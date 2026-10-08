"""DFKI-SLT multilingual DialogPII NER: mmBERT-base with a CRF head and FLERT context.

``model_dir`` holds the hub snapshot: ``pytorch_model.bin``, ``crf_config.json``,
``flert_config.json`` and ``tokenizer.json``. The checkpoint stores the
encoder weights under ``transformer.*`` without the encoder config, so
``base_dir`` must hold the ``config.json`` of jhu-clsp/mmBERT-base.

Inference follows the model card. The detector splits the text into
sentences and words, then tags each sentence inside a window of up to
``context_sentences`` neighbours per side joined with the separator token.
The CRF decodes the whole window with Viterbi; a word takes the tag of its
first sub-token. Entity score is the mean CRF marginal probability
(forward-backward) of the decoded tag over those first sub-tokens.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..taxonomy import DIALOGPII_LABELS
from .base import Detection, Detector
from .chunking import OUTSIDE, Entity, group_bio, to_detection
from .hf_token import fetched_files, file_record, hub_revision, inference_threads, require_dir

if TYPE_CHECKING:
    import torch

REPO = "DFKI-SLT/multilingual_DialogPII_NER"
WEIGHTS = "pytorch_model.bin"
# Training max_length from the model card; windows lose context to fit it.
MAX_TOKENS = 2048
MAX_SENTENCE_WORDS = 128
BATCH_SIZE = 8
SENTENCE_END = frozenset(".!?؟")
SCORE_METHOD = "mean CRF marginal of the Viterbi tag over each word's first sub-token"
_NON_SPACE = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class Word:
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Window:
    """Token ids of one FLERT window and, per target word, the index of its first sub-token."""

    ids: list[int]
    positions: list[int | None]


def split_words(text: str) -> list[Word]:
    """Split on whitespace and cut leading and trailing punctuation into separate words.

    Runs of one punctuation character ("...", "؟؟") stay together. Inner
    punctuation stays inside the word, so emails, URLs and IBANs remain whole.
    """
    words: list[Word] = []
    for match in _NON_SPACE.finditer(text):
        start, end = match.span()
        lead = start
        while lead < end and _is_punctuation(text[lead]):
            lead += 1
        trail = end
        while trail > lead and _is_punctuation(text[trail - 1]):
            trail -= 1
        words += _runs(text, start, lead)
        if lead < trail:
            words.append(Word(lead, trail))
        words += _runs(text, trail, end)
    return words


def _is_punctuation(char: str) -> bool:
    return unicodedata.category(char).startswith("P")


def _runs(text: str, start: int, end: int) -> list[Word]:
    runs: list[Word] = []
    while start < end:
        stop = start + 1
        while stop < end and text[stop] == text[start]:
            stop += 1
        runs.append(Word(start, stop))
        start = stop
    return runs


def split_sentences(text: str, words: Sequence[Word]) -> list[list[Word]]:
    """Group words into sentences at line breaks and after ``. ! ? ؟``, capped at MAX_SENTENCE_WORDS."""
    sentences: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current and ("\n" in text[current[-1].end : word.start] or len(current) >= MAX_SENTENCE_WORDS):
            sentences.append(current)
            current = []
        current.append(word)
        if set(text[word.start : word.end]) <= SENTENCE_END:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def crf_marginals(
    emissions: torch.Tensor,
    mask: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    transitions: torch.Tensor,
) -> torch.Tensor:
    """Posterior tag probabilities per token from the forward-backward algorithm.

    Shapes: ``emissions`` (batch, steps, tags), ``mask`` (batch, steps) with
    the first step set, ``transitions[i, j]`` scores tag ``i`` followed by
    ``j``. Values at masked steps are undefined.
    """
    import torch

    steps = emissions.shape[1]
    alphas = [start + emissions[:, 0]]
    for t in range(1, steps):
        step = torch.logsumexp(alphas[-1].unsqueeze(2) + transitions + emissions[:, t].unsqueeze(1), dim=1)
        alphas.append(torch.where(mask[:, t, None], step, alphas[-1]))
    betas = [end.expand_as(alphas[0])]
    for t in range(steps - 2, -1, -1):
        step = torch.logsumexp(transitions + (emissions[:, t + 1] + betas[0]).unsqueeze(1), dim=2)
        betas.insert(0, torch.where(mask[:, t + 1, None], step, betas[0]))
    alpha, beta = torch.stack(alphas, dim=1), torch.stack(betas, dim=1)
    log_z = torch.logsumexp(alphas[-1] + end, dim=1)
    return torch.exp(alpha + beta - log_z[:, None, None])


class DialogPIIDetector(Detector):
    """PII tagger trained on synthetic dialogues in eleven languages, Arabic and English among them."""

    kind = "dialogpii"

    def __init__(
        self,
        model_dir: str,
        threshold: float = 0.5,
        context_sentences: int = 2,
        base_dir: str = "models/mmbert-base",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.base_dir = Path(base_dir)
        self.threshold = threshold
        self.context_sentences = context_sentences
        self._model: Any = None
        self._tokenizer: Any = None
        self._id2label: dict[int, str] = {}
        self._sep: str | None = None
        self._pad_id = 0
        self._left_only = False

    def prepare(self) -> None:
        import torch
        from tokenizers import Tokenizer

        require_dir(self.model_dir)
        require_dir(self.base_dir)
        crf_config = self._json("crf_config.json")
        flert_config = self._json("flert_config.json")
        tokenizer_config = self._json("tokenizer_config.json")
        self._id2label = {int(k): v for k, v in crf_config["id2label"].items()}
        self._sep = tokenizer_config["sep_token"] if flert_config.get("context_sep_marker", True) else None
        self._left_only = bool(flert_config.get("context_left_only", False))
        self._tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(MAX_TOKENS)
        self._pad_id = self._tokenizer.token_to_id(tokenizer_config["pad_token"])
        torch.set_num_threads(inference_threads())
        model = build_model(self.base_dir, int(crf_config["num_labels"]))
        state = torch.load(self.model_dir / WEIGHTS, map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(state)
        self._model = model.eval()

    def _json(self, name: str) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((self.model_dir / name).read_text(encoding="utf-8"))
        return data

    def info(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "repo": REPO,
            "revision": hub_revision(self.model_dir, WEIGHTS),
            "weights": file_record(self.model_dir / WEIGHTS),
            "fetched_files": fetched_files(self.model_dir),
            "base_config_revision": hub_revision(self.base_dir, "config.json"),
            "base_fetched_files": fetched_files(self.base_dir),
            "threshold": self.threshold,
            "context_sentences": self.context_sentences,
            "max_tokens": MAX_TOKENS,
            "max_sentence_words": MAX_SENTENCE_WORDS,
            "batch_size": BATCH_SIZE,
            "score": SCORE_METHOD,
            "threads": inference_threads(),
        }

    def detect(self, text: str) -> Detection:
        if self._model is None:
            self.prepare()
        sentences = split_sentences(text, split_words(text))
        windows = [self.window(text, sentences, i) for i in range(len(sentences))]
        entities: list[Entity] = []
        for first in range(0, len(windows), BATCH_SIZE):
            batch = windows[first : first + BATCH_SIZE]
            for sentence, tags in zip(sentences[first : first + BATCH_SIZE], self._tag(batch), strict=True):
                items = [(w.start, w.end, tag, prob) for w, (tag, prob) in zip(sentence, tags, strict=True)]
                entities += group_bio(items)
        return to_detection(text, entities, DIALOGPII_LABELS).with_threshold(self.threshold)

    def window(self, text: str, sentences: Sequence[Sequence[Word]], index: int) -> Window:
        """Encode sentence ``index`` with its context, dropping the farthest context sentence until it fits."""
        left = min(index, self.context_sentences)
        right = 0 if self._left_only else min(len(sentences) - 1 - index, self.context_sentences)
        window, overflow = self._encode(text, sentences, index, left, right)
        while overflow and left + right > 0:
            if left >= right:
                left -= 1
            else:
                right -= 1
            window, overflow = self._encode(text, sentences, index, left, right)
        return window

    def _encode(
        self, text: str, sentences: Sequence[Sequence[Word]], index: int, left: int, right: int
    ) -> tuple[Window, bool]:
        """Encode one window; the flag says whether the tokenizer cut tokens off at MAX_TOKENS."""
        words = [text[w.start : w.end] for sentence in sentences[index - left : index] for w in sentence]
        if self._sep and left:
            words.append(self._sep)
        target_start = len(words)
        words += [text[w.start : w.end] for w in sentences[index]]
        target_end = len(words)
        if self._sep and right:
            words.append(self._sep)
        words += [text[w.start : w.end] for sentence in sentences[index + 1 : index + 1 + right] for w in sentence]
        encoding = self._tokenizer.encode(words, is_pretokenized=True)
        first: dict[int, int] = {}
        for position, word_id in enumerate(encoding.word_ids):
            if word_id is not None:
                first.setdefault(word_id, position)
        positions = [first.get(i) for i in range(target_start, target_end)]
        return Window(list(encoding.ids), positions), bool(encoding.overflowing)

    def _tag(self, windows: Sequence[Window]) -> list[list[tuple[str, float]]]:
        """Run one padded batch and return (tag, score) for every target word of every window."""
        import torch

        ids = torch.full((len(windows), max(len(w.ids) for w in windows)), self._pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for row, window in enumerate(windows):
            ids[row, : len(window.ids)] = torch.tensor(window.ids)
            mask[row, : len(window.ids)] = True
        crf = self._model.crf
        with torch.inference_mode():
            hidden = self._model.transformer(input_ids=ids, attention_mask=mask.long()).last_hidden_state
            emissions = self._model.classifier(hidden)
            paths = crf.decode(emissions, mask=mask)
            marginals = crf_marginals(emissions, mask, crf.start_transitions, crf.end_transitions, crf.transitions)
        return [
            self._word_tags(window, path, marginals[row])
            for row, (window, path) in enumerate(zip(windows, paths, strict=True))
        ]

    def _word_tags(self, window: Window, path: Sequence[int], marginals: torch.Tensor) -> list[tuple[str, float]]:
        tags: list[tuple[str, float]] = []
        for position in window.positions:
            if position is None:
                tags.append((OUTSIDE, 0.0))
            else:
                tag = path[position]
                tags.append((self._id2label[tag], float(marginals[position, tag])))
        return tags


def build_model(base_dir: Path, num_labels: int) -> torch.nn.Module:
    """Module tree whose state-dict keys match the checkpoint: ``transformer.*``, ``classifier.*``, ``crf.*``."""
    import torch
    from torchcrf import CRF
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(base_dir)
    model = torch.nn.Module()
    model.add_module("transformer", AutoModel.from_config(config))
    model.add_module("classifier", torch.nn.Linear(config.hidden_size, num_labels))
    model.add_module("crf", CRF(num_labels, batch_first=True))
    return model
