"""Nym PII Multilingual Small through ONNX Runtime on the CPU.

The directory follows the hub layout of Wismut/nym-pii-multilingual-small:
``<model_dir>/<variant>/`` holds ``model_int8.onnx``, ``tokenizer.json`` and
``config.json`` for the ``int8`` and ``edge-int8`` variants.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..dataset import Span
from ..taxonomy import NYM_LABELS
from .base import Detection, Detector
from .chunking import SAME_LINE_GAP, Entity, chunk_text, group_bio, join_fragments, merge_entities, to_detection
from .hf_token import fetched_files, file_record, hub_revision, inference_threads, require_dir

REPO = "Wismut/nym-pii-multilingual-small"
MODEL_FILE = "model_int8.onnx"
PROVIDERS = ("CPUExecutionProvider",)
OVERLAP_CHARS = 200
# Byte fallback turns one character into at most four tokens. The reserve
# covers <bos>, <eos> and a leading word marker split into three bytes.
MAX_TOKENS_PER_CHAR = 4
RESERVED_TOKENS = 5
PERSON = "PERSON"


class NymDetector(Detector):
    """Token classifier with 40 PII types; names, IDs, IBANs and contact data map onto the taxonomy.

    Decoding follows the nym CLI (``src/engine/ner_token.rs``): argmax per
    token, B-/I- grouping, then joining fragments of one label. It differs in
    four places. ``threshold`` applies to the entity score, the mean softmax
    probability of its tokens, where nym drops single tokens below it.
    Fragments join only within a line, where nym joins across line breaks.
    Chunks hold ``chunk_chars`` characters with a 200-character overlap,
    where nym cuts 480-token windows with a 64-token overlap. Entities of one
    character stay, where nym drops them.
    With ``merge_names`` set, given names and surnames separated only by
    spaces on one line become one PERSON span scored with the mean of the
    parts.
    """

    kind = "nym"

    def __init__(
        self,
        model_dir: str,
        variant: str = "int8",
        threshold: float = 0.5,
        merge_names: bool = True,
        chunk_chars: int = 1500,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.variant = variant
        self.threshold = threshold
        self.merge_names = merge_names
        self.chunk_chars = chunk_chars
        self._session: Any = None
        self._tokenizer: Any = None
        self._inputs: list[str] = []
        self._id2label: dict[int, str] = {}
        self._budget = chunk_chars

    @property
    def variant_dir(self) -> Path:
        return self.model_dir / self.variant

    def prepare(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        require_dir(self.variant_dir)
        config = json.loads((self.variant_dir / "config.json").read_text(encoding="utf-8"))
        self._id2label = {int(k): v for k, v in config["id2label"].items()}
        self._budget = self._char_budget(config)
        self._tokenizer = Tokenizer.from_file(str(self.variant_dir / "tokenizer.json"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = inference_threads()
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(self.variant_dir / MODEL_FILE), options, providers=list(PROVIDERS))
        self._inputs = [i.name for i in self._session.get_inputs()]

    def _char_budget(self, config: Mapping[str, Any]) -> int:
        """Chunk size in characters that cannot exceed the model's position limit."""
        return min(self.chunk_chars, (int(config["max_position_embeddings"]) - RESERVED_TOKENS) // MAX_TOKENS_PER_CHAR)

    def info(self) -> dict[str, Any]:
        import onnxruntime as ort

        return {
            "kind": self.kind,
            "repo": REPO,
            "revision": hub_revision(self.model_dir, f"{self.variant}/{MODEL_FILE}"),
            "variant": self.variant,
            "onnx": file_record(self.variant_dir / MODEL_FILE),
            "fetched_files": fetched_files(self.model_dir),
            "threshold": self.threshold,
            "merge_names": self.merge_names,
            "chunk_chars": self.chunk_chars,
            "overlap_chars": OVERLAP_CHARS,
            "intra_op_threads": inference_threads(),
            "providers": list(PROVIDERS),
            "onnxruntime": ort.__version__,
        }

    def detect(self, text: str) -> Detection:
        if self._session is None:
            self.prepare()
        entities: list[Entity] = []
        for chunk in chunk_text(text, self._budget, min(OVERLAP_CHARS, self._budget // 2)):
            if chunk.text.strip():
                entities += [e.shifted(chunk.start) for e in self._infer(chunk.text)]
        detection = to_detection(text, merge_entities(entities), NYM_LABELS)
        if self.merge_names:
            detection.spans = merge_names(text, detection.spans)
        return detection.with_threshold(self.threshold)

    def _infer(self, text: str) -> list[Entity]:
        encoding = self._tokenizer.encode(text)
        ids = np.asarray([encoding.ids], dtype=np.int64)
        arrays = {
            "input_ids": ids,
            "attention_mask": np.asarray([encoding.attention_mask], dtype=np.int64),
            "token_type_ids": np.zeros_like(ids),
        }
        unknown = [name for name in self._inputs if name not in arrays]
        if unknown:
            raise RuntimeError(f"the ONNX graph asks for inputs this detector cannot build: {unknown}")
        logits = self._session.run(None, {name: arrays[name] for name in self._inputs})[0][0]
        return decode_logits(text, logits, encoding.offsets, encoding.special_tokens_mask, self._id2label)


def decode_logits(
    text: str,
    logits: npt.NDArray[np.floating[Any]],
    offsets: Sequence[tuple[int, int]],
    special_tokens: Sequence[int],
    id2label: Mapping[int, str],
) -> list[Entity]:
    """Softmax and argmax per character range, B-/I- grouping and fragment joining, in offsets into ``text``.

    Byte fallback encodes one character, such as an Arabic-Indic digit, as two or more tokens with the same
    offsets. Their probabilities are averaged into one prediction, so a disagreement between the byte pieces
    cannot split an entity in the middle of a character.
    """
    shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    pieces: dict[tuple[int, int], list[int]] = {}
    for i, (start, end) in enumerate(offsets):
        if not special_tokens[i] and end > start:
            pieces.setdefault((start, end), []).append(i)
    tags = []
    for (start, end), rows in pieces.items():
        mean = probs[rows].mean(axis=0)
        best = int(np.argmax(mean))
        tags.append((start, end, id2label[best], float(mean[best])))
    return join_fragments(text, group_bio(tags))


def merge_names(text: str, spans: Sequence[Span]) -> list[Span]:
    """Join PERSON spans that only spaces or tabs separate; the score is the mean of the parts."""
    runs: list[list[Span]] = []
    for span in sorted(spans, key=lambda s: (s.start, s.end)):
        last = runs[-1][-1] if runs else None
        if (
            last is not None
            and last.label == span.label == PERSON
            and last.end <= span.start
            and SAME_LINE_GAP.fullmatch(text, last.end, span.start)
        ):
            runs[-1].append(span)
        else:
            runs.append([span])
    return [_join(text, run) for run in runs]


def _join(text: str, run: Sequence[Span]) -> Span:
    if len(run) == 1:
        return run[0]
    scores = [s.score for s in run if s.score is not None]
    score = sum(scores) / len(scores) if scores else None
    start, end = run[0].start, run[-1].end
    return Span(start, end, run[0].label, text[start:end], score)
