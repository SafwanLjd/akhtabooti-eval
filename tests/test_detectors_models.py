"""Model detectors: Nym (ONNX), transformers pipelines (CAMeLBERT) and DialogPII (mmBERT + CRF).

Unit tests use fake tokenizers, sessions and pipelines, plus a tiny randomly
initialised DialogPII model. Tests marked ``integration`` load the real models
from ``models/`` and skip when the files are missing.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from akheval.dataset import Span
from akheval.detectors import dialogpii
from akheval.detectors.dialogpii import DialogPIIDetector, Word, crf_marginals, split_sentences, split_words
from akheval.detectors.hf_token import (
    HUB_METADATA,
    HFTokenDetector,
    fetched_files,
    fetched_repo,
    file_record,
    hub_revision,
    inference_threads,
)
from akheval.detectors.nym import MODEL_FILE, NymDetector, decode_logits, merge_names

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = REPO_ROOT / "models"

NYM_ID2LABEL = {
    0: "O",
    1: "B-GIVEN_NAME",
    2: "I-GIVEN_NAME",
    3: "B-SURNAME",
    4: "I-SURNAME",
    5: "B-EMAIL",
    6: "I-EMAIL",
    7: "B-DATE",
    8: "I-DATE",
    9: "B-IBAN",
    10: "I-IBAN",
}
NYM_LABEL2ID = {v: k for k, v in NYM_ID2LABEL.items()}

SENTENCES = [
    ("en", "Please contact Omar Haddad at omar.haddad@example.org or +962 79 555 0142 about invoice INV-2024-0042."),
    ("ar", "السيد محمد أحمد الخطيب، الرقم الوطني 9871234567، يسكن في شارع الجامعة، عمّان."),
    ("mixed", "Beneficiary: ليلى حسن, IBAN JO71 CBJO 0010 0000 0000 0131 0003 02, Amman, Jordan."),
    ("en", "P<JORHADDAD<<OMAR<<<<<<<<<<<<<<<<<<<<<<<<<<<\nN1234567<8JOR8501019M3001017<<<<<<<<<<<<<<04"),
    ("ar", "رقم بطاقة اللاجئ 123-45C67890 صادرة عن المفوضية في مخيم الزعتري، هاتف 0791234567."),
]


# --- Nym -------------------------------------------------------------------


@dataclass
class FakeEncoding:
    ids: list[int]
    offsets: list[tuple[int, int]]
    attention_mask: list[int]
    special_tokens_mask: list[int]


@dataclass
class FakeWordTokenizer:
    """Whitespace tokenizer that wraps the words in <bos> and <eos>."""

    vocab: dict[str, int] = field(default_factory=lambda: {"<bos>": 0, "<eos>": 1})

    def encode(self, text: str) -> FakeEncoding:
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        ids = [0] + [self.vocab.setdefault(text[s:e], len(self.vocab)) for s, e in spans] + [1]
        offsets = [(0, 0), *spans, (0, 0)]
        special = [1] + [0] * len(spans) + [1]
        return FakeEncoding(ids, offsets, [1] * len(ids), special)


@dataclass
class FakeSession:
    """Returns logits that put ``tags[word]`` (default O) on top for each token."""

    tokenizer: FakeWordTokenizer
    tags: dict[str, tuple[str, float]]
    feeds: list[dict[str, Any]] = field(default_factory=list)

    def run(self, outputs: None, feeds: dict[str, Any]) -> list[Any]:
        del outputs
        self.feeds.append(feeds)
        words = {i: w for w, i in self.tokenizer.vocab.items()}
        logits = np.zeros((1, feeds["input_ids"].shape[1], len(NYM_ID2LABEL)), dtype=np.float32)
        for position, token in enumerate(feeds["input_ids"][0]):
            tag, logit = self.tags.get(words[int(token)], ("O", 9.0))
            logits[0, position, NYM_LABEL2ID[tag]] = logit
        return [logits]


def fake_nym(tags: dict[str, tuple[str, float]], **params: Any) -> tuple[NymDetector, FakeSession]:
    detector = NymDetector("unused", **params)
    tokenizer = FakeWordTokenizer()
    session = FakeSession(tokenizer, tags)
    detector._tokenizer = tokenizer
    detector._session = session
    detector._inputs = ["input_ids", "attention_mask"]
    detector._id2label = NYM_ID2LABEL
    return detector, session


def test_decode_logits_softmax_grouping_and_special_tokens() -> None:
    text = "Omar Haddad omar@example.org"
    logits = np.full((5, 11), -5.0, dtype=np.float32)
    logits[0, 1] = 9.0  # <bos> predicted as a name; special tokens are skipped
    logits[1, 1] = 4.0
    logits[2, 3] = 4.0
    logits[3, 5] = 4.0
    logits[4, 0] = 9.0
    offsets = [(0, 0), (0, 4), (4, 11), (11, 28), (0, 0)]
    entities = decode_logits(text, logits, offsets, [1, 0, 0, 0, 1], NYM_ID2LABEL)
    top = float(np.exp(4.0) / (np.exp(4.0) + 10 * np.exp(-5.0)))
    assert [(e.start, e.end, e.label, e.tokens) for e in entities] == [
        (0, 4, "GIVEN_NAME", 1),
        (4, 11, "SURNAME", 1),
        (11, 28, "EMAIL", 1),
    ]
    assert [e.score for e in entities] == pytest.approx([top] * 3)


def test_decode_logits_joins_fragments_on_one_line() -> None:
    text = "IBAN JO71 CBJO"
    logits = np.zeros((3, 11), dtype=np.float32)
    logits[0, 9] = logits[1, 9] = logits[2, 0] = 9.0
    entities = decode_logits(text, logits, [(4, 9), (9, 14), (0, 0)], [0, 0, 1], NYM_ID2LABEL)
    assert [(e.start, e.end, e.label, e.tokens) for e in entities] == [(4, 14, "IBAN", 2)]


def test_decode_logits_averages_byte_pieces_of_one_character() -> None:
    text = "IBAN ٩٦"
    logits = np.zeros((4, 11), dtype=np.float32)
    logits[0, 9] = 9.0
    logits[1, 8] = 3.0  # first byte piece of the digit leans to DATE
    logits[2, 10] = 9.0  # second byte piece is confident IBAN
    logits[3, 10] = 9.0
    offsets = [(5, 6), (6, 7), (6, 7), (0, 0)]
    entities = decode_logits(text, logits, offsets, [0, 0, 0, 1], NYM_ID2LABEL)
    assert [(e.start, e.end, e.label, e.tokens) for e in entities] == [(5, 7, "IBAN", 2)]


def test_nym_maps_labels_merges_names_and_counts_unmapped() -> None:
    detector, session = fake_nym(
        {
            "Omar": ("B-GIVEN_NAME", 9.0),
            "Haddad": ("B-SURNAME", 9.0),
            "omar@example.org": ("B-EMAIL", 9.0),
            "2024-01-05": ("B-DATE", 9.0),
        },
        threshold=0.0,
    )
    text = "Omar Haddad wrote to omar@example.org on 2024-01-05"
    detection = detector.detect(text)
    assert [(s.label, s.text) for s in detection.spans] == [("PERSON", "Omar Haddad"), ("EMAIL", "omar@example.org")]
    assert detection.raw_labels == {"GIVEN_NAME": 1, "SURNAME": 1, "EMAIL": 1, "DATE": 1}
    assert detection.unmapped == {"DATE": 1}
    assert set(session.feeds[0]) == {"input_ids", "attention_mask"}


def test_nym_without_name_merging_keeps_parts() -> None:
    detector, _ = fake_nym({"Omar": ("B-GIVEN_NAME", 9.0), "Haddad": ("B-SURNAME", 9.0)}, merge_names=False)
    assert [s.text for s in detector.detect("Omar Haddad").spans] == ["Omar", "Haddad"]


def test_nym_threshold_drops_low_scores() -> None:
    tags = {"Omar": ("B-GIVEN_NAME", 9.0), "Amman": ("B-GIVEN_NAME", 0.5)}
    detector, _ = fake_nym(tags, threshold=0.5)
    detection = detector.detect("Omar\nAmman")
    assert [s.text for s in detection.spans] == ["Omar"]
    assert detection.raw_labels == {"GIVEN_NAME": 2}


def test_nym_rejects_graph_inputs_it_cannot_build() -> None:
    detector, _ = fake_nym({})
    detector._inputs = ["input_ids", "pixel_values"]
    with pytest.raises(RuntimeError, match="pixel_values"):
        detector.detect("text")


def test_nym_chunks_long_text_and_keeps_absolute_offsets() -> None:
    detector, session = fake_nym({"Omar": ("B-GIVEN_NAME", 9.0)}, chunk_chars=40)
    detector._budget = 40
    lines = [f"row {i} filler text Omar" for i in range(12)]
    text = "\n".join(lines)
    detection = detector.detect(text)
    assert len(session.feeds) > 1
    assert [s.start for s in detection.spans] == [m.start() for m in re.finditer("Omar", text)]
    assert all(text[s.start : s.end] == "Omar" for s in detection.spans)


def test_nym_char_budget_respects_position_limit() -> None:
    assert NymDetector("unused", chunk_chars=1500)._char_budget({"max_position_embeddings": 8192}) == 1500
    assert NymDetector("unused", chunk_chars=1500)._char_budget({"max_position_embeddings": 512}) == 126


def test_merge_names_rules() -> None:
    text = "Omar\tHaddad\nSara Khalil, Amman"
    spans = [
        Span(0, 4, "PERSON", "Omar", 0.9),
        Span(5, 11, "PERSON", "Haddad", 0.7),
        Span(12, 16, "PERSON", "Sara", 0.8),
        Span(17, 23, "PERSON", "Khalil", 0.6),
        Span(25, 30, "LOCATION", "Amman", 0.9),
    ]
    merged = merge_names(text, spans)
    assert [(s.text, s.score) for s in merged] == [
        ("Omar\tHaddad", pytest.approx(0.8)),
        ("Sara Khalil", pytest.approx(0.7)),
        ("Amman", 0.9),
    ]


def test_nym_info_reads_revision_and_checksum(tmp_path: Path) -> None:
    variant = tmp_path / "int8"
    variant.mkdir()
    (variant / MODEL_FILE).write_bytes(b"onnx")
    metadata = tmp_path / HUB_METADATA / "int8"
    metadata.mkdir(parents=True)
    (metadata / f"{MODEL_FILE}.metadata").write_text("abc123\netag\n1.0\n", encoding="utf-8")
    info = NymDetector(str(tmp_path), threshold=0.3).info()
    assert info["revision"] == "abc123"
    assert info["onnx"]["bytes"] == 4
    assert info["onnx"]["sha256"] == hashlib.sha256(b"onnx").hexdigest()
    assert info["threshold"] == 0.3
    assert info["intra_op_threads"] >= 1


def test_fetched_files_reads_the_fetch_record(tmp_path: Path) -> None:
    files = {"int8/config.json": "aa", "int8/model_int8.onnx": "bb", "int8/tokenizer.json": "cc"}
    record = {"repo": "Wismut/x", "revision": "abc", "files": files}
    (tmp_path / "FETCHED.json").write_text(json.dumps(record), encoding="utf-8")
    assert fetched_files(tmp_path) == files
    assert NymDetector(str(tmp_path)).info()["fetched_files"] == files
    assert fetched_files(tmp_path / "absent") is None


def test_hub_revision_and_file_record_for_missing_files(tmp_path: Path) -> None:
    assert hub_revision(tmp_path, "config.json") is None
    assert file_record(tmp_path / "absent.bin") == {"path": str(tmp_path / "absent.bin"), "bytes": None, "sha256": None}


@pytest.mark.parametrize(
    ("logical_cpus", "physical_cores", "usable", "threads"),
    [(8, 4, 8, 2), (192, 96, 4, 1), (16, 16, 16, 8), (2, 1, 2, 1)],
)
def test_inference_threads_follow_the_affinity_mask(
    monkeypatch: pytest.MonkeyPatch, logical_cpus: int, physical_cores: int, usable: int, threads: int
) -> None:
    import os

    import psutil

    def cpu_count(logical: bool = True) -> int:
        return logical_cpus if logical else physical_cores

    monkeypatch.setattr(psutil, "cpu_count", cpu_count)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(usable)), raising=False)
    assert inference_threads() == threads


def test_prepare_reports_missing_model_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fetch"):
        NymDetector(str(tmp_path / "absent")).prepare()


# --- transformers pipeline -------------------------------------------------


class FakeHFTokenizer:
    """Counts one token per whitespace-separated word, plus [CLS] and [SEP]."""

    def num_special_tokens_to_add(self) -> int:
        return 2

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        count = len(text.split()) + (2 if add_special_tokens else 0)
        return {"input_ids": list(range(count))}


@dataclass
class FakePipeline:
    """Finds fixed words and labels them like an aggregated token-classification pipeline."""

    entities: dict[str, tuple[str, float]]
    tokenizer: FakeHFTokenizer = field(default_factory=FakeHFTokenizer)
    calls: list[str] = field(default_factory=list)

    def __call__(self, text: str) -> list[dict[str, Any]]:
        self.calls.append(text)
        found = []
        for word, (label, score) in self.entities.items():
            for match in re.finditer(re.escape(word), text):
                found.append({"entity_group": label, "score": score, "start": match.start(), "end": match.end()})
        return found


def test_hf_detector_maps_labels_and_respects_token_budget() -> None:
    detector = HFTokenDetector("unused", labels="camelbert", threshold=0.5, max_length=12)
    pipe = FakePipeline(
        {"محمد": ("PERS", 0.9), "عمّان": ("LOC", 0.4), "الأمم المتحدة": ("ORG", 0.8), "حزيران": ("MISC", 0.9)}
    )
    detector._pipe = pipe
    text = "\n".join(["زار محمد عمّان في حزيران مع وفد الأمم المتحدة"] * 4)
    detection = detector.detect(text)
    assert len(pipe.calls) > 1
    assert all(len(call.split()) <= 10 for call in pipe.calls)
    names = [m.start() for m in re.finditer("محمد", text)]
    orgs = [m.start() for m in re.finditer("الأمم المتحدة", text)]
    assert sorted((s.start, s.label) for s in detection.spans) == sorted(
        [(n, "PERSON") for n in names] + [(o, "ORG") for o in orgs]
    )
    assert detection.unmapped == {"MISC": 4}
    assert detection.raw_labels["LOC"] == 4


def test_hf_detector_info_reads_fetch_record(tmp_path: Path) -> None:
    (tmp_path / "pytorch_model.bin").write_bytes(b"weights")
    (tmp_path / "FETCHED.json").write_text(json.dumps({"repo": "CAMeL-Lab/x", "revision": "abc"}), encoding="utf-8")
    metadata = tmp_path / HUB_METADATA
    metadata.mkdir(parents=True)
    (metadata / "pytorch_model.bin.metadata").write_text("abc\netag\n1.0\n", encoding="utf-8")
    info = HFTokenDetector(str(tmp_path), labels="camelbert").info()
    assert (info["repo"], info["revision"], info["weights"]["bytes"]) == ("CAMeL-Lab/x", "abc", 7)
    assert fetched_repo(tmp_path / "absent") is None


def test_hf_detector_rejects_unknown_label_table() -> None:
    with pytest.raises(ValueError, match="label table"):
        HFTokenDetector("unused", labels="nope")


# --- DialogPII: text handling ------------------------------------------------


def _texts(text: str, words: list[Word]) -> list[str]:
    return [text[w.start : w.end] for w in words]


def test_split_words_keeps_emails_and_splits_edge_punctuation() -> None:
    text = 'Email "omar.haddad@example.org", then call (+962) 79-555... OK؟'
    assert _texts(text, split_words(text)) == [
        "Email",
        '"',
        "omar.haddad@example.org",
        '"',
        ",",
        "then",
        "call",
        "(",
        "+962",
        ")",
        "79-555",
        "...",
        "OK",
        "؟",
    ]


def test_split_words_keeps_arabic_diacritics() -> None:
    text = "مُحَمَّدٌ، في عمّان."
    assert _texts(text, split_words(text)) == ["مُحَمَّدٌ", "،", "في", "عمّان", "."]


def test_split_sentences_on_punctuation_and_newlines() -> None:
    text = "اسمي ليلى. أين تسكن؟ في إربد\nPhone: 0791234567! Thanks"
    sentences = split_sentences(text, split_words(text))
    assert [" ".join(_texts(text, s)) for s in sentences] == [
        "اسمي ليلى .",
        "أين تسكن ؟",
        "في إربد",
        "Phone : 0791234567 !",
        "Thanks",
    ]


def test_split_sentences_caps_length() -> None:
    text = " ".join(["word"] * (dialogpii.MAX_SENTENCE_WORDS * 2 + 5))
    sizes = [len(s) for s in split_sentences(text, split_words(text))]
    assert sizes == [dialogpii.MAX_SENTENCE_WORDS, dialogpii.MAX_SENTENCE_WORDS, 5]


# --- DialogPII: CRF ----------------------------------------------------------


def _brute_force(emissions: Any, start: Any, end: Any, transitions: Any) -> tuple[Any, list[int]]:
    """Marginals and best path of one sequence by enumerating every tag path."""
    steps, tags = emissions.shape
    weights: dict[tuple[int, ...], float] = {}
    for path in itertools.product(range(tags), repeat=steps):
        score = start[path[0]] + end[path[-1]] + sum(emissions[t, k] for t, k in enumerate(path))
        score += sum(transitions[a, b] for a, b in itertools.pairwise(path))
        weights[path] = float(np.exp(float(score)))
    total = sum(weights.values())
    marginals = np.zeros((steps, tags))
    for path, weight in weights.items():
        for t, k in enumerate(path):
            marginals[t, k] += weight / total
    return marginals, list(max(weights.items(), key=lambda item: item[1])[0])


def test_crf_marginals_and_viterbi_match_enumeration() -> None:
    torch = pytest.importorskip("torch")
    torchcrf = pytest.importorskip("torchcrf")
    generator = torch.Generator().manual_seed(7)
    crf = torchcrf.CRF(3, batch_first=True)
    with torch.no_grad():
        for parameter in (crf.start_transitions, crf.end_transitions, crf.transitions):
            parameter.copy_(torch.randn(parameter.shape, generator=generator))
    emissions = torch.randn((2, 4, 3), generator=generator)
    mask = torch.tensor([[True] * 4, [True, True, False, False]])
    marginals = crf_marginals(emissions, mask, crf.start_transitions, crf.end_transitions, crf.transitions)
    paths = crf.decode(emissions, mask=mask)
    for row, length in enumerate([4, 2]):
        expected, best = _brute_force(
            emissions[row, :length].detach().numpy(),
            crf.start_transitions.detach().numpy(),
            crf.end_transitions.detach().numpy(),
            crf.transitions.detach().numpy(),
        )
        assert np.allclose(marginals[row, :length].detach().numpy(), expected, atol=1e-5)
        assert paths[row] == best


# --- DialogPII: tiny random model --------------------------------------------

TINY_LABELS = ["O", "B-PERSON", "I-PERSON", "B-DATETIME", "I-DATETIME", "B-NAME", "I-NAME"]
TINY_TEXT = "Omar Haddad lives in Amman. He was born on 1990-01-01!\nCall Sara at home؟ ليلى تسكن في إربد."
WINDOW_TEXT = "Omar lives here. Sara too. Amman is big. He was born. Call home."


def _tiny_tokenizer(path: Path, text: str) -> None:
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers, processors

    specials = ["<pad>", "<eos>", "<bos>", "<unk>"]
    pieces = sorted({p for word in text.split() for p in re.findall(r".{1,3}", word)})
    vocab = {token: index for index, token in enumerate(specials + pieces)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(".{1,3}"), behavior="isolated")
    tokenizer.add_special_tokens(specials)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>", special_tokens=[("<bos>", 2), ("<eos>", 1)]
    )
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({"sep_token": "<eos>", "pad_token": "<pad>"}))


@pytest.fixture(scope="module")
def tiny_dialogpii(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A model directory with the DialogPII layout and a randomly initialised two-layer encoder."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchcrf")
    from transformers import ModernBertConfig

    root = tmp_path_factory.mktemp("dialogpii")
    model_dir, base_dir = root / "model", root / "base"
    model_dir.mkdir()
    _tiny_tokenizer(model_dir, f"{TINY_TEXT} {WINDOW_TEXT}")
    vocab_size = len(json.loads((model_dir / "tokenizer.json").read_text())["model"]["vocab"])
    ModernBertConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=512,
        local_attention=8,
        layer_types=["full_attention", "sliding_attention"],
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        cls_token_id=2,
        sep_token_id=1,
    ).save_pretrained(base_dir)
    id2label = dict(enumerate(TINY_LABELS))
    (model_dir / "crf_config.json").write_text(json.dumps({"num_labels": len(TINY_LABELS), "id2label": id2label}))
    flert = {"use_flert": True, "context_window": 2, "context_sep_marker": True, "context_left_only": False}
    (model_dir / "flert_config.json").write_text(json.dumps(flert))
    torch.manual_seed(0)
    model = dialogpii.build_model(base_dir, len(TINY_LABELS))
    torch.save(model.state_dict(), model_dir / dialogpii.WEIGHTS)
    return model_dir, base_dir


def _forced(model_dir: Path, tmp_path: Path, label: str) -> Path:
    """Copy of the tiny model whose classifier always emits ``label`` and whose CRF is neutral."""
    import shutil

    import torch

    forced = tmp_path / "forced"
    shutil.copytree(model_dir, forced)
    state = torch.load(forced / dialogpii.WEIGHTS)
    state["classifier.weight"].zero_()
    state["classifier.bias"].zero_()
    state["classifier.bias"][TINY_LABELS.index(label)] = 10.0
    for key in ("crf.start_transitions", "crf.end_transitions", "crf.transitions"):
        state[key].zero_()
    torch.save(state, forced / dialogpii.WEIGHTS)
    return forced


def test_dialogpii_forced_person_tags_every_word(tiny_dialogpii: tuple[Path, Path], tmp_path: Path) -> None:
    model_dir, base_dir = tiny_dialogpii
    detector = DialogPIIDetector(str(_forced(model_dir, tmp_path, "B-PERSON")), base_dir=str(base_dir))
    detection = detector.detect(TINY_TEXT)
    words = split_words(TINY_TEXT)
    assert [(s.start, s.end) for s in detection.spans] == [(w.start, w.end) for w in words]
    assert {s.label for s in detection.spans} == {"PERSON"}
    assert all(s.score is not None and s.score > 0.99 for s in detection.spans)
    assert detection.raw_labels == {"PERSON": len(words)}


def test_dialogpii_forced_datetime_is_unmapped(tiny_dialogpii: tuple[Path, Path], tmp_path: Path) -> None:
    model_dir, base_dir = tiny_dialogpii
    detector = DialogPIIDetector(str(_forced(model_dir, tmp_path, "I-DATETIME")), base_dir=str(base_dir))
    detection = detector.detect(TINY_TEXT)
    assert detection.spans == []
    assert detection.unmapped == {"DATETIME": len(split_sentences(TINY_TEXT, split_words(TINY_TEXT)))}


def test_dialogpii_maps_name_label_from_checkpoint(tiny_dialogpii: tuple[Path, Path], tmp_path: Path) -> None:
    model_dir, base_dir = tiny_dialogpii
    detector = DialogPIIDetector(str(_forced(model_dir, tmp_path, "B-NAME")), base_dir=str(base_dir))
    assert {s.label for s in detector.detect("Omar Haddad").spans} == {"PERSON"}


def test_dialogpii_random_model_is_deterministic_and_aligned(tiny_dialogpii: tuple[Path, Path]) -> None:
    model_dir, base_dir = tiny_dialogpii
    detector = DialogPIIDetector(str(model_dir), threshold=0.0, base_dir=str(base_dir))
    first, second = detector.detect(TINY_TEXT), detector.detect(TINY_TEXT)
    assert first == second
    starts = {w.start for w in split_words(TINY_TEXT)}
    ends = {w.end for w in split_words(TINY_TEXT)}
    for span in first.spans:
        assert span.start in starts and span.end in ends
        assert span.score is not None and 0.0 <= span.score <= 1.0
    assert sum(first.raw_labels.values()) >= len(first.spans)
    assert detector.detect("") == detector.detect("   \n ")


def test_dialogpii_window_layout(tiny_dialogpii: tuple[Path, Path]) -> None:
    model_dir, base_dir = tiny_dialogpii
    detector = DialogPIIDetector(str(model_dir), base_dir=str(base_dir))
    detector.prepare()
    text = WINDOW_TEXT
    sentences = split_sentences(text, split_words(text))
    window = detector.window(text, sentences, 2)
    tokens = [detector._tokenizer.id_to_token(i) for i in window.ids]
    assert tokens.count("<eos>") == 3  # two separators and the closing token
    assert tokens[0] == "<bos>"
    target = [text[w.start : w.end] for w in sentences[2]]
    assert [tokens[p] for p in window.positions if p is not None] == [t[:3] for t in target]
    edge = detector.window(text, sentences, 0)
    assert [detector._tokenizer.id_to_token(i) for i in edge.ids].count("<eos>") == 2


def test_dialogpii_window_drops_context_then_truncates(
    tiny_dialogpii: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir, base_dir = tiny_dialogpii
    monkeypatch.setattr(dialogpii, "MAX_TOKENS", 12)
    detector = DialogPIIDetector(str(model_dir), base_dir=str(base_dir))
    detector.prepare()
    text = WINDOW_TEXT
    sentences = split_sentences(text, split_words(text))
    window = detector.window(text, sentences, 2)
    assert len(window.ids) <= 12
    assert all(p is not None for p in window.positions)
    long_text = "Omar Haddad lives in Amman with Sara and Laila"
    long_window = detector.window(long_text, split_sentences(long_text, split_words(long_text)), 0)
    assert len(long_window.ids) == 12
    assert long_window.positions[0] is not None
    assert long_window.positions[-1] is None
    assert detector.detect(long_text).raw_labels.total() <= len(split_words(long_text))


def test_dialogpii_batches_match_single_windows(tiny_dialogpii: tuple[Path, Path]) -> None:
    model_dir, base_dir = tiny_dialogpii
    detector = DialogPIIDetector(str(model_dir), base_dir=str(base_dir))
    detector.prepare()
    sentences = split_sentences(TINY_TEXT, split_words(TINY_TEXT))
    windows = [detector.window(TINY_TEXT, sentences, i) for i in range(len(sentences))]
    batched = detector._tag(windows)
    for window, tags in zip(windows, batched, strict=True):
        single = detector._tag([window])[0]
        assert [t for t, _ in single] == [t for t, _ in tags]
        assert np.allclose([p for _, p in single], [p for _, p in tags], atol=1e-4)


def test_dialogpii_info_without_model(tmp_path: Path) -> None:
    info = DialogPIIDetector(str(tmp_path), threshold=0.4).info()
    assert info["repo"] == "DFKI-SLT/multilingual_DialogPII_NER"
    assert info["weights"]["sha256"] is None
    assert info["fetched_files"] is None
    assert info["threshold"] == 0.4
    assert "marginal" in info["score"]


# --- integration: real models ----------------------------------------------

NYM_DIR = MODELS / "nym-pii-multilingual-small"
CAMELBERT_DIR = MODELS / "camelbert-mix-ner"
DIALOGPII_DIR = MODELS / "dialogpii"
MMBERT_DIR = MODELS / "mmbert-base"


def _run(detector: Any) -> list[list[tuple[str, str]]]:
    """Detect on every sample sentence and print spans and timing for the test log."""
    detector.prepare()
    results = []
    for language, sentence in SENTENCES:
        started = time.perf_counter()
        detection = detector.detect(sentence)
        elapsed = time.perf_counter() - started
        print(
            f"[{language}] {elapsed * 1000:.0f} ms",
            [(s.label, s.text, round(s.score or 0, 3)) for s in detection.spans],
        )
        results.append([(s.label, s.text) for s in detection.spans])
    return results


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["int8", "edge-int8"])
def test_nym_real_model(variant: str) -> None:
    if not (NYM_DIR / variant / MODEL_FILE).is_file():
        pytest.skip(f"Nym {variant} is not downloaded")
    results = _run(NymDetector(str(NYM_DIR), variant=variant))
    assert ("PERSON", "Omar Haddad") in results[0]
    assert ("EMAIL", "omar.haddad@example.org") in results[0]
    assert ("PHONE", "+962 79 555 0142") in results[0]
    assert ("PERSON", "محمد أحمد الخطيب") in results[1]
    assert ("NATIONAL_ID", "9871234567") in results[1]
    assert ("IBAN", "JO71 CBJO 0010 0000 0000 0131 0003 02") in results[2]
    assert ("PHONE", "0791234567") in results[4]


@pytest.mark.integration
def test_camelbert_real_model() -> None:
    if not (CAMELBERT_DIR / "pytorch_model.bin").is_file():
        pytest.skip("CAMeLBERT Mix NER is not downloaded")
    results = _run(HFTokenDetector(str(CAMELBERT_DIR), labels="camelbert"))
    assert ("PERSON", "محمد أحمد الخطيب") in results[1]
    assert ("PERSON", "ليلى حسن") in results[2]


@pytest.mark.integration
def test_dialogpii_real_model() -> None:
    if not (DIALOGPII_DIR / dialogpii.WEIGHTS).is_file() or not (MMBERT_DIR / "config.json").is_file():
        pytest.skip("DialogPII checkpoint or mmBERT-base config is not downloaded")
    results = _run(DialogPIIDetector(str(DIALOGPII_DIR), base_dir=str(MMBERT_DIR)))
    assert ("PERSON", "Omar Haddad") in results[0]
    assert ("EMAIL", "omar.haddad@example.org") in results[0]
