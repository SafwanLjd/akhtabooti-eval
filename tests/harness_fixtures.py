"""Tiny synthetic dataset, fake components and config writer shared by the harness tests."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from akheval import registry
from akheval.dataset import Dataset, Document, Rendering, Span, write_dataset
from akheval.detectors.base import Detection, Detector
from akheval.extractors.base import ExtractionError, Extractor

EMAIL = re.compile(r"[\w.]+@[\w.]+\.[a-z]{2,}")
DIGITS = re.compile(r"\d{10}")
EMAIL_SCORE = 0.9
DIGITS_SCORE = 0.4
EXPLODE = "EXPLODE"


class FakeDetector(Detector):
    """Emails score 0.9, ten-digit runs score 0.4 and are labelled PHONE; texts with EXPLODE raise."""

    kind = "fake"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.prepared = False

    def prepare(self) -> None:
        self.prepared = True

    def info(self) -> dict[str, Any]:
        return {"kind": self.kind, "threshold": self.threshold}

    def detect(self, text: str) -> Detection:
        if EXPLODE in text:
            raise RuntimeError("fake detector failure")
        spans = [Span(m.start(), m.end(), "EMAIL", m.group(), EMAIL_SCORE) for m in EMAIL.finditer(text)]
        spans += [Span(m.start(), m.end(), "PHONE", m.group(), DIGITS_SCORE) for m in DIGITS.finditer(text)]
        detection = Detection(spans=sorted(spans, key=lambda s: s.start))
        detection.raw_labels.update(s.label for s in spans)
        return detection.with_threshold(self.threshold)


class FakeExtractor(Extractor):
    """Reads files as UTF-8 text; a file name containing "broken" fails."""

    kind = "fake-text"

    def __init__(self, upper: bool = False) -> None:
        self.upper = upper

    def info(self) -> dict[str, Any]:
        return {"kind": self.kind, "upper": self.upper}

    def extract(self, path: Path) -> str:
        if "broken" in path.name:
            raise ExtractionError(f"cannot read {path.name}")
        text = path.read_text(encoding="utf-8")
        return text.upper() if self.upper else text


def register_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(registry.DETECTOR_KINDS, "fake", "harness_fixtures:FakeDetector")
    monkeypatch.setitem(registry.EXTRACTOR_KINDS, "fake-text", "harness_fixtures:FakeExtractor")


def span(text: str, value: str, label: str, start_at: int = 0) -> Span:
    start = text.index(value, start_at)
    return Span(start, start + len(value), label, value)


def _rendering(doc_id: str, suffix: str, fmt: str, channel: str = "native", capture: str = "digital") -> Rendering:
    severity = None if capture == "digital" else "mild"
    return Rendering(f"{doc_id}-{suffix}", f"files/{doc_id}-{suffix}.{fmt}", fmt, channel, capture, severity)


def documents() -> list[Document]:
    en = "Contact John Smith at john.smith@example.org or 0791234567. Invoice INV-2024-001."
    ar = "الاسم: أحمد علي، هاتف 0781234567، البريد ahmad@example.com"
    mixed = "مكتب Office ahmad.ali@example.net, ref 1234567890"
    empty = "No personal data here. Version 1.2.3.4 EXPLODE"
    return [
        Document(
            "doc-ar-1",
            "test",
            "ar",
            "form",
            ar,
            spans=(
                span(ar, "أحمد علي", "PERSON"),
                span(ar, "0781234567", "PHONE"),
                span(ar, "ahmad@example.com", "EMAIL"),
            ),
            renderings=(_rendering("doc-ar-1", "txt", "txt"), _rendering("doc-ar-1", "broken", "png", "ocr", "scan")),
        ),
        Document(
            "doc-en-1",
            "test",
            "en",
            "letter",
            en,
            spans=(
                span(en, "John Smith", "PERSON"),
                span(en, "john.smith@example.org", "EMAIL"),
                span(en, "0791234567", "PHONE"),
            ),
            negatives=(span(en, "INV-2024-001", "invoice_number"),),
            renderings=(_rendering("doc-en-1", "txt", "txt"), _rendering("doc-en-1", "scan", "png", "ocr", "scan")),
        ),
        Document(
            "doc-en-2",
            "test",
            "en",
            "memo",
            empty,
            negatives=(span(empty, "1.2.3.4", "version_number"),),
            renderings=(_rendering("doc-en-2", "txt", "txt"),),
        ),
        Document(
            "doc-mixed-1",
            "dev",
            "mixed",
            "letter",
            mixed,
            spans=(span(mixed, "ahmad.ali@example.net", "EMAIL"),),
            negatives=(span(mixed, "1234567890", "reference_number"),),
            renderings=(_rendering("doc-mixed-1", "txt", "txt"),),
        ),
    ]


def make_dataset(root: Path) -> Dataset:
    """Write the tiny dataset; every rendering file holds the canonical text."""
    docs = documents()
    for doc in docs:
        for rendering in doc.renderings:
            path = root / rendering.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(doc.text, encoding="utf-8")
    write_dataset(docs, root, name="tiny", version="0.0.1")
    return Dataset.load(root)


CARD_FRONT = "Name: Amal Haddad\nNational No: 9861234567"
CARD_BACK = "Address: 12 Rainbow Street, Amman\nPhone: 0791234567"
CARD_TEXT = f"{CARD_FRONT}\n{CARD_BACK}"
CARD_BACK_START = len(CARD_FRONT) + 1


def card_photo(face: str, visible: tuple[tuple[int, int], ...]) -> Rendering:
    return Rendering(f"card-1.{face}.jpg", f"files/card-1/{face}.jpg", "jpg", "ocr", "photo", "mild", visible=visible)


def two_face_card() -> Document:
    """An ID card with one photo per face; each photo shows two of the four gold values."""
    return Document(
        "card-1",
        "test",
        "en",
        "jo_id_card",
        CARD_TEXT,
        spans=(
            span(CARD_TEXT, "Amal Haddad", "PERSON"),
            span(CARD_TEXT, "9861234567", "NATIONAL_ID"),
            span(CARD_TEXT, "12 Rainbow Street, Amman", "ADDRESS"),
            span(CARD_TEXT, "0791234567", "PHONE"),
        ),
        renderings=(
            Rendering("card-1.digital.pdf", "files/card-1/digital.pdf", "pdf", "native", "digital"),
            card_photo("front", ((0, len(CARD_FRONT)),)),
            card_photo("back", ((CARD_BACK_START, len(CARD_TEXT)),)),
        ),
    )


def write_card_dataset(root: Path, doc: Document | None = None) -> Dataset:
    """Write a dataset holding the card; each rendering file holds the text that rendering shows."""
    card = doc or two_face_card()
    for rendering in card.renderings:
        path = root / rendering.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(card.visible_text(rendering), encoding="utf-8")
    write_dataset([card], root, name="card", version="0.0.1")
    return Dataset.load(root)


CONFIG_TEMPLATE = """
[project]
dataset = "{dataset}"
results = "{out}/runs"
tuning = "{out}/tuning"

[detectors.fake]
kind = "fake"
threshold = 0.5

[detectors.regex]
kind = "regex-baseline"

[detectors.both]
kind = "hybrid"
parts = ["regex", "fake"]

[extractors.text]
kind = "fake-text"

[extractors.loud]
kind = "fake-text"
upper = true

[pipelines.plain]
detector = "fake"
extractors = {{ default = "text" }}

[[runs]]
name = "text-fake"
mode = "text"
split = "test"
detector = "fake"

[[runs]]
name = "text-regex"
mode = "text"
split = "test"
detector = "regex"

[[runs]]
name = "text-both"
mode = "text"
split = "test"
detector = "both"

[[runs]]
name = "pipeline-plain"
mode = "pipeline"
split = "test"
pipeline = "plain"

[[runs]]
name = "extract-loud"
mode = "extract"
split = "test"
channel = "ocr"
extractors = {{ default = "loud", txt = "text" }}

[[comparisons]]
name = "regex-vs-fake"
baseline = "text-regex"
candidate = "text-fake"
"""


def write_config(tmp_path: Path, dataset: Path) -> Path:
    path = tmp_path / "eval.toml"
    path.write_text(CONFIG_TEMPLATE.format(dataset=dataset.as_posix(), out=(tmp_path / "out").as_posix()))
    return path
