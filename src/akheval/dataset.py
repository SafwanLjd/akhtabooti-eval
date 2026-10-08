"""Evaluation dataset: documents, gold spans, renderings, manifest and freezing.

A dataset is a directory::

    manifest.json      name, version, frozen flag, counts, sha256 of every file
    documents.jsonl    one Document per line
    files/             rendered files (docx, pdf, xlsx, jpg, ...) referenced by renderings

Offsets are Python ``str`` indices (Unicode code points) into ``Document.text``
and ``end`` is exclusive. ``Document.text`` is the canonical text. A rendering
shows the part of it inside ``Rendering.visible``, or all of it when
``visible`` is None, and its gold values are the spans inside that part.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .taxonomy import LABELS, NEGATIVE_KINDS, TAXONOMY_VERSION

LANGUAGES = ("ar", "en", "mixed")
SPLITS = ("dev", "test")
CHANNELS = ("native", "ocr")
CAPTURES = ("digital", "scan", "photo")
SEVERITIES = ("mild", "moderate", "harsh")
FORMATS = ("txt", "json", "csv", "html", "docx", "odt", "xlsx", "pdf", "png", "jpg")
IMAGE_FORMATS = ("png", "jpg")

MANIFEST = "manifest.json"
DOCUMENTS = "documents.jsonl"


@dataclass(frozen=True, slots=True)
class Span:
    """A labelled character range. Detectors may attach a confidence score."""

    start: int
    end: int
    label: str
    text: str = ""
    score: float | None = None

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"start": self.start, "end": self.end, "label": self.label, "text": self.text}
        if self.score is not None:
            d["score"] = round(self.score, 4)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Span:
        score = d.get("score")
        return cls(
            int(d["start"]),
            int(d["end"]),
            str(d["label"]),
            str(d.get("text", "")),
            None if score is None else float(score),
        )


@dataclass(frozen=True, slots=True)
class ExpectedValue:
    label: str
    value: str


@dataclass(frozen=True, slots=True)
class Rendering:
    """One file that shows the document's canonical text, or part of it.

    ``channel`` says how a scanner gets text out of the file: ``native`` files
    carry a text layer, ``ocr`` files are pixels only. ``capture`` says how the
    file was produced: ``digital`` export, flatbed ``scan`` or phone ``photo``.
    ``visible`` lists the character ranges of ``Document.text`` the file shows,
    for example the front of an ID card; ``None`` means the whole text.
    """

    id: str
    path: str
    format: str
    channel: str
    capture: str
    severity: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    visible: tuple[tuple[int, int], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "format": self.format,
            "channel": self.channel,
            "capture": self.capture,
            "severity": self.severity,
            "params": dict(self.params),
            "visible": None if self.visible is None else [list(r) for r in self.visible],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Rendering:
        return cls(
            id=str(d["id"]),
            path=str(d["path"]),
            format=str(d["format"]),
            channel=str(d["channel"]),
            capture=str(d["capture"]),
            severity=d.get("severity"),
            params=dict(d.get("params", {})),
            visible=None if d.get("visible") is None else tuple((int(a), int(b)) for a, b in d["visible"]),
        )


@dataclass(slots=True)
class Document:
    id: str
    split: str
    language: str
    doc_type: str
    text: str
    spans: tuple[Span, ...] = ()
    negatives: tuple[Span, ...] = ()
    renderings: tuple[Rendering, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def expected_values(self) -> list[ExpectedValue]:
        return [ExpectedValue(s.label, s.text or self.text[s.start : s.end]) for s in self.spans]

    def visible_text(self, rendering: Rendering) -> str:
        """The part of the canonical text that ``rendering`` shows."""
        if rendering.visible is None:
            return self.text
        return "\n".join(self.text[start:end] for start, end in rendering.visible)

    def visible_spans(self, rendering: Rendering) -> tuple[Span, ...]:
        """Gold spans that ``rendering`` shows, with offsets into the canonical text."""
        if rendering.visible is None:
            return self.spans
        return tuple(s for s in self.spans if any(a <= s.start and s.end <= b for a, b in rendering.visible))

    def visible_values(self, rendering: Rendering) -> list[ExpectedValue]:
        return [ExpectedValue(s.label, s.text or self.text[s.start : s.end]) for s in self.visible_spans(rendering)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "split": self.split,
            "language": self.language,
            "doc_type": self.doc_type,
            "text": self.text,
            "spans": [s.to_dict() for s in self.spans],
            "negatives": [s.to_dict() for s in self.negatives],
            "renderings": [r.to_dict() for r in self.renderings],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Document:
        return cls(
            id=str(d["id"]),
            split=str(d["split"]),
            language=str(d["language"]),
            doc_type=str(d["doc_type"]),
            text=str(d["text"]),
            spans=tuple(Span.from_dict(s) for s in d.get("spans", [])),
            negatives=tuple(Span.from_dict(s) for s in d.get("negatives", [])),
            renderings=tuple(Rendering.from_dict(r) for r in d.get("renderings", [])),
            meta=dict(d.get("meta", {})),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Dataset:
    root: Path
    manifest: dict[str, Any]
    documents: list[Document]

    @property
    def name(self) -> str:
        return str(self.manifest.get("name", self.root.name))

    @property
    def version(self) -> str:
        return str(self.manifest.get("version", "unversioned"))

    @property
    def frozen(self) -> bool:
        return bool(self.manifest.get("frozen", False))

    def split(self, name: str | None) -> list[Document]:
        """Documents of one split, or all documents when ``name`` is None."""
        if name is None:
            return list(self.documents)
        if name not in SPLITS:
            raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
        return [d for d in self.documents if d.split == name]

    def renderings(self, split: str | None = None) -> Iterator[tuple[Document, Rendering]]:
        for doc in self.split(split):
            for rendering in doc.renderings:
                yield doc, rendering

    def documents_sha256(self) -> str:
        return sha256_file(self.root / DOCUMENTS)

    @classmethod
    def load(cls, root: Path | str) -> Dataset:
        root = Path(root)
        manifest_path = root / MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        with (root / DOCUMENTS).open(encoding="utf-8") as fh:
            documents = [Document.from_dict(json.loads(line)) for line in fh if line.strip()]
        return cls(root=root, manifest=manifest, documents=documents)


def counts(documents: Iterable[Document]) -> dict[str, Any]:
    docs = list(documents)
    renderings = [r for d in docs for r in d.renderings]
    return {
        "documents": len(docs),
        "by_split": dict(Counter(d.split for d in docs)),
        "by_language": dict(Counter(d.language for d in docs)),
        "by_doc_type": dict(Counter(d.doc_type for d in docs)),
        "entities_by_label": dict(Counter(s.label for d in docs for s in d.spans)),
        "entities_by_language_label": dict(Counter(f"{d.language}/{s.label}" for d in docs for s in d.spans)),
        "negatives_by_kind": dict(Counter(n.label for d in docs for n in d.negatives)),
        "documents_without_pii": sum(1 for d in docs if not d.spans),
        "renderings": len(renderings),
        "renderings_by_format": dict(Counter(r.format for r in renderings)),
        "renderings_by_capture": dict(Counter(r.capture for r in renderings)),
        "renderings_by_severity": dict(Counter(r.severity or "none" for r in renderings)),
    }


def write_dataset(
    documents: list[Document],
    root: Path,
    *,
    name: str,
    version: str,
    generator: Mapping[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Write ``documents.jsonl`` and an unfrozen manifest with file checksums."""
    root.mkdir(parents=True, exist_ok=True)
    doc_path = root / DOCUMENTS
    with doc_path.open("w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
    checksums = {DOCUMENTS: sha256_file(doc_path)}
    for doc in documents:
        for rendering in doc.renderings:
            checksums[rendering.path] = sha256_file(root / rendering.path)
    manifest = {
        "name": name,
        "version": version,
        "frozen": False,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "taxonomy_version": TAXONOMY_VERSION,
        "labels": list(LABELS),
        "generator": dict(generator or {}),
        "counts": counts(documents),
        "checksums": checksums,
        "notes": notes,
    }
    (root / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def freeze(dataset: Dataset) -> dict[str, Any]:
    """Mark a valid dataset as frozen. Raises ValueError listing any problems."""
    problems = validate(dataset)
    if problems:
        raise ValueError("cannot freeze an invalid dataset:\n" + "\n".join(problems))
    manifest = dict(dataset.manifest, frozen=True, frozen_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"))
    (dataset.root / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset.manifest = manifest
    return manifest


def validate(dataset: Dataset) -> list[str]:
    """Return every problem found; an empty list means the dataset is sound."""
    problems: list[str] = []
    seen_docs: set[str] = set()
    seen_renderings: set[str] = set()
    for doc in dataset.documents:
        problems += _validate_document(doc, dataset.root)
        if doc.id in seen_docs:
            problems.append(f"{doc.id}: duplicate document id")
        seen_docs.add(doc.id)
        for rendering in doc.renderings:
            if rendering.id in seen_renderings:
                problems.append(f"{doc.id}: duplicate rendering id {rendering.id}")
            seen_renderings.add(rendering.id)
    problems += _validate_checksums(dataset)
    return problems


def _validate_document(doc: Document, root: Path) -> list[str]:
    problems: list[str] = []
    if doc.split not in SPLITS:
        problems.append(f"{doc.id}: split {doc.split!r} not in {SPLITS}")
    if doc.language not in LANGUAGES:
        problems.append(f"{doc.id}: language {doc.language!r} not in {LANGUAGES}")
    n = len(doc.text)
    for span in doc.spans:
        if span.label not in LABELS:
            problems.append(f"{doc.id}: unknown label {span.label!r}")
        if not 0 <= span.start < span.end <= n:
            problems.append(f"{doc.id}: span {span.start}-{span.end} outside text of length {n}")
        elif span.text and doc.text[span.start : span.end] != span.text:
            problems.append(f"{doc.id}: span text {span.text!r} differs from the text at {span.start}-{span.end}")
    for neg in doc.negatives:
        if neg.label not in NEGATIVE_KINDS:
            problems.append(f"{doc.id}: unknown negative kind {neg.label!r}")
        if not 0 <= neg.start < neg.end <= n:
            problems.append(f"{doc.id}: negative {neg.start}-{neg.end} outside text")
        if any(neg.overlaps(s) for s in doc.spans):
            problems.append(f"{doc.id}: negative {neg.text!r} overlaps a gold span")
    ordered = sorted(doc.spans, key=lambda s: (s.start, s.end))
    for a, b in itertools.pairwise(ordered):
        if a.overlaps(b):
            problems.append(f"{doc.id}: overlapping gold spans {a.start}-{a.end} and {b.start}-{b.end}")
    for rendering in doc.renderings:
        problems += _validate_rendering(doc.id, rendering, root)
        problems += _validate_visible(doc, rendering)
    return problems


def _validate_visible(doc: Document, rendering: Rendering) -> list[str]:
    if rendering.visible is None:
        return []
    problems: list[str] = []
    ranges = list(rendering.visible)
    if not ranges:
        problems.append(f"{doc.id}: rendering {rendering.id} shows no text")
    if ranges != sorted(ranges) or any(a[1] > b[0] for a, b in itertools.pairwise(ranges)):
        problems.append(f"{doc.id}: rendering {rendering.id} has unsorted or overlapping visible ranges")
    for start, end in ranges:
        if not 0 <= start < end <= len(doc.text):
            problems.append(f"{doc.id}: rendering {rendering.id} visible range {start}-{end} is outside the text")
    for span in doc.spans:
        inside = any(a <= span.start and span.end <= b for a, b in ranges)
        touches = any(span.start < b and a < span.end for a, b in ranges)
        if touches and not inside:
            problems.append(f"{doc.id}: rendering {rendering.id} cuts gold span {span.start}-{span.end}")
    return problems


def _validate_rendering(doc_id: str, rendering: Rendering, root: Path) -> list[str]:
    problems: list[str] = []
    if rendering.format not in FORMATS:
        problems.append(f"{doc_id}: rendering {rendering.id} has unknown format {rendering.format!r}")
    if rendering.channel not in CHANNELS:
        problems.append(f"{doc_id}: rendering {rendering.id} has unknown channel {rendering.channel!r}")
    if rendering.capture not in CAPTURES:
        problems.append(f"{doc_id}: rendering {rendering.id} has unknown capture {rendering.capture!r}")
    if rendering.capture != "digital" and rendering.severity not in SEVERITIES:
        problems.append(f"{doc_id}: rendering {rendering.id} needs a severity in {SEVERITIES}")
    if rendering.format in IMAGE_FORMATS and rendering.channel != "ocr":
        problems.append(f"{doc_id}: image rendering {rendering.id} must use the ocr channel")
    if not (root / rendering.path).is_file():
        problems.append(f"{doc_id}: rendering file {rendering.path} is missing")
    return problems


def _validate_checksums(dataset: Dataset) -> list[str]:
    problems: list[str] = []
    for rel, expected in dataset.manifest.get("checksums", {}).items():
        path = dataset.root / rel
        if not path.is_file():
            problems.append(f"manifest: {rel} is listed but missing")
        elif sha256_file(path) != expected:
            problems.append(f"manifest: {rel} changed after the manifest was written")
    return problems
