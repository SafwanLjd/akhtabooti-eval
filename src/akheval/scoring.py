"""Score stored predictions against the gold dataset.

Text runs score spans (strict, lenient, any_type), values and document labels
per document. Pipeline runs score values and document labels per rendering,
since extracted text has no gold offsets, and measure extraction quality.
Extract runs measure extraction quality only. A rendering that shows part of
a document, such as one face of an ID card, is scored against the gold values
and the text of that part only.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from rapidfuzz.distance import Levenshtein

from .config import EXTRACT, PIPELINE, TEXT
from .dataset import Document, ExpectedValue, Rendering, Span
from .detectors.base import Detection
from .matching import (
    ANY,
    ANY_TYPE,
    BIDI_CONTROLS,
    LENIENT,
    SPAN_MODES,
    STRICT,
    VALUE,
    VALUE_LENIENT,
    VALUE_MODES,
    VALUE_STRICT,
    Value,
    align_spans,
    document_counts,
    match_values,
    normalize_value,
    span_counts,
    value_counts,
)
from .metrics import DIGITS, Counts, bootstrap, tally, total
from .taxonomy import LABELS, family

DOCUMENT = "document"
TEXT_LEVELS = (STRICT, LENIENT, ANY_TYPE, VALUE_LENIENT, VALUE, VALUE_STRICT, DOCUMENT)
RENDERING_LEVELS = (VALUE_LENIENT, VALUE, VALUE_STRICT, DOCUMENT)
LEVELS_BY_MODE: dict[str, tuple[str, ...]] = {TEXT: TEXT_LEVELS, PIPELINE: RENDERING_LEVELS, EXTRACT: ()}
MAIN_LEVEL: dict[str, str | None] = {TEXT: LENIENT, PIPELINE: VALUE_LENIENT, EXTRACT: None}

DOCUMENT_FACETS = ("language", "doc_type")
RENDERING_FACETS = (*DOCUMENT_FACETS, "format", "channel", "capture", "severity")
OVERALL = "all"
BY_LABEL = "label"
BY_FAMILY = "family"
CONTEXT_CHARS = 40

FALSE_POSITIVE = "false_positive"
FALSE_NEGATIVE = "false_negative"
BOUNDARY = "boundary"
WRONG_TYPE = "wrong_type"
VALUE_NEAR = "value_near"
VALUE_PARTIAL = "value_partial"
VALUE_MISSED = "value_missed"
VALUE_SPURIOUS = "value_spurious"
DOC_FALSE_POSITIVE = "doc_false_positive"
DOC_FALSE_NEGATIVE = "doc_false_negative"
EXTRACTION_FAILED = "extraction_failed"
DETECTION_FAILED = "detection_failed"
LOOK_ALIKE_ERRORS = (FALSE_POSITIVE, VALUE_SPURIOUS)
NO_LOOK_ALIKE = "none"


@dataclass
class Prediction:
    """One line of ``predictions.jsonl``: what a run produced for one document or rendering."""

    item_id: str
    doc_id: str
    rendering_id: str | None = None
    extractor: str | None = None
    detection: Detection | None = None
    extracted: str | None = None
    extraction_error: str | None = None
    detection_error: str | None = None
    extract_seconds: float | None = None
    detect_seconds: float | None = None

    @property
    def seconds(self) -> float | None:
        parts = [s for s in (self.extract_seconds, self.detect_seconds) if s is not None]
        return sum(parts) if parts else None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["detection"] = None if self.detection is None else self.detection.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Prediction:
        fields = dict(d)
        detection = fields.pop("detection", None)
        return cls(**fields, detection=None if detection is None else Detection.from_dict(detection))


@dataclass(frozen=True, slots=True)
class ExtractionQuality:
    """Edit counts of extracted text against the text a file shows; add them to micro-average.

    ``pii_words`` counts the words of the gold values the file shows, ``pii_found`` those the extracted text keeps.
    """

    items: int = 0
    ref_chars: int = 0
    char_edits: int = 0
    ref_words: int = 0
    word_edits: int = 0
    words_found: int = 0
    pii_words: int = 0
    pii_found: int = 0

    def __add__(self, other: ExtractionQuality) -> ExtractionQuality:
        return ExtractionQuality(
            self.items + other.items,
            self.ref_chars + other.ref_chars,
            self.char_edits + other.char_edits,
            self.ref_words + other.ref_words,
            self.word_edits + other.word_edits,
            self.words_found + other.words_found,
            self.pii_words + other.pii_words,
            self.pii_found + other.pii_found,
        )

    @property
    def cer(self) -> float | None:
        return self.char_edits / self.ref_chars if self.ref_chars else None

    @property
    def wer(self) -> float | None:
        return self.word_edits / self.ref_words if self.ref_words else None

    @property
    def token_recall(self) -> float | None:
        return self.words_found / self.ref_words if self.ref_words else None

    @property
    def pii_token_recall(self) -> float | None:
        return self.pii_found / self.pii_words if self.pii_words else None

    def to_dict(self) -> dict[str, Any]:
        rates = {
            "cer": self.cer,
            "wer": self.wer,
            "token_recall": self.token_recall,
            "pii_token_recall": self.pii_token_recall,
        }
        return asdict(self) | {k: None if v is None else round(v, DIGITS) for k, v in rates.items()}


# Whitespace and list separators split words: a CSV export joins cells with bare commas, the canonical text
# separates table cells with " | ", and log lines write fields as SRC=192.0.2.10.
WORD_BREAKS = re.compile("[\\s,;|=\u060c\u061b]+")
# Punctuation at the edge of a word, such as the full stop after a phone number or the colon after a label.
WORD_EDGES = ".:!?()[]{}\"'\u00ab\u00bb\u061f"
# Angle brackets around an e-mail address go; a "<" at the end of a word stays, since MRZ lines end in filler.
OPENING_EDGES = WORD_EDGES + "<"
CLOSING_EDGES = WORD_EDGES + ">"


def words(text: str) -> list[str]:
    """Words of ``text`` in order, split at whitespace and list separators, without bidi marks or edge punctuation."""
    plain = BIDI_CONTROLS.sub("", unicodedata.normalize("NFKC", text))
    return [bare for word in WORD_BREAKS.split(plain) if (bare := word.lstrip(OPENING_EDGES).rstrip(CLOSING_EDGES))]


def bare_words(text: str) -> Counter[str]:
    """Words of ``text`` as a multiset; see ``words``."""
    return Counter(words(text))


def extraction_quality(reference: str, extracted: str, values: Iterable[str] = ()) -> ExtractionQuality:
    """CER and WER by Levenshtein distance, token recall and PII token recall as order-free multiset overlap.

    All four rates compare the ``words`` of each text, so the " | " and ": " the canonical text puts between table
    cells and after labels, and the bidi marks OCR puts around Latin runs, count as neither words nor characters.
    CER compares the words joined by single spaces. PII token recall counts the words of the gold ``values``.
    """
    ref_seq, hyp_seq = words(reference), words(extracted)
    ref, hyp = " ".join(ref_seq), " ".join(hyp_seq)
    ref_words, hyp_words = Counter(ref_seq), Counter(hyp_seq)
    pii_words = sum((bare_words(value) for value in values), Counter[str]())
    return ExtractionQuality(
        items=1,
        ref_chars=len(ref),
        char_edits=Levenshtein.distance(ref, hyp),
        ref_words=ref_words.total(),
        word_edits=Levenshtein.distance(ref_seq, hyp_seq),
        words_found=(ref_words & hyp_words).total(),
        pii_words=pii_words.total(),
        pii_found=(pii_words & hyp_words).total(),
    )


@dataclass(frozen=True)
class ErrorRecord:
    kind: str
    label: str
    text: str
    doc_id: str
    rendering_id: str | None
    language: str
    context: str = ""
    gold_label: str | None = None
    gold_text: str | None = None
    negative_kind: str | None = None
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ItemScore:
    """Scores of one document (text runs) or one rendering (pipeline and extract runs)."""

    item_id: str
    doc_id: str
    rendering_id: str | None
    facets: dict[str, str]
    levels: dict[str, dict[str, Counts]] = field(default_factory=dict)
    errors: list[ErrorRecord] = field(default_factory=list)
    quality: ExtractionQuality | None = None
    seconds: float | None = None
    chars: int = 0


def context_window(text: str, start: int, end: int, width: int = CONTEXT_CHARS) -> str:
    """Text around ``start:end`` with the hit in double brackets, on one line."""
    before = text[max(0, start - width) : start]
    after = text[end : end + width]
    return " ".join(f"{before}[[{text[start:end]}]]{after}".split())


def _value_context(text: str, value: str) -> str:
    start = text.find(value)
    return context_window(text, start, start + len(value)) if start >= 0 else ""


def _negative_for_span(doc: Document, span: Span) -> str | None:
    return next((n.label for n in doc.negatives if n.overlaps(span)), None)


def _negative_for_value(doc: Document, value: str) -> str | None:
    key = normalize_value(value)
    for negative in doc.negatives:
        other = normalize_value(negative.text or doc.text[negative.start : negative.end])
        if key and other and (key in other or other in key):
            return negative.label
    return None


class _ItemBuilder:
    """Collects the counts and errors of one item.

    ``expected`` holds the gold values the item shows and ``reference`` the canonical text it shows: the whole
    document for a text item, the visible part of the document for a rendering.
    """

    def __init__(self, doc: Document, rendering: Rendering | None, prediction: Prediction) -> None:
        self.doc = doc
        self.expected: list[ExpectedValue] = doc.expected_values()
        self.reference = doc.text
        facets = {"language": doc.language, "doc_type": doc.doc_type}
        if rendering is not None:
            self.expected = doc.visible_values(rendering)
            self.reference = doc.visible_text(rendering)
            facets |= {
                "format": rendering.format,
                "channel": rendering.channel,
                "capture": rendering.capture,
                "severity": rendering.severity or "none",
            }
        self.item = ItemScore(
            item_id=prediction.item_id,
            doc_id=doc.id,
            rendering_id=None if rendering is None else rendering.id,
            facets=facets,
            seconds=prediction.seconds,
        )

    def error(self, kind: str, label: str, text: str, **extra: Any) -> None:
        self.item.errors.append(
            ErrorRecord(kind, label, text, self.doc.id, self.item.rendering_id, self.doc.language, **extra)
        )

    def score_spans(self, detection: Detection) -> None:
        doc = self.doc
        alignment = align_spans(doc.spans, detection.spans)
        for mode in SPAN_MODES:
            self.item.levels[mode] = span_counts(alignment, mode)
        for kind, pairs in ((BOUNDARY, alignment.boundary), (WRONG_TYPE, alignment.wrong_type)):
            for pair in pairs:
                p, g = pair.pred, pair.gold
                context = context_window(doc.text, p.start, p.end)
                self.error(kind, p.label, p.text, context=context, gold_label=g.label, gold_text=g.text, score=p.score)
        for g in alignment.missed:
            self.error(FALSE_NEGATIVE, g.label, g.text, context=context_window(doc.text, g.start, g.end))
        for p in alignment.spurious:
            context = context_window(doc.text, p.start, p.end)
            negative = _negative_for_span(doc, p)
            self.error(FALSE_POSITIVE, p.label, p.text, context=context, negative_kind=negative, score=p.score)

    def score_values(self, predicted: Iterable[Value], text: str, *, report_errors: bool) -> None:
        """Value counts at every value level; ``report_errors`` records each unmatched or inexact value."""
        alignment = match_values([(v.label, v.value) for v in self.expected], predicted)
        for mode in VALUE_MODES:
            self.item.levels[mode] = value_counts(alignment, mode)
        if not report_errors:
            return
        for kind, pairs in ((VALUE_NEAR, alignment.near), (VALUE_PARTIAL, alignment.partial)):
            for pair in pairs:
                context = _value_context(text, pair.pred)
                self.error(kind, pair.label, pair.pred, context=context, gold_text=pair.gold, score=pair.ratio)
        for label, value in alignment.missed:
            self.error(VALUE_MISSED, label, value, context=_value_context(self.reference, value))
        for label, value in alignment.spurious:
            context = _value_context(text, value)
            self.error(
                VALUE_SPURIOUS, label, value, context=context, negative_kind=_negative_for_value(self.doc, value)
            )

    def score_document(self, predicted: set[str]) -> None:
        gold = {v.label for v in self.expected}
        self.item.levels[DOCUMENT] = document_counts(gold, predicted)
        for label in sorted(predicted - gold):
            self.error(DOC_FALSE_POSITIVE, label, "")
        for label in sorted(gold - predicted):
            self.error(DOC_FALSE_NEGATIVE, label, "")


def score_text(doc: Document, prediction: Prediction) -> ItemScore:
    builder = _ItemBuilder(doc, None, prediction)
    detection = prediction.detection or Detection()
    if prediction.detection_error:
        builder.error(DETECTION_FAILED, "", prediction.detection_error)
    builder.score_spans(detection)
    builder.score_values(detection.predicted_values(doc.text), doc.text, report_errors=False)
    builder.score_document(detection.predicted_labels())
    builder.item.chars = len(doc.text)
    return builder.item


def score_rendering(doc: Document, rendering: Rendering, prediction: Prediction, *, detect: bool) -> ItemScore:
    builder = _ItemBuilder(doc, rendering, prediction)
    extracted = prediction.extracted
    if prediction.extraction_error:
        builder.error(EXTRACTION_FAILED, "", prediction.extraction_error)
    if extracted is not None:
        builder.item.quality = extraction_quality(builder.reference, extracted, [v.value for v in builder.expected])
        builder.item.chars = len(extracted)
    if not detect:
        return builder.item
    if prediction.detection_error:
        builder.error(DETECTION_FAILED, "", prediction.detection_error)
    detection = prediction.detection or Detection()
    text = extracted or ""
    builder.score_values(detection.predicted_values(text), text, report_errors=True)
    builder.score_document(detection.predicted_labels())
    return builder.item


def score_predictions(documents: Sequence[Document], mode: str, predictions: Iterable[Prediction]) -> list[ItemScore]:
    """Score every prediction; raises KeyError when one names a document or rendering the dataset lacks."""
    docs = {d.id: d for d in documents}
    renderings = {r.id: r for d in documents for r in d.renderings}
    items = []
    for prediction in sorted(predictions, key=lambda p: p.item_id):
        doc = docs[prediction.doc_id]
        if mode == TEXT:
            items.append(score_text(doc, prediction))
        else:
            rendering = renderings[str(prediction.rendering_id)]
            items.append(score_rendering(doc, rendering, prediction, detect=mode == PIPELINE))
    return items


def family_of(label: str) -> str:
    """Taxonomy family, or "other" for a label outside the taxonomy."""
    return family(label) if label in LABELS else "other"


def _contributions(item: ItemScore, level: str, by: str) -> dict[str, Counts]:
    by_label = {label: c for label, c in item.levels.get(level, {}).items() if label != ANY}
    if by == BY_LABEL:
        return by_label
    if by == BY_FAMILY:
        families: dict[str, Counts] = {}
        for label, counts in by_label.items():
            tally(families, family_of(label), counts)
        return families
    micro = total(by_label.values())
    return {OVERALL: micro} if by == OVERALL else {item.facets[by]: micro}


def _per_document(
    items: Sequence[ItemScore], contribute: Callable[[ItemScore], dict[str, Counts]]
) -> list[dict[str, Counts]]:
    """Sum the contributions of each document's items, in document id order.

    A resample then draws whole documents, so the correlated errors of one document's renderings stay together.
    """
    documents: dict[str, dict[str, Counts]] = {}
    for item in items:
        sums = documents.setdefault(item.doc_id, {})
        for group, counts in contribute(item).items():
            tally(sums, group, counts)
    return [documents[doc_id] for doc_id in sorted(documents)]


def document_count(items: Sequence[ItemScore]) -> int:
    return len({item.doc_id for item in items})


def series(items: Sequence[ItemScore], level: str, by: str = OVERALL) -> dict[str, list[Counts]]:
    """Per-group counts of every document, zero where a document has no item in the group; ready for bootstrap."""
    contributions = _per_document(items, lambda item: _contributions(item, level, by))
    groups = sorted({key for c in contributions for key in c})
    return {group: [c.get(group, Counts()) for c in contributions] for group in groups}


def any_series(items: Sequence[ItemScore]) -> list[Counts]:
    """Document-level counts of the ANY pseudo-label per document."""
    contributions = _per_document(items, lambda item: {ANY: item.levels.get(DOCUMENT, {}).get(ANY, Counts())})
    return [c[ANY] for c in contributions]


def grouped(items: Sequence[ItemScore], level: str, by: str = OVERALL) -> dict[str, Counts]:
    return {group: total(counts) for group, counts in series(items, level, by).items()}


def quality_grouped(items: Sequence[ItemScore], by: str = OVERALL) -> dict[str, ExtractionQuality]:
    groups: dict[str, ExtractionQuality] = {}
    for item in items:
        if item.quality is not None:
            key = OVERALL if by == OVERALL else item.facets[by]
            groups[key] = groups.get(key, ExtractionQuality()) + item.quality
    return dict(sorted(groups.items()))


def label_order(labels: Iterable[str]) -> list[str]:
    """Taxonomy order first, anything else alphabetically after."""
    present = set(labels)
    return [label for label in LABELS if label in present] + sorted(present - set(LABELS))


def look_alike_counts(items: Sequence[ItemScore]) -> dict[str, int]:
    """False positives per look-alike kind they hit; NO_LOOK_ALIKE for those that hit none."""
    kinds = Counter(
        e.negative_kind or NO_LOOK_ALIKE for item in items for e in item.errors if e.kind in LOOK_ALIKE_ERRORS
    )
    return dict(sorted(kinds.items()))


def facets_of(mode: str) -> tuple[str, ...]:
    return DOCUMENT_FACETS if mode == TEXT else RENDERING_FACETS


def summarize(mode: str, items: Sequence[ItemScore]) -> dict[str, Any]:
    """Everything ``summary.json`` holds about scores, errors and extraction quality."""
    levels: dict[str, Any] = {}
    for level in LEVELS_BY_MODE[mode]:
        overall = series(items, level).get(OVERALL, [])
        intervals = {name: None if iv is None else iv.to_dict() for name, iv in bootstrap(overall).items()}
        entry: dict[str, Any] = {
            "overall": total(overall).to_dict() | {"ci95": intervals},
            "by_label": {k: c.to_dict() for k, c in grouped(items, level, BY_LABEL).items()},
            "by_family": {k: c.to_dict() for k, c in grouped(items, level, BY_FAMILY).items()},
        }
        for facet in facets_of(mode):
            entry[f"by_{facet}"] = {k: c.to_dict() for k, c in grouped(items, level, facet).items()}
        levels[level] = entry
    summary: dict[str, Any] = {
        "mode": mode,
        "documents": document_count(items),
        "items": len(items),
        "main_level": MAIN_LEVEL[mode],
        "levels": levels,
        "errors": dict(sorted(Counter(e.kind for item in items for e in item.errors).items())),
        "false_positives_by_look_alike": look_alike_counts(items),
    }
    if DOCUMENT in levels:
        summary["document_any"] = total(any_series(items)).to_dict()
    if mode != TEXT:
        overall_quality = quality_grouped(items).get(OVERALL)
        extraction: dict[str, Any] = {"overall": None if overall_quality is None else overall_quality.to_dict()}
        for facet in RENDERING_FACETS:
            extraction[f"by_{facet}"] = {k: q.to_dict() for k, q in quality_grouped(items, facet).items()}
        summary["extraction"] = extraction
    return summary
