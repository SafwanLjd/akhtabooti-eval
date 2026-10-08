"""Pair predictions with gold annotations at span, value and document level.

Span pairing is one-to-one and greedy: identical boundaries and label first,
then the largest overlap with the same label, then the largest overlap with a
different label. The three span modes read the same pairing:

* ``strict``: a pair counts only with identical boundaries and label.
* ``lenient``: overlapping boundaries with the same label count.
* ``any_type``: any overlap counts, whatever the label. It answers whether the
  scanner flagged that stretch of PII at all.

Value pairing works on (label, string) pairs without offsets and is also
one-to-one and greedy within a label: equal normalised strings first, then near
matches (OCR noise), then partial matches, where one normalised string contains
the other or, for PERSON, ORG, LOCATION and ADDRESS, the two share at least
half of the smaller token set. A contained string needs 4 or more characters,
so a stray fragment such as "t" inside an organisation name earns no credit;
whole words match through the token rule.
The three value modes read that pairing:

* ``value_strict``: equal normalised strings only.
* ``value``: equal or near.
* ``value_lenient``: equal, near or partial. It credits a model that splits an
  address into street and city, as lenient span matching does.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence, Set
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .dataset import Span
from .metrics import FN, FP, TP, Counts, tally
from .taxonomy import CONTEXTUAL

STRICT = "strict"
LENIENT = "lenient"
ANY_TYPE = "any_type"
SPAN_MODES = (STRICT, LENIENT, ANY_TYPE)

ANY = "ANY"
"""Document-level pseudo-label: the document contains PII of any category."""

VALUE_STRICT = "value_strict"
VALUE = "value"
VALUE_LENIENT = "value_lenient"
VALUE_MODES = (VALUE_STRICT, VALUE, VALUE_LENIENT)

NEAR_RATIO = 0.85
"""Minimum rapidfuzz ratio (0..1) between normalised values for a near match."""

MIN_CONTAINED_CHARS = 4
"""Shortest normalised value that counts as contained in another for a partial match."""

Value = tuple[str, str]
"""(label, value) as a detector reports it."""


@dataclass(frozen=True, slots=True)
class SpanPair:
    gold: Span
    pred: Span


@dataclass
class SpanAlignment:
    exact: list[SpanPair] = field(default_factory=list)
    boundary: list[SpanPair] = field(default_factory=list)
    wrong_type: list[SpanPair] = field(default_factory=list)
    missed: list[Span] = field(default_factory=list)
    spurious: list[Span] = field(default_factory=list)


def overlap(a: Span, b: Span) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def _pair_by_overlap(
    gold: list[Span], pred: list[Span], *, same_label: bool
) -> tuple[list[SpanPair], list[Span], list[Span]]:
    candidates = [
        (-overlap(g, p), abs((g.end - g.start) - (p.end - p.start)), gi, pi)
        for gi, g in enumerate(gold)
        for pi, p in enumerate(pred)
        if (g.label == p.label) == same_label and overlap(g, p) > 0
    ]
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    pairs = []
    for _, _, gi, pi in sorted(candidates):
        if gi in used_gold or pi in used_pred:
            continue
        used_gold.add(gi)
        used_pred.add(pi)
        pairs.append(SpanPair(gold[gi], pred[pi]))
    rest_gold = [g for i, g in enumerate(gold) if i not in used_gold]
    rest_pred = [p for i, p in enumerate(pred) if i not in used_pred]
    return pairs, rest_gold, rest_pred


def _span_key(span: Span) -> tuple[int, int, str]:
    return span.start, span.end, span.label


def align_spans(gold: Sequence[Span], pred: Sequence[Span]) -> SpanAlignment:
    alignment = SpanAlignment()
    pred_left = sorted(pred, key=_span_key)
    gold_left = []
    for g in sorted(gold, key=_span_key):
        hit = next((p for p in pred_left if _span_key(p) == _span_key(g)), None)
        if hit is None:
            gold_left.append(g)
        else:
            pred_left.remove(hit)
            alignment.exact.append(SpanPair(g, hit))
    alignment.boundary, gold_left, pred_left = _pair_by_overlap(gold_left, pred_left, same_label=True)
    alignment.wrong_type, alignment.missed, alignment.spurious = _pair_by_overlap(
        gold_left, pred_left, same_label=False
    )
    return alignment


def span_counts(alignment: SpanAlignment, mode: str) -> dict[str, Counts]:
    """Per-label counts of one span mode."""
    if mode not in SPAN_MODES:
        raise ValueError(f"unknown span mode {mode!r}; expected one of {SPAN_MODES}")
    table: dict[str, Counts] = {}
    for pair in alignment.exact:
        tally(table, pair.gold.label, TP)
    for pair in alignment.boundary:
        if mode == STRICT:
            tally(table, pair.gold.label, FN)
            tally(table, pair.pred.label, FP)
        else:
            tally(table, pair.gold.label, TP)
    for pair in alignment.wrong_type:
        if mode == ANY_TYPE:
            tally(table, pair.gold.label, TP)
        else:
            tally(table, pair.gold.label, FN)
            tally(table, pair.pred.label, FP)
    for span in alignment.missed:
        tally(table, span.label, FN)
    for span in alignment.spurious:
        tally(table, span.label, FP)
    return table


ARABIC_INDIC_ZERO = 0x0660
EXTENDED_ARABIC_INDIC_ZERO = 0x06F0
_DIGITS = str.maketrans(
    {chr(zero + d): str(d) for zero in (ARABIC_INDIC_ZERO, EXTENDED_ARABIC_INDIC_ZERO) for d in range(10)}
)
_LETTERS = str.maketrans(
    {
        "\u0623": "\u0627",  # alef with hamza above -> alef
        "\u0625": "\u0627",  # alef with hamza below -> alef
        "\u0622": "\u0627",  # alef with madda -> alef
        "\u0671": "\u0627",  # alef wasla -> alef
        "\u0649": "\u064a",  # alef maksura -> yeh
        "\u06cc": "\u064a",  # Farsi yeh -> yeh
        "\u0629": "\u0647",  # taa marbuta -> heh
    }
)
# Quranic marks, harakat, superscript alef, small high marks, tatweel.
_MARKS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")
# Arabic letter mark, LRM, RLM, embeddings, overrides and isolates: OCR output carries them, the gold text does not.
_BIDI_CHARS = "\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069"
BIDI_CONTROLS = re.compile(f"[{_BIDI_CHARS}]")
# Whitespace, dashes, dots, brackets, MRZ filler, bidi controls.
_SEPARATORS = re.compile("[\\s\\-\u2010-\u2015\u2212.()\\[\\]{}<" + _BIDI_CHARS + "]+")
# Whitespace and the punctuation that separates the words of an address or a name list.
_TOKEN_BREAKS = re.compile("[\\s,;:/|\u060c\u061b]+")


def normalize_value(value: str) -> str:
    """Canonical form for value comparison: NFKC, ASCII digits, plain Arabic letters, no separators, casefolded."""
    text = unicodedata.normalize("NFKC", value).translate(_DIGITS)
    text = _MARKS.sub("", text).translate(_LETTERS)
    return _SEPARATORS.sub("", text).casefold()


def value_tokens(value: str) -> frozenset[str]:
    """Normalised words of a value; separators inside a word, such as the dashes of a phone, stay joined."""
    return frozenset(token for word in _TOKEN_BREAKS.split(value) if (token := normalize_value(word)))


@dataclass(frozen=True, slots=True)
class ValuePair:
    label: str
    gold: str
    pred: str
    ratio: float


@dataclass
class ValueAlignment:
    exact: list[ValuePair] = field(default_factory=list)
    near: list[ValuePair] = field(default_factory=list)
    partial: list[ValuePair] = field(default_factory=list)
    missed: list[Value] = field(default_factory=list)
    spurious: list[Value] = field(default_factory=list)


def _unique(values: Iterable[Value]) -> dict[Value, str]:
    """Map (label, normalised value) to the first original string, dropping duplicates."""
    unique: dict[Value, str] = {}
    for label, value in values:
        unique.setdefault((label, normalize_value(value)), value)
    return unique


def similarity(a: str, b: str) -> float:
    """rapidfuzz ratio of two normalised strings, scaled to 0..1."""
    return fuzz.ratio(a, b) / 100


def is_partial(gold: str, pred: str, *, by_tokens: bool) -> bool:
    """One normalised value contains the other or, with ``by_tokens``, shares at least half of the smaller token set."""
    shorter, longer = sorted((normalize_value(gold), normalize_value(pred)), key=len)
    if len(shorter) >= MIN_CONTAINED_CHARS and shorter in longer:
        return True
    if not by_tokens:
        return False
    gold_tokens, pred_tokens = value_tokens(gold), value_tokens(pred)
    return 2 * len(gold_tokens & pred_tokens) >= min(len(gold_tokens), len(pred_tokens)) > 0


def _pair(
    gold_left: dict[Value, str], pred_left: dict[Value, str], accept: Callable[[Value, Value], bool]
) -> list[ValuePair]:
    """Greedy one-to-one pairing within a label, most similar first; paired entries leave both pools."""
    candidates = sorted(
        (-similarity(g[1], p[1]), g, p) for g in gold_left for p in pred_left if g[0] == p[0] and accept(g, p)
    )
    pairs = []
    for neg_ratio, g, p in candidates:
        if g in gold_left and p in pred_left:
            pairs.append(ValuePair(g[0], gold_left.pop(g), pred_left.pop(p), -neg_ratio))
    return pairs


def match_values(gold: Iterable[Value], pred: Iterable[Value], near_ratio: float = NEAR_RATIO) -> ValueAlignment:
    """Pair distinct (label, value) pairs: exact after normalisation, then near above ``near_ratio``, then partial."""
    gold_left, pred_left = _unique(gold), _unique(pred)
    alignment = ValueAlignment()
    for key in sorted(gold_left.keys() & pred_left.keys()):
        alignment.exact.append(ValuePair(key[0], gold_left.pop(key), pred_left.pop(key), 1.0))
    alignment.near = _pair(gold_left, pred_left, lambda g, p: similarity(g[1], p[1]) >= near_ratio)
    alignment.partial = _pair(
        gold_left, pred_left, lambda g, p: is_partial(gold_left[g], pred_left[p], by_tokens=g[0] in CONTEXTUAL)
    )
    alignment.missed = sorted((key[0], value) for key, value in gold_left.items())
    alignment.spurious = sorted((key[0], value) for key, value in pred_left.items())
    return alignment


_ACCEPTED_PAIR_LISTS = {VALUE_STRICT: 1, VALUE: 2, VALUE_LENIENT: 3}
"""Each value mode counts the first N pair lists, in the order exact, near, partial, as matches."""


def value_counts(alignment: ValueAlignment, mode: str) -> dict[str, Counts]:
    """Per-label counts of one value mode. A pair the mode does not accept counts as one miss and one false alarm."""
    if mode not in VALUE_MODES:
        raise ValueError(f"unknown value mode {mode!r}; expected one of {VALUE_MODES}")
    table: dict[str, Counts] = {}
    for rank, pairs in enumerate((alignment.exact, alignment.near, alignment.partial)):
        outcome = TP if rank < _ACCEPTED_PAIR_LISTS[mode] else FN + FP
        for pair in pairs:
            tally(table, pair.label, outcome)
    for label, _ in alignment.missed:
        tally(table, label, FN)
    for label, _ in alignment.spurious:
        tally(table, label, FP)
    return table


def document_counts(gold: Set[str], pred: Set[str]) -> dict[str, Counts]:
    """Per-label presence counts for one document, plus the ANY pseudo-label."""
    table = {label: TP if label in pred else FN for label in gold}
    table.update(dict.fromkeys(pred - gold, FP))
    if gold or pred:
        table[ANY] = TP if gold and pred else (FN if gold else FP)
    return table
