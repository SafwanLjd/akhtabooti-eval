"""Shared pieces of the token-classification detectors.

The models read a bounded number of tokens, so a detector cuts a long document
into chunks on line boundaries, runs the model per chunk and shifts the
entities back to document offsets. Chunks overlap by a few lines; the same
entity found in two chunks is kept once.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from ..dataset import Span
from ..taxonomy import LabelMap, map_label, strip_bio
from .base import Detection

Measure = Callable[[str], int]

OUTSIDE = "O"
_LINE = re.compile(r"[^\n]*\n|[^\n]+")
_WORD = re.compile(r"\s*\S+\s*|\s+")
SAME_LINE_GAP = re.compile(r"[^\S\n\r]*")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A slice of a document; ``start`` is the offset of ``text`` in the document."""

    start: int
    text: str

    @property
    def end(self) -> int:
        return self.start + len(self.text)


@dataclass(frozen=True, slots=True)
class Entity:
    """An entity with the model's own label (BIO prefix removed).

    ``score`` is the mean probability over ``tokens`` tagged items.
    """

    start: int
    end: int
    label: str
    score: float
    tokens: int = 1

    def overlaps(self, other: Entity) -> bool:
        return self.start < other.end and other.start < self.end

    def shifted(self, offset: int) -> Entity:
        return Entity(self.start + offset, self.end + offset, self.label, self.score, self.tokens)


@dataclass(frozen=True, slots=True)
class _Piece:
    start: int
    end: int
    size: int


def chunk_text(text: str, budget: int, overlap: int = 0, measure: Measure = len) -> list[Chunk]:
    """Split ``text`` into chunks whose size is at most ``budget``.

    ``measure`` gives the size of a piece of text, in characters by default. You
    can pass a token counter instead; it must be additive across whitespace,
    which holds for WordPiece and over-counts slightly for SentencePiece-style
    tokenizers. Chunks end on line boundaries. A line over the budget is cut
    between words, and a word over the budget is cut between characters. Each
    chunk repeats up to ``overlap`` units of whole pieces from the end of the
    previous one.
    """
    if budget < 1:
        raise ValueError(f"chunk budget must be positive, got {budget}")
    if not 0 <= overlap < budget:
        raise ValueError(f"overlap must lie in [0, {budget}), got {overlap}")
    pieces = _pieces(text, budget, measure)
    chunks: list[Chunk] = []
    first = 0
    while first < len(pieces):
        stop, size = first, 0
        while stop < len(pieces) and size + pieces[stop].size <= budget:
            size += pieces[stop].size
            stop += 1
        stop = max(stop, first + 1)
        chunks.append(Chunk(pieces[first].start, text[pieces[first].start : pieces[stop - 1].end]))
        if stop == len(pieces):
            break
        first = _overlap_start(pieces, first, stop, min(overlap, budget - pieces[stop].size))
    return chunks


def _overlap_start(pieces: Sequence[_Piece], first: int, stop: int, room: int) -> int:
    """Index of the first piece the next chunk repeats from the chunk ``pieces[first:stop]``."""
    start, carried = stop, 0
    while start - 1 > first and carried + pieces[start - 1].size <= room:
        start -= 1
        carried += pieces[start].size
    return start


def _pieces(text: str, budget: int, measure: Measure) -> list[_Piece]:
    pieces: list[_Piece] = []
    for line in _LINE.finditer(text):
        size = measure(line.group())
        if size <= budget:
            pieces.append(_Piece(line.start(), line.end(), size))
            continue
        for word in _WORD.finditer(line.group()):
            pieces += _split_word(text, line.start() + word.start(), line.start() + word.end(), budget, measure)
    return pieces


def _split_word(text: str, start: int, end: int, budget: int, measure: Measure) -> list[_Piece]:
    """Cut ``text[start:end]`` into the fewest left-to-right pieces that fit the budget."""
    pieces: list[_Piece] = []
    while start < end:
        size = measure(text[start:end])
        if size <= budget:
            pieces.append(_Piece(start, end, size))
            break
        lo, hi = start + 1, end - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if measure(text[start:mid]) <= budget:
                lo = mid
            else:
                hi = mid - 1
        pieces.append(_Piece(start, lo, measure(text[start:lo])))
        start = lo
    return pieces


def group_bio(tags: Iterable[tuple[int, int, str, float]]) -> list[Entity]:
    """Group ``(start, end, tag, probability)`` items into entities.

    A ``B-`` tag or a change of label starts an entity, an ``I-`` tag with the
    same label extends it and ``O`` ends it. The entity score is the mean
    probability of its items.
    """
    entities: list[Entity] = []
    current: list[tuple[int, int, str, float]] = []
    for item in tags:
        tag = item[2]
        label = strip_bio(tag)
        if current and (tag == OUTSIDE or tag.startswith("B-") or label != current[0][2]):
            entities.append(_entity(current))
            current = []
        if tag != OUTSIDE:
            current.append((item[0], item[1], label, item[3]))
    if current:
        entities.append(_entity(current))
    return entities


def _entity(items: Sequence[tuple[int, int, str, float]]) -> Entity:
    return Entity(items[0][0], items[-1][1], items[0][2], sum(i[3] for i in items) / len(items), len(items))


def join_fragments(text: str, entities: Sequence[Entity]) -> list[Entity]:
    """Join consecutive entities of one label that only spaces or tabs separate.

    Token classifiers often restart an entity with a fresh ``B-`` tag inside a
    street name or an email address. The joined score is the token-weighted
    mean of the parts. A line break keeps entities apart, so list items stay
    separate.
    """
    joined: list[Entity] = []
    for entity in sorted(entities, key=lambda e: (e.start, e.end)):
        last = joined[-1] if joined else None
        if (
            last is not None
            and last.label == entity.label
            and last.end <= entity.start
            and SAME_LINE_GAP.fullmatch(text, last.end, entity.start)
        ):
            tokens = last.tokens + entity.tokens
            score = (last.score * last.tokens + entity.score * entity.tokens) / tokens
            joined[-1] = Entity(last.start, entity.end, last.label, score, tokens)
        else:
            joined.append(entity)
    return joined


def merge_entities(entities: Iterable[Entity]) -> list[Entity]:
    """Drop duplicates that overlapping chunks produced.

    Of two overlapping entities the one with the higher score stays, then the
    longer one. A single chunk never yields overlapping entities, so any
    overlap comes from the shared lines of two chunks.
    """
    ranked = sorted(entities, key=lambda e: (-e.score, e.start - e.end, e.start, e.label))
    kept: list[Entity] = []
    for entity in ranked:
        if not any(entity.overlaps(k) for k in kept):
            kept.append(entity)
    return sorted(kept, key=lambda e: (e.start, e.end, e.label))


def to_detection(text: str, entities: Iterable[Entity], table: LabelMap) -> Detection:
    """Map entity labels onto the taxonomy and trim whitespace from span edges.

    ``raw_labels`` counts every entity by model label; ``unmapped`` counts the
    out-of-scope ones that the detection leaves out.
    """
    detection = Detection()
    for entity in entities:
        detection.raw_labels[entity.label] += 1
        label = map_label(table, entity.label)
        if label is None:
            detection.unmapped[entity.label] += 1
            continue
        span = trimmed_span(text, entity.start, entity.end, label, entity.score)
        if span is not None:
            detection.spans.append(span)
    return detection


def trimmed_span(text: str, start: int, end: int, label: str, score: float | None) -> Span | None:
    """Build a span without leading or trailing whitespace; ``None`` if nothing is left."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start == end:
        return None
    return Span(start, end, label, text[start:end], score)
