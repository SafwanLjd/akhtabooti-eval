"""Chunking, BIO grouping and span merging shared by the model detectors."""

from __future__ import annotations

import itertools
import re

import pytest

from akheval.detectors.chunking import (
    Chunk,
    Entity,
    chunk_text,
    group_bio,
    join_fragments,
    merge_entities,
    to_detection,
    trimmed_span,
)
from akheval.taxonomy import NYM_LABELS

ARABIC_LINES = [
    "السيد محمد أحمد الخطيب يسكن في عمّان.",
    "رقم الهاتف 0791234567 والبريد omar@example.org",
    "تمت مراجعة الطلب في مكتب الزرقاء يوم الأحد.",
    "الرقم الوطني 9871234567 مسجل لدى الدائرة.",
]


def _covers(text: str, chunks: list[Chunk]) -> bool:
    """Every chunk is a verbatim slice, and together they cover the text without gaps."""
    position = 0
    for chunk in chunks:
        if text[chunk.start : chunk.end] != chunk.text or chunk.start > position:
            return False
        position = max(position, chunk.end)
    return position == len(text)


def test_empty_and_short_text() -> None:
    assert chunk_text("", 10) == []
    assert chunk_text("one line\n", 100) == [Chunk(0, "one line\n")]


def test_chunks_end_on_line_boundaries_without_overlap() -> None:
    text = "\n".join(ARABIC_LINES * 5)
    chunks = chunk_text(text, 120)
    assert len(chunks) > 1
    assert _covers(text, chunks)
    assert "".join(c.text for c in chunks) == text
    assert all(len(c.text) <= 120 for c in chunks)
    assert all(c.text.endswith("\n") for c in chunks[:-1])


def test_overlap_repeats_whole_trailing_lines() -> None:
    lines = [f"line {i:02d} " + "x" * 20 + "\n" for i in range(20)]
    text = "".join(lines)
    chunks = chunk_text(text, 100, overlap=40)
    assert _covers(text, chunks)
    for previous, current in itertools.pairwise(chunks):
        shared = previous.end - current.start
        assert 0 < shared <= 40
        assert text[current.start - 1] == "\n"


def test_overlap_never_crowds_out_the_next_line() -> None:
    text = "a" * 30 + "\n" + "b" * 15 + "\n" + "c" * 95 + "\n"
    chunks = chunk_text(text, 100, overlap=20)
    assert _covers(text, chunks)
    assert [c.text[0] for c in chunks] == ["a", "c"]


def test_line_longer_than_budget_splits_between_words() -> None:
    line = " ".join(f"كلمة{i}" for i in range(60))
    chunks = chunk_text(line, 50)
    assert _covers(line, chunks)
    assert all(len(c.text) <= 50 for c in chunks)
    for chunk in chunks[1:]:
        assert line[chunk.start - 1] == " "


def test_word_longer_than_budget_splits_between_characters() -> None:
    text = "start " + "Z" * 95 + " end"
    chunks = chunk_text(text, 30)
    assert _covers(text, chunks)
    assert all(len(c.text) <= 30 for c in chunks)


def test_custom_measure_counts_words() -> None:
    text = "\n".join(" ".join(["w"] * 4) for _ in range(10))
    chunks = chunk_text(text, 9, measure=lambda s: len(s.split()))
    assert all(len(c.text.split()) <= 9 for c in chunks)
    assert _covers(text, chunks)


@pytest.mark.parametrize(("budget", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_budget_or_overlap(budget: int, overlap: int) -> None:
    with pytest.raises(ValueError, match=r"budget|overlap"):
        chunk_text("text", budget, overlap)


def test_group_bio_follows_prefixes_and_labels() -> None:
    tags = [
        (0, 4, "B-GIVEN_NAME", 0.9),
        (5, 9, "I-GIVEN_NAME", 0.7),
        (10, 14, "I-SURNAME", 0.8),
        (15, 18, "O", 1.0),
        (19, 22, "I-CITY", 0.6),
        (23, 26, "B-CITY", 0.4),
    ]
    entities = group_bio(tags)
    assert [(e.start, e.end, e.label, e.tokens) for e in entities] == [
        (0, 9, "GIVEN_NAME", 2),
        (10, 14, "SURNAME", 1),
        (19, 22, "CITY", 1),
        (23, 26, "CITY", 1),
    ]
    assert [e.score for e in entities] == pytest.approx([0.8, 0.8, 0.6, 0.4])


def test_join_fragments_on_one_line_only() -> None:
    text = "شارع الجامعة\nعمّان إربد"  # noqa: RUF001
    entities = [
        Entity(0, 4, "STREET_NAME", 1.0, 1),
        Entity(5, 12, "STREET_NAME", 0.5, 3),
        Entity(13, 18, "CITY", 0.9, 1),
        Entity(19, 23, "CITY", 0.9, 1),
    ]
    joined = join_fragments(text, entities)
    assert [text[e.start : e.end] for e in joined] == ["شارع الجامعة", "عمّان إربد"]
    assert joined[0].score == pytest.approx((1.0 + 0.5 * 3) / 4)
    assert joined[0].tokens == 4
    assert join_fragments("Amman\nIrbid", [Entity(0, 5, "CITY", 1.0), Entity(6, 11, "CITY", 1.0)]) == [
        Entity(0, 5, "CITY", 1.0),
        Entity(6, 11, "CITY", 1.0),
    ]


def test_join_fragments_keeps_different_labels_apart() -> None:
    entities = [Entity(0, 5, "CITY", 1.0), Entity(6, 12, "COUNTRY", 1.0)]
    assert join_fragments("Amman Jordan", entities) == entities


def test_merge_entities_drops_overlap_duplicates() -> None:
    entities = [
        Entity(10, 20, "EMAIL", 0.90),
        Entity(10, 20, "EMAIL", 0.95),
        Entity(12, 20, "EMAIL", 0.99),
        Entity(30, 35, "PHONE", 0.50),
        Entity(0, 4, "CITY", 0.70),
    ]
    assert merge_entities(entities) == [
        Entity(0, 4, "CITY", 0.70),
        Entity(12, 20, "EMAIL", 0.99),
        Entity(30, 35, "PHONE", 0.50),
    ]


def test_merge_entities_prefers_the_longer_span_on_equal_score() -> None:
    assert merge_entities([Entity(5, 8, "CITY", 0.9), Entity(0, 8, "CITY", 0.9)]) == [Entity(0, 8, "CITY", 0.9)]


def test_chunked_detection_reports_each_entity_once_with_absolute_offsets() -> None:
    text = "\n".join(ARABIC_LINES * 6)
    phone = re.compile(r"07\d{8}")
    found: list[Entity] = []
    for chunk in chunk_text(text, 100, overlap=60):
        found += [Entity(*m.span(), "PHONE", 0.9).shifted(chunk.start) for m in phone.finditer(chunk.text)]
    merged = merge_entities(found)
    assert len(found) > len(merged)
    assert [(e.start, e.end) for e in merged] == [m.span() for m in phone.finditer(text)]


def test_to_detection_maps_counts_and_trims() -> None:
    text = "Call  Omar  on 2024-01-01"
    entities = [
        Entity(4, 11, "GIVEN_NAME", 0.9),
        Entity(15, 25, "DATE", 0.8),
        Entity(10, 12, "CITY", 0.7),
        Entity(99, 99, "NOT_A_LABEL", 0.5),
    ]
    detection = to_detection(text, entities, NYM_LABELS)
    assert [(s.label, s.text, s.score) for s in detection.spans] == [("PERSON", "Omar", 0.9)]
    assert detection.raw_labels == {"GIVEN_NAME": 1, "DATE": 1, "CITY": 1, "NOT_A_LABEL": 1}
    assert detection.unmapped == {"DATE": 1, "NOT_A_LABEL": 1}


def test_trimmed_span_handles_arabic_and_blank_ranges() -> None:
    text = " محمد "
    span = trimmed_span(text, 0, len(text), "PERSON", 0.5)
    assert span is not None
    assert (span.start, span.end, span.text) == (1, 5, "محمد")
    assert trimmed_span("   ", 0, 3, "PERSON", 0.5) is None
