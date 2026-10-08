"""Contract of partial renderings: ``Rendering.visible`` and the ``Document.visible_*`` methods."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from akheval.dataset import Rendering, validate
from harness_fixtures import (
    CARD_BACK,
    CARD_BACK_START,
    CARD_FRONT,
    CARD_TEXT,
    card_photo,
    two_face_card,
    write_card_dataset,
)


def test_visible_text_is_the_face_a_rendering_shows() -> None:
    native, front, back = two_face_card().renderings
    card = two_face_card()
    assert card.visible_text(native) == CARD_TEXT
    assert card.visible_text(front) == CARD_FRONT
    assert card.visible_text(back) == CARD_BACK


def test_visible_text_joins_ranges_with_a_line_break() -> None:
    card = two_face_card()
    both = card_photo("both", ((0, 4), (CARD_BACK_START, CARD_BACK_START + 7)))
    assert card.visible_text(both) == "Name\nAddress"


def test_visible_spans_and_values_hold_the_gold_of_one_face() -> None:
    card = two_face_card()
    native, front, back = card.renderings
    assert card.visible_spans(native) == card.spans
    assert [s.label for s in card.visible_spans(front)] == ["PERSON", "NATIONAL_ID"]
    assert [(v.label, v.value) for v in card.visible_values(back)] == [
        ("ADDRESS", "12 Rainbow Street, Amman"),
        ("PHONE", "0791234567"),
    ]
    assert card.visible_values(native) == card.expected_values()


def test_rendering_round_trip_keeps_visible_ranges() -> None:
    front = two_face_card().renderings[1]
    assert Rendering.from_dict(front.to_dict()) == front
    assert Rendering.from_dict(two_face_card().renderings[0].to_dict()).visible is None


def test_a_card_with_one_photo_per_face_is_valid(tmp_path: Path) -> None:
    assert validate(write_card_dataset(tmp_path)) == []


@pytest.mark.parametrize(
    ("visible", "problem"),
    [
        ((), "shows no text"),
        (((CARD_BACK_START, len(CARD_TEXT)), (0, len(CARD_FRONT))), "unsorted or overlapping"),
        (((0, 20), (10, 30)), "unsorted or overlapping"),
        (((0, len(CARD_TEXT) + 5),), "outside the text"),
        (((0, 10),), "cuts gold span"),
    ],
)
def test_validate_rejects_bad_visible_ranges(
    tmp_path: Path, visible: tuple[tuple[int, int], ...], problem: str
) -> None:
    card = two_face_card()
    bad = replace(card.renderings[1], visible=visible)
    dataset = write_card_dataset(tmp_path, replace(card, renderings=(card.renderings[0], bad)))
    assert any(problem in p for p in validate(dataset))
