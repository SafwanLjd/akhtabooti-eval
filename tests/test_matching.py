from __future__ import annotations

import pytest

from akheval.dataset import Span
from akheval.matching import (
    ANY,
    ANY_TYPE,
    LENIENT,
    STRICT,
    VALUE,
    VALUE_LENIENT,
    VALUE_STRICT,
    align_spans,
    document_counts,
    is_partial,
    match_values,
    normalize_value,
    span_counts,
    value_counts,
    value_tokens,
)
from akheval.metrics import FN, FP, TP, Counts

RLM = chr(0x200F)
LRM = chr(0x200E)
TATWEEL = chr(0x0640)
FULLWIDTH_OFFSET = 0xFEE0
FULLWIDTH_JOHN = "".join(chr(ord(c) + FULLWIDTH_OFFSET) for c in "John")

GOLD = [
    Span(0, 10, "PERSON", "a"),
    Span(20, 30, "EMAIL", "b"),
    Span(40, 50, "PHONE", "c"),
    Span(60, 70, "PERSON", "d"),
]
PRED = [
    Span(0, 10, "PERSON", "a"),
    Span(22, 30, "EMAIL", "b'"),
    Span(40, 50, "NATIONAL_ID", "c"),
    Span(80, 90, "EMAIL", "e"),
]


def test_alignment_kinds() -> None:
    alignment = align_spans(GOLD, PRED)
    assert [(p.gold.start, p.pred.start) for p in alignment.exact] == [(0, 0)]
    assert [(p.gold.start, p.pred.start) for p in alignment.boundary] == [(20, 22)]
    assert [(p.gold.label, p.pred.label) for p in alignment.wrong_type] == [("PHONE", "NATIONAL_ID")]
    assert [s.start for s in alignment.missed] == [60]
    assert [s.start for s in alignment.spurious] == [80]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            STRICT,
            {
                "PERSON": Counts(1, 0, 1),
                "EMAIL": Counts(0, 2, 1),
                "PHONE": Counts(0, 0, 1),
                "NATIONAL_ID": Counts(0, 1, 0),
            },
        ),
        (
            LENIENT,
            {
                "PERSON": Counts(1, 0, 1),
                "EMAIL": Counts(1, 1, 0),
                "PHONE": Counts(0, 0, 1),
                "NATIONAL_ID": Counts(0, 1, 0),
            },
        ),
        (ANY_TYPE, {"PERSON": Counts(1, 0, 1), "EMAIL": Counts(1, 1, 0), "PHONE": Counts(1, 0, 0)}),
    ],
)
def test_span_counts_per_mode(mode: str, expected: dict[str, Counts]) -> None:
    assert span_counts(align_spans(GOLD, PRED), mode) == expected


def test_unknown_span_mode() -> None:
    with pytest.raises(ValueError, match="unknown span mode"):
        span_counts(align_spans([], []), "fuzzy")


def test_pairing_prefers_the_largest_overlap() -> None:
    gold = [Span(0, 20, "PERSON")]
    pred = [Span(0, 5, "PERSON"), Span(5, 20, "PERSON")]
    alignment = align_spans(gold, pred)
    assert [p.pred.start for p in alignment.boundary] == [5]
    assert [s.start for s in alignment.spurious] == [0]


def test_pairing_is_one_to_one() -> None:
    gold = [Span(0, 10, "PERSON"), Span(12, 20, "PERSON")]
    pred = [Span(5, 18, "PERSON")]
    alignment = align_spans(gold, pred)
    assert [p.gold.start for p in alignment.boundary] == [12]
    assert [s.start for s in alignment.missed] == [0]
    assert span_counts(alignment, LENIENT) == {"PERSON": Counts(1, 0, 1)}


def test_exact_match_wins_over_overlap() -> None:
    gold = [Span(0, 10, "EMAIL")]
    pred = [Span(0, 12, "EMAIL"), Span(0, 10, "EMAIL")]
    alignment = align_spans(gold, pred)
    assert len(alignment.exact) == 1
    assert [s.end for s in alignment.spurious] == [12]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("٠٧٩-١٢٣ ٤٥٦٧", "0791234567"),
        ("۰۷۹۱۲۳۴۵۶۷", "0791234567"),
        ("P<JOR<<SMITH", "pjorsmith"),
        ("أحمد", "احمد"),
        ("إيمان", "ايمان"),
        ("مُحَمَّد", "محمد"),
        ("فاطمة", "فاطمه"),
        ("مصطفى", "مصطفي"),
        (f"{TATWEEL}{TATWEEL}علي", "علي"),
        (f"{RLM}+962 (79) 123.4567{LRM}", "+962791234567"),
        ("JO94 CBJO 0010", "jo94cbjo0010"),
        (FULLWIDTH_JOHN, "john"),
    ],
)
def test_normalize_value(raw: str, expected: str) -> None:
    assert normalize_value(raw) == expected


def test_value_matching_exact_near_and_duplicates() -> None:
    gold = [
        ("EMAIL", "a@b.com"),
        ("PHONE", "0791234567"),
        ("PERSON", "أحمد علي"),
        ("PHONE", "079 123 4567"),
    ]
    pred = [("EMAIL", "A@B.com"), ("PHONE", "0791234561"), ("PERSON", "احمد علي"), ("EMAIL", "x@y.org")]
    alignment = match_values(gold, pred)
    assert sorted(p.label for p in alignment.exact) == ["EMAIL", "PERSON"]
    assert [(p.gold, p.pred) for p in alignment.near] == [("0791234567", "0791234561")]
    assert alignment.near[0].ratio == pytest.approx(0.9)
    assert alignment.missed == []
    assert alignment.spurious == [("EMAIL", "x@y.org")]
    assert value_counts(alignment, VALUE) == {
        "EMAIL": Counts(1, 1, 0),
        "PERSON": Counts(1, 0, 0),
        "PHONE": Counts(1, 0, 0),
    }
    assert value_counts(alignment, VALUE_STRICT)["PHONE"] == Counts(0, 1, 1)
    assert value_counts(alignment, VALUE_LENIENT) == value_counts(alignment, VALUE)


def test_value_matching_respects_labels() -> None:
    alignment = match_values([("PHONE", "0791234567")], [("NATIONAL_ID", "0791234567")])
    assert alignment.exact == []
    assert alignment.near == []
    assert alignment.missed == [("PHONE", "0791234567")]
    assert alignment.spurious == [("NATIONAL_ID", "0791234567")]


def test_value_below_near_ratio_is_a_miss() -> None:
    alignment = match_values([("PHONE", "0791234567")], [("PHONE", "0799999999")])
    assert alignment.near == []
    assert alignment.partial == []
    assert value_counts(alignment, VALUE_LENIENT) == {"PHONE": Counts(0, 1, 1)}


def test_unknown_value_mode() -> None:
    with pytest.raises(ValueError, match="unknown value mode"):
        value_counts(match_values([], []), "fuzzy")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12 Rainbow St., Amman", {"12", "rainbow", "st", "amman"}),
        ("شارع المدينة، عمّان", {"شارع", "المدينه", "عمان"}),
        ("079-123-4567", {"0791234567"}),
        (" , ", set()),
    ],
)
def test_value_tokens(raw: str, expected: set[str]) -> None:
    assert value_tokens(raw) == expected


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        ("Building 46, Petra Street, Karak", "Karak", True),
        ("0791234567", "Tel0791234567", True),
        ("Al-Amal Charity Society", "t", False),
        ("Sahab", "Sa", False),
        ("علي حازم", "علي", True),
        ("Omar Thamer Al-Nsour", "Omar Thamer Al-NsourField", True),
        ("Al-Sharq College of Applied Sciences", "College Sciences Applied Al-Sharq", True),
        ("Amal Haddad Khoury", "Rami Haddad", True),
        ("Amal Haddad Khoury", "Rami Saleh Odeh", False),
        ("john@example.com", "jane@example.com", False),
        ("-", "Karak", False),
    ],
)
def test_partial_match(gold: str, pred: str, expected: bool) -> None:
    assert is_partial(gold, pred, by_tokens=True) is expected


@pytest.mark.parametrize(
    ("label", "gold", "pred"),
    [
        ("PHONE", "00962 79 211 6889", "00962 79 621 5233"),
        ("IP_ADDRESS", "fd12:3456:c28c::ceda", "fd12:3456:6dd4::a4f6"),
        ("PHONE", "06 3164748", "2026-07-27 04:06:47"),
    ],
)
def test_structured_values_sharing_words_are_no_match(label: str, gold: str, pred: str) -> None:
    assert value_counts(match_values([(label, gold)], [(label, pred)]), VALUE_LENIENT) == {label: Counts(0, 1, 1)}


def test_split_address_earns_lenient_credit_once() -> None:
    gold = [("ADDRESS", "Building 46, Petra Street, Karak"), ("LOCATION", "Karak")]
    pred = [("ADDRESS", "Building 46"), ("ADDRESS", "Karak"), ("LOCATION", "Karak")]
    alignment = match_values(gold, pred)
    assert [p.pred for p in alignment.partial] == ["Building 46"]
    assert alignment.spurious == [("ADDRESS", "Karak")]
    assert value_counts(alignment, VALUE_LENIENT) == {"ADDRESS": Counts(1, 1, 0), "LOCATION": Counts(1, 0, 0)}
    assert value_counts(alignment, VALUE) == {"ADDRESS": Counts(0, 2, 1), "LOCATION": Counts(1, 0, 0)}


def test_exact_and_near_pairs_come_before_partial_ones() -> None:
    gold = [("PERSON", "Amal Haddad"), ("PERSON", "Amal")]
    alignment = match_values(gold, [("PERSON", "Amal"), ("PERSON", "Amal Hadad")])
    assert [(p.gold, p.pred) for p in alignment.exact] == [("Amal", "Amal")]
    assert [(p.gold, p.pred) for p in alignment.near] == [("Amal Haddad", "Amal Hadad")]
    assert alignment.partial == []


def test_document_counts() -> None:
    assert document_counts({"EMAIL", "PHONE"}, {"EMAIL", "PERSON"}) == {
        "EMAIL": TP,
        "PHONE": FN,
        "PERSON": FP,
        ANY: TP,
    }
    assert document_counts(set(), set()) == {}
    assert document_counts(set(), {"EMAIL"}) == {"EMAIL": FP, ANY: FP}
    assert document_counts({"IBAN"}, set()) == {"IBAN": FN, ANY: FN}
