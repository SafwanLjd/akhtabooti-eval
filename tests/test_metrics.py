from __future__ import annotations

import pytest

from akheval.metrics import (
    F1,
    F2,
    FN,
    FP,
    METRICS,
    PRECISION,
    RECALL,
    TP,
    Counts,
    Interval,
    bootstrap,
    f1_of,
    f2_of,
    paired_bootstrap,
    precision_of,
    recall_of,
    tally,
    total,
)


def test_rates_from_counts() -> None:
    c = Counts(tp=3, fp=1, fn=2)
    assert c.support == 5
    assert c.predicted == 4
    assert c.precision == pytest.approx(0.75)
    assert c.recall == pytest.approx(0.6)
    assert c.f1 == pytest.approx(6 / 9)
    assert c.f2 == pytest.approx(15 / 24)


def test_undefined_and_zero_scores() -> None:
    assert Counts().precision is None
    assert Counts().recall is None
    assert Counts().f1 is None
    only_false_alarms = Counts(fp=2)
    assert only_false_alarms.precision == 0
    assert only_false_alarms.recall is None
    assert only_false_alarms.f1 == 0
    only_misses = Counts(fn=3)
    assert only_misses.precision is None
    assert only_misses.f1 == 0


def test_addition_total_and_tally() -> None:
    assert Counts(1, 1, 1) == TP + FP + FN
    assert total([TP, TP, FN]) == Counts(2, 0, 1)
    assert total([]) == Counts()
    table: dict[str, Counts] = {}
    tally(table, "EMAIL", TP)
    tally(table, "EMAIL", FP)
    tally(table, "PHONE", FN)
    assert table == {"EMAIL": Counts(1, 1, 0), "PHONE": Counts(0, 0, 1)}


def test_to_dict_rounds_and_keeps_none() -> None:
    d = Counts(tp=1, fp=2).to_dict()
    assert d["precision"] == 0.333333
    assert d["recall"] == 1.0
    assert Counts().to_dict()["f1"] is None


def test_bootstrap_is_deterministic_and_brackets_the_estimate() -> None:
    items = [Counts(tp=1)] * 6 + [Counts(fp=1, fn=1)] * 4
    first = bootstrap(items)
    assert set(first) == {PRECISION, RECALL, F1, F2}
    assert first == bootstrap(items)
    point = total(items)
    for name, metric in METRICS.items():
        interval, value = first[name], metric(point)
        assert interval is not None
        assert value is not None
        assert interval.low <= value <= interval.high
        assert interval.low < interval.high


def test_bootstrap_of_constant_items_is_a_point() -> None:
    assert bootstrap([Counts(tp=1)] * 5)[F1] == Interval(1.0, 1.0)


def test_bootstrap_needs_two_items() -> None:
    assert bootstrap([Counts(tp=1)]) == dict.fromkeys(METRICS)
    assert bootstrap([], {F1: f1_of}) == {F1: None}


def test_bootstrap_seed_changes_the_resamples() -> None:
    items = [Counts(tp=1)] * 6 + [Counts(fp=1, fn=1)] * 4 + [Counts(tp=2, fn=1)] * 3
    assert bootstrap(items, seed=1) != bootstrap(items, seed=2)


def test_bootstrap_skips_resamples_where_a_rate_is_undefined() -> None:
    interval = bootstrap([Counts(tp=1), Counts(fn=1)] * 5, {RECALL: recall_of})[RECALL]
    assert interval is not None
    assert 0 <= interval.low <= 0.5 <= interval.high <= 1
    assert bootstrap([Counts(fn=1)] * 3, {PRECISION: precision_of}) == {PRECISION: None}


def test_paired_bootstrap_of_identical_runs_is_zero() -> None:
    items = [Counts(tp=1), Counts(fp=1), Counts(fn=1), Counts(tp=2, fn=1)]
    deltas = paired_bootstrap(items, items)
    assert set(deltas) == {PRECISION, RECALL, F1, F2}
    for delta in deltas.values():
        assert delta.value == 0
        assert delta.interval == Interval(0.0, 0.0)
        assert not delta.interval.excludes_zero()


def test_paired_bootstrap_detects_a_clear_gain() -> None:
    base = [Counts(fn=1)] * 8
    candidate = [Counts(tp=1)] * 8
    deltas = paired_bootstrap(base, candidate)
    assert deltas[F1].value == pytest.approx(1.0)
    assert deltas[RECALL].value == pytest.approx(1.0)
    assert deltas[PRECISION].value is None
    interval = deltas[F2].interval
    assert interval is not None
    assert interval.excludes_zero()


def test_paired_bootstrap_of_one_item_has_no_interval() -> None:
    deltas = paired_bootstrap([Counts(fn=1)], [Counts(tp=1)], {F1: f1_of})
    assert deltas[F1].value == pytest.approx(1.0)
    assert deltas[F1].interval is None


def test_paired_bootstrap_rejects_misaligned_runs() -> None:
    with pytest.raises(ValueError, match="aligned"):
        paired_bootstrap([Counts()], [Counts(), Counts()])


def test_metric_helpers() -> None:
    assert f1_of(Counts(tp=1)) == 1.0
    assert recall_of(Counts(tp=1, fn=1)) == 0.5
    assert precision_of(Counts(tp=1, fp=3)) == 0.25
    assert f2_of(Counts(tp=3, fp=1, fn=2)) == pytest.approx(15 / 24)
