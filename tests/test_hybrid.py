from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from akheval.dataset import Span
from akheval.detectors.base import Detection, Detector
from akheval.detectors.hybrid import HybridDetector, merge, restrict

TEXT = "Ahmad Ali ahmad@x.com 0791234567"


class Canned(Detector):
    kind = "canned"

    def __init__(self, detection: Detection) -> None:
        self.detection = detection
        self.calls: list[str] = []

    def prepare(self) -> None:
        self.calls.append("prepare")

    def close(self) -> None:
        self.calls.append("close")

    def info(self) -> dict[str, Any]:
        return {"kind": self.kind}

    def detect(self, _text: str) -> Detection:
        self.calls.append("detect")
        return self.detection


def _rules() -> Detection:
    return Detection(
        spans=[Span(10, 21, "EMAIL", "ahmad@x.com")],
        values=[("EMAIL", "ahmad@x.com"), ("PASSPORT", "Passport")],
        labels_present={"EMAIL", "PASSPORT"},
        raw_labels=Counter({"Email": 1, "Jordanian Passport": 1}),
    )


def _model() -> Detection:
    return Detection(
        spans=[
            Span(0, 9, "PERSON", "Ahmad Ali", 0.9),
            Span(10, 15, "PERSON", "ahmad", 0.6),
            Span(22, 32, "PHONE", "0791234567", 0.8),
        ],
        raw_labels=Counter({"GIVEN_NAME": 2, "PHONE": 1}),
        unmapped=Counter({"DATE": 1}),
    )


def test_first_part_wins_on_overlap() -> None:
    merged = merge(TEXT, [_rules(), _model()])
    assert [(s.start, s.label) for s in merged.spans] == [(0, "PERSON"), (10, "EMAIL"), (22, "PHONE")]
    assert merged.values == [
        ("EMAIL", "ahmad@x.com"),
        ("PASSPORT", "Passport"),
        ("PERSON", "Ahmad Ali"),
        ("PHONE", "0791234567"),
    ]
    assert merged.labels_present == {"EMAIL", "PASSPORT"}
    assert merged.raw_labels == Counter({"Email": 1, "Jordanian Passport": 1, "GIVEN_NAME": 2, "PHONE": 1})
    assert merged.unmapped == Counter({"DATE": 1})
    assert merged.predicted_labels() == {"EMAIL", "PASSPORT", "PERSON", "PHONE"}


def test_order_of_parts_decides() -> None:
    merged = merge(TEXT, [_model(), _rules()])
    assert [(s.start, s.label) for s in merged.spans] == [(0, "PERSON"), (10, "PERSON"), (22, "PHONE")]


def test_values_stay_none_when_no_part_reports_values() -> None:
    assert merge(TEXT, [_model(), Detection()]).values is None


def test_detector_runs_every_part_and_merges() -> None:
    rules, model = Canned(_rules()), Canned(_model())
    hybrid = HybridDetector([rules, model])
    hybrid.prepare()
    merged = hybrid.detect(TEXT)
    hybrid.close()
    assert merged == merge(TEXT, [_rules(), _model()])
    assert rules.calls == model.calls == ["prepare", "detect", "close"]
    assert hybrid.info() == {"kind": "hybrid", "parts": [{"kind": "canned"}, {"kind": "canned"}], "only": [None, None]}


def test_from_config_resolves_parts_by_name() -> None:
    parts = {"rules": Canned(_rules()), "model": Canned(_model())}
    hybrid = HybridDetector.from_config({"parts": ["rules", "model"]}, parts.__getitem__)
    assert hybrid.parts == [parts["rules"], parts["model"]]
    assert hybrid.only == [None, None]


def test_restrict_keeps_only_the_listed_labels() -> None:
    kept = restrict(_rules(), {"PASSPORT"})
    assert kept.spans == []
    assert kept.values == [("PASSPORT", "Passport")]
    assert kept.labels_present == {"PASSPORT"}
    assert kept.raw_labels == _rules().raw_labels
    assert restrict(_model(), {"PHONE"}).values is None


def test_routing_filters_each_part_before_the_merge() -> None:
    parts = {"rules": Canned(_rules()), "model": Canned(_model())}
    params = {"parts": ["rules", "model"], "only": {"rules": ["EMAIL"], "model": ["PERSON"]}}
    hybrid = HybridDetector.from_config(params, parts.__getitem__)
    merged = hybrid.detect(TEXT)
    assert [(s.start, s.label) for s in merged.spans] == [(0, "PERSON"), (10, "EMAIL")]
    assert merged.values == [("EMAIL", "ahmad@x.com"), ("PERSON", "Ahmad Ali")]
    assert merged.labels_present == {"EMAIL"}
    assert hybrid.info()["only"] == [["EMAIL"], ["PERSON"]]


def test_a_routed_out_rule_span_leaves_room_for_the_model() -> None:
    rules = Detection(spans=[Span(22, 32, "NATIONAL_ID", "0791234567")], values=[("NATIONAL_ID", "0791234567")])
    hybrid = HybridDetector([Canned(rules), Canned(_model())], only=[["EMAIL"], None])
    assert [(s.start, s.label) for s in hybrid.detect(TEXT).spans] == [(0, "PERSON"), (10, "PERSON"), (22, "PHONE")]


def test_invalid_hybrids() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        HybridDetector([Canned(Detection())])
    with pytest.raises(ValueError, match="1 filters for 2 parts"):
        HybridDetector([Canned(Detection()), Canned(Detection())], only=[None])
    with pytest.raises(ValueError, match="outside the taxonomy"):
        HybridDetector([Canned(Detection()), Canned(Detection())], only=[["EMAIL"], ["NAME"]])
    parts = {"rules": Canned(_rules()), "model": Canned(_model())}
    with pytest.raises(ValueError, match="not parts of this hybrid"):
        HybridDetector.from_config({"parts": ["rules", "model"], "only": {"ghost": ["EMAIL"]}}, parts.__getitem__)
    with pytest.raises(TypeError, match="prefer"):
        HybridDetector.from_config({"parts": ["rules", "model"], "prefer": "first"}, parts.__getitem__)
