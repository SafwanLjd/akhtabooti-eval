from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from akheval.config import EvalConfig, load_config
from akheval.dataset import Dataset
from akheval.detectors.base import Detection
from akheval.matching import LENIENT, STRICT, VALUE, VALUE_LENIENT
from akheval.metrics import Counts
from akheval.registry import ConfigError
from akheval.tuning import THRESHOLDS, SweepPoint, choose, render, sweep, tune
from harness_fixtures import documents, make_dataset, register_fakes, span, write_config


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EvalConfig:
    register_fakes(monkeypatch)
    return load_config(write_config(tmp_path, make_dataset(tmp_path / "dataset").root))


def test_thresholds_cover_the_open_unit_interval() -> None:
    assert THRESHOLDS[0] == 0.05
    assert THRESHOLDS[-1] == 0.95
    assert len(THRESHOLDS) == 19


def test_sweep_filters_stored_scores() -> None:
    doc = next(d for d in documents() if d.id == "doc-en-1")
    scored = [
        replace(span(doc.text, "john.smith@example.org", "EMAIL"), score=0.9),
        replace(span(doc.text, "0791234567", "PHONE"), score=0.3),
        replace(span(doc.text, "INV-2024-001", "PHONE"), score=0.2),
    ]
    points = sweep([doc], [Detection(spans=scored)], thresholds=(0.1, 0.25, 0.5, 0.95))
    assert [p.lenient for p in points] == [Counts(2, 1, 1), Counts(2, 0, 1), Counts(1, 0, 2), Counts(0, 0, 3)]
    assert choose(points, 1.0).threshold == 0.25
    assert set(points[0].counts) == {LENIENT, STRICT, VALUE_LENIENT, VALUE}
    assert points[0].to_dict()["lenient"]["tp"] == 2
    assert points[0].to_dict()["value_lenient"]["tp"] == 2


def _point(threshold: float, counts: Counts) -> SweepPoint:
    return SweepPoint(threshold, dict.fromkeys((LENIENT, STRICT, VALUE_LENIENT, VALUE), counts))


def test_choose_breaks_ties_towards_the_higher_threshold() -> None:
    points = [_point(0.3, Counts(1)), _point(0.6, Counts(1))]
    assert choose(points, 1.0).threshold == 0.6
    assert "0.60 (chosen)" in render("d", "f1", points, points[1])


def test_f2_prefers_recall() -> None:
    points = [_point(0.3, Counts(tp=8, fp=8, fn=2)), _point(0.7, Counts(tp=6, fp=1, fn=4))]
    assert choose(points, 1.0).threshold == 0.7
    assert choose(points, 2.0).threshold == 0.3
    assert "maximises lenient span F2" in render("d", "f2", points, points[0])


def test_tune_writes_the_curve_and_returns_the_choice(config: EvalConfig) -> None:
    path, threshold = tune(config, "fake")
    assert path == config.project.tuning / "fake.json"
    assert threshold == 0.9
    record = json.loads(path.read_text(encoding="utf-8"))
    assert (record["criterion"], record["criterion_level"]) == ("f2", "lenient")
    assert record["configured_threshold"] == 0.5
    assert record["components"]["fake"]["threshold"] == 0.0
    curve = {p["threshold"]: p for p in record["curve"]}
    assert curve[0.4]["lenient"]["fp"] == 1
    assert curve[0.45]["lenient"]["fp"] == 0
    assert (config.project.tuning / "fake.md").read_text(encoding="utf-8").startswith("# Threshold sweep for fake")
    assert config.detectors["fake"].params["threshold"] == 0.5


def test_tune_sees_dev_documents_only(config: EvalConfig) -> None:
    record = json.loads(tune(config, "fake", "f1")[0].read_text(encoding="utf-8"))
    dataset = Dataset.load(config.project.dataset)
    assert record["dataset"]["split"] == "dev"
    assert record["documents"] == len(dataset.split("dev")) == 1
    assert record["curve"][0]["lenient"]["support"] == sum(len(d.spans) for d in dataset.split("dev"))


def test_tune_rejects_detectors_without_threshold(config: EvalConfig) -> None:
    with pytest.raises(ConfigError, match="no 'threshold' parameter"):
        tune(config, "regex")
    with pytest.raises(ConfigError, match="unknown detector"):
        tune(config, "ghost")
    with pytest.raises(ConfigError, match="criterion"):
        tune(config, "fake", "f3")
