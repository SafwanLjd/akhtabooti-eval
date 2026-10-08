from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from akheval.config import EvalConfig, load_config
from akheval.dataset import Dataset
from akheval.registry import ConfigError
from akheval.runner import (
    ERRORS,
    PREDICTIONS,
    REPORT,
    RUN_JSON,
    SUMMARY,
    environment,
    extract_one,
    helper_peaks,
    load_run,
    percentile,
    run,
    score_run,
    vm_hwm_mb,
)
from harness_fixtures import FakeExtractor, make_dataset, register_fakes, write_config

BAD_DETECTOR = """
[detectors.bad]
kind = "core-rules"
core_sha = "68d1749"
located_categories = ["Email", "Phones"]

[pipelines.bad]
detector = "bad"
extractors = { default = "text" }

[[runs]]
name = "pipeline-bad"
mode = "pipeline"
split = "test"
pipeline = "bad"
"""


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EvalConfig:
    register_fakes(monkeypatch)
    dataset = make_dataset(tmp_path / "dataset")
    return load_config(write_config(tmp_path, dataset.root))


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_text_run_writes_every_file(config: EvalConfig) -> None:
    run_dir = run(config, "text-fake")
    assert run_dir == config.run_dir("text-fake")
    for name in (RUN_JSON, PREDICTIONS, SUMMARY, ERRORS, REPORT):
        assert (run_dir / name).is_file()
    predictions = _lines(run_dir / PREDICTIONS)
    assert [p["item_id"] for p in predictions] == ["doc-ar-1", "doc-en-1", "doc-en-2"]
    failed = predictions[2]
    assert failed["detection"] is None
    assert failed["detection_error"] == "RuntimeError: fake detector failure"
    assert predictions[1]["detection"]["spans"][0]["label"] == "EMAIL"


def test_run_json_records_identity_environment_and_timing(config: EvalConfig) -> None:
    meta = json.loads((run(config, "text-fake") / RUN_JSON).read_text(encoding="utf-8"))
    assert meta["run"]["detector"] == "fake"
    assert meta["components"]["fake"] == {"kind": "fake", "threshold": 0.5}
    assert meta["dataset"]["name"] == "tiny"
    assert meta["dataset"]["documents_sha256"] == Dataset.load(config.project.dataset).documents_sha256()
    timing = meta["timing"]
    assert set(timing["init_seconds"]) == {"fake"}
    assert timing["items"] == 3
    assert set(timing["per_item_seconds"]) == {"mean", "p50", "p95"}
    assert timing["chars_per_second"] > 0
    assert meta["peak_rss_mb"]["self"] > 0
    assert meta["peak_rss_mb"]["helpers"] == {}
    assert {"python", "cpu", "cpu_count", "ram_gb", "packages", "tesseract", "git"} <= set(meta["environment"])


def test_text_run_scores(config: EvalConfig) -> None:
    summary = json.loads((run(config, "text-fake") / SUMMARY).read_text(encoding="utf-8"))
    lenient = summary["levels"]["lenient"]
    assert lenient["overall"]["tp"] == 2
    assert lenient["by_label"]["EMAIL"]["tp"] == 2
    assert lenient["by_label"]["PHONE"]["fn"] == 2
    assert summary["errors"]["detection_failed"] == 1
    assert summary["raw_labels"] == {"EMAIL": 2, "PHONE": 2}
    assert summary["documents"] == 3
    report = (config.run_dir("text-fake") / REPORT).read_text(encoding="utf-8")
    assert report.startswith("# Run text-fake")
    assert "### detection_failed: 1 of 1" in report


def test_pipeline_run_extracts_then_detects(config: EvalConfig) -> None:
    run_dir = run(config, "pipeline-plain")
    predictions = {p["item_id"]: p for p in _lines(run_dir / PREDICTIONS)}
    assert sorted(predictions) == ["doc-ar-1-broken", "doc-ar-1-txt", "doc-en-1-scan", "doc-en-1-txt", "doc-en-2-txt"]
    broken = predictions["doc-ar-1-broken"]
    assert broken["extraction_error"] == "cannot read doc-ar-1-broken.png"
    assert broken["extracted"] == ""
    assert predictions["doc-en-1-txt"]["extractor"] == "text"
    assert predictions["doc-en-1-txt"]["detection"]["spans"]
    assert predictions["doc-en-2-txt"]["detection_error"].startswith("RuntimeError")
    summary = json.loads((run_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["main_level"] == "value_lenient"
    assert summary["extraction"]["by_channel"]["ocr"]["items"] == 2
    kinds = {e["kind"] for e in _lines(run_dir / ERRORS)}
    assert {"extraction_failed", "detection_failed", "value_missed"} <= kinds


def test_extract_run_filters_by_channel(config: EvalConfig) -> None:
    run_dir = run(config, "extract-loud")
    predictions = _lines(run_dir / PREDICTIONS)
    assert [p["item_id"] for p in predictions] == ["doc-ar-1-broken", "doc-en-1-scan"]
    assert all(p["detection"] is None and p["detect_seconds"] is None for p in predictions)
    assert predictions[1]["extracted"].isupper()
    summary = json.loads((run_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["levels"] == {}
    assert 0 < summary["extraction"]["overall"]["cer"] < 1


def test_rerun_overwrites_the_same_directory(config: EvalConfig) -> None:
    first = run(config, "text-regex")
    stale = first / PREDICTIONS
    stale.write_text("stale\n", encoding="utf-8")
    second = run(config, "text-regex")
    assert first == second
    assert sorted(p.name for p in second.iterdir()) == sorted([RUN_JSON, PREDICTIONS, SUMMARY, ERRORS, REPORT])
    assert len(load_run(second)[1]) == 3


def test_rescoring_refuses_a_changed_dataset(config: EvalConfig) -> None:
    run_dir = run(config, "text-regex")
    dataset = Dataset.load(config.project.dataset)
    assert score_run(run_dir, dataset) == run_dir / REPORT
    documents = dataset.root / "documents.jsonl"
    documents.write_text(documents.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"documents\.jsonl"):
        score_run(run_dir, Dataset.load(config.project.dataset))


class _Crash(FakeExtractor):
    """Raises MemoryError on files named crash.txt and records every path it reads."""

    def __init__(self) -> None:
        super().__init__()
        self.read: list[str] = []

    def extract(self, path: Path) -> str:
        self.read.append(path.name)
        if path.name == "crash.txt":
            raise MemoryError("out of memory")
        return super().extract(path)


def test_extract_one_records_each_failure_and_reads_each_file_once(tmp_path: Path) -> None:
    for name in ("a.txt", "crash.txt", "b.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    extractor = _Crash()
    paths = [tmp_path / n for n in ("a.txt", "crash.txt", "broken.txt", "b.txt")]
    results = [extract_one(extractor, path) for path in paths]
    assert [r.text for r in results] == ["a.txt", "", "", "b.txt"]
    assert results[1].error == "MemoryError: out of memory"
    assert results[2].error == "cannot read broken.txt"
    assert extractor.read == ["a.txt", "crash.txt", "broken.txt", "b.txt"]


def test_bad_detector_parameters_fail_before_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: EvalConfig
) -> None:
    config.path.write_text(config.path.read_text(encoding="utf-8") + BAD_DETECTOR, encoding="utf-8")
    read: list[Path] = []
    monkeypatch.setattr(FakeExtractor, "extract", lambda _self, path: read.append(path) or "")
    with pytest.raises(ConfigError, match="bad parameters"):
        run(load_config(config.path), "pipeline-bad")
    assert read == []
    assert not (tmp_path / "out" / "runs" / "pipeline-bad").exists()


def test_vm_hwm_is_read_from_proc_status_text() -> None:
    assert vm_hwm_mb("Name:\takh-ocr\nVmPeak:\t  900000 kB\nVmHWM:\t  501760 kB\nVmRSS:\t 1024 kB\n") == 490.0
    assert vm_hwm_mb("Name:\takh-ocr\nState:\tZ (zombie)\n") is None


def test_helper_peaks_measure_a_live_child() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        peaks = helper_peaks()
    finally:
        child.kill()
        child.wait()
    assert peaks[Path(sys.executable).name] > 0


def test_percentile_nearest_rank() -> None:
    values = [0.4, 0.1, 0.3, 0.2]
    assert percentile(values, 0.5) == 0.2
    assert percentile(values, 0.95) == 0.4
    assert percentile([], 0.5) is None


def test_environment_package_versions_use_metadata() -> None:
    env = environment()
    assert env["packages"]["akheval"] == "1.0.0"
    assert "python-bidi" in env["packages"]
    assert env["cpu_count"] >= 1
