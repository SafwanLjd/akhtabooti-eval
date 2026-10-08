from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from akheval.config import EXTRACT, PIPELINE, TEXT, extractor_for, load_config, parse_config
from akheval.registry import ConfigError

REPO = Path(__file__).resolve().parents[1]
EVAL_V1 = REPO / "configs" / "eval-v1.toml"

BASE: dict[str, Any] = {
    "project": {"dataset": "d", "results": "r/runs", "tuning": "r/tuning"},
    "detectors": {"rules": {"kind": "core-rules", "core_sha": "937e6ca"}, "base": {"kind": "regex-baseline"}},
    "extractors": {"text": {"kind": "extractous"}, "ocr": {"kind": "tesseract", "lang": "eng"}},
    "pipelines": {"p": {"detector": "rules", "extractors": {"default": "text", "png": "ocr"}}},
    "runs": [
        {"name": "t", "mode": "text", "split": "test", "detector": "rules"},
        {"name": "t2", "mode": "text", "split": "test", "detector": "base"},
        {"name": "p", "mode": "pipeline", "split": "test", "pipeline": "p"},
        {"name": "x", "mode": "extract", "split": "dev", "channel": "ocr", "extractors": {"default": "ocr"}},
    ],
    "comparisons": [{"name": "c", "baseline": "t", "candidate": "t2"}],
}


def _with(**changes: Any) -> dict[str, Any]:
    data = {k: (list(v) if isinstance(v, list) else dict(v)) for k, v in BASE.items()}
    data.update(changes)
    return data


def _problems(data: dict[str, Any]) -> str:
    with pytest.raises(ConfigError) as caught:
        parse_config(data, Path("test.toml"))
    return str(caught.value)


def test_valid_config_parses() -> None:
    config = parse_config(BASE, Path("test.toml"))
    assert config.project.results == Path("r/runs")
    assert config.run_dir("t") == Path("r/runs/t")
    assert config.detectors["rules"].params == {"core_sha": "937e6ca"}
    assert [r.mode for r in config.runs.values()] == [TEXT, TEXT, PIPELINE, EXTRACT]
    pipeline_run, extract_run = config.run("p"), config.run("x")
    assert config.detector_of(pipeline_run) == "rules"
    assert config.detector_of(extract_run) is None
    assert extractor_for(config.extractors_of(pipeline_run), "png") == "ocr"
    assert extractor_for(config.extractors_of(pipeline_run), "pdf") == "text"
    assert config.comparison("c").candidate == "t2"


def test_repository_config_loads() -> None:
    config = load_config(EVAL_V1)
    assert {"core-main", "core-release", "core-proposed", "hybrid-small", "regex-baseline"} <= set(config.detectors)
    proposed = config.detectors["core-proposed"].params
    assert proposed["definitions"] == "rules/definitions-proposed.json"
    assert proposed["located_categories"] == ["Email", "Phone Number", "IBAN", "IP Address", "UNHCR Number"]
    assert "located_categories" not in config.detectors["core-main"].params
    assert "text-regex-baseline" not in config.runs
    assert config.pipelines["app-v0.1.0-no-tesseract"].extractors["default"] == "extractous-no-tesseract"
    raw = tomllib.loads(EVAL_V1.read_text(encoding="utf-8"))
    assert len(raw["runs"]) == len(config.runs)
    assert len(raw["comparisons"]) == len(config.comparisons)


def test_mvp_candidate_is_one_detector_and_one_ocr_name() -> None:
    config = load_config(EVAL_V1)
    pipeline = config.pipelines["mvp-candidate"]
    assert pipeline.extractors == {"default": "mvp-extract"}
    assert config.extractors["mvp-extract"].kind == "pdf-ocr"
    assert config.extractors["mvp-extract"].params["ocr"] in config.extractors
    text = config.extractors[config.extractors["mvp-extract"].params["text"]]
    assert (text.kind, text.params["tesseract"]) == ("extractous", False)


def test_routed_hybrids_split_the_labels_between_rules_and_model() -> None:
    config = load_config(EVAL_V1)
    for name, model in (("hybrid-small-routed", "nym-small"), ("hybrid-edge-routed", "nym-edge")):
        params = config.detectors[name].params
        assert params["parts"] == ["core-proposed", model]
        rules, learned = set(params["only"]["core-proposed"]), set(params["only"][model])
        assert rules == {"EMAIL", "PHONE", "IBAN", "IP_ADDRESS", "REFUGEE_ID"}
        assert learned == {"PERSON", "ORG", "LOCATION", "ADDRESS", "NATIONAL_ID", "PASSPORT"}


def test_selection_runs_use_dev_and_reporting_runs_use_test() -> None:
    config = load_config(EVAL_V1)
    for run in config.runs.values():
        assert (run.split == "dev") == run.name.startswith("dev-"), run.name
    for detector in (
        "core-main",
        "core-release",
        "hybrid-small",
        "hybrid-edge",
        "hybrid-small-routed",
        "hybrid-edge-routed",
        "nym-small",
        "nym-edge",
        "core-proposed",
    ):
        assert config.run(f"dev-text-{detector}").detector == detector
        assert f"text-{detector}" in config.runs
    for ocr in ("ppocr-v5-arabic", "ppocr-v5-arabic-en", "tesseract-ara-eng", "ppocr-v6-small"):
        run = config.run(f"dev-extract-{ocr}")
        assert (run.extractors, run.channel) == ({"default": ocr}, "ocr")
    routed = {(c.baseline, c.candidate) for c in config.comparisons.values()}
    for size in ("small", "edge"):
        assert (f"text-hybrid-{size}", f"text-hybrid-{size}-routed") in routed
        assert (f"dev-text-hybrid-{size}", f"dev-text-hybrid-{size}-routed") in routed


def test_unknown_names_raise() -> None:
    config = parse_config(BASE, Path("test.toml"))
    with pytest.raises(ConfigError, match="unknown run 'nope'"):
        config.run("nope")
    with pytest.raises(ConfigError, match="unknown comparison"):
        config.comparison("nope")


def test_missing_file_and_bad_toml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[project\n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(bad)


def test_every_problem_is_listed_at_once() -> None:
    data = _with(
        detectors={"rules": {"kind": "telepathy"}},
        runs=[
            {"name": "t", "mode": "text", "split": "train", "detector": "rules"},
            {"name": "u", "mode": "guess", "split": "test"},
        ],
        comparisons=[],
    )
    message = _problems(data)
    assert "kind 'telepathy'" in message
    assert "split 'train'" in message
    assert "mode 'guess'" in message
    assert "detector 'rules' is not defined" in message


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"project": {"dataset": "d", "results": "r"}}, "'tuning'"),
        ({"surprise": {}}, "unknown top-level table [surprise]"),
        ({"detectors": {"Bad Name": {"kind": "regex-baseline"}}}, "must match"),
        ({"pipelines": {"p": {"detector": "rules", "extractors": {"png": "ocr"}}}}, "'default' key"),
        ({"pipelines": {"p": {"detector": "rules", "extractors": {"default": "text", "tiff": "ocr"}}}}, "'tiff'"),
        ({"pipelines": {"p": {"detector": "rules", "extractors": {"default": "missing"}}}}, "'missing' is not defined"),
        ({"pipelines": {"p": {"detector": "ghost", "extractors": {"default": "text"}}}}, "detector 'ghost'"),
    ],
)
def test_component_and_pipeline_problems(change: dict[str, Any], expected: str) -> None:
    assert expected in _problems(_with(**change))


@pytest.mark.parametrize(
    ("run", "expected"),
    [
        ({"name": "t3", "mode": "text", "split": "test"}, "detector None is not defined"),
        (
            {"name": "t3", "mode": "text", "split": "test", "detector": "rules", "pipeline": "p"},
            "'pipeline' does not apply",
        ),
        ({"name": "t3", "mode": "pipeline", "split": "test", "pipeline": "nope"}, "pipeline 'nope'"),
        ({"name": "t3", "mode": "extract", "split": "test"}, "extractors must be a table"),
        (
            {"name": "t3", "mode": "extract", "split": "test", "channel": "fax", "extractors": {"default": "ocr"}},
            "channel 'fax'",
        ),
        ({"name": "t", "mode": "text", "split": "test", "detector": "rules"}, "defined twice"),
    ],
)
def test_run_problems(run: dict[str, Any], expected: str) -> None:
    assert expected in _problems(_with(runs=[*BASE["runs"], run]))


@pytest.mark.parametrize(
    ("comparison", "expected"),
    [
        ({"name": "c2", "baseline": "t", "candidate": "ghost"}, "run 'ghost' is not defined"),
        ({"name": "c2", "baseline": "t", "candidate": "p"}, "differ in mode"),
        ({"name": "c2", "baseline": "x", "candidate": "x"}, "extract runs have no detection scores"),
    ],
)
def test_comparison_problems(comparison: dict[str, Any], expected: str) -> None:
    assert expected in _problems(_with(comparisons=[comparison]))


def test_run_spec_to_dict_is_sorted() -> None:
    config = parse_config(BASE, Path("test.toml"))
    assert config.run("x").to_dict() == {
        "name": "x",
        "mode": "extract",
        "split": "dev",
        "detector": None,
        "pipeline": None,
        "extractors": {"default": "ocr"},
        "channel": "ocr",
    }
