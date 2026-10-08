from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from akheval.cli import DEFAULT_CONFIG, FAILURE, OK, USAGE, build_parser, main
from akheval.dataset import Rendering
from harness_fixtures import make_dataset, register_fakes, two_face_card, write_card_dataset, write_config


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    register_fakes(monkeypatch)
    return write_config(tmp_path, make_dataset(tmp_path / "dataset").root)


def test_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["tune", "--detector", "nym-small"])
    assert args.config == DEFAULT_CONFIG
    assert args.criterion == "f2"
    assert not hasattr(args, "split")
    args = parser.parse_args(["suite", "--only", "a", "--only", "b"])
    assert args.only == ["a", "b"]
    args = parser.parse_args(["score-app-report", "--report", "r.json"])
    assert (args.split, args.name, args.app_version) == ("test", "app-report", "unknown")
    args = parser.parse_args(["synth", "--config", "s.toml", "--out", "d", "--limit", "3"])
    assert (args.out, args.limit, args.overwrite) == (Path("d"), 3, False)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["dance"],
        ["run"],
        ["tune", "--detector", "x", "--criterion", "f3"],
        ["tune", "--detector", "x", "--split", "test"],
    ],
)
def test_usage_errors_exit_with_two(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == USAGE


def test_unknown_run_is_a_usage_error(config_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--config", str(config_path), "--run", "ghost"]) == USAGE
    assert "unknown run 'ghost'" in capsys.readouterr().err


def test_validate_and_freeze(config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = tmp_path / "dataset"
    assert main(["validate", "--dataset", str(dataset)]) == OK
    assert "valid" in capsys.readouterr().out
    assert main(["freeze", "--config", str(config_path)]) == OK
    assert json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))["frozen"] is True
    (dataset / "files" / "doc-en-2-txt.txt").unlink()
    assert main(["validate", "--dataset", str(dataset)]) == FAILURE
    assert "doc-en-2-txt.txt is missing" in capsys.readouterr().out
    assert main(["freeze", "--dataset", str(dataset)]) == FAILURE


def test_run_report_compare_and_tune(config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = str(config_path)
    assert main(["run", "--config", config, "--run", "text-regex"]) == OK
    assert main(["run", "--config", config, "--run", "text-fake"]) == OK
    assert main(["report", "--config", config, "--run", "text-fake"]) == OK
    assert main(["compare", "--config", config, "--comparison", "regex-vs-fake"]) == OK
    assert (tmp_path / "out" / "comparisons" / "regex-vs-fake.md").is_file()
    assert main(["compare", "--config", config, "--baseline", "text-fake", "--candidate", "text-regex"]) == OK
    assert (tmp_path / "out" / "comparisons" / "text-fake-vs-text-regex.md").is_file()
    assert main(["compare", "--config", config]) == USAGE
    capsys.readouterr()
    assert main(["tune", "--config", config, "--detector", "fake"]) == OK
    assert "set threshold = 0.9 in [detectors.fake]" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("baseline", "candidate", "problem"),
    [("text-fake", "pipeline-plain", "runs differ in mode"), ("extract-loud", "extract-loud", "no detection scores")],
)
def test_ad_hoc_compare_checks_the_runs_are_comparable(
    config_path: Path, capsys: pytest.CaptureFixture[str], baseline: str, candidate: str, problem: str
) -> None:
    argv = ["compare", "--config", str(config_path), "--baseline", baseline, "--candidate", candidate]
    assert main(argv) == USAGE
    assert problem in capsys.readouterr().err


def test_contact_sheet_takes_every_ocr_rendering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from akheval.synth.images import contact

    card = two_face_card()
    scan = Rendering("card-1.scan.pdf", "files/card-1/scan.pdf", "pdf", "ocr", "scan", "mild")
    dataset = write_card_dataset(tmp_path / "card", replace(card, renderings=(*card.renderings, scan)))
    tiled: list[Path] = []
    monkeypatch.setattr(contact, "make_contact_sheet", lambda images, _out: tiled.extend(images))
    assert main(["contact-sheet", "--dataset", str(dataset.root), "--out", str(tmp_path / "sheet.jpg")]) == OK
    assert sorted(p.name for p in tiled) == ["back.jpg", "front.jpg", "scan.pdf"]


def test_fetch_usage_errors_exit_with_two(tmp_path: Path) -> None:
    assert main(["fetch", "--models-dir", str(tmp_path / "nowhere")]) == USAGE
    assert main(["fetch", "--models-dir", str(tmp_path), "--only", "ghost"]) == USAGE


def test_report_of_a_missing_run_fails(config_path: Path) -> None:
    assert main(["report", "--config", str(config_path), "--run", "text-fake"]) == FAILURE


def test_fetch_with_no_entries(tmp_path: Path) -> None:
    assert main(["fetch", "--models-dir", str(tmp_path)]) == OK
