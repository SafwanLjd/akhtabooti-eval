from __future__ import annotations

import json
from pathlib import Path

import pytest

from akheval.app_report import RenderingIndex, entry_detection, merge_entries, path_parts, score_app_report
from akheval.config import EvalConfig, load_config
from akheval.registry import ConfigError
from akheval.runner import ERRORS, PREDICTIONS, RUN_JSON, SUMMARY
from harness_fixtures import make_dataset, register_fakes, write_card_dataset, write_config


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EvalConfig:
    register_fakes(monkeypatch)
    return load_config(write_config(tmp_path, make_dataset(tmp_path / "dataset").root))


def test_path_parts_handles_windows_and_posix_paths() -> None:
    assert path_parts("C:\\Users\\amal\\files\\d1\\scan.pdf")[-3:] == ("files", "d1", "scan.pdf")
    assert path_parts("/home/amal/files/d1/scan.pdf")[-3:] == ("files", "d1", "scan.pdf")
    assert path_parts("scan.pdf") == ("scan.pdf",)


def test_index_matches_the_longest_unambiguous_trailing_path() -> None:
    paths = ["files/d1/digital.pdf", "files/d2/digital.pdf", "files/d2/photo-front.jpg", "files/d2/photo-back.jpg"]
    index = RenderingIndex(paths)
    assert index.match("C:\\Users\\amal\\akh\\files\\d1\\digital.pdf") == "files/d1/digital.pdf"
    assert index.match("/media/usb/d2/digital.pdf") == "files/d2/digital.pdf"
    assert index.match("/home/amal/photo-back.jpg") == "files/d2/photo-back.jpg"
    assert index.match("/home/amal/digital.pdf") is None
    assert index.match("/home/amal/other.pdf") is None


def test_entries_merge_per_rendering_and_list_unmatched_files() -> None:
    index = RenderingIndex(["files/d1/x.pdf", "files/d2/x.pdf"])
    merged, unmatched = merge_entries(
        [
            {"filename": "/a/files/d1/x.pdf", "email_accounts": ["a@b.co"], "phone_numbers": [], "other_piis": []},
            {"filename": "/a/files/d1/x.pdf", "phone_numbers": ["0791234567"], "other_piis": ["IBAN"]},
            {"filename": "/a/x.pdf", "email_accounts": ["c@d.co"]},
        ],
        index.match,
    )
    assert merged == {
        "files/d1/x.pdf": {"email_accounts": ["a@b.co"], "phone_numbers": ["0791234567"], "other_piis": ["IBAN"]}
    }
    assert unmatched == ["x.pdf"]


def test_entry_becomes_values_and_document_labels() -> None:
    detection = entry_detection(
        {"email_accounts": ["a@b.co", "a@b.co"], "phone_numbers": [], "other_piis": ["Jordanian Passport", "Banking"]}
    )
    assert detection.spans == []
    assert detection.values == [("EMAIL", "a@b.co")]
    assert detection.labels_present == {"EMAIL", "PASSPORT"}
    assert detection.unmapped == {"Banking": 1}
    assert entry_detection({}).predicted_labels() == set()


def test_score_app_report_writes_a_pipeline_run(config: EvalConfig, tmp_path: Path) -> None:
    report = tmp_path / "export.json"
    entries = [
        {
            "filename": "C:\\scan\\doc-en-1-txt.txt",
            "email_accounts": ["john.smith@example.org"],
            "phone_numbers": ["0791234567"],
            "other_piis": ["Jordanian Passport"],
        },
        {"filename": "/tmp/unknown.pdf", "email_accounts": [], "phone_numbers": [], "other_piis": []},
    ]
    report.write_text(json.dumps(entries), encoding="utf-8")
    run_dir = score_app_report(config, report, split="test", name="app-test", app_version="0.1.0")
    meta = json.loads((run_dir / RUN_JSON).read_text(encoding="utf-8"))
    assert meta["run"]["mode"] == "pipeline"
    assert meta["run"]["source"] == "desktop-app-report"
    assert meta["unmatched_entries"] == ["unknown.pdf"]
    assert meta["components"]["desktop-app"]["app_version"] == "0.1.0"
    assert (meta["akheval"], meta["config"]) == ("1.0.0", str(config.path))
    predictions = [json.loads(line) for line in (run_dir / PREDICTIONS).read_text(encoding="utf-8").splitlines()]
    assert len(predictions) == 5
    summary = json.loads((run_dir / SUMMARY).read_text(encoding="utf-8"))
    value = summary["levels"]["value"]["by_label"]
    assert value["EMAIL"]["tp"] == 1
    assert value["PHONE"]["tp"] == 1
    assert summary["levels"]["document"]["by_label"]["PASSPORT"]["fp"] == 1
    assert summary["extraction"]["overall"] is None


def test_score_app_report_checks_the_run_name(config: EvalConfig, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="run name"):
        score_app_report(config, tmp_path / "x.json", split="test", name="../escape", app_version="1")


def test_each_card_photo_is_scored_against_the_values_it_shows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    register_fakes(monkeypatch)
    dataset = write_card_dataset(tmp_path / "card")
    config = load_config(write_config(tmp_path, dataset.root))
    report = tmp_path / "export.json"
    entries = [
        {"filename": "D:\\scan\\card-1\\front.jpg", "other_piis": ["Jordanian National ID"]},
        {"filename": "D:\\scan\\card-1\\back.jpg", "phone_numbers": ["0791234567"]},
    ]
    report.write_text(json.dumps(entries), encoding="utf-8")
    run_dir = score_app_report(config, report, split="test", name="app-card", app_version="0.1.0")
    errors = [json.loads(line) for line in (run_dir / ERRORS).read_text(encoding="utf-8").splitlines()]
    missed = sorted((e["rendering_id"], e["label"]) for e in errors if e["kind"] == "value_missed")
    assert missed == [
        ("card-1.back.jpg", "ADDRESS"),
        ("card-1.digital.pdf", "ADDRESS"),
        ("card-1.digital.pdf", "NATIONAL_ID"),
        ("card-1.digital.pdf", "PERSON"),
        ("card-1.digital.pdf", "PHONE"),
        ("card-1.front.jpg", "NATIONAL_ID"),
        ("card-1.front.jpg", "PERSON"),
    ]
    summary = json.loads((run_dir / SUMMARY).read_text(encoding="utf-8"))
    document = summary["levels"]["document"]["by_label"]
    assert (document["NATIONAL_ID"]["tp"], document["NATIONAL_ID"]["fn"]) == (1, 1)
    # Each label sits on the native file and on one of the two photos.
    assert {label: c["support"] for label, c in document.items()} == dict.fromkeys(document, 2)
