from __future__ import annotations

from pathlib import Path

from akheval.suite import INDEX, suite
from harness_fixtures import make_dataset

CONFIG = """
[project]
dataset = "{dataset}"
results = "{out}/runs"
tuning = "{out}/tuning"

[detectors.regex]
kind = "regex-baseline"

[detectors.no-shim]
kind = "core-rules"
core_sha = "0000000"
binary = "{out}/missing-binary"

[[runs]]
name = "text-a"
mode = "text"
split = "test"
detector = "regex"

[[runs]]
name = "text-b"
mode = "text"
split = "dev"
detector = "regex"

[[runs]]
name = "text-c"
mode = "text"
split = "test"
detector = "regex"

[[runs]]
name = "text-broken"
mode = "text"
split = "test"
detector = "no-shim"

[[comparisons]]
name = "a-vs-c"
baseline = "text-a"
candidate = "text-c"

[[comparisons]]
name = "a-vs-broken"
baseline = "text-a"
candidate = "text-broken"
"""


def test_suite_runs_each_entry_in_a_process_and_writes_the_index(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "dataset")
    out = tmp_path / "out"
    config = tmp_path / "eval.toml"
    config.write_text(CONFIG.format(dataset=dataset.root.as_posix(), out=out.as_posix()), encoding="utf-8")
    assert suite(config, ["text-a", "text-c", "text-broken"]) == 1
    assert (out / "runs" / "text-a" / "report.md").is_file()
    assert not (out / "runs" / "text-b").exists()
    assert not (out / "runs" / "text-broken").exists()
    assert (out / "comparisons" / "a-vs-c.md").is_file()
    index = (out / INDEX).read_text(encoding="utf-8")
    assert "[text-a](runs/text-a/report.md)" in index
    assert "| text-b | not run |" in index
    assert "[a-vs-c](comparisons/a-vs-c.md)" in index
    assert "| a-vs-broken | skipped |" in index
