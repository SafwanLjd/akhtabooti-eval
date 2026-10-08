from __future__ import annotations

from pathlib import Path
from typing import Any

from akheval.config import Comparison, RunSpec
from akheval.detectors.base import Detection
from akheval.metrics import F1, RECALL, Counts, Delta, Interval
from akheval.report.compare import DeltaRow, Drop, _drop_table, align, drops, render_comparison
from akheval.report.index import render_index
from akheval.report.markdown import cell, document, interval, pct, points, table
from akheval.report.run import (
    drop_empty_columns,
    helpers_cell,
    peak_rss,
    rate_with_interval,
    render_run,
    sample_errors,
)
from akheval.scoring import PIPELINE, TEXT, ErrorRecord, ItemScore, Prediction, score_predictions, score_text, summarize
from harness_fixtures import documents, span


def _error(label: str, language: str, doc: str) -> ErrorRecord:
    return ErrorRecord("false_negative", label, "x", doc, None, language)


def test_markdown_helpers() -> None:
    assert cell("a|b\nc") == "a\\|b c"
    assert cell(None) == "-"
    assert cell("  ") == "-"
    assert pct(0.1234) == "12.3"
    assert points(-0.021) == "-2.1"
    assert interval(Interval(-0.03, 0.01), signed=True) == "[-3.0, +1.0]"
    assert table(["a"], []) == "No rows.\n"
    assert table(["a", "b"], [[1, None]]) == "| a | b |\n|---|---|\n| 1 | - |\n"


def test_error_sample_spreads_over_labels_and_languages() -> None:
    errors = [_error("EMAIL", "en", f"d{i:02}") for i in range(20)]
    errors += [_error("PHONE", "ar", "d99"), _error("EMAIL", "ar", "d98")]
    picked = sample_errors(errors, limit=4)
    assert len(picked) == 4
    assert {(e.label, e.language) for e in picked} == {("EMAIL", "en"), ("PHONE", "ar"), ("EMAIL", "ar")}
    assert picked == sample_errors(list(reversed(errors)), limit=4)


def test_error_sample_of_few_errors_keeps_all() -> None:
    errors = [_error("EMAIL", "en", "d1"), _error("EMAIL", "en", "d2")]
    assert len(sample_errors(errors)) == 2


def _row(f1: float, interval_: Interval | None) -> DeltaRow:
    delta = {F1: Delta(f1, interval_), RECALL: Delta(None, None)}
    return DeltaRow("scope", "label", "EMAIL", Counts(tp=5), Counts(tp=4, fn=1), delta)


def test_drop_classification() -> None:
    clear = drops([_row(-0.05, Interval(-0.08, -0.01))])
    noisy = drops([_row(-0.05, Interval(-0.08, 0.01))])
    small = drops([_row(-0.01, Interval(-0.015, -0.005))])
    two_points = drops([_row(-0.01999999999999999, Interval(-0.04, -0.01))])
    assert [d.is_regression for d in clear] == [True]
    assert [d.is_regression for d in noisy] == [False]
    assert [d.is_regression for d in small] == [False]
    assert [d.is_regression for d in two_points] == [True]
    assert drops([_row(0.05, Interval(0.01, 0.08))]) == []
    assert isinstance(clear[0], Drop)


def test_drop_table_shows_the_rate_of_each_metric() -> None:
    base, candidate = Counts(tp=4, fn=1), Counts(tp=2, fn=3)
    delta = {F1: Delta(-0.317, Interval(-0.5, -0.1)), RECALL: Delta(-0.4, Interval(-0.6, -0.2))}
    text = _drop_table(drops([DeltaRow("scope", "label", "EMAIL", base, candidate, delta)]))
    assert "| scope | EMAIL | recall | 80.0 | 40.0 | -40.0 |" in text
    assert "| scope | EMAIL | F1 | 88.9 | 57.1 | -31.7 |" in text


def _items(found: bool) -> list[ItemScore]:
    items = []
    for doc in documents():
        spans = [s for s in doc.spans if s.label == "EMAIL"] if found else []
        items.append(score_text(doc, Prediction(item_id=doc.id, doc_id=doc.id, detection=Detection(spans=spans))))
    return items


def _meta(name: str, mode: str = "text") -> dict[str, Any]:
    return {
        "run": {"name": name, "mode": mode, "split": "test"},
        "components": {name: {"kind": "fake"}},
        "dataset": {"name": "tiny", "version": "0.0.1", "frozen": True, "documents_sha256": "ab" * 32},
        "timing": {"init_seconds": {name: 0.1}, "total_seconds": 1.0, "per_item_seconds": {"mean": 0.25}},
        "peak_rss_mb": {"self": 100.0, "helpers": {"akh-ocr": 5.0}},
    }


def test_align_keeps_common_items_in_order() -> None:
    base = _items(True)
    base_aligned, cand_aligned = align(base, list(reversed(_items(False)))[1:])
    assert [i.item_id for i in base_aligned] == [i.item_id for i in cand_aligned]
    assert len(base_aligned) == len(base) - 1


def test_comparison_opens_with_the_headline_and_lists_a_regression() -> None:
    comparison = Comparison("emails-lost", "with-email", "without-email")
    text = render_comparison(comparison, _meta("with-email"), _items(True), _meta("without-email"), _items(False))
    assert text.startswith("# Comparison emails-lost")
    headline = text.split("## Scores by level")[0]
    assert "lenient span on the 4 test documents both runs scored" in headline
    assert "| with-email | baseline | 100.0 | 42.9 | 60.0 | 48.4 | 0.2500 | 100.0 |" in headline
    assert "| Δ | candidate - baseline | - | -42.9 [-100.0, -33.3] | -60.0 [" in headline
    regressions = text.split("## Regressions")[1].split("## Other drops")[0]
    assert "| lenient span, by label | EMAIL | F1 | 100.0 | 0.0 | -100.0 |" in regressions
    by_group = text.split("## By group")[1]
    assert "| family | structured |" in by_group
    assert "| document type | letter |" in by_group
    record = text.split("## Run record")[1]
    assert "| peak RSS MB, run process | 100.0 | 100.0 |" in record
    assert "| peak RSS MB, helper processes | akh-ocr 5.0 | akh-ocr 5.0 |" in record


def test_every_section_of_a_comparison_has_a_caption() -> None:
    comparison = Comparison("c", "a", "b")
    text = render_comparison(comparison, _meta("a"), _items(True), _meta("b"), _items(False))
    for block in text.split("\n## ")[1:]:
        caption = block.split("\n\n")[1]
        assert not caption.startswith("|"), block.splitlines()[0]


def _pipeline_run() -> tuple[dict[str, Any], dict[str, Any], list[ItemScore]]:
    docs = documents()
    predictions = [
        Prediction(
            item_id=r.id,
            doc_id=d.id,
            rendering_id=r.id,
            detection=Detection(values=[(s.label, s.text) for s in d.spans if s.label != "PERSON"]),
            extracted=d.text,
            detect_seconds=0.5,
        )
        for d in docs
        if d.split == "test"
        for r in d.renderings
    ]
    items = score_predictions(docs, PIPELINE, predictions)
    meta = _meta("pipe", "pipeline") | {"environment": {"cpu": "x", "cpu_count": 2, "ram_gb": 4.0}}
    return meta, summarize(PIPELINE, items), items


def test_run_report_opens_with_the_headline() -> None:
    meta, summary, items = _pipeline_run()
    text = render_run(meta, summary, items)
    lines = text.splitlines()
    assert lines[0] == "# Run pipe"
    assert lines[2].startswith("Headline on the test split")
    assert lines[4].startswith("| level | P % | R % | F1 % | F2 % | documents | renderings | s / item | peak RSS MB |")
    assert lines[6].startswith("| lenient value | 100.0 [100.0, 100.0] | 66.7 [")
    assert lines[6].endswith("| 3 | 5 | 0.2500 | 100.0 |")
    assert text.index("## Run record") > text.index("## Errors")
    assert "### doc_false_negative" not in text
    by_label = text.split("## By label\n")[1].split("##")[0]
    assert "| PERSON | contextual | 4 | - | 0.0 | 0.0 | 0.0 | - | 0.0 | 0.0 |" in by_label


def test_run_report_counts_false_positives_by_look_alike() -> None:
    doc = documents()[1]
    detection = Detection(spans=[span(doc.text, "INV-2024-001", "NATIONAL_ID")])
    items = [score_text(doc, Prediction(item_id=doc.id, doc_id=doc.id, detection=detection))]
    text = render_run(_meta("t"), summarize(TEXT, items), items)
    look_alikes = text.split("False positives (spans or values) by the look-alike they hit")[1]
    assert "| look-alike | count |\n|---|---|\n| invoice_number | 1 |" in look_alikes
    assert "| peak RSS MB, helper processes | akh-ocr 5.0 |" in text


def test_every_section_of_a_run_report_has_a_caption() -> None:
    meta, summary, items = _pipeline_run()
    for block in render_run(meta, summary, items).split("\n## ")[1:]:
        caption = block.split("\n\n")[1]
        assert not caption.startswith("|"), block.splitlines()[0]


def test_report_helpers() -> None:
    overall = {"precision": 0.5, "ci95": {"precision": {"low": 0.4, "high": 0.6}}}
    assert rate_with_interval(overall, "precision") == "50.0 [40.0, 60.0]"
    assert rate_with_interval({"recall": None, "ci95": {}}, "recall") == "-"
    assert peak_rss({"self": 10.0, "helpers": {"akh-ocr": 30.0, "python": 20.0}}) == 30.0
    assert peak_rss({"self": 10.0, "helpers": {}}) == 10.0
    assert peak_rss(None) is None
    assert helpers_cell({"self": 10.0, "helpers": {"akh-ocr": 30.0, "python": 20.0}}) == "akh-ocr 30.0, python 20.0"
    assert helpers_cell({"self": 10.0, "helpers": {}}) is None
    headers, rows = drop_empty_columns(["a", "b", "c"], [[1, None, ""], [2, "", "x"]])
    assert (headers, rows) == (["a", "c"], [[1, ""], [2, "x"]])


def test_index_groups_runs_by_split_and_mode(tmp_path: Path) -> None:
    runs = [
        (RunSpec("dev-text-x", "text", "dev", detector="x"), tmp_path / "runs" / "dev-text-x"),
        (RunSpec("extract-y", "extract", "test", extractors={"default": "y"}), tmp_path / "runs" / "extract-y"),
    ]
    text = render_index(runs, [("c1", tmp_path / "comparisons" / "c1.md"), ("c2", None)], tmp_path)
    assert text.index("## Dev split, text runs") < text.index("## Test split, extract runs")
    assert "| dev-text-x | not run | - | - | - | - | - | - | - |" in text
    assert "| extract-y | not run | - | - | - | - | - |" in text
    assert "[c1](comparisons/c1.md)" in text
    assert "| c2 | skipped |" in text


def test_document_ends_with_exactly_one_newline() -> None:
    assert document(["# Title\n", "Body.\n", ""]) == "# Title\n\nBody.\n"
    assert document(["# Title"]) == "# Title\n"
