"""Suite index: one table per split and mode over every run's summary, plus links to the comparisons."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..config import EXTRACT, MODES, PIPELINE, TEXT, RunSpec
from ..dataset import SPLITS
from ..matching import VALUE
from ..metrics import F1
from .markdown import MISSING, captioned, cell, document, interval, pct, section, table
from .run import mean_seconds, peak_rss, quality_rates, to_interval

RUNTIME_HEADERS = ["s / item", "peak RSS MB"]
HEADERS = {
    TEXT: ["run", "P %", "R %", "F1 %", "F1 95% CI", "F2 %", "documents", *RUNTIME_HEADERS],
    PIPELINE: [
        "run",
        "P %",
        "R %",
        "F1 %",
        "F1 95% CI",
        "F2 %",
        "exact/near F1 %",
        "PII token recall %",
        "renderings",
        *RUNTIME_HEADERS,
    ],
    EXTRACT: ["run", "CER %", "WER %", "token recall %", "PII token recall %", "renderings", *RUNTIME_HEADERS],
}
CAPTIONS = {
    TEXT: "Detection on the canonical text at lenient span level, in percent; the interval resamples documents.",
    PIPELINE: (
        "Files through extraction and detection at lenient value level, in percent; exact/near F1 is the "
        "exact or near value level, PII token recall the share of the words of gold values the extractor kept."
    ),
    EXTRACT: (
        "Extraction only, in percent, against the text each file shows; token recall counts every word, "
        "PII token recall the words of gold values."
    ),
}
SPLIT_TITLES = {"dev": "Dev split", "test": "Test split"}


def _link(path: Path, base: Path, text: str) -> str:
    return f"[{text}]({Path(os.path.relpath(path, base)).as_posix()})"


def _detection_cells(mode: str, summary: Mapping[str, Any]) -> list[object]:
    levels = summary["levels"]
    overall = levels[summary["main_level"]]["overall"]
    cells: list[object] = [
        pct(overall["precision"]),
        pct(overall["recall"]),
        pct(overall["f1"]),
        interval(to_interval((overall.get("ci95") or {}).get(F1))),
        pct(overall["f2"]),
    ]
    if mode == PIPELINE:
        quality = (summary.get("extraction") or {}).get("overall") or {}
        cells += [pct(levels[VALUE]["overall"]["f1"]), pct(quality.get("pii_token_recall"))]
    return cells


def _extract_cells(summary: Mapping[str, Any]) -> list[str]:
    return quality_rates((summary.get("extraction") or {}).get("overall") or {})


def run_row(spec: RunSpec, summary: Mapping[str, Any] | None, report: Path, base: Path) -> list[object]:
    """One index row; a run without summary shows as not run."""
    width = len(HEADERS[spec.mode])
    if summary is None:
        return [cell(spec.name), "not run", *[MISSING] * (width - 2)]
    scores = _extract_cells(summary) if spec.mode == EXTRACT else _detection_cells(spec.mode, summary)
    size = summary.get("documents") if spec.mode == TEXT else summary["items"]
    runtime = [mean_seconds(summary.get("timing")), peak_rss(summary.get("peak_rss_mb"))]
    return [_link(report, base, cell(spec.name)), *scores, size, *runtime]


def load_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "summary.json"
    if not path.is_file():
        return None
    summary: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return summary


def _run_sections(runs: Sequence[tuple[RunSpec, Path]], base: Path) -> list[str]:
    parts = []
    for split in SPLITS:
        for mode in MODES:
            selected = [(spec, run_dir) for spec, run_dir in runs if spec.split == split and spec.mode == mode]
            if not selected:
                continue
            rows = [run_row(spec, load_summary(d), d / "report.md", base) for spec, d in selected]
            title = f"{SPLIT_TITLES[split]}, {mode} runs"
            parts.append(section(title, captioned(CAPTIONS[mode], table(HEADERS[mode], rows))))
    return parts


def render_index(
    runs: Sequence[tuple[RunSpec, Path]], comparisons: Sequence[tuple[str, Path | None]], base: Path
) -> str:
    """``runs`` pairs each run with its directory; a comparison path of None means it was not rendered."""
    links = [
        [_link(path, base, cell(name)) if path is not None else cell(name), "written" if path else "skipped"]
        for name, path in comparisons
    ]
    caption = "Comparison reports; skipped means a run is missing or could not be scored."
    parts = [
        "# Evaluation index\n",
        "Dev runs choose thresholds, routing and the MVP configuration; test runs report the chosen ones.\n",
        *_run_sections(runs, base),
        section("Comparisons", captioned(caption, table(["comparison", "status"], links))),
    ]
    return document(parts)
