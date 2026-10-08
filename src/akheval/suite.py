"""Run every configured run in its own process, then render the comparisons and the index.

One process per run keeps each run's peak memory separate and stops a crash
in one model from taking down the others.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import Comparison, EvalConfig, load_config
from .dataset import Dataset
from .report.compare import render_comparison
from .report.index import render_index
from .runner import PREDICTIONS, RUN_JSON, scored_items

log = logging.getLogger(__name__)

INDEX = "index.md"


def run_in_subprocess(config_path: Path, run_name: str) -> int:
    command = [sys.executable, "-m", "akheval", "run", "--config", str(config_path), "--run", run_name]
    return subprocess.run(command, check=False).returncode


def comparison_path(config: EvalConfig, name: str) -> Path:
    return config.project.results.parent / "comparisons" / f"{name}.md"


def write_comparison(config: EvalConfig, comparison: Comparison, dataset: Dataset) -> Path:
    """Score both runs from their stored predictions and write the comparison report."""
    base_meta, base_items = scored_items(config.run_dir(comparison.baseline), dataset)
    cand_meta, cand_items = scored_items(config.run_dir(comparison.candidate), dataset)
    path = comparison_path(config, comparison.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_comparison(comparison, base_meta, base_items, cand_meta, cand_items), encoding="utf-8")
    return path


def _has_run(config: EvalConfig, name: str) -> bool:
    run_dir = config.run_dir(name)
    return (run_dir / RUN_JSON).is_file() and (run_dir / PREDICTIONS).is_file()


def render_comparisons(config: EvalConfig) -> list[tuple[str, Path | None]]:
    """Write every comparison whose two runs exist; None marks a skipped one."""
    dataset = Dataset.load(config.project.dataset)
    written: list[tuple[str, Path | None]] = []
    for comparison in config.comparisons.values():
        missing = [n for n in (comparison.baseline, comparison.candidate) if not _has_run(config, n)]
        if missing:
            log.warning("comparison %s skipped: no stored run for %s", comparison.name, ", ".join(missing))
            written.append((comparison.name, None))
            continue
        try:
            written.append((comparison.name, write_comparison(config, comparison, dataset)))
        except ValueError as exc:
            log.warning("comparison %s skipped: %s", comparison.name, exc)
            written.append((comparison.name, None))
    return written


def write_index(config: EvalConfig, comparisons: Sequence[tuple[str, Path | None]]) -> Path:
    base = config.project.results.parent
    runs = [(spec, config.run_dir(spec.name)) for spec in config.runs.values()]
    path = base / INDEX
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_index(runs, comparisons, base), encoding="utf-8")
    return path


def suite(config_path: Path, only: Sequence[str] = ()) -> int:
    """Returns 0 when every selected run succeeded, 1 otherwise."""
    config = load_config(config_path)
    names = [config.run(name).name for name in only] or list(config.runs)
    failed = []
    for name in names:
        log.info("run %s", name)
        if run_in_subprocess(config_path, name) != 0:
            failed.append(name)
            log.warning("run %s failed", name)
    index = write_index(config, render_comparisons(config))
    log.info("index written to %s", index)
    if failed:
        log.warning("failed runs: %s", ", ".join(failed))
    return 1 if failed else 0
