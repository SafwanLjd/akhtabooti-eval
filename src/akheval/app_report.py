"""Score a JSON report exported by the Akhtabooti desktop app.

The export is a list of ``{"filename", "email_accounts", "phone_numbers",
"other_piis"}``. ``other_piis`` names categories such as "Jordanian Passport"
without the identifiers, so the run scores emails and phones at value level
and every category at document level. The app writes absolute paths. An entry
matches the rendering whose path inside the dataset, such as
``files/<document>/digital.pdf``, ends the exported path. A rendering without
an entry counts as a file where the app found nothing. Scoring treats the
export like any pipeline run, so a file that shows part of a document, such as
one face of an ID card, is scored against the gold values that file shows.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from . import __version__
from .config import NAME, PIPELINE, EvalConfig, RunSpec
from .dataset import Dataset, sha256_file
from .detectors.base import Detection
from .detectors.core_rules import to_detection
from .registry import ConfigError
from .runner import dataset_identity, environment, score_run, selected_renderings, write_run
from .scoring import Prediction

SOURCE = "desktop-app-report"
APP = "desktop-app"
FIELDS = {"email_accounts": "Email", "phone_numbers": "Phone Number"}


Findings = dict[str, list[str]]


def path_parts(filename: str) -> tuple[str, ...]:
    """Components of a POSIX or Windows path."""
    return (PureWindowsPath if "\\" in filename else PurePosixPath)(filename).parts


class RenderingIndex:
    """Finds the rendering an exported path names by the longest trailing run of path components."""

    def __init__(self, paths: Iterable[str]) -> None:
        self._by_suffix: dict[tuple[str, ...], set[str]] = {}
        for path in paths:
            parts = PurePosixPath(path).parts
            for start in range(len(parts)):
                self._by_suffix.setdefault(parts[start:], set()).add(path)

    def match(self, filename: str) -> str | None:
        """The rendering path, or None when no rendering or more than one fits."""
        parts = path_parts(filename)
        for start in range(len(parts)):
            found = self._by_suffix.get(parts[start:])
            if found is not None:
                return next(iter(found)) if len(found) == 1 else None
        return None


def merge_entries(
    entries: Iterable[Mapping[str, Any]], match: Callable[[str], str | None]
) -> tuple[dict[str, Findings], list[str]]:
    """Join the findings of entries per matched rendering path; also return the file names that matched none."""
    merged: dict[str, Findings] = {}
    unmatched: set[str] = set()
    for entry in entries:
        filename = str(entry.get("filename", ""))
        path = match(filename)
        if path is None:
            unmatched.add(path_parts(filename)[-1] if filename else "")
            continue
        found = merged.setdefault(path, {})
        for key in (*FIELDS, "other_piis"):
            found.setdefault(key, []).extend(str(v) for v in entry.get(key, []))
    return merged, sorted(unmatched)


def entry_detection(found: Mapping[str, Sequence[str]]) -> Detection:
    """Read an entry the way the shim reports Core output, with no text to locate spans in."""
    record = {
        "detections": {category: sorted(set(found.get(key, []))) for key, category in FIELDS.items() if found.get(key)},
        "keyword_categories": sorted(set(found.get("other_piis", []))),
    }
    return to_detection("", record)


def score_app_report(config: EvalConfig, report: Path, *, split: str, name: str, app_version: str) -> Path:
    """Write a pipeline-mode run directory for the app's report and score it; returns the run directory."""
    if not NAME.match(name):
        raise ConfigError(f"run name {name!r} must match {NAME.pattern}")
    dataset = Dataset.load(config.project.dataset)
    exported = json.loads(report.read_text(encoding="utf-8"))
    index = RenderingIndex(rendering.path for _, rendering in dataset.renderings())
    entries, unmatched = merge_entries(exported, index.match)
    renderings = selected_renderings(dataset.split(split), None)
    predictions = [
        Prediction(
            item_id=rendering.id,
            doc_id=doc.id,
            rendering_id=rendering.id,
            extractor=APP,
            detection=entry_detection(entries.get(rendering.path, {})),
        )
        for doc, rendering in renderings
    ]
    spec = RunSpec(name=name, mode=PIPELINE, split=split)
    meta = {
        "run": spec.to_dict() | {"source": SOURCE},
        "akheval": __version__,
        "config": str(config.path),
        "components": {
            APP: {
                "kind": SOURCE,
                "app_version": app_version,
                "report": str(report),
                "report_sha256": sha256_file(report),
            }
        },
        "dataset": dataset_identity(dataset, split),
        "environment": environment(),
        "report_entries": len(exported),
        "unmatched_entries": unmatched,
        "timing": None,
        "peak_rss_mb": None,
    }
    run_dir = config.run_dir(name)
    write_run(run_dir, meta, predictions)
    score_run(run_dir, dataset)
    return run_dir
