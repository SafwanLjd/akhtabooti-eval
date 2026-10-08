"""Threshold sweep for a detector with scored spans, on the dev split only.

The detector runs once in text mode with its threshold forced to 0.0 over the
dev documents. The sweep then filters the stored spans at each threshold and
scores them, so every point on the curve comes from the same model outputs.
Lenient span F2 picks the threshold by default, F1 on request; strict span and
value scores are reported alongside. The config is left unchanged, and the
test split is never loaded into the sweep.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import TEXT, EvalConfig, RunSpec
from .dataset import Dataset, Document
from .detectors.base import Detection
from .matching import LENIENT, STRICT, VALUE, VALUE_LENIENT
from .metrics import Counts, total
from .registry import ConfigError
from .report.markdown import captioned, pct, table
from .runner import dataset_identity, execute
from .scoring import OVERALL, Prediction, score_text, series

TUNING_SPLIT = "dev"
THRESHOLD_STEP = 0.05
THRESHOLDS = tuple(round(THRESHOLD_STEP * i, 2) for i in range(1, 20))
CRITERIA = {"f1": 1.0, "f2": 2.0}
DEFAULT_CRITERION = "f2"
THRESHOLD_PARAM = "threshold"
SWEPT_LEVELS = (LENIENT, STRICT, VALUE_LENIENT, VALUE)


@dataclass(frozen=True)
class SweepPoint:
    """Overall counts at one threshold, per level in ``SWEPT_LEVELS``."""

    threshold: float
    counts: dict[str, Counts]

    @property
    def lenient(self) -> Counts:
        return self.counts[LENIENT]

    def to_dict(self) -> dict[str, Any]:
        return {"threshold": self.threshold} | {level: c.to_dict() for level, c in self.counts.items()}


def _overall(items: Sequence[Any], level: str) -> Counts:
    return total(series(items, level).get(OVERALL, []))


def sweep(
    documents: Sequence[Document], detections: Sequence[Detection], thresholds: Sequence[float] = THRESHOLDS
) -> list[SweepPoint]:
    points = []
    for threshold in thresholds:
        items = [
            score_text(doc, Prediction(item_id=doc.id, doc_id=doc.id, detection=det.with_threshold(threshold)))
            for doc, det in zip(documents, detections, strict=True)
        ]
        points.append(SweepPoint(threshold, {level: _overall(items, level) for level in SWEPT_LEVELS}))
    return points


def choose(points: Sequence[SweepPoint], beta: float) -> SweepPoint:
    """Best lenient F-beta; on a tie, the higher threshold."""
    return max(points, key=lambda p: (p.lenient.fbeta(beta) or 0.0, p.threshold))


def render(detector: str, criterion: str, points: Sequence[SweepPoint], best: SweepPoint) -> str:
    beta = CRITERIA[criterion]
    name = criterion.upper()
    rows = [
        [
            f"{p.threshold:.2f}" + (" (chosen)" if p is best else ""),
            pct(p.lenient.precision),
            pct(p.lenient.recall),
            *(pct(p.counts[level].fbeta(beta)) for level in SWEPT_LEVELS),
        ]
        for p in points
    ]
    headers = [
        "threshold",
        "lenient P %",
        "lenient R %",
        f"lenient {name} %",
        f"strict {name} %",
        f"lenient value {name} %",
        f"exact/near value {name} %",
    ]
    caption = (
        f"Dev split, in percent: the chosen threshold maximises lenient span {name}; "
        "the value columns score the same outputs at the levels pipeline runs report."
    )
    return f"# Threshold sweep for {detector}\n\n" + captioned(caption, table(headers, rows))


def tune(config: EvalConfig, detector: str, criterion: str = DEFAULT_CRITERION) -> tuple[Path, float]:
    """Run the sweep and write ``<tuning>/<detector>.json`` and ``.md``; returns the JSON path and the threshold."""
    if criterion not in CRITERIA:
        raise ConfigError(f"criterion {criterion!r} is not one of {', '.join(CRITERIA)}")
    component = config.detectors.get(detector)
    if component is None:
        raise ConfigError(f"unknown detector {detector!r}")
    if THRESHOLD_PARAM not in component.params:
        raise ConfigError(f"detector {detector!r} has no {THRESHOLD_PARAM!r} parameter to tune")
    forced = replace(component, params={**component.params, THRESHOLD_PARAM: 0.0})
    tuned = replace(config, detectors={**config.detectors, detector: forced})
    spec = RunSpec(name=f"tune-{detector}", mode=TEXT, split=TUNING_SPLIT, detector=detector)
    dataset = Dataset.load(config.project.dataset)
    documents = dataset.split(TUNING_SPLIT)
    execution = execute(tuned, spec, documents, dataset.root)
    detections = [p.detection or Detection() for p in execution.predictions]
    points = sweep(documents, detections)
    best = choose(points, CRITERIA[criterion])
    record = {
        "detector": detector,
        "configured_threshold": component.params[THRESHOLD_PARAM],
        "chosen_threshold": best.threshold,
        "criterion": criterion,
        "criterion_level": LENIENT,
        "dataset": dataset_identity(dataset, TUNING_SPLIT),
        "documents": len(documents),
        "components": execution.components,
        "failed_items": sum(1 for p in execution.predictions if p.detection_error),
        "curve": [p.to_dict() for p in points],
    }
    out = config.project.tuning
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{detector}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / f"{detector}.md").write_text(render(detector, criterion, points, best), encoding="utf-8")
    return path, best.threshold
