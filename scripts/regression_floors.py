"""Print proposed regression floors for the MVP runs, per family, language and channel.

Each floor is the lower bound of the 95% bootstrap interval of recall over test documents, rounded down to a
whole point. Run from the repository root:

    .venv/bin/python scripts/regression_floors.py
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

from akheval.dataset import Dataset
from akheval.matching import LENIENT, VALUE_LENIENT
from akheval.metrics import Counts, bootstrap, f1_of, recall_of, total
from akheval.runner import load_run
from akheval.scoring import BY_FAMILY, OVERALL, PIPELINE, TEXT, score_predictions, series

RUNS = (
    ("text-hybrid-small", TEXT, LENIENT, (OVERALL, BY_FAMILY, "language")),
    ("pipeline-mvp-candidate", PIPELINE, VALUE_LENIENT, (OVERALL, BY_FAMILY, "language", "channel")),
)


def rows(documents: Sequence[Counts]) -> tuple[float, float, float, float]:
    """Recall, its 95% interval and F1 of one group's per-document counts."""
    counts = total(documents)
    interval = bootstrap(documents, {"recall": recall_of})["recall"]
    recall, f1 = recall_of(counts), f1_of(counts)
    if interval is None or recall is None or f1 is None:
        raise ValueError("a group needs gold values in at least two documents")
    return recall, interval.low, interval.high, f1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=Path("datasets/akhtabooti-v1"))
    parser.add_argument("--runs", type=Path, default=Path("reports/v1/runs"))
    args = parser.parse_args()
    test = Dataset.load(args.dataset).split("test")
    print("| run | level | group | gold | R % | 95% CI | F1 % | floor % |")
    print("|---|---|---|---|---|---|---|---|")
    for run, mode, level, facets in RUNS:
        _, predictions = load_run(args.runs / run)
        items = score_predictions(test, mode, predictions)
        for facet in facets:
            for group, documents in series(items, level, facet).items():
                recall, low, high, f1 = rows(documents)
                gold = total(documents).tp + total(documents).fn
                floor = math.floor(100 * low)
                print(
                    f"| {run} | {level} | {group} | {gold} | {100 * recall:.1f} | "
                    f"[{100 * low:.1f}, {100 * high:.1f}] | {100 * f1:.1f} | {floor} |"
                )


if __name__ == "__main__":
    main()
