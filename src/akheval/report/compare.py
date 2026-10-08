"""Markdown comparison of a baseline and a candidate run scored on the same items.

Deltas are candidate minus baseline in percentage points, with paired-bootstrap
95% intervals over documents. A drop of at least two points whose interval
lies below zero is a regression; the report lists every other drop under
"Other drops".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..config import PIPELINE, TEXT, Comparison
from ..metrics import DIGITS, F1, METRICS, RECALL, Counts, Delta, Interval, paired_bootstrap, total
from ..scoring import (
    BY_FAMILY,
    BY_LABEL,
    DOCUMENT,
    LEVELS_BY_MODE,
    MAIN_LEVEL,
    OVERALL,
    ItemScore,
    any_series,
    document_count,
    label_order,
    series,
)
from .markdown import POINTS, captioned, cell, document, interval, pct, points, section, table, with_interval
from .run import LEVEL_TITLES, RATES, facet_title, helpers_cell, machine_cell, mean_seconds, peak_rss, per_item_cell

REGRESSION_DROP = 0.02
DELTA_HEADERS = [
    "gold",
    "base P %",
    "base R %",
    "base F1 %",
    "cand P %",
    "cand R %",
    "cand F1 %",
    "ΔF1",
    "ΔF1 95% CI",
    "ΔR",
    "ΔR 95% CI",
]
DELTA_CAPTION = "Rates in percent; Δ is candidate minus baseline in points with a paired 95% bootstrap interval."
GROUP_FACETS = {TEXT: ("language", "doc_type"), PIPELINE: ("language", "channel", "capture")}
DROP_METRICS = {F1: "F1", RECALL: "recall"}


@dataclass(frozen=True)
class DeltaRow:
    """``facet`` names what ``group`` is: a level, a label, a family or a document attribute."""

    scope: str
    facet: str
    group: str
    base: Counts
    candidate: Counts
    delta: Mapping[str, Delta]


@dataclass(frozen=True)
class Drop:
    """``metric`` is a key of DROP_METRICS."""

    row: DeltaRow
    metric: str
    value: float
    interval: Interval | None

    @property
    def is_regression(self) -> bool:
        # Rounding keeps a drop that prints as -2.0, such as 6/50 to 5/50, at the bar.
        big = round(self.value, DIGITS) <= -REGRESSION_DROP
        return big and self.interval is not None and self.interval.high < 0


def align(base: Sequence[ItemScore], candidate: Sequence[ItemScore]) -> tuple[list[ItemScore], list[ItemScore]]:
    """Items present in both runs, in the same order."""
    b = {item.item_id: item for item in base}
    c = {item.item_id: item for item in candidate}
    common = sorted(b.keys() & c.keys())
    return [b[k] for k in common], [c[k] for k in common]


def _row(scope: str, facet: str, group: str, base: list[Counts], candidate: list[Counts]) -> DeltaRow:
    return DeltaRow(scope, facet, group, total(base), total(candidate), paired_bootstrap(base, candidate))


def delta_rows(base: Sequence[ItemScore], candidate: Sequence[ItemScore], level: str, by: str) -> list[DeltaRow]:
    b, c = series(base, level, by), series(candidate, level, by)
    empty = [Counts()] * document_count(base)
    groups = b.keys() | c.keys()
    ordered = label_order(groups) if by == BY_LABEL else sorted(groups)
    scope = f"{LEVEL_TITLES[level]}, by {facet_title(by)}"
    return [_row(scope, facet_title(by), g, b.get(g, empty), c.get(g, empty)) for g in ordered]


def overall_rows(base: Sequence[ItemScore], candidate: Sequence[ItemScore], mode: str) -> list[DeltaRow]:
    empty = [Counts()] * document_count(base)
    rows = []
    for level in LEVELS_BY_MODE[mode]:
        b, c = series(base, level).get(OVERALL, empty), series(candidate, level).get(OVERALL, empty)
        rows.append(_row("overall", "level", LEVEL_TITLES[level], b, c))
    rows.append(_row("overall", "level", "document, any PII", any_series(base), any_series(candidate)))
    return rows


def _delta_cells(row: DeltaRow) -> list[object]:
    b, c, f1, recall = row.base, row.candidate, row.delta[F1], row.delta[RECALL]
    return [
        max(b.support, c.support),
        pct(b.precision),
        pct(b.recall),
        pct(b.f1),
        pct(c.precision),
        pct(c.recall),
        pct(c.f1),
        points(f1.value),
        interval(f1.interval, signed=True),
        points(recall.value),
        interval(recall.interval, signed=True),
    ]


def delta_table(rows: Sequence[DeltaRow], *, with_facet: bool = False) -> str:
    """One row per group; ``with_facet`` adds a column naming the facet of each group."""
    if with_facet:
        return table(["facet", "group", *DELTA_HEADERS], [[r.facet, r.group, *_delta_cells(r)] for r in rows])
    first = rows[0].facet if rows else "group"
    return table([first, *DELTA_HEADERS], [[r.group, *_delta_cells(r)] for r in rows])


def drops(rows: Sequence[DeltaRow]) -> list[Drop]:
    found = []
    for row in rows:
        for metric in DROP_METRICS:
            delta = row.delta[metric]
            if delta.value is not None and delta.value < 0:
                found.append(Drop(row, metric, delta.value, delta.interval))
    return sorted(found, key=lambda d: (d.value, d.row.scope, d.row.group, d.metric))


def _drop_table(found: Sequence[Drop]) -> str:
    rows = []
    for d in found:
        before, after = METRICS[d.metric](d.row.base), METRICS[d.metric](d.row.candidate)
        rows.append(
            [
                d.row.scope,
                d.row.group,
                DROP_METRICS[d.metric],
                pct(before),
                pct(after),
                points(d.value),
                interval(d.interval, signed=True),
            ]
        )
    return table(["scope", "group", "metric", "baseline %", "candidate %", "Δ", "95% CI"], rows)


def _headline(
    comparison: Comparison,
    metas: tuple[Mapping[str, Any], Mapping[str, Any]],
    main: DeltaRow,
    base: Sequence[ItemScore],
) -> str:
    def run_cells(name: str, role: str, counts: Counts, meta: Mapping[str, Any]) -> list[object]:
        rates = [pct(getattr(counts, metric)) for metric in RATES]
        return [name, role, *rates, mean_seconds(meta.get("timing")), peak_rss(meta.get("peak_rss_mb"))]

    deltas = [with_interval(points(d.value), d.interval, signed=True) for d in (main.delta[m] for m in RATES)]
    rows: list[list[object]] = [
        run_cells(comparison.baseline, "baseline", main.base, metas[0]),
        run_cells(comparison.candidate, "candidate", main.candidate, metas[1]),
        ["Δ", "candidate - baseline", *deltas, None, None],
    ]
    run = metas[0]["run"]
    renderings = "" if run["mode"] == TEXT else f"{len(base)} renderings of "
    caption = (
        f"Headline: {main.group} on the {renderings}{document_count(base)} {run['split']} documents both runs "
        "scored; rates in percent, Δ in points with a paired 95% bootstrap interval."
    )
    headers = ["run", "role", "P %", "R %", "F1 %", "F2 %", "s / item", "peak RSS MB"]
    return captioned(caption, table(headers, rows))


def _record(comparison: Comparison, base_meta: Mapping[str, Any], cand_meta: Mapping[str, Any]) -> str:
    def facts(meta: Mapping[str, Any]) -> list[object]:
        timing = meta.get("timing") or {}
        rss = meta.get("peak_rss_mb") or {}
        run = meta["run"]
        return [
            ", ".join(sorted(meta["components"])),
            f"{run['mode']}, {run['split']}, {run.get('channel') or 'all'}",
            round(sum(timing.get("init_seconds", {}).values()), 3),
            timing.get("total_seconds"),
            per_item_cell(timing.get("per_item_seconds")),
            timing.get("chars_per_second"),
            rss.get("self"),
            helpers_cell(rss),
            machine_cell(meta.get("environment")),
        ]

    labels = [
        "components",
        "mode, split, channel",
        "init seconds",
        "total seconds",
        "seconds per item, mean / p50 / p95",
        "characters per second",
        "peak RSS MB, run process",
        "peak RSS MB, helper processes",
        "machine",
    ]
    rows: list[Sequence[object]] = [("run", comparison.baseline, comparison.candidate)]
    rows += zip(labels, facts(base_meta), facts(cand_meta), strict=True)
    caption = "Components and runtime; compare runtime only between runs on the same machine."
    return captioned(caption, table(["", "baseline", "candidate"], rows))


def render_comparison(
    comparison: Comparison,
    base_meta: Mapping[str, Any],
    base_items: Sequence[ItemScore],
    cand_meta: Mapping[str, Any],
    cand_items: Sequence[ItemScore],
) -> str:
    base, candidate = align(base_items, cand_items)
    mode = base_meta["run"]["mode"]
    main = str(MAIN_LEVEL[mode])
    title = LEVEL_TITLES[main]
    overall = overall_rows(base, candidate, mode)
    by_label = delta_rows(base, candidate, main, BY_LABEL)
    by_label_document = delta_rows(base, candidate, DOCUMENT, BY_LABEL)
    by_group = delta_rows(base, candidate, main, BY_FAMILY)
    for facet in GROUP_FACETS[mode]:
        by_group += delta_rows(base, candidate, main, facet)
    found = drops([*overall, *by_label, *by_label_document, *by_group])
    headline = next(row for row in overall if row.group == title)
    threshold = f"{REGRESSION_DROP * POINTS:.0f} points"
    regressions = f"Drops of {threshold} or more whose interval lies below zero."
    other = f"Drops smaller than {threshold}, or whose interval contains zero."
    parts = [
        f"# Comparison {cell(comparison.name)}\n",
        _headline(comparison, (base_meta, cand_meta), headline, base),
        section("Scores by level", captioned(DELTA_CAPTION, delta_table(overall))),
        section(f"By label, {title}", captioned(DELTA_CAPTION, delta_table(by_label))),
        section("By label, document level", captioned(DELTA_CAPTION, delta_table(by_label_document))),
        section(f"By group, {title}", captioned(DELTA_CAPTION, delta_table(by_group, with_facet=True))),
        section("Regressions", captioned(regressions, _drop_table([d for d in found if d.is_regression]))),
        section("Other drops", captioned(other, _drop_table([d for d in found if not d.is_regression]))),
        section("Run record", _record(comparison, base_meta, cand_meta)),
    ]
    return document(parts)
