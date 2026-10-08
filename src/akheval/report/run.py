"""Markdown report of one scored run: headline, scores, errors, then the run record."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from ..config import EXTRACT, PIPELINE, TEXT
from ..matching import ANY_TYPE, LENIENT, STRICT, VALUE, VALUE_LENIENT, VALUE_STRICT
from ..metrics import F1, F2, PRECISION, RECALL, Interval
from ..scoring import (
    BOUNDARY,
    DETECTION_FAILED,
    DOC_FALSE_NEGATIVE,
    DOC_FALSE_POSITIVE,
    DOCUMENT,
    EXTRACTION_FAILED,
    FALSE_NEGATIVE,
    FALSE_POSITIVE,
    LEVELS_BY_MODE,
    NO_LOOK_ALIKE,
    VALUE_MISSED,
    VALUE_NEAR,
    VALUE_PARTIAL,
    VALUE_SPURIOUS,
    WRONG_TYPE,
    ErrorRecord,
    ItemScore,
    facets_of,
    family_of,
    label_order,
)
from .markdown import MISSING, captioned, cell, document, interval, pct, seconds, section, table, with_interval

LEVEL_TITLES = {
    STRICT: "strict span",
    LENIENT: "lenient span",
    ANY_TYPE: "any-type span",
    VALUE_LENIENT: "lenient value",
    VALUE: "exact or near value",
    VALUE_STRICT: "exact value",
    DOCUMENT: "document",
}
SIDE_LEVEL = {TEXT: STRICT, PIPELINE: VALUE}
"""Level shown next to the headline level in the per-label table."""
COLUMN_PREFIX = {LENIENT: "lenient", STRICT: "strict", VALUE_LENIENT: "lenient", VALUE: "exact/near"}
FACET_TITLES = {"doc_type": "document type"}
KIND_NOTES = {
    FALSE_POSITIVE: "Predicted spans that overlap no gold span; look-alike names the negative they hit.",
    FALSE_NEGATIVE: "Gold spans that no prediction overlaps.",
    BOUNDARY: "Predicted spans with the right label and different boundaries; gold shows the gold span.",
    WRONG_TYPE: "Predicted spans over a gold span of another label.",
    VALUE_NEAR: "Values that match a gold value within OCR noise; score is the string similarity.",
    VALUE_PARTIAL: "Values that contain, or share words with, a gold value; score is the string similarity.",
    VALUE_MISSED: "Gold values with no matching predicted value.",
    VALUE_SPURIOUS: "Predicted values that match no gold value; look-alike names the negative they resemble.",
    EXTRACTION_FAILED: "Files the extractor could not read; text is the error.",
    DETECTION_FAILED: "Texts the detector failed on; text is the error.",
}
DOCUMENT_ERRORS = (DOC_FALSE_NEGATIVE, DOC_FALSE_POSITIVE)
"""Kinds the per-label document table already counts; the report shows no samples of them."""
COUNT_HEADERS = ["gold", "TP", "FP", "FN", "P %", "R %", "F1 %", "F2 %"]
QUALITY_HEADERS = ["items", "reference chars", "CER %", "WER %", "token recall %", "PII token recall %"]
QUALITY_KEYS = ("cer", "wer", "token_recall", "pii_token_recall")
RATES = (PRECISION, RECALL, F1, F2)
MAX_ERRORS_PER_KIND = 12
MAX_LABEL_ROWS = 20
SHA_CHARS = 12


def counts_cells(c: Mapping[str, Any]) -> list[object]:
    return [c["support"], c["tp"], c["fp"], c["fn"], pct(c["precision"]), pct(c["recall"]), pct(c["f1"]), pct(c["f2"])]


def quality_rates(q: Mapping[str, Any]) -> list[str]:
    return [pct(q.get(key)) for key in QUALITY_KEYS]


def quality_cells(q: Mapping[str, Any]) -> list[object]:
    return [q["items"], q["ref_chars"], *quality_rates(q)]


def to_interval(d: Mapping[str, float] | None) -> Interval | None:
    return None if d is None else Interval(d["low"], d["high"])


def rate_with_interval(overall: Mapping[str, Any], metric: str) -> str:
    """A rate in percent followed by its 95% interval, such as ``44.3 [40.5, 48.2]``."""
    return with_interval(pct(overall[metric]), to_interval((overall.get("ci95") or {}).get(metric)))


def per_item_cell(stats: Mapping[str, Any] | None) -> str:
    if not stats:
        return MISSING
    return " / ".join(seconds(stats.get(key)) for key in ("mean", "p50", "p95"))


def mean_seconds(timing: Mapping[str, Any] | None) -> str:
    return seconds(((timing or {}).get("per_item_seconds") or {}).get("mean"))


def peak_rss(rss: Mapping[str, Any] | None) -> float | None:
    """The largest peak of the run process and its helper processes, in MiB."""
    values = [v for v in ((rss or {}).get("self"), *_helpers(rss).values()) if v is not None]
    return max(values) if values else None


def _helpers(rss: Mapping[str, Any] | None) -> dict[str, float]:
    helpers: dict[str, float] = (rss or {}).get("helpers") or {}
    return helpers


def helpers_cell(rss: Mapping[str, Any] | None) -> str | None:
    """Each helper program with its peak RSS in MiB, such as ``akh-ocr 489.2``."""
    return ", ".join(f"{program} {peak}" for program, peak in _helpers(rss).items()) or None


def machine_cell(environment: Mapping[str, Any] | None) -> str | None:
    """CPU model, logical CPU count and RAM of the machine a run used."""
    if not environment:
        return None
    return f"{environment.get('cpu')}, {environment.get('cpu_count')} CPUs, {environment.get('ram_gb')} GB RAM"


def _headline(meta: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    mode, split = summary["mode"], meta["run"]["split"]
    runtime = [mean_seconds(meta.get("timing")), peak_rss(meta.get("peak_rss_mb"))]
    if mode == EXTRACT:
        rates = quality_rates((summary.get("extraction") or {}).get("overall") or {})
        headers = [*QUALITY_HEADERS[2:], "documents", "renderings", "s / item", "peak RSS MB"]
        caption = f"Extraction on the {split} split against the text each file shows, in percent."
        return captioned(caption, table(headers, [[*rates, summary["documents"], summary["items"], *runtime]]))
    main = summary["main_level"]
    overall = summary["levels"][main]["overall"]
    sizes = [summary["documents"]] if mode == TEXT else [summary["documents"], summary["items"]]
    size_headers = ["documents"] if mode == TEXT else ["documents", "renderings"]
    headers = ["level", "P %", "R %", "F1 %", "F2 %", *size_headers, "s / item", "peak RSS MB"]
    row: list[object] = [LEVEL_TITLES[main], *(rate_with_interval(overall, m) for m in RATES), *sizes, *runtime]
    caption = (
        f"Headline on the {split} split: rates in percent with 95% bootstrap intervals over documents; "
        "seconds per item is the mean processing time of one item."
    )
    return captioned(caption, table(headers, [row]))


def _scores(summary: Mapping[str, Any]) -> str:
    rows = []
    for level, entry in summary["levels"].items():
        overall = entry["overall"]
        rows.append([LEVEL_TITLES[level], *counts_cells(overall), interval(to_interval(overall["ci95"][F1]))])
    if "document_any" in summary:
        rows.append(["document, any PII", *counts_cells(summary["document_any"]), None])
    caption = (
        "Micro-averaged counts per scoring level; gold counts gold spans, values or document labels, "
        "and the F1 interval resamples documents."
    )
    return captioned(caption, table(["level", *COUNT_HEADERS, "F1 95% CI"], rows))


def _rates(c: Mapping[str, Any] | None, metrics: Sequence[str]) -> list[str]:
    return [pct(None if c is None else c[m]) for m in metrics]


def _by_label(summary: Mapping[str, Any], main: str, side: str) -> str:
    main_labels, side_labels = summary["levels"][main]["by_label"], summary["levels"][side]["by_label"]
    rows = []
    for label in label_order(main_labels.keys() | side_labels.keys()):
        gold = (main_labels.get(label) or side_labels[label])["support"]
        main_rates = _rates(main_labels.get(label), RATES)
        rows.append([label, family_of(label), gold, *main_rates, *_rates(side_labels.get(label), RATES[:3])])
    first, second = COLUMN_PREFIX[main], COLUMN_PREFIX[side]
    headers = [f"{first} {m} %" for m in ("P", "R", "F1", "F2")] + [f"{second} {m} %" for m in ("P", "R", "F1")]
    caption = f"Per label in percent: {LEVEL_TITLES[main]} matching, then {LEVEL_TITLES[side]} matching."
    return captioned(caption, table(["label", "family", "gold", *headers], rows))


def _by_label_document(summary: Mapping[str, Any]) -> str:
    by_label = summary["levels"][DOCUMENT]["by_label"]
    rows = [[label, family_of(label), *counts_cells(by_label[label])] for label in label_order(by_label)]
    caption = (
        "Per label at document level: gold counts items that contain the label, "
        "a hit is the label flagged anywhere in the item."
    )
    return captioned(caption, table(["label", "family", *COUNT_HEADERS], rows))


def facet_title(facet: str) -> str:
    return FACET_TITLES.get(facet, facet)


def _by_group(summary: Mapping[str, Any], main: str) -> str:
    entry = summary["levels"][main]
    rows = [["family", name, *counts_cells(c)] for name, c in entry["by_family"].items()]
    for facet in facets_of(summary["mode"]):
        rows += [[facet_title(facet), name, *counts_cells(c)] for name, c in entry[f"by_{facet}"].items()]
    caption = f"Rates in percent per group at {LEVEL_TITLES[main]} level; gold counts gold items in the group."
    return captioned(caption, table(["facet", "group", *COUNT_HEADERS], rows))


def _quality(extraction: Mapping[str, Any], facets: Sequence[str]) -> str:
    rows: list[list[object]] = []
    if extraction["overall"] is not None:
        rows.append(["all", "all", *quality_cells(extraction["overall"])])
    for facet in facets:
        rows += [[facet_title(facet), k, *quality_cells(q)] for k, q in extraction[f"by_{facet}"].items()]
    caption = (
        "Extracted text against the text each file shows, in percent, compared as words split at whitespace and "
        "list separators without edge punctuation or bidi marks; token recall is the share of reference words found "
        "in any order, PII token recall the share of the words of the gold values."
    )
    return captioned(caption, table(["facet", "group", *QUALITY_HEADERS], rows))


def sample_errors(errors: Sequence[ErrorRecord], limit: int = MAX_ERRORS_PER_KIND) -> list[ErrorRecord]:
    """Deterministic sample spread across labels and languages: round-robin over (label, language) groups."""
    groups: dict[tuple[str, str], list[ErrorRecord]] = defaultdict(list)
    for error in sorted(errors, key=lambda e: (e.doc_id, e.rendering_id or "", e.text)):
        groups[(error.label, error.language)].append(error)
    queues = [groups[key] for key in sorted(groups)]
    picked: list[ErrorRecord] = []
    depth = 0
    while len(picked) < limit and any(depth < len(q) for q in queues):
        picked += [q[depth] for q in queues if depth < len(q)][: limit - len(picked)]
        depth += 1
    return picked


def drop_empty_columns(
    headers: Sequence[str], rows: Sequence[Sequence[object]]
) -> tuple[list[str], list[list[object]]]:
    """Remove every column whose cells are all empty."""
    keep = [i for i in range(len(headers)) if any(row[i] not in (None, "") for row in rows)]
    return [headers[i] for i in keep], [[row[i] for i in keep] for row in rows]


def _error_cells(e: ErrorRecord) -> list[object]:
    return [
        e.rendering_id or e.doc_id,
        e.language,
        e.label,
        e.text,
        e.context,
        e.gold_text if e.gold_label in (None, e.label) else f"{e.gold_label} {e.gold_text}",
        e.negative_kind,
        None if e.score is None else f"{e.score:.3f}",
    ]


def _error_samples(kind: str, errors: Sequence[ErrorRecord]) -> str:
    picked = sample_errors(errors)
    headers, rows = drop_empty_columns(
        ["item", "lang", "label", "text", "context", "gold", "look-alike", "score"], [_error_cells(e) for e in picked]
    )
    title = f"{kind}: {len(picked)} of {len(errors)}"
    return section(title, captioned(KIND_NOTES.get(kind, kind), table(headers, rows)), level=3)


def _errors(summary: Mapping[str, Any], items: Sequence[ItemScore]) -> str:
    by_kind: dict[str, list[ErrorRecord]] = defaultdict(list)
    for item in items:
        for error in item.errors:
            by_kind[error.kind].append(error)
    if not by_kind:
        return "No errors.\n"
    caption = "Error records in errors.jsonl by kind; document-level kinds are counted per label above."
    parts = [captioned(caption, table(["kind", "count"], [[kind, len(by_kind[kind])] for kind in sorted(by_kind)]))]
    if look_alikes := summary.get("false_positives_by_look_alike"):
        note = f"False positives (spans or values) by the look-alike they hit; {NO_LOOK_ALIKE} means no look-alike."
        parts.append(captioned(note, table(["look-alike", "count"], look_alikes.items())))
    parts += [_error_samples(kind, by_kind[kind]) for kind in sorted(by_kind) if kind not in DOCUMENT_ERRORS]
    return "\n".join(parts)


def _labels_table(counts: Mapping[str, int]) -> str:
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_LABEL_ROWS]
    caption = f"Detector labels with no taxonomy label, dropped before scoring; count of predictions, top {len(top)}."
    return captioned(caption, table(["label", "count"], top))


def _record(meta: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    run, dataset = meta["run"], meta["dataset"]
    timing = meta.get("timing") or {}
    rss = meta.get("peak_rss_mb") or {}
    components = ", ".join(f"{name} ({info.get('kind', '?')})" for name, info in sorted(meta["components"].items()))
    init = ", ".join(f"{name} {value:.2f}" for name, value in timing.get("init_seconds", {}).items())
    rows = [
        ("mode, split, channel", f"{run['mode']}, {run['split']}, {run.get('channel') or 'all'}"),
        ("items", summary["items"]),
        ("dataset", f"{dataset['name']} {dataset['version']}, frozen: {'yes' if dataset['frozen'] else 'no'}"),
        ("documents.jsonl sha256", dataset["documents_sha256"][:SHA_CHARS]),
        ("components", components),
        ("init seconds", init),
        ("total seconds", timing.get("total_seconds")),
        ("seconds per item, mean / p50 / p95", per_item_cell(timing.get("per_item_seconds"))),
        ("characters per second", timing.get("chars_per_second")),
        ("peak RSS MB, run process", rss.get("self")),
        ("peak RSS MB, helper processes", helpers_cell(rss)),
        ("machine", machine_cell(meta.get("environment"))),
    ]
    caption = "Provenance and runtime; run.json holds component versions, checksums and the environment."
    return captioned(caption, table(["field", "value"], rows))


def render_run(meta: Mapping[str, Any], summary: Mapping[str, Any], items: Sequence[ItemScore]) -> str:
    mode = summary["mode"]
    parts = [f"# Run {cell(meta['run']['name'])}\n", _headline(meta, summary)]
    if LEVELS_BY_MODE[mode]:
        main = summary["main_level"]
        parts += [
            section("Scores by level", _scores(summary)),
            section("By label", _by_label(summary, main, SIDE_LEVEL[mode])),
            section("By label, document level", _by_label_document(summary)),
            section("By group", _by_group(summary, main)),
        ]
    if "extraction" in summary:
        parts.append(section("Extraction quality", _quality(summary["extraction"], facets_of(mode))))
    parts.append(section("Errors", _errors(summary, items)))
    if summary.get("unmapped_labels"):
        parts.append(section("Detector labels outside the taxonomy", _labels_table(summary["unmapped_labels"])))
    parts.append(section("Run record", _record(meta, summary)))
    return document(parts)
