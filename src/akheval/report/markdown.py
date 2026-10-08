"""Small Markdown helpers shared by the reports."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..metrics import Interval

MISSING = "-"
POINTS = 100


def cell(value: object) -> str:
    """One table cell: pipes escaped, line breaks flattened, an empty value shown as a dash."""
    text = "" if value is None else " ".join(str(value).split())
    return text.replace("|", "\\|") if text else MISSING


def table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    if len(lines) == 2:
        return "No rows.\n"
    return "\n".join(lines) + "\n"


def document(parts: Iterable[str]) -> str:
    """Join report sections into one Markdown file that ends with exactly one newline."""
    return "\n".join(parts).rstrip("\n") + "\n"


def pct(value: float | None) -> str:
    """A rate in percent with one decimal."""
    return MISSING if value is None else f"{POINTS * value:.1f}"


def points(value: float | None) -> str:
    """A difference of rates in signed percentage points."""
    return MISSING if value is None else f"{POINTS * value:+.1f}"


def interval(value: Interval | None, *, signed: bool = False) -> str:
    if value is None:
        return MISSING
    show = points if signed else pct
    return f"[{show(value.low)}, {show(value.high)}]"


def with_interval(estimate: str, bounds: Interval | None, *, signed: bool = False) -> str:
    """A formatted estimate followed by its interval in brackets, when there is one."""
    return estimate if bounds is None else f"{estimate} {interval(bounds, signed=signed)}"


def seconds(value: float | None) -> str:
    return MISSING if value is None else f"{value:.4f}"


def captioned(caption: str, body: str) -> str:
    return f"{caption}\n\n{body}"


def section(title: str, body: str, level: int = 2) -> str:
    return f"{'#' * level} {title}\n\n{body}\n"
