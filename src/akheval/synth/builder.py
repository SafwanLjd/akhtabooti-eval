"""Layout builder: appends blocks of text and records gold spans and negatives at their offsets.

The document text is the blocks' text joined with newlines, so an offset recorded
while a block is written stays valid in the final text.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..dataset import Span
from ..taxonomy import LABELS, NEGATIVE_KINDS

BLOCK_KINDS = ("title", "header", "field", "paragraph", "row", "mrz", "message", "blank")
FRONT = "front"
BACK = "back"
SIDES = (FRONT, BACK)
CELL_SEPARATOR = " | "
FIELD_SEPARATOR = ": "


@dataclass(frozen=True, slots=True)
class Ent:
    """A gold entity value inside a block."""

    value: str
    label: str

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"unknown label {self.label!r}")
        if not self.value.strip():
            raise ValueError(f"empty {self.label} value")


@dataclass(frozen=True, slots=True)
class Neg:
    """A look-alike value that a detector must not report."""

    value: str
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in NEGATIVE_KINDS:
            raise ValueError(f"unknown negative kind {self.kind!r}")
        if not self.value.strip():
            raise ValueError(f"empty {self.kind} value")


Part = str | Ent | Neg
Cell = Part | Sequence[Part]
Blocks = Sequence[Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class Layout:
    text: str
    blocks: list[dict[str, Any]]
    spans: tuple[Span, ...]
    negatives: tuple[Span, ...]


class LayoutBuilder:
    """Collects layout blocks for one document."""

    def __init__(self) -> None:
        self._blocks: list[dict[str, Any]] = []
        self._spans: list[Span] = []
        self._negatives: list[Span] = []
        self._length = 0

    def add(self, kind: str, *parts: Part, field: str | None = None, side: str | None = None) -> None:
        """Append one block whose text is the concatenation of ``parts``."""
        if kind not in BLOCK_KINDS:
            raise ValueError(f"unknown block kind {kind!r}")
        if side is not None and side not in SIDES:
            raise ValueError(f"unknown side {side!r}")
        start = self._length + 1 if self._blocks else 0
        pieces: list[str] = []
        for part in parts:
            value = part if isinstance(part, str) else part.value
            if "\n" in value:
                raise ValueError(f"block text must be one line: {value!r}")
            offset = start + sum(len(p) for p in pieces)
            if isinstance(part, Ent):
                self._spans.append(Span(offset, offset + len(value), part.label, value))
            elif isinstance(part, Neg):
                self._negatives.append(Span(offset, offset + len(value), part.kind, value))
            pieces.append(value)
        text = "".join(pieces)
        block: dict[str, Any] = {"kind": kind, "text": text}
        if side is not None:
            block["side"] = side
        if field is not None:
            block["field"] = field
        self._blocks.append(block)
        self._length = start + len(text)

    def title(self, *parts: Part, side: str | None = None) -> None:
        self.add("title", *parts, side=side)

    def header(self, *parts: Part, side: str | None = None) -> None:
        self.add("header", *parts, side=side)

    def field(self, label: str, *value: Part, key: str | None = None, side: str | None = None) -> None:
        """A "Label: value" line."""
        self.add("field", label, FIELD_SEPARATOR, *value, field=key, side=side)

    def para(self, *parts: Part, side: str | None = None) -> None:
        self.add("paragraph", *parts, side=side)

    def row(self, *cells: Cell) -> None:
        """A table row; cells are joined with " | "."""
        parts: list[Part] = []
        for i, cell in enumerate(cells):
            if i:
                parts.append(CELL_SEPARATOR)
            if isinstance(cell, str | Ent | Neg):
                parts.append(cell)
            else:
                parts.extend(cell)
        self.add("row", *parts)

    def mrz(self, line: str, label: str, side: str | None = None) -> None:
        self.add("mrz", Ent(line, label), side=side)

    def message(self, time: str, sender: Part, *parts: Part) -> None:
        """A chat line "[hh:mm] Name: text"."""
        self.add("message", f"[{time}] ", sender, FIELD_SEPARATOR, *parts)

    def blank(self, side: str | None = None) -> None:
        self.add("blank", side=side)

    def build(self) -> Layout:
        return Layout(
            text="\n".join(block["text"] for block in self._blocks),
            blocks=[dict(b) for b in self._blocks],
            spans=tuple(sorted(self._spans, key=lambda s: (s.start, s.end))),
            negatives=tuple(sorted(self._negatives, key=lambda s: (s.start, s.end))),
        )


def block_side(block: Mapping[str, Any]) -> str:
    """Card side of a layout block; a block without one belongs to the front."""
    return str(block.get("side", FRONT))


def has_back(blocks: Blocks) -> bool:
    return any(block_side(b) == BACK for b in blocks)


def side_ranges(blocks: Blocks, side: str) -> tuple[tuple[int, int], ...]:
    """Character ranges of the layout text that the blocks of one card side cover.

    Each run of consecutive ``side`` blocks gives one range. The range starts at the
    first non-blank block of the run and ends with the last one, so the newlines that
    join runs and blank lines at a run's edges stay outside.
    """
    ranges: list[tuple[int, int]] = []
    offset = 0
    for run_side, run in itertools.groupby(blocks, key=block_side):
        lines: list[tuple[int, int]] = []
        for block in run:
            length = len(str(block["text"]))
            if length:
                lines.append((offset, offset + length))
            offset += length + 1
        if run_side == side and lines:
            ranges.append((lines[0][0], lines[-1][1]))
    return tuple(ranges)
