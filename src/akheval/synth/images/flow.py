"""Layout blocks: parsing, text styles per block kind, placement in columns and drawing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

from PIL import ImageDraw

from ..builder import BLOCK_KINDS, FRONT, SIDES
from .text import RGB, Box, Line, Styled, TextStyle, Typesetter, baseline, draw_line, line_left

HEADING_KINDS = ("title", "header", "blank")

# A blank block is this share of a body line tall.
BLANK_SHARE = 0.6
# Each round of fitting shrinks every style by this factor.
SHRINK_STEP = 0.93
# Width a monospace line is measured at before it is scaled to its box.
PROBE_SIZE = 100


@dataclass(frozen=True, slots=True)
class Block:
    index: int
    kind: str
    text: str
    side: str = FRONT


def parse_layout(layout: Sequence[Mapping[str, Any]]) -> list[Block]:
    """Validate the layout blocks the text generator produced."""
    blocks = []
    for i, raw in enumerate(layout):
        kind, text, side = raw.get("kind"), raw.get("text"), raw.get("side", FRONT)
        if kind not in BLOCK_KINDS:
            raise ValueError(f"layout block {i}: unknown kind {kind!r}")
        if not isinstance(text, str) or "\n" in text:
            raise ValueError(f"layout block {i}: text must be a single line string")
        if kind == "blank" and text:
            raise ValueError(f"layout block {i}: a blank block must have empty text")
        if side not in SIDES:
            raise ValueError(f"layout block {i}: unknown side {side!r}")
        blocks.append(Block(i, kind, text, side))
    return blocks


def leading_headings(blocks: Sequence[Block]) -> int:
    """Number of title, header and blank blocks at the start of ``blocks``."""
    count = 0
    while count < len(blocks) and blocks[count].kind in HEADING_KINDS:
        count += 1
    return count


@dataclass(frozen=True, slots=True)
class KindStyles:
    """Text styles of one document. ``spacing`` is the line pitch as a multiple of the font size."""

    title: TextStyle
    header: TextStyle
    label: TextStyle
    value: TextStyle
    body: TextStyle
    mono: TextStyle
    spacing: float

    def scaled(self, factor: float) -> KindStyles:
        return replace(
            self,
            title=self.title.scaled(factor),
            header=self.header.scaled(factor),
            label=self.label.scaled(factor),
            value=self.value.scaled(factor),
            body=self.body.scaled(factor),
            mono=self.mono.scaled(factor),
        )

    def styled(self, block: Block) -> Styled:
        if block.kind == "field":
            return Styled.field(block.text, self.label, self.value)
        style = {"title": self.title, "header": self.header, "mrz": self.mono}.get(block.kind, self.body)
        return Styled.plain(block.text, style)

    def to_params(self) -> dict[str, Any]:
        sizes = {f.name: getattr(self, f.name).size for f in fields(self) if f.name != "spacing"}
        return {"sizes_px": sizes, "spacing": round(self.spacing, 3), "value_bold": self.value.bold}


@dataclass(frozen=True, slots=True)
class Placed:
    block: Block
    lines: tuple[Line, ...]
    pitch: float
    height: float
    align: str


def place(ts: Typesetter, block: Block, styles: KindStyles, width: float, *, min_scale: float) -> Placed:
    """Lay out one block in ``width``; titles are centred, other blocks follow their direction."""
    if block.kind == "blank":
        return Placed(block, (), 0.0, styles.body.size * styles.spacing * BLANK_SHARE, "start")
    lines = tuple(ts.fit(styles.styled(block), width, min_scale=min_scale))
    pitch = max(line.size for line in lines) * styles.spacing
    return Placed(block, lines, pitch, pitch * len(lines), "center" if block.kind == "title" else "start")


def fit_columns(
    ts: Typesetter,
    groups: Sequence[Sequence[Block]],
    styles: KindStyles,
    width: float,
    height: float,
    *,
    min_scale: float,
    block_min_scale: float,
) -> tuple[list[list[Placed]], float]:
    """Place each group in its own column of ``width``, shrinking all styles until every column fits ``height``."""
    scale = 1.0
    while True:
        scaled = styles.scaled(scale)
        columns = [[place(ts, b, scaled, width, min_scale=block_min_scale) for b in group] for group in groups]
        if all(column_height(column) <= height for column in columns):
            return columns, scale
        if scale <= min_scale:
            raise ValueError(f"layout blocks do not fit a {width:.0f} x {height:.0f} px region")
        scale = max(min_scale, scale * SHRINK_STEP)


def column_height(placed: Sequence[Placed]) -> float:
    return sum(p.height for p in placed)


def spread(placed: Sequence[Placed], height: float, limit: float) -> float:
    """Gap between blocks that spreads them over ``height``, at most ``limit``."""
    if len(placed) < 2:
        return 0.0
    return max(0.0, min(limit, (height - column_height(placed)) / (len(placed) - 1)))


def draw_column(
    draw: ImageDraw.ImageDraw, placed: Sequence[Placed], left: float, right: float, top: float, gap: float = 0.0
) -> list[Box]:
    """Draw placed blocks from ``top`` down, ``gap`` apart, and return the box of every drawn line."""
    boxes: list[Box] = []
    y = top
    for p in placed:
        for i, line in enumerate(p.lines):
            x = line_left(line, left, right, p.align)
            line_top = y + i * p.pitch
            draw_line(draw, line, x, baseline(line_top, p.pitch, line.size))
            boxes.append((x, line_top, x + line.width, line_top + p.pitch))
        y += p.height + gap
    return boxes


def fit_mono(
    ts: Typesetter, blocks: Sequence[Block], color: RGB, width: float, max_height: float, spacing: float
) -> list[Placed]:
    """Place machine-readable-zone lines in the font size that makes the longest line span ``width``."""
    probe = TextStyle(PROBE_SIZE, color, mono=True)
    widest = max(ts.layout(Styled.plain(b.text, probe), "ltr").width for b in blocks)
    by_width = PROBE_SIZE * width / max(widest, 1.0)
    by_height = max_height / (len(blocks) * spacing)
    sized = TextStyle(max(1, int(min(by_width, by_height))), color, mono=True)
    placed = []
    for block in blocks:
        line = ts.layout(Styled.plain(block.text, sized), "ltr")
        placed.append(Placed(block, (line,), sized.size * spacing, sized.size * spacing, "start"))
    return placed


def block_params(placed: Sequence[Placed]) -> list[dict[str, Any]]:
    """Font, size and line count of every drawn block, ordered by block index."""
    out = []
    for p in sorted(placed, key=lambda p: p.block.index):
        if not p.lines:
            continue
        entry: dict[str, Any] = {
            "index": p.block.index,
            "size": max(line.size for line in p.lines),
            "lines": len(p.lines),
            "fonts": sorted(set().union(*(line.faces for line in p.lines))),
        }
        missing = "".join(sorted({char for line in p.lines for char in line.missing}))
        if missing:
            entry["missing_glyphs"] = missing
        out.append(entry)
    return out
