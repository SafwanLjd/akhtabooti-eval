"""A4 page renderer: letters, forms and tables drawn at 150 to 200 dpi, or finer for a fine scan."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw

from ..builder import CELL_SEPARATOR, FRONT
from .flow import Block, KindStyles, Placed, block_params, draw_column, leading_headings, place
from .fonts import SANS_FAMILIES, SERIF_FAMILIES, choose_fonts
from .furniture import accent, emblem, pastel
from .paper import free_spot, signature, stamp, staple
from .sheet import A4_MM, MM_PER_INCH, Face, Sheet
from .text import RGB, Box, Direction, Line, Styled, TextStyle, Typesetter, baseline, line_left, paragraph_direction
from .text import draw_line as draw_text_line

DPI_RANGE = (150.0, 200.0)
MM_PER_POINT = MM_PER_INCH / 72
# Smallest share of the starting type size before the page grows taller instead.
MIN_PAGE_SCALE = 0.8
# Smallest share of the body size a table shrinks to so that its longest words fit their columns.
MIN_TABLE_SCALE = 0.7
# Chances of a signature, of a stamp beside it or on an unsigned page, and of a staple.
SIGN_P = 0.55
STAMP_SIGNED_P = 0.7
STAMP_P = 0.2
STAPLE_P = 0.12
# Lines at the end of the page that count as its closing block.
CLOSING_LINES = 3


@dataclass(frozen=True, slots=True)
class Table:
    blocks: tuple[Block, ...]
    cells: tuple[tuple[tuple[Line, ...], ...], ...]
    col_widths: tuple[float, ...]
    row_heights: tuple[float, ...]
    pitch: float
    pad: float
    direction: Direction
    shaded_header: bool

    @property
    def height(self) -> float:
        return sum(self.row_heights)

    @property
    def width(self) -> float:
        return sum(self.col_widths)


Item = Placed | Table


@dataclass
class _Page:
    ts: Typesetter
    styles: KindStyles
    px_per_mm: float
    rule_color: RGB
    shaded_header: bool
    full_width_tables: bool


def render_page(blocks: Sequence[Block], rng: random.Random, min_dpi: float = 0.0) -> Sheet:
    """Draw the page at a resolution from DPI_RANGE, raised to ``min_dpi`` when that is finer."""
    if any(b.side != FRONT for b in blocks):
        raise ValueError("page layouts have one side; remove 'side': 'back'")
    dpi = max(rng.uniform(*DPI_RANGE), min_dpi)
    ppm = dpi / MM_PER_INCH
    serif = rng.random() < 0.5
    fonts = choose_fonts(rng, SERIF_FAMILIES if serif else SANS_FAMILIES)
    ink_level = rng.randint(0, 40)
    ink = (ink_level, ink_level, ink_level)
    heading = accent(rng) if rng.random() < 0.4 else ink
    pt = ppm * MM_PER_POINT
    body_pt = rng.uniform(10.0, 12.0)
    styles = KindStyles(
        title=TextStyle(round(rng.uniform(15.0, 19.0) * pt), heading, bold=True),
        header=TextStyle(round(rng.uniform(11.5, 13.5) * pt), heading, bold=True),
        label=TextStyle(round(body_pt * pt), ink, bold=rng.random() < 0.7),
        value=TextStyle(round(body_pt * pt), ink),
        body=TextStyle(round(body_pt * pt), ink),
        mono=TextStyle(round(body_pt * pt), ink, mono=True),
        spacing=rng.uniform(1.3, 1.6),
    )
    page = _Page(
        Typesetter(fonts),
        styles,
        ppm,
        rule_color=accent(rng) if rng.random() < 0.3 else (150, 150, 150),
        shaded_header=rng.random() < 0.7,
        full_width_tables=rng.random() < 0.5,
    )
    width = round(A4_MM[0] * ppm)
    side_mm, top_mm, bottom_mm = rng.uniform(18.0, 25.0), rng.uniform(15.0, 25.0), rng.uniform(15.0, 25.0)
    left, right = side_mm * ppm, width - side_mm * ppm
    letterhead = rng.random() < 0.5
    head_count = leading_headings(blocks)
    rule = head_count > 0 and rng.random() < 0.6
    top = top_mm * ppm
    mark = rng.uniform(13.0, 18.0) * ppm if letterhead else 0.0
    if letterhead:
        top += mark + 4 * ppm
    gaps = 3 * ppm if rule else 0.0
    available = A4_MM[1] * ppm - top - bottom_mm * ppm - gaps
    items, scale = _fit_page(page, blocks, right - left, available)
    content = sum(item.height for item in items) + gaps
    height = max(round(A4_MM[1] * ppm), round(top + content + bottom_mm * ppm))
    paper = rng.randint(246, 255)
    image = Image.new("RGB", (width, height), (paper, paper, max(paper - rng.randint(0, 6), 0)))
    draw = ImageDraw.Draw(image)
    params: dict[str, Any] = {
        "dpi": round(dpi, 2),
        "px_per_mm": round(ppm, 4),
        "size_px": [width, height],
        "extended": height > round(A4_MM[1] * ppm),
        "margins_mm": {"side": round(side_mm, 2), "top": round(top_mm, 2), "bottom": round(bottom_mm, 2)},
        "fonts": fonts.to_params(),
        "styles": styles.to_params(),
        "scale": round(scale, 3),
        "letterhead": letterhead,
        "rule": rule,
        "shaded_table_header": page.shaded_header,
        "full_width_tables": page.full_width_tables,
    }
    if letterhead:
        mx = (width - mark) / 2
        params["emblem"] = emblem(draw, (mx, top_mm * ppm, mx + mark, top_mm * ppm + mark), rng, heading, pastel(rng))
    boxes = _draw_items(page, draw, items, left, right, top, rule_after=head_count if rule else 0)
    placed = [item for item in items if isinstance(item, Placed)]
    params["blocks"] = block_params(placed)
    params["tables"] = [{"rows": len(t.blocks), "columns": len(t.col_widths)} for t in items if isinstance(t, Table)]
    texts = [b.text for b in blocks if b.text]
    direction = paragraph_direction(texts[-1]) if texts else "ltr"
    params["marks"] = _marks(image, boxes, (left, right), direction, rng, ppm)
    return Sheet("page", [Face(image, boxes)], ppm, styles.body.size / ppm, params)


def _marks(
    image: Image.Image,
    boxes: Sequence[Box],
    sides: tuple[float, float],
    direction: Direction,
    rng: random.Random,
    ppm: float,
) -> dict[str, Any]:
    """Sign and stamp the page near its closing lines and, now and then, staple its top corner; never over text."""
    marks: dict[str, Any] = {}
    if not boxes:
        return marks
    left, right = sides
    last = max(b[3] for b in boxes)
    closing_top = sorted(b[1] for b in boxes)[-min(CLOSING_LINES, len(boxes))]
    region = (left, closing_top - 20 * ppm, right, image.height - 10 * ppm)
    start = left + 0.3 * (right - left) if direction == "ltr" else right - 0.3 * (right - left)
    signed = None
    if rng.random() < SIGN_P:
        size = (rng.uniform(40, 60) * ppm, rng.uniform(12, 18) * ppm)
        spot = free_spot(boxes, size, region, (start, last + 8 * ppm), rng, ppm)
        if spot is not None:
            signed = (spot[0], spot[1], spot[0] + size[0], spot[1] + size[1])
            marks["signature"] = signature(image, signed, rng, ppm)
    if rng.random() < (STAMP_SIGNED_P if signed else STAMP_P):
        radius = rng.uniform(14, 20) * ppm
        if signed:
            near = ((signed[0] + signed[2]) / 2 + rng.choice((-0.45, 0.45)) * (signed[2] - signed[0]), signed[3])
        else:
            near = (right - (start - left), last + radius)
        spot = free_spot(boxes, (2 * radius, 2 * radius), region, near, rng, ppm)
        if spot is not None:
            marks["stamp"] = stamp(image, (spot[0] + radius, spot[1] + radius), radius, rng)
    if rng.random() < STAPLE_P:
        inset = rng.uniform(9, 12) * ppm
        corner = (inset if direction == "ltr" else image.width - inset, rng.uniform(7, 9) * ppm)
        marks["staple"] = staple(image, corner, rng, ppm)
    return marks


def _fit_page(page: _Page, blocks: Sequence[Block], width: float, available: float) -> tuple[list[Item], float]:
    """Lay out the page, shrinking type down to MIN_PAGE_SCALE; the caller grows the page for the rest."""
    scale = 1.0
    while True:
        items = _layout_items(page, blocks, page.styles.scaled(scale), width)
        if sum(item.height for item in items) <= available or scale <= MIN_PAGE_SCALE:
            return items, scale
        scale = max(MIN_PAGE_SCALE, scale * 0.95)


def _layout_items(page: _Page, blocks: Sequence[Block], styles: KindStyles, width: float) -> list[Item]:
    items: list[Item] = []
    i = 0
    while i < len(blocks):
        if blocks[i].kind == "row":
            j = i
            while j < len(blocks) and blocks[j].kind == "row":
                j += 1
            items.append(_layout_table(page, blocks[i:j], styles, width))
            i = j
        else:
            items.append(place(page.ts, blocks[i], styles, width, min_scale=_min_scale(blocks[i])))
            i += 1
    return items


def _min_scale(block: Block) -> float:
    """Running text wraps at full size; titles and machine-readable lines shrink a little first."""
    return 0.85 if block.kind in ("title", "header", "mrz") else 1.0


def _layout_table(page: _Page, rows: Sequence[Block], styles: KindStyles, width: float) -> Table:
    """Size columns as automatic table layout does; shrink the type only when the longest words cannot fit."""
    direction = paragraph_direction(" ".join(b.text for b in rows))
    texts = [b.text.split(CELL_SEPARATOR) for b in rows]
    columns = max(len(cells) for cells in texts)
    texts = [cells + [""] * (columns - len(cells)) for cells in texts]
    scale = 1.0
    while True:
        body = styles.body.scaled(scale)
        head = TextStyle(body.size, body.color, bold=True)
        pad = body.size * 0.45
        styled = [
            [Styled.plain(t, head if r == 0 and page.shaded_header else body) for t in row]
            for r, row in enumerate(texts)
        ]
        extents = [_column_extent(page.ts, [row[c] for row in styled], pad) for c in range(columns)]
        natural, minimum = [n for n, _ in extents], [m for _, m in extents]
        if sum(minimum) <= width or scale <= MIN_TABLE_SCALE:
            break
        scale = max(MIN_TABLE_SCALE, scale * 0.95)
    col_widths = _column_widths(natural, minimum, width, stretch=page.full_width_tables)
    cells = tuple(
        tuple(tuple(page.ts.fit(styled[r][c], col_widths[c] - 2 * pad, min_scale=0.9)) for c in range(columns))
        for r in range(len(rows))
    )
    pitch = body.size * styles.spacing
    row_heights = tuple(max(len(lines) for lines in row) * pitch + pad for row in cells)
    return Table(tuple(rows), cells, tuple(col_widths), row_heights, pitch, pad, direction, page.shaded_header)


def _column_extent(ts: Typesetter, cells: Sequence[Styled], pad: float) -> tuple[float, float]:
    """Widths of a column's widest cell and of its widest space-free word, each with ``pad`` on both sides."""
    natural = minimum = 0.0
    for cell in cells:
        base = paragraph_direction(cell.text)
        natural = max(natural, ts.layout(cell, base).width)
        start = 0
        for word in cell.text.split(" "):
            minimum = max(minimum, ts.layout(cell.slice(start, start + len(word)), base).width)
            start += len(word) + 1
    return natural + 2 * pad, minimum + 2 * pad


def _column_widths(natural: Sequence[float], minimum: Sequence[float], width: float, *, stretch: bool) -> list[float]:
    """Natural widths if they fit ``width``; else minimums plus slack shared by natural excess; else scaled minimums."""
    if sum(natural) <= width:
        factor = width / sum(natural) if stretch else 1.0
        return [n * factor for n in natural]
    if sum(minimum) <= width:
        share = (width - sum(minimum)) / (sum(natural) - sum(minimum))
        return [m + (n - m) * share for n, m in zip(natural, minimum, strict=True)]
    return [m * width / sum(minimum) for m in minimum]


def _draw_items(
    page: _Page,
    draw: ImageDraw.ImageDraw,
    items: Sequence[Item],
    left: float,
    right: float,
    top: float,
    rule_after: int,
) -> list[Box]:
    boxes: list[Box] = []
    y = top
    for n, item in enumerate(items):
        if isinstance(item, Table):
            boxes += _draw_table(page, draw, item, left, right, y)
        else:
            boxes += draw_column(draw, [item], left, right, y)
        y += item.height
        if rule_after and n + 1 == rule_after:
            line_y = y + 1.2 * page.px_per_mm
            draw.line((left, line_y, right, line_y), fill=page.rule_color, width=max(1, round(0.35 * page.px_per_mm)))
            y += 3 * page.px_per_mm
    return boxes


def _draw_table(
    page: _Page, draw: ImageDraw.ImageDraw, table: Table, left: float, right: float, top: float
) -> list[Box]:
    x0 = right - table.width if table.direction == "rtl" else left
    edges = [x0]
    for w in table.col_widths if table.direction == "ltr" else reversed(table.col_widths):
        edges.append(edges[-1] + w)
    if table.direction == "rtl":
        edges.reverse()
    stroke = max(1, round(0.2 * page.px_per_mm))
    grid = (70, 70, 70)
    boxes: list[Box] = []
    y = top
    for r, row in enumerate(table.cells):
        height = table.row_heights[r]
        if r == 0 and table.shaded_header:
            draw.rectangle((min(edges), y, max(edges), y + height), fill=(232, 232, 232))
        for c, lines in enumerate(row):
            cell_left, cell_right = sorted((edges[c], edges[c + 1]))
            for k, line in enumerate(lines):
                x = line_left(line, cell_left + table.pad, cell_right - table.pad, "start")
                line_top = y + table.pad / 2 + k * table.pitch
                draw_text_line(draw, line, x, baseline(line_top, table.pitch, line.size))
                boxes.append((x, line_top, x + line.width, line_top + table.pitch))
        draw.line((min(edges), y, max(edges), y), fill=grid, width=stroke)
        y += height
    draw.line((min(edges), y, max(edges), y), fill=grid, width=stroke)
    for edge in edges:
        draw.line((edge, top, edge, y), fill=grid, width=stroke)
    return boxes
