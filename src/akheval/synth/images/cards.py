"""ID card (ID-1, 85.6 x 54 mm) and passport data page (ID-3, 125 x 88 mm) renderers."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw

from ..builder import FRONT, SIDES
from .flow import (
    Block,
    KindStyles,
    Placed,
    block_params,
    column_height,
    draw_column,
    fit_columns,
    fit_mono,
    leading_headings,
    spread,
)
from .fonts import SANS_FAMILIES, SERIF_FAMILIES, choose_fonts
from .furniture import accent, emblem, gradient, guilloche, pastel, rounded_mask, shade, silhouette, watermark
from .sheet import Face, Sheet
from .text import RGB, Box, TextStyle, Typesetter

CARD_MM = (85.6, 54.0)
CARD_CORNER_MM = 3.18
CARD_PX_PER_MM = (12.0, 16.0)
PASSPORT_MM = (125.0, 88.0)
PASSPORT_CORNER_MM = 2.5
PASSPORT_PX_PER_MM = (10.0, 13.0)

# Machine-readable zone: line pitch per font size, and the largest share of the face height it may take.
MRZ_SPACING = 1.18
MRZ_SHARE = 0.3
# Largest extra gap between blocks when a face has room to spare, as a multiple of the body font size.
MAX_SPREAD = 0.9
# Padding above and below the MRZ lines, as a share of the margin.
MRZ_PAD = 0.6
HEADER_BAND_P = 0.5
# Space between the passport photo and the field columns, and between two field columns.
GUTTER_MM = 3.0

# Font sizes in millimetres: title, header, label, value, body, monospace.
CARD_TYPE_MM = (3.3, 2.5, 2.0, 2.7, 2.4, 2.8)
PASSPORT_TYPE_MM = (4.0, 2.9, 2.2, 3.0, 2.8, 3.4)


@dataclass(frozen=True, slots=True)
class Palette:
    background: tuple[RGB, RGB]
    lines: RGB
    accent: RGB
    ink: RGB
    label: RGB

    def to_params(self) -> dict[str, Any]:
        return {"background": [list(c) for c in self.background], "accent": list(self.accent), "ink": list(self.ink)}


def _palette(rng: random.Random) -> Palette:
    hue = rng.random()
    first = pastel(rng, hue)
    second = pastel(rng, hue + rng.uniform(-0.12, 0.12))
    ink_level = rng.randint(15, 45)
    strong = accent(rng, hue + rng.uniform(-0.05, 0.05))
    return Palette(
        background=(first, second),
        lines=shade(pastel(rng, hue + 0.5), 0.85),
        accent=strong,
        ink=(ink_level, ink_level, ink_level + 10),
        label=strong if rng.random() < 0.6 else (90, 90, 95),
    )


def _styles(rng: random.Random, sizes_mm: Sequence[float], px_per_mm: float, palette: Palette) -> KindStyles:
    jitter = rng.uniform(0.92, 1.08)
    title, header, label, value, body, mono = (round(mm * px_per_mm * jitter) for mm in sizes_mm)
    return KindStyles(
        title=TextStyle(title, palette.accent, bold=True),
        header=TextStyle(header, palette.accent, bold=rng.random() < 0.7),
        label=TextStyle(label, palette.label),
        value=TextStyle(value, palette.ink, bold=rng.random() < 0.5),
        body=TextStyle(body, palette.ink),
        mono=TextStyle(mono, (20, 20, 20), mono=True),
        spacing=rng.uniform(1.22, 1.38),
    )


@dataclass
class _Context:
    rng: random.Random
    ts: Typesetter
    styles: KindStyles
    palette: Palette
    px_per_mm: float


def _context(rng: random.Random, px_range: tuple[float, float], sizes_mm: Sequence[float]) -> _Context:
    px_per_mm = rng.uniform(*px_range)
    fonts = choose_fonts(rng, SANS_FAMILIES if rng.random() < 0.8 else SERIF_FAMILIES)
    palette = _palette(rng)
    return _Context(rng, Typesetter(fonts), _styles(rng, sizes_mm, px_per_mm, palette), palette, px_per_mm)


def _background(ctx: _Context, width: int, height: int) -> tuple[Image.Image, dict[str, Any]]:
    canvas = gradient(width, height, *ctx.palette.background, ctx.rng.uniform(0, math.pi))
    pattern = guilloche(canvas, ctx.rng, ctx.palette.lines, ctx.px_per_mm)
    return Image.fromarray(canvas), {"guilloche": pattern}


def _mrz_layout(ctx: _Context, blocks: Sequence[Block], width: int, height: int, margin: float) -> list[Placed]:
    """Size machine-readable-zone lines so they span the face between the margins."""
    if not blocks:
        return []
    return fit_mono(ctx.ts, blocks, ctx.styles.mono.color, width - 2 * margin, height * MRZ_SHARE, MRZ_SPACING)


def _draw_mrz(draw: ImageDraw.ImageDraw, placed: Sequence[Placed], width: int, top: float, margin: float) -> list[Box]:
    """Draw MRZ lines on a white band that starts at ``top``."""
    pad = margin * MRZ_PAD
    draw.rectangle((0, top, width, top + column_height(placed) + 2 * pad), fill=(250, 250, 248))
    return draw_column(draw, placed, margin, width - margin, top + pad)


def _mrz_top(placed: Sequence[Placed], bottom: float, margin: float) -> float:
    return bottom - column_height(placed) - 2 * margin * MRZ_PAD


def render_id_card(blocks: Sequence[Block], rng: random.Random) -> Sheet:
    """Draw the card faces the layout has blocks for: the front, the back, or both."""
    ctx = _context(rng, CARD_PX_PER_MM, CARD_TYPE_MM)
    photo_left = rng.random() < 0.5
    faces: list[Face] = []
    params: dict[str, Any] = {
        "px_per_mm": round(ctx.px_per_mm, 3),
        "fonts": ctx.ts.fonts.to_params(),
        "styles": ctx.styles.to_params(),
        "palette": ctx.palette.to_params(),
        "photo_side": "left" if photo_left else "right",
        "faces": [],
    }
    sides = [side for side in SIDES if any(b.side == side for b in blocks)]
    if not sides:
        raise ValueError("an ID card layout needs at least one block")
    for side in sides:
        side_blocks = [b for b in blocks if b.side == side]
        face, face_params = _card_face(ctx, side_blocks, front=side == FRONT, photo_left=photo_left)
        faces.append(face)
        params["faces"].append({"side": side, **face_params})
    return Sheet("id_card", faces, ctx.px_per_mm, ctx.styles.label.size / ctx.px_per_mm, params)


def _card_face(ctx: _Context, blocks: Sequence[Block], *, front: bool, photo_left: bool) -> tuple[Face, dict[str, Any]]:
    ppm = ctx.px_per_mm
    width, height = round(CARD_MM[0] * ppm), round(CARD_MM[1] * ppm)
    margin = 3.0 * ppm
    text_blocks = [b for b in blocks if b.kind != "mrz"]
    mrz = _mrz_layout(ctx, [b for b in blocks if b.kind == "mrz"], width, height, margin)
    mrz_top = _mrz_top(mrz, height - margin * 0.8, margin) if mrz else float(height)
    bottom = mrz_top - margin * 0.4 if mrz else height - margin
    column = ctx.rng.uniform(21.0, 24.0) * ppm if front else 0.0
    col_left = margin if photo_left else width - margin - column
    left, right = margin, width - margin
    if front and photo_left:
        left = col_left + column + 2.5 * ppm
    elif front:
        right = col_left - 2.5 * ppm
    columns, scale = fit_columns(
        ctx.ts, [text_blocks], ctx.styles, right - left, bottom - margin, min_scale=0.5, block_min_scale=0.85
    )
    gap = spread(columns[0], bottom - margin, ctx.styles.body.size * MAX_SPREAD)
    image, params = _background(ctx, width, height)
    draw = ImageDraw.Draw(image)
    heads = columns[0][: leading_headings(text_blocks)]
    params["header_band"] = bool(heads) and ctx.rng.random() < HEADER_BAND_P
    if params["header_band"]:
        band = margin + column_height(heads) + gap * (len(heads) - 0.5)
        draw.rectangle((0, 0, width, band), fill=shade(ctx.palette.background[1], 0.93))
        draw.line((0, band, width, band), fill=ctx.palette.accent, width=max(1, round(0.25 * ppm)))
    photo: Box | None = None
    if front:
        mark = column * 0.5
        params["emblem"] = emblem(
            draw,
            (col_left + (column - mark) / 2, margin, col_left + (column + mark) / 2, margin + mark),
            ctx.rng,
            ctx.palette.accent,
            shade(ctx.palette.background[0], 0.9),
        )
        photo_top = margin + mark + 1.5 * ppm
        photo_h = min(column * 1.28, height - photo_top - margin)
        photo = (col_left, photo_top, col_left + column, photo_top + photo_h)
        params["photo"] = silhouette(image, photo, ctx.rng)
    boxes = _draw_mrz(draw, mrz, width, mrz_top, margin) if mrz else []
    boxes += draw_column(draw, columns[0], left, right, margin, gap=gap)
    image, params["watermark"] = watermark(image, ctx.ts.fonts, ctx.rng, boxes, photo)
    params.update(scale=round(scale, 3), blocks=block_params(mrz + columns[0]))
    return Face(image, boxes, rounded_mask(width, height, CARD_CORNER_MM * ppm)), params


def render_passport(blocks: Sequence[Block], rng: random.Random) -> Sheet:
    if any(b.side != FRONT for b in blocks):
        raise ValueError("passport layouts have one side; remove 'side': 'back'")
    ctx = _context(rng, PASSPORT_PX_PER_MM, PASSPORT_TYPE_MM)
    ppm = ctx.px_per_mm
    width, height = round(PASSPORT_MM[0] * ppm), round(PASSPORT_MM[1] * ppm)
    margin = 4.0 * ppm
    image, params = _background(ctx, width, height)
    params.update(
        px_per_mm=round(ppm, 3),
        fonts=ctx.ts.fonts.to_params(),
        styles=ctx.styles.to_params(),
        palette=ctx.palette.to_params(),
    )
    draw = ImageDraw.Draw(image)
    mark = ctx.rng.uniform(11.0, 14.0) * ppm
    mark_left = ctx.rng.random() < 0.5
    mark_x = margin if mark_left else width - margin - mark
    params["emblem"] = emblem(
        draw, (mark_x, margin, mark_x + mark, margin + mark), ctx.rng, ctx.palette.accent, ctx.palette.background[1]
    )
    text = [b for b in blocks if b.kind != "mrz"]
    mrz = [b for b in blocks if b.kind == "mrz"]
    heads, body = text[: leading_headings(text)], text[leading_headings(text) :]
    head_left = margin + (mark + 2 * ppm if mark_left else 0)
    head_right = width - margin - (0 if mark_left else mark + 2 * ppm)
    head_cols, _ = fit_columns(
        ctx.ts, [heads], ctx.styles, head_right - head_left, height * 0.3, min_scale=0.6, block_min_scale=0.8
    )
    body_top = margin + max(mark, sum(p.height for p in head_cols[0])) + 2 * ppm
    photo_w = ctx.rng.uniform(29.0, 33.0) * ppm
    photo = (margin, body_top, margin + photo_w, body_top + photo_w * 1.28)
    params["photo"] = silhouette(image, photo, ctx.rng)
    placed = _mrz_layout(ctx, mrz, width, height, margin)
    mrz_top = _mrz_top(placed, height - margin * 0.5, margin) if mrz else float(height)
    bottom = mrz_top - ppm if mrz else height - margin
    boxes = _draw_mrz(draw, placed, width, mrz_top, margin) if mrz else []
    boxes += draw_column(draw, head_cols[0], head_left, head_right, margin)
    gutter = GUTTER_MM * ppm
    body_left = margin + photo_w + gutter
    body_cols, scale, columns = _passport_body(ctx, body, width - margin - body_left, bottom - body_top)
    col_w = (width - margin - body_left - gutter * (columns - 1)) / columns
    gap = min(spread(column, bottom - body_top, ctx.styles.body.size * MAX_SPREAD) for column in body_cols)
    for i, column in enumerate(body_cols):
        x = body_left + i * (col_w + gutter)
        boxes += draw_column(draw, column, x, x + col_w, body_top, gap=gap)
    image, params["watermark"] = watermark(image, ctx.ts.fonts, ctx.rng, boxes, photo)
    params.update(
        scale=round(scale, 3),
        body_columns=columns,
        emblem_side="left" if mark_left else "right",
        blocks=block_params(placed + head_cols[0] + [p for c in body_cols for p in c]),
    )
    face = Face(image, boxes, rounded_mask(width, height, PASSPORT_CORNER_MM * ppm))
    return Sheet("passport", [face], ppm, ctx.styles.label.size / ppm, params)


def _passport_body(
    ctx: _Context, body: Sequence[Block], width: float, height: float
) -> tuple[list[list[Placed]], float, int]:
    """Fit the field grid in one column, or in two when one column would need small type."""
    try:
        cols, scale = fit_columns(ctx.ts, [body], ctx.styles, width, height, min_scale=0.85, block_min_scale=0.9)
        return cols, scale, 1
    except ValueError:
        half = math.ceil(len(body) / 2)
        col_w = (width - GUTTER_MM * ctx.px_per_mm) / 2
        groups = [body[:half], body[half:]]
        cols, scale = fit_columns(ctx.ts, groups, ctx.styles, col_w, height, min_scale=0.5, block_min_scale=0.85)
        return cols, scale, 2
