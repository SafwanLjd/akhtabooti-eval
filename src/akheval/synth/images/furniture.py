"""Visual furniture for synthetic documents: backgrounds, guilloche, emblem, photo silhouette, watermark.

Every element is invented and abstract. The emblem is plain geometry, the photo is a grey silhouette and
the watermark marks each card face and passport page as a specimen.
"""

from __future__ import annotations

import colorsys
import math
import random
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from .fonts import FontSet
from .shapes import text_mask
from .text import RGB, Box

WATERMARK_LATIN = "SPECIMEN"
WATERMARK_ARABIC = "عينة"
# Watermark opacity on the 0-255 scale: 0.12 to 0.18.
WATERMARK_ALPHA = (31, 46)
# Space kept between the watermark and a text line, as a share of the face width.
WATERMARK_PAD = 0.01


def hsv(hue: float, saturation: float, value: float) -> RGB:
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, saturation, value)
    return round(r * 255), round(g * 255), round(b * 255)


def pastel(rng: random.Random, hue: float | None = None) -> RGB:
    return hsv(rng.random() if hue is None else hue, rng.uniform(0.06, 0.2), rng.uniform(0.92, 0.98))


def accent(rng: random.Random, hue: float | None = None) -> RGB:
    return hsv(rng.random() if hue is None else hue, rng.uniform(0.45, 0.75), rng.uniform(0.3, 0.5))


def shade(color: RGB, factor: float) -> RGB:
    r, g, b = (min(255, max(0, round(c * factor))) for c in color)
    return r, g, b


def gradient(width: int, height: int, start: RGB, end: RGB, angle: float) -> npt.NDArray[np.uint8]:
    """Linear gradient from ``start`` to ``end`` along ``angle`` (radians)."""
    xs = np.arange(width, dtype=np.float32)[None, :] * math.cos(angle)
    ys = np.arange(height, dtype=np.float32)[:, None] * math.sin(angle)
    proj = xs + ys
    t = (proj - proj.min()) / max(float(np.ptp(proj)), 1.0)
    a = np.array(start, np.float32)
    b = np.array(end, np.float32)
    return np.asarray(a + t[..., None] * (b - a) + 0.5, dtype=np.float32).astype(np.uint8)


def guilloche(canvas: npt.NDArray[np.uint8], rng: random.Random, color: RGB, px_per_mm: float) -> dict[str, Any]:
    """Draw overlapping sine bands and one rosette over ``canvas`` in place."""
    height, width = canvas.shape[:2]
    overlay = canvas.copy()
    thickness = max(1, round(px_per_mm * 0.08))
    curves = rng.randint(14, 26)
    wavelength = rng.uniform(0.35, 0.8) * width
    amplitude = rng.uniform(0.06, 0.14) * height
    xs = np.arange(0, width + 4, 4, dtype=np.float32)
    phase_step = rng.uniform(0.15, 0.4)
    centre_y = rng.uniform(0.3, 0.7) * height
    for k in range(curves):
        offset = (k - curves / 2) * height * 0.035
        ys = (
            centre_y
            + offset
            + amplitude * np.sin(2 * np.pi * xs / wavelength + k * phase_step)
            + 0.4 * amplitude * np.sin(2 * np.pi * xs / (wavelength * 0.37) - k * phase_step * 1.7)
        )
        points = np.stack([xs, ys], axis=1).round().astype(np.int32)
        cv2.polylines(overlay, [points], False, color, thickness, cv2.LINE_AA)
    lobes = rng.randint(7, 16)
    radius = rng.uniform(0.18, 0.3) * height
    cx, cy = rng.uniform(0.2, 0.8) * width, rng.uniform(0.3, 0.7) * height
    theta = np.linspace(0, 2 * np.pi, 720, dtype=np.float32)
    for k in range(10):
        r = radius * (1 + 0.25 * np.sin(lobes * theta + k * 0.3))
        points = np.stack([cx + r * np.cos(theta), cy + r * np.sin(theta)], axis=1).round().astype(np.int32)
        cv2.polylines(overlay, [points], True, color, thickness, cv2.LINE_AA)
    strength = rng.uniform(0.35, 0.6)
    cv2.addWeighted(overlay, strength, canvas, 1 - strength, 0, dst=canvas)
    return {"curves": curves, "rosette_lobes": lobes, "strength": round(strength, 3)}


def emblem(draw: ImageDraw.ImageDraw, box: Box, rng: random.Random, primary: RGB, secondary: RGB) -> dict[str, Any]:
    """Abstract emblem: rings around a star polygon with optional rays."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    radius = min(x1 - x0, y1 - y0) / 2
    width = max(1, round(radius * 0.06))
    rings = rng.randint(1, 3)
    for k in range(rings):
        r = radius * (1 - 0.1 * k)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=primary, width=width)
    points = rng.randint(5, 12)
    inner = rng.uniform(0.38, 0.6)
    outer_r = radius * (0.85 - 0.1 * (rings - 1))
    rotation = rng.uniform(0, math.pi)
    star = []
    for i in range(points * 2):
        r = outer_r if i % 2 == 0 else outer_r * inner
        angle = rotation + i * math.pi / points
        star.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(star, fill=secondary, outline=primary)
    rays = rng.random() < 0.5
    if rays:
        for i in range(points):
            angle = rotation + (i + 0.5) * 2 * math.pi / points
            r0, r1 = outer_r * inner * 0.6, outer_r * 0.95
            draw.line(
                (
                    cx + r0 * math.cos(angle),
                    cy + r0 * math.sin(angle),
                    cx + r1 * math.cos(angle),
                    cy + r1 * math.sin(angle),
                ),
                fill=primary,
                width=width,
            )
    core = outer_r * inner * 0.45
    draw.ellipse((cx - core, cy - core, cx + core, cy + core), fill=primary)
    return {"rings": rings, "points": points, "inner": round(inner, 3), "rays": rays}


def silhouette(image: Image.Image, box: Box, rng: random.Random, mask: Image.Image | None = None) -> dict[str, Any]:
    """Paste a neutral grey head-and-shoulders placeholder for the holder photo into ``box``."""
    w, h = round(box[2] - box[0]), round(box[3] - box[1])
    backdrop = rng.randint(215, 238)
    figure = (rng.randint(120, 160),) * 3
    photo = Image.new("RGB", (w, h), (backdrop, backdrop, backdrop + 4))
    draw = ImageDraw.Draw(photo)
    draw.ellipse((0.08 * w, 0.66 * h, 0.92 * w, 1.45 * h), fill=figure)
    draw.rectangle((0.42 * w, 0.5 * h, 0.58 * w, 0.72 * h), fill=figure)
    draw.ellipse((0.29 * w, 0.14 * h, 0.71 * w, 0.62 * h), fill=figure)
    if mask is None:
        draw.rectangle((0, 0, w - 1, h - 1), outline=tuple(c - 40 for c in figure), width=max(1, round(w * 0.012)))
    image.paste(photo, (round(box[0]), round(box[1])), mask)
    return {"backdrop": backdrop, "figure": figure[0]}


def watermark(
    image: Image.Image, fonts: FontSet, rng: random.Random, boxes: Sequence[Box], prefer: Box | None
) -> tuple[Image.Image, dict[str, Any]]:
    """Return ``image`` with a faint SPECIMEN / عينة mark over ``prefer`` (the photo) or a blank area.

    The mark never touches a text line; when no blank area fits even a small mark, the face stays unmarked.
    """
    width, height = image.size
    color = hsv(rng.choice((0.0, 0.58, 0.02)), rng.uniform(0.5, 0.75), rng.uniform(0.45, 0.65))
    alpha = rng.randint(WATERMARK_ALPHA[0], WATERMARK_ALPHA[1])
    angle = rng.uniform(18, 32) * rng.choice((-1, 1))
    text = text_mask((width, height), boxes, width * WATERMARK_PAD).astype(bool)
    tries: list[tuple[str, float, Box]] = []
    if prefer is not None:
        tries += [("photo", share * (prefer[2] - prefer[0]), prefer) for share in (1.1, 0.9, 0.75)]
    tries += [("blank", share * width, (0.0, 0.0, float(width), float(height))) for share in (0.4, 0.32, 0.25)]
    for where, mark_width, region in tries:
        layer = _mark_layer(fonts, mark_width, (*color, alpha), angle)
        spot = _clear_spot(text, np.asarray(layer.getchannel("A")) > 0, region, rng)
        if spot is not None:
            marked = image.convert("RGBA")
            marked.alpha_composite(layer, spot)
            params = {"where": where, "angle": round(angle, 2), "alpha": alpha, "width_px": round(mark_width)}
            return marked.convert("RGB"), params
    return image, {"where": None}


def _mark_layer(fonts: FontSet, mark_width: float, fill: tuple[int, int, int, int], angle: float) -> Image.Image:
    """Both words, the Latin above the Arabic, sized so that the Latin word spans ``mark_width``, then turned."""
    probe = fonts.latin.bold.font(100)
    size = max(8, round(100 * mark_width / max(probe.getlength(WATERMARK_LATIN), 1.0)))
    latin, arabic = fonts.latin.bold.font(size), fonts.arabic.bold.font(round(size * 1.1))
    layer = Image.new("RGBA", (math.ceil(mark_width * 1.2), math.ceil(size * 2.8)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    mid_x, mid_y = layer.width / 2, layer.height / 2
    draw.text((mid_x, mid_y), WATERMARK_LATIN, font=latin, fill=fill, anchor="md", direction="ltr")
    draw.text((mid_x, mid_y), WATERMARK_ARABIC, font=arabic, fill=fill, anchor="ma", direction="rtl")
    turned = layer.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)
    return turned.crop(turned.getbbox() or (0, 0, 1, 1))


def _clear_spot(
    text: npt.NDArray[np.bool_], ink: npt.NDArray[np.bool_], region: Box, rng: random.Random
) -> tuple[int, int] | None:
    """A position for ``ink`` centred in ``region`` or, failing that, anywhere in it, that overlaps no text."""
    height, width = text.shape
    h, w = ink.shape
    x0, y0 = max(0, round(region[0])), max(0, round(region[1]))
    x1, y1 = min(width, round(region[2])) - w, min(height, round(region[3])) - h
    centre = (round((region[0] + region[2] - w) / 2), round((region[1] + region[3] - h) / 2))
    grid = (
        [(x, y) for y in np.linspace(y0, y1, 7).round() for x in np.linspace(x0, x1, 9).round()]
        if x1 >= x0 and y1 >= y0
        else []
    )
    rng.shuffle(grid)
    for x, y in [centre, *((int(a), int(b)) for a, b in grid)]:
        if 0 <= x <= width - w and 0 <= y <= height - h and not (text[y : y + h, x : x + w] & ink).any():
            return x, y
    return None


def rounded_mask(width: int, height: int, radius: float) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, fill=255)
    return mask
