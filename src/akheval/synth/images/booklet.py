"""An open passport booklet: the data page beside a facing page, the binding between them and the cover around both."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .furniture import guilloche, hsv, rounded_mask, shade
from .sheet import Face, Focus
from .text import Box

COVER_HUES = (0.62, 0.0, 0.33, 0.08)
# Share of booklets photographed with the facing page above the data page; the rest show it beside.
STACKED_P = 0.7
PAGE_EDGES = 3


@dataclass(frozen=True, slots=True)
class Booklet:
    """The open booklet as one face, and its focus: the data page with the cover edge on its three outer sides."""

    face: Face
    focus: Focus
    params: dict[str, Any]


def open_booklet(face: Face, rng: random.Random, ppm: float) -> Booklet:
    """Return the data page ``face`` inside an open booklet, with its text boxes moved to match."""
    width, height = face.image.size
    stacked = rng.random() < STACKED_P
    edge = rng.uniform(2.0, 3.5) * ppm
    pages = (width, 2 * height) if stacked else (2 * width, height)
    total = (round(pages[0] + 2 * edge), round(pages[1] + 2 * edge))
    cover = hsv(rng.choice(COVER_HUES), rng.uniform(0.4, 0.7), rng.uniform(0.14, 0.28))
    canvas = Image.new("RGB", total, cover)
    paper = _paper_tone(face)
    facing, pattern = _facing_page(width, height, paper, rng, ppm)
    first = (round(edge), round(edge))
    second = (round(edge), round(edge + height)) if stacked else (round(edge + width), round(edge))
    data_at, facing_at = (second, first) if stacked or rng.random() < 0.5 else (first, second)
    canvas.paste(facing, facing_at)
    canvas.paste(face.image, data_at)
    pixels = np.asarray(canvas, np.float32)
    _binding(pixels, data_at, facing_at, (width, height), stacked, rng, ppm)
    _page_edges(pixels, data_at, facing_at, (width, height), stacked, paper, ppm)
    image = Image.fromarray(np.clip(pixels + 0.5, 0, 255).astype(np.uint8), "RGB")
    boxes = [(b[0] + data_at[0], b[1] + data_at[1], b[2] + data_at[0], b[3] + data_at[1]) for b in face.boxes]
    binding = "right" if data_at == first else "top" if stacked else "left"
    params = {
        "layout": "stacked" if stacked else "side_by_side",
        "data_page": "first" if data_at == first else "second",
        "binding": binding,
        "cover": list(cover),
        "facing_pattern": pattern,
    }
    focus = Focus(_focus_box(data_at, (width, height), total, binding), binding)
    return Booklet(Face(image, boxes, rounded_mask(*total, 4 * ppm)), focus, params)


def _focus_box(at: tuple[int, int], size: tuple[int, int], total: tuple[int, int], binding: str) -> Box:
    """The data page grown to the booklet's outline on every side but the ``binding`` side."""
    x0, y0, x1, y1 = 0.0, 0.0, float(total[0]), float(total[1])
    if binding == "top":
        y0 = float(at[1])
    elif binding == "left":
        x0 = float(at[0])
    else:
        x1 = float(at[0] + size[0])
    return x0, y0, x1, y1


def _paper_tone(face: Face) -> tuple[int, int, int]:
    """The data page's colour near its top edge, lightened towards plain paper."""
    strip = np.asarray(face.image.convert("RGB"), np.float32)[: max(1, face.image.height // 20)]
    mean = strip.reshape(-1, 3).mean(axis=0)
    r, g, b = (round(0.6 * c + 0.4 * 245) for c in mean)
    return r, g, b


def _facing_page(
    width: int, height: int, paper: tuple[int, int, int], rng: random.Random, ppm: float
) -> tuple[Image.Image, dict[str, Any]]:
    """A page without text: plain paper under a faint security pattern."""
    canvas = np.empty((height, width, 3), np.uint8)
    canvas[:] = paper
    pattern = guilloche(canvas, rng, shade(paper, rng.uniform(0.82, 0.9)), ppm)
    return Image.fromarray(canvas), pattern


def _binding(
    pixels: np.ndarray,
    data_at: tuple[int, int],
    facing_at: tuple[int, int],
    size: tuple[int, int],
    stacked: bool,
    rng: random.Random,
    ppm: float,
) -> None:
    """Darken both pages towards the fold between them."""
    width, height = size
    reach = rng.uniform(5.0, 11.0) * ppm
    depth = rng.uniform(0.25, 0.45)
    axis = 1 if stacked else 0
    fold = max(data_at[axis], facing_at[axis])
    length = pixels.shape[0] if stacked else pixels.shape[1]
    distance = np.abs(np.arange(length, dtype=np.float32) + 0.5 - fold)
    falloff = 1 - depth * np.exp(-distance / (reach / 3)) - 0.25 * np.exp(-((distance / (0.4 * ppm)) ** 2))
    span = slice(data_at[1 - axis], data_at[1 - axis] + (width if stacked else height))
    if stacked:
        pixels[:, span] *= falloff[:, None, None]
    else:
        pixels[span, :] *= falloff[None, :, None]


def _page_edges(
    pixels: np.ndarray,
    data_at: tuple[int, int],
    facing_at: tuple[int, int],
    size: tuple[int, int],
    stacked: bool,
    paper: tuple[int, int, int],
    ppm: float,
) -> None:
    """Thin lines along each page's outer edge, where the stacked pages of the booklet show."""
    width, height = size
    line = np.array(shade(paper, 0.8), np.float32)
    step = max(1, round(0.35 * ppm))
    for x, y in (data_at, facing_at):
        for k in range(1, PAGE_EDGES + 1):
            offset = k * step
            if stacked:
                outer = y - offset if y < max(data_at[1], facing_at[1]) else y + height - 1 + offset
                if 0 <= outer < pixels.shape[0]:
                    pixels[outer, x : x + width] = line
            else:
                outer = x - offset if x < max(data_at[0], facing_at[0]) else x + width - 1 + offset
                if 0 <= outer < pixels.shape[1]:
                    pixels[y : y + height, outer] = line
