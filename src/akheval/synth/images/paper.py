"""Marks on paper: fold creases, an ink stamp, a signature scribble and staple marks. None of them adds text."""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .backgrounds import F32, value_noise
from .text import Box

INK_COLOURS = {"blue": (30, 55, 150), "purple": (92, 44, 128)}
SIGNATURE_INKS = {"blue": (22, 34, 105), "black": (28, 28, 32)}
# Space kept between a mark and any text line.
TEXT_PAD_MM = 1.5
# Resolution of the free-space search, as a share of the page's pixels.
SEARCH_SCALE = 0.25
SEARCH_STEP_MM = 3.0


def crease(image: F32, start: tuple[float, float], end: tuple[float, float], ppm: float, strength: float) -> None:
    """A fold from ``start`` to ``end`` across ``image``: a pale ridge, a shaded side and a lit side."""
    height, width = image.shape[:2]
    reach = 4.0 * ppm
    x0 = max(0, math.floor(min(start[0], end[0]) - reach))
    x1 = min(width, math.ceil(max(start[0], end[0]) + reach))
    y0 = max(0, math.floor(min(start[1], end[1]) - reach))
    y1 = min(height, math.ceil(max(start[1], end[1]) + reach))
    if x1 <= x0 or y1 <= y0:
        return
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    distance = ((xs - start[0]) * np.float32(-dy) + (ys - start[1]) * np.float32(dx)) / np.float32(length)
    ridge = np.exp(-((distance / (0.2 * ppm)) ** 2))
    shaded = np.exp(-np.clip(distance, 0, None) / (1.2 * ppm)) * (distance > 0)
    lit = np.exp(-np.clip(-distance, 0, None) / (0.8 * ppm)) * (distance <= 0)
    factor = 1 + np.float32(strength) * (0.35 * ridge + 0.25 * lit - 0.9 * shaded)
    image[y0:y1, x0:x1] *= factor[..., None]


def fold(image: F32, rng: random.Random, ppm: float, pattern: str, strength: float) -> dict[str, Any]:
    """Fold a page in halves, thirds or quarters; each panel catches the light a little differently."""
    height, width = image.shape[:2]
    rows = {"half": (0.5,), "thirds": (1 / 3, 2 / 3), "quarters": (0.5,)}[pattern]
    columns = (0.5,) if pattern == "quarters" else ()
    lines = []
    for share in rows:
        y = share * height + rng.uniform(-0.01, 0.01) * height
        tilt = math.tan(math.radians(rng.uniform(-0.6, 0.6))) * width
        lines.append(((0.0, y - tilt / 2), (float(width), y + tilt / 2)))
    for share in columns:
        x = share * width + rng.uniform(-0.01, 0.01) * width
        lines.append(((x, 0.0), (x, float(height))))
    edges = [0, *(round((a[1] + b[1]) / 2) for a, b in lines[: len(rows)]), height]
    for top, bottom in itertools.pairwise(edges):
        image[top:bottom] *= np.float32(1 + rng.uniform(-0.035, 0.035))
    for a, b in lines:
        crease(image, a, b, ppm, strength if rng.random() < 0.5 else -strength * 0.6)
    return {"pattern": pattern, "strength": round(strength, 3), "lines": len(lines)}


def free_spot(
    boxes: Sequence[Box],
    size: tuple[float, float],
    region: Box,
    near: tuple[float, float],
    rng: random.Random,
    ppm: float,
) -> tuple[float, float] | None:
    """Top-left corner of a ``size`` rectangle inside ``region`` that touches no text, close to ``near``."""
    scale = SEARCH_SCALE
    width, height = math.ceil(region[2] * scale) + 1, math.ceil(region[3] * scale) + 1
    text = np.zeros((height, width), np.uint8)
    pad = TEXT_PAD_MM * ppm
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(text, (round((x0 - pad) * scale), round((y0 - pad) * scale)),
                      (round((x1 + pad) * scale), round((y1 + pad) * scale)), 1, -1)  # fmt: skip
    integral = cv2.integral(text)
    step = SEARCH_STEP_MM * ppm
    candidates = []
    for y in np.arange(region[1], region[3] - size[1], step):
        for x in np.arange(region[0], region[2] - size[0], step):
            a, b = round(x * scale), round(y * scale)
            c, d = round((x + size[0]) * scale), round((y + size[1]) * scale)
            if integral[d, c] - integral[b, c] - integral[d, a] + integral[b, a] == 0:
                distance = math.hypot(x + size[0] / 2 - near[0], y + size[1] / 2 - near[1])
                candidates.append((distance, float(x), float(y)))
    if not candidates:
        return None
    candidates.sort()
    _, x, y = rng.choice(candidates[:5])
    return x, y


def stamp(page: Image.Image, centre: tuple[float, float], radius: float, rng: random.Random) -> dict[str, Any]:
    """A round ink stamp: two rings with dots and dashes between them and an abstract motif, pressed unevenly."""
    size = math.ceil(radius * 2 + 4)
    coverage = np.zeros((size, size), np.float32)
    c = size / 2
    stroke = max(1, round(radius * 0.05))
    cv2.circle(coverage, (round(c), round(c)), round(radius), 1.0, stroke, cv2.LINE_AA)
    cv2.circle(coverage, (round(c), round(c)), round(radius * 0.7), 1.0, max(1, stroke // 2), cv2.LINE_AA)
    dots = rng.randint(24, 40)
    for k in range(dots):
        angle = 2 * math.pi * k / dots
        r = radius * 0.85
        point = (round(c + r * math.cos(angle)), round(c + r * math.sin(angle)))
        if k % 4 == 0:
            inner = (round(c + radius * 0.77 * math.cos(angle)), round(c + radius * 0.77 * math.sin(angle)))
            outer = (round(c + radius * 0.93 * math.cos(angle)), round(c + radius * 0.93 * math.sin(angle)))
            cv2.line(coverage, inner, outer, 1.0, max(1, stroke // 2), cv2.LINE_AA)
        else:
            cv2.circle(coverage, point, max(1, round(radius * 0.035)), 1.0, -1, cv2.LINE_AA)
    points = rng.randint(5, 8)
    star = []
    for i in range(points * 2):
        r = radius * (0.45 if i % 2 == 0 else 0.2)
        angle = i * math.pi / points
        star.append((c + r * math.cos(angle), c + r * math.sin(angle)))
    cv2.polylines(coverage, [np.round(star).astype(np.int32)], True, 1.0, max(1, stroke // 2), cv2.LINE_AA)
    cv2.circle(coverage, (round(c), round(c)), max(1, round(radius * 0.08)), 1.0, -1, cv2.LINE_AA)
    angle = rng.uniform(0, 360)
    coverage = cv2.warpAffine(coverage, cv2.getRotationMatrix2D((c, c), angle, 1.0), (size, size))
    gen = np.random.default_rng(rng.getrandbits(64))
    pressure = np.clip(0.55 + 0.7 * value_noise(gen, size, size, 3) - 0.25 * value_noise(gen, size, size, 40), 0, 1)
    strength = rng.uniform(0.55, 0.85)
    ink_name = rng.choice(sorted(INK_COLOURS))
    _press(
        page,
        np.asarray(coverage * pressure * strength, np.float32),
        (centre[0] - c, centre[1] - c),
        INK_COLOURS[ink_name],
    )
    return {
        "ink": ink_name,
        "centre_px": [round(centre[0], 1), round(centre[1], 1)],
        "radius_px": round(radius, 1),
        "angle_deg": round(angle, 1),
        "strength": round(strength, 3),
    }


def signature(page: Image.Image, box: Box, rng: random.Random, ppm: float) -> dict[str, Any]:
    """A handwritten-looking scribble inside ``box``: looped strokes with changing pressure and an underline."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    t = np.linspace(0, 1, 500)
    loops = rng.uniform(4, 8)
    lean = rng.uniform(0.1, 0.35)
    amplitude = rng.uniform(0.25, 0.4) * h
    xs = x0 + w * 0.05 + w * 0.8 * t + amplitude * 0.6 * np.cos(2 * math.pi * loops * t)
    envelope = 0.6 + 0.4 * np.sin(math.pi * t * rng.uniform(1.0, 2.0) + rng.uniform(0, math.pi))
    ys = y0 + h * 0.45 - amplitude * envelope * np.sin(2 * math.pi * loops * t + rng.uniform(0, 1))
    xs = xs + lean * (y0 + h * 0.45 - ys)
    sweep = np.linspace(0, 1, 120)
    under_x = x0 + w * (0.95 - 0.9 * sweep)
    under_y = y0 + h * (0.82 + 0.1 * np.sin(math.pi * sweep)) + rng.uniform(-0.05, 0.05) * h
    width = max(1.0, rng.uniform(0.35, 0.6) * ppm)
    coverage = np.zeros((math.ceil(h) + 8, math.ceil(w) + 8), np.float32)
    for line_x, line_y in ((xs, ys), (under_x, under_y)):
        points = np.stack([line_x - x0 + 4, line_y - y0 + 4], axis=1)
        pressure = 0.6 + 0.6 * np.abs(np.sin(np.linspace(0, rng.uniform(4, 9), len(points))))
        for k in range(len(points) - 1):
            a, b = np.round(points[k]).astype(int), np.round(points[k + 1]).astype(int)
            thickness = max(1, round(width * pressure[k]))
            cv2.line(coverage, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 1.0, thickness, cv2.LINE_AA)
    ink_name = rng.choice(sorted(SIGNATURE_INKS))
    _press(
        page, np.clip(coverage, 0, 1) * np.float32(rng.uniform(0.8, 0.95)), (x0 - 4, y0 - 4), SIGNATURE_INKS[ink_name]
    )
    return {"ink": ink_name, "loops": round(loops, 2), "box_px": [round(v, 1) for v in box]}


def staple(page: Image.Image, corner: tuple[float, float], rng: random.Random, ppm: float) -> dict[str, Any]:
    """A staple, or the two holes one left, near ``corner``, turned diagonally across it."""
    angle = math.radians(rng.uniform(30, 60))
    half = rng.uniform(5.5, 6.5) * ppm
    reach = half + 3 * ppm
    x0, y0 = max(0, round(corner[0] - reach)), max(0, round(corner[1] - reach))
    x1, y1 = min(page.width, round(corner[0] + reach)), min(page.height, round(corner[1] + reach))
    pixels = np.asarray(page.crop((x0, y0, x1, y1)), np.float32)
    dx, dy = half * math.cos(angle), half * math.sin(angle)
    a = (corner[0] - x0 - dx, corner[1] - y0 + dy)
    b = (corner[0] - x0 + dx, corner[1] - y0 - dy)
    metal = rng.random() < 0.6
    width = max(1, round(0.7 * ppm))
    if metal:
        shadow = np.zeros(pixels.shape[:2], np.float32)
        offset = 0.5 * ppm
        cv2.line(shadow, _pt(a, offset), _pt(b, offset), 1.0, width + 1, cv2.LINE_AA)
        pixels *= 1 - 0.35 * cv2.GaussianBlur(shadow, (0, 0), max(1.0, 0.4 * ppm))[..., None]
        cv2.line(pixels, _pt(a), _pt(b), (120.0, 122.0, 128.0), width, cv2.LINE_AA)
        cv2.line(pixels, _pt(a), _pt(b), (205.0, 207.0, 212.0), max(1, width // 3), cv2.LINE_AA)
    else:
        for end in (a, b):
            cv2.circle(pixels, _pt(end), max(1, round(0.35 * ppm)), (70.0, 70.0, 70.0), -1, cv2.LINE_AA)
    page.paste(Image.fromarray(np.clip(pixels + 0.5, 0, 255).astype(np.uint8), "RGB"), (x0, y0))
    return {"kind": "staple" if metal else "holes", "angle_deg": round(math.degrees(angle), 1)}


def _pt(point: tuple[float, float], offset: float = 0.0) -> tuple[int, int]:
    return round(point[0] + offset), round(point[1] + offset)


def _press(page: Image.Image, coverage: F32, origin: tuple[float, float], ink: tuple[int, int, int]) -> None:
    """Multiply ink of colour ``ink`` into ``page`` with ``coverage`` placed at ``origin``."""
    left, top = round(origin[0]), round(origin[1])
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(page.width, left + coverage.shape[1]), min(page.height, top + coverage.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    region = np.asarray(page.crop((x0, y0, x1, y1)), np.float32)
    cover = coverage[y0 - top : y1 - top, x0 - left : x1 - left, None]
    tint = np.array(ink, np.float32) * np.float32(1 / 255)
    region *= 1 - cover * (1 - tint)
    page.paste(Image.fromarray(np.clip(region + 0.5, 0, 255).astype(np.uint8), "RGB"), (x0, y0))
