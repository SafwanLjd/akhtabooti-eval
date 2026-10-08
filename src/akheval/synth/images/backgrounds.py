"""Procedural surfaces a document lies on in a phone photo: wood, desk laminate, fabric, stone, a desk under sheets.

Colours come from muted real-world palettes. Every surface is float32 RGB in [0, 1].
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]
RGBf = tuple[float, float, float]
Surface = Callable[[np.random.Generator, int, int], tuple[F32, dict[str, Any]]]

# Light and dark tone of each wood's grain.
WOODS: dict[str, tuple[RGBf, RGBf]] = {
    "oak": ((0.70, 0.56, 0.40), (0.50, 0.36, 0.23)),
    "walnut": ((0.46, 0.33, 0.24), (0.27, 0.18, 0.12)),
    "pine": ((0.80, 0.67, 0.50), (0.64, 0.48, 0.31)),
    "teak": ((0.60, 0.43, 0.29), (0.42, 0.28, 0.17)),
    "grey ash": ((0.66, 0.63, 0.58), (0.47, 0.44, 0.40)),
    "dark stain": ((0.34, 0.24, 0.17), (0.19, 0.13, 0.09)),
}
LAMINATES: dict[str, RGBf] = {
    "white": (0.86, 0.86, 0.84),
    "light grey": (0.73, 0.73, 0.72),
    "beige": (0.79, 0.74, 0.65),
    "graphite": (0.27, 0.27, 0.28),
    "sage": (0.62, 0.66, 0.60),
}
FABRICS: dict[str, RGBf] = {
    "white sheet": (0.88, 0.88, 0.86),
    "cream": (0.85, 0.81, 0.71),
    "pale blue": (0.67, 0.73, 0.79),
    "grey": (0.60, 0.60, 0.61),
    "sage": (0.60, 0.66, 0.57),
    "dusty rose": (0.76, 0.63, 0.61),
    "navy": (0.22, 0.26, 0.35),
    "charcoal": (0.29, 0.29, 0.31),
}
STONES: dict[str, RGBf] = {
    "white marble": (0.87, 0.86, 0.84),
    "grey marble": (0.72, 0.72, 0.71),
    "beige travertine": (0.82, 0.77, 0.69),
    "black granite": (0.20, 0.20, 0.21),
    "grey granite": (0.56, 0.55, 0.54),
}


def value_noise(gen: np.random.Generator, height: int, width: int, cells: float) -> F32:
    """Smooth noise in [0, 1] with about ``cells`` features across the longer side."""
    scale = max(height, width) / max(cells, 1.0)
    grid = gen.random((max(2, math.ceil(height / scale) + 1), max(2, math.ceil(width / scale) + 1)), np.float32)
    return np.asarray(cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC), np.float32).clip(0, 1)


def fractal_noise(gen: np.random.Generator, height: int, width: int, cells: float, octaves: int = 4) -> F32:
    """Value noise summed over ``octaves``, each with twice the features and half the weight of the last."""
    total = np.zeros((height, width), np.float32)
    for k in range(octaves):
        total += value_noise(gen, height, width, cells * 2**k) * np.float32(0.5**k)
    return total / np.float32(2 - 0.5 ** (octaves - 1))


def stretched_noise(gen: np.random.Generator, height: int, width: int, angle: float, cells_u: int, cells_v: int) -> F32:
    """Noise with ``cells_u`` features along direction ``angle`` and ``cells_v`` across it."""
    side = math.ceil(math.hypot(height, width)) + 2
    grid = gen.random((cells_v + 3, cells_u + 3), np.float32)
    canvas = cv2.resize(grid, (side, side), interpolation=cv2.INTER_CUBIC)
    matrix = cv2.getRotationMatrix2D((side / 2, side / 2), -math.degrees(angle), 1.0)
    matrix[:, 2] += (width - side) / 2, (height - side) / 2
    turned = cv2.warpAffine(canvas, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return np.asarray(turned, np.float32)


def _pick(gen: np.random.Generator, options: dict[str, Any]) -> str:
    names = sorted(options)
    return names[int(gen.integers(len(names)))]


def _jitter(gen: np.random.Generator, color: RGBf, amount: float) -> F32:
    return np.clip(np.array(color, np.float32) * np.float32(1 + gen.uniform(-amount, amount)), 0, 1)


def _grain(gen: np.random.Generator, height: int, width: int, amount: float) -> F32:
    return gen.standard_normal((height, width, 1), dtype=np.float32) * np.float32(amount)


def _grain_angle(gen: np.random.Generator) -> float:
    """A direction near the horizontal or the vertical, in radians."""
    return (0.0 if gen.random() < 0.5 else math.pi / 2) + float(gen.uniform(-0.2, 0.2))


def _grain_axes(gen: np.random.Generator, height: int, width: int) -> tuple[F32, F32, float]:
    """Coordinates along and across a grain direction from ``_grain_angle``, scaled to [0, 1]."""
    angle = _grain_angle(gen)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    span = np.float32(math.hypot(height, width))
    u = (xs * np.float32(math.cos(angle)) + ys * np.float32(math.sin(angle))) / span
    v = (-xs * np.float32(math.sin(angle)) + ys * np.float32(math.cos(angle))) / span
    return u - u.min(), v - v.min(), angle


def wood(gen: np.random.Generator, height: int, width: int) -> tuple[F32, dict[str, Any]]:
    """Wood with ring figure, fibre streaks, knots and, on some boards, plank seams."""
    name = _pick(gen, WOODS)
    light, dark = (_jitter(gen, tone, 0.06) for tone in WOODS[name])
    u, v, angle = _grain_axes(gen, height, width)
    rings = float(gen.uniform(14, 34))
    warp = stretched_noise(gen, height, width, angle, 3, 7)
    figure = v * np.float32(rings) + warp * np.float32(gen.uniform(1.5, 3.5))
    knots = int(gen.integers(0, 3))
    for _ in range(knots):
        ku, kv = float(gen.uniform(0.1, 0.9)) * float(np.max(u)), float(gen.uniform(0.1, 0.9)) * float(np.max(v))
        size = float(gen.uniform(0.01, 0.025))
        dist2 = ((u - ku) / np.float32(size * 3)) ** 2 + ((v - kv) / np.float32(size)) ** 2
        figure += np.float32(gen.uniform(1.5, 3.0)) * np.exp(-dist2)
        figure -= np.float32(0.25) * np.exp(-dist2 * 9)
    phase = figure - np.floor(figure)
    band = np.exp(-(((phase - 0.5) / 0.16) ** 2))
    fibres = stretched_noise(gen, height, width, angle, 4, max(8, int(max(height, width) / 2.5)))
    mix = np.clip(0.5 * band + 0.3 * fractal_noise(gen, height, width, 3, 2) + 0.25 * (fibres - 0.5), 0, 1)
    surface = light + (dark - light) * mix[..., None]
    planks = gen.random() < 0.5
    if planks:
        width_v = float(gen.uniform(0.12, 0.25))
        index = np.floor(v / np.float32(width_v))
        tones = gen.uniform(0.92, 1.06, int(index.max()) + 2).astype(np.float32)
        surface *= tones[index.astype(np.int32)][..., None]
        seam = np.abs(v / np.float32(width_v) - np.round(v / np.float32(width_v))) * np.float32(width_v)
        surface *= (1 - 0.45 * np.exp(-((seam * np.float32(math.hypot(height, width)) / 1.2) ** 2)))[..., None]
    params = {"surface": "wood", "wood": name, "grain_deg": round(math.degrees(angle), 1), "knots": knots}
    return surface, params | {"planks": planks}


def laminate(gen: np.random.Generator, height: int, width: int) -> tuple[F32, dict[str, Any]]:
    """Office desk laminate: a flat tone with fine speckle, faint mottling and a soft window reflection."""
    name = _pick(gen, LAMINATES)
    base = _jitter(gen, LAMINATES[name], 0.05)
    mottle = fractal_noise(gen, height, width, 4, 3)
    speckle = value_noise(gen, height, width, max(height, width) / 1.5)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    angle = float(gen.uniform(0, math.pi))
    across = (xs * np.float32(math.cos(angle)) + ys * np.float32(math.sin(angle))) / np.float32(max(height, width))
    sheen = np.exp(-(((across - np.float32(gen.uniform(0.0, 1.0))) / np.float32(gen.uniform(0.15, 0.35))) ** 2))
    shade = 0.95 + 0.06 * mottle + 0.04 * (speckle - 0.5) + float(gen.uniform(0.02, 0.07)) * sheen
    return base * shade[..., None], {"surface": "laminate", "tone": name}


def fabric(gen: np.random.Generator, height: int, width: int) -> tuple[F32, dict[str, Any]]:
    """Bed sheet or tablecloth: long, soft folds lit from one side."""
    name = _pick(gen, FABRICS)
    base = _jitter(gen, FABRICS[name], 0.05)
    angle = _grain_angle(gen)
    folds = stretched_noise(gen, height, width, angle, 2, 6)
    creases = 1 - np.abs(2 * stretched_noise(gen, height, width, angle, 3, 5) - 1)
    relief = cv2.GaussianBlur(folds + creases**4 * np.float32(0.3), (0, 0), max(height, width) / 200)
    gy, gx = np.gradient(relief)
    light = float(gen.uniform(0, 2 * math.pi))
    slope = gx * np.float32(math.cos(light)) + gy * np.float32(math.sin(light))
    depth = float(gen.uniform(0.04, 0.09))
    shade = np.clip(1 + depth * slope / max(float(slope.std()), 1e-6), 0.75, 1.2)
    surface = base * shade[..., None]
    params = {"surface": "fabric", "fabric": name, "fold_deg": round(math.degrees(angle), 1)}
    return surface, params | {"fold_depth": round(depth, 3)}


def stone(gen: np.random.Generator, height: int, width: int) -> tuple[F32, dict[str, Any]]:
    """Kitchen or counter stone: marble with turbulent veins, or speckled granite."""
    name = _pick(gen, STONES)
    base = _jitter(gen, STONES[name], 0.04)
    clouds = fractal_noise(gen, height, width, 3, 4)
    if "granite" in name:
        grains = value_noise(gen, height, width, max(height, width) / 1.5)
        flecks = np.where(grains > 0.72, 1.35, np.where(grains < 0.25, 0.6, 1.0)).astype(np.float32)
        shade = cv2.GaussianBlur(flecks, (0, 0), 0.7) * (0.95 + 0.1 * clouds)
    else:
        u, v, _ = _grain_axes(gen, height, width)
        phase = v * np.float32(gen.uniform(3.0, 6.0)) + fractal_noise(gen, height, width, 2.5, 5) * np.float32(1.3)
        veins = (1 - np.abs(np.sin(np.float32(math.pi) * phase))) ** 14
        fine = (1 - np.abs(np.sin(np.float32(math.pi) * (phase * 3 + u * 5)))) ** 30
        sign = -1.0 if base.mean() > 0.5 else 1.0
        shade = 0.95 + 0.08 * clouds + sign * (float(gen.uniform(0.15, 0.28)) * veins + 0.06 * fine)
    return base * shade[..., None], {"surface": "stone", "stone": name}


def sheets(gen: np.random.Generator, height: int, width: int) -> tuple[F32, dict[str, Any]]:
    """A desk covered by loose blank sheets of paper, each with a soft shadow."""
    desk, desk_params = (wood if gen.random() < 0.5 else laminate)(gen, height, width)
    count = int(gen.integers(2, 5))
    long_side = max(height, width)
    for _ in range(count):
        sheet_w = float(gen.uniform(0.45, 0.8)) * long_side
        sheet_h = sheet_w * math.sqrt(2)
        centre = (float(gen.uniform(0, width)), float(gen.uniform(0, height)))
        angle = float(gen.uniform(-30, 30)) + (90.0 if gen.random() < 0.3 else 0.0)
        corners = cv2.boxPoints((centre, (sheet_w, sheet_h), angle))
        mask = np.zeros((height, width), np.float32)
        cv2.fillPoly(mask, [np.round(corners).astype(np.int32)], 1.0, cv2.LINE_AA)
        shadow = cv2.GaussianBlur(
            np.roll(mask, (round(long_side * 0.004), round(long_side * 0.003)), (0, 1)), (0, 0), 6
        )
        desk *= (1 - 0.3 * shadow)[..., None]
        tone = np.float32(gen.uniform(0.86, 0.95)) * np.array([1.0, 1.0, gen.uniform(0.96, 1.02)], np.float32)
        desk = desk * (1 - mask[..., None]) + tone * mask[..., None]
    return desk, {"surface": "sheets", "sheets": count, "desk": desk_params}


SURFACES: dict[str, Surface] = {"wood": wood, "laminate": laminate, "fabric": fabric, "stone": stone, "sheets": sheets}


def surface(name: str, gen: np.random.Generator, height: int, width: int) -> tuple[F32, dict[str, Any]]:
    """Render surface ``name`` as float32 RGB in [0, 1], drawn at half resolution and enlarged."""
    small, params = SURFACES[name](gen, max(2, height // 2), max(2, width // 2))
    small += _grain(gen, small.shape[0], small.shape[1], 0.008)
    image = cv2.resize(np.clip(small, 0.0, 1.0), (width, height), interpolation=cv2.INTER_LINEAR)
    return np.asarray(image, np.float32), params
