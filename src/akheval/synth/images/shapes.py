"""Shaded solids drawn into a float32 RGB frame: polygon masks, rounded relief, gloss and soft shadows."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from .backgrounds import F32
from .text import Box

Points = npt.NDArray[np.float64]
RGBf = tuple[float, float, float]

# Sub-pixel bits for cv2 polygon drawing.
SHIFT = 4
# Above this sigma in pixels, blurs run on a downscaled copy.
FAST_BLUR_SIGMA = 4.0


@dataclass(frozen=True, slots=True)
class Window:
    """A rectangle of the frame in pixels: columns ``x0:x1`` and rows ``y0:y1``."""

    x0: int
    y0: int
    x1: int
    y1: int

    @classmethod
    def around(cls, polygons: Sequence[Points], margin: float, shape: tuple[int, ...]) -> Window | None:
        """The bounds of ``polygons`` grown by ``margin``, clipped to a frame of ``shape``; None when outside it."""
        points = np.concatenate(polygons)
        low, high = np.min(points, axis=0), np.max(points, axis=0)
        x0, y0 = max(0, math.floor(float(low[0]) - margin)), max(0, math.floor(float(low[1]) - margin))
        x1, y1 = min(shape[1], math.ceil(float(high[0]) + margin)), min(shape[0], math.ceil(float(high[1]) + margin))
        return cls(x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

    @property
    def shape(self) -> tuple[int, int]:
        return self.y1 - self.y0, self.x1 - self.x0

    def view(self, image: F32) -> F32:
        return image[self.y0 : self.y1, self.x0 : self.x1]


def capsule(start: Points, end: Points, radius_start: float, radius_end: float, steps: int = 12) -> Points:
    """Outline of a tapered capsule from ``start`` to ``end`` with round ends."""
    axis = end - start
    angle = math.atan2(float(axis[1]), float(axis[0]))
    head = [angle - math.pi / 2 + math.pi * k / steps for k in range(steps + 1)]
    tail = [angle + math.pi / 2 + math.pi * k / steps for k in range(steps + 1)]
    points = [end + radius_end * np.array([math.cos(a), math.sin(a)]) for a in head]
    points += [start + radius_start * np.array([math.cos(a), math.sin(a)]) for a in tail]
    return np.array(points, np.float64)


def rotation(angle: float) -> Points:
    """Matrix that turns column vectors by ``angle`` radians."""
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])


def ellipse(centre: Points, radii: tuple[float, float], angle: float, steps: int = 48) -> Points:
    """Outline of an ellipse turned by ``angle`` radians."""
    t = np.linspace(0, 2 * math.pi, steps, endpoint=False)
    local = np.stack([radii[0] * np.cos(t), radii[1] * np.sin(t)], axis=1)
    turn = rotation(angle)
    return np.asarray(local @ turn.T + centre, np.float64)


def rectangle(centre: Points, size: tuple[float, float], angle: float, radius: float = 0.0) -> Points:
    """Outline of a rectangle with corners rounded by ``radius``, turned by ``angle`` radians."""
    w, h = size[0] / 2, size[1] / 2
    r = min(radius, w, h)
    points: list[tuple[float, float]] = []
    for cx, cy, start in ((w - r, h - r, 0.0), (-w + r, h - r, 90.0), (-w + r, -h + r, 180.0), (w - r, -h + r, 270.0)):
        for k in range(7 if r > 0 else 1):
            a = math.radians(start + 15 * k)
            points.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    turn = rotation(angle)
    return np.asarray(np.array(points) @ turn.T + centre, np.float64)


def mask(window: Window, polygons: Sequence[Points], holes: Sequence[Points] = ()) -> F32:
    """Anti-aliased coverage of ``polygons`` minus ``holes`` inside ``window``."""
    out = np.zeros(window.shape, np.float32)
    origin = np.array([window.x0, window.y0], np.float64)
    for value, group in ((1.0, polygons), (0.0, holes)):
        for polygon in group:
            points = np.round((polygon - origin) * (1 << SHIFT)).astype(np.int32)
            cv2.fillPoly(out, [points], value, cv2.LINE_AA, SHIFT)
    return out


def text_mask(size: tuple[int, int], boxes: Sequence[Box], pad: float, scale: float = 1.0) -> npt.NDArray[np.uint8]:
    """Boxes grown by ``pad`` and filled with 1 on a ``size`` frame shrunk by ``scale``."""
    out = np.zeros((max(1, round(size[1] * scale)), max(1, round(size[0] * scale))), np.uint8)
    for x0, y0, x1, y1 in boxes:
        corners = (np.array([[x0 - pad, y0 - pad], [x1 + pad, y1 + pad]]) * scale).round().astype(np.int32)
        cv2.rectangle(out, tuple(corners[0]), tuple(corners[1]), 1, -1)
    return out


def smooth(field: F32, sigma: float) -> F32:
    """Gaussian blur; a wide blur runs on a smaller copy, which gives the same result for smooth fields."""
    small, factor = _blurred_small(field, sigma)
    return _enlarge(small, field.shape) if factor > 1 else small


def relief(coverage: F32, bevel: float, light: tuple[float, float], gloss: float = 0.0, depth: float = 0.8) -> F32:
    """Brightness of a solid with edges rounded over ``bevel`` px, lit from image direction ``light``.

    The height is the coverage blurred by half the bevel. A flat interior gives 1; the edge facing the light
    gets brighter and the far edge darker, by up to ``depth``. The shading is computed on the blur's small grid
    and then enlarged, since slopes taken after enlarging ripple with the grid.
    """
    height, factor = _blurred_small(coverage, bevel / 2)
    gy, gx = np.gradient(height * np.float32(max(bevel, 1.0) / factor))
    lx, ly = light[0] * 0.7, light[1] * 0.7
    lz = math.sqrt(max(0.0, 1 - lx * lx - ly * ly))
    norm = np.sqrt(1 + gx * gx + gy * gy)
    diffuse = np.clip((lz - gx * lx - gy * ly) / norm, 0, None) / np.float32(lz)
    shade = 1 + np.float32(depth) * (diffuse - 1)
    if gloss:
        hx, hy, hz = lx, ly, lz + 1
        length = math.sqrt(hx * hx + hy * hy + hz * hz)
        facing = np.clip((hz - gx * hx - gy * hy) / (norm * length), 0, 1)
        shade = shade + np.float32(gloss) * facing**40
    shade = np.asarray(shade, np.float32)
    return _enlarge(shade, coverage.shape) if factor > 1 else shade


def _blurred_small(field: F32, sigma: float) -> tuple[F32, float]:
    """``field`` blurred by ``sigma``, on a grid shrunk by the returned factor when the blur is wide."""
    if sigma <= FAST_BLUR_SIGMA:
        return np.asarray(cv2.GaussianBlur(field, (0, 0), max(sigma, 0.5)), np.float32), 1.0
    factor = sigma / FAST_BLUR_SIGMA
    height, width = field.shape[:2]
    size = (max(2, round(width / factor)), max(2, round(height / factor)))
    small = cv2.GaussianBlur(cv2.resize(field, size, interpolation=cv2.INTER_AREA), (0, 0), FAST_BLUR_SIGMA)
    return np.asarray(small, np.float32), factor


def _enlarge(small: F32, shape: tuple[int, ...]) -> F32:
    return np.asarray(cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC), np.float32)


def soft_offset(coverage: F32, offset: tuple[float, float], sigma: float) -> F32:
    """``coverage`` moved by ``offset`` pixels and blurred by ``sigma``."""
    moved = cv2.warpAffine(coverage, np.array([[1, 0, offset[0]], [0, 1, offset[1]]], np.float32), coverage.shape[::-1])
    return smooth(moved, sigma)


def cast_shadow(target: F32, coverage: F32, offset: tuple[float, float], sigma: float, depth: float) -> None:
    """Darken ``target`` in place under ``coverage`` moved by ``offset`` and softened by ``sigma``."""
    target *= (1 - np.float32(depth) * soft_offset(coverage, offset, sigma))[..., None]


def paint(target: F32, coverage: F32, colour: F32) -> None:
    """Lay ``colour`` (one RGB value or a per-pixel field) over ``target`` where ``coverage`` is set."""
    alpha = coverage[..., None]
    target *= 1 - alpha
    target += colour * alpha
