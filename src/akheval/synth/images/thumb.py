"""The thumb of a hand that holds a card or a passport up to the camera.

Only the thumb shows. It enters from the frame border beyond the document edge it grips and rests on the
document's margin, clear of every text line; the rest of the hand stays outside the frame. The outline is built in
document coordinates and warped with the document, so the thumb follows the same perspective.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from .backgrounds import F32, value_noise
from .shapes import Points, RGBf, Window, mask, rotation, smooth, soft_offset, text_mask
from .text import Box

SKIN_TONES: tuple[RGBf, ...] = (
    (0.91, 0.76, 0.66),
    (0.86, 0.68, 0.56),
    (0.78, 0.58, 0.45),
    (0.70, 0.51, 0.38),
    (0.60, 0.43, 0.31),
    (0.49, 0.34, 0.24),
    (0.39, 0.27, 0.19),
    (0.30, 0.21, 0.15),
)
# Nail colour: this share of the skin tone plus this share of a pale pink.
NAIL_MIX = (0.75, 0.3)
NAIL_TINT: RGBf = (0.96, 0.84, 0.82)
FREE_EDGE_TINT: RGBf = (0.95, 0.93, 0.89)
# Overlap of the thumb tip with the document, deepest first; at each position the deepest that misses text wins.
DEPTHS_MM = (12.0, 9.5, 7.5, 5.5, 4.0, 3.0)
POSITIONS = 15
TEXT_PAD_MM = 1.5
# Resolution of the text-overlap test, as a share of the document's pixels.
CHECK_SCALE = 0.25
# Longest stretch from the thumb tip to the frame border; a longer one would bring the palm into the frame.
REACH_MM = 55.0
# The tip and the middle of the nail stay this far inside the frame.
FRAME_MARGIN_MM = 2.0
LEAN_DEG = 25.0
# Sideways bend of the thumb 50 mm from its tip.
CURVE_MM = 3.0
# Half-width of the thumb at distances from its tip, in mm. Over the first entry the tip rounds off.
WIDTH_MM = ((9.5, 9.8), (20.0, 10.3), (30.0, 9.8), (45.0, 10.9), (70.0, 13.0), (110.0, 15.5))
NAIL_GAP_MM = 1.6  # from the thumb tip to the nail's free edge
NAIL_LENGTH_MM = 13.0
NAIL_HALF_WIDTH_MM = 6.3
# Superellipse exponent of the nail outline: 2 gives an ellipse, higher values squarer corners.
NAIL_ROUNDNESS = 2.2
FREE_EDGE_MM = 1.4
# Crease lines across the thumb's joint, measured from the tip.
JOINT_MM = 29.0
CREASES = 3
CREASE_DEPTH = 0.12
# Shading: share of the light in the image plane, ambient floor, wrap-around of soft indoor light, rim darkening.
LIGHT_TILT = 0.5
AMBIENT = 0.45
WRAP = 0.5
RIM_DARK = 0.2
RIM_TINT = (0.0, 0.1, 0.14)
SHEEN = 0.05
PORE_MM = 0.6
PORE_DEPTH = 0.05
# Fine lines across the back of the thumb: their spacing and darkness.
WRINKLE_MM = 0.8
WRINKLE_DEPTH = 0.07
BLOTCH_MM = 6.0
BLOTCH_DEPTH = 0.05
NAIL_GLOSS = (0.08, 0.16)
# Darkening of the skin fold along the base and sides of the nail.
FOLD_DEPTH = 0.1
# Shadow of the thumb on the document: offset away from the light and blur, and the darker contact under it.
SHADOW_MM = (1.2, 2.0)
SHADOW_DEPTH = 0.4
CONTACT_MM = 0.5
CONTACT_DEPTH = 0.25


@dataclass(frozen=True, slots=True)
class Edge:
    """One document edge: points ``start + t * along + n * out`` in document pixels, ``n`` positive outside."""

    name: str
    start: Points
    along: Points
    out: Points
    length: float

    def at(self, t: float, n: float) -> Points:
        return self.start + self.along * t + self.out * n


@dataclass(frozen=True, slots=True)
class Spot:
    """Thumb placement in document pixels: its tip, its axis towards the hand and its sideways bend."""

    edge: str
    t: float  # position of the tip along the edge
    depth: float  # overlap with the document
    lean: float  # axis turned from the edge normal, radians
    curve: float
    tip: Points
    axis: Points
    reach: float  # length of thumb between the tip and the frame border

    @property
    def normal(self) -> Points:
        return np.array([-self.axis[1], self.axis[0]])

    def point(self, u: float, v: float, ppm: float) -> Points:
        """The point ``u`` mm along the bent axis from the tip and ``v`` mm across it."""
        bend = self.curve * (u / 50.0) ** 2
        return self.tip + self.axis * (u * ppm) + self.normal * ((v * ppm) + bend)


@dataclass(frozen=True, slots=True)
class Outline:
    """The thumb, its nail and the crease lines over its joint, as point arrays."""

    thumb: Points
    nail: Points
    creases: tuple[Points, ...]

    def warped(self, matrix: npt.NDArray[np.float64]) -> Outline:
        return Outline(
            _warp(self.thumb, matrix), _warp(self.nail, matrix), tuple(_warp(c, matrix) for c in self.creases)
        )


@dataclass(frozen=True, slots=True)
class Thumb:
    """Layers in ``window`` of the frame: the shaded thumb, its coverage and its shadow on the document."""

    window: Window
    colour: F32
    alpha: F32
    shadow: F32
    params: dict[str, Any]

    def defocused(self, sigma: float) -> tuple[F32, F32]:
        """Colour and coverage blurred by ``sigma`` pixels, as a lens focused on the document shows the thumb."""
        if sigma < 0.3:
            return self.colour, self.alpha
        alpha = np.asarray(cv2.GaussianBlur(self.alpha, (0, 0), sigma), np.float32)
        colour = cv2.GaussianBlur(self.colour * self.alpha[..., None], (0, 0), sigma)
        return np.asarray(colour / np.maximum(alpha, 1e-4)[..., None], np.float32), alpha


def hold(
    size: tuple[int, int],
    boxes: Sequence[Box],
    matrix: npt.NDArray[np.float64],
    frame: tuple[int, int],
    ppm: float,
    light: tuple[float, float],
    rng: random.Random,
    gen: np.random.Generator,
) -> Thumb | None:
    """Draw the thumb that holds a document of ``size`` pixels, or return None when no edge leaves it room.

    ``matrix`` maps document pixels into a frame of ``frame`` (height, width) pixels; ``ppm`` is document pixels per
    millimetre and ``light`` the unit vector in the image plane that points towards the light.
    """
    spot = _find_spot(size, boxes, matrix, frame, ppm, rng)
    if spot is None:
        return None
    outline = _outline(spot, ppm).warped(matrix)
    window = Window.around([outline.thumb], 6 * ppm, frame)
    if window is None:
        return None
    tone = np.array(rng.choice(SKIN_TONES), np.float32) * np.float32(rng.uniform(0.95, 1.04))
    coverage = mask(window, [outline.thumb])
    shading = _shading(coverage, light, ppm)
    u, v = _axis_coordinates(window, spot, matrix, ppm)
    skin = _skin(tone, shading, u, gen, ppm) * (1 - CREASE_DEPTH * _lines(window, outline.creases, ppm))[..., None]
    gloss = rng.uniform(*NAIL_GLOSS)
    colour = _nail(skin, mask(window, [outline.nail]) * coverage, shading, tone, (u, v), spot, light, gloss, ppm)
    offset = rng.uniform(*SHADOW_MM) * ppm
    shadow = SHADOW_DEPTH * soft_offset(coverage, (-light[0] * offset, -light[1] * offset), offset)
    shadow += CONTACT_DEPTH * smooth(coverage, CONTACT_MM * ppm)
    tip = _warp(spot.tip[None, :], matrix)[0]
    params = {
        "edge": spot.edge,
        "tip_px": [round(float(tip[0]), 1), round(float(tip[1]), 1)],
        "position_mm": round(spot.t / ppm, 1),
        "overlap_mm": round(spot.depth / ppm, 1),
        "reach_mm": round(spot.reach / ppm, 1),
        "lean_deg": round(math.degrees(spot.lean), 1),
        "curve_mm": round(spot.curve / ppm, 2),
        "skin": [round(float(c), 3) for c in tone],
        "nail_gloss": round(gloss, 3),
        "shadow_mm": round(offset / ppm, 2),
    }
    return Thumb(window, colour, coverage, np.clip(shadow, 0, 0.8).astype(np.float32), params)


def _edges(size: tuple[int, int]) -> list[Edge]:
    w, h = float(size[0]), float(size[1])
    return [
        Edge("bottom", np.array([0.0, h]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), w),
        Edge("left", np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([-1.0, 0.0]), h),
        Edge("right", np.array([w, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0]), h),
        Edge("top", np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, -1.0]), w),
    ]


def _find_spot(
    size: tuple[int, int],
    boxes: Sequence[Box],
    matrix: npt.NDArray[np.float64],
    frame: tuple[int, int],
    ppm: float,
    rng: random.Random,
) -> Spot | None:
    """A thumb position on the document edge nearest a frame border that has room, trying farther edges next."""
    text = text_mask(size, boxes, TEXT_PAD_MM * ppm, CHECK_SCALE)
    lean = math.radians(rng.uniform(-LEAN_DEG, LEAN_DEG))
    curve = rng.uniform(-CURVE_MM, CURVE_MM)
    for edge in sorted(_edges(size), key=lambda e: _gap(e, matrix, frame, max(size))):
        spots = []
        for k in range(POSITIONS):
            t = edge.length * (0.08 + 0.84 * k / (POSITIONS - 1))
            spot = _deepest(edge, t, lean, curve, text, matrix, frame, ppm)
            if spot is not None:
                spots.append(spot)
        if spots:
            return rng.choices(spots, weights=[s.depth for s in spots])[0]
    return None


def _deepest(
    edge: Edge,
    t: float,
    lean: float,
    curve: float,
    text: npt.NDArray[np.uint8],
    matrix: npt.NDArray[np.float64],
    frame: tuple[int, int],
    ppm: float,
) -> Spot | None:
    """The deepest overlap at ``t`` that keeps the thumb off the text, its nail in the frame and its reach short."""
    turn = rotation(lean)
    axis = turn @ edge.out
    for depth_mm in DEPTHS_MM:
        tip = edge.at(t, -depth_mm * ppm)
        nail = tip + axis * (NAIL_GAP_MM + NAIL_LENGTH_MM / 2) * ppm
        if not all(_inside(p, matrix, frame, FRAME_MARGIN_MM * ppm) for p in (tip, nail)):
            continue
        reach = _exit(tip, axis, matrix, frame, REACH_MM * ppm)
        if reach is None:
            continue
        spot = Spot(edge.name, t, depth_mm * ppm, lean, curve * ppm, tip, axis, reach)
        if not _covers_text(text, _outline(spot, ppm).thumb):
            return spot
    return None


def _gap(edge: Edge, matrix: npt.NDArray[np.float64], frame: tuple[int, int], longest: float) -> float:
    """Distance from the middle of ``edge`` to the frame border beyond it, in document pixels."""
    gap = _exit(edge.at(edge.length / 2, 0), edge.out, matrix, frame, 4 * longest)
    return math.inf if gap is None else gap


def _exit(
    origin: Points, direction: Points, matrix: npt.NDArray[np.float64], frame: tuple[int, int], limit: float
) -> float | None:
    """Distance in document pixels from ``origin`` along ``direction`` to the frame border, or None beyond ``limit``."""
    steps = np.arange(0.0, limit + 1.0, max(1.0, limit / 400))
    warped = _warp(origin + steps[:, None] * direction, matrix)
    height, width = frame
    outside = (warped[:, 0] < 0) | (warped[:, 0] >= width) | (warped[:, 1] < 0) | (warped[:, 1] >= height)
    return float(steps[int(np.argmax(outside))]) if outside.any() else None


def _inside(point: Points, matrix: npt.NDArray[np.float64], frame: tuple[int, int], margin: float) -> bool:
    x, y = _warp(point[None, :], matrix)[0]
    return margin <= x < frame[1] - margin and margin <= y < frame[0] - margin


def _covers_text(text: npt.NDArray[np.uint8], thumb: Points) -> bool:
    drawn = np.zeros_like(text)
    cv2.fillPoly(drawn, [np.round(thumb * CHECK_SCALE).astype(np.int32)], 1)
    return bool((drawn & text).any())


def _half_width(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Half-width of the thumb in mm at ``u`` mm from its tip, rounded off over the first entry of WIDTH_MM."""
    us, ws = zip(*WIDTH_MM, strict=True)
    tip_length, tip_width = WIDTH_MM[0]
    cap = tip_width * np.sqrt(np.clip(1 - ((tip_length - u) / tip_length) ** 2, 0, 1))
    return np.where(u < tip_length, cap, np.interp(u, us, ws))


def _outline(spot: Spot, ppm: float) -> Outline:
    """Thumb, nail and joint creases in document pixels."""
    tip_length = WIDTH_MM[0][0]
    u = np.concatenate(
        [tip_length * (1 - np.cos(np.linspace(0, math.pi / 2, 14))), np.linspace(tip_length, WIDTH_MM[-1][0], 40)[1:]]
    )
    w = _half_width(u)
    left = [spot.point(a, b, ppm) for a, b in zip(u, w, strict=True)]
    right = [spot.point(a, -b, ppm) for a, b in zip(u, w, strict=True)]
    thumb = np.array(left + right[::-1], np.float64)
    angles = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    centre = NAIL_GAP_MM + NAIL_LENGTH_MM / 2
    shape = 2 / NAIL_ROUNDNESS
    nail_u = centre + NAIL_LENGTH_MM / 2 * np.sign(np.cos(angles)) * np.abs(np.cos(angles)) ** shape
    nail_v = NAIL_HALF_WIDTH_MM * np.sign(np.sin(angles)) * np.abs(np.sin(angles)) ** shape
    nail = np.array([spot.point(a, b, ppm) for a, b in zip(nail_u, nail_v, strict=True)], np.float64)
    creases = []
    across = float(_half_width(np.array([JOINT_MM]))[0]) * 0.7
    for k in range(CREASES):
        v = np.linspace(-across, across, 9)
        bow = JOINT_MM + 1.2 * k - 0.8 * (1 - (v / across) ** 2)
        creases.append(np.array([spot.point(a, b, ppm) for a, b in zip(bow, v, strict=True)], np.float64))
    return Outline(thumb, nail, tuple(creases))


def _warp(points: Points, matrix: npt.NDArray[np.float64]) -> Points:
    return np.asarray(cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2), np.float64)


@dataclass(frozen=True, slots=True)
class _Shading:
    """Light on a rounded thumb: brightness, the rim share (0 on the axis, 1 at the outline) and the sheen."""

    shade: F32
    rim: F32
    sheen: F32


def _shading(coverage: F32, light: tuple[float, float], ppm: float) -> _Shading:
    """Treat the thumb as a tube: its surface normal tilts from the viewer on the axis to sideways at the outline."""
    inside = (coverage >= 0.5).astype(np.uint8)
    distance = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    span = 2 * round(9 * ppm) + 1
    half_width = cv2.dilate(distance, np.ones((span, span), np.uint8))
    rim = np.clip(1 - distance / np.maximum(half_width, 1.0), 0, 1)
    gy, gx = np.gradient(cv2.GaussianBlur(distance, (0, 0), max(1.0, 0.5 * ppm)))
    norm = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx, ny, nz = -gx / norm * rim, -gy / norm * rim, np.sqrt(1 - rim * rim)
    lx, ly, lz = light[0] * LIGHT_TILT, light[1] * LIGHT_TILT, math.sqrt(1 - LIGHT_TILT**2)
    lit = np.clip((nx * lx + ny * ly + nz * lz + WRAP) / (1 + WRAP), 0, 1) / ((lz + WRAP) / (1 + WRAP))
    shade = (AMBIENT + (1 - AMBIENT) * lit) * (1 - RIM_DARK * rim**4)
    hx, hy, hz = lx, ly, lz + 1
    length = math.sqrt(hx * hx + hy * hy + hz * hz)
    sheen = SHEEN * np.clip((nx * hx + ny * hy + nz * hz) / length, 0, 1) ** 16
    return _Shading(np.asarray(shade, np.float32), np.asarray(rim, np.float32), np.asarray(sheen, np.float32))


def _skin(tone: npt.NDArray[np.float32], shading: _Shading, u: F32, gen: np.random.Generator, ppm: float) -> F32:
    """Skin colour under the shading, redder towards the rim, with pores, fine lines across the thumb and blotches.

    ``u`` holds the millimetres along the thumb's axis of every pixel.
    """
    height, width = shading.shade.shape
    side = max(height, width)
    pores = 1 + 2 * PORE_DEPTH * (value_noise(gen, height, width, side / (PORE_MM * ppm)) - 0.5)
    wander = 2 * math.pi * value_noise(gen, height, width, side / (4 * ppm))
    strength = value_noise(gen, height, width, side / (3 * ppm))
    pores *= 1 - WRINKLE_DEPTH * strength * (0.5 + 0.5 * np.sin(2 * math.pi * u / WRINKLE_MM + wander)) ** 4
    blotch = 2 * BLOTCH_DEPTH * (value_noise(gen, height, width, side / (BLOTCH_MM * ppm)) - 0.5)
    tint = 1 - (shading.rim**2)[..., None] * np.array(RIM_TINT, np.float32)
    redness = np.stack([1 + blotch, 1 - blotch / 2, 1 - blotch / 2], axis=-1)
    skin = tone * (shading.shade * pores)[..., None] * tint * redness + shading.sheen[..., None]
    return np.asarray(skin, np.float32)


def _lines(window: Window, lines: Sequence[Points], ppm: float) -> F32:
    """Soft dark strokes along ``lines`` inside ``window``."""
    canvas = np.zeros(window.shape, np.float32)
    origin = np.array([window.x0, window.y0])
    for line in lines:
        cv2.polylines(canvas, [np.round(line - origin).astype(np.int32)], False, 1.0, max(1, round(0.25 * ppm)))
    return np.asarray(cv2.GaussianBlur(canvas, (0, 0), max(0.6, 0.3 * ppm)), np.float32)


def _nail(
    skin: F32,
    coverage: F32,
    shading: _Shading,
    tone: npt.NDArray[np.float32],
    axes: tuple[F32, F32],
    spot: Spot,
    light: tuple[float, float],
    gloss: float,
    ppm: float,
) -> F32:
    """Lay the nail of ``coverage`` over ``skin``.

    The nail is a paler, pinker plate with a whiter free edge, a darker skin fold along its base and sides, and a
    gloss streak towards the light. ``axes`` holds the millimetres along the thumb's axis and across it of every pixel.
    """
    u, v = axes
    soft = smooth(coverage, 0.35 * ppm)
    nail_alpha = soft[..., None]
    plate = NAIL_MIX[0] * tone + NAIL_MIX[1] * np.array(NAIL_TINT, np.float32)
    plate = plate * (0.6 + 0.4 * shading.shade)[..., None]
    free = 0.5 * np.clip((NAIL_GAP_MM + FREE_EDGE_MM - u) / 0.6, 0, 1)[..., None]
    plate = plate * (1 - free) + np.array(FREE_EDGE_TINT, np.float32) * free
    centre_u = NAIL_GAP_MM + 0.45 * NAIL_LENGTH_MM + 1.5 * float(np.dot(light, spot.axis))
    centre_v = 1.5 * float(np.dot(light, spot.normal))
    streak = np.exp(
        -(((u - centre_u) / (0.3 * NAIL_LENGTH_MM)) ** 2) - ((v - centre_v) / (0.3 * NAIL_HALF_WIDTH_MM)) ** 2
    )
    plate += (gloss * streak)[..., None]
    behind_free_edge = np.clip((u - NAIL_GAP_MM - 3.0) / 2.0, 0, 1)
    fold = 4 * soft * (1 - soft) * behind_free_edge
    nailed = skin * (1 - nail_alpha) + plate * nail_alpha
    return np.asarray(nailed * (1 - FOLD_DEPTH * fold)[..., None], np.float32)


def _axis_coordinates(window: Window, spot: Spot, matrix: npt.NDArray[np.float64], ppm: float) -> tuple[F32, F32]:
    """Millimetres along the thumb's axis from its tip, and across it, for every pixel of ``window``."""
    tip, ahead = _warp(np.array([spot.tip, spot.tip + spot.axis * ppm]), matrix)
    axis = (ahead - tip) / np.linalg.norm(ahead - tip)
    normal = np.array([-axis[1], axis[0]])
    scale = float(np.linalg.norm(ahead - tip))
    ys, xs = np.mgrid[window.y0 : window.y1, window.x0 : window.x1].astype(np.float32)
    dx, dy = xs - np.float32(tip[0]), ys - np.float32(tip[1])
    u = (dx * np.float32(axis[0]) + dy * np.float32(axis[1])) / np.float32(scale)
    v = (dx * np.float32(normal[0]) + dy * np.float32(normal[1])) / np.float32(scale)
    return np.asarray(u, np.float32), np.asarray(v, np.float32)
