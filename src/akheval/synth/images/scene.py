"""Objects around a photographed document: a desk edge, a sheet or envelope underneath, a pen, a phone and keys.

Every object is plain shaded geometry without text, sized in millimetres at the document's scale.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .backgrounds import F32, SURFACES, surface
from .shapes import Points, RGBf, Window, capsule, cast_shadow, ellipse, mask, paint, rectangle, relief, rotation

EDGE_P = 0.2
UNDER_P = 0.25
OBJECTS_P = 0.45
# Smallest gap between the document and the frame border that leaves room for a desk edge, as a share of the frame.
EDGE_ROOM = 0.07
OBJECT_TRIES = 30
# Share of an object's bounds that must fall inside the frame.
MIN_INSIDE = 0.25

PEN_COLOURS: tuple[RGBf, ...] = ((0.12, 0.16, 0.35), (0.08, 0.08, 0.09), (0.86, 0.86, 0.84), (0.55, 0.12, 0.12))
PHONE_BACKS: tuple[RGBf, ...] = ((0.08, 0.08, 0.09), (0.85, 0.85, 0.83), (0.42, 0.47, 0.53), (0.66, 0.58, 0.52))
KEY_METALS: tuple[RGBf, ...] = ((0.74, 0.62, 0.38), (0.72, 0.73, 0.75))
KEY_HEADS: tuple[RGBf, ...] = ((0.12, 0.12, 0.13), (0.45, 0.12, 0.12), (0.15, 0.25, 0.45))
STEEL: RGBf = (0.74, 0.75, 0.77)
UNDER_TONES: dict[str, RGBf] = {"white": (0.9, 0.9, 0.88), "kraft": (0.7, 0.56, 0.4), "manila": (0.84, 0.75, 0.56)}


@dataclass(frozen=True, slots=True)
class Part:
    """One piece of an object: its outline, colour, rounded edge width in millimetres and gloss."""

    polygons: tuple[Points, ...]
    colour: RGBf
    bevel_mm: float
    gloss: float = 0.0
    holes: tuple[Points, ...] = ()


@dataclass(frozen=True, slots=True)
class Item:
    name: str
    parts: tuple[Part, ...]
    height_mm: float  # distance above the surface, which sets the shadow offset
    params: dict[str, Any] = field(default_factory=dict)

    def outline(self) -> list[Points]:
        return [p for part in self.parts for p in part.polygons]


@dataclass(frozen=True, slots=True)
class Stage:
    """The frame, the document lying in it and the light, as the scene objects need them."""

    shape: tuple[int, int]
    outline: Points  # document corners in frame pixels
    px_per_mm: float
    light: tuple[float, float]  # unit vector in the image plane pointing towards the light

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        low, high = np.min(self.outline, axis=0), np.max(self.outline, axis=0)
        return float(low[0]), float(low[1]), float(high[0]), float(high[1])

    def shadow_offset(self, height_mm: float) -> tuple[float, float]:
        return -self.light[0] * height_mm * self.px_per_mm, -self.light[1] * height_mm * self.px_per_mm


def dress(image: F32, stage: Stage, rng: random.Random, gen: np.random.Generator, *, held: bool) -> dict[str, Any]:
    """Add a desk edge, a sheet underneath and loose objects to the surface ``image`` in place."""
    params: dict[str, Any] = {}
    if rng.random() < EDGE_P:
        params["desk_edge"] = _desk_edge(image, stage, rng, gen)
    if held:
        return params
    if rng.random() < UNDER_P:
        item = _under(stage, rng)
        _draw(image, item, stage)
        params["under"] = item.params
    if rng.random() < OBJECTS_P:
        makers = [_pen, _phone, _keys]
        rng.shuffle(makers)
        placed = []
        for maker in makers[: rng.randint(1, 2)]:
            loose = _place(maker, stage, rng)
            if loose is not None:
                _draw(image, loose, stage)
                placed.append(loose.params)
        params["objects"] = placed
    return params


def _desk_edge(image: F32, stage: Stage, rng: random.Random, gen: np.random.Generator) -> dict[str, Any] | None:
    """Replace the strip beyond the desk's edge with a darker, defocused second surface."""
    height, width = stage.shape
    x0, y0, x1, y1 = stage.bounds
    gaps = {"left": x0 / width, "right": 1 - x1 / width, "top": y0 / height, "bottom": 1 - y1 / height}
    sides = sorted(side for side, gap in gaps.items() if gap >= EDGE_ROOM)
    if not sides:
        return None
    side = rng.choice(sides)
    reach = gaps[side] * rng.uniform(0.25, 0.75)
    tilt = math.radians(rng.uniform(-5, 5))
    normal = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "top": (0.0, -1.0), "bottom": (0.0, 1.0)}[side]
    nx = normal[0] * math.cos(tilt) - normal[1] * math.sin(tilt)
    ny = normal[0] * math.sin(tilt) + normal[1] * math.cos(tilt)
    limit = width if side in ("left", "right") else height
    edge_at = reach * limit if side in ("left", "top") else (1 - reach) * limit
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    centre = (width / 2, height / 2)
    along = (xs - centre[0]) * np.float32(nx) + (ys - centre[1]) * np.float32(ny)
    edge_offset = abs(edge_at - (centre[0] if side in ("left", "right") else centre[1]))
    beyond_distance = along - np.float32(edge_offset)
    name = rng.choice(sorted(SURFACES))
    floor, floor_params = surface(name, gen, max(2, height // 4), max(2, width // 4))
    floor = cv2.resize(cv2.GaussianBlur(floor, (0, 0), 3.0), (width, height), interpolation=cv2.INTER_LINEAR)
    darkness = rng.uniform(0.45, 0.75)
    lip = max(2.0, 0.004 * max(height, width))
    floor = (
        floor * np.float32(darkness) * (1 - 0.35 * np.exp(-np.clip(beyond_distance, 0, None) / (3 * lip)))[..., None]
    )
    beyond = np.clip(beyond_distance + 0.5, 0, 1)
    image *= (1 + 0.12 * np.exp(-(((beyond_distance + lip) / lip) ** 2)))[..., None]
    paint(image, beyond, floor)
    return {
        "side": side,
        "tilt_deg": round(math.degrees(tilt), 2),
        "floor": floor_params,
        "darkness": round(darkness, 3),
    }


def _under(stage: Stage, rng: random.Random) -> Item:
    """A blank sheet or an envelope under the document, turned against it so that it shows on one or two sides."""
    ppm = stage.px_per_mm
    centre = stage.outline.mean(axis=0)
    top = stage.outline[1] - stage.outline[0]
    angle = math.atan2(float(top[1]), float(top[0])) + math.radians(rng.uniform(3, 18) * rng.choice((-1, 1)))
    kind = rng.choice(("sheet", "envelope"))
    size_mm = (297.0, 210.0) if kind == "sheet" else rng.choice(((220.0, 110.0), (229.0, 162.0)))
    span = float(np.linalg.norm(top))
    direction = rng.uniform(0, 2 * math.pi)
    shift = np.array([math.cos(direction), math.sin(direction)]) * span * rng.uniform(0.1, 0.3)
    middle = centre + shift
    tone = rng.choice(sorted(UNDER_TONES))
    outline = rectangle(middle, (size_mm[0] * ppm, size_mm[1] * ppm), angle, 1.0 * ppm)
    parts = [Part((outline,), UNDER_TONES[tone], 0.6)]
    if kind == "envelope":
        flap = _flap(middle, size_mm, angle, ppm)
        parts.append(Part((flap,), tuple(c * 0.95 for c in UNDER_TONES[tone]), 0.8))  # type: ignore[arg-type]
    return Item(kind, tuple(parts), 0.8, {"kind": kind, "tone": tone, "turn_deg": round(math.degrees(angle), 1)})


def _flap(middle: Points, size_mm: tuple[float, float], angle: float, ppm: float) -> Points:
    """The triangular back flap of an envelope."""
    w, h = size_mm[0] * ppm / 2, size_mm[1] * ppm / 2
    local = np.array([[-w, -h], [w, -h], [0, h * 0.2]], np.float64)
    turn = rotation(angle)
    return np.asarray(local @ turn.T + middle, np.float64)


def _place(
    maker: Callable[[Points, float, float, random.Random], Item], stage: Stage, rng: random.Random
) -> Item | None:
    """Put an object where it misses the document and shows at least partly in the frame."""
    height, width = stage.shape
    x0, y0, x1, y1 = stage.bounds
    pad = 4 * stage.px_per_mm
    for _ in range(OBJECT_TRIES):
        centre = np.array([rng.uniform(-0.15, 1.15) * width, rng.uniform(-0.15, 1.15) * height])
        item = maker(centre, rng.uniform(0, 2 * math.pi), stage.px_per_mm, rng)
        points = np.concatenate(item.outline())
        bx0, by0 = points.min(axis=0)
        bx1, by1 = points.max(axis=0)
        if bx0 < x1 + pad and x0 - pad < bx1 and by0 < y1 + pad and y0 - pad < by1:
            continue
        inside = max(0.0, min(bx1, width) - max(bx0, 0)) * max(0.0, min(by1, height) - max(by0, 0))
        if inside >= MIN_INSIDE * (bx1 - bx0) * (by1 - by0):
            return item
    return None


def _draw(image: F32, item: Item, stage: Stage) -> None:
    offset = stage.shadow_offset(item.height_mm)
    margin = abs(offset[0]) + abs(offset[1]) + 6 * stage.px_per_mm
    window = Window.around(item.outline(), margin, image.shape)
    if window is None:
        return
    view = window.view(image)
    union = mask(window, item.outline())
    cast_shadow(view, union, offset, 1.2 * stage.px_per_mm + item.height_mm * 0.4 * stage.px_per_mm, 0.4)
    for part in item.parts:
        coverage = mask(window, part.polygons, part.holes)
        shade = relief(coverage, part.bevel_mm * stage.px_per_mm, stage.light, part.gloss)
        paint(view, coverage, np.array(part.colour, np.float32) * shade[..., None])


def _unit(angle: float) -> Points:
    return np.array([math.cos(angle), math.sin(angle)], np.float64)


def _pen(centre: Points, angle: float, ppm: float, rng: random.Random) -> Item:
    length, radius = rng.uniform(130, 150) * ppm, rng.uniform(4.0, 5.2) * ppm
    axis = _unit(angle)
    side = _unit(angle + math.pi / 2)
    tip, back = centre - axis * length / 2, centre + axis * length / 2
    body = rng.choice(PEN_COLOURS)
    cap_length = length * rng.uniform(0.35, 0.45)
    cap = Part((capsule(back - axis * cap_length, back, radius * 1.08, radius * 1.08),), body, 4.5, 0.5)
    clip_centre = back - axis * cap_length * 0.45 + side * radius * 0.55
    clip = Part((rectangle(clip_centre, (cap_length * 0.75, 1.6 * ppm), angle, 0.6 * ppm),), STEEL, 0.6, 0.9)
    cone = np.array([tip + side * radius * 0.75, tip - side * radius * 0.75, tip - axis * 10 * ppm], np.float64)
    parts = (Part((capsule(tip, back, radius, radius),), body, 4.5, 0.5), cap, clip, Part((cone,), STEEL, 1.0, 0.9))
    return Item("pen", parts, 4.5, {"object": "pen", "length_mm": round(length / ppm, 1)})


def _phone(centre: Points, angle: float, ppm: float, rng: random.Random) -> Item:
    size = (rng.uniform(70, 77) * ppm, rng.uniform(147, 162) * ppm)
    face_up = rng.random() < 0.6
    colour = PHONE_BACKS[0] if face_up else rng.choice(PHONE_BACKS)
    parts = [Part((rectangle(centre, size, angle, 9 * ppm),), colour, 3.0, 0.5)]
    turn = rotation(angle)
    if face_up:
        screen = rectangle(centre, (size[0] - 3.6 * ppm, size[1] - 3.6 * ppm), angle, 7.5 * ppm)
        parts.append(Part((screen,), (0.03, 0.035, 0.04), 0.5, 0.3))
    else:
        module_centre = centre + np.array([-size[0] / 2 + 21 * ppm, -size[1] / 2 + 21 * ppm]) @ turn.T
        parts.append(Part((rectangle(module_centre, (32 * ppm, 32 * ppm), angle, 8 * ppm),), colour, 1.5, 0.6))
        for dx, dy in ((-7.0, -7.0), (-7.0, 7.0), (7.0, -7.0)):
            lens = module_centre + np.array([dx * ppm, dy * ppm]) @ turn.T
            parts.append(Part((ellipse(lens, (5.5 * ppm, 5.5 * ppm), 0.0),), (0.05, 0.05, 0.06), 1.5, 1.0))
    return Item("phone", tuple(parts), 8.0, {"object": "phone", "face_up": face_up})


def _keys(centre: Points, angle: float, ppm: float, rng: random.Random) -> Item:
    ring_r = rng.uniform(12, 16) * ppm
    metal = rng.choice(KEY_METALS)
    ring = Part(
        (ellipse(centre, (ring_r, ring_r), 0.0),), STEEL, 0.8, 0.9, (ellipse(centre, (ring_r - 1.6 * ppm,) * 2, 0.0),)
    )
    parts = [ring]
    count = rng.randint(1, 3)
    for k in range(count):
        direction = angle + (k - (count - 1) / 2) * rng.uniform(0.5, 0.9)
        parts += _key(centre + _unit(direction) * ring_r, direction, ppm, metal, rng)
    return Item("keys", tuple(parts), 3.0, {"object": "keys", "keys": count})


def _key(start: Points, angle: float, ppm: float, metal: RGBf, rng: random.Random) -> list[Part]:
    axis, side = _unit(angle), _unit(angle + math.pi / 2)
    head_r = rng.uniform(10, 12) * ppm
    head_centre = start + axis * (head_r - 2.5 * ppm)
    hole = ellipse(start + axis * 1.5 * ppm, (2.4 * ppm, 2.4 * ppm), 0.0)
    head_colour = metal if rng.random() < 0.5 else rng.choice(KEY_HEADS)
    blade_length, blade_w = rng.uniform(30, 38) * ppm, rng.uniform(6.5, 8) * ppm
    root = head_centre + axis * head_r * 0.8
    tip = root + axis * blade_length
    teeth = [root + side * blade_w / 2]
    steps = rng.randint(5, 7)
    for k in range(1, steps + 1):
        depth = rng.uniform(0.2, 0.45) * blade_w if k % 2 else 0.0
        teeth.append(root + axis * blade_length * k / (steps + 1) + side * (blade_w / 2 - depth))
    blade = np.array([*teeth, tip + side * blade_w * 0.1, tip - side * blade_w / 2, root - side * blade_w / 2])
    head = ellipse(head_centre, (head_r, head_r * 0.92), angle)
    return [Part((blade,), metal, 1.0, 0.8), Part((head,), head_colour, 2.5, 0.6, (hole,))]
