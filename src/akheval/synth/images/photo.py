"""Phone photo capture: the document on a surface or held by a thumb, with perspective, light, blur and JPEG loss."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from . import camera
from .backgrounds import F32, SURFACES, surface
from .booklet import open_booklet
from .camera import PHOTO, PhotoProfile, Range, draw
from .paper import fold
from .scene import Stage, dress
from .shapes import cast_shadow, paint, rotation, smooth
from .sheet import Face, Focus, Plate, Sheet, arrange, transform_box
from .text import Box
from .thumb import Thumb, hold

FRAME_ASPECT = 0.75  # short side / long side of a phone camera frame
MAX_LONG_SIDE = 2000.0
PHOTO_DPI = 72.0
# Chance that the camera is held across the document, a portrait page in a landscape frame or the reverse.
ORIENTATION_SWAP_P = 0.1
HELD_P = 0.5
# Space kept between the frame border and any text line, and the document outline, as a share of the frame.
TEXT_PAD = 0.015
DOC_PAD = 0.01
# Largest share of the document's width or height that a cropped shot leaves outside the frame.
CROP_SHARE = 0.12
# A face that runs on past the frame keeps its focus in this share of the free play nearest the opposite border.
BLEED_PLAY = 0.25
# Axis and direction of each frame border.
BORDERS = {"left": (0, -1), "right": (0, 1), "top": (1, -1), "bottom": (1, 1)}
# Background defocus of a hand-held shot, as a share of the frame's long side.
DEPTH_OF_FIELD = (0.004, 0.01)
# The thumb lies a little in front of the document, so it blurs by this share of the background's defocus.
THUMB_DEFOCUS = (0.08, 0.14)
# Height of the document's edge above the surface, which sets the drop shadow offset.
LIFT_MM = (0.5, 2.0)
PAPER_GLARE = 0.4
GLARE_TRIES = 40
# A flash hotspot appears this close to the frame centre, as a share of the frame.
HOTSPOT_SPREAD = 0.12
HOTSPOT_RADIUS = (0.05, 0.09)
FOLD_STRENGTH = (0.12, 0.22)


@dataclass(frozen=True, slots=True)
class Framing:
    """The photographer's framing of the document."""

    fill: Range  # share of the frame width that the document, or its focus, takes
    floor: float  # least frame pixels per document millimetre
    bend: float = 0.0  # sag of a curled page as a share of its height
    focus: Focus | None = None  # the part to frame when the document runs on past the frame


def capture_photo(sheet: Sheet, severity: str, rng: random.Random) -> tuple[Image.Image, dict[str, Any]]:
    """Photograph ``sheet`` lying on a surface or held by a thumb; return the image and every parameter drawn."""
    profile = PHOTO[severity]
    gen = np.random.default_rng(rng.getrandbits(64))
    laminated = sheet.style in ("id_card", "passport")
    params: dict[str, Any] = {}
    faces = sheet.faces
    focus = None
    if sheet.style == "passport":
        booklet = open_booklet(faces[0], rng, sheet.px_per_mm)
        faces, focus, params["booklet"] = [booklet.face], booklet.focus, booklet.params
    plate = _plate(faces, rng, sheet.px_per_mm, params)
    if sheet.style == "page" and rng.random() < profile.trifold_p:
        plate = _trifold(plate, rng, sheet.px_per_mm, params)
    placed = _place(plate, rng, profile, sheet.px_per_mm, _framing(sheet.style, profile, focus, rng), params)
    angle = rng.uniform(0, 2 * math.pi)
    light = (math.cos(angle), math.sin(angle))
    thumb = None
    if laminated and len(faces) == 1 and rng.random() < HELD_P:
        thumb = hold(placed.size, placed.boxes, placed.matrix, placed.alpha.shape, placed.ppm, light, rng, gen)
    params["thumb"] = None if thumb is None else thumb.params
    image = _compose(placed, light, thumb, rng, gen, params)
    flash = laminated and rng.random() < profile.flash_p
    camera.illuminate(image, light, flash, rng, profile, params)
    if thumb is None:
        camera.phone_shadow(image, rng, profile, params)
    glare_strength = 1.0 if laminated else PAPER_GLARE
    if rng.random() < profile.glare_p * glare_strength:
        params["glare"] = _glare(image, placed, rng, profile, glare_strength, severity == "harsh")
    if flash:
        params["hotspot"] = _hotspot(image, placed, rng, severity == "harsh")
    low_light = rng.random() < profile.low_light_p
    image = camera.blur(image, rng, profile, sheet.text_mm * placed.ppm, low_light, params)
    image = camera.colour(image, rng, profile, params)
    camera.sensor(image, rng, gen, profile, low_light, params)
    params["px_per_mm"] = round(placed.ppm, 3)
    image *= 255
    image += 0.5
    pixels = Image.fromarray(np.clip(image, 0, 255, out=image).astype(np.uint8), "RGB")
    return camera.finish(pixels, rng, profile, params), params


def _framing(style: str, profile: PhotoProfile, focus: Focus | None, rng: random.Random) -> Framing:
    """Fill and legibility floor by document style; a page also curls."""
    if style == "passport":
        return Framing(profile.fill_passport, profile.min_px_per_mm_card, focus=focus)
    if style == "id_card":
        return Framing(profile.fill_card, profile.min_px_per_mm_card)
    bend = draw(rng, profile.page_bend) * rng.choice((-1, 1))
    return Framing(profile.fill_page, profile.min_px_per_mm_page, bend)


def _plate(faces: Sequence[Face], rng: random.Random, ppm: float, params: dict[str, Any]) -> Plate:
    if len(faces) == 1:
        params["arrangement"] = "single"
        return arrange(faces, vertical=True, gap=0)
    vertical = rng.random() < 0.5
    tilts = [rng.uniform(-3.0, 3.0) for _ in faces]
    params["arrangement"] = {
        "layout": "stacked" if vertical else "side_by_side",
        "tilts_deg": [round(t, 2) for t in tilts],
    }
    return arrange(faces, vertical=vertical, gap=rng.uniform(4.0, 12.0) * ppm, tilts=tilts)


def _trifold(plate: Plate, rng: random.Random, ppm: float, params: dict[str, Any]) -> Plate:
    """Crease a letter where it was folded in three for an envelope."""
    rgb = plate.rgb.astype(np.float32) * np.float32(1 / 255)
    params["folds"] = fold(rgb, rng, ppm, "thirds", rng.uniform(*FOLD_STRENGTH))
    folded = np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)
    return Plate(folded, plate.alpha, plate.boxes, plate.face_boxes)


@dataclass
class _Placement:
    """The document warped into the frame; boxes are in document pixels before the warp."""

    doc: F32  # premultiplied by alpha
    alpha: F32
    size: tuple[int, int]
    boxes: list[Box]
    matrix: npt.NDArray[np.float64]  # document pixels to frame pixels
    outline: npt.NDArray[np.float64]  # document corners in the frame
    ppm: float  # frame pixels per document millimetre

    def frame_boxes(self) -> list[Box]:
        return [transform_box(box, self.matrix) for box in self.boxes]


def _pose(
    plate: Plate, bend: float, rng: random.Random, profile: PhotoProfile, params: dict[str, Any]
) -> npt.NDArray[np.float64]:
    """Corners of the turned and skewed document at plate scale, centred on the origin."""
    pw, ph = plate.size
    ph_bent = ph * (1 + 2 * abs(bend))
    angle = draw(rng, profile.rotation_deg) * rng.choice((-1, 1))
    jitter = draw(rng, profile.corner_jitter) * max(pw, ph_bent)
    src = np.array([[0, 0], [pw, 0], [pw, ph_bent], [0, ph_bent]], np.float64) - [pw / 2, ph_bent / 2]
    theta = math.radians(angle)
    rot = rotation(theta)
    corners = src @ rot.T + np.array([[rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)] for _ in range(4)])
    params.update(rotation_deg=round(angle, 2), corner_jitter_px=round(jitter, 1))
    return corners


def _place(
    plate: Plate, rng: random.Random, profile: PhotoProfile, ppm: float, framing: Framing, params: dict[str, Any]
) -> _Placement:
    """Choose the frame and the document's scale and position, then warp the document into the frame.

    The fill, the fit and a crop apply to the framing's focus when it has one; the plate then runs on past the
    border on the focus's bleed side, and a crop is off.
    """
    unit = _pose(plate, framing.bend, rng, profile, params)
    pw, ph = plate.size
    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32)
    unit_matrix = cv2.getPerspectiveTransform(src, unit.astype(np.float32)).astype(np.float64)
    text = np.array([transform_box(b, unit_matrix) for b in plate.boxes] or [(0.0, 0.0, 0.0, 0.0)])
    text_lo, text_hi = text[:, :2].min(axis=0), text[:, 2:].max(axis=0)
    if framing.focus is None:
        doc_lo, doc_hi = unit.min(axis=0), unit.max(axis=0)
    else:
        bounds = np.array(transform_box(framing.focus.box, unit_matrix))
        doc_lo, doc_hi = bounds[:2], bounds[2:]
    span = doc_hi - doc_lo
    landscape = (span[0] >= span[1]) == (rng.random() >= ORIENTATION_SWAP_P)
    long_side, fill = draw(rng, profile.long_side), draw(rng, framing.fill)
    crop = rng.random() < profile.crop_p and framing.focus is None
    floor = framing.floor
    for _ in range(2):
        frame = np.array([long_side, long_side * FRAME_ASPECT] if landscape else [long_side * FRAME_ASPECT, long_side])
        # The document always fits the frame; a cropped shot then shifts it past one border by at most CROP_SHARE.
        limits = [
            fill * frame[0] / span[0],
            *(frame * (1 - 2 * TEXT_PAD) / np.maximum(text_hi - text_lo, 1)),
            *(frame * (1 - 2 * DOC_PAD) / span),
        ]
        scale = float(min(limits))
        if scale * ppm >= floor or long_side >= MAX_LONG_SIDE:
            break
        long_side = min(MAX_LONG_SIDE, long_side * floor / (scale * ppm))
    size_f = (round(frame[0]), round(frame[1]))
    bleed = None if framing.focus is None else framing.focus.bleed
    offset = _offset(frame, scale, (doc_lo, doc_hi), (text_lo, text_hi), crop, bleed, rng)
    params.update(frame_px=list(size_f), fill=round(float(scale * span[0] / frame[0]), 3), cropped=crop)
    return _warp(plate, scale, unit * scale + offset, size_f, framing.bend, ppm, params)


def _offset(
    frame: npt.NDArray[np.float64],
    scale: float,
    doc: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    text: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    crop: bool,
    bleed: str | None,
    rng: random.Random,
) -> npt.NDArray[np.float64]:
    """The document's centre in the frame.

    A cropped shot pushes one margin past the border, never a text line. With a ``bleed`` side, the document sits
    near the opposite border, so the frame shows what lies beyond it on the bleed side.
    """
    doc_low = -doc[0] * scale + frame * DOC_PAD
    doc_high = frame - doc[1] * scale - frame * DOC_PAD
    text_low = -text[0] * scale + frame * TEXT_PAD
    text_high = frame - text[1] * scale - frame * TEXT_PAD
    low, high = np.maximum(doc_low, text_low), np.minimum(doc_high, text_high)
    share = np.array([rng.random(), rng.random()])
    if bleed is not None:
        axis, direction = BORDERS[bleed]
        near = rng.uniform(0, BLEED_PLAY)
        share[axis] = 1 - near if direction < 0 else near
    offset = low + share * (high - low)
    if crop:
        axis = rng.randrange(2)
        reach = CROP_SHARE * float(doc[1][axis] - doc[0][axis]) * scale
        lowest = max(float(text_low[axis]), float(doc_low[axis]) - reach)
        highest = min(float(text_high[axis]), float(doc_high[axis]) + reach)
        if rng.random() < 0.5 and lowest < doc_low[axis]:
            offset[axis] = rng.uniform(lowest, float(min(doc_low[axis], text_high[axis])))
        elif highest > doc_high[axis]:
            offset[axis] = rng.uniform(float(max(doc_high[axis], text_low[axis])), highest)
    return np.clip(offset, text_low, text_high)


def _warp(
    plate: Plate,
    scale: float,
    corners: npt.NDArray[np.float64],
    frame: tuple[int, int],
    bend: float,
    ppm: float,
    params: dict[str, Any],
) -> _Placement:
    """Scale the plate, curl it by ``bend``, then warp it onto ``corners`` of the frame."""
    pw, ph = plate.size
    size = (max(1, round(pw * scale)), max(1, round(ph * scale)))
    rgb = cv2.resize(plate.rgb, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    alpha = cv2.resize(plate.alpha, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    sx, sy = size[0] / pw, size[1] / ph
    boxes = [(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy) for b in plate.boxes]
    if bend:
        sag = bend * size[1]
        rgb, alpha, pad = _curl(rgb, alpha, sag)
        size = (alpha.shape[1], alpha.shape[0])
        reach = abs(sag)
        boxes = [(b[0], b[1] + pad - reach, b[2], b[3] + pad + reach) for b in boxes]
        params["page_bend_px"] = round(sag, 2)
    src = np.array([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]], np.float32)
    matrix = cv2.getPerspectiveTransform(src, corners.astype(np.float32)).astype(np.float64)
    warped = cv2.warpPerspective(rgb * alpha[..., None], matrix, frame, flags=cv2.INTER_LINEAR)
    warped_alpha = cv2.warpPerspective(alpha, matrix, frame, flags=cv2.INTER_LINEAR)
    return _Placement(warped, warped_alpha, size, boxes, matrix, corners, scale * ppm)


def _curl(rgb: F32, alpha: F32, sag: float) -> tuple[F32, F32, int]:
    """Bend the page so its middle sags by ``sag`` pixels; return the padded images and the padding."""
    pad = math.ceil(abs(sag)) + 1
    rgb = cv2.copyMakeBorder(rgb, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))
    alpha = cv2.copyMakeBorder(alpha, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0.0)
    height, width = alpha.shape
    xs = np.arange(width, dtype=np.float32)
    shift = sag * np.sin(np.pi * xs / max(width - 1, 1))
    map_x = np.ascontiguousarray(np.broadcast_to(xs, (height, width)))
    map_y = np.ascontiguousarray(np.arange(height, dtype=np.float32)[:, None] - shift[None, :])
    curled = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    curled_alpha = cv2.remap(alpha, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return np.asarray(curled, np.float32), np.asarray(curled_alpha, np.float32), pad


def _compose(
    placed: _Placement,
    light: tuple[float, float],
    thumb: Thumb | None,
    rng: random.Random,
    gen: np.random.Generator,
    params: dict[str, Any],
) -> F32:
    """Lay the document on a dressed surface with a drop shadow, or hold it by a thumb before a defocused one."""
    height, width = placed.alpha.shape
    background, params["surface"] = surface(rng.choice(sorted(SURFACES)), gen, height, width)
    params["scene"] = dress(
        background, Stage((height, width), placed.outline, placed.ppm, light), rng, gen, held=thumb is not None
    )
    if thumb is None:
        lift = rng.uniform(*LIFT_MM) * placed.ppm
        sigma = lift * rng.uniform(0.8, 1.6)
        depth = rng.uniform(0.3, 0.5)
        cast_shadow(background, placed.alpha, (-light[0] * lift, -light[1] * lift), sigma, depth)
        params["drop_shadow"] = {"lift_px": round(lift, 1), "sigma_px": round(sigma, 1), "depth": round(depth, 3)}
        return np.asarray(background * (1 - placed.alpha[..., None]) + placed.doc, np.float32)
    sigma = rng.uniform(*DEPTH_OF_FIELD) * max(height, width)
    image = np.asarray(smooth(background, sigma) * (1 - placed.alpha[..., None]) + placed.doc, np.float32)
    view = thumb.window.view(image)
    view *= (1 - thumb.shadow * thumb.window.view(placed.alpha))[..., None]
    blur = sigma * rng.uniform(*THUMB_DEFOCUS)
    colour, alpha = thumb.defocused(blur)
    paint(view, alpha, colour)
    params.update(depth_of_field_px=round(sigma, 1), thumb_defocus_px=round(blur, 2))
    return image


def _glare(
    image: F32, placed: _Placement, rng: random.Random, profile: PhotoProfile, strength: float, may_cross_text: bool
) -> dict[str, Any]:
    """A soft specular streak on the document; below harsh severity it stays clear of every text line."""
    intensity = draw(rng, profile.glare) * strength
    radius = draw(rng, profile.glare_radius) * min(placed.size)
    stretch = rng.uniform(1.0, 2.2)

    def on_document() -> npt.NDArray[np.float64]:
        u, v = rng.uniform(0.1, 0.9) * placed.size[0], rng.uniform(0.1, 0.9) * placed.size[1]
        return np.asarray(cv2.perspectiveTransform(np.array([[[u, v]]], np.float64), placed.matrix)[0, 0])

    return _specular(image, placed, on_document, (radius, stretch, intensity), may_cross_text, rng)


def _hotspot(image: F32, placed: _Placement, rng: random.Random, may_cross_text: bool) -> dict[str, Any]:
    """The phone flash reflected in the laminate near the frame centre: small, bright and nearly white."""
    height, width = placed.alpha.shape
    radius = rng.uniform(*HOTSPOT_RADIUS) * min(placed.size)

    def near_centre() -> npt.NDArray[np.float64]:
        spread = HOTSPOT_SPREAD * np.array([width, height])
        return np.array([width / 2, height / 2]) + np.array([rng.gauss(0, 1), rng.gauss(0, 1)]) * spread

    return _specular(image, placed, near_centre, (radius, 1.2, rng.uniform(0.6, 0.9)), may_cross_text, rng)


def _specular(
    image: F32,
    placed: _Placement,
    where: Callable[[], npt.NDArray[np.float64]],
    shape: tuple[float, float, float],
    may_cross_text: bool,
    rng: random.Random,
) -> dict[str, Any]:
    """Brighten an elliptic blob at the first point from ``where`` on the document that the text rule allows."""
    radius, stretch, intensity = shape
    reach = radius * stretch * 1.6
    boxes = placed.frame_boxes()
    height, width = placed.alpha.shape
    for _ in range(GLARE_TRIES):
        point = where()
        x, y = round(float(point[0])), round(float(point[1]))
        if not (0 <= x < width and 0 <= y < height) or placed.alpha[y, x] < 0.9:
            continue
        if may_cross_text or not any(_near(point, reach, box) for box in boxes):
            break
    else:
        return {"applied": False}
    angle = rng.uniform(0, math.pi)
    ys, xs = np.mgrid[0 : height : camera.LIGHT_GRID, 0 : width : camera.LIGHT_GRID].astype(np.float32)
    dx, dy = xs - np.float32(point[0]), ys - np.float32(point[1])
    along = dx * math.cos(angle) + dy * math.sin(angle)
    across = -dx * math.sin(angle) + dy * math.cos(angle)
    blob = np.exp(-((along / (radius * stretch)) ** 2 + (across / radius) ** 2) * 2.0).astype(np.float32)
    image += (intensity * camera.upsample(blob, image.shape) * placed.alpha)[..., None]
    return {
        "applied": True, "intensity": round(intensity, 3), "radius_px": round(radius, 1), "stretch": round(stretch, 2),
        "centre_px": [round(float(point[0]), 1), round(float(point[1]), 1)],
    }  # fmt: skip


def _near(point: npt.NDArray[np.float64], reach: float, box: Box) -> bool:
    nx = min(max(float(point[0]), box[0]), box[2])
    ny = min(max(float(point[1]), box[1]), box[3])
    return math.hypot(float(point[0]) - nx, float(point[1]) - ny) < reach
