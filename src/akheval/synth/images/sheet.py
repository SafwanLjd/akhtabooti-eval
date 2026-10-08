"""Rendered document faces and their arrangement on one transparent plate."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from .text import Box

U8 = npt.NDArray[np.uint8]
MM_PER_INCH = 25.4
A4_MM = (210.0, 297.0)


@dataclass
class Face:
    """One side of a document. ``mask`` holds its outline (rounded card corners); None means the full rectangle."""

    image: Image.Image
    boxes: list[Box]
    mask: Image.Image | None = None

    def alpha(self) -> U8:
        if self.mask is None:
            return np.full((self.image.height, self.image.width), 255, np.uint8)
        return np.asarray(self.mask, dtype=np.uint8)


@dataclass
class Sheet:
    """A document drawn flat, before any capture: one face, or the front and back of a card."""

    style: str
    faces: list[Face]
    px_per_mm: float
    text_mm: float  # size of the smallest text with words (labels on cards, body on pages); sets the blur it survives
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Focus:
    """The part of a face that a photo frames, and the side on which the face runs on past the frame border."""

    box: Box
    bleed: str  # "top", "bottom", "left" or "right"


@dataclass
class Plate:
    """Faces placed on one transparent canvas, with every text box and face outline in canvas pixels."""

    rgb: U8
    alpha: U8
    boxes: list[Box]
    face_boxes: list[Box]

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]


def arrange(faces: Sequence[Face], *, vertical: bool, gap: float, tilts: Sequence[float] = ()) -> Plate:
    """Place faces in a row or a column, each turned by its tilt in degrees, in slots that do not overlap."""
    turned = [_turn(face, tilts[i] if i < len(tilts) else 0.0) for i, face in enumerate(faces)]
    widths = [plate.size[0] for plate in turned]
    heights = [plate.size[1] for plate in turned]
    if vertical:
        width, height = max(widths), round(sum(heights) + gap * (len(turned) - 1))
    else:
        width, height = round(sum(widths) + gap * (len(turned) - 1)), max(heights)
    rgb = np.zeros((height, width, 3), np.uint8)
    alpha = np.zeros((height, width), np.uint8)
    boxes: list[Box] = []
    face_boxes: list[Box] = []
    cursor = 0.0
    for plate, w, h in zip(turned, widths, heights, strict=True):
        x = round((width - w) / 2) if vertical else round(cursor)
        y = round(cursor) if vertical else round((height - h) / 2)
        rgb[y : y + h, x : x + w] = plate.rgb
        alpha[y : y + h, x : x + w] = plate.alpha
        boxes += [(b[0] + x, b[1] + y, b[2] + x, b[3] + y) for b in plate.boxes]
        face_boxes += [(b[0] + x, b[1] + y, b[2] + x, b[3] + y) for b in plate.face_boxes]
        cursor += (h if vertical else w) + gap
    return Plate(rgb, alpha, boxes, face_boxes)


def _turn(face: Face, degrees: float) -> Plate:
    rgb = np.asarray(face.image.convert("RGB"), dtype=np.uint8)
    alpha = face.alpha()
    h, w = alpha.shape
    if abs(degrees) < 1e-6:
        return Plate(rgb, alpha, list(face.boxes), [(0.0, 0.0, float(w), float(h))])
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    cos, sin = abs(math.cos(math.radians(degrees))), abs(math.sin(math.radians(degrees)))
    new_w, new_h = math.ceil(w * cos + h * sin), math.ceil(w * sin + h * cos)
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2
    turned_rgb = cv2.warpAffine(rgb, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR)
    turned_alpha = cv2.warpAffine(alpha, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR)
    boxes = [transform_box(b, matrix) for b in face.boxes]
    return Plate(turned_rgb, turned_alpha, boxes, [transform_box((0, 0, w, h), matrix)])


def transform_box(box: Box, matrix: npt.NDArray[np.float64]) -> Box:
    """Axis-aligned bounds of ``box`` after a 2x3 affine or 3x3 perspective transform."""
    x0, y0, x1, y1 = box
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
    if matrix.shape == (2, 3):
        moved = corners @ matrix[:, :2].T + matrix[:, 2]
    else:
        homogeneous = np.c_[corners, np.ones(4)] @ matrix.T
        moved = homogeneous[:, :2] / homogeneous[:, 2:3]
    xs, ys = moved[:, 0], moved[:, 1]
    return float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))
