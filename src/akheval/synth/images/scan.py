"""Flatbed scan capture: skew, lid shadow, folds, toner fade, blur, noise, dust and fax banding, grey or colour output.

A two-sided ID card becomes a grey A4 photocopy with both faces one above the other, the way people share ID copies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from .backgrounds import F32, value_noise
from .booklet import open_booklet
from .paper import fold
from .sheet import A4_MM, MM_PER_INCH, Face, Sheet, arrange

Range = tuple[float, float]

# Cards and photocopies are laid out at this resolution, or at the scan's when that is finer, before sampling.
OPTICAL_DPI = 200.0
A4_BED_P = 0.4
GRAY_P = 0.7
# A binarising scanner sets its threshold from the page (Otsu's method) and is off by up to this much.
THRESHOLD_SHIFT = (-0.04, 0.06)
BOOKLET_SCAN_P = 0.5
HOLE_DIAMETER_MM = 6.0
HOLE_EDGE_MM = 12.0
HOLE_SPACING_MM = 80.0
FOLD_PATTERNS = ("half", "thirds", "quarters")
# Depth of the periodic fax ripple as a share of the banding depth.
RIPPLE_SHARE = 0.15
# Weights that turn RGB into luma.
LUMA = np.array([0.299, 0.587, 0.114], np.float32)


@dataclass(frozen=True, slots=True)
class ScanProfile:
    """Distortion ranges for one severity. Each value is drawn uniformly from its range."""

    skew_deg: Range
    blur_mm: Range
    dpi: Range  # sampling resolution of the scanner or fax
    brightness: Range
    contrast_shift: Range
    ink_loss: Range  # share of the ink's darkness lost over the whole sheet: low toner or a bright setting
    noise: Range  # standard deviation on the 0-255 scale
    dust_per_dm2: Range
    edge_shadow_p: float
    edge_shadow_mm: Range
    edge_shadow_depth: Range
    punch_p: float
    fold_p: float  # pages and photocopies
    fold_strength: Range
    fade_p: float  # pages and photocopies
    fade_strength: Range  # share of the toner lost in a faded patch
    speckle_per_dm2: Range  # stray toner dots on pages and photocopies
    streak_p: float
    banding_p: float
    banding_depth: Range
    jpeg_quality: Range
    binarize_p: float


SCAN: dict[str, ScanProfile] = {
    "mild": ScanProfile(
        skew_deg=(0.0, 0.5), blur_mm=(0.02, 0.05), dpi=(200, 300),
        brightness=(0.0, 0.03), contrast_shift=(0.0, 0.05), ink_loss=(0.0, 0.1), noise=(1.0, 2.5),
        dust_per_dm2=(0, 3), edge_shadow_p=0.3, edge_shadow_mm=(2, 6), edge_shadow_depth=(0.3, 0.5), punch_p=0.2,
        fold_p=0.1, fold_strength=(0.08, 0.14), fade_p=0.1, fade_strength=(0.15, 0.3), speckle_per_dm2=(5, 20),
        streak_p=0.0, banding_p=0.0, banding_depth=(0.03, 0.06), jpeg_quality=(90, 95), binarize_p=0.0,
    ),
    "moderate": ScanProfile(
        skew_deg=(0.4, 1.5), blur_mm=(0.13, 0.17), dpi=(120, 150),
        brightness=(0.03, 0.07), contrast_shift=(0.05, 0.12), ink_loss=(0.35, 0.5), noise=(5.0, 8.0),
        dust_per_dm2=(3, 12), edge_shadow_p=0.5, edge_shadow_mm=(4, 10), edge_shadow_depth=(0.45, 0.7), punch_p=0.3,
        fold_p=0.25, fold_strength=(0.12, 0.2), fade_p=0.25, fade_strength=(0.2, 0.35), speckle_per_dm2=(20, 80),
        streak_p=0.3, banding_p=0.0, banding_depth=(0.05, 0.1), jpeg_quality=(48, 62), binarize_p=0.0,
    ),
    "harsh": ScanProfile(
        skew_deg=(1.2, 3.0), blur_mm=(0.15, 0.2), dpi=(90, 110),
        brightness=(0.07, 0.12), contrast_shift=(0.12, 0.22), ink_loss=(0.4, 0.55), noise=(5.5, 8.5),
        dust_per_dm2=(12, 40), edge_shadow_p=0.8, edge_shadow_mm=(8, 16), edge_shadow_depth=(0.6, 0.9), punch_p=0.4,
        fold_p=0.4, fold_strength=(0.18, 0.3), fade_p=0.45, fade_strength=(0.3, 0.5), speckle_per_dm2=(80, 250),
        streak_p=0.6, banding_p=0.5, banding_depth=(0.08, 0.16), jpeg_quality=(42, 55), binarize_p=0.5,
    ),
}  # fmt: skip
EASIER_FIELDS = frozenset({"jpeg_quality", "dpi"})


def _draw(rng: random.Random, span: Range) -> float:
    return rng.uniform(*span)


def scan_dpi(severity: str, rng: random.Random) -> float:
    """Draw the resolution of a scan at ``severity``; a page is rendered at least this fine before its scan."""
    return _draw(rng, SCAN[severity].dpi)


def capture_scan(
    sheet: Sheet, severity: str, dpi: float, rng: random.Random
) -> tuple[Image.Image, dict[str, Any], float]:
    """Scan ``sheet`` at ``dpi``; return the image, the parameters used and the resolution of the image."""
    profile = SCAN[severity]
    gen = np.random.default_rng(rng.getrandbits(64))
    params: dict[str, Any] = {}
    lid = rng.uniform(0.93, 1.0)
    paper = sheet.style == "page"
    copy = sheet.style == "id_card" and len(sheet.faces) > 1
    optical = max(dpi, OPTICAL_DPI) / MM_PER_INCH
    if paper:
        image = np.asarray(sheet.faces[0].image.convert("RGB"), np.float32) * np.float32(1 / 255)
        ppm = sheet.px_per_mm
        if rng.random() < profile.punch_p:
            _punch_holes(image, rng, ppm, lid, params)
    elif copy:
        image, ppm = _photocopy(sheet, rng, gen, optical, params)
    else:
        image, ppm = _bed(sheet, rng, lid, optical, params)
    if paper or copy:
        _wear(image, rng, gen, profile, ppm, params)
    image = _skew(image, rng, profile, lid, params)
    _dust(image, rng, gen, profile.dust_per_dm2, (0.05, 0.5), ppm, "dust", params)
    if rng.random() < profile.edge_shadow_p:
        _edge_shadow(image, rng, profile, ppm, params)
    sigma = _draw(rng, profile.blur_mm) * ppm
    image = np.asarray(cv2.GaussianBlur(image, (0, 0), sigma), np.float32)
    image, ppm = _sample(image, ppm, dpi / MM_PER_INCH)
    params["blur_sigma_px"] = round(sigma, 3)
    image = _tone(image, rng, gen, profile, lid, params)
    if rng.random() < profile.streak_p:
        _streaks(image, rng, params)
    if rng.random() < profile.banding_p:
        _banding(image, rng, profile, params)
    params["dpi"] = round(ppm * MM_PER_INCH, 2)
    pixels = _output(np.clip(image, 0, 1), rng, profile, paper, copy, params)
    return pixels, params, ppm * MM_PER_INCH


def _wear(
    image: F32, rng: random.Random, gen: np.random.Generator, profile: ScanProfile, ppm: float, params: dict[str, Any]
) -> None:
    """Wear of a printed sheet: folds, faded toner and stray toner specks."""
    if rng.random() < profile.fold_p:
        params["folds"] = fold(image, rng, ppm, rng.choice(FOLD_PATTERNS), _draw(rng, profile.fold_strength))
    if rng.random() < profile.fade_p:
        _fade(image, rng, gen, profile, params)
    _dust(image, rng, gen, profile.speckle_per_dm2, (0.02, 0.2), ppm, "toner_speckle", params)


def _tone(
    image: F32, rng: random.Random, gen: np.random.Generator, profile: ScanProfile, lid: float, params: dict[str, Any]
) -> F32:
    """Light print, brightness and contrast of the scanner setting, then sensor noise."""
    ink_loss = _draw(rng, profile.ink_loss)
    brightness = _draw(rng, profile.brightness) * rng.choice((-1, 1))
    contrast = 1 + _draw(rng, profile.contrast_shift) * rng.choice((-1, 1))
    image = ((1 - (1 - image) * (1 - ink_loss)) - 0.5) * contrast + 0.5 + brightness
    noise = _draw(rng, profile.noise)
    image += gen.standard_normal((*image.shape[:2], 1), dtype=np.float32) * np.float32(noise / 255)
    params.update(
        lid=round(lid, 3), ink_loss=round(ink_loss, 3), brightness=round(brightness, 4), contrast=round(contrast, 4),
        noise=round(noise, 3),
    )  # fmt: skip
    return np.asarray(image, np.float32)


def _output(
    image: F32, rng: random.Random, profile: ScanProfile, paper: bool, copy: bool, params: dict[str, Any]
) -> Image.Image:
    """Grey, binarised or colour pixels: only a page may be binarised and a photocopy stays grey.

    JPEG loss applies to the saved file.
    """
    binarize = paper and rng.random() < profile.binarize_p
    gray = copy or binarize or rng.random() < GRAY_P
    quality = round(_draw(rng, profile.jpeg_quality))
    params.update(grayscale=gray, binarized=binarize, jpeg_quality=None if binarize else quality)
    if not gray:
        return Image.fromarray(np.clip(image * 255 + 0.5, 0, 255).astype(np.uint8), "RGB")
    luma = image @ LUMA
    if binarize:
        levels = np.clip(luma * 255, 0, 255).astype(np.uint8)
        otsu, _ = cv2.threshold(levels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = otsu / 255 + rng.uniform(*THRESHOLD_SHIFT)
        luma = (luma > threshold).astype(np.float32)
        params["threshold"] = round(threshold, 3)
    return Image.fromarray(np.clip(luma * 255 + 0.5, 0, 255).astype(np.uint8), "L")


def _sample(image: F32, ppm: float, target: float) -> tuple[F32, float]:
    """Resample the optical image at ``target`` pixels per millimetre, as the scanner's sensor does."""
    height, width = image.shape[:2]
    size = (max(1, round(width * target / ppm)), max(1, round(height * target / ppm)))
    if size == (width, height):
        return image, ppm
    return np.asarray(_resize(image, size), np.float32), ppm * size[0] / width


def _resize(image: npt.NDArray[Any], size: tuple[int, int]) -> npt.NDArray[Any]:
    """Area averaging to shrink, cubic interpolation to enlarge."""
    shrink = size[0] < image.shape[1]
    return np.asarray(cv2.resize(image, size, interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC))


def _faces(sheet: Sheet, rng: random.Random, params: dict[str, Any]) -> list[Face]:
    """The faces that go on the glass; a passport goes as the open booklet half the time."""
    if sheet.style == "passport" and rng.random() < BOOKLET_SCAN_P:
        booklet = open_booklet(sheet.faces[0], rng, sheet.px_per_mm)
        params["booklet"] = booklet.params
        return [booklet.face]
    return list(sheet.faces)


def _bed(sheet: Sheet, rng: random.Random, lid: float, ppm: float, params: dict[str, Any]) -> tuple[F32, float]:
    """Lay one card face or the passport on the scanner glass at ``ppm``: a cropped scan or a full A4 page."""
    a4 = rng.random() < A4_BED_P
    plate = arrange(_faces(sheet, rng, params), vertical=True, gap=0)
    factor = ppm / sheet.px_per_mm
    size = (round(plate.size[0] * factor), round(plate.size[1] * factor))
    rgb_plate = _resize(plate.rgb, size)
    alpha_plate = _resize(plate.alpha, size)
    if a4:
        width, height = round(A4_MM[0] * ppm), round(A4_MM[1] * ppm)
        width, height = max(width, size[0] + round(20 * ppm)), max(height, size[1] + round(20 * ppm))
        x, y = round(rng.uniform(10.0, 30.0) * ppm), round(rng.uniform(10.0, 30.0) * ppm)
        x, y = min(x, width - size[0]), min(y, height - size[1])
        params["bed"] = {"mode": "a4", "offset_mm": [round(x / ppm, 1), round(y / ppm, 1)]}
    else:
        margins = [rng.uniform(6.0, 20.0) * ppm for _ in range(4)]
        width = round(size[0] + margins[0] + margins[2])
        height = round(size[1] + margins[1] + margins[3])
        x, y = round(margins[0]), round(margins[1])
        params["bed"] = {"mode": "crop", "margins_mm": [round(m / ppm, 1) for m in margins]}
    canvas = np.full((height, width, 3), lid, np.float32)
    alpha = np.zeros((height, width), np.float32)
    alpha[y : y + size[1], x : x + size[0]] = alpha_plate.astype(np.float32) / 255
    rgb = np.zeros_like(canvas)
    rgb[y : y + size[1], x : x + size[0]] = rgb_plate.astype(np.float32) / 255
    edge = cv2.GaussianBlur(np.roll(alpha, (round(ppm * 0.3), round(ppm * 0.3)), axis=(0, 1)), (0, 0), ppm * 0.4)
    canvas *= (1 - 0.25 * edge)[..., None]
    return np.asarray(canvas * (1 - alpha[..., None]) + rgb * alpha[..., None], np.float32), ppm


def _photocopy(
    sheet: Sheet, rng: random.Random, gen: np.random.Generator, ppm: float, params: dict[str, Any]
) -> tuple[F32, float]:
    """Both card faces copied one above the other on A4 at ``ppm``: grey card bodies, black text, light grey paper."""
    factor = ppm / sheet.px_per_mm
    faces = []
    for face in sheet.faces:
        size = (round(face.image.width * factor), round(face.image.height * factor))
        mask = None if face.mask is None else face.mask.resize(size, Image.Resampling.BILINEAR)
        boxes = [(b[0] * factor, b[1] * factor, b[2] * factor, b[3] * factor) for b in face.boxes]
        faces.append(Face(face.image.resize(size, Image.Resampling.LANCZOS), boxes, mask))
    tilts = [rng.uniform(-1.5, 1.5) for _ in faces]
    plate = arrange(faces, vertical=True, gap=rng.uniform(10.0, 30.0) * ppm, tilts=tilts)
    width, height = round(A4_MM[0] * ppm), round(A4_MM[1] * ppm)
    paper = rng.uniform(0.84, 0.92)
    white, black = rng.uniform(0.96, 1.04), rng.uniform(0.25, 0.4)
    luma = (plate.rgb.astype(np.float32) / 255) @ LUMA
    copied = np.clip((luma - black) / (white - black), 0, 1) ** np.float32(rng.uniform(1.2, 1.5))
    alpha = plate.alpha.astype(np.float32) / 255
    outline = np.clip(alpha - cv2.erode(alpha, np.ones((5, 5), np.uint8)), 0, 1)
    copied = copied * (1 - 0.7 * outline)
    page = paper * (0.97 + 0.06 * value_noise(gen, height, width, 6))
    x = round((width - plate.size[0]) / 2 + rng.uniform(-10, 10) * ppm)
    y = round(rng.uniform(20.0, 50.0) * ppm)
    x, y = max(0, min(x, width - plate.size[0])), max(0, min(y, height - plate.size[1]))
    region = page[y : y + plate.size[1], x : x + plate.size[0]]
    region *= 1 - alpha * (1 - copied)
    params["photocopy"] = {
        "paper": round(paper, 3), "white": round(white, 3), "black": round(black, 3),
        "offset_mm": [round(x / ppm, 1), round(y / ppm, 1)], "tilts_deg": [round(t, 2) for t in tilts],
    }  # fmt: skip
    return np.repeat(page[..., None], 3, axis=2).astype(np.float32), ppm


def _punch_holes(image: F32, rng: random.Random, ppm: float, lid: float, params: dict[str, Any]) -> None:
    """Two filing holes in the page margin, showing the scanner lid through the paper."""
    height, width = image.shape[:2]
    left = rng.random() < 0.5
    cx = HOLE_EDGE_MM * ppm if left else width - HOLE_EDGE_MM * ppm
    radius = HOLE_DIAMETER_MM / 2 * ppm
    for dy in (-HOLE_SPACING_MM / 2, HOLE_SPACING_MM / 2):
        centre = (round(cx), round(height / 2 + dy * ppm))
        cv2.circle(image, centre, round(radius), (lid * 0.8,) * 3, -1, cv2.LINE_AA)
        cv2.circle(image, centre, round(radius), (lid * 0.55,) * 3, max(1, round(ppm * 0.3)), cv2.LINE_AA)
    params["punch_holes"] = "left" if left else "right"


def _fade(
    image: F32, rng: random.Random, gen: np.random.Generator, profile: ScanProfile, params: dict[str, Any]
) -> None:
    """Patches where the printer or copier laid down too little toner: dark ink turns grey there."""
    height, width = image.shape[:2]
    strength = _draw(rng, profile.fade_strength)
    small = value_noise(gen, max(2, height // 8), max(2, width // 8), rng.uniform(2.5, 5.0))
    patches = np.clip((small - 0.55) / 0.2, 0, 1)
    patches = cv2.resize(patches, (width, height), interpolation=cv2.INTER_LINEAR)
    image[:] = 1 - (1 - image) * (1 - strength * patches)[..., None]
    params["toner_fade"] = {"strength": round(strength, 3), "area": round(float((patches > 0.5).mean()), 3)}


def _skew(image: F32, rng: random.Random, profile: ScanProfile, lid: float, params: dict[str, Any]) -> F32:
    angle = _draw(rng, profile.skew_deg) * rng.choice((-1, 1))
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    params["skew_deg"] = round(angle, 3)
    skewed = cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(lid,) * 3
    )
    return np.asarray(skewed, np.float32)


def _edge_shadow(image: F32, rng: random.Random, profile: ScanProfile, ppm: float, params: dict[str, Any]) -> None:
    """Darken one edge the way an open lid or the page border shows on a scan."""
    height, width = image.shape[:2]
    edge = rng.choice(("left", "right", "top", "bottom"))
    reach = _draw(rng, profile.edge_shadow_mm) * ppm
    depth = _draw(rng, profile.edge_shadow_depth)
    length = width if edge in ("left", "right") else height
    distance = np.arange(length, dtype=np.float32)
    if edge in ("right", "bottom"):
        distance = distance[::-1]
    falloff = 1 - depth * np.exp(-distance / max(reach / 3, 1.0))
    if edge in ("left", "right"):
        image *= falloff[None, :, None]
    else:
        image *= falloff[:, None, None]
    params["edge_shadow"] = {"edge": edge, "reach_mm": round(reach / ppm, 2), "depth": round(depth, 3)}


def _dust(
    image: F32,
    rng: random.Random,
    gen: np.random.Generator,
    density: Range,
    radius_mm: Range,
    ppm: float,
    name: str,
    params: dict[str, Any],
) -> None:
    """Dark specks, ``density`` per square decimetre: dust on the glass or stray toner."""
    height, width = image.shape[:2]
    area_dm2 = height * width / (ppm * 100) ** 2
    count = round(_draw(rng, density) * area_dm2)
    xs = gen.integers(0, width, count)
    ys = gen.integers(0, height, count)
    radii = gen.uniform(*radius_mm, count) * ppm
    shades = gen.uniform(0.05, 0.5, count)
    for x, y, r, s in zip(xs, ys, radii, shades, strict=True):
        cv2.circle(image, (int(x), int(y)), max(1, round(float(r))), (float(s),) * 3, -1, cv2.LINE_AA)
    params[name] = count


def _streaks(image: F32, rng: random.Random, params: dict[str, Any]) -> None:
    """Faint vertical lines from a dirty scanner glass or sensor."""
    width = image.shape[1]
    streaks = []
    for _ in range(rng.randint(1, 3)):
        x = rng.randrange(width)
        thickness = rng.randint(1, 3)
        darkness = rng.uniform(0.05, 0.15) * rng.choice((-1, 1))
        image[:, x : x + thickness] *= 1 - darkness
        streaks.append({"x": x, "px": thickness, "darkness": round(darkness, 3)})
    params["streaks"] = streaks


def _banding(image: F32, rng: random.Random, profile: ScanProfile, params: dict[str, Any]) -> None:
    """Fax-like horizontal bands: a periodic ripple from the stepper motor and a few lighter or darker bands."""
    height = image.shape[0]
    depth = _draw(rng, profile.banding_depth)
    rows = np.arange(height, dtype=np.float32)
    period = rng.uniform(6.0, 18.0)
    factor = 1 + RIPPLE_SHARE * depth * np.sin(2 * math.pi * rows / period + rng.uniform(0, 2 * math.pi))
    bands = rng.randint(3, 8)
    for _ in range(bands):
        top = rng.randrange(height)
        factor[top : top + rng.randint(2, 14)] *= 1 + depth * rng.choice((-1, 1))
    image *= factor[:, None, None].astype(np.float32)
    params["banding"] = {"depth": round(depth, 3), "period_px": round(period, 2), "bands": bands}
