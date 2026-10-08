"""Phone camera model: distortion ranges per severity and the effects a capture adds to the whole frame."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .backgrounds import F32
from .output import jpeg_roundtrip
from .shapes import capsule, rectangle, smooth

Range = tuple[float, float]

# Lighting maps are smooth, so they are computed on a grid this many times coarser than the frame.
LIGHT_GRID = 4
WHITE_BALANCES = {"neutral": 0.35, "tungsten": 0.3, "daylight": 0.2, "mixed": 0.15}
# Channel gains (red, blue) of a warm tungsten lamp and of cool daylight.
TUNGSTEN_GAINS = ((1.06, 1.16), (0.74, 0.86))
DAYLIGHT_GAINS = ((0.93, 0.98), (1.03, 1.09))
SHADOW_SIDES = {"bottom": 0.5, "right": 0.2, "left": 0.2, "top": 0.1}
# Noise gain of a low-light shot, as the camera amplifies a dim signal.
LOW_LIGHT_NOISE = (1.1, 1.3)
# Quality of the phone's own JPEG before a messaging app recompresses it.
CAMERA_JPEG = (88, 95)


@dataclass(frozen=True, slots=True)
class PhotoProfile:
    """Distortion ranges for one severity. Each value is drawn uniformly from its range."""

    long_side: Range  # output long side in px
    fill_card: Range  # card width as a share of the frame width
    fill_passport: Range  # width of the passport's data page and the cover edge beside it, as a share of the frame
    fill_page: Range
    min_px_per_mm_card: float  # legibility floor in the output
    min_px_per_mm_page: float
    crop_p: float  # chance that a document margin leaves the frame
    rotation_deg: Range
    corner_jitter: Range  # perspective corner offset as a share of the document's long side
    light_falloff: Range  # depth of the illumination gradient
    vignette: Range
    phone_shadow: Range  # darkness of the photographer's phone and arm shadow
    phone_shadow_p: float
    glare: Range  # peak brightness added by lamination glare
    glare_radius: Range  # share of the document's short side
    glare_p: float
    flash_p: float  # laminated documents only
    defocus: Range  # blur sigma as a share of the text size, so small card text and page text suffer alike
    motion: Range  # shake length as a share of the text size
    motion_p: float
    low_light_p: float
    exposure: Range  # brightness left in a low-light shot
    noise_luma: Range  # standard deviation on the 0-255 scale
    noise_chroma: Range
    white_balance: Range  # largest channel gain change of a neutral white balance
    gamma_shift: Range
    contrast_shift: Range
    jpeg_quality: Range
    forward_p: float  # chance of a messaging-app downscale and recompression
    forward_long_side: Range
    forward_quality: Range
    page_bend: Range  # sag of a curled page as a share of its height
    trifold_p: float  # pages only


PHOTO: dict[str, PhotoProfile] = {
    "mild": PhotoProfile(
        long_side=(1800, 2000), fill_card=(0.72, 0.9), fill_passport=(0.78, 0.92), fill_page=(0.85, 0.95),
        min_px_per_mm_card=10.0, min_px_per_mm_page=6.0, crop_p=0.1,
        rotation_deg=(0, 3), corner_jitter=(0.0, 0.015),
        light_falloff=(0.05, 0.15), vignette=(0.05, 0.15), phone_shadow=(0.08, 0.15), phone_shadow_p=0.25,
        glare=(0.2, 0.35), glare_radius=(0.08, 0.14), glare_p=0.35, flash_p=0.1,
        defocus=(0.0, 0.02), motion=(0.05, 0.08), motion_p=0.1, low_light_p=0.0, exposure=(0.6, 0.75),
        noise_luma=(1.5, 3.0), noise_chroma=(0.5, 1.5),
        white_balance=(0.0, 0.03), gamma_shift=(0.0, 0.05), contrast_shift=(0.0, 0.05),
        jpeg_quality=(85, 92), forward_p=0.2, forward_long_side=(1500, 1600), forward_quality=(66, 75),
        page_bend=(0.0, 0.004), trifold_p=0.2,
    ),
    "moderate": PhotoProfile(
        long_side=(1400, 1700), fill_card=(0.55, 0.7), fill_passport=(0.68, 0.82), fill_page=(0.75, 0.88),
        min_px_per_mm_card=7.5, min_px_per_mm_page=4.2, crop_p=0.2,
        rotation_deg=(1, 5), corner_jitter=(0.012, 0.03),
        light_falloff=(0.12, 0.28), vignette=(0.12, 0.28), phone_shadow=(0.15, 0.28), phone_shadow_p=0.4,
        glare=(0.3, 0.5), glare_radius=(0.1, 0.18), glare_p=0.5, flash_p=0.2,
        defocus=(0.08, 0.095), motion=(0.1, 0.16), motion_p=0.4, low_light_p=0.0, exposure=(0.6, 0.75),
        noise_luma=(4.0, 6.5), noise_chroma=(1.5, 3.0),
        white_balance=(0.02, 0.06), gamma_shift=(0.05, 0.12), contrast_shift=(0.05, 0.12),
        jpeg_quality=(62, 75), forward_p=0.25, forward_long_side=(1400, 1600), forward_quality=(60, 72),
        page_bend=(0.004, 0.01), trifold_p=0.3,
    ),
    "harsh": PhotoProfile(
        long_side=(1300, 1600), fill_card=(0.55, 0.68), fill_passport=(0.64, 0.74), fill_page=(0.75, 0.88),
        min_px_per_mm_card=7.0, min_px_per_mm_page=4.2, crop_p=0.3,
        rotation_deg=(3, 9), corner_jitter=(0.03, 0.05),
        light_falloff=(0.25, 0.4), vignette=(0.25, 0.4), phone_shadow=(0.25, 0.4), phone_shadow_p=0.55,
        glare=(0.45, 0.7), glare_radius=(0.14, 0.24), glare_p=0.7, flash_p=0.3,
        defocus=(0.085, 0.1), motion=(0.11, 0.19), motion_p=0.5, low_light_p=0.3, exposure=(0.6, 0.75),
        noise_luma=(5.0, 7.5), noise_chroma=(2.5, 4.5),
        white_balance=(0.05, 0.1), gamma_shift=(0.12, 0.2), contrast_shift=(0.12, 0.2),
        jpeg_quality=(55, 68), forward_p=0.35, forward_long_side=(1280, 1500), forward_quality=(55, 65),
        page_bend=(0.01, 0.018), trifold_p=0.4,
    ),
}  # fmt: skip
# Fields where a higher value makes the photo easier to read.
EASIER_FIELDS = frozenset(
    {"long_side", "fill_card", "fill_passport", "fill_page", "min_px_per_mm_card", "min_px_per_mm_page", "jpeg_quality"}
    | {"exposure", "forward_long_side", "forward_quality"}
)


def draw(rng: random.Random, span: Range) -> float:
    return rng.uniform(*span)


def coarse_grid(shape: tuple[int, ...]) -> tuple[F32, F32]:
    """Normalised coordinates in [-1, 1] on a coarse grid over the frame."""
    h, w = max(2, shape[0] // LIGHT_GRID), max(2, shape[1] // LIGHT_GRID)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return xs / (w - 1) * 2 - 1, ys / (h - 1) * 2 - 1


def upsample(field: F32, shape: tuple[int, ...]) -> F32:
    return np.asarray(cv2.resize(field, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR), np.float32)


def illuminate(
    image: F32,
    light: tuple[float, float],
    flash: bool,
    rng: random.Random,
    profile: PhotoProfile,
    params: dict[str, Any],
) -> None:
    """Multiply the frame by a gradient that rises towards the light, vignetting and, with flash, a radial falloff."""
    xs, ys = coarse_grid(image.shape)
    falloff = draw(rng, profile.light_falloff)
    ramp = (1.42 - xs * np.float32(light[0]) - ys * np.float32(light[1])) / 2.84
    field = 1 - falloff * ramp
    vignette = draw(rng, profile.vignette)
    field *= 1 - vignette * (xs**2 + ys**2) / 2
    params["light"] = {"falloff": round(falloff, 3), "vignette": round(vignette, 3)}
    if flash:
        centre = (rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2))
        reach = rng.uniform(0.6, 1.0)
        radial = 1 / (1 + ((xs - centre[0]) ** 2 + (ys - centre[1]) ** 2) / reach**2)
        field *= 0.6 + 0.55 * radial
        params["flash"] = {"centre": [round(c, 3) for c in centre], "reach": round(reach, 3)}
    image *= upsample(field.astype(np.float32), image.shape)[..., None]


def phone_shadow(image: F32, rng: random.Random, profile: PhotoProfile, params: dict[str, Any]) -> None:
    """The soft shadow of the phone and the arm that holds it, entering from one side of the frame."""
    if rng.random() >= profile.phone_shadow_p:
        return
    height, width = (max(2, n // LIGHT_GRID) for n in image.shape[:2])
    names = sorted(SHADOW_SIDES)
    side = rng.choices(names, weights=[SHADOW_SIDES[n] for n in names])[0]
    short = min(height, width)
    inward = {"bottom": (0.0, -1.0), "top": (0.0, 1.0), "left": (1.0, 0.0), "right": (-1.0, 0.0)}[side]
    entry = {
        "bottom": (rng.uniform(0.3, 0.7) * width, height),
        "top": (rng.uniform(0.3, 0.7) * width, 0.0),
        "left": (0.0, rng.uniform(0.3, 0.7) * height),
        "right": (width, rng.uniform(0.3, 0.7) * height),
    }[side]
    depth = rng.uniform(0.3, 0.55) * (height if side in ("top", "bottom") else width)
    centre = np.array(entry) + np.array(inward) * depth
    turn = math.atan2(inward[1], inward[0]) + math.radians(rng.uniform(-20, 20))
    phone = rectangle(centre, (rng.uniform(0.55, 0.7) * short, rng.uniform(0.28, 0.38) * short), turn, 0.05 * short)
    arm_end = np.array(entry) - np.array(inward) * 0.3 * short
    arm = capsule(centre, arm_end, rng.uniform(0.12, 0.17) * short, rng.uniform(0.14, 0.2) * short)
    shape = np.zeros((height, width), np.float32)
    for part in (phone, arm):
        cv2.fillPoly(shape, [np.round(part).astype(np.int32)], 1.0, cv2.LINE_AA)
    softness = rng.uniform(0.03, 0.06) * short
    darkness = draw(rng, profile.phone_shadow)
    shape = smooth(shape, softness)
    image *= upsample(1 - np.float32(darkness) * shape, image.shape)[..., None]
    params["phone_shadow"] = {"side": side, "darkness": round(darkness, 3), "softness": round(softness / short, 3)}


def blur(
    image: F32, rng: random.Random, profile: PhotoProfile, text_px: float, low_light: bool, params: dict[str, Any]
) -> F32:
    """Defocus and shake scaled to text ``text_px`` pixels tall; a low-light shot always shakes."""
    sigma = draw(rng, profile.defocus) * text_px
    if sigma > 0.05:
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    params["defocus_sigma_px"] = round(sigma, 3)
    if low_light or rng.random() < profile.motion_p:
        length = draw(rng, profile.motion) * text_px
        angle = rng.uniform(0, math.pi)
        size = math.ceil(length) * 2 + 1
        kernel = np.zeros((size, size), np.float32)
        half, centre = length / 2, size // 2
        start = (round(centre - half * math.cos(angle)), round(centre - half * math.sin(angle)))
        end = (round(centre + half * math.cos(angle)), round(centre + half * math.sin(angle)))
        cv2.line(kernel, start, end, 1.0, 1, cv2.LINE_AA)
        image = cv2.filter2D(image, -1, kernel / kernel.sum())
        params["motion_blur"] = {"length_px": round(length, 2), "angle_rad": round(angle, 3)}
    return np.asarray(image, np.float32)


def colour(image: F32, rng: random.Random, profile: PhotoProfile, params: dict[str, Any]) -> F32:
    """White balance of a neutral, tungsten, daylight or mixed light source, then gamma and contrast."""
    names = sorted(WHITE_BALANCES)
    mode = rng.choices(names, weights=[WHITE_BALANCES[n] for n in names])[0]
    shift = draw(rng, profile.white_balance) * rng.choice((-1, 1))
    neutral = np.array([1 + shift, 1 + shift * rng.uniform(-0.3, 0.3), 1 - shift], np.float32)
    warm = np.array([draw(rng, TUNGSTEN_GAINS[0]), 1.0, draw(rng, TUNGSTEN_GAINS[1])], np.float32)
    cool = np.array([draw(rng, DAYLIGHT_GAINS[0]), 1.0, draw(rng, DAYLIGHT_GAINS[1])], np.float32)
    if mode == "mixed":
        xs, ys = coarse_grid(image.shape)
        angle = rng.uniform(0, 2 * math.pi)
        share = np.clip(0.5 + 0.6 * (xs * math.cos(angle) + ys * math.sin(angle)), 0, 1)[..., None]
        gains: F32 | np.ndarray = upsample((warm * share + cool * (1 - share)).astype(np.float32), image.shape)
        params["white_balance"] = {"mode": mode, "warm": warm.round(3).tolist(), "cool": cool.round(3).tolist()}
    else:
        gains = {"neutral": neutral, "tungsten": warm, "daylight": cool}[mode]
        params["white_balance"] = {"mode": mode, "gains": np.round(gains, 3).tolist()}
    gamma = 1 + draw(rng, profile.gamma_shift) * rng.choice((-1, 1))
    contrast = 1 + draw(rng, profile.contrast_shift) * rng.choice((-1, 1))
    np.multiply(image, gains, out=image)
    np.clip(image, 0, 1, out=image)
    cv2.pow(image, gamma, image)
    image -= 0.5
    image *= np.float32(contrast)
    image += 0.5
    params["tone"] = {"gamma": round(gamma, 4), "contrast": round(contrast, 4)}
    return image


def sensor(
    image: F32,
    rng: random.Random,
    gen: np.random.Generator,
    profile: PhotoProfile,
    low_light: bool,
    params: dict[str, Any],
) -> None:
    """Exposure loss in low light, then luma noise and coarser chroma noise as a demosaiced phone image shows."""
    height, width = image.shape[:2]
    luma, chroma = draw(rng, profile.noise_luma), draw(rng, profile.noise_chroma)
    if low_light:
        exposure = draw(rng, profile.exposure)
        image *= np.float32(exposure)
        boost = rng.uniform(*LOW_LIGHT_NOISE)
        luma, chroma = luma * boost, chroma * boost
        params["low_light"] = {"exposure": round(exposure, 3), "noise_boost": round(boost, 3)}
    image += gen.standard_normal((height, width, 1), dtype=np.float32) * np.float32(luma / 255)
    coarse = gen.standard_normal((height // 2 + 1, width // 2 + 1, 3), dtype=np.float32) * np.float32(chroma / 255)
    image += cv2.resize(coarse, (width, height), interpolation=cv2.INTER_LINEAR)
    params["noise"] = {"luma": round(luma, 3), "chroma": round(chroma, 3)}


def finish(pixels: Image.Image, rng: random.Random, profile: PhotoProfile, params: dict[str, Any]) -> Image.Image:
    """Set the JPEG quality; a forwarded photo is first saved by the phone, then shrunk and recompressed."""
    if rng.random() >= profile.forward_p:
        params["jpeg_quality"] = round(draw(rng, profile.jpeg_quality))
        return pixels
    camera = rng.randint(*CAMERA_JPEG)
    long_side = round(draw(rng, profile.forward_long_side))
    saved = jpeg_roundtrip(pixels, camera)
    factor = long_side / max(saved.size)
    if factor < 1:
        saved = saved.resize((round(saved.width * factor), round(saved.height * factor)), Image.Resampling.BILINEAR)
    params["forwarded"] = {"camera_jpeg_quality": camera, "long_side": long_side}
    params["jpeg_quality"] = round(draw(rng, profile.forward_quality))
    return saved
