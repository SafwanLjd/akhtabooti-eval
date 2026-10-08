"""Image renderings: which documents get photos, scans or screenshots, and the renderer interface."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .builder import SIDES
from .config import ImageShares
from .doctypes import CARD_IMAGES, CHAT_IMAGES, PAGE_IMAGES

PHOTO = "photo"
SCAN = "scan"
SCREENSHOT = "digital"
PHOTO_FORMAT = "jpg"
SCREENSHOT_FORMAT = "png"


class ImageRenderer(Protocol):
    """Signature of akheval.synth.images.render_image."""

    def __call__(
        self,
        layout: Sequence[Mapping[str, Any]],
        *,
        style: str,
        language: str,
        capture: str,
        severity: str | None,
        rng: random.Random,
        out: Path,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ImageJob:
    capture: str
    fmt: str
    severity: str | None
    face: str | None = None
    """The card side the image shows; None shows every block."""

    @property
    def name(self) -> str:
        return self.capture if self.face is None else f"{self.capture}-{self.face}"

    def relpath(self, doc_id: str) -> str:
        return f"files/{doc_id}/{self.name}.{self.fmt}"

    def rendering_id(self, doc_id: str) -> str:
        return f"{doc_id}.{self.name}.{self.fmt}"


def _severity(shares: ImageShares, rng: random.Random) -> str:
    names = sorted(shares.severity)
    return rng.choices(names, weights=[shares.severity[n] for n in names])[0]


def plan_images(policy: str, shares: ImageShares, rng: random.Random, *, two_sided: bool = False) -> list[ImageJob]:
    """Image renderings for one document under its type's image policy.

    A two-sided card gets one photo per side, both at the same severity; its scan shows both sides on one sheet.
    """
    jobs: list[ImageJob] = []
    if policy == CARD_IMAGES:
        severity = _severity(shares, rng)
        faces: tuple[str | None, ...] = SIDES if two_sided else (None,)
        jobs += [ImageJob(PHOTO, PHOTO_FORMAT, severity, face) for face in faces]
        if rng.random() < shares.card_scan:
            jobs.append(ImageJob(SCAN, rng.choice(shares.card_scan_formats), _severity(shares, rng)))
    elif policy == PAGE_IMAGES and rng.random() < shares.page_image:
        if rng.random() < shares.page_photo:
            jobs.append(ImageJob(PHOTO, PHOTO_FORMAT, _severity(shares, rng)))
        else:
            jobs.append(ImageJob(SCAN, rng.choice(shares.page_scan_formats), _severity(shares, rng)))
    elif policy == CHAT_IMAGES and rng.random() < shares.chat_screenshot:
        jobs.append(ImageJob(SCREENSHOT, SCREENSHOT_FORMAT, None))
    return jobs


def default_renderer() -> ImageRenderer:
    """The project's image renderer, imported on first use so text-only work does not load it."""
    from .images import render_image

    return render_image
