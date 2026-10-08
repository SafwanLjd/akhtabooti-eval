"""Raster renderings of synthetic documents for OCR testing: ID cards, passports, printed pages and chats.

``render_image`` draws a document's layout blocks, captures it as a phone photo, a flatbed scan or a
digital export, and writes JPEG, PNG or an image-only PDF. Every random choice comes from the ``rng``
argument, so the same generator state gives a byte-identical file.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...dataset import LANGUAGES, SEVERITIES
from .cards import render_id_card, render_passport
from .chat import render_chat
from .contact import make_contact_sheet
from .digital import capture_digital
from .flow import Block, parse_layout
from .output import output_format, save_image
from .page import render_page
from .photo import PHOTO_DPI, capture_photo
from .scan import capture_scan, scan_dpi
from .sheet import Sheet

__all__ = ["CAPTURES_BY_STYLE", "RENDERER_VERSION", "STYLES", "make_contact_sheet", "render_image"]

RENDERER_VERSION = "4"
STYLES = ("id_card", "passport", "page", "chat")
CAPTURES_BY_STYLE: dict[str, tuple[str, ...]] = {
    "id_card": ("photo", "scan", "digital"),
    "passport": ("photo", "scan", "digital"),
    "page": ("photo", "scan", "digital"),
    "chat": ("digital",),
}

# A renderer takes the blocks, the generator, the language and the least resolution a scan needs, in dpi.
Renderer = Callable[[Sequence[Block], random.Random, str, float], Sheet]
_RENDERERS: dict[str, Renderer] = {
    "id_card": lambda blocks, rng, _language, _dpi: render_id_card(blocks, rng),
    "passport": lambda blocks, rng, _language, _dpi: render_passport(blocks, rng),
    "page": lambda blocks, rng, _language, dpi: render_page(blocks, rng, dpi),
    "chat": lambda blocks, rng, language, _dpi: render_chat(blocks, rng, language),
}


def render_image(
    layout: Sequence[Mapping[str, Any]],
    *,
    style: str,
    language: str,
    capture: str,
    severity: str | None,
    rng: random.Random,
    out: Path,
) -> dict[str, Any]:
    """Render ``layout`` in ``style``, capture it and write ``out``; return every parameter used.

    An ID card shows the faces its blocks name: front blocks only give the front, back blocks only the back,
    both give both. A scan of both faces is an A4 photocopy with one face above the other.
    """
    _check(style, language, capture, severity)
    output_format(out)
    scan = scan_dpi(severity, rng) if capture == "scan" and severity is not None else 0.0
    sheet = _RENDERERS[style](parse_layout(layout), rng, language, scan)
    quality: int | None = None
    if capture == "photo" and severity is not None:
        image, params = capture_photo(sheet, severity, rng)
        dpi, quality = PHOTO_DPI, params["jpeg_quality"]
    elif capture == "scan" and severity is not None:
        image, params, dpi = capture_scan(sheet, severity, scan, rng)
        quality = params["jpeg_quality"]
    else:
        image, params, dpi = capture_digital(sheet)
    return {
        "renderer_version": RENDERER_VERSION,
        "style": style,
        "language": language,
        "capture": capture,
        "severity": severity,
        "document": sheet.params,
        "capture_params": params,
        "output": save_image(image, out, jpeg_quality=quality, dpi=dpi),
    }


def _check(style: str, language: str, capture: str, severity: str | None) -> None:
    if style not in CAPTURES_BY_STYLE:
        raise ValueError(f"unknown style {style!r}; expected one of {STYLES}")
    if capture not in CAPTURES_BY_STYLE[style]:
        raise ValueError(f"style {style!r} supports captures {CAPTURES_BY_STYLE[style]}, not {capture!r}")
    if language not in LANGUAGES:
        raise ValueError(f"unknown language {language!r}; expected one of {LANGUAGES}")
    if capture == "digital" and severity is not None:
        raise ValueError("digital captures take no severity")
    if capture != "digital" and severity not in SEVERITIES:
        raise ValueError(f"{capture} captures need a severity in {SEVERITIES}")
