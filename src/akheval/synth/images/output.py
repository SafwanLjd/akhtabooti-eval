"""Writing captured images as JPEG, PNG or image-only PDF."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image

SUFFIX_FORMATS = {".jpg": "jpg", ".jpeg": "jpg", ".png": "png", ".pdf": "pdf"}
# JPEG quality for files that carry no capture compression, such as a screenshot saved as .jpg.
LOSSLESS_JPEG_QUALITY = 95
# zlib level 3 writes noisy scans about three times faster than the default 6 for files about 10 % larger.
PNG_COMPRESS_LEVEL = 3


def output_format(out: Path) -> str:
    fmt = SUFFIX_FORMATS.get(out.suffix.lower())
    if fmt is None:
        raise ValueError(f"unsupported image suffix {out.suffix!r}; use one of {sorted(SUFFIX_FORMATS)}")
    return fmt


def save_image(image: Image.Image, out: Path, *, jpeg_quality: int | None, dpi: float) -> dict[str, Any]:
    """Save ``image``. A capture's JPEG loss also applies to PNG output; PDF pages embed the image as JPEG."""
    fmt = output_format(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    quality = jpeg_quality or LOSSLESS_JPEG_QUALITY
    resolution = round(dpi, 2)
    if fmt == "jpg":
        image.save(out, "JPEG", quality=quality, dpi=(resolution, resolution))
    elif fmt == "png":
        if jpeg_quality is not None:
            image = jpeg_roundtrip(image, jpeg_quality)
        image.save(out, "PNG", dpi=(resolution, resolution), compress_level=PNG_COMPRESS_LEVEL)
    else:
        image.save(out, "PDF", resolution=resolution, quality=quality, title=None, creationDate=None, modDate=None)
    return {
        "format": fmt,
        "size_px": [image.width, image.height],
        "mode": image.mode,
        "dpi": resolution,
        "jpeg_quality": quality if fmt != "png" or jpeg_quality is not None else None,
    }


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert(image.mode)
