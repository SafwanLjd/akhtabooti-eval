"""Digital capture: the document exactly as drawn, as an export or a screenshot would save it."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .sheet import MM_PER_INCH, Sheet, arrange

# Gap between the faces of a two-sided card in a digital export.
FACE_GAP_MM = 6.0


def capture_digital(sheet: Sheet) -> tuple[Image.Image, dict[str, Any], float]:
    """Return the flat image, the parameters used and its resolution in dpi."""
    dpi = sheet.px_per_mm * MM_PER_INCH
    faces = sheet.faces
    if len(faces) == 1 and faces[0].mask is None:
        return faces[0].image.convert("RGB"), {"arrangement": "single"}, dpi
    plate = arrange(faces, vertical=True, gap=FACE_GAP_MM * sheet.px_per_mm)
    alpha = plate.alpha.astype(np.float32)[..., None] / 255
    white = np.full_like(plate.rgb, 255, dtype=np.float32)
    flat = plate.rgb.astype(np.float32) * alpha + white * (1 - alpha)
    image = Image.fromarray(np.clip(flat + 0.5, 0, 255).astype(np.uint8), "RGB")
    return image, {"arrangement": "stacked" if len(faces) > 1 else "single"}, dpi
