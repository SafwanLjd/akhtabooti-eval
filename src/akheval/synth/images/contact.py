"""Contact sheets: thumbnails of rendered files in a grid, captioned with their folder and file name."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .fonts import REGULAR_STYLES, find_face

CAPTION_PX = 22
GUTTER_PX = 12
SHEET_BACKGROUND = (48, 48, 52)
CAPTION_COLOR = (230, 230, 230)
PDF_PREVIEW_DPI = 60


def make_contact_sheet(images: Sequence[Path], out: Path, columns: int = 4, thumb_width: int = 480) -> None:
    """Write a grid of thumbnails of ``images`` (JPEG, PNG or PDF first page) to ``out``."""
    if not images:
        raise ValueError("no images for the contact sheet")
    thumbs = [_thumbnail(Path(path), thumb_width) for path in images]
    face = find_face("Noto Sans", REGULAR_STYLES) or find_face("Liberation Sans", REGULAR_STYLES)
    font = face.font(CAPTION_PX) if face else ImageFont.load_default(CAPTION_PX)
    rows = [thumbs[i : i + columns] for i in range(0, len(thumbs), columns)]
    cell_w = thumb_width + GUTTER_PX
    row_heights = [max(t.height for t in row) + CAPTION_PX + 2 * GUTTER_PX for row in rows]
    sheet = Image.new("RGB", (columns * cell_w + GUTTER_PX, sum(row_heights) + GUTTER_PX), SHEET_BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    y = GUTTER_PX
    for r, row in enumerate(rows):
        for c, thumb in enumerate(row):
            x = GUTTER_PX + c * cell_w
            sheet.paste(thumb, (x, y))
            path = Path(images[r * columns + c])
            caption = _fit(f"{path.parent.name}/{path.name}", font, thumb_width)
            draw.text((x, y + thumb.height + 4), caption, font=font, fill=CAPTION_COLOR)
        y += row_heights[r]
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def _fit(caption: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, width: int) -> str:
    """``caption``, cut at the start with an ellipsis until it fits ``width`` pixels."""
    if font.getlength(caption) <= width:
        return caption
    while len(caption) > 1 and font.getlength("…" + caption) > width:
        caption = caption[1:]
    return "…" + caption


def _thumbnail(path: Path, width: int) -> Image.Image:
    with _open(path) as image:
        rgb = image.convert("RGB")
    height = max(1, round(rgb.height * width / rgb.width))
    return rgb.resize((width, height), Image.Resampling.LANCZOS)


def _open(path: Path) -> Image.Image:
    if path.suffix.lower() != ".pdf":
        return Image.open(path)
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise RuntimeError("pdftoppm from poppler is required to preview PDF files")
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run(
            [pdftoppm, "-r", str(PDF_PREVIEW_DPI), "-png", "-singlefile", "-f", "1", "-l", "1", str(path), str(prefix)],
            check=True,
            capture_output=True,
        )
        with Image.open(prefix.with_suffix(".png")) as page:
            page.load()
            return page.copy()
