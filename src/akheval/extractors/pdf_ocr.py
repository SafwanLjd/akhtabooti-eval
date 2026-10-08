"""File sniffing, PDF rasterisation and the text-layer-then-OCR extractor."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Self

from .base import ExtractionError, Extractor, ExtractorResolver

RASTER_DPI = 200
PDF_MAGIC = b"%PDF-"
IMAGE_MAGIC = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"BM", b"II*\x00", b"MM\x00*")
HEAD_BYTES = 16


def _head(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(HEAD_BYTES)
    except OSError as exc:
        raise ExtractionError(f"cannot read {path}: {exc}") from exc


def is_pdf(path: Path) -> bool:
    return _head(path).startswith(PDF_MAGIC)


def is_image(path: Path) -> bool:
    """Match PNG, JPEG, GIF, BMP, TIFF and WebP by magic bytes, as the Core's ``infer`` check does."""
    head = _head(path)
    return head.startswith(IMAGE_MAGIC) or (head[:4] == b"RIFF" and head[8:12] == b"WEBP")


@contextmanager
def rasterized_pages(pdf: Path, dpi: int = RASTER_DPI) -> Iterator[list[Path]]:
    """Render every page of ``pdf`` to PNG with pdftoppm; the files are deleted when the block exits."""
    with tempfile.TemporaryDirectory(prefix="akh-raster-") as tmp:
        prefix = Path(tmp) / "page"
        try:
            subprocess.run(["pdftoppm", "-r", str(dpi), "-png", str(pdf), str(prefix)], check=True, capture_output=True)
        except FileNotFoundError as exc:
            raise ExtractionError("pdftoppm is not installed (poppler-utils)") from exc
        except subprocess.CalledProcessError as exc:
            raise ExtractionError(f"pdftoppm failed on {pdf}: {exc.stderr.decode(errors='replace').strip()}") from exc
        # pdftoppm pads page numbers to a common width, so name order is page order.
        yield sorted(Path(tmp).glob("page-*.png"))


def visible_chars(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


class PdfOcrFallbackExtractor(Extractor):
    """Send images to OCR, PDFs to the text layer with OCR when it is near empty, other files to the text extractor.

    A PDF falls back to OCR when its text layer has fewer than ``min_chars`` non-space characters. ``text`` and
    ``ocr`` name other extractors in the config; the OCR extractor rasterises PDF pages itself.
    """

    kind: ClassVar[str] = "pdf-ocr"

    def __init__(self, text: Extractor, ocr: Extractor, min_chars: int = 20) -> None:
        self.text = text
        self.ocr = ocr
        self.min_chars = min_chars

    @classmethod
    def from_config(cls, params: Mapping[str, Any], resolve: ExtractorResolver) -> Self:
        options = dict(params)
        return cls(text=resolve(options.pop("text")), ocr=resolve(options.pop("ocr")), **options)

    def prepare(self) -> None:
        self.text.prepare()
        self.ocr.prepare()

    def close(self) -> None:
        self.text.close()
        self.ocr.close()

    def info(self) -> dict[str, Any]:
        return {"kind": self.kind, "min_chars": self.min_chars, "text": self.text.info(), "ocr": self.ocr.info()}

    def extract(self, path: Path) -> str:
        if is_image(path):
            return self.ocr.extract(path)
        text = self.text.extract(path)
        if is_pdf(path) and visible_chars(text) < self.min_chars:
            return self.ocr.extract(path)
        return text
