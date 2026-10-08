"""Tesseract OCR through pytesseract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytesseract

from .base import ExtractionError, Extractor
from .pdf_ocr import RASTER_DPI, is_pdf, rasterized_pages


class TesseractExtractor(Extractor):
    """OCR an image with tesseract; a PDF is rasterised at 200 dpi and each page read in turn."""

    kind: ClassVar[str] = "tesseract"

    def __init__(self, lang: str = "eng", psm: int = 3) -> None:
        self.lang = lang
        self.psm = psm

    def info(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "tesseract": str(pytesseract.get_tesseract_version()),
            "lang": self.lang,
            "psm": self.psm,
            "pdf_dpi": RASTER_DPI,
        }

    def extract(self, path: Path) -> str:
        if not is_pdf(path):
            return self._read(path)
        with rasterized_pages(path) as pages:
            return "\n".join(self._read(page) for page in pages)

    def _read(self, image: Path) -> str:
        try:
            return str(pytesseract.image_to_string(str(image), lang=self.lang, config=f"--psm {self.psm}"))
        except (OSError, RuntimeError) as exc:
            raise ExtractionError(f"tesseract failed on {image}: {exc}") from exc
