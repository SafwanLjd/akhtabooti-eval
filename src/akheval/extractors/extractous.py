"""Apache Tika through extractous with the library defaults, the text path of the Akhtabooti Core.

The Core calls ``Extractor::new().extract_file_to_string(path)`` from extractous 0.3.0; the Python binding wraps
the same native Tika build. Measured on this machine with the defaults:

* docx: body paragraphs and tables. Headers and footers are dropped (``include_headers_and_footers`` is off).
  Embedded pictures are OCR'd when tesseract is on PATH.
* odt: body, tables, headers and footers. Embedded pictures are OCR'd when tesseract is on PATH, including the
  thumbnail LibreOffice stores in every ODF file.
* xlsx: the sheet name, then one line per row with tab-separated cells.
* html, txt, json, csv: the text as written; json and csv keep their markup characters.
* PDF with a text layer: the text layer through PDFBox. Pictures on such pages are not OCR'd. Arabic from
  LibreOffice exports comes out in visual order with broken ligatures.
* PDF without a text layer, jpg, png: Tika runs tesseract (language ``eng``, 300 dpi) when the ``tesseract``
  executable is on PATH and returns no text otherwise.
* Text beyond 500,000 characters is cut off.

Set ``tesseract = false`` to reproduce a machine without tesseract: extraction then runs in a child process whose
PATH has no tesseract.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import subprocess
import weakref
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from importlib.metadata import version
from pathlib import Path
from typing import Any, ClassVar

import extractous

from .base import ExtractionError, Extractor

DEFAULTS: dict[str, Any] = {
    "string_max_length": 500_000,
    "encoding": "UTF-8",
    "pdf_ocr_strategy": "AUTO",
    "pdf_extract_inline_images": False,
    "pdf_extract_annotation_text": True,
    "office_include_headers_and_footers": False,
    "office_include_shape_based_content": True,
    "ocr_language": "eng",
    "ocr_density": 300,
}


@cache
def _library() -> extractous.Extractor:
    return extractous.Extractor()


def _load_library() -> None:
    _library()


def _extract_text(path: str) -> str:
    try:
        text, _metadata = _library().extract_file_to_string(path)
    except (TypeError, OSError) as exc:  # extractous reports parse and I/O errors as TypeError
        raise ExtractionError(f"extractous failed on {path}: {exc}") from exc
    return str(text)


def _path_without_tesseract() -> str:
    entries = os.environ.get("PATH", "").split(os.pathsep)
    return os.pathsep.join(entry for entry in entries if not shutil.which("tesseract", path=entry))


def _hide_tesseract() -> None:
    os.environ["PATH"] = _path_without_tesseract()


def _tesseract_version(executable: str) -> str:
    output = subprocess.run([executable, "--version"], check=True, capture_output=True, text=True).stdout
    return output.splitlines()[0]


class ExtractousExtractor(Extractor):
    """Text from any file type through extractous ``Extractor()`` defaults."""

    kind: ClassVar[str] = "extractous"

    def __init__(self, tesseract: bool = True) -> None:
        self.tesseract = tesseract
        self._pool: ProcessPoolExecutor | None = None

    def prepare(self) -> None:
        if self.tesseract:
            _load_library()
        elif self._pool is None:
            # Tika fails once PATH changes after its first tesseract lookup, so a fresh process starts without it.
            context = multiprocessing.get_context("spawn")
            self._pool = ProcessPoolExecutor(max_workers=1, mp_context=context, initializer=_hide_tesseract)
            weakref.finalize(self, self._pool.shutdown)
            self._pool.submit(_load_library).result()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def info(self) -> dict[str, Any]:
        tesseract_path = shutil.which("tesseract") if self.tesseract else None
        return {
            "kind": self.kind,
            "extractous": version("extractous"),
            "tesseract_on_path": tesseract_path is not None,
            "tesseract": _tesseract_version(tesseract_path) if tesseract_path else None,
            "defaults": DEFAULTS,
        }

    def extract(self, path: Path) -> str:
        self.prepare()
        if self._pool is None:
            return _extract_text(str(path))
        return self._pool.submit(_extract_text, str(path)).result()
