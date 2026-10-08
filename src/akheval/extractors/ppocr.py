"""PaddleOCR models on MNN through the ``akh-ocr`` binary (rust/ocr), the engine the Core uses for images."""

from __future__ import annotations

import itertools
import json
import subprocess
import tempfile
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, ClassVar

from bidi import get_display

from ..dataset import sha256_file
from .base import ExtractionError, Extractor
from .pdf_ocr import RASTER_DPI, is_pdf, rasterized_pages

ORDERS = ("engine", "reading")
ARABIC_BLOCKS = (
    ("\u0600", "\u06ff"),
    ("\u0750", "\u077f"),
    ("\u08a0", "\u08ff"),
    ("\ufb50", "\ufdff"),
    ("\ufe70", "\ufeff"),
)
STDERR_TAIL_CHARS = 2000
CLOSE_TIMEOUT_SECONDS = 10


def _is_arabic(char: str) -> bool:
    return any(low <= char <= high for low, high in ARABIC_BLOCKS)


def visual_to_logical(line: str) -> str:
    """Return a recognised line in logical order.

    The PaddleOCR Arabic recognisers emit characters left to right as they appear on the page. Running the Unicode
    bidi algorithm over that visual string restores logical order, with Latin words and digit groups kept intact.
    A right-to-left paragraph ends on the page with its first letter, so the base direction is right to left when
    the rightmost letter is Arabic.
    """
    letters = [char for char in line if char.isalpha()]
    base = "R" if letters and _is_arabic(letters[-1]) else "L"
    return str(get_display(line, base_dir=base))


@dataclass(frozen=True, slots=True)
class OcrLine:
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class OcrResult:
    """One response line of akh-ocr."""

    id: str
    text: str
    lines: list[OcrLine]
    seconds: float
    error: str | None

    @classmethod
    def from_json(cls, raw: str) -> OcrResult:
        data = json.loads(raw)
        lines = [OcrLine(str(line["text"]), float(line["score"])) for line in data["lines"]]
        return cls(str(data["id"]), str(data["text"]), lines, float(data["seconds"]), data.get("error"))


class OcrProcessError(ExtractionError):
    """akh-ocr exited or wrote something outside the protocol."""


class OcrProcess:
    """A running akh-ocr. Requests go one at a time, so the pipes never fill up."""

    def __init__(self, command: Sequence[str]) -> None:
        self._stderr: IO[bytes] = tempfile.TemporaryFile()  # noqa: SIM115 - closed in close()
        self._proc = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            encoding="utf-8",
        )
        self._ids = itertools.count()
        try:
            ready = self._read_line()
            if not json.loads(ready).get("ready"):
                raise OcrProcessError(f"akh-ocr did not report ready: {ready!r}")
        except (OcrProcessError, ValueError, AttributeError):
            self.close()
            raise

    def _stderr_tail(self) -> str:
        self._stderr.seek(0)
        return self._stderr.read().decode(errors="replace")[-STDERR_TAIL_CHARS:].strip()

    def _read_line(self) -> str:
        assert self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            code = self._proc.wait()
            raise OcrProcessError(f"akh-ocr exited with code {code}: {self._stderr_tail()}")
        return line

    def read(self, image: Path) -> OcrResult:
        assert self._proc.stdin is not None
        request_id = str(next(self._ids))
        try:
            self._proc.stdin.write(json.dumps({"id": request_id, "path": str(image)}) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise OcrProcessError(f"akh-ocr is not running: {self._stderr_tail()}") from exc
        raw = self._read_line()
        try:
            result = OcrResult.from_json(raw)
        except (ValueError, KeyError, TypeError) as exc:
            raise OcrProcessError(f"unreadable akh-ocr response {raw!r}: {exc}") from exc
        if result.id != request_id:
            raise OcrProcessError(f"akh-ocr answered request {result.id!r}, expected {request_id!r}")
        return result

    def close(self) -> None:
        if self._proc.stdin is not None and not self._proc.stdin.closed:
            self._proc.stdin.close()
        try:
            self._proc.wait(timeout=CLOSE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        if self._proc.stdout is not None:
            self._proc.stdout.close()
        self._stderr.close()


def _as_paths(value: str | Sequence[str]) -> list[Path]:
    return [Path(value)] if isinstance(value, str) else [Path(item) for item in value]


class PaddleOcrExtractor(Extractor):
    """OCR with akh-ocr. One process serves every file; PDF pages are rasterised at 200 dpi first.

    ``rec`` and ``keys`` take one path or a list of paths. With several pairs, akh-ocr detects boxes once and keeps
    the recognition with the higher score per box. ``fix_rtl`` converts each recognised line from visual to logical
    order; set it for the PaddleOCR Arabic recognisers.
    """

    kind: ClassVar[str] = "ppocr"

    def __init__(
        self,
        binary: str,
        det: str,
        rec: str | Sequence[str],
        keys: str | Sequence[str],
        order: str = "engine",
        fix_rtl: bool = False,
    ) -> None:
        self.binary = Path(binary)
        self.det = Path(det)
        self.recs = _as_paths(rec)
        self.keys = _as_paths(keys)
        if len(self.recs) != len(self.keys):
            raise ValueError(f"got {len(self.recs)} recognisers and {len(self.keys)} key files; pass them in pairs")
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
        self.order = order
        self.fix_rtl = fix_rtl
        self._process: OcrProcess | None = None

    def command(self) -> list[str]:
        command = [str(self.binary), "--det", str(self.det), "--order", self.order]
        for rec, keys in zip(self.recs, self.keys, strict=True):
            command += ["--rec", str(rec), "--keys", str(keys)]
        return command

    def prepare(self) -> None:
        if self._process is not None:
            return
        if not self.binary.is_file():
            raise ExtractionError(f"{self.binary} not found; build it with rust/ocr/build.sh")
        self._process = OcrProcess(self.command())
        weakref.finalize(self, self._process.close)

    def close(self) -> None:
        if self._process is not None:
            self._process.close()
            self._process = None

    def info(self) -> dict[str, Any]:
        version = subprocess.run([*self.command(), "--version"], check=True, capture_output=True, text=True).stdout
        models = [self.det, *self.recs, *self.keys]
        return {
            "kind": self.kind,
            "binary": str(self.binary),
            "version": json.loads(version),
            "order": self.order,
            "fix_rtl": self.fix_rtl,
            "pdf_dpi": RASTER_DPI,
            "models": {str(path): sha256_file(path) for path in models},
        }

    def extract(self, path: Path) -> str:
        if not is_pdf(path):
            return self._read(path)
        with rasterized_pages(path) as pages:
            return "".join(self._read(page) for page in pages)

    def _read(self, image: Path) -> str:
        self.prepare()
        assert self._process is not None
        try:
            result = self._process.read(image)
        except OcrProcessError:
            self.close()
            raise
        if result.error:
            raise ExtractionError(result.error)
        if not self.fix_rtl:
            return result.text
        return "".join(visual_to_logical(line.text) + "\n" for line in result.lines)
