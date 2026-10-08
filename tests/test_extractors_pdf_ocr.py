from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from akheval.extractors.base import ExtractionError, Extractor
from akheval.extractors.pdf_ocr import (
    PdfOcrFallbackExtractor,
    is_image,
    is_pdf,
    rasterized_pages,
    visible_chars,
)


class FakeExtractor(Extractor):
    kind = "fake"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[Path] = []
        self.prepared = False

    def prepare(self) -> None:
        self.prepared = True

    def info(self) -> dict[str, Any]:
        return {"kind": self.kind}

    def extract(self, path: Path) -> str:
        self.calls.append(path)
        return self.text


def write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    return {
        "pdf": write(tmp_path / "doc.pdf", b"%PDF-1.7\n%rest"),
        "png": write(tmp_path / "scan.png", b"\x89PNG\r\n\x1a\n" + bytes(8)),
        "jpg": write(tmp_path / "photo.bin", b"\xff\xd8\xff\xe0" + bytes(12)),
        "webp": write(tmp_path / "photo.webp", b"RIFF\x00\x00\x00\x00WEBPVP8 "),
        "docx": write(tmp_path / "letter.docx", b"PK\x03\x04" + bytes(12)),
    }


def test_sniffs_by_magic_bytes(files: dict[str, Path]) -> None:
    assert is_pdf(files["pdf"])
    assert not is_pdf(files["png"])
    assert all(is_image(files[name]) for name in ("png", "jpg", "webp"))
    assert not is_image(files["docx"])
    assert not is_image(files["pdf"])


def test_sniffing_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError):
        is_pdf(tmp_path / "absent.pdf")


def test_visible_chars_ignores_whitespace() -> None:
    assert visible_chars(" a\tb\n\u00a0c ") == 3


def test_image_goes_straight_to_ocr(files: dict[str, Path]) -> None:
    text, ocr = FakeExtractor("ignored"), FakeExtractor("ocr text")
    assert PdfOcrFallbackExtractor(text, ocr).extract(files["png"]) == "ocr text"
    assert text.calls == []


def test_pdf_with_short_text_layer_falls_back_to_ocr(files: dict[str, Path]) -> None:
    ocr = FakeExtractor("ocr text")
    extractor = PdfOcrFallbackExtractor(FakeExtractor(" page 1 \n"), ocr, min_chars=20)
    assert extractor.extract(files["pdf"]) == "ocr text"
    assert ocr.calls == [files["pdf"]]


def test_pdf_with_text_layer_keeps_it(files: dict[str, Path]) -> None:
    ocr = FakeExtractor("ocr text")
    extractor = PdfOcrFallbackExtractor(FakeExtractor("x" * 20), ocr, min_chars=20)
    assert extractor.extract(files["pdf"]) == "x" * 20
    assert ocr.calls == []


def test_other_files_never_fall_back(files: dict[str, Path]) -> None:
    ocr = FakeExtractor("ocr text")
    assert PdfOcrFallbackExtractor(FakeExtractor(""), ocr).extract(files["docx"]) == ""
    assert ocr.calls == []


def test_from_config_resolves_named_extractors() -> None:
    named = {"tika": FakeExtractor("t"), "paddle": FakeExtractor("o")}
    extractor = PdfOcrFallbackExtractor.from_config(
        {"text": "tika", "ocr": "paddle", "min_chars": 5}, named.__getitem__
    )
    extractor.prepare()
    assert extractor.text is named["tika"]
    assert extractor.ocr is named["paddle"]
    assert extractor.min_chars == 5
    assert all(component.prepared for component in named.values())
    assert extractor.info()["ocr"] == {"kind": "fake"}


@pytest.mark.integration
def test_rasterized_pages_renders_each_page_in_order(tmp_path: Path) -> None:
    if shutil.which("pdftoppm") is None:
        pytest.skip("pdftoppm is not installed")
    pdf = tmp_path / "two.pdf"
    Image.init()  # registers the JPEG encoder the PDF writer uses
    first, second = Image.new("RGB", (200, 100), "white"), Image.new("RGB", (300, 100), "white")
    first.save(pdf, save_all=True, append_images=[second], resolution=100)
    with rasterized_pages(pdf) as pages:
        assert [page.name for page in pages] == ["page-1.png", "page-2.png"]
        widths = [Image.open(page).width for page in pages]
        assert widths == [400, 600]
    assert not any(page.exists() for page in pages)


@pytest.mark.integration
def test_rasterized_pages_reports_a_broken_pdf(tmp_path: Path) -> None:
    if shutil.which("pdftoppm") is None:
        pytest.skip("pdftoppm is not installed")
    broken = write(tmp_path / "broken.pdf", b"%PDF-1.7\nnot a pdf")
    with pytest.raises(ExtractionError, match="pdftoppm failed"), rasterized_pages(broken):
        pass
