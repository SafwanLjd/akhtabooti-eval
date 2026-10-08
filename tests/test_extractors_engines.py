"""Integration tests that run the real OCR and extraction engines on generated files."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont, features

from akheval.extractors.pdf_ocr import PdfOcrFallbackExtractor
from akheval.extractors.ppocr import PaddleOcrExtractor

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "ppocr"
AKH_OCR = ROOT / "rust" / "ocr" / "bin" / "akh-ocr"
LATIN_FONT = Path("/usr/share/fonts/liberation/LiberationSans-Regular.ttf")
ARABIC_FONT = Path("/usr/share/fonts/TTF/Amiri-Regular.ttf")
FONT_SIZE = 36
EMAIL = "sara.k@example.org"
ENGLISH = f"Contact: {EMAIL} phone 0791234567"
# A form row: the label reads right to left and the phone number sits after a wide gap.
ARABIC_ROW = "رقم الهاتف:" + "\u00a0" * 8 + "+962 79 555 1234"


def render(path: Path, text: str, font_path: Path, direction: str = "ltr") -> Path:
    if not font_path.exists():
        pytest.skip(f"font {font_path} is not installed")
    if direction == "rtl" and not features.check("raqm"):
        pytest.skip("Pillow was built without raqm")
    font = ImageFont.truetype(str(font_path), FONT_SIZE, layout_engine=ImageFont.Layout.RAQM)
    image = Image.new("RGB", (1000, 120), "white")
    anchor, x = ("ra", 980) if direction == "rtl" else ("la", 20)
    ImageDraw.Draw(image).text((x, 40), text, font=font, fill="black", direction=direction, anchor=anchor)
    image.save(path)
    return path


def image_pdf(path: Path, image: Path) -> Path:
    Image.init()  # registers the JPEG encoder the PDF writer uses
    Image.open(image).save(path, resolution=100)
    return path


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        pytest.skip(f"{name} is not installed")


def paddle(
    rec: list[str], keys: list[str], det: str = "PP-OCRv5_mobile_det.mnn", order: str = "engine", fix_rtl: bool = False
) -> PaddleOcrExtractor:
    paths = [MODELS / name for name in (det, *rec, *keys)]
    if not AKH_OCR.exists() or not all(path.exists() for path in paths):
        pytest.skip(
            "akh-ocr or the PaddleOCR models are missing; run rust/ocr/build.sh and fetch configs/models/ocr.toml"
        )
    recs, key_files = [str(MODELS / name) for name in rec], [str(MODELS / name) for name in keys]
    return PaddleOcrExtractor(str(AKH_OCR), str(MODELS / det), recs, key_files, order=order, fix_rtl=fix_rtl)


def core_main_ocr() -> PaddleOcrExtractor:
    return paddle(["PP-OCRv6_small_rec.mnn"], ["ppocr_keys_v6_small.txt"], det="PP-OCRv6_small_det.mnn")


def test_tesseract_reads_an_image_and_a_scanned_pdf(tmp_path: Path) -> None:
    require_tool("tesseract")
    require_tool("pdftoppm")
    from akheval.extractors.tesseract import TesseractExtractor

    png = render(tmp_path / "line.png", ENGLISH, LATIN_FONT)
    extractor = TesseractExtractor()
    assert EMAIL in extractor.extract(png)
    assert EMAIL in extractor.extract(image_pdf(tmp_path / "scan.pdf", png))
    assert extractor.info()["lang"] == "eng"


def test_ppocr_core_defaults_read_english(tmp_path: Path) -> None:
    extractor = core_main_ocr()
    text = extractor.extract(render(tmp_path / "line.png", ENGLISH, LATIN_FONT))
    assert EMAIL in text
    assert text.endswith("\n")
    assert extractor.info()["version"]["ocr_rs"] == "2.4.1"


def test_ppocr_arabic_with_fix_rtl_returns_logical_order(tmp_path: Path) -> None:
    extractor = paddle(["arabic_PP-OCRv5_mobile_rec_infer.mnn"], ["ppocr_keys_arabic.txt"], fix_rtl=True)
    text = extractor.extract(render(tmp_path / "ar.png", "الاسم الكامل: سارة خالد", ARABIC_FONT, "rtl"))
    assert "سارة خالد" in text


def test_ppocr_multi_recogniser_reading_order_restores_the_phone_number(tmp_path: Path) -> None:
    extractor = paddle(
        ["arabic_PP-OCRv5_mobile_rec_infer.mnn", "en_PP-OCRv5_mobile_rec_infer.mnn"],
        ["ppocr_keys_arabic.txt", "ppocr_keys_en.txt"],
        order="reading",
        fix_rtl=True,
    )
    text = extractor.extract(render(tmp_path / "row.png", ARABIC_ROW, ARABIC_FONT, "rtl"))
    assert "+962 79 555 1234" in text
    assert text.startswith("رقم")


def test_extractous_defaults(tmp_path: Path) -> None:
    pytest.importorskip("extractous")
    docx = pytest.importorskip("docx")
    from akheval.extractors.extractous import ExtractousExtractor

    document = docx.Document()
    document.sections[0].header.paragraphs[0].text = "header.person@example.org"
    document.add_paragraph(ENGLISH)
    document.save(str(tmp_path / "letter.docx"))
    (tmp_path / "note.txt").write_text(ENGLISH + "\n", encoding="utf-8")

    extractor = ExtractousExtractor()
    assert extractor.extract(tmp_path / "note.txt").strip() == ENGLISH
    letter = extractor.extract(tmp_path / "letter.docx")
    assert EMAIL in letter
    assert "header.person" not in letter
    assert extractor.info()["defaults"]["ocr_language"] == "eng"


def test_extractous_ocr_depends_on_tesseract_on_path(tmp_path: Path) -> None:
    pytest.importorskip("extractous")
    require_tool("tesseract")
    from akheval.extractors.extractous import ExtractousExtractor

    png = render(tmp_path / "line.png", ENGLISH, LATIN_FONT)
    assert EMAIL in ExtractousExtractor().extract(png)
    without = ExtractousExtractor(tesseract=False)
    assert without.extract(png).strip() == ""
    assert without.info()["tesseract_on_path"] is False


def test_fallback_ocrs_a_pdf_without_text_layer(tmp_path: Path) -> None:
    pytest.importorskip("extractous")
    require_tool("pdftoppm")
    from akheval.extractors.extractous import ExtractousExtractor

    png = render(tmp_path / "line.png", ENGLISH, LATIN_FONT)
    pdf = image_pdf(tmp_path / "scan.pdf", png)
    extractor = PdfOcrFallbackExtractor(ExtractousExtractor(tesseract=False), core_main_ocr())
    assert EMAIL in extractor.extract(pdf)
    assert EMAIL in extractor.extract(png)
