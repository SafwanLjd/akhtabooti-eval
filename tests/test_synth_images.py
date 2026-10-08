"""Tests for the raster renderer of synthetic documents (akheval.synth.images)."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from akheval.synth.builder import BLOCK_KINDS, SIDES
from akheval.synth.images import CAPTURES_BY_STYLE, make_contact_sheet, render_image
from akheval.synth.images.camera import EASIER_FIELDS as PHOTO_EASIER
from akheval.synth.images.camera import PHOTO
from akheval.synth.images.flow import parse_layout
from akheval.synth.images.fonts import SANS_FAMILIES, choose_fonts
from akheval.synth.images.scan import EASIER_FIELDS as SCAN_EASIER
from akheval.synth.images.scan import SCAN
from akheval.synth.images.shapes import rotation, text_mask
from akheval.synth.images.text import Styled, TextStyle, Typesetter, direction_runs, paragraph_direction

Layout = list[dict[str, str]]


def _blocks(*items: tuple[str, str] | tuple[str, str, str]) -> Layout:
    out = []
    for item in items:
        block = {"kind": item[0], "text": item[1]}
        if len(item) == 3:
            block["side"] = item[2]
        out.append(block)
    return out


# Sample layouts with invented people, numbers and issuers.
ID_CARD: dict[str, Layout] = {
    "ar": _blocks(
        ("header", "سجل الأحوال المدنية"),
        ("title", "بطاقة شخصية"),
        ("field", "الاسم: ليث سامي محمود العمري"),
        ("field", "الرقم الوطني: 9861234567"),
        ("field", "تاريخ الولادة: 14/03/1986"),
        ("field", "مكان الولادة: إربد"),
        ("field", "اسم الأم: سلمى"),
        ("field", "العنوان: عمّان، جبل الحسين، شارع خالد بن الوليد، عمارة 12", "back"),
        ("field", "تاريخ الانتهاء: 22/09/2031", "back"),
        ("field", "رقم البطاقة: ZXK48213", "back"),
        ("mrz", "IDJORZXK4821397986123456<<<<<<", "back"),
        ("mrz", "8603145M3109221JOR<<<<<<<<<<<6", "back"),
        ("mrz", "ALOMARI<<LAITH<SAMI<<<<<<<<<<<", "back"),
    ),
    "en": _blocks(
        ("header", "CIVIL REGISTRY"),
        ("title", "IDENTITY CARD"),
        ("field", "Name: Laith Sami Al-Omari"),
        ("field", "National No.: 9861234567"),
        ("field", "Date of Birth: 14/03/1986"),
        ("field", "Place of Birth: Irbid"),
        ("field", "Expiry: 22/09/2031"),
    ),
    "mixed": _blocks(
        ("header", "سجل الأحوال المدنية"),
        ("header", "CIVIL REGISTRY"),
        ("title", "بطاقة شخصية / IDENTITY CARD"),
        ("field", "الاسم: ريم خالد الحسن"),
        ("field", "Name: Reem Khaled Al-Hasan"),
        ("field", "الرقم الوطني / National No.: 2000456789"),
        ("field", "تاريخ الولادة / DOB: 02/11/1999"),
        ("field", "العنوان / Address: Amman, Tla' Al-Ali, Street 15", "back"),
        ("field", "Phone / الهاتف: +962 79 555 0142", "back"),
        ("mrz", "IDJORQBT5528104200045678<<<<<<", "back"),
        ("mrz", "9911023F3011027JOR<<<<<<<<<<<2", "back"),
        ("mrz", "ALHASAN<<REEM<KHALED<<<<<<<<<<", "back"),
    ),
}

PASSPORT: dict[str, Layout] = {
    "en": _blocks(
        ("header", "TRAVEL DOCUMENT AUTHORITY"),
        ("title", "PASSPORT"),
        ("field", "Type: P"),
        ("field", "Country code: JOR"),
        ("field", "Passport No.: N123456"),
        ("field", "Surname: AL-OMARI"),
        ("field", "Given names: LAITH SAMI"),
        ("field", "Nationality: JORDANIAN"),
        ("field", "Date of birth: 14 MAR 1986"),
        ("field", "Sex: M"),
        ("field", "Place of birth: IRBID"),
        ("field", "Date of expiry: 04 JAN 2027"),
        ("mrz", "P<JORAL<OMARI<<LAITH<SAMI<<<<<<<<<<<<<<<<<<<"),
        ("mrz", "N123456<<4JOR8603145M2701045<<<<<<<<<<<<<<02"),
    ),
    "ar": _blocks(
        ("header", "هيئة وثائق السفر"),
        ("title", "جواز سفر"),
        ("field", "النوع: P"),
        ("field", "رقم الجواز: R7741029"),
        ("field", "الاسم: هدى صالح يوسف"),
        ("field", "الجنسية: سوري"),
        ("field", "تاريخ الولادة: 21/07/1990"),
        ("field", "مكان الولادة: حمص"),
        ("field", "تاريخ الانتهاء: 30/06/2028"),
        ("mrz", "P<SYRYOUSEF<<HUDA<SALEH<<<<<<<<<<<<<<<<<<<<<"),
        ("mrz", "R7741029<3SYR9007214F2806304<<<<<<<<<<<<<<06"),
    ),
    "mixed": _blocks(
        ("header", "هيئة وثائق السفر / TRAVEL DOCUMENT AUTHORITY"),
        ("title", "جواز سفر / PASSPORT"),
        ("field", "رقم الجواز / Passport No.: N884512"),
        ("field", "الاسم / Name: عمر خليل"),
        ("field", "Name: OMAR KHALIL"),
        ("field", "الجنسية / Nationality: JORDANIAN"),
        ("field", "تاريخ الولادة / Date of birth: 09/12/1978"),
        ("field", "مكان الولادة / Place of birth: الزرقاء"),
        ("field", "تاريخ الإصدار / Date of issue: 11/02/2023"),
        ("field", "تاريخ الانتهاء / Date of expiry: 10/02/2028"),
        ("mrz", "P<JORKHALIL<<OMAR<<<<<<<<<<<<<<<<<<<<<<<<<<<"),
        ("mrz", "N884512<<8JOR7812092M2802104<<<<<<<<<<<<<<04"),
    ),
}

PAGE: dict[str, Layout] = {
    "en": _blocks(
        ("header", "Community Legal Aid Network"),
        ("header", "Case referral service"),
        ("blank", ""),
        ("title", "Referral Letter"),
        ("paragraph", "Date: 03/10/2026"),
        (
            "paragraph",
            "We refer the family below for legal counselling on residency documents. The case worker met the "
            "family on 28/09/2026 and confirmed the details. Please contact Dana Haddad at "
            "dana.haddad@example.org or on +962 79 555 0142 to arrange the first appointment.",
        ),
        ("field", "Case worker: Dana Haddad"),
        ("field", "Reference: REF-2026-00417"),
        ("field", "IBAN: JO71 CBJO 0010 0000 0000 0131 0003 02"),
        ("blank", ""),
        ("row", "Name | National No. | Phone"),
        ("row", "Omar Khalil | 9871002345 | 0791234567"),
        ("row", "Huda Saleh | 2001123456 | +962 78 555 0199"),
        ("blank", ""),
        ("paragraph", "Kind regards,"),
        ("paragraph", "Dana Haddad"),
    ),
    "ar": _blocks(
        ("header", "شبكة المساعدة القانونية المجتمعية"),
        ("blank", ""),
        ("title", "كتاب إحالة"),
        ("paragraph", "التاريخ: ٠٣/١٠/٢٠٢٦"),  # noqa: RUF001
        (
            "paragraph",
            "نحيل إليكم العائلة المذكورة أدناه للحصول على استشارة قانونية بخصوص وثائق الإقامة. وقد التقت "
            "الباحثة الاجتماعية بالعائلة وتأكدت من البيانات. يرجى التواصل مع السيدة دانا حداد على الرقم "
            "0795550142 لترتيب الموعد الأول.",
        ),
        ("field", "الاسم: محمود أحمد الخطيب"),
        ("field", "الرقم الوطني: ٩٨٧١٠٠٢٣٤٥"),
        ("field", "العنوان: الزرقاء، حي معصوم، شارع الملك طلال"),
        ("blank", ""),
        ("row", "الاسم | الرقم الوطني | الهاتف"),
        ("row", "عمر خليل | 9871002345 | 0791234567"),
        ("row", "هدى صالح | 2001123456 | 0785550199"),
        ("blank", ""),
        ("paragraph", "وتفضلوا بقبول فائق الاحترام،"),
    ),
    "mixed": _blocks(
        ("header", "Community Legal Aid Network / شبكة المساعدة القانونية"),
        ("title", "نموذج تسجيل حالة"),
        ("field", "اسم المستفيد: سارة يوسف"),
        ("field", "Email: sara.yousef@example.net"),
        ("field", "رقم الهاتف: +962 77 123 4567"),
        ("field", "IBAN: JO94 ARAB 1180 0000 0011 8000 1234 56"),
        (
            "paragraph",
            "تم استلام الطلب من خلال البريد الإلكتروني intake@example.org بتاريخ 12/09/2026، ورقم المرجع "
            "REF-2026-0912 وليس رقم هوية.",
        ),
        ("row", "الوثيقة | الرقم | Notes"),
        ("row", "Passport | N884512 | valid until 2028"),
        ("row", "بطاقة لاجئ | 345-22C01234 | UNHCR"),
    ),
}

CHAT: dict[str, Layout] = {
    "en": _blocks(
        ("header", "Today"),
        ("message", "[10:02] Sami: hey, send me the contract to sami.k@example.com"),
        ("message", "[10:03] Rana: sure, and my new number is +962 79 555 0142 by the way"),
        ("message", "[10:04] Sami: cool, see you tomorrow at Bright Path Training at 9"),
        ("message", "[10:05] Rana: ok, the invoice number is INV-2026-00871 btw"),
    ),
    "ar": _blocks(
        ("title", "سامي"),
        ("header", "اليوم"),
        ("message", "[10:02] سامي: أهلاً، ابعتلي العقد على sami.k@example.com"),
        ("message", "[10:03] رنا: تمام، ورقمي الجديد 0795550142 بالمناسبة"),
        ("message", "[10:04] سامي: خلص، بشوفك بكرا في مركز الأمل الساعة 9"),
        ("message", "[10:05] رنا: رقم الفاتورة INV-2026-00871 للعلم"),
    ),
    "mixed": _blocks(
        ("message", "[09:41] Lina: hey, send me the contract to lina.h@example.org"),
        ("message", "[09:42] كريم: تمام، ورقمي الجديد +962 78 555 0199 بالمناسبة"),
        ("message", "[09:43] Lina: خلص، بشوفك بكرا في Amman office الساعة 9"),
        ("blank", ""),
        ("message", "[09:45] كريم: ok, the invoice number is INV-2026-00912 btw"),
    ),
}

LAYOUTS: dict[str, dict[str, Layout]] = {"id_card": ID_CARD, "passport": PASSPORT, "page": PAGE, "chat": CHAT}
SEVERITY_LEVELS = ("mild", "moderate", "harsh")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render(
    tmp_path: Path, style: str, language: str, capture: str, severity: str | None, seed: int, suffix: str
) -> tuple[Path, dict[str, Any]]:
    out = tmp_path / f"{style}-{language}-{capture}-{severity}-{seed}{suffix}"
    params = render_image(
        LAYOUTS[style][language],
        style=style,
        language=language,
        capture=capture,
        severity=severity,
        rng=random.Random(seed),
        out=out,
    )
    return out, params


def _cases() -> list[tuple[str, str, str, str | None]]:
    """Every style, capture and severity once; the language rotates so each style and capture meets all three."""
    cases: list[tuple[str, str, str, str | None]] = []
    for style, layouts in LAYOUTS.items():
        languages = sorted(layouts)
        for c, capture in enumerate(CAPTURES_BY_STYLE[style]):
            if capture == "digital":
                cases += [(style, language, capture, None) for language in languages]
            else:
                cases += [
                    (style, languages[(c + s) % len(languages)], capture, severity)
                    for s, severity in enumerate(SEVERITY_LEVELS)
                ]
    return cases


@pytest.mark.parametrize(("style", "language", "capture", "severity"), _cases())
def test_every_style_capture_and_severity_renders(
    tmp_path: Path, style: str, language: str, capture: str, severity: str | None
) -> None:
    suffix = ".png" if capture == "digital" else ".jpg"
    out, params = _render(tmp_path, style, language, capture, severity, 7, suffix)
    with Image.open(out) as image:
        image.load()
        assert min(image.size) >= 400
    assert json.loads(json.dumps(params)) == params
    assert params["style"] == style and params["capture"] == capture and params["severity"] == severity
    assert not any("missing_glyphs" in block for block in params["document"].get("blocks", []))


@pytest.mark.parametrize(
    ("style", "capture", "severity"),
    [
        ("id_card", "photo", "moderate"),
        ("id_card", "scan", "mild"),
        ("passport", "photo", "harsh"),
        ("page", "photo", "moderate"),
        ("page", "scan", "harsh"),
        ("chat", "digital", None),
    ],
)
def test_same_rng_state_gives_identical_bytes(tmp_path: Path, style: str, capture: str, severity: str | None) -> None:
    first, params_a = _render(tmp_path / "a", style, "mixed", capture, severity, 11, ".png")
    second, params_b = _render(tmp_path / "b", style, "mixed", capture, severity, 11, ".png")
    assert _hash(first) == _hash(second)
    assert params_a == params_b
    third, _ = _render(tmp_path / "c", style, "mixed", capture, severity, 12, ".png")
    assert _hash(first) != _hash(third)


@pytest.mark.integration
def test_pdf_output_is_one_image_page_without_text(tmp_path: Path) -> None:
    out, params = _render(tmp_path, "page", "ar", "scan", "mild", 3, ".pdf")
    assert out.read_bytes().startswith(b"%PDF")
    assert params["output"]["format"] == "pdf"
    pdfinfo, pdftotext = shutil.which("pdfinfo"), shutil.which("pdftotext")
    if pdfinfo is None or pdftotext is None:
        pytest.skip("poppler tools are not installed")
    info = subprocess.run([pdfinfo, str(out)], capture_output=True, text=True, check=True).stdout
    assert any(line.split() == ["Pages:", "1"] for line in info.splitlines())
    text = subprocess.run([pdftotext, str(out), "-"], capture_output=True, text=True, check=True).stdout
    assert text.strip() == ""


def test_pdf_output_is_deterministic(tmp_path: Path) -> None:
    first, _ = _render(tmp_path / "a", "passport", "en", "photo", "mild", 5, ".pdf")
    second, _ = _render(tmp_path / "b", "passport", "en", "photo", "mild", 5, ".pdf")
    assert _hash(first) == _hash(second)


@pytest.mark.parametrize(("profiles", "easier"), [(PHOTO, PHOTO_EASIER), (SCAN, SCAN_EASIER)])
def test_severity_ranges_grow_monotonically(profiles: Mapping[str, Any], easier: frozenset[str]) -> None:
    mild, moderate, harsh = (profiles[s] for s in SEVERITY_LEVELS)
    for field in fields(mild):
        values = [getattr(p, field.name) for p in (mild, moderate, harsh)]
        if field.name in easier:
            values.reverse()
        flat = [v if isinstance(v, tuple) else (v, v) for v in values]
        for low, high in itertools.pairwise(flat):
            assert low[0] <= high[0] and low[1] <= high[1], field.name


def test_parse_layout_rejects_bad_blocks() -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        parse_layout([{"kind": "caption", "text": "x"}])
    with pytest.raises(ValueError, match="single line"):
        parse_layout([{"kind": "paragraph", "text": "a\nb"}])
    with pytest.raises(ValueError, match="blank"):
        parse_layout([{"kind": "blank", "text": "x"}])


def test_parse_layout_accepts_every_builder_kind_and_side() -> None:
    raw = [
        {"kind": kind, "text": "" if kind == "blank" else "x", "side": side} for kind in BLOCK_KINDS for side in SIDES
    ]
    assert [(b.kind, b.side) for b in parse_layout(raw)] == [(r["kind"], r["side"]) for r in raw]


@pytest.mark.parametrize(
    ("style", "capture", "severity"), [("chat", "photo", "mild"), ("page", "digital", "mild"), ("page", "scan", None)]
)
def test_render_image_rejects_unsupported_combinations(
    tmp_path: Path, style: str, capture: str, severity: str | None
) -> None:
    with pytest.raises(ValueError):
        render_image(
            LAYOUTS[style]["en"],
            style=style,
            language="en",
            capture=capture,
            severity=severity,
            rng=random.Random(0),
            out=tmp_path / "x.png",
        )


def test_render_image_rejects_unknown_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="suffix"):
        render_image(
            PAGE["en"],
            style="page",
            language="en",
            capture="digital",
            severity=None,
            rng=random.Random(0),
            out=tmp_path / "x.tiff",
        )


def test_rotation_turns_column_vectors_counterclockwise() -> None:
    assert np.allclose(rotation(math.pi / 2) @ np.array([1.0, 0.0]), [0.0, 1.0])


def test_text_mask_fills_padded_boxes_at_its_scale() -> None:
    box = (10.0, 4.0, 20.0, 8.0)
    full = text_mask((40, 20), [box], pad=2.0)
    assert full.shape == (20, 40) and full.sum() == 15 * 9 and full[2:11, 8:23].all()
    small = text_mask((40, 20), [box], pad=2.0, scale=0.5)
    assert small.shape == (10, 20) and small.sum() == 8 * 5 and small[1:6, 4:12].all()


def test_paragraph_direction_uses_first_strong_letter() -> None:
    assert paragraph_direction("[10:03] رنا: ok") == "rtl"
    assert paragraph_direction("123 Name: سامي") == "ltr"
    assert paragraph_direction("12/03/2026") == "ltr"


def test_numbers_inside_arabic_stay_in_one_left_to_right_run() -> None:
    text = "الهاتف: +962 79 555 0142 فقط"
    runs = direction_runs(text, "rtl")
    ltr = [text[a:b] for a, b, direction in runs if direction == "ltr"]
    assert ltr == ["+962 79 555 0142"]
    assert "".join(text[a:b] for a, b, _ in runs) == text


def test_bracketed_timestamp_joins_its_run() -> None:
    text = "[10:03] رنا: تمام"
    ltr = [text[a:b] for a, b, direction in direction_runs(text, "rtl") if direction == "ltr"]
    assert ltr == ["[10:03]"]


def test_wrapping_keeps_every_word() -> None:
    ts = Typesetter(choose_fonts(random.Random(1), SANS_FAMILIES))
    text = "تم استلام الطلب من خلال البريد الإلكتروني intake@example.org بتاريخ 12/09/2026 ورقم المرجع REF-2026"
    styled = Styled.plain(text, TextStyle(40))
    lines = ts.fit(styled, 500)
    assert len(lines) > 1
    assert " ".join(line.text for line in lines) == text
    assert all(line.width <= 500 for line in lines)


def test_shrink_to_keeps_one_line_within_width() -> None:
    ts = Typesetter(choose_fonts(random.Random(3), SANS_FAMILIES))
    text = "مجموعة العائلة الكبيرة وأصدقاء الحي القديم"
    line = ts.shrink_to(Styled.plain(text, TextStyle(60)), 300)
    assert line.text == text
    assert line.width <= 300
    assert line.size < 60


def test_layout_draws_every_character_once_with_a_covering_font() -> None:
    ts = Typesetter(choose_fonts(random.Random(2), SANS_FAMILIES))
    line = ts.layout(Styled.plain("الاسم / Name: ريم (Reem) ٢٠٠٠٤٥٦٧٨٩", TextStyle(30)), "rtl")
    assert line.missing == ""
    assert sorted("".join(piece.text for piece in line.pieces)) == sorted(line.text)


@pytest.mark.integration
def test_contact_sheet_tiles_images_and_pdfs(tmp_path: Path) -> None:
    png, _ = _render(tmp_path, "chat", "en", "digital", None, 1, ".png")
    pdf, _ = _render(tmp_path, "id_card", "en", "scan", "mild", 1, ".pdf")
    sheet = tmp_path / "sheet.png"
    if shutil.which("pdftoppm") is None:
        pytest.skip("pdftoppm is not installed")
    make_contact_sheet([png, pdf], sheet, columns=2, thumb_width=200)
    with Image.open(sheet) as image:
        assert image.width == 2 * 212 + 12


@pytest.mark.integration
@pytest.mark.parametrize(("style", "language"), [("page", "en"), ("page", "ar"), ("id_card", "en"), ("passport", "en")])
def test_tesseract_reads_most_words_of_mild_renders(tmp_path: Path, style: str, language: str) -> None:
    pytesseract = pytest.importorskip("pytesseract")
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract is not installed")
    recall = {}
    for severity in ("mild", "harsh"):
        out, _ = _render(tmp_path, style, language, "scan", severity, 21, ".png")
        with Image.open(out) as image:
            text = pytesseract.image_to_string(image, lang="ara+eng")
        recall[severity] = word_recall(LAYOUTS[style][language], text)
    assert recall["mild"] >= 0.6
    assert recall["harsh"] <= recall["mild"] + 0.05


def word_recall(layout: Sequence[Mapping[str, str]], ocr_text: str) -> float:
    """Share of the layout's words of three or more characters that the OCR output contains.

    Words are compared without surrounding punctuation and without Arabic diacritics.
    """
    expected = [w for block in layout for w in map(_normalise, block["text"].split()) if len(w) >= 3]
    found = {_normalise(w) for w in ocr_text.split()}
    return sum(w in found for w in expected) / max(len(expected), 1)


def _normalise(word: str) -> str:
    return re.sub("[\u064b-\u0652\u0640]", "", word).strip(".,:;،؛|()[]/")
