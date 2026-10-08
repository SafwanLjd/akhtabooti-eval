"""Tests for capture realism: card sides, photocopies, thumbs, watermarks, page marks, chat bubbles and framing."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from akheval.synth.images import photo, render_image
from akheval.synth.images.booklet import open_booklet
from akheval.synth.images.camera import PHOTO, finish
from akheval.synth.images.cards import render_id_card
from akheval.synth.images.chat import message_parts, render_chat
from akheval.synth.images.flow import parse_layout
from akheval.synth.images.fonts import SANS_FAMILIES, choose_fonts
from akheval.synth.images.furniture import WATERMARK_ALPHA, watermark
from akheval.synth.images.page import render_page
from akheval.synth.images.photo import CROP_SHARE, Framing, _glare, _place
from akheval.synth.images.scan import SCAN
from akheval.synth.images.sheet import A4_MM, MM_PER_INCH, Face, Plate, arrange, transform_box
from akheval.synth.images.text import Box, Typesetter
from akheval.synth.images.thumb import REACH_MM, Thumb, hold
from test_synth_images import ID_CARD, PAGE, PASSPORT

# Share of photos per severity in the dataset config.
SEVERITY_SHARES = {"mild": 0.4, "moderate": 0.4, "harsh": 0.2}


def _overlaps(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


@pytest.mark.parametrize("sides", [("front",), ("back",), ("front", "back")])
def test_card_draws_only_the_sides_its_layout_has(sides: tuple[str, ...]) -> None:
    blocks = [b for b in parse_layout(ID_CARD["ar"]) if b.side in sides]
    sheet = render_id_card(blocks, random.Random(1))
    assert [face["side"] for face in sheet.params["faces"]] == list(sides)
    assert len(sheet.faces) == len(sides)


def test_two_sided_card_scan_is_a_grey_a4_photocopy(tmp_path: Path) -> None:
    both = render_image(
        ID_CARD["ar"], style="id_card", language="ar", capture="scan", severity="moderate",
        rng=random.Random(4), out=tmp_path / "both.png",
    )  # fmt: skip
    back = [b for b in ID_CARD["ar"] if b.get("side") == "back"]
    one = render_image(
        back, style="id_card", language="ar", capture="scan", severity="moderate",
        rng=random.Random(4), out=tmp_path / "back.png",
    )  # fmt: skip
    assert "photocopy" in both["capture_params"] and both["capture_params"]["grayscale"]
    assert "photocopy" not in one["capture_params"]
    with Image.open(tmp_path / "both.png") as image:
        assert image.mode == "L"
        a4 = [mm * both["capture_params"]["dpi"] / MM_PER_INCH for mm in A4_MM]
        assert all(abs(side - expected) <= 1 for side, expected in zip(image.size, a4, strict=True))


def test_watermark_is_faint_and_never_touches_text() -> None:
    fonts = choose_fonts(random.Random(0), SANS_FAMILIES)
    plain = Image.new("RGB", (900, 560), (238, 240, 236))
    boxes = [(300.0, 40.0 + 70 * k, 860.0, 80.0 + 70 * k) for k in range(5)] + [(40.0, 420.0, 860.0, 520.0)]
    for seed in range(6):
        for prefer in ((40.0, 40.0, 260.0, 320.0), None):
            marked, params = watermark(plain, fonts, random.Random(seed), boxes, prefer)
            changed = np.abs(np.asarray(marked, np.int16) - np.asarray(plain, np.int16)).sum(axis=2) > 0
            assert params["where"] == ("photo" if prefer else "blank")
            assert params["alpha"] <= WATERMARK_ALPHA[1] <= round(0.18 * 255)
            assert changed.any()
            assert not any(changed[round(y0) : round(y1), round(x0) : round(x1)].any() for x0, y0, x1, y1 in boxes)


FRAME = (1500, 2000)
CARD_PX = (1284, 810)


def _held(boxes: list[Box], seed: int, offset: tuple[int, int] = (350, 300)) -> tuple[Thumb | None, np.ndarray]:
    """The thumb holding a card of 85.6 x 54 mm at 15 px/mm placed at ``offset`` in a 2000 x 1500 frame.

    Returns the thumb and its coverage over the whole frame.
    """
    matrix = np.array([[1.0, 0.0, offset[0]], [0.0, 1.0, offset[1]], [0.0, 0.0, 1.0]])
    thumb = hold(CARD_PX, boxes, matrix, FRAME, 15.0, (0.6, -0.8), random.Random(seed), np.random.default_rng(seed))
    coverage = np.zeros(FRAME, np.float32)
    if thumb is not None:
        coverage[thumb.window.y0 : thumb.window.y1, thumb.window.x0 : thumb.window.x1] = thumb.alpha
    return thumb, coverage


def test_thumb_never_covers_a_text_line() -> None:
    boxes = [(45.0, 45.0 + 60 * k, 900.0, 85.0 + 60 * k) for k in range(12)]
    dx, dy = 350, 300
    for seed in range(8):
        thumb, coverage = _held(boxes, seed)
        assert thumb is not None
        assert coverage[dy : dy + CARD_PX[1], dx : dx + CARD_PX[0]].max() > 0.5
        for x0, y0, x1, y1 in boxes:
            assert coverage[round(y0) + dy : round(y1) + dy, round(x0) + dx : round(x1) + dx].max() < 0.01


def test_thumb_enters_from_the_nearest_frame_border_and_leaves_the_hand_outside() -> None:
    boxes = [(45.0, 45.0, 900.0, 85.0)]
    for offset, edge in (((680, 300), "right"), ((350, 640), "bottom"), ((40, 300), "left"), ((350, 30), "top")):
        thumb, coverage = _held(boxes, 3, offset)
        assert thumb is not None
        assert thumb.params["edge"] == edge
        assert thumb.params["reach_mm"] <= REACH_MM
        border = np.concatenate([coverage[0], coverage[-1], coverage[:, 0], coverage[:, -1]])
        assert border.max() > 0.5
        tip = np.array(thumb.params["tip_px"])
        ys, xs = np.nonzero(coverage > 0.05)
        reach = np.hypot(xs - tip[0], ys - tip[1]).max() / 15.0
        assert reach <= REACH_MM + 25


def test_card_without_free_margin_is_not_held() -> None:
    thumb, _ = _held([(20.0, 20.0, 1264.0, 790.0)], 0)
    assert thumb is None


def test_page_marks_stay_clear_of_text() -> None:
    signed = stamped = 0
    for seed in range(14):
        sheet = render_page(parse_layout(PAGE["en"]), random.Random(seed))
        marks, boxes = sheet.params["marks"], sheet.faces[0].boxes
        if "signature" in marks:
            signed += 1
            assert not any(_overlaps(tuple(marks["signature"]["box_px"]), box) for box in boxes)
        if "stamp" in marks:
            stamped += 1
            (x, y), r = marks["stamp"]["centre_px"], marks["stamp"]["radius_px"]
            assert not any(_overlaps((x - r, y - r, x + r, y + r), box) for box in boxes)
    assert signed and stamped


CONTACT_ROWS = [
    ("الاسم", "الجهة", "الهاتف", "البريد الإلكتروني", "المدينة"),
    ("فراس جورج عواد", "جمعية الأمل الخيرية للتنمية", "00962774980379", "firas.awad99@amal-charity.org", "المدينة"),
    ("يارا النابلسي", "مستشفى الرحمة التخصصي", "+962794856075", "y.nabulsi@rahma-hospital.jo", "طرطوس"),
    ("صالح جورج الجندي", "شركة مرايا للتصميم والطباعة", "٠٠٩٦٢٧٨٠٥٧٥٩١٠", "qais.zoubi@amal-charity.org", "الرصيفة"),
    ("مصطفى الشريف", "مؤسسة النهضة للتنمية", "+963974974846", "mustafasharif@maraya-design.com", "القامشلي"),
]


def test_contact_list_columns_keep_every_word_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object) -> None:
        raise AssertionError("a table cell split a word")

    monkeypatch.setattr(Typesetter, "_break_word", refuse)
    layout = [{"kind": "row", "text": " | ".join(row)} for row in CONTACT_ROWS]
    for seed in range(4):
        sheet = render_page(parse_layout(layout), random.Random(seed))
        assert sheet.faces[0].boxes


def test_chat_message_draws_name_text_and_time_once() -> None:
    assert message_parts("[10:02] Sami: hey, see you") == ("10:02", "Sami", "hey, see you")
    assert message_parts("Today") is None
    layout = [
        {"kind": "message", "text": "[10:02] Sami: hello there"},
        {"kind": "message", "text": "[10:03] Rana: ok"},
    ]
    sheet = render_chat(parse_layout(layout), random.Random(0), "en")
    assert [block["lines"] for block in sheet.params["blocks"]] == [3, 3]
    assert len(sheet.faces[0].boxes) == 2 * 3 + 1


def test_forwarded_photo_is_downscaled_and_recompressed() -> None:
    profile = replace(PHOTO["moderate"], forward_p=1.0)
    params: dict[str, object] = {}
    out = finish(Image.new("RGB", (2000, 1500), (128, 128, 128)), random.Random(0), profile, params)
    forwarded = params["forwarded"]
    assert isinstance(forwarded, dict)
    assert max(out.size) == forwarded["long_side"]
    assert profile.forward_long_side[0] <= forwarded["long_side"] <= profile.forward_long_side[1]
    assert profile.forward_quality[0] <= params["jpeg_quality"] <= profile.forward_quality[1]  # type: ignore[operator]


def test_about_a_quarter_of_photos_are_forwarded() -> None:
    share = sum(PHOTO[severity].forward_p * weight for severity, weight in SEVERITY_SHARES.items())
    assert 0.2 <= share <= 0.3


def _plate(width: int, height: int, boxes: list[Box]) -> Plate:
    rgb = np.full((height, width, 3), 250, np.uint8)
    return Plate(rgb, np.full((height, width), 255, np.uint8), boxes, [(0.0, 0.0, float(width), float(height))])


@pytest.mark.parametrize("severity", ["mild", "moderate", "harsh"])
def test_card_fills_its_share_of_the_frame_width(severity: str) -> None:
    profile = PHOTO[severity]
    plate = _plate(1284, 810, [(300.0, 100.0, 1200.0, 140.0)])
    for seed in range(6):
        params: dict[str, object] = {}
        _place(plate, random.Random(seed), profile, 15.0, Framing(profile.fill_card, 0.0), params)
        fill = params["fill"]
        assert isinstance(fill, float)
        assert profile.fill_card[0] - 0.01 <= fill <= profile.fill_card[1] + 0.01


@pytest.mark.parametrize("severity", ["mild", "moderate", "harsh"])
def test_passport_photo_frames_the_data_page_beside_part_of_the_facing_page(severity: str) -> None:
    profile = PHOTO[severity]
    page = Face(Image.new("RGB", (1250, 880), (236, 230, 240)), [(320.0, 120.0, 1190.0, 180.0)])
    bleeds = set()
    for seed in range(8):
        booklet = open_booklet(page, random.Random(seed), 10.0)
        plate = arrange([booklet.face], vertical=True, gap=0)
        params: dict[str, object] = {}
        framing = Framing(profile.fill_passport, 0.0, focus=booklet.focus)
        placed = _place(plate, random.Random(seed), profile, 10.0, framing, params)
        fill = params["fill"]
        assert isinstance(fill, float)
        assert profile.fill_passport[0] - 0.01 <= fill <= profile.fill_passport[1] + 0.01
        assert 0.6 <= fill * 1250 / (booklet.focus.box[2] - booklet.focus.box[0]) <= 0.9
        height, width = placed.alpha.shape
        f = placed.size[0] / plate.size[0]
        left, top, right, bottom = booklet.focus.box
        x0, y0, x1, y1 = transform_box((left * f, top * f, right * f, bottom * f), placed.matrix)
        assert x0 >= 0 and y0 >= 0 and x1 <= width and y1 <= height
        facing = {"top": y0 / height, "left": x0 / width, "right": 1 - x1 / width}[booklet.focus.bleed]
        assert facing >= 0.05
        bleeds.add(booklet.focus.bleed)
    assert "top" in bleeds and len(bleeds) > 1


def test_cropped_page_photo_keeps_text_in_the_frame_and_loses_little_margin() -> None:
    profile = replace(PHOTO["harsh"], crop_p=1.0)
    plate = _plate(1240, 1754, [(150.0, 150.0 + 40 * k, 1090.0, 180.0 + 40 * k) for k in range(35)])
    crossed = 0
    for seed in range(10):
        placed = _place(plate, random.Random(seed), profile, 5.9, Framing(profile.fill_page, 0.0), {})
        height, width = placed.alpha.shape
        for x0, y0, x1, y1 in placed.frame_boxes():
            assert x0 >= 0 and x1 <= width and y0 >= 0 and y1 <= height
        low, high = placed.outline.min(axis=0), placed.outline.max(axis=0)
        outside = np.maximum(0, -low) + np.maximum(0, high - np.array([width, height]))
        assert (outside <= CROP_SHARE * (high - low) + 1).all()
        crossed += bool(outside.any())
    assert crossed > 0


def test_cropped_page_across_the_frame_keeps_its_text_inside(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(photo, "ORIENTATION_SWAP_P", 1.0)
    profile = replace(PHOTO["moderate"], crop_p=1.0)
    plate = _plate(1240, 1754, [(150.0, 150.0 + 40 * k, 1090.0, 180.0 + 40 * k) for k in range(12)])
    for seed in range(10):
        placed = _place(plate, random.Random(seed), profile, 5.9, Framing(profile.fill_page, 0.0), {})
        height, width = placed.alpha.shape
        assert width > height
        for x0, y0, x1, y1 in placed.frame_boxes():
            assert x0 >= 0 and x1 <= width and y0 >= 0 and y1 <= height
        low, high = placed.outline.min(axis=0), placed.outline.max(axis=0)
        outside = np.maximum(0, -low) + np.maximum(0, high - np.array([width, height]))
        assert (outside <= CROP_SHARE * (high - low) + 1).all()


def test_thumb_held_forwarded_photo_is_deterministic(tmp_path: Path) -> None:
    front = [b for b in ID_CARD["en"] if b.get("side", "front") == "front"]
    renders = []
    for name in ("a.jpg", "b.jpg"):
        params = render_image(
            front, style="id_card", language="en", capture="photo", severity="moderate",
            rng=random.Random(6), out=tmp_path / name,
        )  # fmt: skip
        renders.append(((tmp_path / name).read_bytes(), params))
    assert renders[0] == renders[1]
    assert renders[0][1]["capture_params"]["thumb"] and renders[0][1]["capture_params"]["forwarded"]


def test_passport_photo_shows_the_open_booklet_around_the_data_page(tmp_path: Path) -> None:
    for seed in range(2):
        params = render_image(
            PASSPORT["en"], style="passport", language="en", capture="photo", severity="moderate",
            rng=random.Random(seed), out=tmp_path / f"{seed}.jpg",
        )["capture_params"]  # fmt: skip
        assert params["booklet"]["binding"] in ("top", "left", "right")
        assert PHOTO["moderate"].fill_passport[0] - 0.01 <= params["fill"] <= PHOTO["moderate"].fill_passport[1] + 0.01
        assert not params["cropped"]


@pytest.mark.parametrize(
    ("style", "language", "severity"),
    [
        ("page", "en", "mild"),
        ("page", "ar", "harsh"),
        ("id_card", "ar", "moderate"),
        ("passport", "mixed", "mild"),
        ("id_card", "en", "harsh"),
    ],
)
def test_scan_resolution_follows_its_severity(tmp_path: Path, style: str, language: str, severity: str) -> None:
    layout = {"page": PAGE, "id_card": ID_CARD, "passport": PASSPORT}[style][language]
    params = render_image(
        layout, style=style, language=language, capture="scan", severity=severity, rng=random.Random(2),
        out=tmp_path / "scan.png",
    )  # fmt: skip
    low, high = SCAN[severity].dpi
    assert low * 0.99 <= params["capture_params"]["dpi"] <= high * 1.01
    assert params["output"]["dpi"] == round(params["capture_params"]["dpi"], 2)
    if style == "page":
        assert params["document"]["dpi"] >= low


def test_glare_below_harsh_keeps_clear_of_text() -> None:
    profile = PHOTO["moderate"]
    plate = _plate(1284, 810, [(60.0, 60.0 + 90 * k, 700.0, 110.0 + 90 * k) for k in range(8)])
    applied = 0
    for seed in range(8):
        placed = _place(plate, random.Random(seed), profile, 15.0, Framing(profile.fill_card, 0.0), {})
        image = np.zeros((*placed.alpha.shape, 3), np.float32)
        glare = _glare(image, placed, random.Random(seed), profile, 1.0, False)
        if not glare["applied"]:
            continue
        applied += 1
        x, y = glare["centre_px"]
        reach = glare["radius_px"] * glare["stretch"] * 1.6
        for x0, y0, x1, y1 in placed.frame_boxes():
            assert np.hypot(x - min(max(x, x0), x1), y - min(max(y, y0), y1)) >= reach - 1
    assert applied
