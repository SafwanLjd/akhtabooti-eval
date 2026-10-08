from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from akheval.extractors.base import ExtractionError
from akheval.extractors.ppocr import OcrProcessError, OcrResult, PaddleOcrExtractor, visual_to_logical

# Speaks the akh-ocr protocol. The image file holds the recognised lines, one per row; "ERROR <msg>" makes the
# fake report an error and "CRASH" makes it exit.
FAKE_AKH_OCR = """
import json, sys
if "--version" in sys.argv:
    print(json.dumps({"akh_ocr": "fake", "argv": sys.argv[1:]}))
    sys.exit(0)
print(json.dumps({"ready": True, "seconds": 0.0}), flush=True)
for raw in sys.stdin:
    request = json.loads(raw)
    content = open(request["path"], encoding="utf-8").read()
    if content.startswith("CRASH"):
        sys.stderr.write("fake engine crashed\\n")
        sys.exit(3)
    error = content[len("ERROR "):] if content.startswith("ERROR ") else None
    rows = [] if error else content.splitlines()
    lines = [{"text": row, "score": 0.9} for row in rows]
    text = "".join(row + "\\n" for row in rows)
    print(json.dumps({"id": request["id"], "text": text, "lines": lines, "seconds": 0.01, "error": error}), flush=True)
"""


@pytest.fixture
def fake_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "akh-ocr"
    binary.write_text(f"#!{sys.executable}\n{FAKE_AKH_OCR}", encoding="utf-8")
    binary.chmod(0o755)
    return binary


@pytest.fixture
def models(tmp_path: Path) -> dict[str, Path]:
    names = ("det.mnn", "ar.mnn", "ar.txt", "en.mnn", "en.txt")
    paths = {name: tmp_path / name for name in names}
    for name, path in paths.items():
        path.write_text(name, encoding="utf-8")
    return paths


def make(binary: Path, models: dict[str, Path], order: str = "engine", fix_rtl: bool = False) -> PaddleOcrExtractor:
    det, rec, keys = (str(models[name]) for name in ("det.mnn", "ar.mnn", "ar.txt"))
    return PaddleOcrExtractor(str(binary), det, rec, keys, order=order, fix_rtl=fix_rtl)


def image(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_extract_returns_the_core_text(tmp_path: Path, fake_binary: Path, models: dict[str, Path]) -> None:
    extractor = make(fake_binary, models)
    assert extractor.extract(image(tmp_path, "a.png", "Sara\nsara@example.org")) == "Sara\nsara@example.org\n"


def test_extract_reuses_one_process(tmp_path: Path, fake_binary: Path, models: dict[str, Path]) -> None:
    extractor = make(fake_binary, models)
    paths = [image(tmp_path, f"{i}.png", f"line {i}") for i in range(3)]
    assert [extractor.extract(path) for path in paths] == ["line 0\n", "line 1\n", "line 2\n"]


def test_engine_error_becomes_extraction_error(tmp_path: Path, fake_binary: Path, models: dict[str, Path]) -> None:
    extractor = make(fake_binary, models)
    with pytest.raises(ExtractionError, match="cannot decode"):
        extractor.extract(image(tmp_path, "bad.png", "ERROR cannot decode"))
    assert extractor.extract(image(tmp_path, "good.png", "ok")) == "ok\n"


def test_crash_is_reported_and_process_restarts(tmp_path: Path, fake_binary: Path, models: dict[str, Path]) -> None:
    extractor = make(fake_binary, models)
    with pytest.raises(OcrProcessError, match="fake engine crashed"):
        extractor.extract(image(tmp_path, "crash.png", "CRASH"))
    assert extractor.extract(image(tmp_path, "after.png", "still here")) == "still here\n"


def test_missing_binary_is_an_extraction_error(tmp_path: Path, models: dict[str, Path]) -> None:
    with pytest.raises(ExtractionError, match="build it with"):
        make(tmp_path / "absent", models).prepare()


def test_fix_rtl_converts_each_line(tmp_path: Path, fake_binary: Path, models: dict[str, Path]) -> None:
    extractor = make(fake_binary, models, fix_rtl=True)
    visual = "1234 555 079 :فتاه\nFull name: Omar"
    assert extractor.extract(image(tmp_path, "ar.png", visual)) == "هاتف: 079 555 1234\nFull name: Omar\n"


def test_command_pairs_recognisers_with_keys(fake_binary: Path, models: dict[str, Path]) -> None:
    extractor = PaddleOcrExtractor(
        str(fake_binary),
        str(models["det.mnn"]),
        [str(models["ar.mnn"]), str(models["en.mnn"])],
        [str(models["ar.txt"]), str(models["en.txt"])],
        order="reading",
    )
    command = extractor.command()
    assert command[command.index("--order") + 1] == "reading"
    recs = [command[i + 1] for i, flag in enumerate(command) if flag == "--rec"]
    keys = [command[i + 1] for i, flag in enumerate(command) if flag == "--keys"]
    assert recs == [str(models["ar.mnn"]), str(models["en.mnn"])]
    assert keys == [str(models["ar.txt"]), str(models["en.txt"])]


def test_rejects_unpaired_recognisers(fake_binary: Path, models: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="pairs"):
        PaddleOcrExtractor(str(fake_binary), str(models["det.mnn"]), [str(models["ar.mnn"])], [])


def test_rejects_unknown_order(fake_binary: Path, models: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="order"):
        make(fake_binary, models, order="columns")


def test_info_records_version_and_model_hashes(fake_binary: Path, models: dict[str, Path]) -> None:
    info = make(fake_binary, models).info()
    assert info["version"]["akh_ocr"] == "fake"
    assert "--version" in info["version"]["argv"]
    expected = hashlib.sha256(b"det.mnn").hexdigest()
    assert info["models"][str(models["det.mnn"])] == expected
    assert set(info["models"]) == {str(models[name]) for name in ("det.mnn", "ar.mnn", "ar.txt")}


def test_ocr_result_parses_a_protocol_line() -> None:
    raw = json.dumps({"id": "7", "text": "a\n", "lines": [{"text": "a", "score": 0.5}], "seconds": 0.2, "error": None})
    result = OcrResult.from_json(raw)
    assert result.id == "7"
    assert result.lines[0].text == "a"
    assert result.lines[0].score == pytest.approx(0.5)
    assert result.error is None


@pytest.mark.parametrize(
    ("visual", "logical"),
    [
        ("يرمعلا دلاخ ةراس لماكلا مسالا", "الاسم الكامل سارة خالد العمري"),
        ("9981023456 :ينطولا مقرلا", "الرقم الوطني: 9981023456"),
        ("1234 555 79 962+ :فتاهلا مقر", "رقم الهاتف: +962 79 555 1234"),
        ("lina.m@example.org :ينورتكلإلا ديربلا", "البريد الإلكتروني: lina.m@example.org"),
        ("دلاخ ةراس Name: Sara Khaled مسالا", "الاسم Name: Sara Khaled سارة خالد"),
        ("Passport No. N1234567", "Passport No. N1234567"),
        ("Amman نامع city", "Amman عمان city"),
    ],
)
def test_visual_to_logical(visual: str, logical: str) -> None:
    assert visual_to_logical(visual) == logical
