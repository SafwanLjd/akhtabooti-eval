"""Measure how much text each capture severity leaves legible.

The script renders a seeded sample of every style, capture, severity and language through ``render_image``, using
layouts from the dataset generator. It reads each image with tesseract (``ara+eng``) and with akh-ocr (PP-OCRv5
detection with the Arabic and English recognisers, reading order), scores the token recall of both readings against
the text the image shows, and prints the median of the better reading per style, capture and severity, for English
and for Arabic with mixed.

Token recall counts the words of three or more characters, compared without edge punctuation, Arabic diacritics or
case. Tesseract reads photos with page segmentation mode 11 (sparse text) and every other image with mode 6 (one
block of text).

Run it from the repository root after ``rust/ocr/build.sh`` and ``akheval fetch``:

    .venv/bin/python scripts/calibrate_captures.py --out /tmp/calibration --seeds 4
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from akheval.dataset import LANGUAGES, SEVERITIES
from akheval.extractors.ppocr import PaddleOcrExtractor
from akheval.extractors.tesseract import TesseractExtractor
from akheval.scoring import bare_words
from akheval.synth.build import generate_document, plan_documents
from akheval.synth.builder import SIDES, block_side, has_back
from akheval.synth.config import load_config
from akheval.synth.doctypes import DOC_TYPES, NO_IMAGES
from akheval.synth.images import CAPTURES_BY_STYLE, STYLES, render_image

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs/dataset-v1.toml"
AKH_OCR = REPO / "rust/ocr/bin/akh-ocr"
MODELS = REPO / "models/ppocr"
RECOGNISERS = ("arabic", "en")
DIACRITICS = re.compile("[ً-ْـ]")
MIN_TOKEN = 3
SUFFIXES = {"photo": ".jpg", "scan": ".png", "digital": ".png"}
SPARSE_TEXT_PSM = 11
TEXT_BLOCK_PSM = 6
GROUPS = ("en", "ar+mixed")

Layout = list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class Job:
    style: str
    capture: str
    severity: str | None
    language: str
    seed: int
    doc_id: str
    face: str | None
    blocks: Layout

    @property
    def name(self) -> str:
        face = self.face or "all"
        return f"{self.style}.{self.capture}.{self.severity or 'none'}.{self.language}.{self.seed}.{self.doc_id}.{face}"


@dataclass(frozen=True, slots=True)
class Reading:
    name: str
    style: str
    capture: str
    severity: str | None
    language: str
    tokens: int
    tesseract: float
    akh_ocr: float

    @property
    def best(self) -> float:
        return max(self.tesseract, self.akh_ocr)

    @property
    def group(self) -> str:
        return "en" if self.language == "en" else "ar+mixed"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    pool = _layouts()
    tesseract = {psm: TesseractExtractor("ara+eng", psm) for psm in (SPARSE_TEXT_PSM, TEXT_BLOCK_PSM)}
    akh_ocr = PaddleOcrExtractor(
        str(AKH_OCR),
        str(MODELS / "PP-OCRv5_mobile_det.mnn"),
        [str(MODELS / f"{name}_PP-OCRv5_mobile_rec_infer.mnn") for name in RECOGNISERS],
        [str(MODELS / f"ppocr_keys_{name}.txt") for name in RECOGNISERS],
        order="reading",
        fix_rtl=True,
    )
    readings = []
    try:
        for job in _jobs(pool, args.seeds):
            path = out / f"{job.name}{SUFFIXES[job.capture]}"
            render_image(
                job.blocks,
                style=job.style,
                language=job.language,
                capture=job.capture,
                severity=job.severity,
                rng=random.Random(f"calibrate:{job.name}"),
                out=path,
            )
            psm = SPARSE_TEXT_PSM if job.capture == "photo" else TEXT_BLOCK_PSM
            readings.append(_read(job, path, tesseract[psm], akh_ocr))
    finally:
        akh_ocr.close()
    _write(readings, out / "recall.jsonl")
    print(_table(readings))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render, OCR and score a sample of captures per severity.")
    parser.add_argument("--out", type=Path, required=True, help="directory for the renders and recall.jsonl")
    parser.add_argument("--seeds", type=int, default=3, help="samples per style, capture, severity and language")
    return parser


def _layouts() -> dict[tuple[str, str], list[tuple[str, Layout]]]:
    """Layouts of every dataset document that gets images, by style and language."""
    cfg = load_config(CONFIG)
    pool: dict[tuple[str, str], list[tuple[str, Layout]]] = defaultdict(list)
    for spec in plan_documents(cfg.documents):
        if DOC_TYPES[spec.doc_type].images == NO_IMAGES:
            continue
        doc = generate_document(spec, seed=cfg.seed, split="test")
        pool[(doc.meta["style"], doc.language)].append((doc.id, doc.meta["layout"]))
    return pool


def _jobs(pool: Mapping[tuple[str, str], list[tuple[str, Layout]]], seeds: int) -> Iterator[Job]:
    """One document per seed for each style, capture, severity and language; a card photo shows one side."""
    for seed in range(seeds):
        for style in STYLES:
            for capture in CAPTURES_BY_STYLE[style]:
                for severity in (None,) if capture == "digital" else SEVERITIES:
                    for language in LANGUAGES:
                        rng = random.Random(f"calibrate:{seed}:{style}:{capture}:{severity}:{language}")
                        doc_id, layout = rng.choice(pool[(style, language)])
                        face = rng.choice(SIDES) if capture == "photo" and has_back(layout) else None
                        blocks = layout if face is None else [b for b in layout if block_side(b) == face]
                        yield Job(style, capture, severity, language, seed, doc_id, face, blocks)


def _read(job: Job, path: Path, tesseract: TesseractExtractor, akh_ocr: PaddleOcrExtractor) -> Reading:
    expected = _tokens("\n".join(block["text"] for block in job.blocks))
    scores = [_recall(expected, extractor.extract(path)) for extractor in (tesseract, akh_ocr)]
    reading = Reading(job.name, job.style, job.capture, job.severity, job.language, expected.total(), *scores)
    print(f"{reading.best:.2f} tesseract {scores[0]:.2f} akh-ocr {scores[1]:.2f} {path.name}", flush=True)
    return reading


def _tokens(text: str) -> Counter[str]:
    words = bare_words(DIACRITICS.sub("", text).casefold())
    return Counter({word: n for word, n in words.items() if len(word) >= MIN_TOKEN})


def _recall(expected: Counter[str], text: str) -> float:
    return (expected & _tokens(text)).total() / max(expected.total(), 1)


def _write(readings: Sequence[Reading], path: Path) -> None:
    lines = [json.dumps(asdict(r) | {"best": round(r.best, 4)}, sort_keys=True) for r in readings]
    path.write_text("".join(f"{line}\n" for line in sorted(lines)), encoding="utf-8")


def _table(readings: Sequence[Reading]) -> str:
    """Median best recall and image count per style, capture and severity, then per capture and severity."""
    cells: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in readings:
        severity = r.severity or "-"
        cells[(r.style, r.capture, severity)][r.group].append(r.best)
        if r.capture != "digital":
            cells[("all", r.capture, severity)][r.group].append(r.best)
    order = {name: k for k, name in enumerate((*STYLES, "all", "photo", "scan", "digital", *SEVERITIES, "-"))}
    rows = ["| style | capture | severity | English | n | Arabic and mixed | n |", "|---|---|---|---:|---:|---:|---:|"]
    for key in sorted(cells, key=lambda k: tuple(order[part] for part in k)):
        values = [cells[key][group] for group in GROUPS]
        medians = [f"{statistics.median(v):.2f} | {len(v)}" if v else "- | 0" for v in values]
        rows.append(f"| {' | '.join(key)} | {' | '.join(medians)} |")
    return "\n".join(rows)


if __name__ == "__main__":
    raise SystemExit(main())
