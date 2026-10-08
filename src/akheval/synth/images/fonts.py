"""Installed fonts: discovery through fontconfig, glyph coverage and the font set of one document."""

from __future__ import annotations

import random
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import TypeVar

from fontTools.ttLib import TTFont
from PIL import ImageFont

ARABIC_FAMILIES = ("Amiri", "Noto Naskh Arabic", "Noto Kufi Arabic", "Noto Sans Arabic", "Droid Arabic Kufi")
SANS_FAMILIES = ("Noto Sans", "DejaVu Sans", "Liberation Sans", "FreeSans")
SERIF_FAMILIES = ("Liberation Serif", "FreeSerif", "Noto Serif")
MONO_FAMILIES = ("DejaVu Sans Mono", "FreeMono", "Liberation Mono", "Noto Sans Mono")
# Both families carry Arabic and Latin glyphs. The typesetter uses them for characters the document fonts lack.
FALLBACK_FAMILIES = ("FreeSerif", "Amiri")

REGULAR_STYLES = ("Regular", "Book")
BOLD_STYLES = ("Bold",)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class FontFace:
    family: str
    style: str
    path: Path

    @property
    def name(self) -> str:
        return f"{self.family} {self.style}"

    def covers(self, char: str) -> bool:
        return ord(char) in _cmap(self.path)

    def font(self, size: int) -> ImageFont.FreeTypeFont:
        return _load(self.path, size)


@dataclass(frozen=True, slots=True)
class FontPair:
    regular: FontFace
    bold: FontFace

    def pick(self, bold: bool) -> FontFace:
        return self.bold if bold else self.regular


@dataclass(frozen=True, slots=True)
class FontSet:
    """The fonts one document uses: an Arabic and a Latin family, a monospace face and the fallbacks."""

    arabic: FontPair
    latin: FontPair
    mono: FontFace
    fallbacks: tuple[FontPair, ...]

    def preferences(self, *, arabic_first: bool, bold: bool) -> tuple[FontFace, ...]:
        first, second = (self.arabic, self.latin) if arabic_first else (self.latin, self.arabic)
        return tuple(pair.pick(bold) for pair in (first, second, *self.fallbacks))

    def to_params(self) -> dict[str, str]:
        return {"arabic": self.arabic.regular.family, "latin": self.latin.regular.family, "mono": self.mono.name}


@lru_cache(maxsize=1)
def installed_faces() -> dict[tuple[str, str], Path]:
    """Map (family, style) to a font file for every face fontconfig knows."""
    fc_list = shutil.which("fc-list")
    if fc_list is None:
        raise RuntimeError("fc-list from fontconfig is required to locate fonts")
    listing = subprocess.run(
        [fc_list, "--format", "%{family[0]}|%{style[0]}|%{file}\n"], capture_output=True, text=True, check=True
    ).stdout
    faces: dict[tuple[str, str], Path] = {}
    for line in sorted(listing.splitlines()):
        parts = line.split("|", 2)
        if len(parts) == 3:
            faces.setdefault((parts[0], parts[1]), Path(parts[2]))
    return faces


def find_face(family: str, styles: Sequence[str]) -> FontFace | None:
    faces = installed_faces()
    for style in styles:
        path = faces.get((family, style))
        if path is not None:
            return FontFace(family, style, path)
    return None


def font_pairs(families: Sequence[str]) -> list[FontPair]:
    """Regular and bold faces of each installed family; a family without a bold face reuses the regular one."""
    pairs = []
    for family in families:
        regular = find_face(family, REGULAR_STYLES)
        if regular is not None:
            pairs.append(FontPair(regular, find_face(family, BOLD_STYLES) or regular))
    return pairs


def choose_fonts(rng: random.Random, latin_families: Sequence[str]) -> FontSet:
    """Pick one installed Arabic family, one Latin family from ``latin_families`` and a monospace face."""
    monos = [face for family in MONO_FAMILIES if (face := find_face(family, REGULAR_STYLES)) is not None]
    fallbacks = tuple(font_pairs(FALLBACK_FAMILIES))
    if not fallbacks:
        raise RuntimeError(f"none of the fallback font families {FALLBACK_FAMILIES} is installed")
    return FontSet(
        arabic=_choose(rng, font_pairs(ARABIC_FAMILIES), "Arabic"),
        latin=_choose(rng, font_pairs(latin_families), "Latin"),
        mono=_choose(rng, monos, "monospace"),
        fallbacks=fallbacks,
    )


def _choose(rng: random.Random, options: Sequence[T], what: str) -> T:
    if not options:
        raise RuntimeError(f"no {what} font is installed")
    return rng.choice(options)


@cache
def _cmap(path: Path) -> frozenset[int]:
    with TTFont(str(path), lazy=True, fontNumber=0) as font:
        return frozenset(font.getBestCmap())


@lru_cache(maxsize=256)
def _load(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size, layout_engine=ImageFont.Layout.RAQM)
