"""Line layout for mixed Arabic and Latin text: direction runs, font fallback, wrapping and drawing.

A line is split into direction runs. Text of the opposite direction (Latin words and numbers inside an
Arabic paragraph, Arabic words inside a Latin one) forms isolated runs, so a phone number such as
``+962 79 123 4567`` reads left to right inside Arabic text the way a typeset document shows it. Each run
is split again wherever the font changes, because most Arabic fonts lack Latin glyphs and the reverse.
libraqm shapes every piece with an explicit direction.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from PIL import ImageDraw, ImageFont, features

from .fonts import FontFace, FontSet

Direction = Literal["ltr", "rtl"]
RGB = tuple[int, int, int]
Box = tuple[float, float, float, float]

# Characters that stay with a Latin or numeric run inside Arabic text when they touch it.
_LEADING = "+#$([{"
_CLOSERS = {")": "(", "]": "[", "}": "{"}
_TRAILING = "%"

# Baseline position inside a line box, as a share of the font size above the box's vertical centre.
_BASELINE_FROM_CENTRE = 0.3


@dataclass(frozen=True, slots=True)
class TextStyle:
    size: int
    color: RGB = (25, 25, 25)
    bold: bool = False
    mono: bool = False

    def scaled(self, factor: float) -> TextStyle:
        return replace(self, size=max(1, round(self.size * factor)))


@dataclass(frozen=True, slots=True)
class Styled:
    """Logical text with style spans that cover it from start to end."""

    text: str
    spans: tuple[tuple[int, int, TextStyle], ...]

    @classmethod
    def plain(cls, text: str, style: TextStyle) -> Styled:
        return cls(text, ((0, len(text), style),))

    @classmethod
    def field(cls, text: str, label: TextStyle, value: TextStyle) -> Styled:
        """Style a ``Label: value`` line; text without ``": "`` gets the value style."""
        cut = text.find(": ")
        if cut < 0:
            return cls.plain(text, value)
        return cls(text, ((0, cut + 1, label), (cut + 1, len(text), value)))

    def styles(self) -> list[TextStyle]:
        """The style of every character."""
        out: list[TextStyle] = []
        for start, end, style in self.spans:
            out.extend([style] * (end - start))
        return out

    def slice(self, start: int, end: int) -> Styled:
        spans = tuple(
            (max(a, start) - start, min(b, end) - start, style)
            for a, b, style in self.spans
            if min(b, end) > max(a, start)
        )
        return Styled(self.text[start:end], spans or ((0, 0, self.spans[0][2]),))

    def scaled(self, factor: float) -> Styled:
        return Styled(self.text, tuple((a, b, style.scaled(factor)) for a, b, style in self.spans))

    @property
    def size(self) -> int:
        return max(style.size for _, _, style in self.spans)


@dataclass(frozen=True, slots=True)
class Piece:
    text: str
    face: FontFace
    style: TextStyle
    direction: Direction
    width: float

    @property
    def font(self) -> ImageFont.FreeTypeFont:
        return self.face.font(self.style.size)


@dataclass(frozen=True, slots=True)
class Line:
    """One drawn line: its logical text and the pieces in visual order, left to right."""

    text: str
    direction: Direction
    pieces: tuple[Piece, ...]
    size: int
    missing: str = ""

    @property
    def width(self) -> float:
        return sum(piece.width for piece in self.pieces)

    @property
    def faces(self) -> set[str]:
        return {piece.face.name for piece in self.pieces}


def strong_direction(char: str) -> Direction | None:
    """Direction class used for runs: Latin letters and all digits are LTR, Arabic letters RTL."""
    bidi = unicodedata.bidirectional(char)
    if bidi in ("R", "AL"):
        return "rtl"
    if bidi in ("L", "EN", "AN"):
        return "ltr"
    return None


def paragraph_direction(text: str) -> Direction:
    """Direction of the first strong letter; digits do not count. Defaults to LTR."""
    for char in text:
        bidi = unicodedata.bidirectional(char)
        if bidi == "L":
            return "ltr"
        if bidi in ("R", "AL"):
            return "rtl"
    return "ltr"


def direction_runs(text: str, base: Direction) -> list[tuple[int, int, Direction]]:
    """Split ``text`` into logical runs that cover it, each with the direction it is drawn in."""
    opposite: Direction = "rtl" if base == "ltr" else "ltr"
    isolated: list[tuple[int, int]] = []
    start = end = -1
    for i, char in enumerate(text):
        direction = strong_direction(char)
        if direction == opposite:
            start = i if start < 0 else start
            end = i + 1
        elif direction == base and start >= 0:
            isolated.append((start, end))
            start = -1
    if start >= 0:
        isolated.append((start, end))
    if base == "rtl":
        isolated = [_absorb_affixes(text, a, b) for a, b in isolated]
    runs: list[tuple[int, int, Direction]] = []
    cursor = 0
    for a, b in isolated:
        if a > cursor:
            runs.append((cursor, a, base))
        runs.append((a, b, opposite))
        cursor = b
    if cursor < len(text) or not runs:
        runs.append((cursor, len(text), base))
    return runs


def _absorb_affixes(text: str, start: int, end: int) -> tuple[int, int]:
    """Extend an LTR run over touching signs and brackets, as in ``+962``, ``[10:03]`` or ``(06) 555``."""
    while start > 0 and text[start - 1] in _LEADING:
        start -= 1
    while end < len(text):
        char = text[end]
        opener = _CLOSERS.get(char)
        run = text[start:end]
        if char in _TRAILING or (opener is not None and run.count(opener) > run.count(char)):
            end += 1
        else:
            break
    return start, end


class Typesetter:
    """Lays out and draws styled lines with one document's fonts."""

    def __init__(self, fonts: FontSet) -> None:
        if not features.check("raqm"):
            raise RuntimeError("Pillow was built without libraqm, which Arabic shaping needs")
        self.fonts = fonts

    def layout(self, styled: Styled, base: Direction) -> Line:
        styles = styled.styles()
        runs: list[list[Piece]] = []
        missing: list[str] = []
        for start, end, direction in direction_runs(styled.text, base):
            pieces = self._pieces(styled.text, styles, start, end, direction, base, missing)
            runs.append(pieces if direction == "ltr" else pieces[::-1])
        if base == "rtl":
            runs.reverse()
        return Line(
            text=styled.text,
            direction=base,
            pieces=tuple(piece for run in runs for piece in run),
            size=styled.size,
            missing="".join(sorted(set(missing))),
        )

    def fit(self, styled: Styled, width: float, *, min_scale: float = 1.0) -> list[Line]:
        """Lay out ``styled`` within ``width``: shrink it down to ``min_scale``, then wrap at spaces."""
        base = paragraph_direction(styled.text)
        line = self.layout(styled, base)
        scale = 1.0
        while line.width > width and scale > min_scale:
            scale = max(min_scale, scale * min(0.97, width / line.width))
            line = self.layout(styled.scaled(scale), base)
        if line.width <= width:
            return [line]
        return self.wrap(styled.scaled(scale), base, width)

    def shrink_to(self, styled: Styled, width: float) -> Line:
        """Lay out ``styled`` on one line, scaled down until it fits ``width``."""
        base = paragraph_direction(styled.text)
        line = self.layout(styled, base)
        while line.width > width and line.size > 1:
            line = self.layout(styled.scaled(min(0.97, width / line.width) * line.size / styled.size), base)
        return line

    def wrap(self, styled: Styled, base: Direction, width: float) -> list[Line]:
        """Greedy wrap at single spaces; joining the line texts with one space restores the input."""
        text = styled.text
        cuts = [i for i, char in enumerate(text) if char == " "] + [len(text)]
        lines: list[Line] = []
        start, k = 0, 0
        while True:
            line: Line | None = None
            line_end = start
            while k < len(cuts):
                candidate = self.layout(styled.slice(start, cuts[k]), base)
                if line is not None and candidate.width > width:
                    break
                line, line_end = candidate, cuts[k]
                k += 1
                if candidate.width > width:
                    break
            if line is None:
                return lines
            if line.width > width:
                lines.extend(self._break_word(styled.slice(start, line_end), base, width))
            else:
                lines.append(line)
            if line_end >= len(text):
                return lines
            start = line_end + 1

    def _break_word(self, styled: Styled, base: Direction, width: float) -> list[Line]:
        """Split text without usable spaces at character boundaries."""
        lines: list[Line] = []
        start = 0
        while start < len(styled.text):
            end = start + 1
            line = self.layout(styled.slice(start, end), base)
            while end < len(styled.text):
                candidate = self.layout(styled.slice(start, end + 1), base)
                if candidate.width > width:
                    break
                line, end = candidate, end + 1
            lines.append(line)
            start = end
        return lines

    def _pieces(
        self,
        text: str,
        styles: Sequence[TextStyle],
        start: int,
        end: int,
        direction: Direction,
        base: Direction,
        missing: list[str],
    ) -> list[Piece]:
        arabic_first = direction == "rtl" or (base == "rtl" and not _has_latin_letter(text[start:end]))
        pieces: list[Piece] = []
        piece_start = start
        current: tuple[FontFace, TextStyle] | None = None
        for i in range(start, end):
            style = styles[i]
            face = self._face(text[i], style, arabic_first, current[0] if current else None, missing)
            if current is not None and (face, style) != current:
                pieces.append(self._piece(text[piece_start:i], *current, direction))
                piece_start = i
            current = (face, style)
        if current is not None:
            pieces.append(self._piece(text[piece_start:end], *current, direction))
        return pieces

    def _face(
        self, char: str, style: TextStyle, arabic_first: bool, current: FontFace | None, missing: list[str]
    ) -> FontFace:
        if current is not None and char.isspace() and current.covers(char):
            return current
        preferences = self.fonts.preferences(arabic_first=arabic_first, bold=style.bold)
        if style.mono:
            preferences = (self.fonts.mono, *preferences)
        for face in preferences:
            if face.covers(char):
                return face
        if not char.isspace() and unicodedata.category(char) != "Cf":
            missing.append(char)
        return current or preferences[0]

    @staticmethod
    def _piece(text: str, face: FontFace, style: TextStyle, direction: Direction) -> Piece:
        width = face.font(style.size).getlength(text, direction=direction) if text else 0.0
        return Piece(text, face, style, direction, width)


def _has_latin_letter(text: str) -> bool:
    return any(unicodedata.bidirectional(char) == "L" for char in text)


def baseline(top: float, pitch: float, size: int) -> float:
    """Baseline y of a line whose box starts at ``top`` and is ``pitch`` tall."""
    return top + pitch / 2 + size * _BASELINE_FROM_CENTRE


def draw_line(draw: ImageDraw.ImageDraw, line: Line, x: float, y_baseline: float) -> None:
    """Draw ``line`` with its left edge at ``x``."""
    for piece in line.pieces:
        if piece.text:
            draw.text(
                (x, y_baseline),
                piece.text,
                font=piece.font,
                fill=piece.style.color,
                anchor="ls",
                direction=piece.direction,
            )
        x += piece.width


def line_left(line: Line, left: float, right: float, align: str) -> float:
    """Left x for ``line`` in [left, right]: ``center``, or ``start`` which follows the line direction."""
    if align == "center":
        return left + (right - left - line.width) / 2
    return right - line.width if line.direction == "rtl" else left
