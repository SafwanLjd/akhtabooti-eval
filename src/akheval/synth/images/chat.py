"""Phone chat screenshot renderer: status bar, app bar, message bubbles and an input bar."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw

from ..builder import FRONT
from .flow import Block, Placed, block_params
from .fonts import choose_fonts
from .furniture import silhouette
from .sheet import MM_PER_INCH, Face, Sheet
from .text import RGB, Box, Line, Styled, TextStyle, Typesetter, baseline, draw_line, line_left

SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 2340
DP = 3.0  # screen pixels per density-independent pixel
SCREEN_DPI = 72.0  # resolution written into screenshot metadata
CHAT_FAMILIES = ("Noto Sans", "Liberation Sans", "DejaVu Sans", "FreeSans")
GENERIC_TITLES = {"ar": "المحادثات", "en": "Messages"}
SPEAKER = re.compile(r"^\[[^\]]*\]\s*([^:]+):")
MESSAGE = re.compile(r"^\[(?P<time>[^\]]*)\]\s*(?P<name>[^:]+):\s?(?P<body>.*)$")
# Sizes of the sender header and the time as a share of the message size, and their line pitch.
NAME_SHARE = 0.86
TIME_SHARE = 0.68
HEADER_SPACING = 1.3
TIME_GAP_DP = 8
NAME_COLOURS: tuple[RGB, ...] = ((31, 122, 140), (196, 82, 32), (120, 66, 160), (34, 128, 64), (180, 50, 98))
NAME_COLOURS_DARK: tuple[RGB, ...] = ((83, 189, 235), (255, 160, 100), (190, 150, 255), (110, 210, 140))

STATUS_DP, BAR_DP, INPUT_DP, EDGE_DP, GAP_DP, TITLE_DP, TITLE_WIDTH_DP, TICKS_DP = 24, 56, 64, 10, 6, 82, 236, 8


@dataclass(frozen=True, slots=True)
class Theme:
    name: str
    wallpaper: RGB
    doodle: RGB
    bar: RGB
    bar_text: RGB
    outgoing: RGB
    incoming: RGB
    text: RGB
    chip: RGB
    chip_text: RGB
    input: RGB


THEMES = (
    Theme("light-green", (236, 229, 221), (222, 213, 203), (240, 242, 245), (17, 27, 33), (217, 253, 211),
          (255, 255, 255), (17, 27, 33), (255, 250, 230), (84, 101, 111), (255, 255, 255)),
    Theme("light-blue", (245, 246, 248), (235, 237, 241), (255, 255, 255), (20, 20, 25), (209, 230, 255),
          (233, 233, 235), (20, 20, 25), (226, 228, 232), (95, 99, 104), (255, 255, 255)),
    Theme("dark", (11, 20, 26), (20, 32, 40), (32, 44, 51), (233, 237, 239), (0, 92, 75),
          (32, 44, 51), (233, 237, 239), (24, 34, 41), (134, 150, 160), (42, 57, 66)),
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class _Item:
    """A bubble or a chip. A message bubble has its sender's name above the text and the time below it."""

    block: Block
    side: str  # "left", "right" or "center"
    width: float
    height: float
    body: tuple[Line, ...]
    pitch: float
    name: Line | None = None
    time: Line | None = None
    time_row: bool = False  # the time sits on a row of its own instead of after the last line

    def lines(self) -> tuple[Line, ...]:
        return tuple(line for line in (self.name, *self.body, self.time) if line is not None)


@dataclass(frozen=True, slots=True)
class _Styles:
    body: TextStyle
    chip: TextStyle
    time: TextStyle
    spacing: float
    pad: tuple[float, float]
    max_width: float


def render_chat(blocks: Sequence[Block], rng: random.Random, language: str) -> Sheet:
    if any(b.side != FRONT for b in blocks):
        raise ValueError("chat layouts have one side; remove 'side': 'back'")
    theme = rng.choice(THEMES)
    fonts = choose_fonts(rng, CHAT_FAMILIES)
    ts = Typesetter(fonts)
    size = round(rng.uniform(14.5, 16.5) * DP)
    styles = _Styles(
        body=TextStyle(size, theme.text),
        chip=TextStyle(round(size * 0.8), theme.chip_text),
        time=TextStyle(round(size * TIME_SHARE), theme.chip_text),
        spacing=rng.uniform(1.3, 1.45),
        pad=(11 * DP, 7 * DP),
        max_width=SCREEN_WIDTH * rng.uniform(0.74, 0.8),
    )
    title_block = next((b for b in blocks if b.kind == "title"), None)
    me = _pick_me(blocks, rng)
    end = "left" if language == "ar" else "right"
    names = _name_colours(blocks, theme)
    items = []
    for block in blocks:
        if block.kind == "blank":
            items.append(_Item(block, "center", 0, GAP_DP * DP, (), 0))
        elif block.kind == "message" and (parts := message_parts(block.text)) is not None:
            items.append(_bubble(ts, block, parts, styles, names, me, end))
        elif block is not title_block:
            items.append(_plain(ts, block, styles, me))
    top = (STATUS_DP + BAR_DP) * DP
    bottom = INPUT_DP * DP
    content = sum(item.height + GAP_DP * DP for item in items) + EDGE_DP * DP
    height = max(SCREEN_HEIGHT, round(top + content + bottom))
    image = Image.new("RGB", (SCREEN_WIDTH, height), theme.wallpaper)
    draw = ImageDraw.Draw(image)
    doodles = rng.random() < 0.6
    if doodles:
        _doodles(draw, rng, theme, height)
    y = height - bottom - content + EDGE_DP * DP
    boxes: list[Box] = []
    for item in items:
        boxes += _draw_item(draw, item, theme, y, styles.pad, end)
        y += item.height + GAP_DP * DP
    if title_block is not None:
        app_title = title_block.text
    else:
        app_title = GENERIC_TITLES.get(language) or rng.choice(sorted(GENERIC_TITLES.values()))
    title_style = TextStyle(round(size * 1.12), theme.bar_text, bold=True)
    title_line = ts.shrink_to(Styled.plain(app_title, title_style), TITLE_WIDTH_DP * DP)
    boxes += _draw_bars(image, rng, theme, title_line)
    drawn = [Placed(item.block, item.lines(), item.pitch, item.height, "start") for item in items]
    params: dict[str, Any] = {
        "size_px": [SCREEN_WIDTH, height],
        "theme": theme.name,
        "doodles": doodles,
        "fonts": fonts.to_params(),
        "font_size": size,
        "spacing": round(styles.spacing, 3),
        "time_corner": end,
        "app_title": "title block" if title_block else "generic",
        "speakers": len(names),
        "blocks": block_params(drawn + ([Placed(title_block, (title_line,), 0, 0, "start")] if title_block else [])),
    }
    ppm = SCREEN_DPI / MM_PER_INCH
    return Sheet("chat", [Face(image, boxes)], ppm, size / ppm, params)


def message_parts(text: str) -> tuple[str, str, str] | None:
    """The time, sender and message of a ``[hh:mm] Name: message`` line; None for other text."""
    match = MESSAGE.match(text)
    return None if match is None else (match["time"], match["name"].strip(), match["body"])


def _bubble(
    ts: Typesetter,
    block: Block,
    parts: tuple[str, str, str],
    styles: _Styles,
    names: dict[str, RGB],
    me: str | None,
    end: str,
) -> _Item:
    """Lay out a message as its sender's name, the message and the time, each drawn once.

    The time shares the last line when it fits beside it in the ``end`` corner, and takes a row of its own otherwise.
    """
    stamp, sender, message = parts
    side = "right" if sender == me else "left"
    inner = styles.max_width - 2 * styles.pad[0]
    name_style = TextStyle(round(styles.body.size * NAME_SHARE), names[sender], bold=True)
    name = ts.shrink_to(Styled.plain(sender, name_style), inner)
    body = tuple(ts.fit(Styled.plain(message, styles.body), inner, min_scale=1.0)) if message else ()
    time = ts.layout(Styled.plain(stamp, styles.time), "ltr")
    pitch = styles.body.size * styles.spacing
    trail = time.width + TIME_GAP_DP * DP + (TICKS_DP * 2 * DP if side == "right" else 0)
    last = body[-1].width if body else 0.0
    starts_at_end = bool(body) and (body[-1].direction == "rtl") == (end == "right")
    time_row = not body or starts_at_end or last + trail > inner
    width = max(name.width, *(line.width for line in body), trail if time_row else last + trail) + 2 * styles.pad[0]
    height = name.size * HEADER_SPACING + pitch * len(body) + (time.size * HEADER_SPACING if time_row else 0)
    return _Item(block, side, width, height + 2 * styles.pad[1], body, pitch, name, time, time_row)


def _plain(ts: Typesetter, block: Block, styles: _Styles, me: str | None) -> _Item:
    """A day chip, a notice or a message without a timestamp, drawn as one text."""
    bubble = block.kind == "message"
    style = styles.body if bubble else styles.chip
    lines = tuple(ts.fit(Styled.plain(block.text, style), styles.max_width - 2 * styles.pad[0], min_scale=1.0))
    pitch = style.size * styles.spacing
    width = max(line.width for line in lines) + 2 * styles.pad[0]
    side = ("right" if _speaker(block.text) == me else "left") if bubble else "center"
    return _Item(block, side, width, pitch * len(lines) + 2 * styles.pad[1], lines, pitch)


def _name_colours(blocks: Sequence[Block], theme: Theme) -> dict[str, RGB]:
    """A header colour for each sender, in order of first appearance."""
    palette = NAME_COLOURS_DARK if theme.name == "dark" else NAME_COLOURS
    senders: dict[str, RGB] = {}
    for block in blocks:
        name = _speaker(block.text) if block.kind == "message" else None
        if name is not None and name not in senders:
            senders[name] = palette[len(senders) % len(palette)]
    return senders


def _speaker(text: str) -> str | None:
    match = SPEAKER.match(text)
    return match.group(1).strip() if match else None


def _pick_me(blocks: Sequence[Block], rng: random.Random) -> str | None:
    """The speaker whose bubbles sit on the right: the first or second distinct speaker."""
    speakers: list[str] = []
    for block in blocks:
        name = _speaker(block.text) if block.kind == "message" else None
        if name is not None and name not in speakers:
            speakers.append(name)
    return rng.choice(speakers[:2]) if speakers else None


def _draw_item(
    draw: ImageDraw.ImageDraw, item: _Item, theme: Theme, top: float, pad: tuple[float, float], end: str
) -> list[Box]:
    """Draw a bubble or a chip and return the box of every text line; the time goes in the ``end`` corner."""
    if not item.body and item.name is None:
        return []
    if item.side == "right":
        x1 = SCREEN_WIDTH - EDGE_DP * DP
        x0 = x1 - item.width
    elif item.side == "left":
        x0 = EDGE_DP * DP
        x1 = x0 + item.width
    else:
        x0 = (SCREEN_WIDTH - item.width) / 2
        x1 = x0 + item.width
    fill = {"right": theme.outgoing, "left": theme.incoming}.get(item.side, theme.chip)
    draw.rounded_rectangle((x0, top, x1, top + item.height), radius=8 * DP, fill=fill)
    if item.side == "right":
        draw.polygon([(x1 - 8 * DP, top), (x1 + 7 * DP, top), (x1, top + 9 * DP)], fill=fill)
    elif item.side == "left":
        draw.polygon([(x0 + 8 * DP, top), (x0 - 7 * DP, top), (x0, top + 9 * DP)], fill=fill)
    left, right = x0 + pad[0], x1 - pad[0]
    boxes: list[Box] = []
    y = top + pad[1]
    if item.name is not None:
        pitch = item.name.size * HEADER_SPACING
        boxes.append(_put(draw, item.name, line_left(item.name, left, right, "start"), y, pitch))
        y += pitch
    for line in item.body:
        boxes.append(_put(draw, line, line_left(line, left, right, "start"), y, item.pitch))
        y += item.pitch
    if item.time is not None:
        ticks = TICKS_DP * 2 * DP if item.side == "right" else 0.0
        row = y if item.time_row else y - item.pitch
        row_pitch = item.time.size * HEADER_SPACING if item.time_row else item.pitch
        time_y = row + row_pitch - item.time.size * HEADER_SPACING
        x = right - ticks - item.time.width if end == "right" else left
        boxes.append(_put(draw, item.time, x, time_y, item.time.size * HEADER_SPACING))
        if ticks:
            _ticks(draw, x + item.time.width + ticks, time_y + item.time.size * 1.1, theme)
    return boxes


def _put(draw: ImageDraw.ImageDraw, line: Line, x: float, top: float, pitch: float) -> Box:
    draw_line(draw, line, x, baseline(top, pitch, line.size))
    return (x, top, x + line.width, top + pitch)


def _ticks(draw: ImageDraw.ImageDraw, right: float, bottom: float, theme: Theme) -> None:
    color = (83, 189, 235) if theme.name != "light-blue" else (120, 130, 140)
    for shift in (0.0, 4 * DP):
        x = right - 12 * DP + shift
        draw.line(
            [(x, bottom - 3 * DP), (x + 3 * DP, bottom), (x + 8 * DP, bottom - 6 * DP)], fill=color, width=round(DP)
        )


def _doodles(draw: ImageDraw.ImageDraw, rng: random.Random, theme: Theme, height: int) -> None:
    """Faint wallpaper shapes: rings, dots and short strokes."""
    for _ in range(int(height * SCREEN_WIDTH / 9000)):
        x, y = rng.uniform(0, SCREEN_WIDTH), rng.uniform(0, height)
        r = rng.uniform(4, 14) * DP / 3
        shape = rng.random()
        if shape < 0.4:
            draw.ellipse((x - r, y - r, x + r, y + r), outline=theme.doodle, width=2)
        elif shape < 0.7:
            draw.ellipse((x - r / 3, y - r / 3, x + r / 3, y + r / 3), fill=theme.doodle)
        else:
            draw.line((x, y, x + rng.uniform(-2, 2) * r, y + rng.uniform(-2, 2) * r), fill=theme.doodle, width=2)


def _draw_bars(image: Image.Image, rng: random.Random, theme: Theme, title: Line) -> list[Box]:
    """Status bar icons, app bar with back arrow, avatar and title, and the empty input bar."""
    draw = ImageDraw.Draw(image)
    height = image.height
    status, bar = STATUS_DP * DP, BAR_DP * DP
    draw.rectangle((0, 0, SCREEN_WIDTH, status + bar), fill=theme.bar)
    icon = theme.bar_text
    right = SCREEN_WIDTH - 16 * DP
    draw.rounded_rectangle((right - 22 * DP, 7 * DP, right, 17 * DP), radius=2 * DP, outline=icon, width=round(DP))
    draw.rectangle((right - 20 * DP, 9 * DP, right - 20 * DP + rng.uniform(4, 18) * DP, 15 * DP), fill=icon)
    for k in range(4):
        x = right - 50 * DP + k * 5 * DP
        draw.rectangle((x, 17 * DP - (k + 1) * 2.5 * DP, x + 3 * DP, 17 * DP), fill=icon)
    mid = status + bar / 2
    draw.line([(26 * DP, mid - 8 * DP), (18 * DP, mid), (26 * DP, mid + 8 * DP)], fill=icon, width=round(2 * DP))
    avatar = (36 * DP, mid - 18 * DP, 72 * DP, mid + 18 * DP)
    mask = Image.new("L", (round(avatar[2] - avatar[0]), round(avatar[3] - avatar[1])), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, mask.width - 1, mask.height - 1), fill=255)
    silhouette(image, avatar, rng, mask)
    for k in range(3):
        cy = mid - 7 * DP + k * 7 * DP
        draw.ellipse((SCREEN_WIDTH - 22 * DP, cy - 2 * DP, SCREEN_WIDTH - 18 * DP, cy + 2 * DP), fill=icon)
    x = TITLE_DP * DP
    draw_line(draw, title, x, baseline(mid - title.size * 0.7, title.size * 1.4, title.size))
    input_top = height - INPUT_DP * DP
    draw.rectangle((0, input_top, SCREEN_WIDTH, height), fill=theme.wallpaper)
    draw.rounded_rectangle(
        (8 * DP, input_top + 8 * DP, SCREEN_WIDTH - 64 * DP, height - 8 * DP), radius=24 * DP, fill=theme.input
    )
    cx, cy = SCREEN_WIDTH - 34 * DP, input_top + INPUT_DP * DP / 2
    draw.ellipse((cx - 24 * DP, cy - 24 * DP, cx + 24 * DP, cy + 24 * DP), fill=(0, 168, 132))
    draw.polygon([(cx - 8 * DP, cy - 9 * DP), (cx + 10 * DP, cy), (cx - 8 * DP, cy + 9 * DP)], fill=(255, 255, 255))
    return [(x, mid - title.size * 0.7, x + title.width, mid + title.size * 0.7)]
