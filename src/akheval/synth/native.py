"""Native renderings: files with a text layer (txt, json, csv, html, docx, xlsx, and pdf/odt through LibreOffice).

Every writer draws the document's layout blocks, so each file holds every line of
``Document.text``. PDF and ODT files are LibreOffice conversions of the DOCX
writer's output, made in one batch call per format.
"""

from __future__ import annotations

import csv
import html
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import docx
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Emu, Length, Pt
from docx.text.paragraph import Paragraph
from openpyxl import Workbook

from ..dataset import Document, Rendering
from .builder import CELL_SEPARATOR, FIELD_SEPARATOR, Blocks

CAPTURE = "digital"
CHANNEL = "native"
LIBREOFFICE_FORMATS = ("pdf", "odt")
ARABIC_SCRIPT = re.compile("[\u0600-\u06ff]")
ARABIC_FONT = "Amiri"
"""Complex-script font for Arabic runs; with it, pdftotext returns whole Arabic words from LibreOffice PDFs."""
LATIN_FONT = "Liberation Serif"
MONO_FONT = "Liberation Mono"
MRZ_FONT_SIZE = Pt(10)
TABLE_FONT_SIZE = Pt(8)
LANDSCAPE_MIN_COLUMNS = 5
"""Tables this wide get a landscape page so e-mail addresses fit their cells without wrapping."""
MIN_COLUMN_CHARS = 4
COLUMN_PADDING_CHARS = 3
"""Characters added to every column's width share to cover cell margins and wide Arabic-Indic digits."""
SOFFICE_VERSION_TIMEOUT = 120
WRITERS = {
    "txt": "text",
    "json": "json",
    "csv": "csv",
    "html": "html",
    "docx": "python-docx",
    "xlsx": "openpyxl",
    "pdf": "libreoffice from python-docx",
    "odt": "libreoffice from python-docx",
}
HTML_TAGS = {"title": "h1", "header": "h2", "field": "p", "paragraph": "p", "mrz": "pre", "message": "p"}
DOCX_HEADING_LEVELS = {"title": 1, "header": 2}
TABLE_TBLPR_SUCCESSORS = (
    "w:tblStyleRowBandSize", "w:tblStyleColBandSize", "w:tblW", "w:jc", "w:tblCellSpacing", "w:tblInd",
    "w:tblBorders", "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook", "w:tblCaption", "w:tblDescription",
)  # fmt: skip


def native_relpath(doc_id: str, fmt: str) -> str:
    return f"files/{doc_id}/{CAPTURE}.{fmt}"


def is_rtl(language: str) -> bool:
    """Arabic and mixed documents run right to left."""
    return language != "en"


def _cells(block: Mapping[str, Any]) -> list[str]:
    """Table cells of one block: row cells, a field's label and value, or the whole line."""
    text = str(block["text"])
    if block["kind"] == "blank":
        return []
    if block["kind"] == "row":
        return text.split(CELL_SEPARATOR)
    if block["kind"] == "field":
        label, _, value = text.partition(FIELD_SEPARATOR)
        return [label, value]
    return [text]


def _row_groups(blocks: Blocks) -> Iterator[tuple[str, list[Mapping[str, Any]]]]:
    """Consecutive "row" blocks as one ("table", rows) group; every other block alone."""
    rows: list[Mapping[str, Any]] = []
    for block in blocks:
        if block["kind"] == "row":
            rows.append(block)
            continue
        if rows:
            yield "table", rows
            rows = []
        yield str(block["kind"]), [block]
    if rows:
        yield "table", rows


def write_txt(doc: Document, path: Path) -> None:
    path.write_text(doc.text + "\n", encoding="utf-8")


def write_json(doc: Document, path: Path) -> None:
    path.write_text(json.dumps({"lines": doc.text.split("\n")}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(doc: Document, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerows(_cells(block) for block in doc.meta["layout"])


def write_xlsx(doc: Document, path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    if sheet is None:
        raise RuntimeError("openpyxl created a workbook without a sheet")
    sheet.title = "Document"
    sheet.sheet_view.rightToLeft = is_rtl(doc.language)
    for block in doc.meta["layout"]:
        sheet.append(_cells(block))
    workbook.save(path)


def write_html(doc: Document, path: Path) -> None:
    body: list[str] = []
    for kind, blocks in _row_groups(doc.meta["layout"]):
        if kind == "table":
            rows = ("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in _cells(b)) + "</tr>" for b in blocks)
            body.append('<table dir="auto">' + "".join(rows) + "</table>")
        elif kind == "blank":
            body.append("<br>")
        else:
            tag = HTML_TAGS[kind]
            body.append(f'<{tag} dir="auto">{html.escape(str(blocks[0]["text"]))}</{tag}>')
    direction = "rtl" if is_rtl(doc.language) else "ltr"
    lang = "en" if doc.language == "en" else "ar"
    page = [
        "<!DOCTYPE html>",
        f'<html lang="{lang}" dir="{direction}">',
        '<head><meta charset="utf-8"></head>',
        "<body>",
        *body,
        "</body>",
        "</html>",
    ]
    path.write_text("\n".join(page) + "\n", encoding="utf-8")


def _format_runs(paragraph: Paragraph, *, mono: bool, size: Length | None = None) -> None:
    """Set fonts and sizes and, for Arabic text, right-to-left paragraph and run properties.

    ``size`` applies to complex-script runs too, which Arabic letters and Arabic-Indic digits use.
    """
    rtl = bool(ARABIC_SCRIPT.search(paragraph.text))
    if rtl:
        paragraph._p.get_or_add_pPr().append(OxmlElement("w:bidi"))
        paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in paragraph.runs:
        run.font.name = MONO_FONT if mono else LATIN_FONT
        if rtl:
            run.font.rtl = True
        rpr = run._r.get_or_add_rPr()
        rpr.get_or_add_rFonts().set(qn("w:cs"), ARABIC_FONT)
        if size is not None:
            run.font.size = size
            size_cs = OxmlElement("w:szCs")
            size_cs.set(qn("w:val"), str(round(size.pt * 2)))
            rpr.get_or_add_sz().addnext(size_cs)


def _use_landscape(document: Any) -> None:
    section = document.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width


def _docx_table(document: Any, blocks: Sequence[Mapping[str, Any]], rtl: bool) -> None:
    """A table whose column widths follow the longest cell of each column."""
    rows = [_cells(b) for b in blocks]
    width = max(len(r) for r in rows)
    if width >= LANDSCAPE_MIN_COLUMNS:
        _use_landscape(document)
    section = document.sections[0]
    usable = section.page_width - section.left_margin - section.right_margin
    chars = [
        COLUMN_PADDING_CHARS + max(MIN_COLUMN_CHARS, *(len(r[i]) for r in rows if i < len(r))) for i in range(width)
    ]
    table = document.add_table(rows=len(rows), cols=width)
    table.style = "Table Grid"
    table.autofit = False
    if rtl:
        table._tbl.tblPr.insert_element_before(OxmlElement("w:bidiVisual"), *TABLE_TBLPR_SUCCESSORS)
    for column, n_chars in zip(table.columns, chars, strict=True):
        column.width = Emu(usable * n_chars // sum(chars))
        for cell in column.cells:
            cell.width = column.width
    for row, cells in zip(table.rows, rows, strict=True):
        for cell, value in zip(row.cells, cells, strict=False):
            paragraph = cell.paragraphs[0]
            paragraph.add_run(value)
            _format_runs(paragraph, mono=False, size=TABLE_FONT_SIZE)


def write_docx(doc: Document, path: Path) -> None:
    document = docx.Document()
    for kind, blocks in _row_groups(doc.meta["layout"]):
        if kind == "table":
            _docx_table(document, blocks, is_rtl(doc.language))
            continue
        text = str(blocks[0]["text"])
        if kind in DOCX_HEADING_LEVELS:
            paragraph = document.add_heading(text, level=DOCX_HEADING_LEVELS[kind])
        else:
            paragraph = document.add_paragraph(text)
        _format_runs(paragraph, mono=kind == "mrz", size=MRZ_FONT_SIZE if kind == "mrz" else None)
    document.save(str(path))


DIRECT_WRITERS: dict[str, Callable[[Document, Path], None]] = {
    "txt": write_txt,
    "json": write_json,
    "csv": write_csv,
    "html": write_html,
    "docx": write_docx,
    "xlsx": write_xlsx,
}


def libreoffice_version(soffice: str) -> str:
    result = subprocess.run(
        [soffice, "--version"], capture_output=True, text=True, check=True, timeout=SOFFICE_VERSION_TIMEOUT
    )
    return result.stdout.strip()


def convert_with_libreoffice(sources: Sequence[Path], fmt: str, outdir: Path, *, soffice: str, timeout: int) -> None:
    """Convert ``sources`` to ``fmt`` in one headless LibreOffice call with a private profile."""
    with tempfile.TemporaryDirectory(prefix="akheval-lo-profile-") as profile:
        command = [
            soffice,
            f"-env:UserInstallation={Path(profile).as_uri()}",
            "--headless",
            "--convert-to",
            fmt,
            "--outdir",
            str(outdir),
            *map(str, sources),
        ]
        subprocess.run(command, capture_output=True, check=True, timeout=timeout)
    missing = [s.name for s in sources if not (outdir / f"{s.stem}.{fmt}").is_file()]
    if missing:
        raise RuntimeError(f"LibreOffice did not produce {fmt} for {len(missing)} file(s): {missing[:5]}")


def write_native_files(
    documents: Sequence[Document],
    formats: Mapping[str, str],
    root: Path,
    *,
    soffice: str,
    timeout: int,
) -> dict[str, Rendering]:
    """Write one native file per document in the format ``formats[doc.id]``; return the renderings by doc id."""
    renderings: dict[str, Rendering] = {}
    converted: dict[str, list[Document]] = {fmt: [] for fmt in LIBREOFFICE_FORMATS}
    for doc in documents:
        fmt = formats[doc.id]
        (root / "files" / doc.id).mkdir(parents=True, exist_ok=True)
        if fmt in converted:
            converted[fmt].append(doc)
        else:
            DIRECT_WRITERS[fmt](doc, root / native_relpath(doc.id, fmt))
        renderings[doc.id] = Rendering(
            id=f"{doc.id}.{CAPTURE}.{fmt}",
            path=native_relpath(doc.id, fmt),
            format=fmt,
            channel=CHANNEL,
            capture=CAPTURE,
            params={"writer": WRITERS[fmt]},
        )
    for fmt, docs in converted.items():
        if docs:
            _convert_batch(docs, fmt, root, soffice=soffice, timeout=timeout)
    return renderings


def _convert_batch(docs: Sequence[Document], fmt: str, root: Path, *, soffice: str, timeout: int) -> None:
    with tempfile.TemporaryDirectory(prefix="akheval-native-") as tmp:
        staging, outdir = Path(tmp) / "docx", Path(tmp) / fmt
        staging.mkdir()
        outdir.mkdir()
        sources = []
        for doc in docs:
            source = staging / f"{doc.id}.docx"
            write_docx(doc, source)
            sources.append(source)
        convert_with_libreoffice(sources, fmt, outdir, soffice=soffice, timeout=timeout)
        for doc in docs:
            shutil.move(outdir / f"{doc.id}.{fmt}", root / native_relpath(doc.id, fmt))
