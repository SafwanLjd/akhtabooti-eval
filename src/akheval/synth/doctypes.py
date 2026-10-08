"""Document types: the template that writes each one, its native file formats and its image renderings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .context import Ctx
from .templates import digital, identity, prose, records

PROSE_FORMATS = ("docx", "pdf", "odt", "html", "txt")
TABLE_FORMATS = ("xlsx", "csv", "pdf", "docx")
CHAT_FORMATS = ("txt", "json", "html")
LOG_FORMATS = ("txt", "html", "json", "pdf")
CARD_FORMATS = ("pdf", "docx")

CARD_IMAGES = "card"
PAGE_IMAGES = "page"
CHAT_IMAGES = "chat"
NO_IMAGES = "none"

Template = Callable[[Ctx], str]


@dataclass(frozen=True, slots=True)
class DocType:
    name: str
    template: Template
    native_formats: tuple[str, ...]
    images: str


DOC_TYPES: dict[str, DocType] = {
    t.name: t
    for t in (
        DocType("letter", prose.letter, PROSE_FORMATS, PAGE_IMAGES),
        DocType("hr_record", records.hr_record, PROSE_FORMATS, PAGE_IMAGES),
        DocType("invoice", records.invoice, TABLE_FORMATS, PAGE_IMAGES),
        DocType("contact_list", records.contact_list, TABLE_FORMATS, PAGE_IMAGES),
        DocType("jo_id_card", identity.jo_id_card, CARD_FORMATS, CARD_IMAGES),
        DocType("jo_passport", identity.jo_passport, CARD_FORMATS, CARD_IMAGES),
        DocType("sy_id_card", identity.sy_id_card, CARD_FORMATS, CARD_IMAGES),
        DocType("refugee_certificate", identity.refugee_certificate, CARD_FORMATS, CARD_IMAGES),
        DocType("chat", digital.chat, CHAT_FORMATS, CHAT_IMAGES),
        DocType("press_note", prose.press_note, PROSE_FORMATS, NO_IMAGES),
        DocType("cv", prose.cv, PROSE_FORMATS, PAGE_IMAGES),
        DocType("it_incident", digital.it_incident, LOG_FORMATS, NO_IMAGES),
        DocType("notice_no_pii", digital.notice_no_pii, PROSE_FORMATS, PAGE_IMAGES),
    )
}
