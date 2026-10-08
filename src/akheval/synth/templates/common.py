"""Render styles and blocks that several templates share."""

from __future__ import annotations

from collections.abc import Sequence

from ..builder import Part
from ..context import Ctx
from ..pools import OrgSpec
from ..values import Person

PAGE = "page"
ID_CARD = "id_card"
PASSPORT = "passport"
CHAT = "chat"

FOOTER_SHARE = 0.5
"""Share of pages that end with a footer line such as a page number or a form code."""
FORM_CODES = ("HR", "FIN", "ADM", "PRG")

Field = tuple[str, Sequence[Part]]


def letterhead(c: Ctx, org: OrgSpec) -> None:
    """Organisation name, address and contact line at the top of a page."""
    c.b.title(c.org(org))
    c.b.header(c.address())
    c.b.header(c.lbl("هاتف", "Tel."), ": ", c.landline(), " | ", c.lbl("البريد", "Email"), ": ", c.org_email(org))


def signature(c: Ctx, person: Person, role: str, org: OrgSpec) -> None:
    c.b.para(c.t("مع الاحترام،", "Kind regards,", c.pick("مع الاحترام،", "Best regards,")))
    c.b.para(c.person(person, parts=c.pick(2, 3)))
    c.b.para(role, "، " if not c.en else ", ", c.org(org))


def dated_reference(c: Ctx) -> None:
    c.b.field(c.lbl("التاريخ", "Date"), c.date())
    c.b.field(c.lbl("الرقم", "Ref."), c.reference())


def joined(c: Ctx, *parts: Part) -> list[Part]:
    """Join parts with the list separator of the document's language."""
    sep = ", " if c.en else "، "
    out: list[Part] = []
    for i, part in enumerate(parts):
        if i:
            out.append(sep)
        out.append(part)
    return out


def shuffled_fields(c: Ctx, fields: Sequence[Field]) -> None:
    """Write "Label: value" lines in a random order."""
    order = list(fields)
    c.rng.shuffle(order)
    for label, value in order:
        c.b.field(label, *value)


def footer(c: Ctx) -> None:
    """End half of the pages with a page number, a confidentiality mark, a form code or a print date."""
    if c.rng.random() >= FOOTER_SHARE:
        return
    form = f"{c.pick(*FORM_CODES)}-{c.rng.randint(1, 40):02d} rev. {c.rng.randint(1, 9)}"
    lines: list[list[Part]] = [
        [c.num(c.t("صفحة 1 من 1", "Page 1 of 1", "Page 1 of 1 / صفحة 1 من 1"))],
        [c.t("سري - للاستخدام الداخلي", "CONFIDENTIAL - internal use", "سري / CONFIDENTIAL")],
        [c.t("نموذج رقم ", "Form ", "Form / نموذج "), form],
        [c.t("طبع بتاريخ ", "Printed on ", "Printed on / طبع بتاريخ "), c.date(c.v.past_date(30))],
    ]
    c.b.blank()
    c.b.para(*c.pick(*lines))
