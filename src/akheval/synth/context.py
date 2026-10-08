"""Per-document generation context: language rules, the value source and the layout builder."""

from __future__ import annotations

import datetime as dt
import random
from collections.abc import Sequence
from typing import Any, TypeVar

from .builder import Ent, LayoutBuilder, Neg, Part
from .pools import OrgSpec, Pair
from .values import Person, Values, to_indic

T = TypeVar("T")

LATIN_ENTITY_SHARE_MIXED = 0.7
"""Share of entities written in Latin script inside mixed-language documents."""
INDIC_DIGIT_SHARE_AR = 0.35
"""Share of Arabic documents that write numbers with Arabic-Indic digits."""
MIXED_PHRASING_SHARE = 0.7
"""Share of mixed-language paragraphs written with a code-switching phrasing when the template has one."""

HONORIFICS = {
    ("ar", False): ("السيد", "الأستاذ", "د.", "م."),
    ("ar", True): ("السيدة", "الآنسة", "د.", "م."),
    ("en", False): ("Mr.", "Dr.", "Eng."),
    ("en", True): ("Ms.", "Mrs.", "Dr.", "Eng."),
}


class Ctx:
    """Everything a template needs to write one document in one language."""

    def __init__(self, lang: str, rng: random.Random) -> None:
        self.lang = lang
        self.rng = rng
        self.v = Values(rng)
        self.b = LayoutBuilder()
        self.indic = lang == "ar" and rng.random() < INDIC_DIGIT_SHARE_AR
        self.meta: dict[str, Any] = {}

    @property
    def en(self) -> bool:
        """True when the running text is English."""
        return self.lang == "en"

    def t(self, ar: str, en: str, mixed: str | None = None) -> str:
        """Pick the text for this document's language; mixed documents default to the Arabic text."""
        if self.lang == "en":
            return en
        if self.lang == "mixed" and mixed is not None:
            return mixed
        return ar

    def lbl(self, ar: str, en: str) -> str:
        """A field label; mixed documents carry both languages."""
        return self.t(ar, en, f"{ar} / {en}")

    def pick(self, *options: T) -> T:
        return self.rng.choice(options)

    def say(
        self,
        ar: Sequence[Sequence[Part]],
        en: Sequence[Sequence[Part]],
        mixed: Sequence[Sequence[Part]] | None = None,
    ) -> Sequence[Part]:
        """Choose one phrasing among the variants written for this document's language."""
        if self.lang == "en":
            return self.rng.choice(en)
        if self.lang == "mixed" and mixed and self.rng.random() < MIXED_PHRASING_SHARE:
            return self.rng.choice(mixed)
        return self.rng.choice(ar)

    def script(self) -> str:
        """Script for the next entity: Arabic or Latin, mixed documents lean Latin."""
        if self.lang == "mixed":
            return "en" if self.rng.random() < LATIN_ENTITY_SHARE_MIXED else "ar"
        return "en" if self.lang == "en" else "ar"

    def num(self, text: str) -> str:
        """Write digits the way this document writes them."""
        return to_indic(text) if self.indic else text

    def honorific(self, person: Person, script: str | None = None) -> str:
        return self.rng.choice(HONORIFICS[(script or self.script(), person.female)])

    def person(self, person: Person, parts: int | None = None, script: str | None = None) -> Ent:
        n = parts or self.rng.choice((2, 2, 3, 4))
        return Ent(person.name(script or self.script(), n), "PERSON")

    def given(self, person: Person, script: str | None = None) -> Ent:
        return Ent(getattr(person.given, script or self.script()), "PERSON")

    def org(self, org: OrgSpec | Pair, script: str | None = None) -> Ent:
        return Ent(getattr(org, script or self.script()), "ORG")

    def place(self, place: Pair, script: str | None = None) -> Ent:
        return Ent(getattr(place, script or self.script()), "LOCATION")

    def address(self, country: str = "JO", script: str | None = None) -> Ent:
        return Ent(self.num(self.v.address(script or self.script(), country)), "ADDRESS")

    def mobile(self) -> Ent:
        return Ent(self.num(self.v.mobile()), "PHONE")

    def landline(self) -> Ent:
        return Ent(self.num(self.v.landline()), "PHONE")

    def syrian_mobile(self) -> Ent:
        return Ent(self.num(self.v.syrian_mobile()), "PHONE")

    def any_phone(self) -> Ent:
        """Mostly Jordanian mobiles, some landlines and Syrian mobiles."""
        return self.rng.choices((self.mobile, self.landline, self.syrian_mobile), weights=(7, 2, 1))[0]()

    def email(self, person: Person, domain: str | None = None) -> Ent:
        return Ent(self.v.email(person.given.en, person.family.en, domain), "EMAIL")

    def org_email(self, org: OrgSpec) -> Ent:
        return Ent(self.v.org_email(org), "EMAIL")

    def iban(self, bank: OrgSpec | None = None) -> Ent:
        return Ent(self.v.iban(bank), "IBAN")

    def _date_text(self, day: dt.date) -> str:
        return self.num(self.v.format_date(day, "en" if self.lang == "en" else "ar"))

    def date(self, day: dt.date | None = None) -> Neg:
        """An issue, expiry, invoice or other non-personal date: a "date" look-alike."""
        return Neg(self._date_text(day or self.v.past_date()), "date")

    def birth_date(self, person: Person) -> str:
        """A date of birth: personal data outside the taxonomy, so it carries no label and is no look-alike."""
        return self._date_text(person.birth)

    def amount(self, fils: int | None = None) -> Neg:
        lang = self.lang if self.lang != "mixed" else self.pick("ar", "en")
        return Neg(self.num(self.v.amount(lang, fils)), "amount")

    def reference(self) -> Neg:
        return Neg(self.num(self.v.reference_number()), "reference_number")

    def invoice_number(self) -> Neg:
        return Neg(self.num(self.v.invoice_number()), "invoice_number")

    def order_number(self) -> Neg:
        return Neg(self.num(self.v.order_number()), "numeric_sequence")

    def tax_number(self) -> Neg:
        return Neg(self.num(self.v.tax_number()), "numeric_sequence")

    def year(self, year: int) -> Neg:
        return Neg(self.num(str(year)), "date")

    def invalid_iban(self) -> Neg:
        return Neg(self.v.invalid_iban(), "invalid_identifier")

    def nine_digits(self) -> Neg:
        return Neg(self.num(self.v.nine_digit_number()), "invalid_identifier")

    def version(self) -> Neg:
        return Neg(self.v.version_number(), "version_number")
