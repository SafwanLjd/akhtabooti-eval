"""Random but plausible values: people, places, organisations, identifiers and look-alikes."""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass

from . import pools
from .identifiers import group_iban, iban_check_digits
from .pools import OrgSpec, Pair, Relation, latin_slug

INDIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
REFERENCE_DATE = dt.date(2026, 10, 1)


def add_years(day: dt.date, years: int) -> dt.date:
    """Shift ``day`` by whole calendar years; 29 February becomes 28 February in common years."""
    try:
        return day.replace(year=day.year + years)
    except ValueError:
        return day.replace(year=day.year + years, day=28)


"""The dataset's fixed "today": every relative date is computed from it, never from the clock."""

MONTHS_EN = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip
MONTHS_LEVANT = (
    "كانون الثاني", "شباط", "آذار", "نيسان", "أيار", "حزيران",
    "تموز", "آب", "أيلول", "تشرين الأول", "تشرين الثاني", "كانون الأول",
)  # fmt: skip
MONTHS_AR = (
    "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
    "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
)  # fmt: skip

LANDLINE_AREAS = ("6", "6", "6", "5", "2", "3")
REFUGEE_OFFICES = ("945", "946", "947")
SY_GOVERNORATE_CODES = range(1, 15)
JO_POSTCODES = range(11100, 11999)
MRZ_SEXES = ("M", "F")


def to_indic(text: str) -> str:
    """Write the ASCII digits of ``text`` as Arabic-Indic digits."""
    return text.translate(INDIC_DIGITS)


@dataclass(frozen=True, slots=True)
class Person:
    given: Pair
    father: Pair
    grandfather: Pair
    family: Pair
    female: bool
    birth: dt.date

    def name(self, script: str, parts: int = 2) -> str:
        """Given name, then ``parts - 2`` ancestor names, then the family name."""
        middle = (self.father, self.grandfather)[: max(parts - 2, 0)]
        return " ".join(getattr(p, script) for p in (self.given, *middle, self.family))

    def given_names(self, script: str, parts: int = 3) -> str:
        """Given name followed by father's and grandfather's names, as on a passport."""
        return " ".join(getattr(p, script) for p in (self.given, self.father, self.grandfather)[: parts - 1])

    @property
    def sex(self) -> str:
        return MRZ_SEXES[self.female]


class Values:
    """Draws every value from one seeded ``random.Random``."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def digits(self, n: int) -> str:
        return "".join(self.rng.choice("0123456789") for _ in range(n))

    def person(self, *, female: bool | None = None, min_age: int = 18, max_age: int = 70) -> Person:
        is_female = self.rng.random() < 0.5 if female is None else female
        father, grandfather = self.rng.sample(pools.MALE_GIVEN, 2)
        return Person(
            given=self.rng.choice(pools.FEMALE_GIVEN if is_female else pools.MALE_GIVEN),
            father=father,
            grandfather=grandfather,
            family=self.rng.choice(pools.FAMILY),
            female=is_female,
            birth=self.birth_date(min_age, max_age),
        )

    def relations(self, of: Person, n: int) -> list[Relation]:
        """``n`` distinct relations of ``of``: a woman gets no wife, a man no husband."""
        excluded = "Wife" if of.female else "Husband"
        return self.rng.sample([r for r in pools.RELATIONS if r.name.en != excluded], n)

    def relatives(self, of: Person, n: int) -> list[tuple[Relation, Person]]:
        """``n`` family members of ``of``. Children carry their father's family name, so a woman's husband
        and children share one family name and her siblings share hers."""
        married_family = self.rng.choice([f for f in pools.FAMILY if f != of.family]) if of.female else of.family
        return [(r, self.relative(of, r, married_family)) for r in self.relations(of, n)]

    def relative(self, of: Person, relation: Relation, married_family: Pair | None = None) -> Person:
        """A family member of ``of`` with a plausible birth date.

        A father carries the names that ``of`` gives for him; spouses and mothers keep their own family name.
        ``married_family`` is the family name of the children of ``of`` (default: that of ``of``).
        """
        other = self.person(female=relation.female)
        gap = dt.timedelta(days=self.rng.randint(18 * 365, 38 * 365))
        if relation.generation > 0:
            birth = of.birth - gap
        elif relation.generation < 0:
            birth = min(of.birth + gap, REFERENCE_DATE - dt.timedelta(days=self.rng.randint(30, 3 * 365)))
        else:
            birth = of.birth + dt.timedelta(days=self.rng.randint(-8 * 365, 8 * 365))
        family = other.family if relation.name.en in pools.OWN_FAMILY_NAME else of.family
        if relation.generation < 0 or relation.name.en == "Husband":
            family = married_family or of.family
        if relation.name.en == "Father":
            return Person(of.father, of.grandfather, other.grandfather, family, False, birth)
        return Person(other.given, other.father, other.grandfather, family, relation.female, birth)

    def org(self, *kinds: str) -> OrgSpec:
        candidates = [o for o in pools.ORGS if not kinds or o.kind in kinds]
        return self.rng.choice(candidates)

    def orgs(self, n: int, *kinds: str) -> list[OrgSpec]:
        candidates = [o for o in pools.ORGS if not kinds or o.kind in kinds]
        return self.rng.sample(candidates, n)

    def city(self, country: str = "JO") -> Pair:
        table = {"JO": pools.JO_CITIES, "SY": pools.SY_CITIES, "PS": pools.PS_CITIES}
        return self.rng.choice(table[country])

    def birth_place(self) -> Pair:
        return self.city(self.rng.choices(("JO", "SY", "PS"), weights=(6, 3, 1))[0])

    def address(self, script: str, country: str = "JO") -> str:
        """A full postal address in Arabic or Latin script; the district matches the city when one is known."""
        capital = pools.JO_CITIES[0] if country == "JO" else pools.SY_CITIES[0]
        city = capital if self.rng.random() < 0.5 else self.city(country)
        districts = pools.DISTRICTS.get(city.en, ())
        district = self.rng.choice(districts) if districts else None
        street = self.rng.choice(pools.STREETS)
        building, floor, flat = self.rng.randint(1, 160), self.rng.randint(1, 6), self.rng.randint(1, 24)
        box, postcode = self.rng.randint(100, 9999), self.rng.choice(JO_POSTCODES)
        if script == "ar":
            area = f"{district.ar}، {city.ar}" if district else city.ar
            forms = [
                f"{area}، {street.ar}، بناية رقم {building}",
                f"{street.ar}، عمارة {building}، الطابق {floor}، {area}",
                f"{city.ar} - {district.ar if district else street.ar} - بناء {building} - شقة {flat}",
            ]
            if country == "JO":
                forms.append(f"ص.ب {box} {city.ar} {postcode}")
        else:
            area = f"{district.en}, {city.en}" if district else city.en
            forms = [
                f"Building {building}, {street.en}, {area}",
                f"{building} {street.en}, {area}",
                f"Apartment {flat}, Building {building}, {street.en}, {area}",
            ]
            if country == "JO":
                forms += [
                    f"P.O. Box {box}, {city.en} {postcode}, Jordan",
                    f"{building} {street.en}, {city.en} {postcode}",
                ]
        return self.rng.choice(forms)

    def mobile(self) -> str:
        """A Jordanian mobile number (077, 078 or 079): local or international, compact, spaced or dashed."""
        rest = self.rng.choice("789") + self.digits(7)
        a, b, c = rest[0], rest[1:4], rest[4:]
        return self.rng.choice(
            [
                f"07{rest}",
                f"07{a} {b} {c}",
                f"07{a}-{b}-{c}",
                f"+9627{rest}",
                f"+962 7{a} {b} {c}",
                f"009627{rest}",
                f"00962 7{a} {b} {c}",
            ]
        )

    def landline(self) -> str:
        area, rest = self.rng.choice(LANDLINE_AREAS), self.digits(7)
        return self.rng.choice(
            [f"0{area} {rest}", f"0{area}-{rest}", f"+962 {area} {rest[:3]} {rest[3:]}", f"0{area}{rest}"]
        )

    def syrian_mobile(self) -> str:
        rest = self.rng.choice("3456789") + self.digits(7)
        return self.rng.choice(
            [
                f"09{rest}",
                f"09{rest[:2]} {rest[2:5]} {rest[5:]}",
                f"+963 9{rest[:2]} {rest[2:5]} {rest[5:]}",
                f"+9639{rest}",
            ]
        )

    def email(self, given: str, family: str, domain: str | None = None) -> str:
        """An address whose local part comes from a Latin given and family name."""
        g, f = latin_slug(given), latin_slug(family)
        year = self.rng.randint(70, 99) if self.rng.random() < 0.5 else self.rng.randint(1980, 2004)
        local = self.rng.choice(
            [f"{g}.{f}", f"{g[0]}.{f}", f"{g}{f}", f"{g}_{f}{self.rng.randint(1, 99)}", f"{f}.{g}{year}", f"{g}{year}"]
        )
        return f"{local}@{domain or self.rng.choice(pools.WEBMAIL_DOMAINS)}"

    def org_email(self, org: OrgSpec) -> str:
        return f"{self.rng.choice(['info', 'contact', 'office', 'admin', 'finance', 'hr'])}@{org.domain}"

    def iban(self, bank: OrgSpec | None = None, *, grouped: bool | None = None) -> str:
        """A Jordanian IBAN with valid check digits."""
        code = (bank or self.org("bank")).bank_code
        bban = f"{code}{self.digits(4)}{self.digits(18)}"
        iban = f"JO{iban_check_digits('JO', bban)}{bban}"
        group = self.rng.random() < 0.5 if grouped is None else grouped
        return group_iban(iban) if group else iban

    def invalid_iban(self) -> str:
        """A Jordanian IBAN whose check digits fail the mod-97 test."""
        valid = self.iban(grouped=False)
        wrong = f"{(int(valid[2:4]) + self.rng.randint(1, 96)) % 97:02d}"
        if wrong == valid[2:4]:
            wrong = f"{(int(wrong) + 1) % 97:02d}"
        iban = f"JO{wrong}{valid[4:]}"
        return group_iban(iban) if self.rng.random() < 0.5 else iban

    def jo_national_id(self, person: Person) -> str:
        """Ten digits: 9 and the birth year for people born before 2000, otherwise 20 and the year."""
        year, sex = person.birth.year, "2" if person.female else "1"
        if year < 2000:
            return f"9{year % 100:02d}{sex}{self.digits(6)}"
        return f"20{year % 100:02d}{sex}{self.digits(5)}"

    def sy_national_id(self) -> str:
        """Eleven digits starting with a two-digit governorate code."""
        return f"{self.rng.choice(SY_GOVERNORATE_CODES):02d}{self.digits(9)}"

    def jo_card_number(self) -> str:
        letters = "".join(self.rng.choice("ABCDEFGHJKLMNPRSTUVXYZ") for _ in range(3))
        return f"{letters}{self.digits(5)}"

    def passport_number(self) -> str:
        return f"{self.rng.choice('NPRKLT')}{self.digits(self.rng.choice((6, 7)))}"

    def refugee_case_number(self) -> str:
        return f"{self.rng.choice(REFUGEE_OFFICES)}-{self.rng.randint(13, 26):02d}C{self.digits(5)}"

    def refugee_individual_number(self, case: str | None = None) -> str:
        office = case[:3] if case else self.rng.choice(REFUGEE_OFFICES)
        return f"{office}-{self.digits(8)}"

    def moi_card_number(self) -> str:
        letters = "".join(self.rng.choice("ABCDEFGHJK") for _ in range(2))
        return f"{letters}{self.digits(7)}"

    def ipv4(self) -> str:
        r = self.rng.randint
        return self.rng.choice(
            [
                f"10.{r(0, 255)}.{r(0, 255)}.{r(1, 254)}",
                f"192.168.{r(0, 255)}.{r(1, 254)}",
                f"172.{r(16, 31)}.{r(0, 255)}.{r(1, 254)}",
                f"192.0.2.{r(1, 254)}",
                f"198.51.100.{r(1, 254)}",
                f"203.0.113.{r(1, 254)}",
            ]
        )

    def ipv6(self) -> str:
        groups = [f"{self.rng.randint(0, 0xFFFF):x}" for _ in range(4)]
        return self.rng.choice(
            [
                f"2001:db8:{groups[0]}:{groups[1]}::{groups[2]}",
                f"fe80::{groups[0]}:{groups[1]}:{groups[2]}:{groups[3]}",
                "2001:0db8:" + ":".join(f"{int(g, 16):04x}" for g in groups) + ":0000:0001",
                f"fd12:3456:{groups[0]}::{groups[1]}",
            ]
        )

    def version_number(self) -> str:
        """Four dot-separated numbers that read like an IPv4 address."""
        r = self.rng.randint
        return f"{r(1, 12)}.{r(0, 9)}.{r(0, 20)}.{r(1, 250)}"

    def birth_date(self, min_age: int = 18, max_age: int = 70) -> dt.date:
        """A birth date that makes the person ``min_age`` to ``max_age`` full years old on REFERENCE_DATE.

        The person passed ``min_age`` at least one day earlier, so a document issued at that age can predate
        REFERENCE_DATE.
        """
        latest = add_years(REFERENCE_DATE, -min_age) - dt.timedelta(days=1)
        earliest = add_years(REFERENCE_DATE, -(max_age + 1)) + dt.timedelta(days=1)
        return earliest + dt.timedelta(days=self.rng.randint(0, (latest - earliest).days))

    def past_date(self, max_days: int = 3 * 365) -> dt.date:
        return REFERENCE_DATE - dt.timedelta(days=self.rng.randint(1, max_days))

    def future_date(self, max_days: int = 6 * 365) -> dt.date:
        return REFERENCE_DATE + dt.timedelta(days=self.rng.randint(30, max_days))

    def format_date(self, day: dt.date, lang: str) -> str:
        """Write ``day`` in one of the styles common in documents of ``lang``."""
        styles = [f"{day:%d/%m/%Y}", f"{day:%d-%m-%Y}", f"{day:%Y-%m-%d}", f"{day.day}/{day.month}/{day.year}"]
        if lang == "en":
            styles += [
                f"{day.day} {MONTHS_EN[day.month - 1]} {day.year}",
                f"{MONTHS_EN[day.month - 1]} {day.day}, {day.year}",
            ]
        else:
            styles += [
                f"{day.day} {MONTHS_LEVANT[day.month - 1]} {day.year}",
                f"{day.day} {MONTHS_AR[day.month - 1]} {day.year}",
            ]
        return self.rng.choice(styles)

    def amount(self, lang: str, fils: int | None = None) -> str:
        """Write ``fils`` thousandths of a dinar as a JOD amount; draw a random amount when it is None."""
        if fils is None:
            fils = self.rng.randint(15, 25000) * 1000 + self.rng.choice((0, 0, 250, 500, 750))
        value, rest = divmod(fils, 1000)
        if lang == "en":
            exact, whole = [f"{value:,}.{rest:03d} JOD", f"JOD {value:,}.{rest:03d}"], [f"JD {value:,}"]
        else:
            exact, whole = [f"{value:,}.{rest:03d} د.أ"], [f"{value:,} دينار", f"{value} دينار أردني"]
        return self.rng.choice(exact + whole if rest == 0 else exact)

    def invoice_number(self) -> str:
        year = self.rng.randint(2024, 2026)
        return self.rng.choice(
            [
                f"INV-{year}-{self.digits(5)}",
                f"{year}/{self.digits(4)}",
                f"F-{self.digits(6)}",
                f"TI{year % 100}{self.digits(5)}",
            ]
        )

    def reference_number(self) -> str:
        year = self.rng.randint(2024, 2026)
        prefix = self.rng.choice(["REF", "HR", "ADM", "FIN", "PRG"])
        return self.rng.choice(
            [f"{prefix}/{year}/{self.digits(4)}", f"{self.digits(4)}-{self.digits(4)}", f"{year}/{self.digits(3)}"]
        )

    def order_number(self) -> str:
        """Ten digits; half of them start with 07 like a Jordanian mobile number."""
        if self.rng.random() < 0.5:
            return f"07{self.rng.choice('789')}{self.digits(7)}"
        return f"{self.rng.randint(1, 9)}{self.digits(9)}"

    def nine_digit_number(self) -> str:
        """Nine digits starting with 9: one digit short of a Jordanian national number."""
        return f"9{self.digits(8)}"

    def tax_number(self) -> str:
        return self.digits(self.rng.choice((8, 9)))
