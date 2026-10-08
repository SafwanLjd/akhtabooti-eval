"""Check digits and machine-readable zones for synthetic identity documents (ICAO 9303, ISO 13616)."""

from __future__ import annotations

import datetime as dt
import re

MRZ_WEIGHTS = (7, 3, 1)
TD1_WIDTH = 30
TD3_WIDTH = 44
MRZ_FILLER = "<"
IBAN_MODULUS = 97
IBAN_CHECK_BASE = 98
_LETTER_OFFSET = ord("A") - 10


def mrz_value(char: str) -> int:
    """Numeric value of one MRZ character: digits as is, A=10 ... Z=35, filler 0."""
    if char.isdigit():
        return int(char)
    if char == MRZ_FILLER:
        return 0
    if "A" <= char <= "Z":
        return ord(char) - _LETTER_OFFSET
    raise ValueError(f"character {char!r} is not allowed in an MRZ")


def mrz_check_digit(data: str) -> str:
    """ICAO 9303 check digit: weights 7, 3, 1 repeated, sum modulo 10."""
    total = sum(mrz_value(char) * MRZ_WEIGHTS[i % 3] for i, char in enumerate(data))
    return str(total % 10)


def mrz_field(text: str, width: int) -> str:
    """Upper-case ``text``, replace spaces and punctuation with fillers and pad or cut it to ``width``."""
    cleaned = re.sub(r"[^A-Z0-9]", MRZ_FILLER, text.upper())
    return cleaned.ljust(width, MRZ_FILLER)[:width]


def mrz_name(surname: str, given_names: str, width: int) -> str:
    """Name field: SURNAME<<GIVEN<NAMES, padded with fillers."""
    return mrz_field(f"{surname}  {given_names}", width)


def mrz_date(day: dt.date) -> str:
    return day.strftime("%y%m%d")


def td1_lines(
    *,
    issuer: str,
    document_number: str,
    optional1: str,
    birth: dt.date,
    sex: str,
    expiry: dt.date,
    nationality: str,
    surname: str,
    given_names: str,
) -> tuple[str, str, str]:
    """The three 30-character lines of an ID-card MRZ (TD1)."""
    number = mrz_field(document_number, 9)
    line1 = f"ID{mrz_field(issuer, 3)}{number}{mrz_check_digit(number)}{mrz_field(optional1, 15)}"
    born, expires = mrz_date(birth), mrz_date(expiry)
    head = f"{born}{mrz_check_digit(born)}{sex}{expires}{mrz_check_digit(expires)}{mrz_field(nationality, 3)}"
    head += MRZ_FILLER * 11
    composite = mrz_check_digit(line1[5:30] + head[0:7] + head[8:15] + head[18:29])
    line3 = mrz_name(surname, given_names, TD1_WIDTH)
    return line1, head + composite, line3


def td3_lines(
    *,
    issuer: str,
    document_number: str,
    birth: dt.date,
    sex: str,
    expiry: dt.date,
    nationality: str,
    personal_number: str,
    surname: str,
    given_names: str,
) -> tuple[str, str]:
    """The two 44-character lines of a passport MRZ (TD3)."""
    line1 = f"P<{mrz_field(issuer, 3)}{mrz_name(surname, given_names, TD3_WIDTH - 5)}"
    number = mrz_field(document_number, 9)
    born, expires = mrz_date(birth), mrz_date(expiry)
    personal = mrz_field(personal_number, 14)
    number_part = number + mrz_check_digit(number)
    birth_part = born + mrz_check_digit(born)
    expiry_part = expires + mrz_check_digit(expires)
    personal_part = personal + mrz_check_digit(personal)
    composite = mrz_check_digit(number_part + birth_part + expiry_part + personal_part)
    line2 = f"{number_part}{mrz_field(nationality, 3)}{birth_part}{sex}{expiry_part}{personal_part}{composite}"
    return line1, line2


def _iban_digits(text: str) -> int:
    return int("".join(str(mrz_value(char)) if char.isalpha() else char for char in text))


def iban_check_digits(country: str, bban: str) -> str:
    """ISO 13616 check digits for ``country`` + ``bban``."""
    remainder = _iban_digits(bban + country + "00") % IBAN_MODULUS
    return f"{IBAN_CHECK_BASE - remainder:02d}"


def iban_is_valid(iban: str) -> bool:
    """True when the mod-97 check of ``iban`` (spaces allowed) passes."""
    compact = iban.replace(" ", "").upper()
    if len(compact) < 5 or not compact.isalnum():
        return False
    return _iban_digits(compact[4:] + compact[:4]) % IBAN_MODULUS == 1


def group_iban(iban: str) -> str:
    """Print an IBAN in groups of four characters."""
    return " ".join(iban[i : i + 4] for i in range(0, len(iban), 4))
