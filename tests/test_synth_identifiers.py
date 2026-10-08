import datetime as dt
import random

import pytest

from akheval.synth import pools
from akheval.synth.identifiers import (
    group_iban,
    iban_check_digits,
    iban_is_valid,
    mrz_check_digit,
    td1_lines,
    td3_lines,
)
from akheval.synth.values import REFERENCE_DATE, Values, add_years

ICAO_BIRTH = dt.date(1974, 8, 12)
ICAO_EXPIRY = dt.date(2012, 4, 15)


def test_mrz_check_digit_matches_icao_worked_example() -> None:
    assert mrz_check_digit("L898902C3") == "6"
    assert mrz_check_digit("740812") == "2"
    assert mrz_check_digit("120415") == "9"


def test_td3_lines_match_icao_specimen() -> None:
    line1, line2 = td3_lines(
        issuer="UTO",
        document_number="L898902C3",
        birth=ICAO_BIRTH,
        sex="F",
        expiry=ICAO_EXPIRY,
        nationality="UTO",
        personal_number="ZE184226B",
        surname="Eriksson",
        given_names="Anna Maria",
    )
    assert line1 == "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    assert line2 == "L898902C36UTO7408122F1204159ZE184226B<<<<<10"


def test_td1_lines_match_icao_specimen() -> None:
    line1, line2, line3 = td1_lines(
        issuer="UTO",
        document_number="D23145890",
        optional1="",
        birth=ICAO_BIRTH,
        sex="F",
        expiry=ICAO_EXPIRY,
        nationality="UTO",
        surname="Eriksson",
        given_names="Anna Maria",
    )
    assert line1 == "IDUTOD231458907<<<<<<<<<<<<<<<"
    assert line2 == "7408122F1204159UTO<<<<<<<<<<<6"
    assert line3 == "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"


def test_mrz_rejects_lower_case_input() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        mrz_check_digit("abc")


def test_iban_check_digits_reproduce_a_published_example() -> None:
    assert iban_check_digits("GB", "WEST12345698765432") == "82"
    assert iban_is_valid("GB82 WEST 1234 5698 7654 32")
    assert not iban_is_valid("GB83 WEST 1234 5698 7654 32")


@pytest.mark.parametrize("seed", range(20))
def test_generated_ibans_pass_and_invalid_ibans_fail(seed: int) -> None:
    values = Values(random.Random(seed))
    iban = values.iban()
    assert iban_is_valid(iban)
    assert len(iban.replace(" ", "")) == 30
    assert iban.startswith("JO")
    assert not iban_is_valid(values.invalid_iban())


def test_group_iban_uses_blocks_of_four() -> None:
    assert group_iban("JO94CBJO0010000000000131000302") == "JO94 CBJO 0010 0000 0000 0131 0003 02"


def test_add_years_keeps_the_day_and_clamps_29_february() -> None:
    assert add_years(dt.date(2016, 3, 30), 10) == dt.date(2026, 3, 30)
    assert add_years(dt.date(2024, 2, 29), 1) == dt.date(2025, 2, 28)
    assert add_years(dt.date(2026, 10, 1), -16) == dt.date(2010, 10, 1)


@pytest.mark.parametrize("seed", range(200))
def test_birth_date_gives_full_years_of_age(seed: int) -> None:
    born = Values(random.Random(seed)).birth_date(min_age=16, max_age=17)
    assert add_years(born, 16) < REFERENCE_DATE < add_years(born, 18)


@pytest.mark.parametrize("seed", range(50))
def test_children_carry_the_fathers_family_name(seed: int) -> None:
    values = Values(random.Random(seed))
    mother = values.person(female=True)
    members = values.relatives(mother, len(pools.RELATIONS) - 1)
    children = [p for r, p in members if r.generation < 0]
    husbands = [p for r, p in members if r.name.en == "Husband"]
    siblings = [p for r, p in members if r.generation == 0 and r.name.en != "Husband"]
    assert len({p.family for p in children + husbands}) <= 1
    assert all(p.family != mother.family for p in children + husbands)
    assert all(p.family == mother.family for p in siblings)
