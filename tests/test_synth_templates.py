import datetime as dt
import random
import re
from collections.abc import Sequence
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from akheval.dataset import LANGUAGES, Dataset, Document, validate
from akheval.synth import pools
from akheval.synth.build import DocSpec, generate_document
from akheval.synth.builder import Ent, Part
from akheval.synth.context import Ctx
from akheval.synth.doctypes import DOC_TYPES
from akheval.synth.identifiers import iban_is_valid, mrz_check_digit
from akheval.synth.pools import Pair
from akheval.synth.templates.digital import _email, _syrian_phone, _third_person
from akheval.synth.templates.identity import _issue_date
from akheval.synth.templates.records import VAT_RATE
from akheval.synth.values import INDIC_DIGITS, REFERENCE_DATE, Person

SAMPLES_PER_STRATUM = 6
ARABIC = re.compile("[\u0600-\u06ff]")
LATIN = re.compile(r"[A-Za-z]")
TO_ASCII = {v: k for k, v in INDIC_DIGITS.items()}
IDENTITY_TYPES = ("jo_id_card", "jo_passport", "sy_id_card", "refugee_certificate")
AUTHORITIES = {
    name
    for pair in (pools.JO_KINGDOM, pools.JO_MOI, pools.JO_CSPD, pools.SY_REPUBLIC, pools.SY_MOI, pools.UNHCR)
    for name in pair
}
BIRTH_LABELS = ("تاريخ الولادة", "تاريخ الميلاد", "Date of Birth", "Date of birth")
SENDER = Person(
    Pair("سامي", "Sami"),
    Pair("علي", "Ali"),
    Pair("عمر", "Omar"),
    Pair("الخطيب", "Al-Khatib"),
    False,
    dt.date(1990, 5, 1),
)
OTHER = replace(SENDER, given=Pair("رنا", "Rana"), family=Pair("عواد", "Awad"), female=True)


@pytest.fixture(scope="module")
def documents() -> list[Document]:
    return [
        generate_document(DocSpec(doc_type, lang, index), seed=7, split="test")
        for doc_type in DOC_TYPES
        for lang in LANGUAGES
        for index in range(1, SAMPLES_PER_STRATUM + 1)
    ]


def _of(documents: list[Document], doc_type: str) -> list[Document]:
    return [d for d in documents if d.doc_type == doc_type]


def _values(doc: Document, label: str) -> list[str]:
    return [s.text for s in doc.spans if s.label == label]


def test_every_template_in_every_language_validates(documents: list[Document]) -> None:
    assert validate(Dataset(root=Path(), manifest={}, documents=documents)) == []


def test_text_is_the_layout_joined_by_newlines(documents: list[Document]) -> None:
    for doc in documents:
        assert doc.text == "\n".join(block["text"] for block in doc.meta["layout"])


@pytest.mark.parametrize("doc_type", sorted(DOC_TYPES))
@pytest.mark.parametrize("lang", LANGUAGES)
def test_each_stratum_has_gold_except_the_notice(documents: list[Document], doc_type: str, lang: str) -> None:
    docs = [d for d in _of(documents, doc_type) if d.language == lang]
    assert len(docs) == SAMPLES_PER_STRATUM
    if doc_type == "notice_no_pii":
        assert all(not d.spans and d.negatives for d in docs)
    else:
        assert all(d.spans for d in docs)


def test_mixed_documents_carry_both_scripts(documents: list[Document]) -> None:
    for doc in documents:
        if doc.language == "mixed":
            assert ARABIC.search(doc.text), doc.id
            assert LATIN.search(doc.text), doc.id
        if doc.language == "en":
            assert not ARABIC.search(doc.text), doc.id


def test_jordanian_id_cards_end_with_a_valid_td1_mrz(documents: list[Document]) -> None:
    for doc in _of(documents, "jo_id_card"):
        lines = [b["text"] for b in doc.meta["layout"] if b["kind"] == "mrz"]
        assert len(lines) == 3
        assert all(len(line) == 30 for line in lines)
        line1, line2, _ = lines
        assert line1.startswith("IDJOR")
        assert mrz_check_digit(line1[5:14]) == line1[14]
        assert mrz_check_digit(line2[0:6]) == line2[6]
        assert mrz_check_digit(line2[8:14]) == line2[14]
        assert mrz_check_digit(line1[5:30] + line2[0:7] + line2[8:15] + line2[18:29]) == line2[29]
        assert set(lines) <= set(_values(doc, "NATIONAL_ID"))


def test_jordanian_passports_end_with_a_valid_td3_mrz(documents: list[Document]) -> None:
    for doc in _of(documents, "jo_passport"):
        line1, line2 = (b["text"] for b in doc.meta["layout"] if b["kind"] == "mrz")
        assert len(line1) == len(line2) == 44
        assert line1.startswith("P<JOR")
        assert mrz_check_digit(line2[0:9]) == line2[9]
        assert mrz_check_digit(line2[0:10] + line2[13:20] + line2[21:43]) == line2[43]
        assert line2[0:9].rstrip("<") in _values(doc, "PASSPORT")
        assert len(_values(doc, "PERSON")) >= 2


def test_national_ids_have_the_expected_length(documents: list[Document]) -> None:
    for doc in documents:
        for value in _values(doc, "NATIONAL_ID"):
            if "<" in value:
                continue
            digits = value.translate(TO_ASCII)
            assert digits.isdigit(), value
            if doc.doc_type == "sy_id_card":
                assert len(digits) == 11
            else:
                assert len(digits) == 10
                assert digits[0] in "92"


def test_refugee_ids_follow_case_individual_and_card_formats(documents: list[Document]) -> None:
    pattern = re.compile(r"^(94[5-7]-\d{2}C\d{5}|94[5-7]-\d{8}|[A-K]{2}\d{7})$")
    values = [v for d in documents for v in _values(d, "REFUGEE_ID")]
    assert values
    assert all(pattern.match(v) for v in values), [v for v in values if not pattern.match(v)]


def test_gold_ibans_are_valid_and_invalid_look_alikes_are_not(documents: list[Document]) -> None:
    gold = [v for d in documents for v in _values(d, "IBAN")]
    assert gold
    assert all(iban_is_valid(v) for v in gold)
    invalid = [
        n.text for d in documents for n in d.negatives if n.label == "invalid_identifier" and n.text.startswith("JO")
    ]
    assert invalid
    assert not any(iban_is_valid(v) for v in invalid)


def test_some_arabic_documents_use_arabic_indic_digits(documents: list[Document]) -> None:
    arabic = [d for d in documents if d.language == "ar"]
    flagged = [d for d in arabic if d.meta["indic_digits"]]
    assert 0 < len(flagged) < len(arabic)
    assert all(any(ord(ch) in TO_ASCII for ch in d.text) for d in flagged)


def test_issuing_authorities_carry_no_label(documents: list[Document]) -> None:
    for doc in documents:
        if doc.doc_type in IDENTITY_TYPES:
            assert any(name in doc.text for name in AUTHORITIES), doc.id
            assert not {s.text for s in (*doc.spans, *doc.negatives)} & AUTHORITIES, doc.id


def _lines(doc: Document) -> list[tuple[int, int, dict[str, str]]]:
    """Each layout block with its start and end offsets in the text."""
    out, offset = [], 0
    for block in doc.meta["layout"]:
        out.append((offset, offset + len(block["text"]), block))
        offset += len(block["text"]) + 1
    return out


def _holds_birth_date(doc: Document, block: dict[str, str]) -> bool:
    """A "Date of Birth" field, or a family-member row of an asylum certificate."""
    label = block["text"].partition(": ")[0]
    if block["kind"] == "field":
        return any(name in label for name in BIRTH_LABELS)
    return doc.doc_type == "refugee_certificate" and block["kind"] == "row"


def test_dates_of_birth_are_neither_gold_nor_look_alikes(documents: list[Document]) -> None:
    checked = 0
    for doc in documents:
        for start, end, block in _lines(doc):
            if _holds_birth_date(doc, block):
                checked += 1
                assert not [n for n in doc.negatives if start <= n.start < end], (doc.id, block["text"])
        if doc.doc_type in ("jo_id_card", "jo_passport"):
            assert any(n.label == "date" for n in doc.negatives), doc.id
    assert checked > len(documents) // 2


def test_card_documents_mark_front_and_back(documents: list[Document]) -> None:
    for doc in _of(documents, "jo_id_card"):
        sides = {b.get("side") for b in doc.meta["layout"]}
        assert sides == {"front", "back"}


def test_chat_lines_are_messages_with_a_named_sender(documents: list[Document]) -> None:
    message = re.compile(r"^\[\d{2}:\d{2}\] [^:]+: .+$")
    for doc in _of(documents, "chat"):
        messages = [b["text"] for b in doc.meta["layout"] if b["kind"] == "message"]
        assert messages
        assert all(message.match(m) for m in messages)


def test_incident_reports_carry_ip_addresses_and_version_look_alikes(documents: list[Document]) -> None:
    docs = _of(documents, "it_incident")
    assert all(_values(d, "IP_ADDRESS") for d in docs)
    assert any(n.label == "version_number" for d in docs for n in d.negatives)


def _ents(parts: Sequence[Part], label: str) -> list[Ent]:
    return [p for p in parts if isinstance(p, Ent) and p.label == label]


def _plain(parts: Sequence[Part]) -> str:
    return "".join(p if isinstance(p, str) else p.value for p in parts)


def test_organisation_names_domains_and_bank_codes_are_distinct() -> None:
    assert len(pools.ORGS) == len(pools.ORG_STEMS) * len(pools.ORG_PATTERNS)
    for field in ("ar", "en", "domain"):
        assert len({getattr(o, field) for o in pools.ORGS}) == len(pools.ORGS), field
    codes = [o.bank_code for o in pools.ORGS if o.kind == "bank"]
    assert len(set(codes)) == len(codes) and all(re.fullmatch("[A-Z]{4}", code) for code in codes)


def test_access_log_timestamps_are_date_look_alikes(documents: list[Document]) -> None:
    stamp = re.compile(r"\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} \+0300)\]")
    checked = 0
    for doc in _of(documents, "it_incident"):
        dates = {(n.start, n.end) for n in doc.negatives if n.label == "date"}
        for m in stamp.finditer(doc.text):
            checked += 1
            assert (m.start(1), m.end(1)) in dates, (doc.id, m.group(1))
    assert checked


@pytest.mark.parametrize("lang", LANGUAGES)
def test_syrian_phone_message_labels_the_country(lang: str) -> None:
    for seed in range(10):
        parts = _syrian_phone(Ctx(lang, random.Random(seed)), SENDER, [OTHER])
        assert [e.value for e in _ents(parts, "LOCATION")] in (["Syria"], ["سوريا"])
        assert not any(isinstance(p, str) and ("Syria" in p or "سوريا" in p) for p in parts)


@pytest.mark.parametrize("lang", LANGUAGES)
def test_chat_email_belongs_to_the_sender(lang: str) -> None:
    for seed in range(20):
        (email,) = _ents(_email(Ctx(lang, random.Random(seed)), SENDER, [OTHER]), "EMAIL")
        local = email.value.split("@")[0]
        assert re.search("sami|khatib", local) and not re.search("rana|awad", local)


def test_third_person_pronoun_follows_the_named_person() -> None:
    female = {p.ar for p in pools.FEMALE_GIVEN}
    for seed in range(30):
        parts = _third_person(Ctx("ar", random.Random(seed)), SENDER, [OTHER])
        (name,) = _ents(parts, "PERSON")
        assert ("رقمها " if name.value.split()[0] in female else "رقمه ") in _plain(parts)


def test_relations_are_distinct_and_fit_the_holder() -> None:
    for seed in range(50):
        c = Ctx("en", random.Random(seed))
        for holder in (SENDER, OTHER):
            names = [r.name.en for r in c.v.relations(holder, 4)]
            assert len(set(names)) == 4
            assert ("Wife" if holder.female else "Husband") not in names


def test_father_carries_the_holder_patronymic() -> None:
    father = Ctx("en", random.Random(3)).v.relative(SENDER, pools.RELATIONS[0])
    assert (father.given, father.father, father.family) == (SENDER.father, SENDER.grandfather, SENDER.family)
    assert not father.female and father.birth < SENDER.birth


@pytest.mark.parametrize("min_age", [14, 16])
def test_issue_date_falls_after_the_minimum_age(min_age: int) -> None:
    for seed in range(100):
        c = Ctx("en", random.Random(seed))
        holder = c.v.person(min_age=min_age, max_age=min_age + 10)
        issued = _issue_date(c, holder, min_age, 10 * 365)
        assert (issued - holder.birth).days >= min_age * 365
        assert REFERENCE_DATE - dt.timedelta(days=10 * 365) <= issued < REFERENCE_DATE


def test_family_rows_name_each_relation_once(documents: list[Document]) -> None:
    tables = 0
    for doc in _of(documents, "refugee_certificate"):
        rows = [b["text"].split(" | ") for b in doc.meta["layout"] if b["kind"] == "row"]
        relations = [row[2] for row in rows[1:]]
        tables += bool(relations)
        assert len(set(relations)) == len(relations), doc.id
    assert tables


def _fils(amount: str) -> int:
    return int(Decimal(re.sub(r"[^\d.]", "", amount)) * 1000)


def test_invoice_figures_add_up(documents: list[Document]) -> None:
    for doc in _of(documents, "invoice"):
        if doc.language != "en":
            continue
        amounts = [_fils(n.text) for n in sorted(doc.negatives, key=lambda n: n.start) if n.label == "amount"]
        *lines, subtotal, tax, total = amounts
        assert sum(lines[1::2]) == subtotal
        assert tax == round(subtotal * VAT_RATE) and total == subtotal + tax
