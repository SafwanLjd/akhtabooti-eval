"""Identity documents: Jordanian ID card and passport, Syrian ID card, refugee certificates."""

from __future__ import annotations

import datetime as dt

from .. import pools
from ..builder import BACK, FRONT, Ent
from ..context import Ctx
from ..identifiers import td1_lines, td3_lines
from ..pools import Pair
from ..values import REFERENCE_DATE, Person, add_years
from .common import ID_CARD, PAGE, PASSPORT

ID_CARD_VALIDITY_YEARS = 10
PASSPORT_VALIDITY_YEARS = 5
SEX = {False: Pair("ذكر", "M"), True: Pair("أنثى", "F")}
NATIONALITY_JO = Pair("أردني", "JORDANIAN")
NATIONALITY_SY = Pair("سوري", "Syrian")
BLOOD_GROUPS = ("A+", "A-", "B+", "O+", "O-", "AB+")
ORIGIN_COUNTRIES = (Pair("سوريا", "Syria"), Pair("العراق", "Iraq"), Pair("اليمن", "Yemen"), Pair("السودان", "Sudan"))


def _issue_date(c: Ctx, holder: Person, min_age: int, max_days: int) -> dt.date:
    """A past date within ``max_days`` on which ``holder`` was at least ``min_age`` years old."""
    eligible = (REFERENCE_DATE - add_years(holder.birth, min_age)).days
    return c.v.past_date(max(1, min(max_days, eligible)))


def _authority_title(c: Ctx, pair: Pair, side: str | None = None) -> None:
    """Issuing state or agency in the document language; mixed documents add the English name as a header line.

    Issuing-authority furniture is not personal data, so it carries no label.
    """
    c.b.title(pair.en if c.en else pair.ar, side=side)
    if c.lang == "mixed":
        c.b.header(pair.en, side=side)


def _authority(c: Ctx, pair: Pair) -> str:
    """A ministry or department of the issuing state, unlabelled like the title."""
    return pair.en if c.script() == "en" else pair.ar


def _bilingual_header(c: Ctx, ar: str, en: str, side: str | None = None) -> None:
    c.b.header(c.t(ar, en, f"{ar} / {en}"), side=side)


def _sex(c: Ctx, person: Person) -> str:
    value = SEX[person.female]
    return c.t(value.ar, value.en, f"{value.ar} / {value.en}")


def _name_fields(c: Ctx, person: Person, parts: int, side: str | None = None) -> None:
    """Name in Arabic, Latin or both scripts, depending on the document language."""
    if c.lang != "en":
        c.b.field(c.lbl("الاسم", "Name"), Ent(person.name("ar", parts), "PERSON"), key="name", side=side)
    if c.lang != "ar":
        c.b.field("Name", Ent(person.name("en", parts).upper(), "PERSON"), key="name_latin", side=side)


def jo_id_card(c: Ctx) -> str:
    holder, mother = c.v.person(min_age=16, max_age=75), c.v.person(female=True)
    issued = _issue_date(c, holder, 16, ID_CARD_VALIDITY_YEARS * 365 - 30)
    expiry = add_years(issued, ID_CARD_VALIDITY_YEARS)
    national_id = c.v.jo_national_id(holder)
    _authority_title(c, pools.JO_KINGDOM, FRONT)
    c.b.header(_authority(c, pools.JO_MOI), side=FRONT)
    c.b.header(_authority(c, pools.JO_CSPD), side=FRONT)
    _bilingual_header(c, "بطاقة شخصية", "Identity Card", FRONT)
    _name_fields(c, holder, 4, FRONT)
    c.b.field(
        c.lbl("الرقم الوطني", "National No."), Ent(c.num(national_id), "NATIONAL_ID"), key="national_id", side=FRONT
    )
    c.b.field(c.lbl("تاريخ الولادة", "Date of Birth"), c.birth_date(holder), key="birth_date", side=FRONT)
    c.b.field(c.lbl("مكان الولادة", "Place of Birth"), c.place(c.v.birth_place()), key="birth_place", side=FRONT)
    c.b.field(c.lbl("الجنس", "Sex"), _sex(c, holder), key="sex", side=FRONT)
    c.b.field(c.lbl("اسم الأم", "Mother's Name"), c.person(mother, parts=2), key="mother", side=FRONT)
    c.b.field(c.lbl("تاريخ الانتهاء", "Expiry Date"), c.date(expiry), key="expiry", side=FRONT)
    c.b.field(c.lbl("مكان الإقامة", "Residence"), c.place(c.v.city()), key="residence", side=BACK)
    c.b.field(c.lbl("مكان الإصدار", "Place of Issue"), c.place(c.v.city()), key="issue_place", side=BACK)
    c.b.field(c.lbl("تاريخ الإصدار", "Date of Issue"), c.date(issued), key="issue_date", side=BACK)
    c.b.field(c.lbl("فصيلة الدم", "Blood Group"), c.pick(*BLOOD_GROUPS), key="blood_group", side=BACK)
    c.b.blank(side=BACK)
    lines = td1_lines(
        issuer="JOR",
        document_number=c.v.jo_card_number(),
        optional1=national_id,
        birth=holder.birth,
        sex=holder.sex,
        expiry=expiry,
        nationality="JOR",
        surname=holder.family.en,
        given_names=holder.given_names("en", 3),
    )
    for line in lines:
        c.b.mrz(line, "NATIONAL_ID", side=BACK)
    return ID_CARD


def jo_passport(c: Ctx) -> str:
    holder = c.v.person(min_age=1, max_age=80)
    expiry, number = c.v.future_date(PASSPORT_VALIDITY_YEARS * 365), c.v.passport_number()
    national_id = c.v.jo_national_id(holder)
    _authority_title(c, pools.JO_KINGDOM)
    _bilingual_header(c, "جواز سفر", "PASSPORT")
    c.b.field(c.lbl("النوع", "Type"), "P", key="type")
    c.b.field(c.lbl("رمز الدولة", "Code"), "JOR", key="code")
    c.b.field(c.lbl("رقم الجواز", "Passport No."), Ent(number, "PASSPORT"), key="passport_no")
    if c.lang == "ar":
        c.b.field("اسم العائلة", Ent(holder.family.ar, "PERSON"), key="surname")
        c.b.field("الاسم", Ent(holder.given_names("ar"), "PERSON"), key="given_names")
    else:
        c.b.field(c.lbl("اسم العائلة", "Surname"), Ent(holder.family.en.upper(), "PERSON"), key="surname")
        c.b.field(c.lbl("الأسماء", "Given Names"), Ent(holder.given_names("en").upper(), "PERSON"), key="given_names")
    if c.lang == "mixed":
        c.b.field("الاسم بالعربية", Ent(holder.name("ar", 4), "PERSON"), key="name_ar")
    c.b.field(c.lbl("الجنسية", "Nationality"), c.t(NATIONALITY_JO.ar, NATIONALITY_JO.en, "أردني / JORDANIAN"))
    c.b.field(c.lbl("تاريخ الولادة", "Date of Birth"), c.birth_date(holder), key="birth_date")
    c.b.field(c.lbl("الجنس", "Sex"), _sex(c, holder), key="sex")
    place = c.v.birth_place()
    c.b.field(
        c.lbl("مكان الولادة", "Place of Birth"), c.place(place) if c.lang == "ar" else Ent(place.en.upper(), "LOCATION")
    )
    issued = add_years(expiry, -PASSPORT_VALIDITY_YEARS)
    c.b.field(c.lbl("تاريخ الإصدار", "Date of Issue"), c.date(issued), key="issue_date")
    c.b.field(c.lbl("تاريخ الانتهاء", "Date of Expiry"), c.date(expiry), key="expiry")
    c.b.field(c.lbl("جهة الإصدار", "Authority"), c.place(c.v.city()), key="authority")
    c.b.field(c.lbl("الرقم الوطني", "National No."), Ent(c.num(national_id), "NATIONAL_ID"), key="national_id")
    c.b.blank()
    lines = td3_lines(
        issuer="JOR",
        document_number=number,
        birth=holder.birth,
        sex=holder.sex,
        expiry=expiry,
        nationality="JOR",
        personal_number=national_id,
        surname=holder.family.en,
        given_names=holder.given_names("en"),
    )
    for line in lines:
        c.b.mrz(line, "PASSPORT")
    return PASSPORT


def sy_id_card(c: Ctx) -> str:
    """Syrian ID card; the English version is a certified translation printed on a page."""
    holder, mother = c.v.person(min_age=14, max_age=75), c.v.person(female=True)
    front, back = (None, None) if c.en else (FRONT, BACK)
    if c.en:
        c.b.title("Certified translation from Arabic")
    _authority_title(c, pools.SY_REPUBLIC, front)
    c.b.header(_authority(c, pools.SY_MOI), side=front)
    _bilingual_header(c, "بطاقة شخصية", "Personal Identity Card", front)
    c.b.field(
        c.lbl("الرقم الوطني", "National No."),
        Ent(c.num(c.v.sy_national_id()), "NATIONAL_ID"),
        key="national_id",
        side=front,
    )
    script = "en" if c.en else c.script()
    c.b.field(c.lbl("الاسم", "Name"), c.given(holder, script), key="given_name", side=front)
    c.b.field(c.lbl("الكنية", "Surname"), Ent(getattr(holder.family, script), "PERSON"), key="surname", side=front)
    c.b.field(
        c.lbl("اسم الأب", "Father's Name"), Ent(getattr(holder.father, script), "PERSON"), key="father", side=front
    )
    c.b.field(c.lbl("اسم الأم", "Mother's Name"), c.person(mother, parts=2, script=script), key="mother", side=front)
    c.b.field(
        c.lbl("مكان وتاريخ الولادة", "Place and Date of Birth"),
        c.place(c.v.city("SY"), script),
        " ",
        c.birth_date(holder),
        key="birth",
        side=front,
    )
    c.b.field(
        c.lbl("القيد", "Civil Registry"),
        c.place(c.v.city("SY"), script),
        " / ",
        c.num(str(c.rng.randint(1, 999))),
        key="registry",
        side=back,
    )
    c.b.field(
        c.lbl("الجنس", "Sex"),
        _sex(c, holder) if not c.en else ("Female" if holder.female else "Male"),
        key="sex",
        side=back,
    )
    c.b.field(
        c.lbl("تاريخ المنح", "Date of Issue"), c.date(_issue_date(c, holder, 14, 15 * 365)), key="issue_date", side=back
    )
    if c.en:
        c.b.blank()
        translator, office = c.v.person(), c.v.org("company")
        c.b.para(
            "Translated by ", c.person(translator, parts=2, script="en"), ", sworn translator, ", c.org(office, "en")
        )
        c.b.para("Translation reference: ", c.reference(), ". Translator phone: ", c.mobile())
        return PAGE
    return ID_CARD


def refugee_certificate(c: Ctx) -> str:
    """A UNHCR-style asylum-seeker certificate (page) or a Jordanian MOI service card for Syrians (card)."""
    variant = c.rng.choices(("asylum_certificate", "moi_service_card"), weights=(3, 2))[0]
    c.meta["variant"] = variant
    if variant == "moi_service_card":
        return _moi_service_card(c)
    return _asylum_certificate(c)


def _asylum_certificate(c: Ctx) -> str:
    holder = c.v.person(min_age=18, max_age=70)
    case = c.v.refugee_case_number()
    _authority_title(c, pools.UNHCR)
    _bilingual_header(c, "شهادة طالب لجوء", "Asylum-Seeker Certificate")
    c.b.field(c.lbl("المكتب", "Office"), c.place(c.v.city()))
    c.b.field(c.lbl("رقم الملف", "Case No."), Ent(case, "REFUGEE_ID"))
    c.b.field(c.lbl("الرقم الفردي", "Individual No."), Ent(c.v.refugee_individual_number(case), "REFUGEE_ID"))
    c.b.field(c.lbl("الاسم", "Name"), c.person(holder, parts=c.pick(3, 4)))
    c.b.field(c.lbl("تاريخ الولادة", "Date of Birth"), c.birth_date(holder))
    c.b.field(c.lbl("الجنس", "Sex"), _sex(c, holder))
    c.b.field(
        c.lbl("بلد الأصل", "Country of Origin"), c.place(c.rng.choices(ORIGIN_COUNTRIES, weights=(8, 2, 1, 1))[0])
    )
    c.b.field(c.lbl("تاريخ التسجيل", "Date of Registration"), c.date(c.v.past_date(10 * 365)))
    c.b.field(c.lbl("صالحة حتى", "Valid Until"), c.date(c.v.future_date(2 * 365)))
    if c.rng.random() < 0.7:
        c.b.field(c.lbl("العنوان", "Address"), c.address())
    else:
        c.b.field(c.lbl("مكان السكن", "Residence"), c.place(c.pick(*pools.CAMPS)))
    c.b.field(c.lbl("رقم الهاتف", "Phone"), c.pick(c.mobile, c.syrian_mobile)())
    unhcr = pools.UNHCR.ar if c.lang == "ar" else pools.UNHCR.en
    c.b.para(
        *c.say(
            ar=[["تشهد ", unhcr, " بأن الشخص المذكور أعلاه مسجل لديها كطالب لجوء، ويجب حمايته من الإعادة القسرية."]],
            en=[
                [
                    "This certificate confirms that the person named above is registered with ",
                    unhcr,
                    " as an asylum-seeker and must be protected from forcible return.",
                ]
            ],
        )
    )
    c.b.blank()
    _bilingual_header(c, "أفراد الأسرة", "Family Members")
    c.b.row(
        c.lbl("الرقم الفردي", "Individual No."),
        c.lbl("الاسم", "Name"),
        c.lbl("الصلة", "Relationship"),
        c.lbl("تاريخ الولادة", "Date of Birth"),
    )
    for relation, member in c.v.relatives(holder, c.rng.randint(1, 4)):
        c.b.row(
            Ent(c.v.refugee_individual_number(case), "REFUGEE_ID"),
            c.person(member, parts=2),
            c.t(relation.name.ar, relation.name.en),
            c.birth_date(member),
        )
    c.b.blank()
    c.b.para(
        c.t("للتحقق من صحة الشهادة اتصل على ", "To verify this certificate, call ", "للتحقق / To verify, call "),
        c.landline(),
    )
    return PAGE


def _moi_service_card(c: Ctx) -> str:
    holder = c.v.person(min_age=12, max_age=75)
    _authority_title(c, pools.JO_KINGDOM, FRONT)
    c.b.header(_authority(c, pools.JO_MOI), side=FRONT)
    _bilingual_header(c, "بطاقة الخدمة للجالية السورية", "Service Card for Syrian Nationals", FRONT)
    c.b.field(c.lbl("الاسم", "Name"), c.person(holder, parts=c.pick(3, 4)), key="name", side=FRONT)
    c.b.field(c.lbl("رقم البطاقة", "Card No."), Ent(c.v.moi_card_number(), "REFUGEE_ID"), key="card_no", side=FRONT)
    c.b.field(
        c.lbl("رقم ملف المفوضية", "UNHCR Case No."),
        Ent(c.v.refugee_case_number(), "REFUGEE_ID"),
        key="case_no",
        side=FRONT,
    )
    c.b.field(c.lbl("تاريخ الولادة", "Date of Birth"), c.birth_date(holder), key="birth_date", side=FRONT)
    c.b.field(
        c.lbl("الجنسية", "Nationality"),
        c.t(NATIONALITY_SY.ar, NATIONALITY_SY.en, "سوري / Syrian"),
        key="nationality",
        side=FRONT,
    )
    c.b.field(c.lbl("المحافظة", "Governorate"), c.place(c.v.city()), key="governorate", side=FRONT)
    c.b.field(c.lbl("تاريخ الإصدار", "Date of Issue"), c.date(c.v.past_date(5 * 365)), key="issue_date", side=FRONT)
    c.b.field(c.lbl("العنوان", "Address"), c.address(), key="address", side=BACK)
    c.b.field(c.lbl("عدد أفراد الأسرة", "Family Size"), c.num(str(c.rng.randint(1, 9))), key="family_size", side=BACK)
    c.b.para(
        c.t(
            "يجب حمل هذه البطاقة في جميع الأوقات وإبرازها عند الطلب.",
            "Carry this card at all times and show it on request.",
            "يجب حمل البطاقة دائماً / Carry this card at all times.",
        ),
        side=BACK,
    )
    return ID_CARD
