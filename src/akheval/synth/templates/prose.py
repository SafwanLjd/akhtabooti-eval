"""Running-text documents: letters, press notes and CVs."""

from __future__ import annotations

from collections.abc import Callable

from .. import pools
from ..builder import Ent, Neg
from ..context import Ctx
from ..pools import Pair
from ..values import Person
from .common import PAGE, dated_reference, footer, joined, letterhead, shuffled_fields, signature

TOPICS = (
    Pair("السلامة الرقمية", "digital safety"),
    Pair("حماية البيانات الشخصية", "personal data protection"),
    Pair("التمكين الاقتصادي للنساء", "women's economic empowerment"),
    Pair("الدعم النفسي الاجتماعي", "psychosocial support"),
    Pair("التعليم غير النظامي", "non-formal education"),
    Pair("الصحة المجتمعية", "community health"),
)
SERVICES = (
    Pair("استشارة قانونية", "legal counselling"),
    Pair("دعم نفسي اجتماعي", "psychosocial support"),
    Pair("مساعدة نقدية", "cash assistance"),
    Pair("متابعة طبية", "medical follow-up"),
)
AMBIGUOUS_PLACE_WORDS = (Pair("عمّان", "Amman"), Pair("الأردن", "Jordan"), Pair("البتراء", "Petra"))

DONATION_SHARE = 0.5
"""Share of press notes that end with a donation account."""

LetterBody = Callable[[Ctx, Ent, Person], None]


def ambiguous_word(c: Ctx) -> Neg:
    """A place name used as the name of a room, street or product."""
    word = c.pick(*AMBIGUOUS_PLACE_WORDS)
    return Neg(word.en if c.en else word.ar, "ambiguous_name")


def _subject(c: Ctx, ar: str, en: str) -> None:
    c.b.field(c.lbl("الموضوع", "Subject"), c.t(ar, en, f"{ar} / {en}"))
    c.b.para(c.t("تحية طيبة وبعد،", "Dear Sir or Madam,", c.pick("تحية طيبة وبعد،", "Dear all,")))


def _data_update(c: Ctx, org: Ent, recipient: Person) -> None:
    del recipient
    _subject(c, "تحديث البيانات الشخصية", "Update of personal records")
    date = c.date()
    c.b.para(
        *c.say(
            ar=[
                ["نشير إلى طلبكم المقدم إلى ", org, " بتاريخ ", date, " بخصوص تحديث بياناتكم الشخصية."],
                ["بالإشارة إلى زيارتكم لمكتبنا بتاريخ ", date, "، قامت ", org, " بمراجعة البيانات المسجلة لديها."],
            ],
            en=[
                ["We refer to your request to ", org, " dated ", date, " about updating your personal records."],
                ["Following your visit on ", date, ", ", org, " has reviewed the information it holds about you."],
            ],
        )
    )
    phone, email = c.mobile(), c.email(c.v.person())
    c.b.para(
        *c.say(
            ar=[
                ["يرجى تأكيد رقم هاتفكم ", phone, " وعنوان بريدكم الإلكتروني ", email, "."],
                [
                    "سنتواصل معكم على الرقم ",
                    phone,
                    " أو عبر البريد الإلكتروني ",
                    email,
                    " عند الحاجة إلى وثائق إضافية.",
                ],
            ],
            en=[
                ["Please confirm that your mobile number is ", phone, " and your email address is ", email, "."],
                ["We will contact you on ", phone, " or at ", email, " if we need further documents."],
            ],
            mixed=[["يرجى تأكيد الـ mobile number تبعكم ", phone, " والـ email ", email, " قبل الـ deadline."]],
        )
    )
    address = c.address()
    c.b.para(
        *c.say(
            ar=[["العنوان المسجل لدينا هو ", address, ". يرجى إعلامنا خلال أسبوعين في حال تغييره."]],
            en=[["Our records list your address as ", address, ". Tell us within two weeks if it has changed."]],
            mixed=[["الـ address المسجل عندنا: ", address, ". Please update us إذا تغير."]],
        )
    )


def _visa_support(c: Ctx, org: Ent, recipient: Person) -> None:
    del recipient
    _subject(c, "رسالة تأييد لطلب تأشيرة", "Visa support letter")
    employee = c.v.person()
    name, passport, job = (
        c.person(employee, parts=c.pick(3, 4)),
        Ent(c.v.passport_number(), "PASSPORT"),
        c.pick(*pools.JOB_TITLES),
    )
    works = "تعمل" if employee.female else "يعمل"
    since = c.date(c.v.past_date(8 * 365))
    if c.rng.random() < 0.5:
        nid = Ent(c.num(c.v.jo_national_id(employee)), "NATIONAL_ID")
        c.b.para(
            *c.say(
                ar=[
                    [
                        "تفيد ",
                        org,
                        " بأن ",
                        name,
                        " (رقم جواز السفر ",
                        passport,
                        "، الرقم الوطني ",
                        nid,
                        ") ",
                        works,
                        " لديها بوظيفة ",
                        job.ar,
                        ".",
                    ]
                ],
                en=[
                    [
                        "We confirm that ",
                        name,
                        " (passport no. ",
                        passport,
                        ", national number ",
                        nid,
                        ") is employed by ",
                        org,
                        " as ",
                        job.en,
                        ".",
                    ]
                ],
            )
        )
    else:
        c.b.para(
            *c.say(
                ar=[
                    [
                        "نشهد بأن ",
                        name,
                        "، حامل جواز السفر رقم ",
                        passport,
                        "، ",
                        works,
                        " لدى ",
                        org,
                        " بوظيفة ",
                        job.ar,
                        " منذ ",
                        since,
                        ".",
                    ]
                ],
                en=[
                    [
                        "This letter confirms that ",
                        name,
                        ", holder of passport number ",
                        passport,
                        ", has worked at ",
                        org,
                        " as ",
                        job.en,
                        " since ",
                        since,
                        ".",
                    ]
                ],
                mixed=[
                    [
                        "نشهد بأن ",
                        name,
                        "، passport no. ",
                        passport,
                        "، ",
                        works,
                        " في ",
                        org,
                        " كـ ",
                        job.en,
                        " من ",
                        since,
                        ".",
                    ]
                ],
            )
        )
    country = c.pick(*pools.COUNTRIES[3:])
    city, nation = c.place(c.v.city("JO")), c.place(country)
    start, end = (c.date(day) for day in sorted((c.v.future_date(120), c.v.future_date(200))))
    c.b.para(
        *c.say(
            ar=[
                [
                    "سيشارك الموظف في تدريب يعقد في ",
                    nation,
                    " خلال الفترة من ",
                    start,
                    " إلى ",
                    end,
                    "، وتتحمل ",
                    org,
                    " كافة تكاليف السفر.",
                ]
            ],
            en=[
                [
                    "The employee will attend a training in ",
                    nation,
                    " from ",
                    start,
                    " to ",
                    end,
                    ", and ",
                    org,
                    " covers all travel costs.",
                ]
            ],
            mixed=[
                [
                    "رح يشارك في training في ",
                    nation,
                    " من ",
                    start,
                    " لغاية ",
                    end,
                    "، و ",
                    org,
                    " بتغطي كل الـ travel costs.",
                ]
            ],
        )
    )
    phone, email = c.landline(), c.email(c.v.person())
    c.b.para(
        *c.say(
            ar=[["للتحقق يرجى التواصل معنا في مكتب ", city, " على الرقم ", phone, " أو عبر البريد ", email, "."]],
            en=[["For verification, contact our ", city, " office on ", phone, " or at ", email, "."]],
        )
    )


def _referral(c: Ctx, org: Ent, recipient: Person) -> None:
    del org
    _subject(c, "إحالة حالة", "Case referral")
    client = c.v.person()
    name, case = c.person(client, parts=c.pick(3, 4)), Ent(c.v.refugee_case_number(), "REFUGEE_ID")
    service = c.pick(*SERVICES)
    c.b.para(
        *c.say(
            ar=[
                ["نحيل إليكم ", name, "، رقم ملف اللجوء ", case, "، للحصول على ", service.ar, "."],
                ["نرجو التكرم باستقبال ", name, " (رقم الملف ", case, ") وتقديم ", service.ar, " له ولعائلته."],
            ],
            en=[
                ["We refer ", name, ", asylum case number ", case, ", to your organisation for ", service.en, "."],
                ["Please receive ", name, " (case no. ", case, ") and provide ", service.en, " to the family."],
            ],
            mixed=[["نحيل لكم ", name, "، case no. ", case, "، لخدمات الـ ", service.en, "."]],
        )
    )
    phone = c.pick(c.mobile, c.syrian_mobile)()
    if c.rng.random() < 0.5:
        individual = Ent(c.v.refugee_individual_number(case.value), "REFUGEE_ID")
        c.b.para(
            *c.say(
                ar=[["الرقم الفردي لمقدم الطلب الرئيسي هو ", individual, "، ورقم التواصل ", phone, "."]],
                en=[
                    [
                        "The individual number of the principal applicant is ",
                        individual,
                        ". The family can be reached on ",
                        phone,
                        ".",
                    ]
                ],
            )
        )
    else:
        address = c.address()
        c.b.para(
            *c.say(
                ar=[["تقيم العائلة في ", address, "، ويمكن التواصل معها على الرقم ", phone, "."]],
                en=[["The family lives at ", address, " and can be reached on ", phone, "."]],
            )
        )
    email = c.email(recipient)
    c.b.para(
        *c.say(
            ar=[["يرجى إرسال الرد إلى البريد الإلكتروني ", email, " خلال عشرة أيام عمل."]],
            en=[["Please send your feedback to ", email, " within ten working days."]],
            mixed=[["Please send the feedback على ", email, " خلال 10 working days."]],
        )
    )


def _refund(c: Ctx, org: Ent, recipient: Person) -> None:
    del recipient
    _subject(c, "إعادة مبلغ", "Refund of payment")
    amount, invoice, date = c.amount(), c.invoice_number(), c.date()
    c.b.para(
        *c.say(
            ar=[["استلمنا دفعتكم بقيمة ", amount, " عن الفاتورة رقم ", invoice, " بتاريخ ", date, "."]],
            en=[["We received your payment of ", amount, " for invoice ", invoice, " on ", date, "."]],
        )
    )
    bank = c.v.org("bank")
    iban, bank_e = c.iban(bank), c.org(bank)
    c.b.para(
        *c.say(
            ar=[["ستقوم ", org, " بإعادة المبلغ إلى حسابكم ", iban, " لدى ", bank_e, " خلال أربعة عشر يوماً."]],
            en=[[org, " will refund the amount to your account ", iban, " at ", bank_e, " within 14 days."]],
            mixed=[["رح نعمل refund للمبلغ على الحساب ", iban, " في ", bank_e, " خلال 14 يوم."]],
        )
    )
    email, phone = c.org_email(c.v.org()), c.landline()
    c.b.para(
        *c.say(
            ar=[["في حال وجود خطأ في بيانات الحساب يرجى التواصل مع قسم المالية على ", email, " أو الرقم ", phone, "."]],
            en=[["If the account details are wrong, contact our finance team at ", email, " or on ", phone, "."]],
        )
    )


def _invitation(c: Ctx, org: Ent, recipient: Person) -> None:
    del recipient
    _subject(c, "دعوة لحضور ورشة عمل", "Workshop invitation")
    topic, host = c.pick(*TOPICS), c.v.org("ngo", "university", "company")
    date, room, host_e, city = c.date(c.v.future_date(90)), ambiguous_word(c), c.org(host), c.place(c.v.city())
    c.b.para(
        *c.say(
            ar=[
                [
                    "يسر ",
                    org,
                    " دعوتكم لحضور ورشة عمل حول ",
                    topic.ar,
                    " بتاريخ ",
                    date,
                    " في قاعة ",
                    room,
                    " لدى ",
                    host_e,
                    " في ",
                    city,
                    ".",
                ]
            ],
            en=[
                [
                    org,
                    " invites you to a workshop on ",
                    topic.en,
                    " on ",
                    date,
                    " in the ",
                    room,
                    " Hall at ",
                    host_e,
                    ", ",
                    city,
                    ".",
                ]
            ],
            mixed=[
                [
                    "يسر ",
                    org,
                    " دعوتكم لـ workshop عن ",
                    topic.en,
                    " يوم ",
                    date,
                    " في قاعة ",
                    room,
                    "، ",
                    host_e,
                    "، ",
                    city,
                    ".",
                ]
            ],
        )
    )
    email, phone = c.email(c.v.person()), c.mobile()
    contact = c.v.person()
    c.b.para(
        *c.say(
            ar=[
                [
                    "يرجى تأكيد الحضور عبر البريد ",
                    email,
                    " أو الاتصال بـ ",
                    c.person(contact, parts=2),
                    " على الرقم ",
                    phone,
                    ".",
                ]
            ],
            en=[["Please confirm by email to ", email, " or call ", c.person(contact, parts=2), " on ", phone, "."]],
            mixed=[
                ["Please confirm attendance على ", email, " أو كلموا ", c.person(contact, parts=2), " على ", phone, "."]
            ],
        )
    )


LETTER_BODIES: dict[str, LetterBody] = {
    "data_update": _data_update,
    "visa_support": _visa_support,
    "referral": _referral,
    "refund": _refund,
    "invitation": _invitation,
}


def letter(c: Ctx) -> str:
    org = c.v.org("ngo", "company", "hospital", "university")
    letterhead(c, org)
    c.b.blank()
    dated_reference(c)
    c.b.blank()
    recipient, script = c.v.person(), c.script()
    honorific = c.honorific(recipient, script)
    respected = c.t(" المحترمة" if recipient.female else " المحترم", "")
    c.b.para(c.lbl("إلى", "To"), ": ", honorific, " ", c.person(recipient, script=script), respected)
    if c.rng.random() < 0.5:
        c.b.para(c.address())
    variant = c.pick(*sorted(LETTER_BODIES))
    c.meta["variant"] = variant
    LETTER_BODIES[variant](c, c.org(org), recipient)
    c.b.blank()
    job = c.pick(*pools.JOB_TITLES)
    signature(c, c.v.person(), job.en if c.en else job.ar, org)
    footer(c)
    return PAGE


def press_note(c: Ctx) -> str:
    org, partner = c.v.orgs(2, "ngo", "company", "university")
    org_e, partner_e, topic = c.org(org), c.org(partner), c.pick(*TOPICS)
    city, gov1, gov2 = (c.place(p) for p in c.rng.sample(pools.JO_CITIES, 3))
    date = c.date(c.v.past_date(60))
    c.b.title(c.t("بيان صحفي", "Press release", "بيان صحفي - Press Release"))
    c.b.blank()
    beneficiaries = str(c.rng.randint(80, 2400))
    c.b.para(
        *c.say(
            ar=[
                [
                    city,
                    " - ",
                    date,
                    " - أطلقت ",
                    org_e,
                    " اليوم برنامجاً في مجال ",
                    topic.ar,
                    " في محافظتي ",
                    gov1,
                    " و",
                    gov2,
                    "، بمشاركة ",
                    c.num(beneficiaries),
                    " مستفيد في السنة الأولى.",
                ],
                [
                    city,
                    " - ",
                    date,
                    " - وقعت ",
                    org_e,
                    " اتفاقية شراكة مع ",
                    partner_e,
                    " لتوسيع خدمات ",
                    topic.ar,
                    " في ",
                    gov1,
                    " و",
                    gov2,
                    ".",
                ],
            ],
            en=[
                [
                    city,
                    " - ",
                    date,
                    " - ",
                    org_e,
                    " launched a programme today to support ",
                    topic.en,
                    " in ",
                    gov1,
                    " and ",
                    gov2,
                    ", with ",
                    beneficiaries,
                    " participants in its first year.",
                ],
                [
                    city,
                    " - ",
                    date,
                    " - ",
                    org_e,
                    " signed a partnership agreement with ",
                    partner_e,
                    " to expand ",
                    topic.en,
                    " services across ",
                    gov1,
                    " and ",
                    gov2,
                    ".",
                ],
            ],
            mixed=[
                [
                    city,
                    " - ",
                    date,
                    " - أطلقت ",
                    org_e,
                    " اليوم programme جديد في مجال ",
                    topic.en,
                    " في ",
                    gov1,
                    " و ",
                    gov2,
                    ".",
                ]
            ],
        )
    )
    speaker = c.v.person()
    speaker_e, year = c.person(speaker, parts=c.pick(2, 3)), c.year(c.rng.randint(2019, 2025))
    said, director, stressed = (
        ("وقالت ", "المديرة التنفيذية", "وأكدت ") if speaker.female else ("وقال ", "المدير التنفيذي", "وأكد ")
    )
    c.b.para(
        *c.say(
            ar=[
                [
                    said,
                    speaker_e,
                    "، ",
                    director,
                    " في ",
                    org_e,
                    "، إن البرنامج يستجيب لاحتياجات طرحتها المجموعات المحلية خلال عام ",
                    year,
                    ".",
                ],
                [stressed, speaker_e, " أن ", org_e, " بنت البرنامج بالشراكة مع المجتمعات المحلية منذ عام ", year, "."],
            ],
            en=[
                [
                    speaker_e,
                    ", executive director of ",
                    org_e,
                    ", said the programme answers needs that local groups raised during ",
                    year,
                    ".",
                ],
                [
                    '"We have built this programme with the communities we serve since ',
                    year,
                    '," said ',
                    speaker_e,
                    ", who chairs ",
                    org_e,
                    ".",
                ],
            ],
        )
    )
    until, office = c.date(c.v.future_date(180)), c.place(c.v.city())
    c.b.para(
        *c.say(
            ar=[
                ["ويحظى البرنامج بتمويل من ", partner_e, " ويستمر حتى ", until, "."],
                ["ويستمر التسجيل حتى ", until, " عبر مكاتب ", org_e, " في ", office, "."],
            ],
            en=[
                ["The programme is funded by ", partner_e, " and runs until ", until, "."],
                ["Registration is open until ", until, " at the offices of ", org_e, " in ", office, "."],
            ],
            mixed=[["الـ registration مفتوح لغاية ", until, " في مكاتب ", org_e, " في ", office, "."]],
        )
    )
    if c.rng.random() < DONATION_SHARE:
        bank = c.v.org("bank")
        c.b.para(
            *c.say(
                ar=[["ويمكن إرسال التبرعات إلى الحساب ", c.iban(bank), " لدى ", c.org(bank), "."]],
                en=[["Donations can be sent to account ", c.iban(bank), " at ", c.org(bank), "."]],
            )
        )
    c.b.blank()
    contact = c.v.person()
    c.b.para(
        c.lbl("للتواصل الإعلامي", "Media contact"),
        ": ",
        *joined(c, c.person(contact, parts=2), c.mobile(), c.email(contact, org.domain)),
    )
    footer(c)
    return PAGE


def cv(c: Ctx) -> str:
    person = c.v.person(min_age=22, max_age=55)
    script = c.script()
    job = c.pick(*pools.JOB_TITLES)
    c.b.title(c.person(person, parts=c.pick(2, 3), script=script))
    c.b.header(c.t(job.ar, job.en, job.en))
    shuffled_fields(
        c,
        [
            (c.lbl("الهاتف", "Phone"), [c.mobile()]),
            (c.lbl("البريد الإلكتروني", "Email"), [c.email(person)]),
            (c.lbl("العنوان", "Address"), [c.address()]),
            (c.lbl("تاريخ الميلاد", "Date of birth"), [c.birth_date(person)]),
        ],
    )
    c.b.blank()
    c.b.header(c.lbl("الخبرة العملية", "Work experience"))
    year = c.rng.randint(2010, 2019)
    for i, org in enumerate(c.v.orgs(c.rng.randint(2, 3), "ngo", "company", "hospital")):
        role = c.pick(*pools.JOB_TITLES)
        end = c.year(year + c.rng.randint(1, 3)) if i else c.t("حتى الآن", "present", "present")
        start = c.year(year)
        c.b.para(
            start, " - ", end, ": ", c.t(role.ar, role.en, role.en), *joined(c, "", c.org(org), c.place(c.v.city()))[1:]
        )
        year -= c.rng.randint(2, 4)
    c.b.blank()
    c.b.header(c.lbl("التعليم", "Education"))
    degree, school = c.pick(*pools.DEGREES), c.v.org("university")
    c.b.para(*joined(c, c.t(degree.ar, degree.en), c.org(school), c.place(c.v.city()), c.year(year - 1)))
    c.b.blank()
    skills = c.rng.sample(pools.SKILLS, 3)
    c.b.field(c.lbl("المهارات", "Skills"), *joined(c, *(c.t(s.ar, s.en, s.en) for s in skills)))
    c.b.field(
        c.lbl("اللغات", "Languages"),
        c.t(
            "العربية (اللغة الأم)، الإنجليزية (جيد جداً)",
            "Arabic (native), English (fluent)",
            "Arabic (native), English (fluent)",
        ),
    )
    c.b.blank()
    c.b.header(c.lbl("المعرّفون", "References"))
    for _ in range(c.rng.randint(1, 2)):
        ref, ref_org = c.v.person(), c.v.org()
        contact = c.mobile() if c.rng.random() < 0.6 else c.email(ref, ref_org.domain)
        c.b.para(*joined(c, c.person(ref, parts=2), c.org(ref_org), contact))
    footer(c)
    return PAGE
