"""Form-like and tabular documents: HR records, invoices and contact lists."""

from __future__ import annotations

from .. import pools
from ..builder import Ent, Part
from ..context import Ctx
from .common import PAGE, footer, joined, letterhead, shuffled_fields

VAT_RATE = 0.16


def hr_record(c: Ctx) -> str:
    org = c.v.org("ngo", "company", "hospital", "university")
    employee = c.v.person(min_age=20, max_age=60)
    c.b.title(c.org(org))
    c.b.header(c.t("سجل موظف", "Employee record", "سجل موظف / Employee Record"))
    c.b.blank()
    c.b.field(c.lbl("الرقم الوظيفي", "Employee no."), c.reference())
    c.b.field(c.lbl("الاسم الكامل", "Full name"), c.person(employee, parts=c.pick(3, 4)))
    nationality = c.rng.choices(("JO", "SY", "OTHER"), weights=(6, 3, 1))[0]
    c.meta["nationality"] = nationality
    if nationality == "JO":
        c.b.field(c.lbl("الرقم الوطني", "National no."), Ent(c.num(c.v.jo_national_id(employee)), "NATIONAL_ID"))
    elif nationality == "SY":
        c.b.field(c.lbl("رقم جواز السفر", "Passport no."), Ent(c.v.passport_number(), "PASSPORT"))
        c.b.field(c.lbl("رقم ملف المفوضية", "UNHCR case no."), Ent(c.v.refugee_case_number(), "REFUGEE_ID"))
        if c.rng.random() < 0.5:
            c.b.field(c.lbl("رقم بطاقة الخدمة", "MOI card no."), Ent(c.v.moi_card_number(), "REFUGEE_ID"))
    else:
        c.b.field(c.lbl("رقم جواز السفر", "Passport no."), Ent(c.v.passport_number(), "PASSPORT"))
    shuffled_fields(
        c,
        [
            (c.lbl("تاريخ الميلاد", "Date of birth"), [c.birth_date(employee)]),
            (c.lbl("مكان الولادة", "Place of birth"), [c.place(c.v.birth_place())]),
            (c.lbl("الهاتف المحمول", "Mobile"), [c.mobile()]),
            (c.lbl("البريد الإلكتروني", "Email"), [c.email(employee)]),
            (c.lbl("العنوان", "Address"), [c.address()]),
        ],
    )
    job, department = c.pick(*pools.JOB_TITLES), c.pick(*pools.DEPARTMENTS)
    c.b.field(c.lbl("القسم", "Department"), c.t(department.ar, department.en, department.en))
    c.b.field(c.lbl("المسمى الوظيفي", "Job title"), c.t(job.ar, job.en, job.en))
    c.b.field(c.lbl("تاريخ المباشرة", "Start date"), c.date(c.v.past_date(10 * 365)))
    c.b.field(c.lbl("الراتب الأساسي", "Basic salary"), c.amount())
    bank = c.v.org("bank")
    c.b.field(c.lbl("البنك", "Bank"), c.org(bank))
    c.b.field("IBAN" if c.en else c.lbl("رقم الحساب الدولي", "IBAN"), c.iban(bank))
    relation, contact = c.v.relatives(employee, 1)[0]
    c.b.field(
        c.lbl("جهة الاتصال للطوارئ", "Emergency contact"),
        *joined(c, c.person(contact, parts=2), c.t(relation.name.ar, relation.name.en), c.any_phone()),
    )
    c.b.blank()
    c.b.para(
        *c.say(
            ar=[
                ["ملاحظات: تم تجديد العقد بتاريخ ", c.date(), " وانتهت فترة التجربة."],
                ["ملاحظات: رقم الملف الورقي ", c.reference(), "، يحفظ في قسم الموارد البشرية."],
            ],
            en=[
                ["Notes: contract renewed on ", c.date(), "; probation completed."],
                ["Notes: paper file ", c.reference(), " is kept by Human Resources."],
            ],
            mixed=[["ملاحظات: الـ contract تجدد بتاريخ ", c.date(), " و الـ probation خلصت."]],
        )
    )
    footer(c)
    return PAGE


def invoice(c: Ctx) -> str:
    vendor = c.v.org("company")
    letterhead(c, vendor)
    c.b.field(c.lbl("الرقم الضريبي", "Tax no."), c.tax_number())
    c.b.blank()
    c.b.title(c.t("فاتورة ضريبية", "TAX INVOICE", "فاتورة ضريبية / TAX INVOICE"))
    c.b.field(c.lbl("رقم الفاتورة", "Invoice no."), c.invoice_number())
    c.b.field(c.lbl("التاريخ", "Date"), c.date(c.v.past_date(400)))
    c.b.field(c.lbl("رقم الطلب", "Order no."), c.order_number())
    if c.rng.random() < 0.5:
        customer = c.v.org("ngo", "hospital", "university")
        c.b.field(c.lbl("العميل", "Bill to"), c.org(customer))
        attention = c.v.person()
        c.b.field(c.lbl("بعناية", "Attn."), c.person(attention, parts=2), " - ", c.mobile())
    else:
        buyer = c.v.person()
        c.b.field(c.lbl("العميل", "Bill to"), c.person(buyer, parts=c.pick(2, 3)), " - ", c.mobile())
    if c.rng.random() < 0.6:
        c.b.field(c.lbl("عنوان العميل", "Customer address"), c.address())
    c.b.blank()
    c.b.row(
        c.lbl("البند", "Item"), c.lbl("الكمية", "Qty"), c.lbl("سعر الوحدة", "Unit price"), c.lbl("المجموع", "Total")
    )
    subtotal = 0
    for item in c.rng.sample(pools.INVOICE_ITEMS, c.rng.randint(2, 5)):
        qty, unit = c.rng.randint(1, 40), c.rng.randint(20, 4000) * 250
        subtotal += qty * unit
        c.b.row(c.t(item.ar, item.en), c.num(str(qty)), c.amount(unit), c.amount(qty * unit))
    tax = round(subtotal * VAT_RATE)
    c.b.blank()
    c.b.field(c.lbl("المجموع قبل الضريبة", "Subtotal"), c.amount(subtotal))
    c.b.field(c.lbl(f"ضريبة المبيعات {VAT_RATE:.0%}", f"Sales tax {VAT_RATE:.0%}"), c.amount(tax))
    c.b.field(c.lbl("الإجمالي", "Total due"), c.amount(subtotal + tax))
    c.b.blank()
    banks = c.v.orgs(c.rng.choice((1, 2)), "bank")
    for bank in banks:
        c.b.field(c.lbl("التحويل إلى", "Pay to"), c.org(bank), " - IBAN ", c.iban(bank))
    c.b.para(
        *c.say(
            ar=[["يستحق الدفع خلال 30 يوماً. للاستفسار: ", c.org_email(vendor)]],
            en=[["Payment is due within 30 days. Queries: ", c.org_email(vendor)]],
            mixed=[["Payment due خلال 30 يوم. للاستفسار: ", c.org_email(vendor)]],
        )
    )
    footer(c)
    return PAGE


def contact_list(c: Ctx) -> str:
    owner = c.v.org("ngo", "company")
    c.b.title(c.t("قائمة جهات الاتصال", "Contact list", "قائمة جهات الاتصال / Contact list"))
    c.b.header(c.org(owner))
    c.b.row(
        c.lbl("الاسم", "Name"), c.lbl("الجهة", "Organisation"), c.lbl("الهاتف", "Phone"),
        c.lbl("البريد الإلكتروني", "Email"), c.lbl("المدينة", "City"),
    )  # fmt: skip
    for _ in range(c.rng.randint(6, 12)):
        person, org = c.v.person(), c.v.org()
        email: Part = c.email(person, org.domain if c.rng.random() < 0.5 else None) if c.rng.random() < 0.85 else "-"
        city = c.v.city(c.rng.choices(("JO", "SY"), weights=(4, 1))[0])
        c.b.row(c.person(person, parts=c.pick(2, 2, 3)), c.org(org), c.any_phone(), email, c.place(city))
    c.b.blank()
    c.b.para(
        c.t("آخر تحديث: ", "Last updated: ", "Last updated / آخر تحديث: "),
        c.date(c.v.past_date(200)),
        c.t(" - للاستخدام الداخلي فقط", " - internal use only", " - internal use only"),
    )
    footer(c)
    return PAGE
