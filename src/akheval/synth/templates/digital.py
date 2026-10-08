"""Digital-native documents: chat exports, IT incident reports and access logs, and a notice without PII."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence

from ..builder import Ent, Neg, Part
from ..context import Ctx
from ..pools import Pair
from ..values import Person
from .common import CHAT, PAGE, footer
from .prose import ambiguous_word

MINUTES_PER_DAY = 24 * 60
LOG_USERNAMES = ("root", "admin", "deploy", "backup", "test", "ubuntu", "oracle", "svc_web")
HOSTNAMES = ("srv-db-01", "srv-web-02", "fw-edge-01", "nas-office", "vpn-gw-01", "mail-01")
HTTP_PATHS = ("/admin", "/wp-login.php", "/api/v1/users", "/login", "/.env", "/index.php", "/reports/2026")
HTTP_STATUSES = (200, 301, 401, 403, 404, 500)
SYRIA = Pair("سوريا", "Syria")

Message = Callable[[Ctx, Person, Sequence[Person]], Sequence[Part]]


def _greeting(c: Ctx, sender: Person, others: Sequence[Person]) -> Sequence[Part]:
    del sender, others
    return c.say(
        ar=[["مرحبا، كيفك؟"], ["صباح الخير"], ["السلام عليكم"]],
        en=[["hi, how are you?"], ["morning!"], ["hey, got a minute?"]],
        mixed=[["hi, كيفك؟"], ["hala, shu el akhbar?"], ["good morning يا جماعة"]],
    )


def _filler(c: Ctx, sender: Person, others: Sequence[Person]) -> Sequence[Part]:
    del sender, others
    return c.say(
        ar=[["تمام"], ["يعطيك العافية"], ["شكراً كتير"], ["ماشي، بس أوصل بحكيلك"]],
        en=[["ok"], ["thanks!"], ["sure, will do"], ["on my way"]],
        mixed=[["ok تمام"], ["thanks كتير"], ["tamam, 5 minutes"], ["yalla bye"]],
    )


def _email(c: Ctx, sender: Person, others: Sequence[Person]) -> Sequence[Part]:
    del others
    email = c.email(sender)
    return c.say(
        ar=[["ابعتلي العقد على ", email], ["إيميلي ", email, " ابعت عليه"]],
        en=[["can you send the contract to ", email, "?"], ["my email is ", email]],
        mixed=[["ابعتلي الـ contract على ", email, " please"], ["ib3atli el file 3ala ", email]],
    )


def _phone(c: Ctx, sender: Person, others: Sequence[Person]) -> Sequence[Part]:
    del sender, others
    phone = c.mobile()
    return c.say(
        ar=[["رقمي الجديد ", phone, " سجله عندك"], ["اتصل على ", phone, " بعد الساعة 4"]],
        en=[["my new number is ", phone, ", save it"], ["call me on ", phone, " after 4"]],
        mixed=[["رقمي الجديد ", phone, ", save it"], ["hay ra2mi el jdeed ", phone]],
    )


def _syrian_phone(c: Ctx, sender: Person, others: Sequence[Person]) -> Sequence[Part]:
    del sender, others
    phone, syria = c.syrian_mobile(), c.place(SYRIA)
    return c.say(
        ar=[["رقم أخوي في ", syria, " ", phone]],
        en=[["my brother's number in ", syria, " is ", phone]],
        mixed=[["رقم أخوي in ", syria, " ", phone]],
    )


def _meeting(c: Ctx, sender: Person, others: Sequence[Person]) -> Sequence[Part]:
    del sender, others
    org, city = c.org(c.v.org()), c.place(c.v.city())
    return c.say(
        ar=[["بكرا الاجتماع بمكتب ", org, " في ", city, " الساعة 10"]],
        en=[["meeting tomorrow at the ", org, " office in ", city, ", 10am"]],
        mixed=[["el meeting bukra 3ind ", org, " f ", city], ["الـ meeting بكرا في ", org, "، ", city]],
    )


def _third_person(c: Ctx, sender: Person, others: Sequence[Person]) -> Sequence[Part]:
    del sender, others
    other = c.v.person()
    name, org, phone = c.person(other, parts=2), c.org(c.v.org()), c.any_phone()
    ar_number, mixed_number = ("رقمها", "her number") if other.female else ("رقمه", "his number")
    return c.say(
        ar=[["كلّم ", name, " من ", org, f"، {ar_number} ", phone]],
        en=[["call ", name, " from ", org, ", number is ", phone]],
        mixed=[["كلّم ", name, " from ", org, f", {mixed_number} ", phone]],
    )


def _iban(c: Ctx, sender: Person, others: Sequence[Person]) -> Sequence[Part]:
    del sender, others
    iban = c.iban()
    return c.say(
        ar=[["هاد الآيبان للتحويل ", iban]], en=[["IBAN for the transfer: ", iban]], mixed=[["el IBAN: ", iban]]
    )


def _address(c: Ctx, sender: Person, others: Sequence[Person]) -> Sequence[Part]:
    del sender, others
    address = c.address()
    return c.say(ar=[["العنوان: ", address]], en=[["the address is ", address]], mixed=[["el address: ", address]])


def _look_alike(c: Ctx, sender: Person, others: Sequence[Person]) -> Sequence[Part]:
    del sender, others
    return c.say(
        ar=[
            ["رقم الفاتورة ", c.invoice_number()],
            ["الطلبية رقم ", c.order_number(), " وصلت"],
            ["المجموع ", c.amount()],
        ],
        en=[
            ["invoice number is ", c.invoice_number()],
            ["order ", c.order_number(), " arrived"],
            ["total is ", c.amount()],
        ],
        mixed=[["el invoice no. ", c.invoice_number()], ["الـ order ", c.order_number(), " وصل"]],
    )


def _mention(c: Ctx, sender: Person, others: Sequence[Person]) -> Sequence[Part]:
    del sender
    name = c.given(c.rng.choice(others))
    return c.say(
        ar=[[name, " وصلت؟"], ["استنى ", name, " يرد"]],
        en=[["did ", name, " arrive?"], ["waiting for ", name]],
        mixed=[["wen ", name, "?"], ["استنى ", name, " please"]],
    )


CHAT_MESSAGES: tuple[Message, ...] = (
    _email, _phone, _syrian_phone, _meeting, _third_person, _iban, _address, _look_alike, _mention, _filler,
)  # fmt: skip


def _clock(minutes: int) -> str:
    return f"{minutes // 60 % 24:02d}:{minutes % 60:02d}"


def chat(c: Ctx) -> str:
    people = [c.v.person() for _ in range(c.rng.choice((2, 2, 3)))]
    script = c.script()
    names = [Ent(getattr(p.given, script), "PERSON") for p in people]
    c.b.header(c.date(c.v.past_date(400)))
    minutes = c.rng.randint(7 * 60, 21 * 60)
    messages: list[Message] = [_greeting, *c.rng.sample(CHAT_MESSAGES, c.rng.randint(4, 8)), _filler]
    for i, message in enumerate(messages):
        sender = i % len(people) if c.rng.random() < 0.8 else c.rng.randrange(len(people))
        others = people[:sender] + people[sender + 1 :]
        c.b.message(_clock(minutes), names[sender], *message(c, people[sender], others))
        minutes = (minutes + c.rng.randint(0, 25)) % MINUTES_PER_DAY
    return CHAT


def _log_time(c: Ctx, base: dt.datetime) -> tuple[dt.datetime, Neg]:
    stamp = base + dt.timedelta(seconds=c.rng.randint(0, 3600))
    return stamp, Neg(f"{stamp:%Y-%m-%d %H:%M:%S}", "date")


def _log_line(c: Ctx, base: dt.datetime, source: Ent, target: Ent) -> list[Part]:
    _, stamp = _log_time(c, base)
    user = c.pick(*LOG_USERNAMES)
    port = c.rng.randint(1024, 65535)
    return c.pick(
        [
            stamp,
            f" sshd[{c.rng.randint(900, 9999)}]: Failed password for invalid user {user} from ",
            source,
            f" port {port} ssh2",
        ],
        [stamp, " kernel: [UFW BLOCK] SRC=", source, " DST=", target, f" PROTO=TCP SPT={port} DPT=22"],
        [stamp, f" sshd[{c.rng.randint(900, 9999)}]: Accepted publickey for {user} from ", source, f" port {port}"],
        [
            stamp,
            " nginx: ",
            Ent(c.v.ipv6(), "IP_ADDRESS"),
            f' "GET {c.pick(*HTTP_PATHS)} HTTP/1.1" {c.pick(*HTTP_STATUSES)}',
        ],
    )


def it_incident(c: Ctx) -> str:
    variant = c.rng.choices(("incident_report", "access_log"), weights=(3, 2))[0]
    c.meta["variant"] = variant
    org = c.v.org("ngo", "company", "hospital")
    day = c.v.past_date(300)
    base = dt.datetime(day.year, day.month, day.day, c.rng.randint(0, 22))
    if variant == "access_log":
        return _access_log(c, c.org(org), base)
    reporter, owner = c.v.person(), c.v.person()
    attacker, host = Ent(c.v.ipv4(), "IP_ADDRESS"), Ent(c.v.ipv4(), "IP_ADDRESS")
    c.b.title(c.org(org))
    c.b.header(c.t("تقرير حادثة أمنية", "Security incident report", "تقرير حادثة أمنية / Security Incident Report"))
    c.b.field(c.lbl("رقم الحادثة", "Incident no."), c.reference())
    c.b.field(c.lbl("التاريخ", "Date"), Neg(f"{day:%Y-%m-%d}", "date"))
    c.b.field(
        c.lbl("أعد التقرير", "Reported by"), c.person(reporter, parts=2), " <", c.email(reporter, org.domain), ">"
    )
    c.b.field(c.lbl("الجهاز المتأثر", "Affected host"), f"{c.pick(*HOSTNAMES)} (", host, ")")
    c.b.field(c.lbl("إصدار الوكيل", "Agent build"), c.version())
    c.b.blank()
    failures = str(c.rng.randint(12, 400))
    c.b.para(
        *c.say(
            ar=[
                [
                    "رصد النظام ",
                    failures,
                    " محاولة دخول فاشلة عبر SSH من العنوان ",
                    attacker,
                    " استهدفت الخادم ",
                    host,
                    ".",
                ]
            ],
            en=[["The system logged ", failures, " failed SSH logins from ", attacker, " against ", host, "."]],
            mixed=[["النظام سجل ", failures, " failed SSH logins من ", attacker, " على الـ server ", host, "."]],
        )
    )
    c.b.para(
        *c.say(
            ar=[
                [
                    "قام وكيل الحماية (الإصدار ",
                    c.version(),
                    ") بحظر المصدر، وتمت إعادة تعيين كلمة مرور حساب ",
                    c.person(owner, parts=2),
                    ".",
                ]
            ],
            en=[
                [
                    "The endpoint agent (build ",
                    c.version(),
                    ") blocked the source, and the password of ",
                    c.person(owner, parts=2),
                    " was reset.",
                ]
            ],
            mixed=[
                [
                    "الـ endpoint agent (build ",
                    c.version(),
                    ") عمل block للـ source، وعملنا reset لباسوورد ",
                    c.person(owner, parts=2),
                    ".",
                ]
            ],
        )
    )
    c.b.blank()
    c.b.header(c.lbl("مقتطف من السجل", "Log excerpt"))
    for _ in range(c.rng.randint(3, 6)):
        c.b.para(*_log_line(c, base, attacker if c.rng.random() < 0.7 else Ent(c.v.ipv4(), "IP_ADDRESS"), host))
    c.b.blank()
    c.b.para(
        *c.say(
            ar=[["للمتابعة تواصلوا مع فريق تقنية المعلومات على ", c.org_email(org), " أو ", c.landline(), "."]],
            en=[["Follow-up: contact the IT team at ", c.org_email(org), " or ", c.landline(), "."]],
        )
    )
    footer(c)
    return PAGE


def _access_log(c: Ctx, org: Ent, base: dt.datetime) -> str:
    host = c.pick(*HOSTNAMES)
    c.b.title(c.t(f"سجل الوصول - {host}", f"access.log - {host}", f"سجل الوصول / access.log - {host}"))
    c.b.header(org)
    for _ in range(c.rng.randint(8, 16)):
        stamp, _ = _log_time(c, base)
        source = Ent(c.v.ipv4() if c.rng.random() < 0.8 else c.v.ipv6(), "IP_ADDRESS")
        agent: list[Part] = ["curl/8.5.0"] if c.rng.random() < 0.5 else ["backup-agent/", c.version()]
        c.b.para(
            source,
            " - - [",
            Neg(f"{stamp:%d/%b/%Y:%H:%M:%S} +0300", "date"),
            f'] "{c.pick("GET", "POST")} {c.pick(*HTTP_PATHS)} HTTP/1.1" ',
            f'{c.pick(*HTTP_STATUSES)} {c.rng.randint(120, 9000)} "-" "',
            *agent,
            '"',
        )
    exporter = c.v.person()
    c.b.blank()
    c.b.para(
        c.t("صدّره ", "Exported by ", "صدّره / Exported by "),
        c.person(exporter, parts=2),
        " (",
        c.email(exporter),
        c.t(") بتاريخ ", ") on ", ") on "),
        c.date(c.v.past_date(30)),
    )
    return PAGE


PLACEHOLDERS = ("name@example.com", "user@example.org", "+962 7X XXX XXXX", "07XXXXXXXX", "9XXXXXXXXX")


def notice_no_pii(c: Ctx) -> str:
    """An internal circular with look-alike values only: dates, amounts, references and placeholders."""
    c.b.title(c.t("تعميم داخلي", "Internal circular", "تعميم داخلي / Internal Circular"))
    c.b.field(c.lbl("التاريخ", "Date"), c.date())
    c.b.field(c.lbl("الرقم", "No."), c.reference())
    c.b.blank()
    deadline, budget = c.date(c.v.future_date(60)), c.amount()
    c.b.para(
        *c.say(
            ar=[
                [
                    "على جميع الأقسام تسليم تقارير الربع الثالث قبل ",
                    deadline,
                    ". وقد اعتمدت الموازنة بقيمة ",
                    budget,
                    ".",
                ]
            ],
            en=[
                [
                    "All departments must submit their third-quarter reports before ",
                    deadline,
                    ". The approved budget is ",
                    budget,
                    ".",
                ]
            ],
            mixed=[["على كل الأقسام تسليم الـ Q3 reports قبل ", deadline, ". الـ budget المعتمد ", budget, "."]],
        )
    )
    email, phone = c.pick(*PLACEHOLDERS[:2]), c.pick(*PLACEHOLDERS[2:])
    c.b.para(
        *c.say(
            ar=[
                [
                    "عند تعبئة النموذج استخدموا الصيغة التالية: ",
                    Neg(email, "placeholder"),
                    " و ",
                    Neg(phone, "placeholder"),
                    "، ولا ترسلوا بيانات شخصية.",
                ]
            ],
            en=[
                [
                    "Fill in the form fields as in the example ",
                    Neg(email, "placeholder"),
                    " and ",
                    Neg(phone, "placeholder"),
                    "; do not send personal data.",
                ]
            ],
        )
    )
    c.b.para(
        *c.say(
            ar=[["المرجع: الفاتورة رقم ", c.invoice_number(), " والطلب رقم ", c.order_number(), "."]],
            en=[["Reference: invoice ", c.invoice_number(), " and order no. ", c.order_number(), "."]],
        )
    )
    c.b.para(
        *c.say(
            ar=[
                ["الحساب التجريبي ", c.invalid_iban(), " غير صالح ولا يستخدم للتحويل."],
                ["رقم الملف القديم ", c.nine_digits(), " ألغي ولا يستخدم."],
            ],
            en=[
                ["The test account ", c.invalid_iban(), " is invalid; do not use it for transfers."],
                ["The old file number ", c.nine_digits(), " is no longer in use."],
            ],
        )
    )
    word, until = ambiguous_word(c), c.date(c.v.future_date(40))
    c.b.para(
        *c.say(
            ar=[
                ["قاعة ", word, " مغلقة للصيانة حتى ", until, "."],
                ["تستمر أعمال الصيانة في شارع ", word, " حتى ", until, "."],
            ],
            en=[
                ["The ", word, " meeting room is closed for maintenance until ", until, "."],
                ["Road works continue on ", word, " Street until ", until, "."],
            ],
            mixed=[
                ["قاعة ", word, " closed for maintenance لغاية ", until, "."],
                ["الـ road works في شارع ", word, " مستمرة لغاية ", until, "."],
            ],
        )
    )
    c.b.para(
        *c.say(
            ar=[["يرجى تحديث برنامج المكتب إلى الإصدار ", c.version(), "."]],
            en=[["Please update the office software to version ", c.version(), "."]],
            mixed=[["Please update الـ software للـ version ", c.version(), "."]],
        )
    )
    c.b.blank()
    c.b.para(c.t("الإدارة", "The Administration", "الإدارة / Administration"))
    footer(c)
    return PAGE
