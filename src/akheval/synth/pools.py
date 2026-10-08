"""Value pools for the synthetic dataset. Arabic and Latin forms come in fixed pairs.

Names are common given and family names from Jordan, Syria and Palestine, combined at
random. Organisation names put invented stems into common name patterns; bank codes come from the stems.
"""

from __future__ import annotations

import re
from typing import NamedTuple


def latin_slug(name: str) -> str:
    """Lower-case ASCII letters of a Latin name, without an "Al-" prefix."""
    return re.sub(r"[^a-z]", "", re.sub(r"^al-", "", name.lower()))


class Pair(NamedTuple):
    ar: str
    en: str


def _pairs(*items: tuple[str, str]) -> tuple[Pair, ...]:
    return tuple(Pair(ar, en) for ar, en in items)


MALE_GIVEN = _pairs(
    ("محمد", "Mohammad"),
    ("أحمد", "Ahmad"),
    ("محمود", "Mahmoud"),
    ("خالد", "Khaled"),
    ("عمر", "Omar"),
    ("يوسف", "Yousef"),
    ("علي", "Ali"),
    ("حسن", "Hasan"),
    ("حسين", "Hussein"),
    ("إبراهيم", "Ibrahim"),
    ("عبدالله", "Abdullah"),
    ("عبدالرحمن", "Abdelrahman"),
    ("سامر", "Samer"),
    ("طارق", "Tareq"),
    ("فراس", "Firas"),
    ("ليث", "Laith"),
    ("زيد", "Zaid"),
    ("معاذ", "Moath"),
    ("باسل", "Basel"),
    ("نضال", "Nidal"),
    ("رامي", "Rami"),
    ("سليمان", "Suleiman"),
    ("مصطفى", "Mustafa"),
    ("ماجد", "Majed"),
    ("وليد", "Walid"),
    ("هاني", "Hani"),
    ("سامي", "Sami"),
    ("فادي", "Fadi"),
    ("عماد", "Imad"),
    ("جمال", "Jamal"),
    ("كريم", "Karim"),
    ("مالك", "Malek"),
    ("يزن", "Yazan"),
    ("قيس", "Qais"),
    ("أنس", "Anas"),
    ("بلال", "Bilal"),
    ("حمزة", "Hamza"),
    ("صالح", "Saleh"),
    ("عادل", "Adel"),
    ("نبيل", "Nabil"),
    ("منذر", "Munther"),
    ("غسان", "Ghassan"),
    ("رائد", "Raed"),
    ("مهند", "Muhannad"),
    ("إياد", "Iyad"),
    ("أيمن", "Ayman"),
    ("وسيم", "Wassim"),
    ("مازن", "Mazen"),
    ("ثامر", "Thamer"),
    ("عدي", "Odai"),
    ("سيف", "Saif"),
    ("تامر", "Tamer"),
    ("هشام", "Hisham"),
    ("ياسر", "Yaser"),
    ("زياد", "Ziad"),
    ("ناصر", "Naser"),
    ("عمار", "Ammar"),
    ("أسامة", "Osama"),
    ("شادي", "Shadi"),
    ("نزار", "Nizar"),
    ("جورج", "George"),
    ("عيسى", "Issa"),
    ("حازم", "Hazem"),
    ("مراد", "Murad"),
)

FEMALE_GIVEN = _pairs(
    ("سارة", "Sara"),
    ("ليان", "Layan"),
    ("رنا", "Rana"),
    ("هبة", "Heba"),
    ("دانا", "Dana"),
    ("نور", "Noor"),
    ("آية", "Aya"),
    ("لينا", "Lina"),
    ("رهف", "Rahaf"),
    ("تالا", "Tala"),
    ("مريم", "Maryam"),
    ("فرح", "Farah"),
    ("ديمة", "Dima"),
    ("يارا", "Yara"),
    ("سلمى", "Salma"),
    ("رزان", "Razan"),
    ("عبير", "Abeer"),
    ("منى", "Mona"),
    ("هالة", "Hala"),
    ("سوسن", "Sawsan"),
    ("رغد", "Raghad"),
    ("جنى", "Jana"),
    ("لمى", "Lama"),
    ("ريم", "Reem"),
    ("أسماء", "Asmaa"),
    ("فاطمة", "Fatima"),
    ("خديجة", "Khadija"),
    ("زينب", "Zainab"),
    ("نادية", "Nadia"),
    ("سهى", "Suha"),
    ("ميس", "Mais"),
    ("بيان", "Bayan"),
    ("روان", "Rawan"),
    ("دعاء", "Duaa"),
    ("إسراء", "Israa"),
    ("حنين", "Haneen"),
    ("غدير", "Ghadeer"),
    ("وفاء", "Wafaa"),
    ("سناء", "Sanaa"),
    ("لبنى", "Lubna"),
    ("نسرين", "Nisreen"),
    ("ربى", "Ruba"),
    ("عائشة", "Aisha"),
    ("هيا", "Haya"),
    ("شهد", "Shahd"),
    ("ملك", "Malak"),
    ("رولا", "Rola"),
    ("سمر", "Samar"),
    ("رشا", "Rasha"),
    ("نوال", "Nawal"),
)

FAMILY = _pairs(
    ("الخطيب", "Al-Khatib"),
    ("العبادي", "Al-Abbadi"),
    ("الزعبي", "Al-Zoubi"),
    ("الحوراني", "Al-Hourani"),
    ("المجالي", "Al-Majali"),
    ("الطراونة", "Al-Tarawneh"),
    ("الشوابكة", "Al-Shawabkeh"),
    ("القيسي", "Al-Qaisi"),
    ("النعيمات", "Al-Nuaimat"),
    ("حدادين", "Haddadin"),
    ("الرفاعي", "Al-Rifai"),
    ("الدباس", "Al-Dabbas"),
    ("العمري", "Al-Omari"),
    ("بني هاني", "Bani Hani"),
    ("الخصاونة", "Al-Khasawneh"),
    ("عبيدات", "Obeidat"),
    ("الروسان", "Al-Rousan"),
    ("المعايطة", "Al-Maaytah"),
    ("الحباشنة", "Al-Habashneh"),
    ("العدوان", "Al-Adwan"),
    ("السرحان", "Al-Sarhan"),
    ("البطوش", "Al-Btoush"),
    ("العتوم", "Al-Atoum"),
    ("بني ياسين", "Bani Yaseen"),
    ("الكساسبة", "Al-Kasasbeh"),
    ("الضمور", "Al-Dmour"),
    ("النسور", "Al-Nsour"),
    ("القضاة", "Al-Qudah"),
    ("الصرايرة", "Al-Sarayrah"),
    ("الخلايلة", "Al-Khalayleh"),
    ("الشديفات", "Al-Shdeifat"),
    ("الحمايدة", "Al-Hmaideh"),
    ("الحلبي", "Al-Halabi"),
    ("الحمصي", "Al-Homsi"),
    ("الشامي", "Al-Shami"),
    ("الأتاسي", "Al-Atassi"),
    ("القباني", "Kabbani"),
    ("العظمة", "Al-Azmeh"),
    ("الحريري", "Al-Hariri"),
    ("المصري", "Al-Masri"),
    ("الإدلبي", "Al-Idlibi"),
    ("السباعي", "Al-Sibai"),
    ("الجندي", "Al-Jundi"),
    ("الخوري", "Al-Khoury"),
    ("حداد", "Haddad"),
    ("الأحمد", "Al-Ahmad"),
    ("الحسن", "Al-Hasan"),
    ("العلي", "Al-Ali"),
    ("الموسى", "Al-Mousa"),
    ("الصالح", "Al-Saleh"),
    ("الكردي", "Al-Kurdi"),
    ("الزين", "Al-Zein"),
    ("الديري", "Al-Deiri"),
    ("البيطار", "Al-Bitar"),
    ("الحمود", "Al-Hammoud"),
    ("العيسى", "Al-Issa"),
    ("الخليل", "Al-Khalil"),
    ("النابلسي", "Al-Nabulsi"),
    ("الخليلي", "Al-Khalili"),
    ("البرغوثي", "Barghouti"),
    ("الشريف", "Al-Sharif"),
    ("أبو زيد", "Abu Zaid"),
    ("أبو غوش", "Abu Ghosh"),
    ("الطويل", "Al-Taweel"),
    ("عودة", "Odeh"),
    ("حمدان", "Hamdan"),
    ("عبد الهادي", "Abdelhadi"),
    ("القواسمي", "Al-Qawasmi"),
    ("الجعبري", "Al-Jabari"),
    ("أبو عيشة", "Abu Aisheh"),
    ("الدجاني", "Al-Dajani"),
    ("صوالحة", "Sawalha"),
    ("زيادة", "Ziadeh"),
    ("أبو صالح", "Abu Saleh"),
    ("عواد", "Awad"),
    ("شحادة", "Shehadeh"),
    ("داود", "Dawood"),
    ("منصور", "Mansour"),
    ("سلامة", "Salameh"),
    ("يعقوب", "Yacoub"),
    ("جرار", "Jarrar"),
    ("طوقان", "Touqan"),
)

JO_CITIES = _pairs(
    ("عمّان", "Amman"),
    ("إربد", "Irbid"),
    ("الزرقاء", "Zarqa"),
    ("العقبة", "Aqaba"),
    ("السلط", "Salt"),
    ("مادبا", "Madaba"),
    ("الكرك", "Karak"),
    ("جرش", "Jerash"),
    ("المفرق", "Mafraq"),
    ("معان", "Ma'an"),
    ("عجلون", "Ajloun"),
    ("الطفيلة", "Tafilah"),
    ("الرمثا", "Ramtha"),
    ("الرصيفة", "Russeifa"),
    ("سحاب", "Sahab"),
    ("الفحيص", "Fuheis"),
)
SY_CITIES = _pairs(
    ("دمشق", "Damascus"),
    ("حلب", "Aleppo"),
    ("حمص", "Homs"),
    ("حماة", "Hama"),
    ("درعا", "Daraa"),
    ("اللاذقية", "Latakia"),
    ("إدلب", "Idlib"),
    ("دير الزور", "Deir ez-Zor"),
    ("الرقة", "Raqqa"),
    ("طرطوس", "Tartus"),
    ("السويداء", "As-Suwayda"),
    ("القامشلي", "Qamishli"),
    ("الحسكة", "Al-Hasakah"),
)
PS_CITIES = _pairs(
    ("نابلس", "Nablus"),
    ("الخليل", "Hebron"),
    ("جنين", "Jenin"),
    ("رام الله", "Ramallah"),
    ("بيت لحم", "Bethlehem"),
    ("طولكرم", "Tulkarm"),
    ("غزة", "Gaza"),
)
COUNTRIES = _pairs(
    ("الأردن", "Jordan"),
    ("سوريا", "Syria"),
    ("فلسطين", "Palestine"),
    ("لبنان", "Lebanon"),
    ("العراق", "Iraq"),
    ("مصر", "Egypt"),
    ("تركيا", "Turkey"),
    ("ألمانيا", "Germany"),
    ("كندا", "Canada"),
    ("السويد", "Sweden"),
)
CAMPS = _pairs(("مخيم الزعتري", "Zaatari Camp"), ("مخيم الأزرق", "Azraq Camp"))

DISTRICTS: dict[str, tuple[Pair, ...]] = {
    "Amman": _pairs(
        ("الصويفية", "Sweifieh"),
        ("تلاع العلي", "Tla' Al-Ali"),
        ("خلدا", "Khalda"),
        ("الجبيهة", "Jubeiha"),
        ("الشميساني", "Shmeisani"),
        ("جبل الحسين", "Jabal Al-Hussein"),
        ("عبدون", "Abdoun"),
        ("ماركا", "Marka"),
        ("طبربور", "Tabarbour"),
        ("أبو نصير", "Abu Nseir"),
        ("اللويبدة", "Weibdeh"),
        ("مرج الحمام", "Marj Al-Hamam"),
        ("الهاشمي الشمالي", "Hashmi Al-Shamali"),
        ("ضاحية الرشيد", "Dahiyat Al-Rasheed"),
    ),
    "Irbid": _pairs(("الحي الشرقي", "Al-Hay Al-Sharqi"), ("حي التركمان", "Hay Al-Turkman"), ("الحصن", "Al-Husn")),
    "Zarqa": _pairs(("الزرقاء الجديدة", "New Zarqa"), ("حي معصوم", "Hay Masoum"), ("الغويرية", "Al-Ghweiriyeh")),
    "Damascus": _pairs(
        ("المزة", "Mezzeh"), ("باب توما", "Bab Touma"), ("الميدان", "Al-Midan"), ("ركن الدين", "Rukn Al-Din")
    ),
    "Aleppo": _pairs(("الشعار", "Al-Shaar"), ("السليمانية", "Al-Sulaymaniyah"), ("حلب الجديدة", "New Aleppo")),
    "Homs": _pairs(("الوعر", "Al-Waer"), ("الإنشاءات", "Al-Inshaat"), ("الخالدية", "Al-Khalidiya")),
}
STREETS = _pairs(
    ("شارع الملكة رانيا", "Queen Rania Street"),
    ("شارع مكة", "Mecca Street"),
    ("شارع الجامعة", "University Street"),
    ("شارع وصفي التل", "Wasfi Al-Tal Street"),
    ("شارع الاستقلال", "Al-Istiqlal Street"),
    ("شارع الأمير حسن", "Prince Hassan Street"),
    ("شارع المدينة المنورة", "Al-Madina Al-Munawwara Street"),
    ("شارع زهران", "Zahran Street"),
    ("شارع الحرية", "Al-Hurriya Street"),
    ("شارع ابن خلدون", "Ibn Khaldoun Street"),
    ("شارع الشهيد", "Al-Shaheed Street"),
    ("شارع الياسمين", "Al-Yasmeen Street"),
    ("شارع البتراء", "Petra Street"),
    ("شارع الملك عبدالله الثاني", "King Abdullah II Street"),
    ("شارع الثورة", "Al-Thawra Street"),
    ("شارع بغداد", "Baghdad Street"),
    ("شارع النصر", "Al-Nasr Street"),
    ("شارع القدس", "Al-Quds Street"),
)


class OrgSpec(NamedTuple):
    ar: str
    en: str
    kind: str
    domain: str
    bank_code: str = ""


ORG_STEMS = _pairs(
    ("الأمل", "Al-Amal"),
    ("النهضة", "Nahda"),
    ("السنديان", "Sindyan"),
    ("النخيل", "Nakheel"),
    ("الكرمة", "Al-Karma"),
    ("الفجر", "Al-Fajr"),
    ("الشروق", "Al-Shorouq"),
    ("الرحمة", "Al-Rahma"),
    ("البلسم", "Balsam"),
    ("الندى", "Al-Nada"),
    ("الأفق", "Ufuq"),
    ("الضفاف", "Difaf"),
    ("الوادي", "Al-Wadi"),
    ("الشرق", "Al-Sharq"),
    ("المعرفة", "Maarifa"),
    ("الحوار", "Hiwar"),
    ("صدى", "Sada"),
    ("دروب", "Duroob"),
    ("غيث", "Ghaith"),
    ("منارة", "Manara"),
    ("سنابل", "Sanabel"),
    ("ينابيع", "Yanabee"),
    ("رواحل", "Rawahel"),
    ("أرجوان", "Orjowan"),
    ("قنديل", "Qandeel"),
    ("مرايا", "Maraya"),
    ("الوئام", "Al-Wiam"),
    ("الصنوبر", "Al-Sanobar"),
    ("الريادة", "Riyada"),
    ("الجسور", "Al-Jusoor"),
    ("القمم", "Al-Qimam"),
    ("الواحة", "Al-Waha"),
    ("الربيع", "Al-Rabee"),
    ("الياقوت", "Al-Yaqout"),
    ("المرجان", "Al-Murjan"),
    ("النرجس", "Narjis"),
    ("البنفسج", "Banafsaj"),
    ("الخزامى", "Khuzama"),
    ("الإشراق", "Ishraq"),
    ("الضياء", "Al-Diyaa"),
)
"""Invented name stems. The first four letters of their Latin slugs differ, so each bank gets its own code."""

ORG_PATTERNS = (
    ("ngo", "جمعية {} الخيرية", "{} Charity Society", "charity.org"),
    ("ngo", "مؤسسة {} للتنمية المجتمعية", "{} Community Development Foundation", "dev.org"),
    ("ngo", "مركز {} للدراسات", "{} Centre for Studies", "studies.org"),
    ("company", "شركة {} للتقنية", "{} Technologies", "tech.jo"),
    ("company", "شركة {} للتجارة العامة", "{} General Trading", "trading.com"),
    ("company", "شركة {} للاستشارات", "{} Consulting", "consult.com"),
    ("bank", "بنك {}", "{} Bank", "bank.jo"),
    ("hospital", "مستشفى {} التخصصي", "{} Specialty Hospital", "hospital.jo"),
    ("hospital", "مركز {} الطبي", "{} Medical Centre", "medical.jo"),
    ("university", "كلية {} للعلوم التطبيقية", "{} College of Applied Sciences", "college.edu.jo"),
    ("university", "أكاديمية {} للتدريب", "{} Training Academy", "academy.jo"),
)
"""Kind, Arabic and Latin name pattern, and domain suffix of the organisations built on every stem."""


def _org(stem: Pair, kind: str, ar: str, en: str, suffix: str) -> OrgSpec:
    slug = latin_slug(stem.en)
    code = slug[:4].upper() if kind == "bank" else ""
    return OrgSpec(ar.format(stem.ar), en.format(stem.en), kind, f"{slug}-{suffix}", code)


ORGS: tuple[OrgSpec, ...] = tuple(sorted(_org(stem, *pattern) for stem in ORG_STEMS for pattern in ORG_PATTERNS))

JO_MOI = Pair("وزارة الداخلية", "Ministry of Interior")
JO_CSPD = Pair("دائرة الأحوال المدنية والجوازات", "Civil Status and Passports Department")
JO_KINGDOM = Pair("المملكة الأردنية الهاشمية", "The Hashemite Kingdom of Jordan")
SY_REPUBLIC = Pair("الجمهورية العربية السورية", "Syrian Arab Republic")
SY_MOI = Pair("وزارة الداخلية", "Ministry of Interior")
UNHCR = Pair("المفوضية السامية للأمم المتحدة لشؤون اللاجئين", "UNHCR")

WEBMAIL_DOMAINS = ("gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com", "live.com")

JOB_TITLES = _pairs(
    ("منسق مشاريع", "Project Coordinator"),
    ("محاسب", "Accountant"),
    ("مهندس برمجيات", "Software Engineer"),
    ("مسؤول موارد بشرية", "HR Officer"),
    ("محلل بيانات", "Data Analyst"),
    ("باحث ميداني", "Field Researcher"),
    ("مدير برامج", "Programme Manager"),
    ("مسؤول حماية", "Protection Officer"),
    ("مصمم جرافيك", "Graphic Designer"),
    ("مسؤول تقنية المعلومات", "IT Officer"),
    ("مترجم", "Translator"),
    ("ممرض", "Nurse"),
)
DEPARTMENTS = _pairs(
    ("المالية", "Finance"),
    ("الموارد البشرية", "Human Resources"),
    ("البرامج", "Programmes"),
    ("تقنية المعلومات", "IT"),
    ("الاتصال", "Communications"),
    ("العمليات", "Operations"),
)
SKILLS = _pairs(
    ("تحليل البيانات", "data analysis"),
    ("إدارة المشاريع", "project management"),
    ("الترجمة", "translation"),
    ("المحاسبة", "accounting"),
    ("بايثون", "Python"),
    ("التصميم الجرافيكي", "graphic design"),
    ("كتابة التقارير", "report writing"),
    ("الدعم النفسي الاجتماعي", "psychosocial support"),
    ("إدارة الحالات", "case management"),
    ("Excel", "Excel"),
)
DEGREES = _pairs(
    ("بكالوريوس نظم المعلومات", "BSc Information Systems"),
    ("بكالوريوس المحاسبة", "BA Accounting"),
    ("ماجستير العمل الاجتماعي", "MA Social Work"),
    ("بكالوريوس اللغة الإنجليزية", "BA English Language"),
    ("دبلوم التمريض", "Diploma in Nursing"),
    ("ماجستير إدارة الأعمال", "MBA"),
)
INVOICE_ITEMS = _pairs(
    ("خدمات استشارية", "Consulting services"),
    ("تدريب ميداني", "Field training"),
    ("طباعة كتيبات", "Booklet printing"),
    ("استضافة موقع إلكتروني", "Website hosting"),
    ("ترجمة وثائق", "Document translation"),
    ("أجهزة حاسوب محمولة", "Laptops"),
    ("قرطاسية", "Stationery"),
    ("نقل وتوصيل", "Transport and delivery"),
    ("صيانة شبكة", "Network maintenance"),
    ("تصميم هوية بصرية", "Brand design"),
)


class Relation(NamedTuple):
    name: Pair
    female: bool
    generation: int
    """1 for a parent, 0 for a spouse or sibling, -1 for a child."""


RELATIONS = (
    Relation(Pair("الأب", "Father"), False, 1),
    Relation(Pair("الأم", "Mother"), True, 1),
    Relation(Pair("الزوج", "Husband"), False, 0),
    Relation(Pair("الزوجة", "Wife"), True, 0),
    Relation(Pair("الأخ", "Brother"), False, 0),
    Relation(Pair("الأخت", "Sister"), True, 0),
    Relation(Pair("الابن", "Son"), False, -1),
    Relation(Pair("الابنة", "Daughter"), True, -1),
)
OWN_FAMILY_NAME = ("Husband", "Wife", "Mother")
"""Relatives who keep their own family name: Arab women keep theirs after marriage."""
