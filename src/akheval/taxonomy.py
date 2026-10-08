"""PII taxonomy and the label maps that translate each detector's labels into it."""

from __future__ import annotations

from collections.abc import Mapping

TAXONOMY_VERSION = "2"

STRUCTURED: tuple[str, ...] = (
    "EMAIL",
    "PHONE",
    "NATIONAL_ID",
    "PASSPORT",
    "REFUGEE_ID",
    "IBAN",
    "IP_ADDRESS",
)
CONTEXTUAL: tuple[str, ...] = ("PERSON", "ORG", "LOCATION", "ADDRESS")
LABELS: tuple[str, ...] = STRUCTURED + CONTEXTUAL

NEGATIVE_KINDS: tuple[str, ...] = (
    "invoice_number",
    "reference_number",
    "numeric_sequence",
    "invalid_identifier",
    "date",
    "amount",
    "placeholder",
    "ambiguous_name",
    "version_number",
)

LabelMap = Mapping[str, str | None]


def family(label: str) -> str:
    """Return "structured" or "contextual" for a taxonomy label."""
    if label in STRUCTURED:
        return "structured"
    if label in CONTEXTUAL:
        return "contextual"
    raise KeyError(label)


def strip_bio(raw: str) -> str:
    """Drop a B-/I-/E-/S-/L-/U- prefix from a token-classification label."""
    if len(raw) > 2 and raw[1] == "-" and raw[0] in "BIESLU":
        return raw[2:]
    return raw


def map_label(table: LabelMap, raw: str) -> str | None:
    """Map a detector label onto the taxonomy; ``None`` means out of scope."""
    base = strip_bio(raw)
    return table.get(base, table.get(base.upper()))


CORE_LABELS: LabelMap = {
    "Email": "EMAIL",
    "Phone Number": "PHONE",
    "Jordanian National ID": "NATIONAL_ID",
    "Jordanian Passport": "PASSPORT",
    "IBAN": "IBAN",
    "IP Address": "IP_ADDRESS",
    "Syrian Refugee Document": "REFUGEE_ID",
    "UNHCR Number": "REFUGEE_ID",
    # Document classifiers from the August rule file; not PII entities.
    "Banking": None,
    "Payment Card": None,
    "UUID4": None,
    "Professional Resume": None,
}

NYM_LABELS: LabelMap = {
    "ACCOUNT_NUMBER": None,
    "AGE": None,
    "API_KEY": None,
    "BUILDING_NUMBER": "ADDRESS",
    "CITY": "LOCATION",
    "COMPANY_NAME": "ORG",
    "COUNTRY": "LOCATION",
    "CREDIT_DEBIT_CARD": None,
    "CUSTOMER_ID": None,
    "CVV": None,
    "DATE": None,
    "DATE_OF_BIRTH": None,
    "DRIVERS_LICENSE": None,
    "EMAIL": "EMAIL",
    "EMPLOYEE_ID": None,
    "FAX_NUMBER": "PHONE",
    "GENDER": None,
    "GIVEN_NAME": "PERSON",
    "GOVERNMENT_ID": "NATIONAL_ID",
    "IBAN": "IBAN",
    "LICENSE_PLATE": None,
    "MAC_ADDRESS": None,
    "MEDICAL_RECORD_NUMBER": None,
    "PASSPORT": "PASSPORT",
    "PASSWORD": None,
    "PHONE": "PHONE",
    "PIN": None,
    "ROUTING_NUMBER": None,
    "SECONDARY_ADDRESS": "ADDRESS",
    "SSN": "NATIONAL_ID",
    "STATE": "LOCATION",
    "STREET_ADDRESS": "ADDRESS",
    "STREET_NAME": "ADDRESS",
    "SURNAME": "PERSON",
    "SWIFT_BIC": None,
    "TAX_ID": None,
    "TIME": None,
    "URL": None,
    "USERNAME": None,
    "ZIP_CODE": "ADDRESS",
}

DIALOGPII_LABELS: LabelMap = {
    "PERSON": "PERSON",
    "PERSON_EMAIL": "EMAIL",
    "PERSON_SOCIAL_RELATION": None,
    "ORG": "ORG",
    "LOC_CITY": "LOCATION",
    "LOC_COUNTRY": "LOCATION",
    "LOC_STREET": "ADDRESS",
    "LOC_ZIP": "ADDRESS",
    "LOC_HOUSENUMBER": "ADDRESS",
    "LOC_OTHER": "LOCATION",
    "DATETIME": None,
    "DATETIME_AGE": None,
    "CODE": None,
    "CODE_PHONE": "PHONE",
    "CODE_URL": None,
    "PROFESSION": None,
    "PRODUCT": None,
    "QUANTITY": None,
    "MISC": None,
    # Labels the released checkpoint emits (crf_config.json) beyond those on its model card.
    "CITY": "LOCATION",
    "LOC_DISTRICT": "LOCATION",
    "LOC_NAME": "LOCATION",
    "NAME": "PERSON",
    "DISEASE": None,
}

CAMELBERT_LABELS: LabelMap = {
    "PERS": "PERSON",
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOCATION",
    "MISC": None,
}
