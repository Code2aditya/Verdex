"""Module 1 — field extraction from raw OCR lines.

Label-based extraction: OCR lines shaped like "Label: Value" are matched
against a per-document-type alias map (built from the YAML rule field
names) and normalised into the field dict Module 2 consumes. Ported from
VerdexZip/app/ocr/field_extractor.py and extended with pan_card.
"""

from __future__ import annotations

import re
from datetime import datetime

from backend.config.rule_loader import supported_document_types

_DATE_FIELDS = {
    "date_of_birth", "date_of_issue", "date_of_expiry",
    "validity_start", "validity_end",
}

_DATE_INPUT_FORMATS = [
    "%d %b %Y",
    "%d %B %Y",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
]


def normalize_date(raw: str) -> str:
    """Best-effort: convert a printed date string to ISO YYYY-MM-DD."""
    cleaned = raw.strip().upper()
    for fmt in _DATE_INPUT_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return raw


_FIELD_ALIASES: dict[str, dict[str, str]] = {
    "pan_card": {
        "permanent account number": "pan_number",
        "pan": "pan_number",
        "pan no": "pan_number",
        "name": "full_name",
        "father's name": "father_name",
        "father name": "father_name",
        "date of birth": "date_of_birth",
        "dob": "date_of_birth",
        "signature": "signature",
    },
    "passport": {
        "surname": "surname", "given names": "given_names",
        "passport no": "passport_number", "passport number": "passport_number",
        "nationality": "nationality", "date of birth": "date_of_birth",
        "date of issue": "date_of_issue", "date of expiry": "date_of_expiry",
        "sex": "gender", "place of birth": "place_of_birth",
    },
    "visa": {
        "visa no": "visa_number", "visa number": "visa_number",
        "visa type": "visa_type", "entry": "entry_type",
        "duration of stay": "stay_duration_days", "valid from": "validity_start",
        "valid until": "validity_end", "issuing country": "issuing_country",
    },
    "voter_id": {
        "epic no": "epic_number", "epic number": "epic_number",
        "name": "full_name", "father's name": "relative_name",
        "husband's name": "relative_name", "date of birth": "date_of_birth",
        "age": "age", "sex": "gender", "address": "address",
    },
    "driving_licence": {
        "dl no": "dl_number", "date of birth": "date_of_birth",
        "date of issue": "date_of_issue", "valid till": "date_of_expiry",
        "cov": "vehicle_classes", "address": "address",
    },
    "citizenship_certificate": {
        "certificate no": "certificate_number", "name": "full_name",
        "date of birth": "date_of_birth", "address": "address",
        "issuing authority": "issuing_authority", "date of issue": "date_of_issue",
    },
    "aadhaar": {
        "aadhaar no": "aadhaar_number", "name": "full_name",
        "dob": "date_of_birth", "date of birth": "date_of_birth",
        "gender": "gender", "address": "address", "vid": "vid",
    },
}

_LABEL_LINE_RE = re.compile(r"^(?P<label>[A-Za-z][A-Za-z /'.]*?)\s*:\s*(?P<value>.+)$")


def extract_fields(document_type: str, lines: list[str]) -> dict[str, str]:
    """Extract named fields from raw OCR lines for a document type."""
    if document_type not in _FIELD_ALIASES:
        raise ValueError(
            f"No field extractor configured for document_type={document_type!r}. "
            f"Known types: {sorted(_FIELD_ALIASES)}"
        )
    aliases = _FIELD_ALIASES[document_type]
    fields: dict[str, str] = {}

    for line in lines:
        match = _LABEL_LINE_RE.match(str(line).strip())
        if not match:
            continue
        raw_label = match.group("label").strip().lower()
        raw_label = raw_label.split("/")[0].strip()
        value = match.group("value").strip()

        field_name = aliases.get(raw_label)
        if field_name:
            if field_name in _DATE_FIELDS:
                value = normalize_date(value)
            fields[field_name] = value

    if document_type == "passport" and "full_name" not in fields:
        surname = fields.get("surname", "")
        given = fields.get("given_names", "")
        combined = f"{surname} {given}".strip()
        if combined:
            fields["full_name"] = combined

    return fields


def supported_document_types() -> list[str]:
    configured = set(supported_document_types())
    extractable = set(_FIELD_ALIASES)
    missing = configured - extractable
    if missing:
        raise RuntimeError(f"Missing field extractor(s) for: {sorted(missing)}")
    return sorted(extractable)