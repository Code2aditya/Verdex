"""Module 1 — document-type classification from OCR text.

Lightweight keyword classifier that picks the template before field
extraction runs. Extended to the Indian document set used by the app:
aadhaar, pan_card, passport, voter_id, driving_licence, visa,
citizenship_certificate.
"""

from __future__ import annotations

_CLASSIFIER_KEYWORDS: dict[str, list[str]] = {
    "pan_card": [
        "PERMANENT ACCOUNT NUMBER",
        "INCOME TAX DEPARTMENT",
        "INCOME TAX PAN",
        "GOVT OF INDIA",
    ],
    "aadhaar": ["AADHAAR", "UIDAI", "UNIQUE IDENTIFICATION"],
    "voter_id": ["ELECTION COMMISSION", "EPIC", "VOTER"],
    "driving_licence": ["DRIVING LICENCE", "DRIVING LICENSE", "DL NO"],
    "citizenship_certificate": ["CITIZENSHIP CERTIFICATE", "CITIZENSHIP"],
    "visa": ["VISA"],
    "passport": ["PASSPORT", "PASSEPORT"],
}


def classify_document_type(lines: list[str]) -> str | None:
    """Return the best-guess document type from raw OCR lines, or None."""
    joined = " ".join(lines).upper()
    for doc_type, keywords in _CLASSIFIER_KEYWORDS.items():
        if any(kw in joined for kw in keywords):
            return doc_type
    return None