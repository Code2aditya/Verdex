"""Module 1 — MRZ line extraction and ICAO 9303 TD3 parsing.

Only TD3 (passport, 2 lines x 44 chars) is supported, matching the
Indian passport format. Check digits use the ICAO Doc 9303 algorithm:
characters map to 0-35, weighted by [7, 3, 1], summed, modulo 10.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_WEIGHTS = [7, 3, 1]


def _char_value(ch: str) -> int:
    if ch == "<":
        return 0
    if ch.isdigit():
        return int(ch)
    if ch.isalpha():
        return ord(ch.upper()) - ord("A") + 10
    raise ValueError(f"Invalid MRZ character: {ch!r}")


def compute_check_digit(data: str) -> int:
    total = sum(_char_value(ch) * _WEIGHTS[i % 3] for i, ch in enumerate(data))
    return total % 10


def verify_check_digit(data: str, check_digit: str) -> bool:
    if check_digit == "<" and set(data) <= {"<"}:
        return True
    try:
        return str(compute_check_digit(data)) == check_digit
    except ValueError:
        return False


@dataclass
class MRZResult:
    valid: bool
    document_number: str = ""
    date_of_birth: str = ""
    date_of_expiry: str = ""
    nationality: str = ""
    sex: str = ""
    names: str = ""
    surname: str = ""
    given_names: str = ""
    errors: list = field(default_factory=list)


def parse_and_validate_td3(line1: str, line2: str) -> MRZResult:
    errors = []
    line1 = line1.rstrip("\n")
    line2 = line2.rstrip("\n")

    if len(line1) != 44:
        errors.append(f"Line 1 length is {len(line1)}, expected 44")
    if len(line2) != 44:
        errors.append(f"Line 2 length is {len(line2)}, expected 44")
    if errors:
        return MRZResult(valid=False, errors=errors)

    doc_type = line1[0:2]
    issuing_country = line1[2:5]
    names_field = line1[5:44]
    surname, _, given = names_field.partition("<<")
    surname = surname.replace("<", " ").strip()
    given_names = given.replace("<", " ").strip()

    if doc_type[0] != "P":
        errors.append(f"Unexpected document type code: {doc_type!r} (expected 'P..')")

    doc_number = line2[0:9]
    doc_number_check = line2[9]
    nationality = line2[10:13]
    dob = line2[13:19]
    dob_check = line2[19]
    sex = line2[20]
    expiry = line2[21:27]
    expiry_check = line2[27]
    personal_number = line2[28:42]
    personal_number_check = line2[42]
    overall_check = line2[43]

    if not verify_check_digit(doc_number, doc_number_check):
        errors.append("Document number checksum failed")
    if not verify_check_digit(dob, dob_check):
        errors.append("Date of birth checksum failed")
    if not verify_check_digit(expiry, expiry_check):
        errors.append("Date of expiry checksum failed")
    if not verify_check_digit(personal_number, personal_number_check):
        errors.append("Personal number checksum failed")

    composite = (
        doc_number + doc_number_check
        + dob + dob_check
        + expiry + expiry_check
        + personal_number + personal_number_check
    )
    if not verify_check_digit(composite, overall_check):
        errors.append("Overall composite checksum failed")

    if sex not in ("M", "F", "<"):
        errors.append(f"Unexpected sex field value: {sex!r}")

    return MRZResult(
        valid=len(errors) == 0,
        document_number=doc_number.replace("<", ""),
        date_of_birth=dob,
        date_of_expiry=expiry,
        nationality=nationality.replace("<", ""),
        sex=sex,
        surname=surname,
        given_names=given_names,
        names=f"{surname} {given_names}".strip(),
        errors=errors,
    )


def extract_mrz_lines(ocr_output: dict) -> list[str]:
    """Heuristic: MRZ lines are long alnum/'<' strings (>= 30 chars)."""
    candidates = []
    for line in ocr_output.get("lines", []):
        text = str(line.get("text", "")).upper().replace(" ", "")
        if len(text) >= 30 and all(c.isalnum() or c == "<" for c in text):
            candidates.append(text)
    return candidates