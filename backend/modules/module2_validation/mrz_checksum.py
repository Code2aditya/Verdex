"""Module 2 — MRZ checksum validation (ICAO Doc 9303, TD3).

Re-exports the TD3 parser from module1_ocr with an ISO-date helper
used by the rules engine's mrz-vs-visual cross-checks.
"""

from __future__ import annotations

from datetime import date

from backend.modules.module1_ocr.mrz_parser import (
    compute_check_digit,
    parse_and_validate_td3,
    verify_check_digit,
)

__all__ = ["compute_check_digit", "verify_check_digit", "parse_and_validate_td3", "mrz_date_to_iso"]


def mrz_date_to_iso(mrz_date: str, field: str = "dob") -> str | None:
    """Convert an MRZ YYMMDD date to ISO YYYY-MM-DD.

    ICAO 9303 does not encode a century. DOB uses the standard reader
    heuristic (year > current 2-digit year -> 19xx, else 20xx); expiry
    always resolves to 20xx.
    """
    if not mrz_date or len(mrz_date) != 6 or not mrz_date.isdigit():
        return None
    yy, mm, dd = mrz_date[0:2], mrz_date[2:4], mrz_date[4:6]
    if field == "expiry":
        century = "20"
    else:
        current_yy = date.today().year % 100
        century = "19" if int(yy) > current_yy else "20"
    try:
        return date(int(century + yy), int(mm), int(dd)).isoformat()
    except ValueError:
        return None