"""Indian document checksum utilities.

- Aadhaar: 12-digit number validated with the Verhoeff check-digit
  algorithm (the official UIDAI scheme). Digits only; spacing like
  "1234 5678 9012" is ignored.
- PAN: the 10th character is a check character derived from the first
  nine characters. The Income-Tax Department has not published the
  algorithm; this is the widely used heuristic implementation, treated
  as an advisory signal rather than an authoritative failure.
"""

from __future__ import annotations

import re
from typing import Iterable

# ── Verhoeff tables ────────────────────────────────────────────────────────────
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_valid(digits: Iterable[int]) -> bool:
    """Verhoeff checksum over the full digit sequence (last digit is the check)."""
    c = 0
    for i, d in enumerate(reversed(list(digits))):
        c = _D[c][_P[i % 8][d]]
    return c == 0


def _aadhaar_digits(value: object) -> list[int] | None:
    if value is None:
        return None
    text = str(value).replace(" ", "").replace("-", "").strip()
    if not re.fullmatch(r"[0-9]{12}", text):
        return None
    return [int(ch) for ch in text]


def is_valid_aadhaar(value: object) -> bool:
    """True when the value is a 12-digit Aadhaar number passing Verhoeff."""
    digits = _aadhaar_digits(value)
    return digits is not None and verhoeff_valid(digits)


# ── PAN check character (heuristic) ────────────────────────────────────────────
_ALPH = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def pan_check_char(pan9: str) -> str:
    """Derive the 10th (check) character from the first nine PAN characters."""
    total = sum(_ALPH.index(ch) * (2 if i % 2 else 1) for i, ch in enumerate(pan9))
    return _ALPH[total % 36]


def is_valid_pan(value: object) -> bool:
    """True when the value matches the PAN shape AND its check character."""
    text = (str(value) if value is not None else "").strip().upper()
    if not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", text):
        return False
    return pan_check_char(text[:9]) == text[9]