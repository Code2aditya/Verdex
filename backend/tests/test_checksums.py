"""Checksum tests — Aadhaar Verhoeff + PAN check character heuristics."""

from __future__ import annotations

import pytest

from backend.security.checksums import (
    is_valid_aadhaar,
    is_valid_pan,
    pan_check_char,
    verhoeff_valid,
)

VALID_AADHAARS = ["526018159089", "301661318609", "913909960301"]
VALID_PANS = ["ABCPF1234F", "ABCPG1234G", "ABCPH1234H", "ABCPJ1234J"]


@pytest.mark.parametrize("aadhaar", VALID_AADHAARS)
def test_valid_aadhaars(aadhaar: str) -> None:
    assert is_valid_aadhaar(aadhaar) is True


@pytest.mark.parametrize("aadhaar", ["000000000000", "526018159081", "123456789012", "367594039629"])
def test_invalid_aadhaars(aadhaar: str) -> None:
    assert is_valid_aadhaar(aadhaar) is False


def test_spaced_aadhaar_is_accepted() -> None:
    assert is_valid_aadhaar("5260 1815 9089") is True


@pytest.mark.parametrize("pan", VALID_PANS)
def test_valid_pan_check_chars(pan: str) -> None:
    assert is_valid_pan(pan) is True
    assert pan_check_char(pan[:9]) == pan[9]


@pytest.mark.parametrize("pan", ["ABCDE1234X", "AAAAA0000A", "ABCPF1234Z", "abcde1234f"])
def test_invalid_pan_check_chars(pan: str) -> None:
    assert is_valid_pan(pan) is False


def test_verhoeff_digit_sanity() -> None:
    assert verhoeff_valid([int(c) for c in "123456789099"]) is True
    assert verhoeff_valid([int(c) for c in "123456789098"]) is False


def test_pan_check_char_generation() -> None:
    assert pan_check_char("ABCPF1234") == "F"
    assert len({pan_check_char(p[:9]) for p in VALID_PANS}) == len(VALID_PANS)