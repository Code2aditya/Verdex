"""Module 2 validation tests — rules engine + watchlist."""

from __future__ import annotations

import pytest

from backend.modules.module2_validation import validate_document, watchlist_check


def test_aadhaar_valid_document() -> None:
    report = validate_document(
        "aadhaar",
        {
            "aadhaar_number": "5260 1815 9089",
            "full_name": "RAHUL KUMAR",
            "date_of_birth": "1995-04-11",
            "gender": "M",
            "address": "14 PARK STREET, KOLKATA 700016",
        },
    )
    assert report.result == "PASS"
    assert report.checks


def test_aadhaar_invalid_checksum_fails() -> None:
    report = validate_document(
        "aadhaar",
        {
            "aadhaar_number": "5260 1815 9081",
            "full_name": "RAHUL KUMAR",
            "date_of_birth": "1995-04-11",
            "gender": "M",
            "address": "14 PARK STREET, KOLKATA 700016",
        },
    )
    assert report.result == "FAIL"
    assert any(c.rule.startswith("checksum:") and not c.passed for c in report.checks)


def test_pan_requires_father_name() -> None:
    report = validate_document("pan_card", {"pan_number": "ABCPF1234F", "full_name": "ARJUN MEHTA"})
    assert report.result == "FAIL"
    assert any("father" in c.detail.lower() for c in report.checks)


def test_pan_with_father_name_passes() -> None:
    report = validate_document(
        "pan_card",
        {
            "pan_number": "ABCPF1234F",
            "full_name": "ARJUN MEHTA",
            "father_name": "SURESH MEHTA",
            "date_of_birth": "1988-02-29",
        },
    )
    assert report.result == "PASS"


def test_expired_document_flagged() -> None:
    report = validate_document(
        "passport",
        {
            "passport_number": "N1234567",
            "full_name": "ANITA SHARMA",
            "date_of_birth": "1990-01-15",
            "date_of_expiry": "2020-01-01",
            "nationality": "IND",
        },
    )
    assert report.result == "FAIL"
    assert any(c.rule == "expiry_status" and not c.passed for c in report.checks)


def test_watchlist_passport_hit() -> None:
    watch = watchlist_check("passport", "P9988776", "ANITA SHARMA")
    assert watch["hit"] is True
    assert watch["matches"][0]["doc_number"] == "P9988776"


def test_watchlist_name_alt_hit() -> None:
    watch = watchlist_check("passport", "N9999999", "deepak karki")
    assert watch["hit"] is True


def test_watchlist_clean() -> None:
    watch = watchlist_check("aadhaar", "526018159089", "RAHUL KUMAR")
    assert watch["hit"] is False
    assert watch["matches"] == []