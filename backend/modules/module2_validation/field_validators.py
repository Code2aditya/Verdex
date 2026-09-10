"""Module 2 — field-level validators and Indian checksum entry points.

validate_field() applies regex / allowed-values / checksum rules from
the YAML rule config and returns check records in the same shape the
rules engine produces.
"""

from __future__ import annotations

import re
from typing import Any

from backend.security.checksums import is_valid_aadhaar, is_valid_pan


def checksum_check(rule: str, algorithm: str | None, value: Any) -> dict | None:
    """Run the configured checksum (verhoeff for Aadhaar, pan_char for PAN).

    Returns a check record, or None when no checksum is configured.
    The PAN check-character algorithm is heuristic, so a failure is
    reported as a warning rather than a hard error.
    """
    if not algorithm or value is None or str(value).strip() == "":
        return None

    text = str(value).strip()
    if algorithm == "verhoeff":
        passed = is_valid_aadhaar(text)
        return {
            "rule": f"checksum:{text}",
            "passed": passed,
            "severity": "error" if not passed else "info",
            "detail": (
                "Aadhaar Verhoeff checksum valid"
                if passed
                else "Aadhaar Verhoeff checksum FAILED — number may be synthetically generated"
            ),
        }
    if algorithm == "pan_char":
        passed = is_valid_pan(text)
        return {
            "rule": f"checksum:{text}",
            "passed": passed,
            "severity": "warning",
            "detail": (
                "PAN check character valid"
                if passed
                else "PAN check character mismatch — heuristic algorithm (advisory)"
            ),
        }
    return None


def validate_field(name: str, field_spec: dict, value: Any) -> list[dict]:
    """Apply required/format/allowed-values/checksum rules for one field."""
    checks: list[dict] = []
    if value is None or str(value).strip() == "":
        if field_spec.get("required"):
            checks.append(
                {
                    "rule": f"required:{name}",
                    "passed": False,
                    "severity": "error",
                    "detail": f"Required field '{name}' is missing",
                }
            )
        return checks

    regex = field_spec.get("format_regex")
    if regex and not re.match(regex, str(value)):
        checks.append(
            {
                "rule": f"format:{name}",
                "passed": False,
                "severity": "error",
                "detail": f"Field '{name}'={value!r} does not match expected format {regex!r}",
            }
        )
    elif regex:
        checks.append(
            {
                "rule": f"format:{name}",
                "passed": True,
                "severity": "info",
                "detail": f"Field '{name}' matches expected format",
            }
        )

    allowed = field_spec.get("allowed_values")
    if allowed and str(value) not in allowed:
        checks.append(
            {
                "rule": f"allowed_values:{name}",
                "passed": False,
                "severity": "error",
                "detail": f"Field '{name}'={value!r} not in allowed values {allowed}",
            }
        )

    cs = checksum_check(name, field_spec.get("checksum"), value)
    if cs:
        checks.append(cs)
    return checks