"""Module 2 — Document Validation Engine (deterministic rules).

Checks extracted field values against the versioned YAML rule config
for a document type:
  - required fields present
  - field format conformance (regex / allowed values)
  - Indian checksums (Aadhaar Verhoeff, PAN check char)
  - date logic (e.g. DOB < issue < expiry) and expiry status
  - MRZ checksum + visual-vs-MRZ cross-field consistency

Ported from VerdexZip/app/validation/rules_engine.py and extended with
checksum handling. Rule definitions live in Backend/rules/*.yaml.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from datetime import date, datetime
from typing import Any

from backend.config.rule_loader import load_rule
from backend.modules.module2_validation.field_validators import validate_field
from backend.modules.module2_validation.mrz_checksum import (
    mrz_date_to_iso,
    parse_and_validate_td3,
)

_OPERATORS = {
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
}
_OP_PATTERN = re.compile(r"(<=|>=|==|<|>)")


@dataclass
class CheckResult:
    rule: str
    passed: bool
    severity: str  # "error" | "warning" | "info"
    detail: str


@dataclass
class ValidationReport:
    document_type: str
    result: str  # "PASS" | "FAIL"
    checks: list = dc_field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "document_type": self.document_type,
            "result": self.result,
            "checks": [c.__dict__ for c in self.checks],
        }


def _parse_date(value: str, fmt: str = "%Y-%m-%d") -> date | None:
    try:
        return datetime.strptime(value, fmt).date()
    except (ValueError, TypeError):
        return None


def _eval_date_expr(expr: str, fields: dict[str, Any]) -> CheckResult:
    match = _OP_PATTERN.search(expr)
    if not match:
        return CheckResult(
            rule=expr, passed=False, severity="warning",
            detail=f"Could not parse date-logic expression: {expr!r}",
        )

    op_str = match.group(1)
    parts = [s.strip() for s in _OP_PATTERN.split(expr, maxsplit=1)]
    left_name, right_name = parts[0], parts[2]

    def resolve(name: str) -> date | None:
        if name == "today":
            return date.today()
        raw = fields.get(name)
        return _parse_date(raw) if raw else None

    left_val = resolve(left_name)
    right_val = resolve(right_name)

    if left_val is None or right_val is None:
        missing = left_name if left_val is None else right_name
        return CheckResult(
            rule=expr, passed=False, severity="warning",
            detail=f"Cannot evaluate '{expr}' — field '{missing}' missing or unparsable",
        )

    passed = _OPERATORS[op_str](left_val, right_val)
    return CheckResult(
        rule=expr, passed=passed, severity="error" if not passed else "info",
        detail=f"{left_name}={left_val} {op_str} {right_name}={right_val} -> {passed}",
    )


def _eval_cross_field_expr(expr: str, namespaces: dict[str, dict[str, Any]]) -> CheckResult | None:
    if "==" not in expr:
        return CheckResult(
            rule=expr, passed=False, severity="warning",
            detail=f"Unsupported cross-field expression: {expr!r}",
        )

    left_ref, right_ref = (s.strip() for s in expr.split("==", 1))

    def resolve(ref: str):
        ns, _, key = ref.partition(".")
        if ns not in namespaces:
            return "__SKIP__"
        return namespaces[ns].get(key)

    left_val = resolve(left_ref)
    right_val = resolve(right_ref)

    if left_val == "__SKIP__" or right_val == "__SKIP__":
        return None
    if left_val is None or right_val is None:
        return CheckResult(
            rule=expr, passed=False, severity="warning",
            detail=f"Cannot evaluate '{expr}' — one side is missing",
        )

    passed = str(left_val).strip().upper() == str(right_val).strip().upper()
    return CheckResult(
        rule=expr, passed=passed, severity="error" if not passed else "info",
        detail=f"{left_ref}={left_val!r} vs {right_ref}={right_val!r} -> {'match' if passed else 'MISMATCH'}",
    )


def validate_document(
    document_type: str,
    fields: dict[str, Any],
    mrz_lines: list[str] | None = None,
) -> ValidationReport:
    """Run the full Module 2 rule set for one document."""
    config = load_rule(document_type)
    checks: list[CheckResult] = []
    mrz_fields: dict[str, Any] = {}
    fields = dict(fields)

    if mrz_lines and "mrz_raw" not in fields:
        fields["mrz_raw"] = "\n".join(mrz_lines)

    for field_spec in config.get("fields", []):
        name = field_spec["name"]
        checks.extend(
            CheckResult(**c) for c in validate_field(name, field_spec, fields.get(name))
        )

    for expr in config.get("date_logic", []):
        checks.append(_eval_date_expr(expr, fields))

    expiry_field_name = "date_of_expiry" if "date_of_expiry" in fields else (
        "validity_end" if "validity_end" in fields else None
    )
    if expiry_field_name:
        expiry_date = _parse_date(fields.get(expiry_field_name))
        if expiry_date:
            expired = expiry_date < date.today()
            checks.append(CheckResult(
                rule="expiry_status", passed=not expired,
                severity="error" if expired else "info",
                detail=f"Document {'EXPIRED' if expired else 'valid'} — expiry {expiry_date}",
            ))

    if config.get("mrz", {}).get("present") and mrz_lines and len(mrz_lines) == 2:
        mrz_result = parse_and_validate_td3(mrz_lines[0], mrz_lines[1])
        checks.append(CheckResult(
            rule="mrz_checksum", passed=mrz_result.valid,
            severity="error" if not mrz_result.valid else "info",
            detail="; ".join(mrz_result.errors) if mrz_result.errors else "All MRZ checksums valid",
        ))
        mrz_fields = {
            "passport_number": mrz_result.document_number,
            "date_of_birth": mrz_date_to_iso(mrz_result.date_of_birth, field="dob"),
            "date_of_expiry": mrz_date_to_iso(mrz_result.date_of_expiry, field="expiry"),
        }

    namespaces = {"mrz": mrz_fields, "visual": fields}
    for expr in config.get("cross_field_checks", []):
        result = _eval_cross_field_expr(expr, namespaces)
        if result is not None:
            checks.append(result)

    has_error = any(c.severity == "error" and not c.passed for c in checks)
    return ValidationReport(
        document_type=document_type,
        result="FAIL" if has_error else "PASS",
        checks=checks,
    )