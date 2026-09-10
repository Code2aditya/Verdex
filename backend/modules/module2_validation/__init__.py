"""module2_validation — Document Validation Engine.

    from backend.modules.module2_validation import validate_document, watchlist_check

    report = validate_document("aadhaar", fields, mrz_lines=None)
    watch  = watchlist_check("passport", doc_number="P1234567", name="ANITA SHARMA")
"""

from __future__ import annotations

from backend.modules.module2_validation.rules_engine import (
    CheckResult,
    ValidationReport,
    validate_document,
)
from backend.modules.module2_validation.watchlist_lookup import (
    watchlist_check,
    watchlist_status,
)

__all__ = [
    "validate_document",
    "ValidationReport",
    "CheckResult",
    "watchlist_check",
    "watchlist_status",
]