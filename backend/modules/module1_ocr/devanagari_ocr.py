"""Module 1 — Devanagari / Indic script OCR support.

Indian and Nepali identity documents (voter ID, DL, citizenship cert,
Aadhaar) mix Devanagari and Latin script. PaddleOCR's built-in
"hi"/"ne" recognition models are selected automatically via the
document type's `script:` rule field. This module reports whether the
Devanagari-capable engine is available in the current environment.
"""

from __future__ import annotations

from backend.modules.module1_ocr.ocr_engine import paddle_available


def devanagari_supported() -> bool:
    """True when PaddleOCR (the Devanagari-capable engine) is installed."""
    return paddle_available()


def get_script_for_document_type(document_type: str | None) -> str:
    if not document_type:
        return "latin"
    try:
        from backend.config.rule_loader import load_rule

        return load_rule(document_type).get("script", "latin")
    except Exception:
        return "latin"