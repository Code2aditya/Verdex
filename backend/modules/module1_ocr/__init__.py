"""module1_ocr — OCR & MRZ extraction pipeline.

Primary entry point
-------------------
    from backend.modules.module1_ocr import run_ocr_pipeline

    result = run_ocr_pipeline(image_bytes, document_type=None)
    # result["fields"]      → dict of extracted fields
    # result["mrz_lines"]   → list of MRZ candidate lines
    # result["lines"]       → raw OCR lines with confidence
    # result["document_type"] → detected or supplied type
"""

from __future__ import annotations

import logging
from typing import Any

from backend.modules.module1_ocr.doc_classifier import classify_document_type
from backend.modules.module1_ocr.field_extractor import (
    extract_fields,
    normalize_date,
)
from backend.modules.module1_ocr.mrz_parser import extract_mrz_lines
from backend.modules.module1_ocr.ocr_engine import (
    easy_available,
    lang_for_document_type,
    paddle_available,
    run_ocr,
)

logger = logging.getLogger(__name__)


def run_ocr_pipeline(image_bytes: bytes, document_type: str | None = None) -> dict[str, Any]:
    """Full OCR step: raw OCR -> classification -> fields -> MRZ lines."""
    if image_bytes is None or len(image_bytes) == 0:
        return {
            "status": "unavailable",
            "reason": "no image bytes supplied",
            "fields": {},
            "mrz_lines": [],
            "lines": [],
            "document_type": document_type,
            "confidence": 0.0,
        }

    first = run_ocr(image_bytes, document_type)
    lines = first.get("lines", [])

    detected = document_type
    if detected is None:
        detected = classify_document_type([ln.get("text", "") for ln in lines])

    # Re-run with the Devanagari model when OCR in English produced
    # nothing but the classifier or caller expects an Indic document.
    if (not lines) or (detected and lang_for_document_type(detected) == "hi"):
        second = run_ocr(image_bytes, detected)
        if second.get("line_count", 0) > first.get("line_count", 0):
            lines = second.get("lines", [])
            if detected is None:
                detected = classify_document_type([ln.get("text", "") for ln in lines])

    fields: dict[str, str] = {}
    if detected:
        try:
            fields = extract_fields(detected, [ln.get("text", "") for ln in lines])
        except ValueError as exc:
            logger.warning("Field extraction skipped: %s", exc)

    mrz_lines = extract_mrz_lines({"lines": lines})
    confidences = [ln.get("confidence", 0.0) for ln in lines if isinstance(ln.get("confidence"), (int, float))]
    confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    status = first.get("status", "unavailable")
    if status == "ok" and len(lines) == 0:
        status = "degraded"

    return {
        "status": status,
        "engine": "paddleocr",
        "document_type": detected,
        "line_count": len(lines),
        "lines": lines,
        "fields": fields,
        "mrz_lines": mrz_lines,
        "confidence": confidence,
        "devanagari_available": paddle_available(),
    }


__all__ = ["run_ocr_pipeline", "extract_fields", "normalize_date", "classify_document_type"]