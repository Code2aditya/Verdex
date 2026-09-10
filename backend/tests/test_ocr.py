"""Module 1 OCR tests — graceful degradation without PaddleOCR."""

from __future__ import annotations

from backend.modules.module1_ocr import run_ocr_pipeline


def test_empty_bytes_returns_unavailable() -> None:
    result = run_ocr_pipeline(b"")
    assert result["status"] == "unavailable"
    assert result["fields"] == {}


def test_none_returns_unavailable() -> None:
    result = run_ocr_pipeline(None, document_type="aadhaar")
    assert result["status"] == "unavailable"
    assert result["document_type"] == "aadhaar"


def test_pipeline_never_raises_on_garbage(sample_document_bytes) -> None:
    result = run_ocr_pipeline(sample_document_bytes, document_type="aadhaar")
    assert set(["status", "fields", "mrz_lines", "lines"]).issubset(result)
    assert result["status"] in ("ok", "unavailable", "degraded")