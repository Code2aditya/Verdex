"""Module 3 tamper tests — composite score bounds + API payload shape."""

from __future__ import annotations

from backend.modules.module3_tampering import run_tamper_detection
from backend.modules.module3_tampering.tamper_pipeline import _cnn_available


def test_cnn_guard_is_callable() -> None:
    assert isinstance(_cnn_available(), bool)


def test_blank_canvas_scores_in_bounds(sample_document_bytes) -> None:
    result = run_tamper_detection(sample_document_bytes)
    assert 0.0 <= result.tamper_score <= 1.0
    assert result.risk_tier in ("LOW", "MEDIUM", "HIGH")
    assert isinstance(result.flags, list)


def test_api_dict_shape(sample_document_bytes) -> None:
    payload = run_tamper_detection(sample_document_bytes, ocr_fields={"doc_number": "N1234567"}).to_api_dict()
    assert set(["tamper_score", "risk_tier", "heatmap_url", "flags", "signals"]).issubset(payload)
    assert set(["ela", "cnn", "noise", "qr"]).issubset(payload["signals"])
    assert "processing_ms" in payload


def test_numpy_source_supported() -> None:
    import cv2

    import numpy as np

    img = np.full((120, 160, 3), 200, dtype=np.uint8)
    result = run_tamper_detection(img)
    assert 0.0 <= result.tamper_score <= 1.0