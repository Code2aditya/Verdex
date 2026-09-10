"""health.py — application & module health endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends

from backend.audit.chain import get_chain_root
from backend.config.rule_loader import supported_document_types
from backend.modules.module1_ocr import easy_available, paddle_available, tesseract_available
from backend.modules.module2_validation import watchlist_status
from backend.modules.module3_tampering.tamper_pipeline import _cnn_available
from backend.security.offline_queue import OfflineQueue
from backend.config.settings import get_settings

router = APIRouter(tags=["System"])


def _module_status() -> dict[str, str]:
    cnn = _cnn_available()
    settings = get_settings()
    if paddle_available():
        ocr_status = "active (paddleocr)"
    elif easy_available():
        quant = "quantized" if settings.easyocr_quantize else "full"
        ocr_status = f"active (easyocr {quant})"
    elif tesseract_available():
        ocr_status = "active (tesseract)"
    else:
        ocr_status = "degraded (no OCR engine installed)"
    face_status = (
        "active (insightface)"
        if settings.enable_insightface
        else "degraded (ENABLE_INSIGHTFACE=false — histogram fallback)"
    )
    return {
        "module1_ocr": ocr_status,
        "module2_validation": "active",
        "module3_tamper": "active",
        "module3_cnn": "active" if cnn else "degraded (torch not installed — algorithmic signals only)",
        "module4_face": face_status,
        "risk_engine": "active",
        "blockchain": "degraded (sepolia simulated unless INFURA_URL/WALLET_ADDRESS/PRIVATE_KEY set)",
        "audit_chain": "active",
    }


@router.get("/api/health", summary="Global health check")
def health() -> dict[str, Any]:
    queue = OfflineQueue(get_settings().audit_db_path)
    return {
        "ok": True,
        "service": "VERIDEX",
        "problem": "SIH26188",
        "time": datetime.now(timezone.utc).isoformat(),
        "modules": _module_status(),
        "supported_document_types": supported_document_types(),
        "chain_root": get_chain_root(),
        "offline_pending": queue.pending_count(),
        "watchlist": watchlist_status(),
    }


@router.get("/api/v1/health", summary="Health check (v1)")
def health_v1() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "VERIDEX",
        "version": "1.0.0",
        "modules": _module_status(),
    }