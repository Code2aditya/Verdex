"""
tamper.py — FastAPI route for Module 3 Tampering & Forgery Detection
VERIDEX | /api/v1/tamper

Endpoints
---------
  POST /api/v1/tamper/analyze
      Upload a document image → returns {tamper_score, heatmap_url, flags, ...}

  GET  /api/v1/tamper/status
      Health check — reports engine availability (torch / zxing optional).
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from backend.modules.module3_tampering.tamper_pipeline import _cnn_available

# The CNN classifier and Grad-CAM require torch; when absent the
# pipeline degrades to algorithmic signals (ELA/noise/QR). All imports
# are guarded so the API never boot-fails on a CI box without torch.
try:
    from backend.modules.module3_tampering.cnn_classifier import _FINE_TUNED, get_model
except Exception:  # pragma: no cover - optional dependency
    _FINE_TUNED = False

    def get_model():
        raise RuntimeError("torch is not installed — CNN/Grad-CAM unavailable.")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tamper", tags=["Module 3 — Tampering Detection"])


@router.post(
    "/analyze",
    summary="Analyze a document image for tampering/forgery",
)
async def analyze(
    file: UploadFile = File(..., description="Document image file (JPEG/PNG/BMP)"),
    doc_number: Optional[str] = Form(None, description="OCR-extracted document number"),
    dob: Optional[str] = Form(None),
    expiry: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    nationality: Optional[str] = Form(None),
) -> JSONResponse:
    """Analyze a document image for signs of tampering or forgery."""
    from backend.modules.module3_tampering import run_tamper_detection

    allowed = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
    ct = (file.content_type or "").lower()
    if ct and ct not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ct}'. Accepted: JPEG, PNG, BMP, TIFF, WEBP.",
        )

    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {exc}")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    ocr_fields: dict | None = None
    provided = {k: v for k, v in {
        "doc_number": doc_number, "dob": dob, "expiry": expiry,
        "name": name, "nationality": nationality,
    }.items() if v is not None}
    if provided:
        ocr_fields = provided

    try:
        result = run_tamper_detection(image_bytes, ocr_fields=ocr_fields)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Could not process image: {exc}")
    except Exception as exc:
        logger.exception("Unexpected error in tamper detection pipeline")
        raise HTTPException(status_code=500, detail=f"Internal error during tamper analysis: {exc}")

    api_dict = result.to_api_dict()
    logger.info("Tamper analysis — score=%.4f tier=%s flags=%s",
                api_dict["tamper_score"], api_dict["risk_tier"], api_dict["flags"])
    return JSONResponse(content=api_dict)


@router.get("/status", summary="Module 3 health check")
async def status_check():
    """Returns engine availability and fine-tune status."""
    model_ok, device = False, "cpu"
    message = "Algorithmic signals active (ELA + noise + QR)."
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        get_model()
        model_ok = True
        message = (
            "Fine-tuned checkpoint loaded — CNN calibrated for document forgery."
            if _FINE_TUNED
            else "Using ImageNet-pretrained weights — run scripts/train_cnn.py to calibrate."
        )
    except Exception as exc:
        message += f" CNN/Grad-CAM unavailable: {exc}"

    return {
        "status": "ok" if _cnn_available() else "degraded",
        "fine_tuned": _FINE_TUNED,
        "device": device,
        "torch": model_ok,
        "message": message,
    }