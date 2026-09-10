"""ocr.py — standalone Module 1 OCR endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.modules.module1_ocr import run_ocr_pipeline
from backend.security.jwt_auth import get_current_user

router = APIRouter(prefix="/api/v1/ocr", tags=["Module 1 — OCR"])


@router.post("/analyze", summary="OCR + field extraction for a document image")
async def analyze(
    file: UploadFile = File(...),
    document_type: str | None = Form(None),
    _: Any = Depends(get_current_user),
) -> dict[str, Any]:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    result = run_ocr_pipeline(image_bytes, document_type=document_type)
    hint = document_type or result.get("document_type")
    result["suggested_validation_doc_type"] = hint or "unknown"
    return result


@router.get("/status", summary="OCR engine availability")
def status() -> dict[str, Any]:
    from backend.modules.module1_ocr import paddle_available

    return {"paddleocr": "available" if paddle_available() else "not installed"}