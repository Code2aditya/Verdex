"""face.py — standalone Module 4 face analysis endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.modules.module4_face import run_face_analysis
from backend.security.jwt_auth import get_current_user

router = APIRouter(prefix="/api/v1/face", tags=["Module 4 — Face"])


@router.post("/analyze", summary="Analyze a document portrait against a live capture")
async def analyze(
    file: UploadFile = File(...),
    live_photo: UploadFile | None = File(None),
    _: Any = Depends(get_current_user),
) -> dict[str, Any]:
    document_bytes = await file.read()
    if not document_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    live_bytes = None
    if live_photo is not None:
        live_bytes = await live_photo.read() or None
    return run_face_analysis(document_bytes, live_bytes)


@router.get("/status", summary="Module 4 availability")
def status() -> dict[str, Any]:
    import logging

    logger = logging.getLogger("face.status")

    def probe(module: str) -> str:
        try:
            __import__(module)
            return "available"
        except Exception:
            return "not installed"

    return {
        "insightface": probe("insightface"),
        "mediapipe": probe("mediapipe"),
        "deepface": probe("deepface"),
        "degradation": "matching falls back to HSV histogram descriptor; morphing skipped.",
    }