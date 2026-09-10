"""validation.py — standalone Module 2 validation endpoint."""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, Form, HTTPException

from backend.config.rule_loader import supported_document_types
from backend.modules.module2_validation import validate_document
from backend.security.jwt_auth import get_current_user

router = APIRouter(prefix="/api/v1/validation", tags=["Module 2 — Validation"])


@router.post("/analyze", summary="Validate extracted fields against the rule set")
async def analyze(
    document_type: str = Form(...),
    fields_json: str = Form(...),
    mrz_lines_json: Optional[str] = Form(None),
    _: Any = Depends(get_current_user),
) -> dict[str, Any]:
    if document_type not in supported_document_types():
        raise HTTPException(status_code=422, detail=f"Unsupported document type '{document_type}'.")
    try:
        fields = json.loads(fields_json)
        mrz_lines = json.loads(mrz_lines_json) if mrz_lines_json else None
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {exc}")
    if not isinstance(fields, dict):
        raise HTTPException(status_code=422, detail="fields_json must decode to an object.")
    report = validate_document(document_type, fields, mrz_lines=([str(x) for x in mrz_lines] if mrz_lines else None))
    return report.as_dict()


@router.get("/documents", summary="Supported document types")
def documents() -> dict[str, Any]:
    return {"document_types": supported_document_types()}