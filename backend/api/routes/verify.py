"""verify.py — unified document verification endpoints.

Both the Flutter officer app and the web dashboard call these:

  POST /api/verify                  (Flutter AppConstants.verifyEndpoint)
  POST /api/v1/verify-document      (Flutter FastApiUploadService.defaultEndpoint)

Multipart form:
  file:            image bytes (JPEG/PNG...)
  document_type:   optional hint e.g. "aadhaar", "pan_card", "passport"...
  officer_id:      optional
  terminal_node:   optional
  sha256_checksum: optional (client-computed digest)
  live_photo:      OPTIONAL selfie bytes — enables Module 4 facial verify.

Response matches the Flutter Verification contract:
  risk_score, risk_tier, ocr, validation, tamper, face, liveness, blockchain
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.audit.chain import ScreeningPayload, append_record
from backend.config.settings import get_settings
from backend.db.session import get_db
from backend.pipeline.verify import run_verification
from backend.security.offline_queue import OfflineQueue
from backend.security.jwt_auth import decode_token

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Verify"])

_bearer = HTTPBearer(auto_error=False)

_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}


def _optional_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any] | None:
    if creds is None or creds.scheme.lower() != "bearer":
        return None
    try:
        return decode_token(creds.credentials)
    except HTTPException:
        return None


def _verify_body(
    file: UploadFile = File(..., description="Document image"),
    document_type: str | None = Form(None, description="Optional doc-type hint"),
    officer_id: str | None = Form(None),
    terminal_node: str | None = Form(None),
    sha256_checksum: str | None = Form(None),
    client_timestamp: str | None = Form(None),
    live_photo: UploadFile | None = File(None, description="Optional live selfie"),
):
    return {**locals()}


async def _run_verify(args: dict[str, Any], user: dict[str, Any] | None) -> dict[str, Any]:
    file: UploadFile = args["file"]
    ct = (file.content_type or "").lower()
    if ct and ct not in _IMAGE_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{ct}'.")

    document_bytes = await file.read()
    if not document_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    live_bytes = None
    if args.get("live_photo") is not None:
        live_bytes = await args["live_photo"].read() or None

    hint = args.get("document_type")
    if not hint or hint.lower() in ("auto", "unknown", "icao-9303"):
        hint = None

    officer = user.get("officer_id") if user else (args.get("officer_id") or "OFFICER-API")
    checkpoint = user.get("checkpoint_id") if user and user.get("checkpoint_id") else "CHK-001"

    contract = run_verification(
        document_bytes,
        document_type=hint,
        live_bytes=live_bytes,
        officer_id=officer,
        checkpoint_id=checkpoint,
    )

    audit_info = contract["audit"]
    try:
        record = append_record(
            ScreeningPayload(
                checkpoint_id=checkpoint,
                officer_id=officer,
                doc_type=contract["document_type"],
                doc_number=audit_info["doc_number"],
                risk_score=contract["risk_score"],
                decision=audit_info["decision"],
            )
        )
        contract["audit"]["seq"] = record.seq
        contract["audit"]["hash_chain"] = record.hash_chain
        contract["queued_offline"] = None
    except Exception as exc:
        logger.warning("Audit append failed — queuing offline: %s", exc)
        queue = OfflineQueue(get_settings().audit_db_path)
        queued = queue.enqueue(officer, "verify", {
            "checkpoint": checkpoint,
            "doc_type": contract["document_type"],
            "doc_number": audit_info["doc_number"],
            "risk_score": contract["risk_score"],
            "decision": audit_info["decision"],
        })
        contract["queued_offline"] = queued
        contract["audit"]["seq"] = None

    del contract["audit"]
    return contract


@router.post(
    "/api/verify",
    summary="Verify a document image",
    description="Full OCR → validation → tampering → face → risk pipeline.",
)
async def verify(
    args: dict[str, Any] = Depends(_verify_body),
    user: dict[str, Any] | None = Depends(_optional_user),
    db=Depends(get_db),  # noqa: ARG001 - keeps session lifecycle consistent
) -> dict[str, Any]:
    return await _run_verify(args, user)


@router.post(
    "/api/v1/verify-document",
    summary="Verify a document image (v1 multipart contract)",
)
async def verify_document(
    args: dict[str, Any] = Depends(_verify_body),
    user: dict[str, Any] | None = Depends(_optional_user),
    db=Depends(get_db),  # noqa: ARG001
) -> dict[str, Any]:
    return await _run_verify(args, user)