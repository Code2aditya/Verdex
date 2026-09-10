"""screen.py — officer screening console endpoint (ported from module4).

POST /api/screen  with a demo preset (A–E) or raw signals. Result is
logged to the tamper-evident audit chain (or the encrypted offline queue
when `offline=true`).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.audit.chain import ScreeningPayload, append_record
from backend.config.settings import get_settings
from backend.risk_engine import compute_risk, evaluate_preset, list_presets
from backend.security.jwt_auth import get_current_user
from backend.security.offline_queue import OfflineQueue

router = APIRouter(prefix="/api", tags=["Officer Console"])

_queue: OfflineQueue | None = None


def _offline_queue() -> OfflineQueue:
    global _queue
    if _queue is None:
        _queue = OfflineQueue(get_settings().audit_db_path)
    return _queue


class ScreenBody(BaseModel):
    preset: str | None = Field(default=None, description="Demo preset A–E")
    signals: dict[str, float] | None = None
    flags: dict[str, bool] | None = None
    document_type: str = "passport"
    traveller_name: str = ""
    doc_number: str = ""
    offline: bool = False
    officer_decision: str = "pending"
    policy_escalation: bool = True


@router.get("/screen/presets", summary="List officer console presets")
def screen_presets(_: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {"presets": list_presets(), "count": 5}


@router.post("/screen", summary="Run an officer screening")
def screen(body: ScreenBody, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    if body.preset:
        try:
            result = evaluate_preset(body.preset)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        doc_number = result["preset"]["doc_number"]
        event_type = f"screen.preset.{body.preset.upper()}"
    else:
        if not body.signals:
            raise HTTPException(status_code=400, detail="Provide preset A–E or a signals object.")
        result = compute_risk(
            body.signals,
            flags=body.flags,
            document_type=body.document_type,
            traveller_name=body.traveller_name,
            policy_escalation=body.policy_escalation,
        )
        result["preset"] = None
        doc_number = body.doc_number
        event_type = "screen.custom"

    audit_payload = {
        "score": result["score"],
        "level": result["level"],
        "document_type": result["document_type"],
        "traveller_name": result["traveller_name"],
        "doc_number": doc_number,
        "reasons": [r["code"] for r in result["reasons"]],
        "post": user.get("post"),
    }

    if body.offline:
        queued = _offline_queue().enqueue(user["officer_id"], user.get("post", "unknown"), audit_payload)
        result["queued_offline"] = queued
        result["audit"] = None
        return result

    try:
        record = append_record(
            ScreeningPayload(
                checkpoint_id="CHK-001",
                officer_id=user["officer_id"],
                doc_type=result["document_type"],
                doc_number=doc_number,
                risk_score=result["score"],
                decision=_decision(result["level"]),
                extra=audit_payload,
            )
        )
        result["audit"] = {
            "seq": record.seq,
            "event_type": event_type,
            "hash_chain": record.hash_chain,
            "officer_id": user["officer_id"],
        }
    except Exception as exc:
        result["audit"] = None
        result["audit_error"] = str(exc)
    result["queued_offline"] = None

    if result.get("preset") and result["preset"].get("triggers_tamper_demo"):
        result["tamper_hint"] = (
            "Demo E: call POST /api/audit/tamper-demo (supervisor) to mutate the ledger "
            "and watch /api/audit/verify fail."
        )
    return result


def _decision(level: str) -> str:
    if level in ("CRITICAL", "HIGH"):
        return "ALERT"
    if level == "MEDIUM":
        return "REVIEW"
    return "CLEAR"