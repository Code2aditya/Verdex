"""risk.py — risk engine presets, reason catalog and preview endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.risk_engine import compute_risk, evaluate_preset, list_presets
from backend.risk_engine.scorer import REASON_CATALOG
from backend.security.jwt_auth import get_current_user

router = APIRouter(prefix="/api", tags=["Risk Engine"])


class PreviewBody(BaseModel):
    signals: dict[str, float]
    flags: dict[str, bool] | None = None
    document_type: str = "passport"
    traveller_name: str = ""
    policy_escalation: bool = False


@router.get("/presets", summary="List demo risk presets (A–E)")
def presets(_: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {"presets": list_presets(), "count": 5}


@router.post("/risk/preview", summary="Compute risk from raw signals")
def preview(body: PreviewBody) -> dict[str, Any]:
    return compute_risk(
        body.signals,
        flags=body.flags,
        document_type=body.document_type,
        traveller_name=body.traveller_name,
        policy_escalation=body.policy_escalation,
    )


@router.get("/risk/reasons", summary="Risk reason-code catalog")
def reasons() -> dict[str, Any]:
    return {"catalog": REASON_CATALOG}


@router.get("/risk/presets/{preset_id}", summary="Evaluate one demo preset")
def preset_detail(preset_id: str) -> dict[str, Any]:
    try:
        return evaluate_preset(preset_id)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc