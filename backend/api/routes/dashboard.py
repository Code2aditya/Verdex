"""dashboard.py — officer dashboard summary endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.audit.chain import get_chain_root, get_records, verify_chain
from backend.config.settings import get_settings
from backend.db.session import get_db
from backend.security.jwt_auth import get_current_user
from backend.security.offline_queue import OfflineQueue

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])

_TIERS = ["CLEAR", "REVIEW", "ALERT"]


@router.get("/summary", summary="Dashboard aggregate summary")
def summary(
    _: dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    records = get_records(db, limit=200)
    by_decision = {t: 0 for t in _TIERS}
    by_type: dict[str, int] = {}
    for r in records:
        by_decision[r.decision] = by_decision.get(r.decision, 0) + 1
        by_type[r.doc_type] = by_type.get(r.doc_type, 0) + 1

    verification = verify_chain(db)
    queue = OfflineQueue(get_settings().audit_db_path)

    return {
        "total_screenings": verification.total_records,
        "by_decision": by_decision,
        "by_document_type": by_type,
        "chain_intact": verification.valid,
        "chain_root": get_chain_root(db),
        "offline_pending": queue.pending_count(),
        "recent": [_record_slim(r) for r in get_records(db, limit=10)],
    }


def _record_slim(r: Any) -> dict[str, Any]:
    return {
        "seq": r.seq,
        "officer_id": r.officer_id,
        "doc_type": r.doc_type,
        "doc_number": r.doc_number,
        "risk_score": r.risk_score,
        "decision": r.decision,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }