"""audit.py — tamper-evident audit chain endpoints."""

from __future__ import annotations

import datetime
import hashlib
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.audit.chain import (
    ScreeningPayload,
    append_record,
    get_chain_root,
    get_records,
    verify_chain,
)
from backend.db.models import AuditRecord, BlockchainAnchor
from backend.db.session import get_db
from backend.security.jwt_auth import get_current_user, require_roles

router = APIRouter(prefix="/api/audit", tags=["Audit Chain"])


@router.get("/verify", summary="Verify the hash-chain integrity")
def audit_verify(
    _: dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    report = verify_chain(db)
    recent = get_records(db, limit=12)
    return {
        "intact": report.valid,
        "records": report.total_records,
        "first_broken_seq": report.first_broken_seq,
        "error_message": report.error_message,
        "checked_at": report.checked_at.isoformat(),
        "chain_root": get_chain_root(db),
        "recent": [_record_to_dict(r) for r in recent],
    }


@router.get("/log", summary="Audit log page")
def audit_log(
    limit: int = 50,
    offset: int = 0,
    _: dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    records = get_records(db, limit=limit, offset=offset)
    return {
        "records": [_record_to_dict(r) for r in records],
        "total": verify_chain(db).total_records,
    }


@router.post("/anchor", summary="Anchor the chain (supervisor; simulated if no RPC)")
def audit_anchor(
    user: dict[str, Any] = Depends(require_roles("supervisor")),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    try:
        from backend.blockchain.sepolia_anchor import anchor_now

        result = anchor_now(db)
        if result.success:
            append_record(
                ScreeningPayload(
                    checkpoint_id="CHK-001",
                    officer_id=user["officer_id"],
                    doc_type="anchor.root",
                    doc_number=None,
                    risk_score=None,
                    decision="ANCHOR",
                    extra={"tx_hash": result.tx_hash, "root": result.chain_root},
                ),
                db=db,
            )
            return {
                "success": True,
                "tx_hash": result.tx_hash,
                "chain_root": result.chain_root,
                "seq_from": result.seq_from,
                "seq_to": result.seq_to,
                "mode": "live",
                "error": result.error,
            }
    except Exception:
        pass

    # Offline simulation — no RPC configured. Deterministic, inspectable.
    root = get_chain_root(db)
    unanchored = get_records(db, limit=1000, unanchored_only=True)
    if not unanchored:
        raise HTTPException(status_code=400, detail="Nothing to anchor — ledger empty.")
    seq_from = unanchored[-1].seq
    seq_to = unanchored[0].seq
    tx_hash = "0x" + hashlib.sha256(f"sepolia-sim:{root}".encode()).hexdigest()

    db.query(AuditRecord).filter(
        AuditRecord.seq <= seq_to, AuditRecord.anchored == False  # noqa: E712
    ).update(
        {
            "anchored": True,
            "anchor_tx_hash": tx_hash,
            "anchor_block": 0,
        },
        synchronize_session=False,
    )
    db.add(BlockchainAnchor(
        chain_root_hash=root,
        seq_from=seq_from,
        seq_to=seq_to,
        tx_hash=tx_hash,
        block_number=0,
        status="SIMULATED",
        confirmed_at=datetime.datetime.utcnow(),
    ))
    db.commit()

    append_record(
        ScreeningPayload(
            checkpoint_id="CHK-001",
            officer_id=user["officer_id"],
            doc_type="anchor.root",
            doc_number=None,
            risk_score=None,
            decision="ANCHOR",
            extra={"tx_hash": tx_hash, "root": root, "mode": "simulated"},
        ),
        db=db,
    )
    return {
        "success": True,
        "tx_hash": tx_hash,
        "chain_root": root,
        "seq_from": seq_from,
        "seq_to": seq_to,
        "mode": "simulated",
        "note": "Set INFURA_URL + WALLET_ADDRESS + PRIVATE_KEY for a live Sepolia transaction.",
    }


@router.post("/tamper-demo", summary="Mutate a past ledger row (supervisor demo)")
def audit_tamper_demo(
    user: dict[str, Any] = Depends(require_roles("supervisor")),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    target = (
        db.query(AuditRecord)
        .order_by(AuditRecord.seq.asc())
        .first()
    )
    if target is None:
        raise HTTPException(status_code=400, detail="Ledger is empty — screen a document first.")

    original_score = target.risk_score
    target.risk_score = 0.0
    target.decision = "TAMPERED_DEMO"
    db.commit()

    append_record(
        ScreeningPayload(
            checkpoint_id="CHK-001",
            officer_id=user["officer_id"],
            doc_type="demo.tamper",
            doc_number=None,
            risk_score=None,
            decision="DEMO",
            extra={"note": "Supervisor triggered Demo E mutation", "seq": target.seq},
        ),
        db=db,
    )

    verification = verify_chain(db)
    return {
        "tampered_seq": target.seq,
        "mutation": (
            f"Forced stored risk_score {original_score} -> 0 and decision -> TAMPERED_DEMO "
            "without recomputing record_hash."
        ),
        "verification": {
            "intact": verification.valid,
            "records": verification.total_records,
            "first_broken_seq": verification.first_broken_seq,
            "error_message": verification.error_message,
        },
    }


def _record_to_dict(r: AuditRecord) -> dict[str, Any]:
    return {
        "seq": r.seq,
        "checkpoint_id": r.checkpoint_id,
        "officer_id": r.officer_id,
        "doc_type": r.doc_type,
        "doc_number": r.doc_number,
        "risk_score": r.risk_score,
        "decision": r.decision,
        "record_hash": r.record_hash,
        "prev_hash": r.prev_hash,
        "hash_chain": r.hash_chain,
        "anchored": r.anchored,
        "anchor_tx_hash": r.anchor_tx_hash,
        "anchor_block": r.anchor_block,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }