"""blockchain.py — Sepolia anchor status and history."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.db.models import BlockchainAnchor
from backend.db.session import get_db
from backend.security.jwt_auth import get_current_user
from backend.blockchain.sepolia_anchor import anchor_now, get_anchor_history

router = APIRouter(prefix="/api/blockchain", tags=["Blockchain"])


@router.get("/status", summary="Blockchain anchor status")
def status(_: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {
        "network": "sepolia",
        "chain_id": 11155111,
        "mode": "calldata-anchor (zero-value tx)",
        "live": bool(__import__("os").environ.get("INFURA_URL")),
    }


@router.get("/history", summary="Anchor history")
def history(
    user: dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    rows = get_anchor_history(db, limit=20)
    return {
        "anchors": [
            {
                "chain_root": r.chain_root_hash,
                "seq_from": r.seq_from,
                "seq_to": r.seq_to,
                "tx_hash": r.tx_hash,
                "block_number": r.block_number,
                "status": r.status,
                "confirmed_at": r.confirmed_at.isoformat() if r.confirmed_at else None,
            }
            for r in rows
        ]
    }


@router.post("/anchor", summary="Anchor the chain root (simulated unless configured)")
def anchor(
    user: dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    result = anchor_now(db)
    if result.success:
        return {
            "success": True,
            "tx_hash": result.tx_hash,
            "block_number": result.block_number,
            "chain_root": result.chain_root,
            "seq_from": result.seq_from,
            "seq_to": result.seq_to,
        }
    if result.error and "No unanchored" in result.error:
        return {"success": True, "note": result.error}
    return {
        "success": False,
        "error": result.error or "anchor failed",
        "hint": "Set INFURA_URL + WALLET_ADDRESS + PRIVATE_KEY for a live Sepolia transaction.",
    }