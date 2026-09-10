"""Blockchain anchor tests — no-RPC degradation and history shape."""

from __future__ import annotations

import pytest

from backend.audit.chain import ScreeningPayload, append_record
from backend.blockchain.sepolia_anchor import anchor_now, get_anchor_history
from backend.db.session import SessionLocal


def test_history_is_empty_initially() -> None:
    db = SessionLocal()
    try:
        assert get_anchor_history(db) == []
    finally:
        db.close()


def test_anchor_now_without_rpc_reports_config_missing() -> None:
    db = SessionLocal()
    try:
        append_record(
            ScreeningPayload(
                checkpoint_id="CHK-001",
                officer_id="OFF-1",
                doc_type="passport",
                doc_number="N1234567",
                risk_score=12.0,
                decision="CLEAR",
            ),
            db=db,
        )
        result = anchor_now(db)
        assert result.success is False
        assert "INFURA_URL" in (result.error or "")
    finally:
        db.close()


def test_anchor_now_empty_chain_is_noop() -> None:
    result = anchor_now()
    assert result.success is True
    assert "No unanchored records" in (result.error or "")