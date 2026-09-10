"""Audit chain tests — append, replay, the int/float canonical regression, tamper detection."""

from __future__ import annotations

from backend.audit.chain import (
    GENESIS_HASH,
    ScreeningPayload,
    append_record,
    get_chain_root,
    verify_chain,
)
from backend.db.session import SessionLocal


def _append(db, seq_hint: str, score: int | float | None, decision: str):
    return append_record(
        ScreeningPayload(
            checkpoint_id="CHK-001",
            officer_id="OFF-1",
            doc_type="passport",
            doc_number=seq_hint,
            risk_score=score,
            decision=decision,
        ),
        db=db,
    )


def test_empty_chain_is_valid() -> None:
    assert verify_chain().valid is True
    assert get_chain_root() == GENESIS_HASH


def test_chain_int_float_regression() -> None:
    """Records written with int risk_score must replay cleanly."""
    db = SessionLocal()
    try:
        first = _append(db, "R1", 62, "ALERT")
        _append(db, "R2", 12, "CLEAR")
        _append(db, "R3", 88, "ALERT")

        assert first.seq == 1
        result = verify_chain(db)
        assert result.valid is True
        assert result.total_records == 3
        root = get_chain_root(db)
        assert root != GENESIS_HASH
        assert len(root) == 64
    finally:
        db.close()


def test_tampering_middle_record_breaks_chain() -> None:
    db = SessionLocal()
    try:
        _append(db, "R1", 20, "CLEAR")
        _append(db, "R2", 55, "REVIEW")
        _append(db, "R3", 90, "ALERT")
        assert verify_chain(db).valid is True

        from backend.db.models import AuditRecord

        row = db.query(AuditRecord).filter(AuditRecord.seq == 2).one()
        row.decision = "HACKED"
        row.risk_score = 1.0
        db.commit()

        result = verify_chain(db)
        assert result.valid is False
        assert result.first_broken_seq == 2
    finally:
        db.close()


def test_float_scores_roundtrip_exact() -> None:
    db = SessionLocal()
    try:
        _append(db, "R1", 17.5, "REVIEW")
        _append(db, "R2", 0.0, "CLEAR")
        assert verify_chain(db).valid is True
    finally:
        db.close()