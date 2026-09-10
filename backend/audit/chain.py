"""
chain.py — Tamper-evident hash-chain audit log
VERIDEX | audit | Anubhav

Every document screening event produces one AuditRecord. Records are
SHA-256 chained: each record's hash_chain field is:

    hash_chain[n] = SHA-256( record_hash[n] + hash_chain[n-1] )

Breaking the chain (editing any past record) is detectable by replaying
the chain from the genesis record and comparing computed hashes to stored
hashes. This is done offline by scripts/verify_chain.py.

Optionally, the chain root is periodically anchored to the Ethereum Sepolia
testnet by sepolia_anchor.py, providing a publicly verifiable timestamp.

Public API
----------
    append_record(db, payload)   -> AuditRecord
    verify_chain(db)             -> ChainVerifyResult
    get_chain_root(db)           -> str   (latest hash_chain value)
    get_records(db, limit, offset) -> list[AuditRecord]
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from backend.config.settings import get_settings
from backend.db.models import AuditRecord
from backend.db.session import SessionLocal, init_db

logger  = logging.getLogger(__name__)
_cfg    = get_settings()

# SHA-256 of the string "VERIDEX_GENESIS" — the anchor for the first record
GENESIS_HASH = hashlib.sha256(b"VERIDEX_GENESIS").hexdigest()


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ScreeningPayload:
    """
    Data produced by the risk engine and passed to append_record().
    All fields are optional except checkpoint_id — missing fields are
    stored as None and excluded from the hash input.
    """
    checkpoint_id:  str
    officer_id:     str  | None = None
    doc_type:       str  | None = None
    doc_number:     str  | None = None
    risk_score:     float| None = None
    decision:       str  | None = None    # "CLEAR" | "REVIEW" | "ALERT"
    extra:          dict | None = None    # arbitrary extra data (not stored in DB)


@dataclass
class ChainVerifyResult:
    """Result of replaying and verifying the full audit chain."""
    valid:          bool
    total_records:  int
    first_broken_seq: int | None = None
    error_message:  str | None   = None
    checked_at:     datetime.datetime = field(
        default_factory=datetime.datetime.utcnow
    )


# ── Public API ────────────────────────────────────────────────────────────────

def append_record(
    payload: ScreeningPayload,
    db: Session | None = None,
) -> AuditRecord:
    """
    Append a new screening event to the audit chain.

    Parameters
    ----------
    payload : ScreeningPayload
        The screening event data from the risk engine.
    db : Session | None
        Optional SQLAlchemy session. If None, a new session is created
        and closed automatically (useful for standalone scripts).

    Returns
    -------
    AuditRecord — the newly created, committed record.
    """
    _own_session = db is None
    if _own_session:
        init_db()
        db = SessionLocal()

    try:
        # ── 1. Determine sequence number ──────────────────────────────────────
        last = _get_last_record(db)
        seq       = (last.seq + 1) if last else 1
        prev_hash = last.hash_chain if last else GENESIS_HASH

        # ── 2. Build canonical record payload for hashing ──────────────────────
        canonical = _build_canonical(seq, payload)
        record_hash = _sha256(canonical)

        # ── 3. Compute chain link ─────────────────────────────────────────────
        hash_chain = _sha256(record_hash + prev_hash)

        # ── 4. Persist ────────────────────────────────────────────────────────
        record = AuditRecord(
            seq           = seq,
            checkpoint_id = payload.checkpoint_id,
            officer_id    = payload.officer_id,
            doc_type      = payload.doc_type,
            doc_number    = payload.doc_number,
            risk_score    = payload.risk_score,
            decision      = payload.decision,
            record_hash   = record_hash,
            prev_hash     = prev_hash,
            hash_chain    = hash_chain,
            anchored      = False,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        logger.info(
            "AuditRecord appended: seq=%d decision=%s chain=%s...",
            record.seq, record.decision, record.hash_chain[:12],
        )
        return record

    except Exception:
        db.rollback()
        raise
    finally:
        if _own_session:
            db.close()


def verify_chain(db: Session | None = None) -> ChainVerifyResult:
    """
    Replay the entire audit chain from the genesis record and verify that
    every stored hash_chain value matches the recomputed value.

    A mismatch at sequence N means record N (or any record before it) was
    tampered with after insertion.

    Returns a ChainVerifyResult with valid=True if the chain is intact.
    """
    _own_session = db is None
    if _own_session:
        db = SessionLocal()

    try:
        records = (
            db.query(AuditRecord)
            .order_by(AuditRecord.seq.asc())
            .all()
        )

        if not records:
            return ChainVerifyResult(valid=True, total_records=0)

        prev_hash = GENESIS_HASH

        for record in records:
            canonical   = _build_canonical(record.seq, _record_to_payload(record))
            record_hash = _sha256(canonical)
            hash_chain  = _sha256(record_hash + prev_hash)

            if hash_chain != record.hash_chain:
                logger.error(
                    "Chain broken at seq=%d  stored=%s  computed=%s",
                    record.seq, record.hash_chain, hash_chain,
                )
                return ChainVerifyResult(
                    valid             = False,
                    total_records     = len(records),
                    first_broken_seq  = record.seq,
                    error_message     = (
                        f"Hash mismatch at seq={record.seq}. "
                        "Record may have been tampered with."
                    ),
                )
            prev_hash = hash_chain

        logger.info("Chain verified: %d records intact.", len(records))
        return ChainVerifyResult(valid=True, total_records=len(records))

    finally:
        if _own_session:
            db.close()


def get_chain_root(db: Session | None = None) -> str:
    """
    Return the hash_chain value of the most recent record.
    This is the value anchored to Sepolia by sepolia_anchor.py.
    Returns GENESIS_HASH if the chain is empty.
    """
    _own_session = db is None
    if _own_session:
        db = SessionLocal()
    try:
        last = _get_last_record(db)
        return last.hash_chain if last else GENESIS_HASH
    finally:
        if _own_session:
            db.close()


def get_records(
    db: Session,
    limit:  int = 50,
    offset: int = 0,
    unanchored_only: bool = False,
) -> list[AuditRecord]:
    """Fetch a page of audit records ordered by sequence number (newest first)."""
    q = db.query(AuditRecord)
    if unanchored_only:
        q = q.filter(AuditRecord.anchored == False)
    return (
        q.order_by(AuditRecord.seq.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def mark_anchored(
    db: Session,
    seq_to: int,
    tx_hash: str,
    block_number: int,
) -> int:
    """
    Mark all records up to seq_to as anchored on-chain.
    Called by sepolia_anchor.py after a confirmed transaction.
    Returns the number of records updated.
    """
    updated = (
        db.query(AuditRecord)
        .filter(AuditRecord.seq <= seq_to, AuditRecord.anchored == False)
        .all()
    )
    for r in updated:
        r.anchored       = True
        r.anchor_tx_hash = tx_hash
        r.anchor_block   = block_number
    db.commit()
    logger.info("Marked %d records as anchored (tx=%s)", len(updated), tx_hash)
    return len(updated)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_last_record(db: Session) -> AuditRecord | None:
    return (
        db.query(AuditRecord)
        .order_by(AuditRecord.seq.desc())
        .first()
    )


def _build_canonical(seq: int, payload: ScreeningPayload) -> str:
    """
    Produce a stable, deterministic string from (seq, payload) for hashing.
    Uses sorted-key JSON serialisation to avoid key-order sensitivity.
    Excludes the 'extra' field — it is informational only and not hashed.
    """
    obj = {
        "seq":           seq,
        "checkpoint_id": payload.checkpoint_id,
        "officer_id":    payload.officer_id,
        "doc_type":      payload.doc_type,
        "doc_number":    payload.doc_number,
        "risk_score":    float(payload.risk_score) if payload.risk_score is not None else None,
        "decision":      payload.decision,
    }
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _record_to_payload(record: AuditRecord) -> ScreeningPayload:
    """Reconstruct a ScreeningPayload from a stored AuditRecord for re-hashing."""
    return ScreeningPayload(
        checkpoint_id = record.checkpoint_id,
        officer_id    = record.officer_id,
        doc_type      = record.doc_type,
        doc_number    = record.doc_number,
        risk_score    = record.risk_score,
        decision      = record.decision,
    )


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
