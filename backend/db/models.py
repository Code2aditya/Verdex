"""
models.py — SQLAlchemy ORM models for VERIDEX
Covers the audit chain table used by chain.py (Anubhav)
and stubs for other modules to extend.
"""

from __future__ import annotations

import datetime
from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Float,
    Integer, String, Text, func,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# ── Audit Chain (Anubhav) ──────────────────────────────────────────────────────

class AuditRecord(Base):
    """
    One entry per document screening event.

    The hash_chain column stores SHA-256(record_data + prev_hash),
    forming a tamper-evident linked list anchored to the genesis record.
    Any modification of a past row will break the chain, which is
    verifiable offline via scripts/verify_chain.py.
    """
    __tablename__ = "audit_records"

    id              = Column(Integer,    primary_key=True, autoincrement=True)
    seq             = Column(BigInteger, nullable=False, unique=True)  # monotonic chain index

    # Screening event metadata
    checkpoint_id   = Column(String(64),  nullable=False)
    officer_id      = Column(String(128), nullable=True)
    doc_type        = Column(String(64),  nullable=True)
    doc_number      = Column(String(128), nullable=True)
    risk_score      = Column(Float,       nullable=True)
    decision        = Column(String(32),  nullable=True)   # CLEAR / REVIEW / ALERT

    # Chain fields
    record_hash     = Column(String(64),  nullable=False)  # SHA-256 of this record's payload
    prev_hash       = Column(String(64),  nullable=False)  # SHA-256 of the previous record
    hash_chain      = Column(String(64),  nullable=False)  # SHA-256(record_hash + prev_hash)

    # Blockchain anchor (populated when anchored to Sepolia)
    anchored        = Column(Boolean,     nullable=False, default=False)
    anchor_tx_hash  = Column(String(128), nullable=True)
    anchor_block    = Column(BigInteger,  nullable=True)

    # Timestamps
    created_at      = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    synced_at       = Column(DateTime, nullable=True)  # when synced to central platform

    def __repr__(self) -> str:
        return (
            f"<AuditRecord seq={self.seq} checkpoint={self.checkpoint_id} "
            f"decision={self.decision} anchored={self.anchored}>"
        )


# ── Blockchain Anchor Log (Anubhav) ───────────────────────────────────────────

class BlockchainAnchor(Base):
    """
    Tracks every Sepolia anchor transaction submitted by sepolia_anchor.py.
    Allows the system to resume after a crash without double-submitting.
    """
    __tablename__ = "blockchain_anchors"

    id              = Column(Integer,    primary_key=True, autoincrement=True)
    chain_root_hash = Column(String(64), nullable=False)   # hash of last AuditRecord
    seq_from        = Column(BigInteger, nullable=False)   # first seq in this anchor window
    seq_to          = Column(BigInteger, nullable=False)   # last  seq in this anchor window
    tx_hash         = Column(String(128),nullable=True)    # Sepolia tx hash (null if pending)
    block_number    = Column(BigInteger, nullable=True)
    gas_used        = Column(BigInteger, nullable=True)
    status          = Column(String(16), nullable=False, default="PENDING")  # PENDING/CONFIRMED/FAILED
    submitted_at    = Column(DateTime,   nullable=False, default=datetime.datetime.utcnow)
    confirmed_at    = Column(DateTime,   nullable=True)

    def __repr__(self) -> str:
        return (
            f"<BlockchainAnchor seq={self.seq_from}-{self.seq_to} "
            f"status={self.status} tx={self.tx_hash}>"
        )
