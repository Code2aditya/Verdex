"""Module 2 — watchlist / identity-graph lookup.

SQLite-backed watchlist with the ability to add/remove entries at
runtime. Comes pre-seeded with a few demo entries for testing.
In production this would connect to an encrypted national watchlist
and SLTD database.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

_WATCHLIST_DB_PATH = os.environ.get(
    "WATCHLIST_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "watchlist.db"),
)


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_WATCHLIST_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_WATCHLIST_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_type TEXT NOT NULL,
            doc_number TEXT NOT NULL,
            name TEXT NOT NULL,
            ref TEXT DEFAULT '',
            severity TEXT DEFAULT 'MEDIUM',
            reason TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def _seed_if_empty(conn: sqlite3.Connection) -> None:
    count = conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
    if count > 0:
        return
    seeds = [
        ("passport", "P9988776", "BINOD THAPA", "SLTD-88234", "HIGH",
         "Reported lost passport — flagged for lookalike abuse."),
        ("citizenship_certificate", "CC-441902", "DEEPAK KARKI", "INT-99102", "CRITICAL",
         "Duplicate identity linked to two active certificate numbers."),
    ]
    conn.executemany(
        "INSERT INTO watchlist (doc_type, doc_number, name, ref, severity, reason) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        seeds,
    )
    conn.commit()


def _norm(value: object) -> str:
    return str(value or "").upper().replace("<", "").replace(" ", "").strip()


def watchlist_check(
    document_type: str | None = None,
    doc_number: Any = None,
    name: Any = None,
) -> dict[str, Any]:
    """Look up a document number / name against the watchlist database."""
    num = _norm(doc_number)
    nm = _norm(name)
    matches = []

    try:
        conn = _get_conn()
        _ensure_table(conn)
        _seed_if_empty(conn)

        if num:
            rows = conn.execute(
                "SELECT * FROM watchlist WHERE UPPER(REPLACE(doc_number, ' ', '')) = ?",
                (num,),
            ).fetchall()
            for row in rows:
                if document_type and row["doc_type"] != document_type:
                    continue
                matches.append(dict(row))

        if nm and not matches:
            rows = conn.execute(
                "SELECT * FROM watchlist WHERE UPPER(REPLACE(name, ' ', '')) = ?",
                (nm,),
            ).fetchall()
            for row in rows:
                matches.append(dict(row))

        conn.close()
    except Exception as exc:
        logger.warning("Watchlist DB query failed: %s", exc)

    return {
        "hit": len(matches) > 0,
        "matches": matches,
        "checked": True,
        "note": f"Watchlist: {len(matches)} match(es) found.",
    }


def watchlist_add(
    doc_type: str,
    doc_number: str,
    name: str,
    ref: str = "",
    severity: str = "MEDIUM",
    reason: str = "",
) -> dict[str, Any]:
    """Add an entry to the watchlist."""
    try:
        conn = _get_conn()
        _ensure_table(conn)
        conn.execute(
            "INSERT INTO watchlist (doc_type, doc_number, name, ref, severity, reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (doc_type, doc_number.upper(), name.upper(), ref, severity, reason),
        )
        conn.commit()
        conn.close()
        return {"ok": True, "message": f"Added {doc_number} to watchlist."}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


def watchlist_remove(doc_number: str) -> dict[str, Any]:
    """Remove entries by document number."""
    try:
        conn = _get_conn()
        _ensure_table(conn)
        cur = conn.execute(
            "DELETE FROM watchlist WHERE UPPER(REPLACE(doc_number, ' ', '')) = ?",
            (doc_number.upper().replace(" ", ""),),
        )
        conn.commit()
        removed = cur.rowcount
        conn.close()
        return {"ok": True, "removed": removed}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


def watchlist_status() -> dict[str, Any]:
    try:
        conn = _get_conn()
        _ensure_table(conn)
        _seed_if_empty(conn)
        count = conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
        conn.close()
        return {"size": count, "enabled": True, "mode": "sqlite", "db_path": _WATCHLIST_DB_PATH}
    except Exception as exc:
        return {"size": 0, "enabled": False, "mode": "error", "error": str(exc)}
