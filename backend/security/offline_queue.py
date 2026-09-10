"""Encrypted offline queue for remote posts that lose connectivity.

Payloads are Fernet-encrypted before INSERT. sync_batch() decrypts a
batch and marks rows synced inside one SQLite transaction (atomic).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.security.aes_encrypt import decrypt_pii, encrypt_pii


class OfflineQueue:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS offline_queue (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    post TEXT NOT NULL,
                    officer_id TEXT NOT NULL,
                    ciphertext TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    synced_at TEXT
                )
                """
            )
            conn.commit()

    def enqueue(self, officer_id: str, post: str, payload: dict[str, Any]) -> dict[str, Any]:
        item_id = str(uuid.uuid4())
        created = datetime.now(timezone.utc).isoformat()
        blob = encrypt_pii(json.dumps(payload, sort_keys=True))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO offline_queue (id, created_at, post, officer_id, ciphertext, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
                """,
                (item_id, created, post, officer_id, blob),
            )
            conn.commit()
        return {"id": item_id, "created_at": created, "status": "pending", "encrypted": True}

    def pending_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM offline_queue WHERE status = 'pending'"
            ).fetchone()
        return int(row["n"])

    def list_pending(self, limit: int = 50, decrypt: bool = False) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, post, officer_id, status, ciphertext
                FROM offline_queue
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            item = {
                "id": r["id"],
                "created_at": r["created_at"],
                "post": r["post"],
                "officer_id": r["officer_id"],
                "status": r["status"],
                "mode": "offline",
            }
            if decrypt:
                item["payload"] = json.loads(decrypt_pii(r["ciphertext"]))
            out.append(item)
        return out

    def sync_batch(self, limit: int = 25) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        synced: list[dict[str, Any]] = []
        with self._connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                rows = conn.execute(
                    """
                    SELECT id, ciphertext, post, officer_id, created_at
                    FROM offline_queue
                    WHERE status = 'pending'
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                for row in rows:
                    payload = json.loads(decrypt_pii(row["ciphertext"]))
                    conn.execute(
                        "UPDATE offline_queue SET status = 'synced', synced_at = ? WHERE id = ?",
                        (now, row["id"]),
                    )
                    synced.append(
                        {
                            "id": row["id"],
                            "created_at": row["created_at"],
                            "post": row["post"],
                            "officer_id": row["officer_id"],
                            "payload": payload,
                        }
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {
            "synced": len(synced),
            "remaining": self.pending_count(),
            "items": synced,
            "atomic": True,
        }