"""Central platform sync — pushes the encrypted offline queue upstream
and records each sync in the tamper-evident audit chain."""

from __future__ import annotations

from typing import Any

from backend.audit.chain import append_record, ScreeningPayload
from backend.security.offline_queue import OfflineQueue


class CentralSync:
    def __init__(self, offline_queue: OfflineQueue):
        self._queue = offline_queue

    def status(self) -> dict[str, Any]:
        return {
            "pending": self._queue.pending_count(),
            "mode": "encrypted-offline-queue",
            "enabled": True,
        }

    def push_batch(self, officer_id: str, post: str, limit: int = 25) -> dict[str, Any]:
        batch = self._queue.sync_batch(limit=limit)
        for item in batch["items"]:
            append_record(
                ScreeningPayload(
                    checkpoint_id="CHK-001",
                    officer_id=officer_id,
                    doc_type="offline.sync",
                    doc_number=None,
                    risk_score=None,
                    decision="SYNC",
                )
            )
            _ = post
        return batch