"""sync.py — offline queue + central platform sync endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.audit.chain import ScreeningPayload, append_record
from backend.config.settings import get_settings
from backend.security.jwt_auth import get_current_user
from backend.security.offline_queue import OfflineQueue
from backend.sync.central_sync import CentralSync

router = APIRouter(prefix="/api", tags=["Offline Sync"])

_QUEUE = OfflineQueue(get_settings().audit_db_path)
_SYNC = CentralSync(_QUEUE)


class EnqueueBody(BaseModel):
    payload: dict[str, Any]


class SyncBody(BaseModel):
    limit: int = 25


@router.get("/offline/status", summary="Encrypted offline queue status")
def offline_status(_: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {"pending": _QUEUE.pending_count(), "items": _QUEUE.list_pending(decrypt=True)}


@router.post("/offline/enqueue", summary="Enqueue an encrypted offline item")
def offline_enqueue(
    body: EnqueueBody,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    return _QUEUE.enqueue(user["officer_id"], user.get("post", "unknown"), body.payload)


@router.post("/offline/sync", summary="Sync encrypted queue to the central platform")
def offline_sync(
    body: SyncBody | None = None,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    limit = body.limit if body else 25
    batch = _SYNC.push_batch(user["officer_id"], user.get("post", "unknown"), limit=limit)
    return batch


@router.get("/sync/status", summary="Central sync service status")
def sync_status(_: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return _SYNC.status()