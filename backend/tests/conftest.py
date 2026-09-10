"""Shared test fixtures. Uses a sandboxed SQLite DB so CI never touches the depot DB."""

from __future__ import annotations

import os
import tempfile

_SCRATCH = tempfile.mkdtemp(prefix="veridex-tests-")
os.environ["AUDIT_DB_PATH"] = os.path.join(_SCRATCH, "test_audit.db")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from backend.db.models import AuditRecord, BlockchainAnchor  # noqa: E402
from backend.db.session import SessionLocal, init_db  # noqa: E402
from backend.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    init_db()
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def sample_document_bytes():
    """Small synthetic JPEG that is cheap to run through the tamper/face stacks."""
    import cv2

    import numpy as np

    canvas = np.full((240, 340, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (40, 40), (300, 60), 0, -1)
    cv2.putText(
        canvas,
        "N1234567",
        (50, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (50, 100), (290, 200), 90, 2)
    ok, buf = cv2.imencode(".jpg", canvas)
    assert ok
    return buf.tobytes()


@pytest.fixture(autouse=True)
def _clean_fields():
    """Wipe audit + anchor + offline tables before every test (fresh chain)."""
    init_db()
    db = SessionLocal()
    try:
        db.query(AuditRecord).delete()
        db.query(BlockchainAnchor).delete()
        db.execute(text("DELETE FROM offline_queue"))
        db.commit()
    finally:
        db.close()
    yield