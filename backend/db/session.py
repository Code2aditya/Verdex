"""
session.py — SQLAlchemy engine + session factory for VERIDEX
Uses SQLite locally (offline-first) and PostgreSQL for the central platform.
"""

from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from backend.config.settings import get_settings
from backend.db.models import Base

_settings = get_settings()

# ── Engine ────────────────────────────────────────────────────────────────────
# Use SQLite for the local edge node (offline audit chain).
# Switch to DATABASE_URL (PostgreSQL) when running on the central platform.
_SQLITE_URL = f"sqlite:///{_settings.audit_db_path}"

engine = create_engine(
    _SQLITE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite + FastAPI
    echo=(_settings.app_env == "development"),
)

# Enable WAL mode on SQLite for better concurrent read performance
@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, _):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# ── Session factory ───────────────────────────────────────────────────────────
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    class_=Session,
)


def init_db() -> None:
    """Create all tables if they don't exist. Called at application startup."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    FastAPI dependency that yields a DB session and ensures it is closed
    after the request, even if an exception is raised.

    Usage:
        @router.get("/...")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
