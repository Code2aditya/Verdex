"""
settings.py — Centralised configuration loaded from .env
VERIDEX backend | config
"""

from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_env:        str = "development"
    secret_key:     str = "change-this-to-a-random-string"
    checkpoint_id:  str = "CHK-001"

    # ── PostgreSQL (central platform) ─────────────────────────────────────────
    database_url:   str = "postgresql://user:password@localhost:5432/veridex"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url:      str = "redis://localhost:6379"

    # ── Blockchain (Anubhav) ──────────────────────────────────────────────────
    infura_url:      str = ""
    wallet_address:  str = ""
    private_key:     str = ""

    # ── Audit chain local SQLite DB ───────────────────────────────────────────
    # Path to the SQLite file used for the offline audit chain.
    # In production this should be a SQLCipher-encrypted file.
    audit_db_path:  str = "veridex_audit.db"

    # ── Encryption ────────────────────────────────────────────────────────────
    fernet_key:     str = ""

    # ── JWT ───────────────────────────────────────────────────────────────────
    jwt_secret:        str = "change-this-to-a-random-string"
    jwt_expiry_hours:  int = 8


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (reads .env once at startup)."""
    return Settings()
