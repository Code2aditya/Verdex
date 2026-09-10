"""API route blueprints for the unified VERIDEX backend."""

from __future__ import annotations

from backend.api.routes import audit, auth, blockchain, dashboard, face, health, ocr, risk, screen, sync, tamper, validation, verify

__all__ = [
    "audit",
    "auth",
    "blockchain",
    "dashboard",
    "face",
    "health",
    "ocr",
    "risk",
    "screen",
    "sync",
    "tamper",
    "validation",
    "verify",
]