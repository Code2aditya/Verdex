"""
main.py — VERIDEX unified FastAPI application (SIH26188 / SSB Police II)
Document Verification System for Indian identity documents.

Modules wired:
  Module 1 — OCR & MRZ extraction        → /api/v1/ocr/
  Module 2 — Field validation + checksum → /api/v1/validation/
  Module 3 — Tampering & forgery         → /api/v1/tamper/
  Module 4 — Face / liveness / morphing  → /api/v1/face/
  Risk engine + presets                  → /api/risk/, /api/presets
  Officer screening console              → /api/screen
  Tamper-evident audit chain             → /api/audit/
  Blockchain (Sepolia anchor)            → /api/blockchain/
  Offline encrypted sync queue           → /api/offline/, /api/sync/
  Dashboard                              → /api/dashboard/
  Unified verification (Flutter)         → POST /api/verify
                                          POST /api/v1/verify-document
  Health                                 → /api/health, /api/v1/health

Run:
  cd D:\\Verdex_full\\Backend
  .\\venv\\Scripts\\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("veridex")

app = FastAPI(
    title="VERIDEX — Indian Document Verification API",
    description=(
        "AI document fraud detection for Aadhaar / PAN / Passport / Voter ID / "
        "Driving Licence / Visa / Citizenship Certificate. "
        "OCR + Validation + Tampering + Face + Risk Engine + Blockchain Audit."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (Grad-CAM / ELA heatmaps) ───────────────────────────────────
_STATIC_DIR = Path(__file__).resolve().parent / "static"
(_STATIC_DIR / "heatmaps").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ── Routers ───────────────────────────────────────────────────────────────────
from backend.api.routes import (  # noqa: E402
    audit,
    auth,
    blockchain,
    dashboard,
    face,
    health,
    ocr,
    risk,
    screen,
    sync,
    tamper,
    validation,
    verify,
)

for router in (
    auth.router,
    health.router,
    verify.router,
    tamper.router,
    risk.router,
    screen.router,
    audit.router,
    blockchain.router,
    sync.router,
    dashboard.router,
    face.router,
    ocr.router,
    validation.router,
):
    app.include_router(router)


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({
        "message": "VERIDEX API is running. Visit /docs for the interactive API.",
        "docs": "/docs",
        "verify": "/api/verify (multipart POST)",
        "verify_v1": "/api/v1/verify-document (multipart POST)",
        "presets": "/api/presets",
    })


# ── Startup / shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    from backend.db.session import init_db

    init_db()
    from backend.audit.chain import get_chain_root

    logger.info(
        "VERIDEX API ready. Docs: http://127.0.0.1:8000/docs  chain_root=%s...",
        get_chain_root()[:16],
    )


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("VERIDEX API shutting down.")