# VERIDEX — AI-Powered Identity & Document Forensics Platform
**SIH26188 | Ministry of Home Affairs — SSB, Police II Division**

> Real-time AI document verification: OCR · Tampering Detection · Face Verification · Blockchain Audit

---

## Team

| Name | Team | Role |
|---|---|---|
| Mayank | Alpha | Module 1 — OCR / MRZ Extraction |
| Anisha | Alpha | Module 2 — Document Validation Engine |
| Raj | Beta | Module 3 — ELA + CNN Classifier + Grad-CAM |
| Anubhav | Beta | Module 3 — Preprocessing + QR Check + API · Blockchain |
| Anshit | Gamma | Module 4 — Face Verification + Liveness + Morph Detection |
| Rishabh | Gamma | Risk Engine + Officer Dashboard + Central Platform |

---

## Architecture

```
Document Image
      │
      ▼
┌─────────────────────────────────────────────────┐
│  Module 1 — OCR & MRZ Extraction                │  Team Alpha
│  PaddleOCR · passporteye · Devanagari support   │
└───────────────────┬─────────────────────────────┘
                    │ structured fields
                    ▼
┌─────────────────────────────────────────────────┐
│  Module 2 — Document Validation Engine          │  Team Alpha
│  Field consistency · expiry · watchlist lookup  │
└───────────────────┬─────────────────────────────┘
                    │ pass / fail + reason
                    ▼
┌─────────────────────────────────────────────────┐
│  Module 3 — Tampering & Forgery Detection       │  Team Beta
│  ELA · EfficientNet CNN · Grad-CAM · QR check   │
└───────────────────┬─────────────────────────────┘
                    │ tamper_score · heatmap_url · flags
                    ▼
┌─────────────────────────────────────────────────┐
│  Module 4 — Face Verification & Liveness        │  Team Gamma
│  InsightFace · MediaPipe · morph detection      │
└───────────────────┬─────────────────────────────┘
                    │ face_score · liveness · match
                    ▼
┌─────────────────────────────────────────────────┐
│  Risk Engine — Composite Scoring                │  Team Gamma
│  OCR 20% · Validation 20% · Tamper 25%         │
│  Face 20% · Morph 10% · Watchlist 5%           │
└───────────────────┬─────────────────────────────┘
                    │ risk_score · tier · recommended action
                    ▼
┌─────────────────────────────────────────────────┐
│  Officer Dashboard + Blockchain Audit           │  Team Gamma
│  Sepolia anchor · hash-chained audit log · JWT  │
└─────────────────────────────────────────────────┘
```

---

## Module Status

| Module | Owner | Status |
|---|---|---|
| Module 1 — OCR & MRZ | Team Alpha | ✅ Complete |
| Module 2 — Validation | Team Alpha | ✅ Complete |
| Module 3 — Tampering Detection | Team Beta | ✅ Complete |
| Module 4 — Face & Liveness | Team Gamma | 🔄 In Progress |
| Risk Engine + Dashboard | Team Gamma | 🔄 In Progress |
| Blockchain Audit (Sepolia) | Anubhav | ✅ Complete |
| FastAPI Backend | Anubhav | ✅ Active (`/api/v1/tamper/`) |

---

## Setup

```bash
git clone https://github.com/PanchalAnubhav/Veridex-.git
cd Veridex-

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

Copy `.env.example` → `.env` and fill in your Infura key, DB URL, and JWT secret.

---

## Run (Backend)

```bash
# From project root
uvicorn backend.main:app --reload --port 8000
```

| URL | Description |
|---|---|
| http://localhost:8000/docs | Interactive API docs (Swagger UI) |
| http://localhost:8000/api/v1/health | System health check |
| http://localhost:8000/api/v1/tamper/analyze | **POST** — tamper detection |
| http://localhost:8000/api/v1/tamper/status | **GET** — CNN model status |

---

## CNN Fine-tuning (Module 3)

```bash
# 1. Add genuine document images to dataset/genuine/
# 2. Generate synthetic tampered variants:
python scripts/generate_dataset.py

# 3. Fine-tune the EfficientNet binary head (~10 mins on CPU):
python scripts/train_cnn.py
# Checkpoint saved to: models/checkpoints/efficientnet_tamper.pth
```

---

## Handover

See [`HANDOVER.md`](./HANDOVER.md) for full details on completed work, empty stub files, and risk engine integration instructions.

---

## Deadline
**10 September 2026 — 11:59 PM**
