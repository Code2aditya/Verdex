# VERIDEX — Handover Report
**Author:** Anubhav (Team Beta)  
**Date:** 9 September 2026  
**Branch:** `main`  
**Repo:** https://github.com/PanchalAnubhav/Veridex-

---

## ✅ Completed Work

### Module 3 — Tampering & Forgery Detection

| File | Description |
|---|---|
| `backend/modules/module3_tampering/preprocessing.py` | Image load, deskew, denoise, CNN tensor prep |
| `backend/modules/module3_tampering/qr_check.py` | QR/barcode decode + OCR cross-check |
| `backend/modules/module3_tampering/ela.py` | Error Level Analysis — detects JPEG re-compression artefacts |
| `backend/modules/module3_tampering/noise_analysis.py` | Sensor noise inconsistency + copy-move detection |
| `backend/modules/module3_tampering/cnn_classifier.py` | EfficientNet-B0 forgery classifier (fine-tuned binary head) |
| `backend/modules/module3_tampering/gradcam.py` | Grad-CAM heatmap generation + overlay saved to `/static/heatmaps/` |
| `backend/modules/module3_tampering/tamper_pipeline.py` | Orchestrates ELA + CNN + Noise + QR → `{tamper_score, heatmap_url, flags}` |
| `backend/modules/module3_tampering/__init__.py` | Public API exports |

### Blockchain & Audit

| File | Description |
|---|---|
| `backend/blockchain/sepolia_anchor.py` | Sepolia testnet anchoring — 30/30 tests pass |
| `backend/audit/chain.py` | Hash-chained audit log |
| `backend/audit/__init__.py` | Public exports |

### Infrastructure

| File | Description |
|---|---|
| `backend/main.py` | FastAPI app — CORS, static files, startup model pre-load |
| `backend/api/routes/tamper.py` | `POST /api/v1/tamper/analyze` + `GET /api/v1/tamper/status` |
| `backend/config/settings.py` | App settings / env config |
| `backend/db/models.py` | SQLAlchemy models |
| `backend/db/session.py` | DB session management |

### Scripts

| File | Purpose |
|---|---|
| `scripts/generate_dataset.py` | Generates synthetic tampered document images from genuine source images |
| `scripts/train_cnn.py` | Fine-tunes EfficientNet-B0 binary head (~10 mins on CPU) |
| `scripts/test_step1.py` | Module 3 step 1 tests |
| `scripts/test_step2.py` | Module 3 step 2 tests |

---

## 🚀 How to Run

```bash
cd f:\SIh\veridex
venv\Scripts\activate
uvicorn backend.main:app --reload --port 8000
```

| Endpoint | Method | Description |
|---|---|---|
| `/docs` | GET | Interactive Swagger UI |
| `/api/v1/health` | GET | System health check |
| `/api/v1/tamper/status` | GET | CNN model load status |
| `/api/v1/tamper/analyze` | POST | Upload document → tamper score + heatmap |

### CNN Fine-tuning (if not done)

```bash
# 1. Drop genuine document images into dataset/genuine/
# 2. Generate tampered variants:
python scripts/generate_dataset.py

# 3. Fine-tune the CNN head (~10 mins on CPU):
python scripts/train_cnn.py
# Saves checkpoint to: models/checkpoints/efficientnet_tamper.pth
```

---

## 🔌 Risk Engine Integration (Team Gamma)

Call Module 3 from `backend/risk_engine/scorer.py`:

```python
import httpx

response = httpx.post(
    "http://localhost:8000/api/v1/tamper/analyze",
    files={"file": ("doc.jpg", image_bytes, "image/jpeg")},
    data={
        "doc_number": ocr_fields.get("doc_number"),
        "dob":        ocr_fields.get("dob"),
        "expiry":     ocr_fields.get("expiry"),
        "name":       ocr_fields.get("name"),
    }
)

data         = response.json()
tamper_score = data["tamper_score"]   # float 0.0 – 1.0
risk_tier    = data["risk_tier"]      # "LOW" / "MEDIUM" / "HIGH"
heatmap_url  = data["heatmap_url"]    # "/static/heatmaps/<uuid>.jpg"
flags        = data["flags"]          # list[str]
```

**Sprint doc weight for Module 3 in composite risk score: 25%**

---

## 🗑️ Empty Files — Team Gamma / Team Alpha to Fill or Delete

These files exist in the repo as blank placeholders:

```
# Routes (empty — to be implemented or deleted)
backend/api/routes/audit.py
backend/api/routes/auth.py
backend/api/routes/screen.py
backend/api/routes/sync.py

# Module 1 — OCR (Team Alpha)
backend/modules/module1_ocr/*.py

# Module 2 — Validation (Team Alpha)
backend/modules/module2_validation/*.py

# Module 4 — Face (Team Gamma — Anshit)
backend/modules/module4_face/*.py

# Risk Engine (Team Gamma — Rishabh)
backend/risk_engine/scorer.py
backend/risk_engine/explainer.py

# Security
backend/security/aes_encrypt.py
backend/security/jwt_auth.py

# Sync
backend/sync/central_sync.py

# Empty test files
backend/tests/test_*.py
```

> If a file will not be implemented before the deadline, **delete it** — empty files in a handover make the codebase look incomplete.

---

## Git History

```
b3b4b46  chore: merge all branches into main, clean up gitignore
438d312  feat module 3: initialize backend service and implement tamper detection API route
5e297f3  feat module3: implement document tampering detection pipeline
249f77b  feat(blockchain): audit chain + Sepolia anchor (30/30 tests pass)
3b8293c  feat/module3: add module3_tampering package with QR/barcode decoding
b86cfe0  feat/module3: implement preprocessing module for tampering detection
0c673a5  init: VERIDEX project structure
```
