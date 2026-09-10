"""Module 1 — OCR engine (PaddleOCR primary, EasyOCR fallback).

PaddleOCR is the preferred engine and natively ships a Devanagari
recognition model (lang="hi" / "ne") used for Indian/Nepali documents.
When PaddleOCR is not available, EasyOCR is used as a fallback with
broad language support. If neither is installed the engine reports
"unavailable" instead of crashing.
"""

from __future__ import annotations

import logging
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from paddleocr import PaddleOCR

    _PADDLE_AVAILABLE = True
except Exception:
    _PADDLE_AVAILABLE = False

try:
    import easyocr

    _EASY_AVAILABLE = True
except Exception:
    _EASY_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image as _PILImage
    from io import BytesIO as _BytesIO

    _TESS_AVAILABLE = True
except Exception:
    _TESS_AVAILABLE = False


def paddle_available() -> bool:
    return _PADDLE_AVAILABLE


def easy_available() -> bool:
    return _EASY_AVAILABLE


def tesseract_available() -> bool:
    if not _TESS_AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _wanted_engines() -> list[str]:
    """Engine priority, overridable via OCR_ENGINE (auto/paddle/easyocr/tesseract)."""
    try:
        from backend.config.settings import get_settings

        wanted = str(get_settings().ocr_engine or "auto").strip().lower()
    except Exception:
        wanted = "auto"
    if wanted in ("paddle", "paddleocr"):
        return ["paddle"]
    if wanted in ("easy", "easyocr"):
        return ["easyocr"]
    if wanted in ("tess", "tesseract"):
        return ["tesseract"]
    return ["paddle", "easyocr", "tesseract"]


_SCRIPT_TO_LANG = {
    "latin": "en",
    "devanagari": "hi",
    "devanagari_latin_mixed": "hi",
}


@lru_cache(maxsize=4)
def get_ocr_engine(lang: str = "en"):
    if _PADDLE_AVAILABLE:
        return PaddleOCR(use_textline_orientation=True, lang=lang)
    return None


@lru_cache(maxsize=4)
def get_easyocr_reader(lang: str = "en"):
    if not _EASY_AVAILABLE:
        return None
    lang_list = ["en"]
    if lang == "hi":
        lang_list = ["hi", "en"]
    try:
        from backend.config.settings import get_settings

        quantize = bool(get_settings().easyocr_quantize)
    except Exception:
        quantize = False
    return easyocr.Reader(lang_list, gpu=False, quantize=quantize)


def lang_for_document_type(document_type: str | None) -> str:
    if not document_type:
        return "en"
    try:
        from backend.config.rule_loader import load_rule

        script = load_rule(document_type).get("script", "latin")
    except Exception:
        script = "latin"
    return _SCRIPT_TO_LANG.get(script, "en")


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "engine": "none",
        "reason": reason,
        "line_count": 0,
        "lines": [],
    }


def run_ocr(image_bytes: bytes, document_type: str | None = None) -> dict[str, Any]:
    """Run OCR on raw image bytes.

    Engine priority: PaddleOCR -> EasyOCR -> Tesseract (all optional).
    OCR_ENGINE env setting can pin one engine (auto/paddle/easyocr/tesseract).
    """
    if not image_bytes:
        return _unavailable("no image bytes supplied")

    lang = lang_for_document_type(document_type)
    wanted = _wanted_engines()

    # --- PaddleOCR ---
    if "paddle" in wanted and _PADDLE_AVAILABLE:
        try:
            engine = get_ocr_engine(lang)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name
            try:
                results = engine.predict(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            lines = []
            for res in results:
                for text, score, box in zip(
                    res.get("rec_texts", []),
                    res.get("rec_scores", []),
                    res.get("rec_polys", []),
                ):
                    lines.append(
                        {
                            "text": text,
                            "confidence": float(score),
                            "box": box.tolist() if hasattr(box, "tolist") else box,
                        }
                    )
            return {"status": "ok", "engine": "paddleocr", "line_count": len(lines), "lines": lines, "lang_used": lang}
        except Exception as exc:
            logger.warning("PaddleOCR inference failed: %s", exc)

    # --- EasyOCR fallback ---
    if "easyocr" in wanted and _EASY_AVAILABLE:
        try:
            reader = get_easyocr_reader(lang)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return _unavailable("could not decode image for OCR")

            results = reader.readtext(img, detail=1)
            lines = []
            for (box, text, score) in results:
                lines.append({
                    "text": str(text),
                    "confidence": float(score),
                    "box": [[int(p[0]), int(p[1])] for p in box],
                })
            return {"status": "ok", "engine": "easyocr", "line_count": len(lines), "lines": lines, "lang_used": lang}
        except Exception as exc:
            logger.warning("EasyOCR inference failed: %s", exc)
            return _unavailable(f"easyocr inference error: {exc}")

    # --- Tesseract fallback (lightweight, no torch; for 1GB containers) ---
    if "tesseract" in wanted and tesseract_available():
        try:
            return _run_tesseract(image_bytes, lang)
        except Exception as exc:
            logger.warning("Tesseract inference failed: %s", exc)
            return _unavailable(f"tesseract inference error: {exc}")

    return _unavailable("no OCR engine installed (paddleocr, easyocr and tesseract all unavailable)")


def _run_tesseract(image_bytes: bytes, lang: str) -> dict[str, Any]:
    """OCR via the Tesseract native binary. Returns the same shape as others."""
    tess_lang = "hin+eng" if lang == "hi" else "eng"
    img = _PILImage.open(_BytesIO(image_bytes)).convert("RGB")
    # Upscale small captures: Tesseract needs ~300 DPI equivalent.
    w, h = img.size
    if max(w, h) < 1200:
        scale = 1200 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
    data = pytesseract.image_to_data(img, lang=tess_lang, output_type=pytesseract.Output.DICT)
    lines: dict[tuple[int, int, int], dict[str, Any]] = {}
    n = len(data.get("text", []))
    for i in range(n):
        text = str(data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if not text or conf < 0:
            continue
        key = (int(data["page_num"][i]), int(data["block_num"][i]), int(data["line_num"][i]))
        slot = lines.setdefault(key, {"text": [], "conf": [], "box": None})
        slot["text"].append(text)
        slot["conf"].append(conf)
        l, t, wd, ht = (int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i]))
        if slot["box"] is None:
            slot["box"] = [l, t, l + wd, t + ht]
        else:
            b = slot["box"]
            slot["box"] = [min(b[0], l), min(b[1], t), max(b[2], l + wd), max(b[3], t + ht)]
    out = []
    for slot in lines.values():
        joined = " ".join(slot["text"]).strip()
        if not joined:
            continue
        avg_conf = sum(slot["conf"]) / len(slot["conf"]) if slot["conf"] else 0.0
        x0, y0, x1, y1 = slot["box"]
        out.append({
            "text": joined,
            "confidence": round(avg_conf / 100.0, 4),
            "box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        })
    return {"status": "ok", "engine": "tesseract", "line_count": len(out), "lines": out, "lang_used": lang}


def extract_mrz_lines(ocr_output: dict[str, Any]) -> list[str]:
    """Heuristic: MRZ lines are long, use only A-Z/0-9/'<', and sit at
    the bottom of the document. Filters candidate lines by shape."""
    candidates = []
    for line in ocr_output.get("lines", []):
        text = str(line.get("text", "")).upper().replace(" ", "")
        if len(text) >= 30 and all(c.isalnum() or c == "<" for c in text):
            candidates.append(text)
    return candidates
