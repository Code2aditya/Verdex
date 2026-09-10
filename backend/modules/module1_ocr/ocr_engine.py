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


def paddle_available() -> bool:
    return _PADDLE_AVAILABLE


def easy_available() -> bool:
    return _EASY_AVAILABLE


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
    return easyocr.Reader(lang_list, gpu=False)


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

    Tries PaddleOCR first, falls back to EasyOCR if unavailable.
    """
    if not image_bytes:
        return _unavailable("no image bytes supplied")

    lang = lang_for_document_type(document_type)

    # --- PaddleOCR ---
    if _PADDLE_AVAILABLE:
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
    if _EASY_AVAILABLE:
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

    return _unavailable("no OCR engine installed (paddleocr and easyocr both unavailable)")


def extract_mrz_lines(ocr_output: dict[str, Any]) -> list[str]:
    """Heuristic: MRZ lines are long, use only A-Z/0-9/'<', and sit at
    the bottom of the document. Filters candidate lines by shape."""
    candidates = []
    for line in ocr_output.get("lines", []):
        text = str(line.get("text", "")).upper().replace(" ", "")
        if len(text) >= 30 and all(c.isalnum() or c == "<" for c in text):
            candidates.append(text)
    return candidates
