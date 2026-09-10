"""Module 4 — face embedding extraction.

Uses InsightFace's buffalo_l model when installed. Without it, a stable
HSV-colour-histogram descriptor is used so the demo can still compare
a document portrait against a live capture. The embedding 'method' is
always reported so the officer knows what produced the score.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_INSIGHT = None


def _load_insightface():
    global _INSIGHT
    if _INSIGHT is None:
        try:
            from backend.config.settings import get_settings

            if not get_settings().enable_insightface:
                logger.info("InsightFace disabled by ENABLE_INSIGHTFACE=false — using histogram descriptor.")
                _INSIGHT = False
                return None
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(224, 224))
            _INSIGHT = app
        except Exception as exc:
            logger.warning("InsightFace unavailable (%s) — using histogram descriptor.", exc)
            _INSIGHT = False
    return _INSIGHT or None


def embed(image_bgr: np.ndarray) -> tuple[list[float], str]:
    """Return (embedding, method). method is 'insightface' or 'histogram'."""
    app = _load_insightface()
    if app is not None:
        faces = app.get(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if faces:
            emb = np.asarray(faces[0].normed_embedding, dtype=np.float64)
            return emb.tolist(), "insightface"
    return _histogram_embedding(image_bgr)


def _histogram_embedding(image_bgr: np.ndarray) -> tuple[list[float], str]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    vec = np.concatenate([hist_h, hist_s, hist_v])
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist(), "histogram"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))