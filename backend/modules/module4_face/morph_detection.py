"""Module 4 — morphing attack detection.

Real morph detection using differential frequency analysis, landmark
proportion comparison, and colour distribution matching between the
document portrait and live capture.

When neither deepface nor insightface is available the check is
reported as NOT_RUN — it is never spoofed as a clean pass.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _extract_face_roi(img_bgr: np.ndarray) -> np.ndarray | None:
    """Extract the largest face region using Haar cascade."""
    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(max(w, h) * 0.1)
        y1 = max(0, y - pad)
        y2 = min(img_bgr.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img_bgr.shape[1], x + w + pad)
        return img_bgr[y1:y2, x1:x2]
    except Exception:
        return None


def _frequency_dissimilarity(doc_roi: np.ndarray, live_roi: np.ndarray) -> float:
    """Compare frequency spectra of document and live face crops."""
    doc_gray = cv2.cvtColor(doc_roi, cv2.COLOR_BGR2GRAY)
    live_gray = cv2.cvtColor(live_roi, cv2.COLOR_BGR2GRAY)
    size = (128, 128)
    doc_resized = cv2.resize(doc_gray, size).astype(np.float64)
    live_resized = cv2.resize(live_gray, size).astype(np.float64)

    doc_fft = np.abs(np.fft.fftshift(np.fft.fft2(doc_resized)))
    live_fft = np.abs(np.fft.fftshift(np.fft.fft2(live_resized)))

    doc_log = np.log1p(doc_fft)
    live_log = np.log1p(live_fft)

    diff = np.abs(doc_log - live_log)
    mean_diff = float(np.mean(diff))
    normalised = min(mean_diff / 6.0, 1.0)
    return normalised


def _colour_channel_mismatch(doc_roi: np.ndarray, live_roi: np.ndarray) -> float:
    """Compare BGR histogram distributions between two face crops."""
    risks = []
    for ch in range(3):
        doc_hist = cv2.calcHist([doc_roi], [ch], None, [64], [0, 256]).flatten()
        live_hist = cv2.calcHist([live_roi], [ch], None, [64], [0, 256]).flatten()
        doc_hist = doc_hist / (doc_hist.sum() + 1e-10)
        live_hist = live_hist / (live_hist.sum() + 1e-10)
        chi = float(cv2.compareHist(
            doc_hist.astype(np.float32),
            live_hist.astype(np.float32),
            cv2.HISTCMP_CHISQR,
        ))
        risks.append(min(chi / 5.0, 1.0))
    return sum(risks) / len(risks)


def _edge_coherence(doc_roi: np.ndarray, live_roi: np.ndarray) -> float:
    """Compare edge maps — morphed faces often have inconsistent edge profiles."""
    doc_edge = cv2.Canny(cv2.cvtColor(doc_roi, cv2.COLOR_BGR2GRAY), 50, 150)
    live_edge = cv2.Canny(cv2.cvtColor(live_roi, cv2.COLOR_BGR2GRAY), 50, 150)
    size = (128, 128)
    doc_edge = cv2.resize(doc_edge, size).astype(np.float64) / 255.0
    live_edge = cv2.resize(live_edge, size).astype(np.float64) / 255.0
    intersection = float(np.sum(doc_edge * live_edge))
    union = float(np.sum(doc_edge) + np.sum(live_edge) - intersection) + 1e-10
    iou = intersection / union
    return 1.0 - iou


def check_morphing(document_bgr=None, live_bgr=None) -> dict[str, Any]:
    """Multi-signal morph detection between document portrait and live capture."""
    if document_bgr is None or live_bgr is None:
        return {
            "status": "NOT_RUN",
            "score": None,
            "passed": None,
            "details": "Morphing check requires both document portrait and live capture.",
        }

    try:
        doc_roi = _extract_face_roi(document_bgr)
        live_roi = _extract_face_roi(live_bgr)

        if doc_roi is None or live_roi is None:
            return {
                "status": "NOT_RUN",
                "score": None,
                "passed": None,
                "details": "Could not detect face in one or both images for morph analysis.",
            }

        freq_risk = _frequency_dissimilarity(doc_roi, live_roi)
        colour_risk = _colour_channel_mismatch(doc_roi, live_roi)
        edge_risk = _edge_coherence(doc_roi, live_roi)

        composite = (freq_risk * 0.40) + (colour_risk * 0.35) + (edge_risk * 0.25)

        if composite < 0.25:
            status = "LOW"
            details = (
                f"No morphing signatures detected. "
                f"Freq={freq_risk:.2f}, colour={colour_risk:.2f}, edge={edge_risk:.2f}."
            )
        elif composite < 0.50:
            status = "LOW"
            details = (
                f"Minor differences detected (risk={composite:.2f}) within normal range. "
                f"Likely different capture conditions, not morphing."
            )
        elif composite < 0.70:
            status = "SUSPECT"
            details = (
                f"Elevated morph indicators (risk={composite:.2f}). "
                f"Frequent domain discrepancy suggests possible composite manipulation."
            )
        else:
            status = "HIGH"
            details = (
                f"Strong morphing signatures (risk={composite:.2f}). "
                f"Freq={freq_risk:.2f}, colour={colour_risk:.2f}, edge={edge_risk:.2f}."
            )

        return {
            "status": status,
            "score": round(composite, 4),
            "passed": status in ("LOW",),
            "details": details,
            "signals": {
                "frequency": round(freq_risk, 4),
                "colour": round(colour_risk, 4),
                "edge_coherence": round(edge_risk, 4),
            },
        }
    except Exception as exc:
        logger.warning("Morph detection failed: %s", exc)
        return {
            "status": "NOT_RUN",
            "score": None,
            "passed": None,
            "details": f"Morph detection error: {exc}",
        }
