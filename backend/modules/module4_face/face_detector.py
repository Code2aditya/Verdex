"""Module 4 — face detection (OpenCV Haar cascade fallback)."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_CASCADE = None


def _load_cascade():
    global _CASCADE
    if _CASCADE is None:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            _CASCADE = cv2.CascadeClassifier(cascade_path)
            if _CASCADE.empty():
                _CASCADE = False
                logger.warning("Face cascade is empty (model file not bundled).")
        except Exception as exc:
            logger.warning("Face cascade unavailable: %s", exc)
            _CASCADE = False
    return _CASCADE or None


def detect_faces(image_bgr: np.ndarray) -> list[dict]:
    """Return facial bounding boxes {x,y,w,h} for a BGR image."""
    cascade = _load_cascade()
    if cascade is None:
        return []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(min(60, gray.shape[1]), min(60, gray.shape[0])),
    )
    return [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]