"""module4_face — face verification, liveness and morphing checks.

    from backend.modules.module4_face import run_face_analysis

    result = run_face_analysis(document_bytes, live_bytes)
    # result["face_verification"]["status"] -> MATCH | NOMATCH | NOT_RUN
    # result["liveness"]["status"]           -> GENUINE | NOT_RUN
    # result["morphing"]["status"]           -> LOW | NOT_RUN
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from backend.modules.module4_face.face_detector import detect_faces
from backend.modules.module4_face.face_matcher import match_faces
from backend.modules.module4_face.liveness import check_liveness
from backend.modules.module4_face.morph_detection import check_morphing

logger = logging.getLogger(__name__)


def _decode(source: bytes | None):
    if source is None or len(source) == 0:
        return None
    try:
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def run_face_analysis(document_bytes: bytes | None, live_bytes: bytes | None = None) -> dict[str, Any]:
    """Full Module 4 step: detection -> matching -> liveness -> morphing.
    Never crashes the pipeline: every signal degrades to NOT_RUN."""
    warnings: list[str] = []
    doc_img = _decode(document_bytes)
    live_img = _decode(live_bytes)

    doc_faces = detect_faces(doc_img) if doc_img is not None else []
    live_faces = detect_faces(live_img) if live_img is not None else []

    if doc_img is None and live_img is None:
        warnings.append("No images decoded — face analysis skipped.")

    match = match_faces(doc_img, live_img)
    liveness = check_liveness(live_img)
    morph = check_morphing(doc_img, live_img)

    if match["status"] == "NOT_RUN":
        warnings.append(match["details"])
    if liveness["status"] == "NOT_RUN":
        warnings.append(liveness["details"])
    if morph["status"] == "NOT_RUN":
        warnings.append(morph["details"])

    return {
        "status": "checked" if live_img is not None else "not_checked",
        "document_faces": len(doc_faces),
        "live_faces": len(live_faces),
        "face_verification": match,
        "liveness": liveness,
        "morphing": morph,
        "warnings": warnings,
    }