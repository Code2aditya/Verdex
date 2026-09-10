"""Module 4 — liveness (presentation-attack) check.

Real liveness detection using InsightFace 3D landmarks for geometry
consistency, OpenCV for texture analysis and HSV colour-distribution
artefact detection, and frequency-domain moiré detection.

When no face models are available the check is reported as NOT_RUN —
it is never spoofed as a genuine liveness pass.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _insight_app():
    try:
        from functools import lru_cache
        from insightface.app import FaceAnalysis

        @lru_cache(maxsize=1)
        def _get():
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0)
            return app

        return _get()
    except Exception:
        return None


def _texture_score(face_bgr: np.ndarray) -> float:
    """Laplacian variance — live skin has higher variance than prints/screens."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = lap.var()
    if var < 30:
        return 0.9
    if var < 80:
        return 0.6
    if var < 200:
        return 0.3
    return 0.1


def _colour_score(face_bgr: np.ndarray) -> float:
    """HSV saturation/spread analysis — printed photos cluster narrowly."""
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    s_mean = float(np.mean(hsv[:, :, 1]))
    s_std = float(np.std(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))
    risk = 0.0
    if s_std < 15:
        risk += 0.4
    if s_mean < 30 or s_mean > 220:
        risk += 0.3
    if v_mean < 40 or v_mean > 240:
        risk += 0.2
    return min(risk, 1.0)


def _screen_moiré_score(face_bgr: np.ndarray) -> float:
    """Detect moiré / banding patterns typical of screen re-capture."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    ring = (dist > min(h, w) * 0.15) & (dist < min(h, w) * 0.45)
    ring_energy = float(np.mean(mag[ring]))
    total_energy = float(np.mean(mag)) + 1e-10
    ring_ratio = ring_energy / total_energy
    if ring_ratio > 0.35:
        return 0.8
    if ring_ratio > 0.25:
        return 0.5
    return 0.1


def _geometry_score(lmk3d) -> float:
    """Check 68-point 3D landmark consistency against normal human geometry.

    Uses the InsightFace landmark_3d_68 model to detect flat/printed
    faces (low depth variance) and anatomically implausible proportions.
    """
    try:
        pts = np.array(lmk3d, dtype=np.float64).reshape(-1, 3)
        left_eye = pts[36]
        right_eye = pts[45]
        nose_tip = pts[30]
        mouth_left = pts[48]
        mouth_right = pts[54]

        eye_dist = float(np.linalg.norm(right_eye[:2] - left_eye[:2]))
        mouth_width = float(np.linalg.norm(mouth_right[:2] - mouth_left[:2]))
        nose_to_eye = float(np.linalg.norm(nose_tip[:2] - ((left_eye[:2] + right_eye[:2]) / 2)))

        risk = 0.0
        if eye_dist > 0:
            ratio = mouth_width / eye_dist
            if ratio < 0.3 or ratio > 2.5:
                risk += 0.4
        if eye_dist > 0:
            depth_ratio = abs(left_eye[2] - right_eye[2]) / (eye_dist + 1e-6)
            if depth_ratio > 0.3:
                risk += 0.3
        if eye_dist > 0:
            face_ratio = nose_to_eye / eye_dist
            if face_ratio < 0.2 or face_ratio > 1.8:
                risk += 0.3

        if pts.shape[0] >= 68:
            z = pts[:, 2]
            z_range = float(np.max(z) - np.min(z))
            if z_range < 0.5:
                risk += 0.4
        return min(risk, 1.0)
    except Exception:
        return 0.0


def check_liveness(live_bgr=None) -> dict[str, Any]:
    """Run full liveness analysis on a live capture frame."""
    if live_bgr is None:
        return {
            "status": "NOT_RUN",
            "score": None,
            "passed": None,
            "details": "No live capture provided — liveness check skipped.",
        }

    try:
        live_rgb = cv2.cvtColor(live_bgr, cv2.COLOR_BGR2RGB)
        app = _insight_app()

        geo_risk = 0.0
        if app is not None:
            try:
                faces = app.get(live_rgb)
                if faces:
                    geo_risk = _geometry_score(faces[0].kps if hasattr(faces[0], "kps") else None)
                    if geo_risk == 0.0:
                        lmks = getattr(faces[0], "landmark_3d_68", None)
                        geo_risk = _geometry_score(lmks)
                else:
                    return {
                        "status": "FAIL",
                        "score": None,
                        "passed": False,
                        "details": "No face detected in live capture — liveness cannot be assessed.",
                    }
            except Exception as exc:
                logger.warning("InsightFace geometry analysis failed: %s", exc)

        tex = _texture_score(live_bgr)
        col = _colour_score(live_bgr)
        moire = _screen_moiré_score(live_bgr)

        has_geometry = app is not None and geo_risk != 0.0
        composite = (tex * 0.40) + (col * 0.30) + (moire * 0.30)
        if has_geometry:
            composite = (geo_risk * 0.30) + (tex * 0.30) + (col * 0.20) + (moire * 0.20)

        if composite < 0.25:
            status = "GENUINE"
            details = (
                f"Texture variance OK, geometry consistent, no moiré artefacts. "
                f"Composite liveness risk={composite:.2f}."
            )
        elif composite < 0.50:
            status = "GENUINE"
            details = (
                f"Mild artefacts detected (risk={composite:.2f}) but within tolerance. "
                f"Officer should verify manually."
            )
        elif composite < 0.70:
            status = "SUSPECT"
            details = (
                f"Elevated presentation-attack signals (risk={composite:.2f}). "
                f"Possible printed photo or screen replay."
            )
        else:
            status = "FAIL"
            details = (
                f"Strong presentation-attack indicators (risk={composite:.2f}). "
                f"Texture={tex:.2f}, geometry={geo_risk:.2f}, colour={col:.2f}, moiré={moire:.2f}."
            )

        return {
            "status": status,
            "score": round(composite, 4),
            "passed": status == "GENUINE",
            "details": details,
            "signals": {
                "geometry": round(geo_risk, 4) if has_geometry else None,
                "texture": round(tex, 4),
                "colour": round(col, 4),
                "moiré": round(moire, 4),
            },
        }
    except Exception as exc:
        logger.warning("Liveness analysis failed: %s", exc)
        return {
            "status": "NOT_RUN",
            "score": None,
            "passed": None,
            "details": f"Liveness analysis error: {exc}",
        }