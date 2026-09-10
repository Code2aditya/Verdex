"""
tamper_pipeline.py — Module 3 orchestration pipeline
Module 3 | VERIDEX | Raj + Anubhav

Entry point for the entire Module 3 tampering & forgery detection pipeline.
Orchestrates all sub-components into a single call that returns the
{tamper_score, heatmap_url, flags} dict consumed by the risk engine.

Signal weights (calibrated for demo without fine-tuned CNN):
  ELA             : 35%  (algorithmic, highly reliable)
  Noise analysis  : 25%  (algorithmic, reliable)
  CNN classifier  : 25%  (less weight until fine-tuned; bumped to 35% after)
  QR cross-check  : 15%  (highly reliable when QR present; 0 when not)

Usage
-----
  from backend.modules.module3_tampering import run_tamper_detection

  result = run_tamper_detection(
      source="path/to/doc.jpg",
      ocr_fields={"doc_number": "A1234567", "dob": "1990-01-15", ...},
  )
  print(result.tamper_score)   # 0.0 – 1.0
  print(result.heatmap_url)    # "/static/heatmaps/<uuid>.jpg"
  print(result.flags)          # ["ELA_HIGH_INTENSITY", "QR_MISMATCH_doc_number"]
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from .preprocessing   import preprocess, PreprocessResult
from .ela             import run_ela,       ELAResult
from .noise_analysis  import analyze_noise, NoiseResult
from .qr_check        import check_qr,      QRCheckResult

# The CNN classifier and Grad-CAM heatmap require torch(+torchvision).
# They are optional: when unavailable, their share of the composite
# score is redistributed across the algorithmic signals (ELA/noise/QR).
try:
    from .cnn_classifier import classify as _classify_impl, _FINE_TUNED
    from .gradcam        import generate_heatmap as _heatmap_impl

    def _cnn_available() -> bool:
        return True

except Exception:  # pragma: no cover - depends on environment
    _classify_impl = None
    _heatmap_impl = None

    def _cnn_available() -> bool:
        return False

    _FINE_TUNED = False

logger = logging.getLogger(__name__)

# ── Signal weights ─────────────────────────────────────────────────────────────
# Switch to FINE_TUNED weights when a calibrated checkpoint is present
_WEIGHTS_UNCALIBRATED = {"ela": 0.35, "noise": 0.25, "cnn": 0.25, "qr": 0.15}
_WEIGHTS_CALIBRATED   = {"ela": 0.25, "noise": 0.20, "cnn": 0.40, "qr": 0.15}

# Risk tiers
_TIER_HIGH   = 0.65
_TIER_MEDIUM = 0.40


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class TamperResult:
    """
    Final result of run_tamper_detection().  This is what the risk engine
    (Module 4) and the FastAPI endpoint consume.

    Attributes
    ----------
    tamper_score    : Composite forgery score [0.0 – 1.0].
                      0.0 = almost certainly genuine.
                      1.0 = almost certainly tampered.
    risk_tier       : "LOW" | "MEDIUM" | "HIGH"
    heatmap_url     : URL of the Grad-CAM overlay JPEG saved to disk.
    flags           : List of string flags raised by any sub-component.
    ela_result      : Full ELAResult for downstream inspection.
    cnn_result      : Full CNNResult for downstream inspection.
    noise_result    : Full NoiseResult for downstream inspection.
    qr_result       : Full QRCheckResult for downstream inspection.
    gradcam_result  : Full GradCAMResult for downstream inspection.
    preprocess_result: Full PreprocessResult (image representations).
    processing_ms   : Wall-clock time taken by the pipeline (milliseconds).
    warnings        : Non-fatal issues from any sub-component.
    """
    tamper_score:      float
    risk_tier:         str
    heatmap_url:       str
    flags:             list
    ela_result:        ELAResult
    cnn_result:        CNNResult
    noise_result:      NoiseResult
    qr_result:         QRCheckResult
    gradcam_result:    GradCAMResult
    preprocess_result: PreprocessResult
    processing_ms:     float
    warnings: list = field(default_factory=list)

    # ── Convenience serialisation ──────────────────────────────────────────────
    def to_api_dict(self) -> dict:
        """
        Return a JSON-serialisable dict for the FastAPI response.
        Omits numpy arrays (not serialisable); returns only scalars and lists.
        """
        return {
            "tamper_score":    self.tamper_score,
            "risk_tier":       self.risk_tier,
            "heatmap_url":     self.heatmap_url,
            "flags":           self.flags,
            "processing_ms":   self.processing_ms,
            "signals": {
                "ela": {
                    "risk_delta":      self.ela_result.risk_delta,
                    "mean_intensity":  self.ela_result.mean_intensity,
                    "max_intensity":   self.ela_result.max_intensity,
                },
                "cnn": {
                    "tamper_prob":  self.cnn_result.tamper_prob,
                    "genuine_prob": self.cnn_result.genuine_prob,
                    "fine_tuned":   self.cnn_result.fine_tuned,
                },
                "noise": {
                    "risk_delta":    self.noise_result.risk_delta,
                    "flagged_pct":   self.noise_result.flagged_pct,
                    "flagged_blocks": self.noise_result.flagged_blocks,
                    "total_blocks":  self.noise_result.total_blocks,
                },
                "qr": {
                    "found":       self.qr_result.found,
                    "mismatches":  self.qr_result.mismatches,
                    "risk_delta":  self.qr_result.risk_delta,
                    "explanation": self.qr_result.explanation,
                },
            },
            "warnings": self.warnings,
        }


# ── Placeholder results when torch is unavailable ────────────────────────────

class _CnnUnavailable:
    """API-compatible stand-in for CNNResult when the classifier is absent."""
    def __init__(self) -> None:
        self.tamper_prob = 0.5
        self.genuine_prob = 0.5
        self.flags: list = []
        self.fine_tuned = False
        self.warnings: list = []


class _GradCamUnavailable:
    """API-compatible stand-in for GradCAMResult when torch is absent."""
    def __init__(self) -> None:
        self.heatmap_url = ""
        self.warnings: list = []


# ── Public API ────────────────────────────────────────────────────────────────

def run_tamper_detection(
    source: Union[str, Path, bytes, bytearray, np.ndarray, Image.Image],
    ocr_fields: dict | None = None,
) -> TamperResult:
    """
    Full Module 3 tampering detection pipeline.

    Parameters
    ----------
    source      : Document image — any format accepted by preprocessing.preprocess().
    ocr_fields  : Optional dict of canonical OCR field values from Module 1/2.
                  Used by the QR cross-check sub-component.

    Returns
    -------
    TamperResult with tamper_score, heatmap_url, and flags.
    """
    t_start = time.perf_counter()
    all_warnings: list[str] = []
    all_flags:    list[str] = []

    # ── 1. Preprocessing ──────────────────────────────────────────────────────
    logger.info("[Module3] Starting preprocessing...")
    prep = preprocess(source)
    all_warnings.extend(prep.warnings)

    # ── 2. ELA ────────────────────────────────────────────────────────────────
    logger.info("[Module3] Running ELA...")
    ela = run_ela(prep.canvas)
    all_warnings.extend(ela.warnings)
    if ela.risk_delta > 0.6:
        all_flags.append("ELA_HIGH_INTENSITY")
    elif ela.risk_delta > 0.35:
        all_flags.append("ELA_MODERATE_INTENSITY")

    # ── 3. Noise analysis ─────────────────────────────────────────────────────
    logger.info("[Module3] Running noise analysis...")
    noise = analyze_noise(prep.canvas, prep.gray)
    all_warnings.extend(noise.warnings)
    if noise.risk_delta > 0.6:
        all_flags.append("NOISE_HIGH_ANOMALY")
    elif noise.risk_delta > 0.35:
        all_flags.append("NOISE_MODERATE_ANOMALY")

    # ── 4. CNN classifier (optional — needs torch) ────────────────────────────
    cnn_used = False
    cnn = _CnnUnavailable()
    if _classify_impl is not None:
        logger.info("[Module3] Running CNN classifier...")
        try:
            cnn = _classify_impl(prep.cnn_input)
            all_warnings.extend(cnn.warnings)
            all_flags.extend(cnn.flags)
            cnn_used = True
        except Exception as exc:  # pragma: no cover - e.g. weight download offline
            logger.warning("[Module3] CNN classifier unavailable at runtime: %s", exc)
            all_warnings.append(f"CNN classifier unavailable: {exc}")
    else:
        all_warnings.append(
            "CNN classifier not available (torch missing in this environment) — "
            "signal excluded; ELA/noise/QR weights redistributed."
        )

    # ── 5. Grad-CAM heatmap (optional — needs torch) ──────────────────────────
    gcam = _GradCamUnavailable()
    if _heatmap_impl is not None:
        logger.info("[Module3] Generating Grad-CAM heatmap...")
        try:
            from .cnn_classifier import get_model

            model = get_model()
            gcam = _heatmap_impl(model, prep.cnn_input, prep.canvas)
            all_warnings.extend(gcam.warnings)
        except Exception as exc:  # pragma: no cover
            logger.warning("[Module3] Grad-CAM unavailable at runtime: %s", exc)
            all_warnings.append(f"Grad-CAM heatmap unavailable: {exc}")

    # ── 6. QR cross-check ─────────────────────────────────────────────────────
    logger.info("[Module3] Running QR cross-check...")
    qr = check_qr(prep.canvas, ocr_fields)
    all_warnings.extend(qr.warnings)
    for mismatch in qr.mismatches:
        all_flags.append(f"QR_MISMATCH_{mismatch.upper()}")

    # ── 7. Composite tamper score ──────────────────────────────────────────────
    weights = _WEIGHTS_CALIBRATED if _FINE_TUNED else _WEIGHTS_UNCALIBRATED

    # Signal map: only include signals that actually produced a reading.
    signals = {
        "ela":   ela.risk_delta,
        "noise": noise.risk_delta,
    }
    if cnn_used:
        signals["cnn"] = cnn.tamper_prob
    if qr.found:
        signals["qr"] = qr.risk_delta

    used_weight = sum(weights[k] for k in signals)
    total_weight = sum(weights.values())
    scale = total_weight / used_weight if used_weight else 1.0

    tamper_score = sum(weights[k] * v for k, v in signals.items()) * scale
    tamper_score = round(min(float(tamper_score), 1.0), 4)

    # ── 8. Risk tier ──────────────────────────────────────────────────────────
    if tamper_score >= _TIER_HIGH:
        risk_tier = "HIGH"
        all_flags.append("MODULE3_HIGH_RISK")
    elif tamper_score >= _TIER_MEDIUM:
        risk_tier = "MEDIUM"
        all_flags.append("MODULE3_MEDIUM_RISK")
    else:
        risk_tier = "LOW"

    processing_ms = round((time.perf_counter() - t_start) * 1000, 1)

    logger.info(
        "[Module3] Done — tamper_score=%.4f  tier=%s  flags=%s  time=%.0fms",
        tamper_score, risk_tier, all_flags, processing_ms,
    )

    return TamperResult(
        tamper_score      = tamper_score,
        risk_tier         = risk_tier,
        heatmap_url       = gcam.heatmap_url,
        flags             = all_flags,
        ela_result        = ela,
        cnn_result        = cnn,
        noise_result      = noise,
        qr_result         = qr,
        gradcam_result    = gcam,
        preprocess_result = prep,
        processing_ms     = processing_ms,
        warnings          = all_warnings,
    )
