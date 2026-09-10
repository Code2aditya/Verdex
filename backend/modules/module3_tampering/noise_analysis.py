"""
noise_analysis.py — Sensor noise inconsistency & copy-move detection
Module 3 | VERIDEX | Raj

Principle
---------
A genuine scanned document exhibits statistically uniform camera/scanner
sensor noise across the entire image.  A forged document — where a region
has been digitally inserted, cloned, or replaced — introduces a patch with
a DIFFERENT noise signature:

  * Copy-move forgery: the noise of the pasted patch matches a different
    region of the same image (or a different image entirely).
  * Digital insertion: synthesised regions have near-zero sensor noise.
  * JPEG block boundary inconsistency: re-encoded patches produce 8×8 block
    artefacts misaligned with the rest of the image.

Approach
--------
  1. Extract residual noise via median-filter subtraction.
  2. Divide image into non-overlapping blocks (default 16×16 px).
  3. Compute local noise variance per block.
  4. Flag blocks whose variance deviates > 2σ from the image-wide mean.
  5. Generate an anomaly heat-map and a scalar risk_delta.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
_BLOCK_SIZE      = 16    # pixels per side of each analysis block
_SIGMA_THRESHOLD = 2.0   # blocks > mean + N*sigma are flagged
_MEDIAN_KSIZE    = 3     # kernel size for median filter (noise extractor)
_MIN_FLAGGED_PCT = 0.05  # at least 5% flagged blocks before raising risk


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class NoiseResult:
    """
    Output of analyze_noise().

    Attributes
    ----------
    noise_map       : float32 grayscale array — per-pixel residual noise map.
    anomaly_mask    : uint8 image (same size as canvas) — 255 where flagged.
    anomaly_overlay : BGR image with flagged blocks highlighted in red.
    flagged_blocks  : number of anomalous blocks found.
    total_blocks    : total number of blocks analysed.
    flagged_pct     : fraction of blocks flagged (0-1).
    risk_delta      : float [0-1] contribution to composite score.
    warnings        : non-fatal issues.
    """
    noise_map:       np.ndarray
    anomaly_mask:    np.ndarray
    anomaly_overlay: np.ndarray
    flagged_blocks:  int
    total_blocks:    int
    flagged_pct:     float
    risk_delta:      float
    warnings: list = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_noise(
    canvas: np.ndarray,
    gray:   np.ndarray,
    block_size: int = _BLOCK_SIZE,
) -> NoiseResult:
    """
    Run noise inconsistency analysis on a BGR document image.

    Parameters
    ----------
    canvas     : BGR numpy array (PreprocessResult.canvas).
    gray       : uint8 grayscale array (PreprocessResult.gray).
    block_size : side length of analysis blocks in pixels.

    Returns
    -------
    NoiseResult
    """
    warnings: list[str] = []
    h, w = gray.shape[:2]

    # ── 1. Extract residual noise (image − median-filtered version) ───────────
    median_filtered = cv2.medianBlur(gray, _MEDIAN_KSIZE)
    noise_map = (gray.astype(np.float32) - median_filtered.astype(np.float32))
    # Shift to non-negative for variance computation
    noise_abs = np.abs(noise_map)

    # ── 2. Block-wise local noise variance ────────────────────────────────────
    rows = h // block_size
    cols = w // block_size

    if rows == 0 or cols == 0:
        warnings.append("Image too small for block-wise noise analysis.")
        empty = np.zeros((h, w), dtype=np.uint8)
        return NoiseResult(
            noise_map=noise_abs, anomaly_mask=empty,
            anomaly_overlay=canvas.copy(),
            flagged_blocks=0, total_blocks=0, flagged_pct=0.0,
            risk_delta=0.0, warnings=warnings,
        )

    variances = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            patch = noise_abs[
                r * block_size:(r + 1) * block_size,
                c * block_size:(c + 1) * block_size,
            ]
            variances[r, c] = float(np.var(patch))

    # ── 3. Statistical flagging ───────────────────────────────────────────────
    mean_var = float(np.mean(variances))
    std_var  = float(np.std(variances))
    threshold = mean_var + _SIGMA_THRESHOLD * std_var

    anomaly_blocks = variances > threshold
    flagged = int(np.sum(anomaly_blocks))
    total   = rows * cols
    flagged_pct = flagged / total if total > 0 else 0.0

    logger.debug(
        "Noise analysis — mean_var=%.3f  std_var=%.3f  threshold=%.3f  "
        "flagged=%d/%d (%.1f%%)",
        mean_var, std_var, threshold, flagged, total, flagged_pct * 100,
    )

    # ── 4. Build anomaly mask and visual overlay ───────────────────────────────
    anomaly_mask = np.zeros((h, w), dtype=np.uint8)
    overlay      = canvas.copy()

    for r in range(rows):
        for c in range(cols):
            if anomaly_blocks[r, c]:
                y1 = r * block_size
                y2 = min(y1 + block_size, h)
                x1 = c * block_size
                x2 = min(x1 + block_size, w)
                anomaly_mask[y1:y2, x1:x2] = 255
                # Semi-transparent red highlight
                roi = overlay[y1:y2, x1:x2].astype(np.float32)
                roi[:, :, 2] = np.clip(roi[:, :, 2] * 0.5 + 127, 0, 255)
                roi[:, :, 0] = roi[:, :, 0] * 0.6
                roi[:, :, 1] = roi[:, :, 1] * 0.6
                overlay[y1:y2, x1:x2] = roi.astype(np.uint8)

    # ── 5. Risk delta ─────────────────────────────────────────────────────────
    risk = _compute_risk(flagged_pct, mean_var, std_var)

    if flagged_pct < _MIN_FLAGGED_PCT and flagged > 0:
        warnings.append(
            f"Only {flagged_pct*100:.1f}% of blocks flagged — likely noise, "
            "not a strong tampering indicator."
        )

    return NoiseResult(
        noise_map       = noise_abs,
        anomaly_mask    = anomaly_mask,
        anomaly_overlay = overlay,
        flagged_blocks  = flagged,
        total_blocks    = total,
        flagged_pct     = flagged_pct,
        risk_delta      = risk,
        warnings        = warnings,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_risk(flagged_pct: float, mean_var: float, std_var: float) -> float:
    """
    Map noise anomaly statistics to risk_delta in [0, 1].

      - flagged_pct drives the score: 0% → 0.0, 25%+ → 1.0
      - high std_var (very uneven noise) adds a small additional bump
    """
    pct_score = min(flagged_pct / 0.25, 1.0)
    # Normalise std / mean ratio (coefficient of variation) — high CV = uneven
    cv = (std_var / mean_var) if mean_var > 0 else 0.0
    cv_score = min(cv / 3.0, 1.0)

    risk = 0.75 * pct_score + 0.25 * cv_score
    return round(min(risk, 1.0), 4)
