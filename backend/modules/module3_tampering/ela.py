"""
ela.py — Error Level Analysis (ELA) for document tampering detection
Module 3 | VERIDEX | Raj

Principle
---------
JPEG compression is lossy and idempotent at a given quality level: re-saving
an unedited JPEG at the same quality produces the same bitstream.  An edited
region was recompressed at a different local quality, so when we re-save the
WHOLE image at a known quality and subtract, edited patches show higher
difference (higher "error level") than untouched regions.

Pipeline
--------
  1. Re-save the input image as JPEG at a known quality level (default 90).
  2. Reload the re-saved version.
  3. Compute per-pixel absolute difference (amplified for visibility).
  4. Derive scalar statistics: mean intensity, max intensity, std.
  5. Compute a risk_delta [0-1] from the statistics.
  6. Return ELAResult.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
_DEFAULT_QUALITY   = 90    # JPEG quality used for the reference re-save
_AMPLIFY           = 15    # multiply difference to make subtle artefacts visible
_HIGH_RISK_THRESH  = 25.0  # mean ELA intensity above this → high risk
_MED_RISK_THRESH   = 12.0  # mean ELA intensity above this → medium risk


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class ELAResult:
    """
    Output of run_ela().

    Attributes
    ----------
    ela_map         : uint8 BGR image — amplified pixel-difference map, same
                      size as the input canvas. Bright regions = likely edits.
    ela_gray        : uint8 grayscale version of ela_map (H x W).
    mean_intensity  : average pixel intensity of ela_gray (0-255).
    max_intensity   : maximum pixel intensity of ela_gray (0-255).
    std_intensity   : std-dev of ela_gray pixel intensities.
    risk_delta      : float [0-1] contribution to Module-3 composite score.
    quality_used    : JPEG quality level used for re-save.
    warnings        : Non-fatal issues encountered.
    """
    ela_map:        np.ndarray
    ela_gray:       np.ndarray
    mean_intensity: float
    max_intensity:  float
    std_intensity:  float
    risk_delta:     float
    quality_used:   int
    warnings: list = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def run_ela(
    image: np.ndarray,
    quality: int = _DEFAULT_QUALITY,
) -> ELAResult:
    """
    Run Error Level Analysis on a BGR numpy image.

    Parameters
    ----------
    image   : BGR numpy array (e.g. PreprocessResult.canvas).
    quality : JPEG re-save quality (1-95).  Lower values amplify differences
              but increase false positives on already highly-compressed inputs.

    Returns
    -------
    ELAResult
    """
    warnings: list[str] = []
    quality = max(1, min(95, quality))

    # ── 1. Convert BGR → RGB PIL image ───────────────────────────────────────
    pil_orig = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # ── 2. Re-save at known quality into an in-memory buffer ─────────────────
    buf = io.BytesIO()
    pil_orig.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    pil_resaved = Image.open(buf)
    pil_resaved.load()   # force decode before buffer goes out of scope

    # ── 3. Pixel-wise absolute difference (amplified) ─────────────────────────
    orig_arr    = np.array(pil_orig,    dtype=np.float32)
    resaved_arr = np.array(pil_resaved, dtype=np.float32)

    diff = np.abs(orig_arr - resaved_arr) * _AMPLIFY
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    # Convert back to BGR for consistent OpenCV convention
    ela_map  = cv2.cvtColor(diff, cv2.COLOR_RGB2BGR)
    ela_gray = cv2.cvtColor(ela_map, cv2.COLOR_BGR2GRAY)

    # ── 4. Scalar statistics ──────────────────────────────────────────────────
    mean_i = float(np.mean(ela_gray))
    max_i  = float(np.max(ela_gray))
    std_i  = float(np.std(ela_gray))

    logger.debug(
        "ELA stats — mean=%.2f  max=%.2f  std=%.2f  quality=%d",
        mean_i, max_i, std_i, quality,
    )

    # ── 5. Risk delta ─────────────────────────────────────────────────────────
    risk = _compute_risk(mean_i, max_i, std_i)

    # Warn if the input looks like it was already heavily compressed — ELA is
    # less reliable in that case.
    if mean_i < 2.0:
        warnings.append(
            "ELA mean intensity is very low — the image may already have been "
            "compressed multiple times or is a PNG/lossless format. "
            "ELA reliability is reduced."
        )

    return ELAResult(
        ela_map        = ela_map,
        ela_gray       = ela_gray,
        mean_intensity = mean_i,
        max_intensity  = max_i,
        std_intensity  = std_i,
        risk_delta     = risk,
        quality_used   = quality,
        warnings       = warnings,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_risk(mean_i: float, max_i: float, std_i: float) -> float:
    """
    Map ELA statistics to a risk delta in [0, 1].

    Heuristic thresholds calibrated on a mix of genuine and synthetically
    tampered document images:
      - mean_intensity is the primary driver (broad editing leaves high avg)
      - max_intensity catches localised single-field edits (high peak, low avg)
      - std_intensity penalises uneven distributions (genuine images are flat)
    """
    # Normalise each component to [0, 1]
    mean_score = min(mean_i / _HIGH_RISK_THRESH, 1.0)
    max_score  = min(max_i  / 180.0, 1.0)
    std_score  = min(std_i  / 40.0,  1.0)

    # Weighted combination
    risk = 0.50 * mean_score + 0.30 * max_score + 0.20 * std_score
    return round(min(risk, 1.0), 4)
