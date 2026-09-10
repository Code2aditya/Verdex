"""
preprocessing.py — Document image preprocessing pipeline
Module 3 | VERIDEX | Anubhav

Responsibilities:
  - Load image from file path, bytes, or numpy array
  - Auto-rotate / deskew via Hough-line voting
  - Denoise  (bilateral filter + fastNlMeansDenoisingColored)
  - Adaptive histogram equalisation (CLAHE) for CNN contrast boost
  - Resize to a standard canvas without squashing aspect ratio (letterbox)
  - Normalise pixel values to float32 [0,1] for CNN input
  - Return a PreprocessResult dataclass consumed by all downstream modules
    (ELA, CNN classifier, QR check, noise analysis, Grad-CAM)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_WIDTH   = 1200         # standard canvas width  (pixels)
TARGET_HEIGHT  = 850          # standard canvas height (pixels)
CNN_SIZE       = (224, 224)   # input size for EfficientNet / ResNet
DESKEW_MAX_ANGLE = 45.0       # reject deskew estimates beyond ±45°
PAD_COLOR      = (255, 255, 255)  # white padding for letterbox


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class PreprocessResult:
    """
    All intermediate image representations needed by Module-3 sub-components.
    Every field is a numpy array in BGR colour order (matching OpenCV convention)
    EXCEPT cnn_input which is float32 RGB [0,1] ready for torch/ONNX.
    """
    original:   np.ndarray   # BGR — exact copy of what was loaded
    canvas:     np.ndarray   # BGR — resized + letterboxed to TARGET_W x TARGET_H
    gray:       np.ndarray   # uint8 grayscale of canvas
    denoised:   np.ndarray   # BGR bilateral + NlMeans denoised canvas
    cnn_input:  np.ndarray   # float32 RGB (224,224,3) in [0,1]
    skew_angle: float = 0.0  # degrees corrected by deskew (+ = clockwise)
    width:  int = 0
    height: int = 0
    warnings: list = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess(
    source: Union[str, Path, bytes, bytearray, np.ndarray, Image.Image],
) -> PreprocessResult:
    """
    Full preprocessing pipeline.

    Accepts any of:
      - str / Path        -> file path on disk
      - bytes / bytearray -> raw image bytes (JPEG, PNG, BMP ...)
      - np.ndarray        -> already-loaded BGR image (cv2.imread output)
      - PIL.Image         -> RGB PIL image

    Returns a PreprocessResult with all representations ready
    for ELA, CNN, QR, noise and face modules.

    Raises
    ------
    ValueError  if the source cannot be decoded as an image.
    TypeError   if the source type is unsupported.
    """
    original = _load(source)
    if original is None or original.size == 0:
        raise ValueError("Could not decode image from the provided source.")

    warnings = []

    # ── 1. Deskew ─────────────────────────────────────────────────────────────
    deskewed, angle = _deskew(original)
    if abs(angle) > DESKEW_MAX_ANGLE:
        warnings.append(
            f"Deskew estimate ({angle:.1f}) exceeds +-{DESKEW_MAX_ANGLE} — "
            "skipped to avoid distortion."
        )
        deskewed = original
        angle    = 0.0
    elif abs(angle) > 1.0:
        logger.debug("Deskew applied: %.2f degrees", angle)

    # ── 2. Letterbox to standard canvas ───────────────────────────────────────
    canvas = _letterbox(deskewed, TARGET_WIDTH, TARGET_HEIGHT)

    # ── 3. Grayscale ──────────────────────────────────────────────────────────
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # ── 4. Denoise ────────────────────────────────────────────────────────────
    denoised = _denoise(canvas)

    # ── 5. CNN input tensor ───────────────────────────────────────────────────
    cnn_input = _to_cnn_tensor(denoised)

    h, w = canvas.shape[:2]
    return PreprocessResult(
        original   = original,
        canvas     = canvas,
        gray       = gray,
        denoised   = denoised,
        cnn_input  = cnn_input,
        skew_angle = angle,
        width      = w,
        height     = h,
        warnings   = warnings,
    )


# ── Utility converters (used by other modules) ────────────────────────────────

def to_pil(image: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array to a PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def from_pil(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to a BGR numpy array."""
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def load_from_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes to a BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes.")
    return img


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load(source) -> np.ndarray | None:
    """Dispatch to the appropriate loader based on source type."""
    if isinstance(source, np.ndarray):
        return source.copy()
    if isinstance(source, Image.Image):
        return from_pil(source)
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if img is None:
            logger.error("cv2.imread returned None for path: %s", source)
        return img
    if isinstance(source, (bytes, bytearray)):
        arr = np.frombuffer(source, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    raise TypeError(f"Unsupported source type: {type(source).__name__}")


def _deskew(img: np.ndarray) -> tuple:
    """
    Estimate and correct document skew using Hough-line voting on a
    Canny edge map of the grayscale image.

    Algorithm:
      1. Gaussian blur to reduce noise before edge detection.
      2. Canny edges.
      3. Probabilistic Hough lines -- long lines only (>= quarter image width).
      4. Compute angle of each line; take the median.
      5. Rotate image by -median_angle around the image centre.

    Falls back to (original, 0.0) if no clear dominant angle is found.
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150, apertureSize=3)

    min_line_len = img.shape[1] // 4
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_len,
        maxLineGap=20,
    )

    if lines is None:
        return img, 0.0

    angles = []
    for line in lines:
        line = np.asarray(line).reshape(-1)
        if line.size < 4:
            continue
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        if abs(angle) < DESKEW_MAX_ANGLE:
            angles.append(angle)

    if not angles:
        return img, 0.0

    median_angle = float(np.median(angles))

    # Skip negligible angles — warpAffine is not free
    if abs(median_angle) < 0.5:
        return img, 0.0

    h, w   = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M      = cv2.getRotationMatrix2D((cx, cy), median_angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, median_angle


def _letterbox(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Resize img to fit inside (target_w x target_h) while preserving
    the original aspect ratio, then pad with white to fill the canvas.

    This ensures every downstream module receives images of identical
    spatial dimensions without geometric distortion.
    """
    h, w  = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w, 3), PAD_COLOR, dtype=np.uint8)
    x_off  = (target_w - new_w) // 2
    y_off  = (target_h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def _denoise(img: np.ndarray) -> np.ndarray:
    """
    Two-stage denoising that preserves text legibility:

    Stage 1 — Bilateral filter (d=9, sigma=75):
        Smooths flat regions while keeping sharp text/line edges.

    Stage 2 — fastNlMeansDenoisingColored (h=6):
        Removes colour noise introduced by scanner/camera sensor.
    """
    bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    denoised  = cv2.fastNlMeansDenoisingColored(bilateral, None, 6, 6, 7, 21)
    return denoised


def _to_cnn_tensor(img: np.ndarray) -> np.ndarray:
    """
    Produce a float32 RGB array of shape (224, 224, 3) with pixel values
    in [0, 1], suitable for direct use as a CNN input.

    Pipeline:
      1. CLAHE on the L channel of LAB colour space to boost local contrast
         -- helps the CNN see subtle ink/compression artefacts.
      2. Resize to CNN_SIZE using area interpolation.
      3. Convert BGR -> RGB (PyTorch / ONNX convention).
      4. Cast to float32 and scale to [0, 1].
    """
    # CLAHE contrast enhancement in LAB colour space
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Resize and convert to RGB float
    resized = cv2.resize(enhanced, CNN_SIZE, interpolation=cv2.INTER_AREA)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0)
