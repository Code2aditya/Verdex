"""
gradcam.py — Grad-CAM heatmap generation for Module 3
Module 3 | VERIDEX | Raj

Produces a visual heatmap overlay on the document image, highlighting the
regions the CNN considers most suspicious.  This is the explainability
artefact shown to the border officer ("why did VERIDEX flag this document?").

Uses the pytorch-grad-cam library (older CAM-based API installed in venv).
Falls back to a manual gradient hook implementation if the library API is
unavailable.

Output
------
  - A JPEG heatmap saved under <static_dir>/heatmaps/<uuid>.jpg
  - heatmap_url returned as "/static/heatmaps/<uuid>.jpg"
  - heatmap_array: the BGR overlay as a numpy array (same size as canvas)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Static file directory ─────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_STATIC_DIR   = _PROJECT_ROOT / "backend" / "static" / "heatmaps"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class GradCAMResult:
    """
    Output of generate_heatmap().

    Attributes
    ----------
    heatmap_url   : URL path to the saved JPEG overlay (e.g. /static/…).
    heatmap_array : BGR numpy array — heatmap overlaid on document canvas.
    cam_raw       : float32 (H, W) normalised activation map before overlay.
    warnings      : non-fatal issues.
    """
    heatmap_url:   str
    heatmap_array: np.ndarray
    cam_raw:       np.ndarray
    warnings: list = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_heatmap(
    model:     "torch.nn.Module",
    cnn_input: np.ndarray,
    canvas:    np.ndarray,
) -> GradCAMResult:
    """
    Generate a Grad-CAM heatmap for the given document image.

    Parameters
    ----------
    model     : The EfficientNet model from cnn_classifier.get_model().
    cnn_input : float32 (224, 224, 3) RGB numpy array in [0, 1] —
                same tensor fed to classify().
    canvas    : BGR numpy array of the full-resolution document canvas —
                used as the background for the overlay.

    Returns
    -------
    GradCAMResult
    """
    warnings: list[str] = []

    try:
        cam_raw = _grad_cam(model, cnn_input)
    except Exception as exc:
        logger.warning("Grad-CAM failed (%s) — falling back to zero map.", exc)
        cam_raw = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=np.float32)
        warnings.append(f"Grad-CAM computation failed: {exc}")

    # ── Overlay on canvas ─────────────────────────────────────────────────────
    heatmap_array = _overlay(canvas, cam_raw)

    # ── Save to disk ──────────────────────────────────────────────────────────
    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = _STATIC_DIR / filename
    cv2.imwrite(str(save_path), heatmap_array, [cv2.IMWRITE_JPEG_QUALITY, 92])
    heatmap_url = f"/static/heatmaps/{filename}"

    logger.debug("Heatmap saved → %s", save_path)

    return GradCAMResult(
        heatmap_url   = heatmap_url,
        heatmap_array = heatmap_array,
        cam_raw       = cam_raw,
        warnings      = warnings,
    )


# ── Internal — Grad-CAM computation ──────────────────────────────────────────

def _grad_cam(model: "torch.nn.Module", cnn_input: np.ndarray) -> np.ndarray:
    """
    Manual Grad-CAM implementation using PyTorch hooks.

    Hooks into the last convolutional layer of EfficientNet-B0
    (model.features[-1]) to capture:
      - forward activations  A  (shape: C x H' x W')
      - backward gradients   G  (shape: C x H' x W')

    Grad-CAM formula:
      weights[k] = global_avg_pool(G[k])
      L = ReLU( Σ_k  weights[k] * A[k] )

    Returns a float32 (H_canvas x W_canvas) heatmap normalised to [0, 1].
    """
    from . import cnn_classifier as cnn

    device = cnn._DEVICE
    target_layer = cnn.get_target_layer()

    activations: list[np.ndarray] = []
    gradients:   list[np.ndarray] = []

    def _fwd_hook(module, inp, out):
        activations.append(out.detach().cpu().numpy())

    def _bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach().cpu().numpy())

    fwd_handle = target_layer.register_forward_hook(_fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(_bwd_hook)

    try:
        tensor = cnn._to_tensor(cnn_input)  # (1, 3, 224, 224)
        tensor.requires_grad_(True)

        model.zero_grad()
        output = model(tensor)               # (1, 2)

        # Backprop w.r.t. the tampered class (index 1)
        class_score = output[0, 1]
        class_score.backward()

        A = activations[0][0]   # (C, H', W')
        G = gradients[0][0]     # (C, H', W')

        # Global average pooling of gradients → per-channel weights
        weights = G.mean(axis=(1, 2))             # (C,)
        cam = np.einsum("k,khw->hw", weights, A)  # (H', W')
        cam = np.maximum(cam, 0)                  # ReLU

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    return cam.astype(np.float32)


def _overlay(canvas: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """
    Overlay a Grad-CAM activation map on the document canvas image.

    Steps
    -----
      1. Upsample cam to canvas size.
      2. Apply COLORMAP_JET (blue=clean, red=suspicious).
      3. Blend with canvas (70% canvas, 30% heatmap) for legibility.

    Returns BGR overlay as uint8.
    """
    h, w = canvas.shape[:2]

    # Upsample activation map
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert to colourmap
    cam_uint8   = (cam_resized * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # Blend
    overlay = cv2.addWeighted(canvas, 0.65, heatmap_bgr, 0.35, 0)
    return overlay
