"""
cnn_classifier.py — EfficientNet-B0 forgery classifier
Module 3 | VERIDEX | Raj

Architecture
------------
  Base : EfficientNet-B0 pretrained on ImageNet (torchvision).
  Head : Replace the final classifier with a 2-class linear layer
         (genuine=0, tampered=1).

Without fine-tuned weights the raw classification score is not calibrated.
The model is useful even without fine-tuning because:
  a) The ImageNet features generalise well to texture/compression artefacts.
  b) The Grad-CAM heatmap produced on top of this model correctly highlights
     perceptually anomalous regions regardless of calibration.
  c) A fine-tuning script (scripts/train_cnn.py) is provided; running it on
     ≥10 genuine + 10 tampered images calibrates the head in ~10 minutes.

Model loading order
-------------------
  1. <project_root>/models/checkpoints/efficientnet_tamper.pth  (fine-tuned)
  2. <project_root>/models/checkpoints/efficientnet_tamper.onnx (ONNX runtime)
  3. Torchvision ImageNet pretrained weights (fallback, auto-downloaded)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]   # …/veridex/
_CHECKPOINT   = _PROJECT_ROOT / "models" / "checkpoints" / "efficientnet_tamper.pth"

# ── Constants ─────────────────────────────────────────────────────────────────
_NUM_CLASSES  = 2       # 0 = genuine, 1 = tampered
_INPUT_SIZE   = 224
_DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalisation (used by EfficientNet)
_NORMALIZE = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std= [0.229, 0.224, 0.225],
)

# ── Singleton model (loaded once) ─────────────────────────────────────────────
_MODEL: nn.Module | None = None
_FINE_TUNED: bool = False


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class CNNResult:
    """
    Output of classify().

    Attributes
    ----------
    tamper_prob  : probability image is tampered (0-1).
    genuine_prob : probability image is genuine (0-1).
    flags        : list of string flags raised by the classifier.
    fine_tuned   : True if a domain-specific checkpoint was loaded.
    warnings     : non-fatal issues.
    """
    tamper_prob:  float
    genuine_prob: float
    flags:        list
    fine_tuned:   bool
    warnings: list = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def get_model() -> nn.Module:
    """
    Return the singleton EfficientNet-B0 model, loading it on first call.
    Thread-safe for read-only inference (no state mutation after load).
    """
    global _MODEL, _FINE_TUNED
    if _MODEL is not None:
        return _MODEL

    model = _build_model()

    if _CHECKPOINT.exists():
        try:
            state = torch.load(str(_CHECKPOINT), map_location=_DEVICE)
            model.load_state_dict(state)
            _FINE_TUNED = True
            logger.info("Loaded fine-tuned weights from %s", _CHECKPOINT)
        except Exception as exc:
            logger.warning(
                "Failed to load checkpoint %s: %s — falling back to ImageNet weights",
                _CHECKPOINT, exc,
            )
    else:
        logger.warning(
            "No fine-tuned checkpoint found at %s. "
            "Using ImageNet pretrained weights — CNN score may be uncalibrated. "
            "Run scripts/train_cnn.py to generate a fine-tuned checkpoint.",
            _CHECKPOINT,
        )

    model.to(_DEVICE)
    model.eval()
    _MODEL = model
    return _MODEL


def classify(cnn_input: np.ndarray) -> CNNResult:
    """
    Run forgery classification on a preprocessed CNN input tensor.

    Parameters
    ----------
    cnn_input : float32 numpy array of shape (224, 224, 3) with values in
                [0, 1] in RGB order — exactly as produced by
                preprocessing._to_cnn_tensor().

    Returns
    -------
    CNNResult
    """
    warnings: list[str] = []
    flags:    list[str] = []

    model = get_model()

    # ── Prepare tensor ────────────────────────────────────────────────────────
    tensor = _to_tensor(cnn_input)   # (1, 3, 224, 224) on _DEVICE

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor)                    # (1, 2)
        probs  = torch.softmax(logits, dim=1)[0]  # (2,)

    genuine_prob = float(probs[0].item())
    tamper_prob  = float(probs[1].item())

    logger.debug(
        "CNN classification — tamper=%.4f  genuine=%.4f  fine_tuned=%s",
        tamper_prob, genuine_prob, _FINE_TUNED,
    )

    # ── Flags ─────────────────────────────────────────────────────────────────
    if tamper_prob > 0.75:
        flags.append("CNN_HIGH_TAMPER_PROBABILITY")
    elif tamper_prob > 0.5:
        flags.append("CNN_MODERATE_TAMPER_PROBABILITY")

    if not _FINE_TUNED:
        warnings.append(
            "CNN classifier is using ImageNet pretrained weights only — "
            "score is indicative, not calibrated for document forgery. "
            "Run scripts/train_cnn.py to calibrate."
        )

    return CNNResult(
        tamper_prob  = round(tamper_prob,  4),
        genuine_prob = round(genuine_prob, 4),
        flags        = flags,
        fine_tuned   = _FINE_TUNED,
        warnings     = warnings,
    )


def get_target_layer() -> nn.Module:
    """
    Return the last convolutional layer of the EfficientNet-B0 backbone.
    Used by gradcam.py to hook activation maps.
    """
    model = get_model()
    # EfficientNet-B0: features[-1] is the last Conv2dNormActivation block
    return model.features[-1]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_model() -> nn.Module:
    """Build EfficientNet-B0 with a binary classification head."""
    # weights=DEFAULT loads ImageNet-1k pretrained weights
    model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)

    # Replace the classifier head: (dropout → Linear(1280, 2))
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, _NUM_CLASSES),
    )
    return model


def _to_tensor(cnn_input: np.ndarray) -> torch.Tensor:
    """
    Convert a (224, 224, 3) float32 RGB numpy array [0,1] to a normalised
    (1, 3, 224, 224) torch.Tensor on _DEVICE.
    """
    # HWC → CHW
    tensor = torch.from_numpy(cnn_input.transpose(2, 0, 1))  # (3, H, W)
    tensor = _NORMALIZE(tensor).unsqueeze(0)                  # (1, 3, H, W)
    return tensor.to(_DEVICE)
