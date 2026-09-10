"""risk_engine — weighted multi-modal risk scoring + explainability."""

from __future__ import annotations

from backend.risk_engine.explainer import explain_result
from backend.risk_engine.scorer import (
    SIGNAL_LABELS,
    WEIGHTS,
    classify_threat,
    compute_risk,
    evaluate_preset,
    flags_from_signals,
    fused_risk,
    list_presets,
)

__all__ = [
    "WEIGHTS",
    "SIGNAL_LABELS",
    "classify_threat",
    "compute_risk",
    "fused_risk",
    "flags_from_signals",
    "explain_result",
    "evaluate_preset",
    "list_presets",
]