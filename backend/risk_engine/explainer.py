"""Risk score explainability — per-channel contributions and officer notes."""

from __future__ import annotations

from typing import Any

from backend.risk_engine.scorer import SIGNAL_LABELS, WEIGHTS, REASON_CATALOG


def explain_result(result: dict[str, Any]) -> dict[str, Any]:
    """Build a human-readable breakdown of a risk result."""
    signals = result.get("signals", {})
    contribution_rows = []
    total_points = 0.0
    for key in WEIGHTS:
        value = signals.get(key, 0.0)
        points = round(value * WEIGHTS[key], 1)
        total_points += points
        contribution_rows.append(
            {
                "key": key,
                "label": SIGNAL_LABELS.get(key, key),
                "signal_score": round(value, 1),
                "weight": WEIGHTS[key],
                "contribution_points": points,
            }
        )

    reasons = []
    for reason in result.get("reasons", []):
        code = reason.get("code", "")
        entry = REASON_CATALOG.get(code, {})
        reasons.append(
            {
                "code": code,
                "severity": reason.get("severity"),
                "message": reason.get("message", entry.get("message")),
                "detail": reason.get("detail", entry.get("detail")),
            }
        )

    return {
        "weighted_points": round(total_points, 1),
        "declared_score": result.get("score"),
        "level": result.get("level"),
        "contributions": contribution_rows,
        "reasons": reasons,
        "escalations": result.get("escalations", []),
        "channels_available": result.get("channels_available"),
        "channels_omitted": result.get("channels_omitted"),
        "action": result.get("action"),
    }