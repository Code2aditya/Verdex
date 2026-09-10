"""Risk engine tests — presets, thresholds, partial fusion."""

from __future__ import annotations

import pytest

from backend.risk_engine import compute_risk, evaluate_preset, fused_risk, list_presets
from backend.risk_engine.scorer import PRESETS, classify_threat


def test_preset_d_watchlist_is_critical() -> None:
    result = evaluate_preset("D")
    assert result["level"] == "CRITICAL"
    assert result["score"] >= 85
    assert any(r["code"] == "WATCHLIST_HIT" for r in result["reasons"])


def test_preset_c_morph_is_high() -> None:
    result = evaluate_preset("C")
    assert result["level"] == "HIGH"
    assert 60 <= result["score"] < 85
    assert any(r["code"] == "MORPH_DMAD" for r in result["reasons"])


def test_preset_a_clean_is_low() -> None:
    result = evaluate_preset("A")
    assert result["level"] == "LOW"
    assert result["score"] < 30


def test_preset_e_runs_and_flags_demo() -> None:
    result = evaluate_preset("E")
    assert result["level"] == "LOW"
    assert result["preset"]["triggers_tamper_demo"] is True


def test_unknown_preset_raises() -> None:
    with pytest.raises(KeyError):
        evaluate_preset("Z")


def test_list_presets_has_five() -> None:
    presets = list_presets()
    assert len(presets) == 5
    assert {p["id"] for p in presets} == set(PRESETS)


def test_classify_threat_boundaries() -> None:
    assert classify_threat(85)["level"] == "CRITICAL"
    assert classify_threat(60)["level"] == "HIGH"
    assert classify_threat(30)["level"] == "MEDIUM"
    assert classify_threat(0)["level"] == "LOW"


def test_fused_risk_omits_missing_channels() -> None:
    result = fused_risk(
        {"tampering": 10.0, "watchlist": 0.0},
        document_type="passport",
        available=["tampering", "watchlist"],
    )
    assert result["partial_verification"] is True
    assert "biometrics" in result["channels_omitted"]
    assert "structure" in result["channels_omitted"]
    assert result["channels_available"] == ["tampering", "watchlist"]
    assert 0 <= result["score"] <= 100


def test_compute_risk_quality_insufficient_is_inconclusive() -> None:
    result = compute_risk(
        {"structure": 100.0},
        flags={"quality_insufficient": True},
        document_type="passport",
    )
    assert result["level"] == "INCONCLUSIVE"
    assert result["score"] == 0.0


def test_compute_risk_escalation_on_watchlist() -> None:
    result = compute_risk(
        {"structure": 0.0, "ocr_mrz": 0.0, "tampering": 0.0, "biometrics": 0.0, "morphing": 0.0, "watchlist": 100.0},
        flags={"watchlist_hit": True},
        document_type="passport",
    )
    assert result["level"] == "CRITICAL"
    assert result["escalations"]