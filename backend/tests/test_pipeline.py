"""Pipeline tests — run_verification never raises and matches the Flutter contract."""

from __future__ import annotations

from backend.pipeline.verify import run_verification

CONTRACT_KEYS = {
    "risk_score",
    "risk_tier",
    "ocr",
    "validation",
    "tamper",
    "face",
    "liveness",
    "blockchain",
    "watchlist",
    "risk",
    "explanation",
    "audit",
}


def test_contract_keys_present(sample_document_bytes) -> None:
    result = run_verification(sample_document_bytes, document_type="aadhaar")
    assert CONTRACT_KEYS.issubset(result)
    assert isinstance(result["risk_score"], int)
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_tier"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL", "INCONCLUSIVE")
    assert result["blockchain"]["anchored"] is True
    assert result["blockchain"]["network"] == "sepolia-simulated"


def test_with_empty_bytes_never_raises() -> None:
    result = run_verification(b"", document_type="passport")
    assert result["risk_score"] in range(0, 101)


def test_aadhaar_hint_surfaces_document_type(sample_document_bytes) -> None:
    result = run_verification(sample_document_bytes, document_type="aadhaar")
    assert result["document_type"] == "aadhaar"


def test_officer_metadata_flows_into_audit(sample_document_bytes) -> None:
    result = run_verification(
        sample_document_bytes,
        document_type="pan_card",
        officer_id="SSB-OP-1042",
        checkpoint_id="CHK-004",
    )
    assert result["audit"]["officer_id"] == "SSB-OP-1042"
    assert result["audit"]["checkpoint_id"] == "CHK-004"