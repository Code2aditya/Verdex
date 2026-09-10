"""Unified real-document verification pipeline — wires Modules 1–4,
the risk engine, the tamper-evident audit chain and the blockchain blob
into the single response object consumed by the Flutter officer app.

    from backend.pipeline.verify import run_verification

    result = run_verification(document_image_bytes, document_type, ...)
    # result["risk_score"], result["risk_tier"], result["ocr"], ...
"""

from __future__ import annotations

import hashlib
import logging
import time
from types import SimpleNamespace
from typing import Any

from backend.audit.chain import get_chain_root
from backend.modules.module1_ocr import run_ocr_pipeline
from backend.modules.module2_validation import validate_document, watchlist_check
from backend.modules.module3_tampering import TamperResult, run_tamper_detection
from backend.modules.module4_face import run_face_analysis
from backend.risk_engine import explain_result, fused_risk
from backend.risk_engine.scorer import REASON_CATALOG

logger = logging.getLogger(__name__)

_WEIGHTS = {
    "ocr_mrz": 0.20,
    "structure": 0.20,
    "tampering": 0.25,
    "biometrics": 0.20,
    "morphing": 0.10,
    "watchlist": 0.05,
}

FLUTTER_KEY_FIELDS = {
    "doc_number": [
        "passport_number", "aadhaar_number", "pan_number", "epic_number",
        "dl_number", "visa_number", "certificate_number",
        "document_number", "id_number", "number",
    ],
    "name": ["full_name", "name", "holder_name", "surname", "given_name"],
    "dob": ["date_of_birth", "dob"],
    "expiry": ["date_of_expiry", "validity_end", "valid_until", "expiry_date"],
    "nationality": ["nationality", "citizenship"],
    "gender": ["gender", "sex"],
}


def _canonical(fields: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for canon, aliases in FLUTTER_KEY_FIELDS.items():
        for alias in aliases:
            value = fields.get(alias)
            if value:
                out[canon] = str(value).strip()
                break
    return out


def _first(d: dict[str, Any], keys: list[str]) -> str | None:
    for k in keys:
        if d.get(k):
            return str(d[k])
    return None


def _validation_checks(report) -> list[dict[str, bool]]:
    """Index the validation report by rule name for easy flag extraction."""
    index: dict[str, bool] = {}
    for check in report.checks:
        index[check.rule] = index.get(check.rule, True) and check.passed
    return index


def _structure_signal(report) -> float:
    if report is None or report.result != "FAIL":
        return 8.0
    errors = [c for c in report.checks if c.severity == "error" and not c.passed]
    warnings = [c for c in report.checks if c.severity == "warning" and not c.passed]
    value = 45.0 + min(45.0, 12.0 * len(errors)) + min(15.0, 4.0 * len(warnings))
    return min(100.0, value)


def _ocr_signal(report, ocr_status: str) -> float:
    if ocr_status == "unavailable":
        return 5.0
    if report is None:
        return 15.0
    index = _validation_checks(report)
    if not index.get("mrz_checksum", True):
        return 90.0
    if not index.get("expiry_status", True):
        return 65.0
    for check in report.checks:
        if check.rule.startswith("date") and not check.passed:
            return 70.0
    if report.result == "FAIL":
        return 55.0
    return 6.0


def _biometric_signals(face_result: dict[str, Any]) -> tuple[float | None, float | None]:
    """Return (biometrics_risk, morphing_risk). None = channel unavailable."""
    if not face_result or face_result.get("status") != "checked":
        return None, None

    match = face_result.get("face_verification", {})
    match_score = match.get("match_score")
    base = (1.0 - float(match_score)) * 100.0 if isinstance(match_score, (int, float)) else 50.0

    liveness = face_result.get("liveness", {})
    liveness_status = liveness.get("status")
    if liveness_status == "GENUINE":
        base *= 0.4
    elif liveness_status in ("FAIL", "REPLAY", "SUSPECT"):
        base = max(base, 85.0)

    morph = face_result.get("morphing", {})
    morph_status = morph.get("status")
    morph_risk: float | None = None
    if morph_status in ("HIGH", "CRITICAL"):
        base = max(base, 90.0)
        morph_risk = 95.0
    elif morph_status == "LOW":
        morph_risk = 5.0
    elif morph_status == "SUSPECT":
        morph_risk = 60.0

    return min(100.0, base), morph_risk


def _tamper_anomalies(tamper_dict: dict[str, Any]) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    flags = tamper_dict.get("flags", [])
    signals = tamper_dict.get("signals", {})
    for flag in flags:
        if flag.upper().startswith("ELA_"):
            anomalies.append({
                "title": "ELA High Intensity",
                "description": "Error Level Analysis shows recompression artefacts consistent with local editing.",
                "severity": "HIGH" if "HIGH" in flag.upper() else "MEDIUM",
                "x": 0.5, "y": 0.5,
            })
        elif flag.upper().startswith("NOISE_"):
            anomalies.append({
                "title": "Noise Anomaly",
                "description": "Residual noise statistics diverge from the rest of the document.",
                "severity": "HIGH" if "HIGH" in flag.upper() else "MEDIUM",
                "x": 0.5, "y": 0.5,
            })
        elif flag.upper().startswith("QR_MISMATCH"):
            anomalies.append({
                "title": "QR/OCR Field Mismatch",
                "description": "The embedded barcode disagrees with the printed field. Strong tamper indicator.",
                "severity": "HIGH",
                "x": 0.5, "y": 0.5,
            })
        elif flag.upper() in ("FORGERY", "TAMPERED"):
            anomalies.append({
                "title": "Forensic Model Alert",
                "description": "CNN classifier strongly believes the image was synthetically altered.",
                "severity": "CRITICAL",
                "x": 0.5, "y": 0.5,
            })
    if not anomalies and tamper_dict.get("tamper_score", 0) >= 0.4:
        anomalies.append({
            "title": "Elevated Tamper Signal",
            "description": "Composite forensic score indicates possible document manipulation.",
            "severity": "MEDIUM",
            "x": 0.5, "y": 0.5,
        })
    return anomalies


def run_verification(
    document_bytes: bytes,
    document_type: str | None = None,
    live_bytes: bytes | None = None,
    officer_id: str = "SYSTEM",
    checkpoint_id: str = "CHK-001",
    doc_number_hint: str | None = None,
    name_hint: str | None = None,
) -> dict[str, Any]:
    """Run the full verification stack. Never raises — returns a contract dict."""
    t0 = time.perf_counter()

    # ── Module 1 — OCR ─────────────────────────────────────────────────────────
    ocr = run_ocr_pipeline(document_bytes, document_type=document_type)
    detected_type = ocr.get("document_type") or document_type or "unknown"
    fields: dict[str, Any] = ocr.get("fields", {}) or {}
    canonical = _canonical(fields)

    doc_number = doc_number_hint or canonical.get("doc_number") or _first(fields, FLUTTER_KEY_FIELDS["doc_number"])
    person_name = name_hint or canonical.get("name") or _first(fields, ["full_name", "name"])

    # ── Module 2 — Validation ──────────────────────────────────────────────────
    report = None
    if ocr.get("status") != "unavailable":
        try:
            report = validate_document(detected_type, fields, mrz_lines=ocr.get("mrz_lines"))
        except ValueError as exc:
            logger.warning("Validation skipped (%s): %s", detected_type, exc)

    watchlist = watchlist_check(detected_type, doc_number, person_name)

    validation_checks = _validation_checks(report) if report else {}

    # ── Module 3 — Tampering ───────────────────────────────────────────────────
    try:
        tamper = run_tamper_detection(document_bytes, ocr_fields=canonical)
    except Exception as exc:
        # Corrupt / undecodable image — degrade instead of crashing the officer
        # terminal. A neutral 0.5 reading keeps fraud signals OFF the record.
        logger.warning("Tamper analysis skipped (%s: %s)", type(exc).__name__, exc)
        tamper = _degraded_tamper(f"Tamper analysis skipped: {exc}")
    tamper_dict = tamper.to_api_dict()

    # ── Module 4 — Face / liveness / morphing ────────────────────────────────
    face = run_face_analysis(document_bytes, live_bytes)

    # ── Risk engine — signals over available channels ─────────────────────────
    biometrics_risk, morphing_risk = _biometric_signals(face)

    signals = {
        "structure": _structure_signal(report),
        "ocr_mrz": _ocr_signal(report, ocr.get("status", "")),
        "tampering": tamper.tamper_score * 100.0,
        "watchlist": 100.0 if watchlist.get("hit") else 0.0,
    }
    if biometrics_risk is not None:
        signals["biometrics"] = biometrics_risk
    if morphing_risk is not None:
        signals["morphing"] = morphing_risk

    available = ["structure", "ocr_mrz", "tampering", "watchlist"]
    if biometrics_risk is not None:
        available.append("biometrics")
    if morphing_risk is not None:
        available.append("morphing")

    match_result = face.get("face_verification", {})
    liveness_result = face.get("liveness", {})
    flags: dict[str, bool] = {
        "mrz_checksum_fail": not validation_checks.get("mrz_checksum", True),
        "date_logic_fail": any(
            c.rule.startswith("date") and not c.passed for c in (report.checks if report else [])
        ),
        "face_mismatch": match_result.get("status") == "NOMATCH",
        "liveness_fail": liveness_result.get("status") in ("FAIL", "REPLAY", "SUSPECT"),
        "watchlist_hit": bool(watchlist.get("hit")),
        "morph_detected": face.get("morphing", {}).get("status") in ("HIGH", "CRITICAL"),
    }

    risk = fused_risk(
        signals,
        flags=flags,
        document_type=detected_type,
        traveller_name=person_name or fields.get("full_name") or "UNKNOWN",
        available=available,
    )
    explanation = explain_result(risk)

    # ── Blockchain blob (hash-chain root + simulated Sepolia anchor) ─────────
    chain_root = get_chain_root()
    tx_hash = "0x" + hashlib.sha256(f"sepolia-sim:{chain_root}".encode()).hexdigest()
    blockchain = {
        "anchored": True,
        "tx_hash": tx_hash,
        "sha256": f"SHA256: {chain_root}",
        "enclave_digest": f"HC:{chain_root[:16].upper()}",
        "chain_root": chain_root,
        "network": "sepolia-simulated",
        "note": "Hash-chain root anchored (simulated). Set WEB3_RPC_URL + WEB3_PRIVATE_KEY for a live Sepolia tx.",
    }

    ocr_detail = (
        f"OCR engine parsed {ocr.get('line_count', 0)} line(s); detected type '{detected_type}'; "
        f"confidence {ocr.get('confidence', 0.0):.2f}."
        if ocr.get("status") != "unavailable"
        else "OCR engine unavailable on this deployment — field extraction skipped."
    )

    validation_detail = (
        f"Module 2 ran {len(report.checks)} rule checks → {report.result}."
        if report is not None
        else "Validation skipped — no OCR fields available."
    )

    tamper_passed = tamper.tamper_score < 0.40
    tamper_detail = (
        f"Composite forensic score {tamper.tamper_score:.2f}; tier {tamper.risk_tier}. "
        f"Signals: ELA {tamper_dict['signals']['ela']['risk_delta']:.2f}, "
        f"noise {tamper_dict['signals']['noise']['risk_delta']:.2f}, "
        f"QR found={tamper_dict['signals']['qr']['found']}."
    )

    face_status = match_result.get("status", "NOT_RUN")
    face_passed = face_status == "MATCH"
    liveness_passed = liveness_result.get("status") == "GENUINE"

    contract = {
        "risk_score": int(risk["score"]),
        "risk_tier": risk["level"],
        "risk": risk,
        "explanation": explanation,
        "document_type": detected_type,
        "processing_ms": round((time.perf_counter() - t0) * 1000, 1),
        "ocr": {
            "status": ocr.get("status", "unavailable"),
            "confidence": ocr.get("confidence", 0.0),
            "passed": ocr.get("status") != "unavailable" and not flags["mrz_checksum_fail"],
            "details": ocr_detail,
            "document_type": detected_type,
            "fields": fields,
            "mrz_lines": ocr.get("mrz_lines", []),
            "line_count": ocr.get("line_count", 0),
        },
        "validation": {
            "status": report.result if report else "NOT_RUN",
            "confidence": 0.92 if report and report.result == "PASS" else (0.4 if report else 0.0),
            "passed": bool(report and report.result == "PASS"),
            "details": validation_detail,
            "checks": [c.__dict__ for c in report.checks] if report else [],
        },
        "tamper": {
            "confidence": round(1.0 - tamper.tamper_score, 4),
            "passed": tamper_passed,
            "details": tamper_detail,
            "score": tamper.tamper_score,
            "tier": tamper.risk_tier,
            "anomalies": _tamper_anomalies(tamper_dict),
            "signals": tamper_dict["signals"],
            "flags": tamper_dict["flags"],
            "heatmap_url": tamper_dict.get("heatmap_url", ""),
            "processing_ms": tamper_dict.get("processing_ms"),
        },
        "face": {
            "status": face_status,
            "match_score": match_result.get("match_score"),
            "passed": face_passed,
            "details": match_result.get("details", "Face verification skipped."),
            "method": match_result.get("method"),
            "document_faces": face.get("document_faces", 0),
            "live_faces": face.get("live_faces", 0),
        },
        "liveness": {
            "status": liveness_result.get("status", "NOT_RUN"),
            "score": liveness_result.get("score"),
            "passed": liveness_passed,
            "details": liveness_result.get("details", "Liveness check skipped."),
        },
        "watchlist": watchlist,
        "blockchain": blockchain,
        "audit": {
            "officer_id": officer_id,
            "checkpoint_id": checkpoint_id,
            "doc_number": doc_number,
            "decision": _decision_for(risk["level"]),
        },
    }
    return contract


def _degraded_tamper(error: str) -> TamperResult:
    """Minimal TamperResult used when the tamper stack cannot decode the image."""

    def stub(**attrs):  # noqa: ANN001, ANN202
        return SimpleNamespace(**attrs)

    ela = stub(risk_delta=0.5, mean_intensity=0.0, max_intensity=0.0, warnings=[error])
    cnn = stub(tamper_prob=0.5, genuine_prob=0.5, fine_tuned=False)
    noise = stub(risk_delta=0.5, flagged_pct=0.0, flagged_blocks=0, total_blocks=0)
    qr = stub(found=False, mismatches=[], risk_delta=0.0, explanation="Image could not be decoded.")
    gcam = stub(heatmap_url="", warnings=[])
    prep = stub(warnings=[error])
    return TamperResult(
        tamper_score=0.5,
        risk_tier="MEDIUM",
        heatmap_url="",
        flags=["QUALITY_INSUFFICIENT"],
        ela_result=ela,
        cnn_result=cnn,
        noise_result=noise,
        qr_result=qr,
        gradcam_result=gcam,
        preprocess_result=prep,
        processing_ms=0.0,
        warnings=[error],
    )


def _decision_for(level: str) -> str:
    if level in ("CRITICAL", "HIGH"):
        return "ALERT"
    if level == "MEDIUM":
        return "REVIEW"
    return "CLEAR"


def risk_reason_help(code: str) -> dict[str, str]:
    return REASON_CATALOG.get(code, {"message": code, "detail": "Unknown reason code."})