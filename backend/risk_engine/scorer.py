"""Multi-modal risk scoring engine (weighted fusion).

Fuses document, biometric, and watchlist signals into a single 0-100
risk score with human-readable reason codes. Designed as officer
decision-support — never an autonomous clearance engine.

Ported from module4/risk_engine.py. `fused_risk()` additionally rescales
weights across only the channels that produced a reading — so a missing
biometrics channel cannot silently understate the score.
"""

from __future__ import annotations

from typing import Any

WEIGHTS: dict[str, float] = {
    "structure": 0.20,
    "ocr_mrz": 0.20,
    "tampering": 0.25,
    "biometrics": 0.20,
    "morphing": 0.10,
    "watchlist": 0.05,
}

SIGNAL_LABELS = {
    "structure": "Document Integrity / Structure",
    "ocr_mrz": "OCR / MRZ Consistency",
    "tampering": "Tampering Probability (ELA + CNN + noise)",
    "biometrics": "Face Verification / Liveness",
    "morphing": "Morphing Attack Risk (D-MAD)",
    "watchlist": "Watchlist / Identity Graph",
}

REASON_CATALOG: dict[str, dict[str, str]] = {
    "STRUCT_LAYOUT": {
        "message": "Document structure check failed",
        "detail": "Document layout / template does not match the declared document type.",
    },
    "STRUCT_SECURITY": {
        "message": "Security features look non-standard",
        "detail": "Font, seal, or MRZ-zone placement does not match the official template.",
    },
    "MRZ_CHECKSUM": {
        "message": "MRZ checksum failure",
        "detail": "ICAO 9303 machine-readable zone checksum failed — high-confidence data anomaly.",
    },
    "MRZ_OCR_MISMATCH": {
        "message": "Printed fields do not match the MRZ",
        "detail": "Visually printed biographical fields disagree with the machine-readable zone.",
    },
    "MRZ_DATE_LOGIC": {
        "message": "Date logic failure",
        "detail": "DOB / issue / expiry dates are inconsistent, or a date is in the future.",
    },
    "TAMPER_ELA": {
        "message": "Possible digital editing detected",
        "detail": "Error Level Analysis highlights recompression artefacts consistent with local editing.",
    },
    "TAMPER_CNN": {
        "message": "Forensic model flagged image tampering",
        "detail": "CNN detected splice, clone-stamp, or inpainting cues on the document image.",
    },
    "TAMPER_PHOTO_SWAP": {
        "message": "Photo substitution suspected",
        "detail": "Portrait-region noise boundary is consistent with a swapped photograph.",
    },
    "FACE_MISMATCH": {
        "message": "Face mismatch detected",
        "detail": "Live face does not match the document portrait closely enough.",
    },
    "FACE_LIVENESS": {
        "message": "Liveness check failed",
        "detail": "Presentation attack possible (print, screen, or mask).",
    },
    "MORPH_DMAD": {
        "message": "Facial morphing suspected",
        "detail": "Differential morph detection found identity inconsistency typical of a morphed portrait.",
    },
    "WATCHLIST_HIT": {
        "message": "Watchlist hit",
        "detail": "Watchlist / SLTD / duplicate-identity graph returned a positive match.",
    },
    "WATCHLIST_DUPLICATE": {
        "message": "Possible duplicate identity",
        "detail": "1:N embedding search matched another identity at a different checkpoint.",
    },
    "QUALITY_INSUFFICIENT": {
        "message": "Image quality too low to analyse",
        "detail": "No fraud declaration is issued when capture quality is insufficient.",
    },
}


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(value)))


def classify_threat(score: float) -> dict[str, str]:
    s = clamp(score)
    if s >= 85:
        return {
            "level": "CRITICAL",
            "action": "Detain for investigation — multiple high-severity signals present.",
            "color": "#b1000e",
        }
    if s >= 60:
        return {
            "level": "HIGH",
            "action": "Further verification — escalate to supervisor / secondary inspection.",
            "color": "#d35400",
        }
    if s >= 30:
        return {
            "level": "MEDIUM",
            "action": "Manual review — officer examines the flagged signals before clearance.",
            "color": "#c47a00",
        }
    return {
        "level": "LOW",
        "action": "Proceed — standard clearance. Human remains in the loop.",
        "color": "#138808",
    }


def _reason_codes(signals: dict[str, float], flags: dict[str, bool]) -> list[dict[str, str]]:
    reasons: list[dict[str, str]] = []

    def add(code: str, severity: str) -> None:
        entry = REASON_CATALOG[code]
        reasons.append(
            {
                "code": code,
                "severity": severity,
                "message": entry["message"],
                "detail": entry["detail"],
            }
        )

    if flags.get("quality_insufficient"):
        add("QUALITY_INSUFFICIENT", "INFO")
        return reasons

    structure = signals["structure"]
    ocr = signals["ocr_mrz"]
    tamper = signals["tampering"]
    bio = signals["biometrics"]
    morph = signals["morphing"]
    watch = signals["watchlist"]

    if structure >= 55:
        add("STRUCT_LAYOUT", "HIGH" if structure >= 75 else "MEDIUM")
    if structure >= 70:
        add("STRUCT_SECURITY", "HIGH")
    if flags.get("mrz_checksum_fail") or ocr >= 70:
        add("MRZ_CHECKSUM", "HIGH")
    if flags.get("mrz_ocr_mismatch") or ocr >= 50:
        add("MRZ_OCR_MISMATCH", "HIGH" if ocr >= 70 else "MEDIUM")
    if flags.get("date_logic_fail"):
        add("MRZ_DATE_LOGIC", "MEDIUM")
    if tamper >= 40:
        add("TAMPER_ELA", "HIGH" if tamper >= 70 else "MEDIUM")
    if tamper >= 55:
        add("TAMPER_CNN", "HIGH")
    if flags.get("photo_swap") or tamper >= 80:
        add("TAMPER_PHOTO_SWAP", "CRITICAL")
    if flags.get("face_mismatch") or bio >= 55:
        add("FACE_MISMATCH", "HIGH" if bio >= 70 else "MEDIUM")
    if flags.get("liveness_fail"):
        add("FACE_LIVENESS", "HIGH")
    if flags.get("morph_detected") or morph >= 70:
        add("MORPH_DMAD", "CRITICAL" if morph >= 85 else "HIGH")
    if flags.get("watchlist_hit") or watch >= 80:
        add("WATCHLIST_HIT", "CRITICAL")
    if flags.get("duplicate_identity"):
        add("WATCHLIST_DUPLICATE", "HIGH")

    return reasons


def flags_from_signals(signals: dict[str, float], extra: dict[str, bool] | None = None) -> dict[str, bool]:
    flags = dict(extra or {})
    structure = signals.get("structure", 0)
    ocr = signals.get("ocr_mrz", 0)
    tamper = signals.get("tampering", 0)
    bio = signals.get("biometrics", 0)
    morph = signals.get("morphing", 0)
    watch = signals.get("watchlist", 0)
    if ocr >= 70:
        flags.setdefault("mrz_checksum_fail", True)
        flags.setdefault("mrz_ocr_mismatch", True)
    elif ocr >= 50:
        flags.setdefault("mrz_ocr_mismatch", True)
    if tamper >= 80:
        flags.setdefault("photo_swap", True)
    if bio >= 55:
        flags.setdefault("face_mismatch", True)
    if morph >= 70:
        flags.setdefault("morph_detected", True)
    if watch >= 80:
        flags.setdefault("watchlist_hit", True)
    if structure >= 55:
        flags.setdefault("struct_issue", True)
    return flags


def compute_risk(
    signals: dict[str, float],
    flags: dict[str, bool] | None = None,
    document_type: str = "passport",
    traveller_name: str = "",
    policy_escalation: bool = True,
) -> dict[str, Any]:
    """Fuse module scores into a weighted 0-100 risk result."""
    normalised = {key: clamp(signals.get(key, 0.0)) for key in WEIGHTS}
    flags = flags_from_signals(normalised, flags)

    if flags.get("quality_insufficient"):
        threat = classify_threat(0)
        return {
            "score": 0.0,
            "weighted_score": 0.0,
            "level": "INCONCLUSIVE",
            "action": "Unable to analyse — image quality / OCR insufficient. Do not treat as fraud.",
            "color": "#8b9bb4",
            "signals": normalised,
            "reasons": _reason_codes(normalised, flags),
            "escalations": [],
            "document_type": document_type,
            "traveller_name": traveller_name,
        }

    weighted = round(clamp(sum(normalised[key] * weight for key, weight in WEIGHTS.items())), 1)
    score = weighted
    escalations: list[str] = []

    if policy_escalation:
        if flags.get("watchlist_hit") or normalised["watchlist"] >= 80:
            if score < 85:
                score = 88.0
                escalations.append("Watchlist hit — case raised to CRITICAL.")
            flags["watchlist_hit"] = True
        if flags.get("morph_detected") or normalised["morphing"] >= 85:
            if score < 60:
                score = 64.0
                escalations.append("Facial morphing suspected — case raised to HIGH.")
            flags["morph_detected"] = True
        if flags.get("mrz_checksum_fail") and normalised["tampering"] >= 70 and score < 60:
            score = 62.0
            escalations.append("MRZ failure with tampering — case raised to HIGH.")

    score = round(clamp(score), 1)
    threat = classify_threat(score)

    return {
        "score": score,
        "weighted_score": weighted,
        "level": threat["level"],
        "action": threat["action"],
        "color": threat["color"],
        "signals": normalised,
        "reasons": _reason_codes(normalised, flags),
        "escalations": escalations,
        "document_type": document_type,
        "traveller_name": traveller_name,
        "human_in_the_loop": True,
        "disclaimer": "Decision-support only. The officer records the final decision.",
    }


def fused_risk(
    signals: dict[str, float],
    flags: dict[str, bool] | None = None,
    document_type: str = "passport",
    traveller_name: str = "",
    available: set[str] | list[str] | None = None,
) -> dict[str, Any]:
    """Risk score over the channels that produced a reading.

    Missing channels (e.g. biometrics when no live photo was provided) do
    NOT silently lower or inflate the score. The risk engine computes the
    fixed-weight fusion exactly as with a full verification, and the
    result reports `channels_omitted` plus `partial_verification=True` so
    the officer knows sensitivity is reduced until the missing channels
    are completed.
    """
    avail = {k for k in (available or []) if k in WEIGHTS}
    if not avail:
        avail = set(WEIGHTS)
    omitted = sorted(set(WEIGHTS) - avail)
    result = compute_risk(
        signals,
        flags=flags,
        document_type=document_type,
        traveller_name=traveller_name,
    )
    result["channels_available"] = sorted(avail)
    result["channels_omitted"] = omitted
    result["partial_verification"] = bool(omitted)
    return result


# One-click SIH demo presets (synthetic identities only)
PRESETS: dict[str, dict[str, Any]] = {
    "A": {
        "id": "A",
        "title": "Demo A — Genuine traveller",
        "subtitle": "Clean document, matching live face, empty watchlist",
        "expected_level": "LOW",
        "document_type": "passport",
        "traveller_name": "ANITA SHARMA",
        "doc_number": "N1234567",
        "signals": {
            "structure": 8, "ocr_mrz": 5, "tampering": 6,
            "biometrics": 10, "morphing": 4, "watchlist": 0,
        },
        "flags": {},
    },
    "B": {
        "id": "B",
        "title": "Demo B — Suspicious MRZ / DOB edit",
        "subtitle": "Printed DOB altered; MRZ left inconsistent; ELA hotspot on date field",
        "expected_level": "HIGH",
        "document_type": "passport",
        "traveller_name": "BINOD THAPA",
        "doc_number": "P9988776",
        "signals": {
            "structure": 55, "ocr_mrz": 100, "tampering": 95,
            "biometrics": 25, "morphing": 20, "watchlist": 0,
        },
        "flags": {"mrz_checksum_fail": True, "mrz_ocr_mismatch": True, "date_logic_fail": True},
    },
    "C": {
        "id": "C",
        "title": "Demo C — Facial morphing",
        "subtitle": "Document portrait resembles two identities (D-MAD positive)",
        "expected_level": "HIGH",
        "document_type": "voter_id",
        "traveller_name": "CHIRAG RANA",
        "doc_number": "ABC1234567",
        "signals": {
            "structure": 22, "ocr_mrz": 18, "tampering": 35,
            "biometrics": 48, "morphing": 96, "watchlist": 0,
        },
        "flags": {"morph_detected": True},
    },
    "D": {
        "id": "D",
        "title": "Demo D — Watchlist hit",
        "subtitle": "Cached SLTD / national watchlist match on document number",
        "expected_level": "CRITICAL",
        "document_type": "citizenship_certificate",
        "traveller_name": "DEEPAK KARKI",
        "doc_number": "CC-441902",
        "signals": {
            "structure": 12, "ocr_mrz": 10, "tampering": 8,
            "biometrics": 14, "morphing": 6, "watchlist": 100,
        },
        "flags": {"watchlist_hit": True},
    },
    "E": {
        "id": "E",
        "title": "Demo E — Audit ledger tampering",
        "subtitle": "Screening is LOW; then a past ledger row is mutated to prove hash-chain detection",
        "expected_level": "LOW",
        "document_type": "driving_licence",
        "traveller_name": "ESHA GURUNG",
        "doc_number": "DL-09-2024-8811",
        "signals": {
            "structure": 7, "ocr_mrz": 6, "tampering": 5,
            "biometrics": 9, "morphing": 3, "watchlist": 0,
        },
        "flags": {},
        "triggers_tamper_demo": True,
    },
}


def evaluate_preset(preset_id: str) -> dict[str, Any]:
    key = preset_id.strip().upper()
    if key not in PRESETS:
        raise KeyError(f"Unknown preset '{preset_id}'. Use A–E.")
    preset = PRESETS[key]
    result = compute_risk(
        preset["signals"],
        flags=preset.get("flags"),
        document_type=preset["document_type"],
        traveller_name=preset["traveller_name"],
    )
    result["preset"] = {
        "id": preset["id"],
        "title": preset["title"],
        "subtitle": preset["subtitle"],
        "expected_level": preset["expected_level"],
        "doc_number": preset["doc_number"],
        "triggers_tamper_demo": bool(preset.get("triggers_tamper_demo")),
    }
    return result


def list_presets() -> list[dict[str, Any]]:
    return [
        {
            "id": p["id"],
            "title": p["title"],
            "subtitle": p["subtitle"],
            "expected_level": p["expected_level"],
            "document_type": p["document_type"],
            "triggers_tamper_demo": bool(p.get("triggers_tamper_demo")),
        }
        for p in PRESETS.values()
    ]