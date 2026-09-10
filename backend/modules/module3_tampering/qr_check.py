"""
qr_check.py — QR / barcode decode and cross-check against OCR fields
Module 3 | VERIDEX | Anubhav

Many modern identity documents embed a QR code or PDF417 / Code128 barcode
that encodes a subset of the printed fields (name, DOB, document number,
expiry).  A forged document that swaps the photo or edits a field will often
leave the barcode unchanged, creating a detectable mismatch.

Pipeline
--------
  1. Decode all barcodes in the document image using zxing-cpp.
  2. If nothing found, apply image enhancement and retry once.
  3. Parse the decoded payload (JSON, ICAO 9303 MRZ, or key=value pairs).
  4. Cross-check against OCR-extracted fields provided by Module 1.
  5. Return a QRCheckResult with match status, mismatch list, and risk delta.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

try:
    import zxingcpp
    _ZXING_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _ZXING_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Risk weights ──────────────────────────────────────────────────────────────
# Each mismatched field contributes this to the Module-3 composite risk score.
FIELD_RISK_WEIGHTS: dict[str, float] = {
    "doc_number":   0.40,
    "dob":          0.30,
    "expiry":       0.20,
    "name":         0.15,
    "nationality":  0.10,
    "gender":       0.05,
}
NO_QR_RISK      = 0.05    # small bump when no QR found (not all docs have one)
DECODE_FAIL_RISK = 0.10   # QR region detected but text unreadable -> higher suspicion


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class BarcodeHit:
    """Raw decode result for a single barcode found in the image."""
    format:   str    # e.g. "QR_CODE", "PDF_417", "CODE_128"
    raw_text: str    # decoded text payload
    position: tuple  # (x, y, w, h) bounding box in pixels


@dataclass
class QRCheckResult:
    """
    Result returned to tamper_pipeline.py.

    Attributes
    ----------
    found           : True if at least one barcode was decoded.
    barcodes        : List of all BarcodeHit objects found.
    parsed_fields   : Fields extracted from the barcode payload, keyed by
                      canonical field name (e.g. "doc_number", "dob").
    mismatches      : Canonical field names where QR value != OCR value.
    risk_delta      : 0.0-1.0 contribution to the composite risk score.
    explanation     : Human-readable summary shown to the border officer.
    warnings        : Non-fatal issues encountered during processing.
    """
    found:         bool
    barcodes:      list
    parsed_fields: dict
    mismatches:    list
    risk_delta:    float
    explanation:   str
    warnings:      list = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def check_qr(
    image: np.ndarray,
    ocr_fields: dict | None = None,
) -> QRCheckResult:
    """
    Decode QR/barcodes from *image* (BGR numpy array) and cross-check
    against *ocr_fields* extracted by Module 1 / Module 2.

    Parameters
    ----------
    image       : BGR numpy array (e.g. PreprocessResult.canvas).
    ocr_fields  : Optional dict of field_name -> value from Module 1.
                  Recognised canonical keys:
                    doc_number, name, dob, expiry, nationality, gender
                  Non-canonical keys are mapped via _ALIAS_TO_CANONICAL.

    Returns
    -------
    QRCheckResult
    """
    ocr_fields = _normalise_keys(ocr_fields or {})
    warnings   = []

    # ── 1. Decode barcodes ────────────────────────────────────────────────────
    barcodes = _decode_barcodes(image)

    if not barcodes:
        # Retry with contrast-enhanced version
        enhanced = _enhance_for_barcode(image)
        barcodes = _decode_barcodes(enhanced)
        if barcodes:
            warnings.append("Barcode only detected after image enhancement — "
                            "possible low-quality scan or print.")

    # ── 2. No barcode found ───────────────────────────────────────────────────
    if not barcodes:
        return QRCheckResult(
            found         = False,
            barcodes      = [],
            parsed_fields = {},
            mismatches    = [],
            risk_delta    = NO_QR_RISK,
            explanation   = (
                "No QR code or barcode detected on document. "
                "Not all document types include one; absence alone is not "
                "a tamper indicator."
            ),
            warnings=warnings,
        )

    # ── 3. Parse payloads ─────────────────────────────────────────────────────
    parsed: dict[str, str] = {}
    for hit in barcodes:
        parsed.update(_parse_payload(hit.raw_text))

    # ── 4. Cross-check against OCR fields ────────────────────────────────────
    mismatches: list[str] = []
    if ocr_fields and parsed:
        for canon_name in FIELD_RISK_WEIGHTS:
            ocr_val = ocr_fields.get(canon_name)
            qr_val  = parsed.get(canon_name)
            if ocr_val and qr_val:
                if not _values_match(ocr_val, qr_val):
                    mismatches.append(canon_name)
                    logger.warning(
                        "QR<->OCR mismatch [%s]:  QR=%r  OCR=%r",
                        canon_name, qr_val, ocr_val,
                    )
    elif not parsed:
        warnings.append(
            "Barcode decoded but payload could not be mapped to document "
            "fields. Raw text stored for officer review."
        )

    # ── 5. Composite risk delta ───────────────────────────────────────────────
    risk = sum(FIELD_RISK_WEIGHTS[f] for f in mismatches)
    risk = min(risk, 1.0)

    # ── 6. Officer-facing explanation ─────────────────────────────────────────
    if mismatches:
        mismatch_str = ", ".join(mismatches)
        explanation = (
            f"WARNING: QR/barcode data DOES NOT match printed fields: "
            f"{mismatch_str}. "
            "This strongly suggests field tampering — the barcode retains "
            "original values while visible text has been altered."
        )
    elif parsed and ocr_fields:
        explanation = (
            "QR/barcode decoded successfully. "
            "All verifiable fields match OCR output."
        )
    else:
        explanation = (
            "QR/barcode decoded. No OCR fields provided for cross-check; "
            "manual field verification recommended."
        )

    return QRCheckResult(
        found         = True,
        barcodes      = barcodes,
        parsed_fields = parsed,
        mismatches    = mismatches,
        risk_delta    = risk,
        explanation   = explanation,
        warnings      = warnings,
    )


# ── Barcode decoding ──────────────────────────────────────────────────────────

def _decode_barcodes(image: np.ndarray) -> list:
    """
    Attempt to decode all barcodes in a BGR numpy image using zxing-cpp.
    Supports QR Code, PDF417, Code 128, Data Matrix, Aztec, and more.
    Returns a list of BarcodeHit objects (may be empty).
    """
    # Preferred: zxing-cpp (QR, PDF417, Code128, Aztec, Data Matrix ...)
    if _ZXING_AVAILABLE:
        return _decode_with_zxing(image)

    # Fallback: OpenCV's built-in QRCodeDetector (QR only). Other
    # barcode formats need zxing-cpp; absence is not itself a flag.
    return _decode_with_cv2(image)


def _decode_with_zxing(image: np.ndarray) -> list:
    """Decode all barcodes using zxing-cpp (full format support)."""
    try:
        rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = zxingcpp.read_barcodes(rgb)
        hits    = []
        for r in results:
            if not r.valid or not r.text:
                continue
            pos = r.position
            pts = [
                (pos.top_left.x,     pos.top_left.y),
                (pos.top_right.x,    pos.top_right.y),
                (pos.bottom_right.x, pos.bottom_right.y),
                (pos.bottom_left.x,  pos.bottom_left.y),
            ]
            xs   = [p[0] for p in pts]
            ys   = [p[1] for p in pts]
            bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            hits.append(BarcodeHit(
                format   = str(r.format),
                raw_text = r.text,
                position = bbox,
            ))
        return hits
    except Exception as exc:
        logger.error("zxing-cpp decode error: %s", exc)
        return []


def _decode_with_cv2(image: np.ndarray) -> list:
    """Fallback decode using OpenCV's QRCodeDetector (QR codes only)."""
    try:
        detector = cv2.QRCodeDetector()
        decoded, points, _ = detector.detectAndDecode(image)
        hits = []
        if decoded:
            pts = points.reshape(-1, 2)
            xs = [int(p[0]) for p in pts]
            ys = [int(p[1]) for p in pts]
            bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            hits.append(BarcodeHit(
                format   = "QR_CODE",
                raw_text = decoded,
                position = bbox,
            ))
        return hits
    except Exception as exc:
        logger.error("OpenCV QR decode error: %s", exc)
        return []


def _enhance_for_barcode(image: np.ndarray) -> np.ndarray:
    """
    Image enhancement specifically aimed at improving barcode detectability:
      - Upscale small documents to at least 1000px on the long edge.
      - Unsharp masking to sharpen QR module edges.
      - CLAHE on L channel for local contrast boost.
    """
    h, w = image.shape[:2]

    # Upscale if image is too small for reliable decode
    if max(h, w) < 1000:
        scale = 1000.0 / max(h, w)
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    # Unsharp masking
    blurred   = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    # CLAHE on L channel
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ── Payload parsers ───────────────────────────────────────────────────────────

def _parse_payload(text: str) -> dict[str, str]:
    """
    Try multiple parsing strategies for the decoded barcode text.
    Strategies are tried in order; the first successful one wins.

    1. JSON object
    2. ICAO 9303 TD3 MRZ  (two lines of 44 chars)
    3. ICAO 9303 TD1 MRZ  (three lines of 30 chars)
    4. key=value or key:value pairs (newline / semicolon delimited)
    5. Raw text stored under key "raw" for officer review

    Returns a normalised dict of canonical_field_name -> value.
    """
    text = text.strip()

    result = _try_json(text)
    if result:
        return _normalise_keys(result)

    result = _try_mrz_td3(text)
    if result:
        return result

    result = _try_mrz_td1(text)
    if result:
        return result

    result = _try_kv(text)
    if result:
        return _normalise_keys(result)

    return {"raw": text}


def _try_json(text: str) -> dict | None:
    """Attempt JSON parse; return dict or None."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return {k: str(v) for k, v in obj.items()}
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _try_mrz_td3(text: str) -> dict | None:
    """
    Parse ICAO 9303 TD3 MRZ (passport-style, 2 lines x 44 chars).
    Often embedded verbatim in document QR codes.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2 or len(lines[0]) != 44 or len(lines[1]) != 44:
        return None
    l1, l2 = lines[0], lines[1]
    try:
        name_field   = l1[5:44].split("<<", 1)
        surname      = name_field[0].replace("<", " ").strip()
        given        = name_field[1].replace("<", " ").strip() if len(name_field) > 1 else ""
        return {
            "doc_number":  l2[0:9].replace("<", "").strip(),
            "name":        f"{surname} {given}".strip(),
            "dob":         _mrz_date(l2[13:19]),
            "gender":      l2[20] if l2[20] in ("M", "F") else "X",
            "expiry":      _mrz_date(l2[21:27]),
            "nationality": l1[2:5].replace("<", "").strip(),
        }
    except Exception:
        return None


def _try_mrz_td1(text: str) -> dict | None:
    """
    Parse ICAO 9303 TD1 MRZ (national ID card, 3 lines x 30 chars).
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3 or any(len(ln) != 30 for ln in lines[:3]):
        return None
    l1, l2, l3 = lines[0], lines[1], lines[2]
    try:
        name_field = l3.split("<<", 1)
        surname    = name_field[0].replace("<", " ").strip()
        given      = name_field[1].replace("<", " ").strip() if len(name_field) > 1 else ""
        return {
            "doc_number":  l1[5:14].replace("<", "").strip(),
            "nationality": l2[15:18].replace("<", "").strip(),
            "dob":         _mrz_date(l2[0:6]),
            "gender":      l2[7] if l2[7] in ("M", "F") else "X",
            "expiry":      _mrz_date(l2[8:14]),
            "name":        f"{surname} {given}".strip(),
        }
    except Exception:
        return None


def _try_kv(text: str) -> dict | None:
    """
    Parse key=value or key:value pairs separated by newlines or semicolons.
    Returns the dict if at least 2 pairs are found, else None.
    """
    pairs: dict[str, str] = {}
    for part in re.split(r"[\n;|]", text):
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_\s]*?)\s*[:=]\s*(.+)\s*$", part)
        if m:
            pairs[m.group(1).strip()] = m.group(2).strip()
    return pairs if len(pairs) >= 2 else None


# ── Field normalisation & matching ────────────────────────────────────────────

# Maps common raw key names to canonical field names
_FIELD_ALIASES: dict[str, list[str]] = {
    "doc_number":   ["document_number", "passport_number", "id_number",
                     "docno", "passportno", "number", "no"],
    "name":         ["full_name", "surname", "given_name", "holder_name",
                     "firstname", "lastname"],
    "dob":          ["date_of_birth", "birth_date", "birthdate",
                     "dateofbirth", "born"],
    "expiry":       ["expiry_date", "date_of_expiry", "valid_until",
                     "expiration", "expires", "expirydate"],
    "nationality":  ["nation", "country", "citizen", "citizenship"],
    "gender":       ["sex"],
}

_ALIAS_TO_CANONICAL: dict[str, str] = {
    alias: canonical
    for canonical, aliases in _FIELD_ALIASES.items()
    for alias in aliases
}


def _normalise_keys(d: dict) -> dict[str, str]:
    """Map raw key names to canonical field names where possible."""
    result: dict[str, str] = {}
    for k, v in d.items():
        key       = k.lower().replace(" ", "_").replace("-", "_")
        canonical = _ALIAS_TO_CANONICAL.get(key, key)
        result[canonical] = str(v).strip()
    return result


def _values_match(a: str, b: str) -> bool:
    """
    Fuzzy match two field values accounting for:
      - Case differences
      - MRZ filler characters (<)
      - Date format variations (YYYYMMDD / DDMMYYYY / DD-MM-YYYY)
      - Whitespace and punctuation
    """
    return _normalise_value(a) == _normalise_value(b)


def _normalise_value(v: str) -> str:
    """Normalise a field value to a canonical comparison form."""
    v = v.upper().strip()
    # Strip MRZ fillers and common separators
    v = re.sub(r"[<\-/\s]", "", v)
    # Normalise DDMMYYYY -> YYYYMMDD
    m = re.match(r"^(\d{2})(\d{2})(\d{4})$", v)
    if m:
        v = f"{m.group(3)}{m.group(2)}{m.group(1)}"
    return v


def _mrz_date(s: str) -> str:
    """
    Convert 6-digit MRZ date string YYMMDD to YYYY-MM-DD.
    Century heuristic: YY <= 30 -> 2000s, else -> 1900s.
    """
    if len(s) != 6 or not s.isdigit():
        return s
    yy, mm, dd = s[:2], s[2:4], s[4:6]
    century    = "20" if int(yy) <= 30 else "19"
    return f"{century}{yy}-{mm}-{dd}"
