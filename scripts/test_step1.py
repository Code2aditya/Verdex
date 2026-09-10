"""
test_step1.py — Manual smoke-test for preprocessing.py and qr_check.py
Run from the veridex/ root:

    venv\Scripts\python.exe scripts/test_step1.py

What it does:
  1. Builds a synthetic passport-like document image (white bg + printed fields)
  2. Artificially tilts it 7 degrees to verify deskewing
  3. Runs preprocess() and prints all result fields + saves output images
  4. Embeds a JSON QR code in the document image
  5. Runs check_qr() with matching and mismatching OCR fields
  6. Prints a PASS/FAIL summary for every check
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import cv2
import numpy as np
import qrcode
from PIL import Image, ImageDraw, ImageFont

from backend.modules.module3_tampering.preprocessing import preprocess, to_pil
from backend.modules.module3_tampering.qr_check import check_qr

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "test_step1_output")
os.makedirs(OUT_DIR, exist_ok=True)

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"

results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}")
    if detail:
        print(f"         {detail}")
    results.append((label, condition))


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Build a synthetic document image
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Building synthetic document image ===")

DOC_W, DOC_H = 1000, 700

# White canvas
doc_pil = Image.new("RGB", (DOC_W, DOC_H), (255, 255, 255))
draw    = ImageDraw.Draw(doc_pil)

# Header bar
draw.rectangle([(0, 0), (DOC_W, 80)], fill=(30, 60, 120))
draw.text((20, 20), "REPUBLIC OF INDIA — IDENTITY DOCUMENT", fill=(255, 255, 255))

# Printed fields (simulating a national ID)
fields_text = [
    ("Document No :", "IND-2024-987654"),
    ("Full Name   :", "SURESH MEHTA"),
    ("Date of Birth:", "15-08-1998"),
    ("Expiry Date :", "14-08-2029"),
    ("Nationality :", "INDIAN"),
    ("Gender      :", "M"),
]
y = 110
for label, value in fields_text:
    draw.text((60, y),   label, fill=(60, 60, 60))
    draw.text((260, y),  value, fill=(20, 20, 20))
    y += 40

# Photo placeholder
draw.rectangle([(780, 100), (950, 300)], outline=(180, 180, 180), width=2)
draw.text((800, 185), "[PHOTO]", fill=(150, 150, 150))

# Bottom MRZ-like strip
draw.rectangle([(0, 620), (DOC_W, DOC_H)], fill=(240, 240, 240))
draw.text((20, 635), "IND<<MEHTA<<SURESH<<<<<<<<<<<<<<<<<<<<<<<<<<", fill=(20, 20, 20))
draw.text((20, 660), "IND2024987654M2908145<<<<<<<<<<<<<<<<<<<<<<<<", fill=(20, 20, 20))

# Save clean version
clean_path = os.path.join(OUT_DIR, "doc_clean.jpg")
doc_pil.save(clean_path, quality=95)
print(f"  Saved clean document: {clean_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Test preprocessing on CLEAN image
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 1: preprocess() on clean image ===")

result = preprocess(clean_path)

check("PreprocessResult returned",         result is not None)
check("original shape is 3-channel",       len(result.original.shape) == 3)
check("canvas width == 1200",              result.width == 1200)
check("canvas height == 850",              result.height == 850)
check("gray is 2D",                        len(result.gray.shape) == 2)
check("denoised shape matches canvas",     result.denoised.shape == result.canvas.shape)
check("cnn_input shape == (224,224,3)",    result.cnn_input.shape == (224, 224, 3))
check("cnn_input dtype float32",           result.cnn_input.dtype == np.float32)
check("cnn_input values in [0,1]",
      result.cnn_input.min() >= 0.0 and result.cnn_input.max() <= 1.0)
check("skew_angle near 0 for clean image", abs(result.skew_angle) < 3.0,
      f"skew_angle={result.skew_angle:.2f}°")
check("no warnings on clean image",        len(result.warnings) == 0,
      f"warnings: {result.warnings}")

# Save canvas output
cv2.imwrite(os.path.join(OUT_DIR, "canvas_clean.jpg"), result.canvas)
cv2.imwrite(os.path.join(OUT_DIR, "denoised_clean.jpg"), result.denoised)
cv2.imwrite(os.path.join(OUT_DIR, "gray_clean.jpg"),    result.gray)
print(f"  Saved canvas, denoised, gray to: {OUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Test preprocessing on TILTED image (deskew test)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 2: preprocess() deskew on 7-degree tilted image ===")

TILT_ANGLE = 7.0
doc_cv  = cv2.cvtColor(np.array(doc_pil), cv2.COLOR_RGB2BGR)
h, w    = doc_cv.shape[:2]
M       = cv2.getRotationMatrix2D((w / 2, h / 2), TILT_ANGLE, 1.0)
tilted  = cv2.warpAffine(doc_cv, M, (w, h), borderValue=(255, 255, 255))

tilted_path = os.path.join(OUT_DIR, "doc_tilted.jpg")
cv2.imwrite(tilted_path, tilted)

result_tilt = preprocess(tilted_path)

check("Deskew applied (angle detected)",
      abs(result_tilt.skew_angle) > 0.5,
      f"detected skew_angle={result_tilt.skew_angle:.2f}°  (tilted by {TILT_ANGLE}°)")
check("Canvas still 1200x850 after deskew",
      result_tilt.width == 1200 and result_tilt.height == 850)

cv2.imwrite(os.path.join(OUT_DIR, "canvas_deskewed.jpg"), result_tilt.canvas)
print(f"  Saved deskewed canvas to: {OUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — Test preprocess() from bytes (upload simulation)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 3: preprocess() from raw bytes ===")

with open(clean_path, "rb") as f:
    raw_bytes = f.read()

result_bytes = preprocess(raw_bytes)
check("Bytes source works",          result_bytes is not None)
check("Canvas size correct (bytes)", result_bytes.width == 1200)

# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — Build document image WITH embedded QR code
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Building document with embedded QR code ===")

QR_PAYLOAD = {
    "doc_number":  "IND-2024-987654",
    "name":        "SURESH MEHTA",
    "dob":         "1998-08-15",
    "expiry":      "2029-08-14",
    "nationality": "INDIAN",
    "gender":      "M",
}

qr = qrcode.QRCode(version=3, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=4, border=2)
qr.add_data(json.dumps(QR_PAYLOAD))
qr.make(fit=True)
qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

# Paste QR onto clean document
doc_with_qr = doc_pil.copy()
qr_resized  = qr_img.resize((160, 160))
doc_with_qr.paste(qr_resized, (60, 460))
draw2 = ImageDraw.Draw(doc_with_qr)
draw2.text((60, 625), "QR verified", fill=(80, 80, 80))

qr_doc_path = os.path.join(OUT_DIR, "doc_with_qr.jpg")
doc_with_qr.save(qr_doc_path, quality=95)
print(f"  Saved document with QR: {qr_doc_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 6 — Test check_qr() with MATCHING OCR fields
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 4: check_qr() — matching OCR fields ===")

qr_doc_cv   = cv2.imread(qr_doc_path)
result_qr   = preprocess(qr_doc_cv)

ocr_matching = {
    "doc_number":  "IND-2024-987654",
    "name":        "SURESH MEHTA",
    "dob":         "1998-08-15",
    "expiry":      "2029-08-14",
    "nationality": "INDIAN",
    "gender":      "M",
}

qr_result_match = check_qr(result_qr.canvas, ocr_matching)

print(f"  found:          {qr_result_match.found}")
print(f"  barcodes:       {len(qr_result_match.barcodes)} detected")
print(f"  parsed_fields:  {qr_result_match.parsed_fields}")
print(f"  mismatches:     {qr_result_match.mismatches}")
print(f"  risk_delta:     {qr_result_match.risk_delta:.2f}")
print(f"  explanation:    {qr_result_match.explanation}")

check("QR found in document",             qr_result_match.found)
check("At least 1 barcode decoded",       len(qr_result_match.barcodes) >= 1)
check("doc_number parsed from QR",        "doc_number" in qr_result_match.parsed_fields)
check("No mismatches (fields match OCR)", len(qr_result_match.mismatches) == 0,
      f"mismatches={qr_result_match.mismatches}")
check("Risk delta == 0.0 (clean match)",  qr_result_match.risk_delta == 0.0,
      f"risk_delta={qr_result_match.risk_delta}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 7 — Test check_qr() with MISMATCHING OCR fields (tamper simulation)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 5: check_qr() — TAMPERED fields (mismatch) ===")

ocr_tampered = {
    "doc_number":  "IND-2024-000000",   # changed doc number
    "name":        "FAKE PERSON",        # changed name
    "dob":         "1998-08-15",         # same
    "expiry":      "2029-08-14",         # same
    "nationality": "INDIAN",
    "gender":      "M",
}

qr_result_tamper = check_qr(result_qr.canvas, ocr_tampered)

print(f"  mismatches:  {qr_result_tamper.mismatches}")
print(f"  risk_delta:  {qr_result_tamper.risk_delta:.2f}")
print(f"  explanation: {qr_result_tamper.explanation}")

check("Mismatch detected on doc_number",  "doc_number" in qr_result_tamper.mismatches)
check("Mismatch detected on name",        "name"       in qr_result_tamper.mismatches)
check("Risk delta > 0 (tampering flagged)", qr_result_tamper.risk_delta > 0.0,
      f"risk_delta={qr_result_tamper.risk_delta:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 8 — Test check_qr() on image with NO QR code
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 6: check_qr() — no QR in image ===")

no_qr_result = check_qr(result.canvas, ocr_matching)
check("found=False when no QR",     not no_qr_result.found)
check("risk_delta == 0.05 (NO_QR)", no_qr_result.risk_delta == 0.05,
      f"risk_delta={no_qr_result.risk_delta}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print(f"\n{'='*55}")
print(f"  RESULTS: {passed}/{total} passed  |  {failed} failed")
print(f"  Output images saved to: {os.path.abspath(OUT_DIR)}")
print(f"{'='*55}\n")

if failed:
    sys.exit(1)
