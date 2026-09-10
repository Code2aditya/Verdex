"""
generate_dataset.py — Synthetic tampered document dataset generator
VERIDEX | Module 3 Sprint Tool

Generates a balanced genuine/tampered dataset from any source document images.
Designed to be run ONCE to populate dataset/genuine/ and dataset/tampered/
so that train_cnn.py can fine-tune the EfficientNet binary head.

Tampering strategies applied (all programmatic, fully ground-truth labelled):
  1. DATE_SWAP       — replaces a text region with a different date string
  2. PHOTO_BLUR      — blurs the photo region (simulates photo substitution)
  3. CLONE_STAMP     — copies a patch from one region and pastes it elsewhere
  4. TEXT_INSERT     — draws new text over an existing field
  5. BRIGHTNESS_PATCH— applies localised brightness shift (editing artefact)
  6. JPEG_GHOST      — saves at lower quality then re-embeds (JPEG ghost)

Usage
-----
  python scripts/generate_dataset.py --source path/to/docs/ --count 10

  --source  : directory of genuine document images OR a single image path.
              If only 1 image is provided it is duplicated with augmentation.
  --count   : number of tampered images to generate (default 15).
  --out-dir : project dataset root (default: dataset/).
  --seed    : random seed for reproducibility (default: 42).
"""

from __future__ import annotations

import argparse
import io
import logging
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATASET = _ROOT / "dataset"

# Dummy dates to swap in
_FAKE_DATES = [
    "01 JAN 2035", "15 MAR 2028", "31 DEC 2030", "07 JUL 2029",
    "20 APR 2027", "03 SEP 2033", "11 NOV 2031", "25 FEB 2026",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic tampered document dataset")
    p.add_argument("--source",  default=str(_DEFAULT_DATASET / "genuine"),
                   help="Directory of genuine images or a single image path")
    p.add_argument("--count",   type=int, default=15,
                   help="Number of tampered images to generate")
    p.add_argument("--out-dir", default=str(_DEFAULT_DATASET),
                   help="Dataset root directory")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    source_path = Path(args.source)
    out_dir     = Path(args.out_dir)
    genuine_dir = out_dir / "genuine"
    tampered_dir = out_dir / "tampered"
    genuine_dir.mkdir(parents=True, exist_ok=True)
    tampered_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect source images ─────────────────────────────────────────────────
    if source_path.is_file():
        source_images = [source_path]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        source_images = [p for p in source_path.iterdir() if p.suffix.lower() in exts]

    if not source_images:
        logger.error("No source images found at %s", source_path)
        sys.exit(1)

    logger.info("Found %d source image(s)", len(source_images))

    # ── Copy/augment genuine images ───────────────────────────────────────────
    genuine_count = max(len(source_images), args.count)
    genuine_paths: list[Path] = []
    for i in range(genuine_count):
        src = source_images[i % len(source_images)]
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Could not read %s — skipping", src)
            continue
        # Light augmentation for genuine copies (only if duplicating)
        if i >= len(source_images):
            img = _augment_genuine(img)
        dst = genuine_dir / f"genuine_{i:04d}.jpg"
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        genuine_paths.append(dst)
        logger.info("Genuine [%d/%d] → %s", i + 1, genuine_count, dst.name)

    # ── Generate tampered images ──────────────────────────────────────────────
    strategies = [
        _date_swap, _photo_blur, _clone_stamp,
        _text_insert, _brightness_patch, _jpeg_ghost,
    ]

    for i in range(args.count):
        src = source_images[i % len(source_images)]
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            continue

        strategy = strategies[i % len(strategies)]
        tampered = strategy(img.copy())

        dst = tampered_dir / f"tampered_{strategy.__name__}_{i:04d}.jpg"
        cv2.imwrite(str(dst), tampered, [cv2.IMWRITE_JPEG_QUALITY, 92])
        logger.info("Tampered [%d/%d] %s → %s", i + 1, args.count, strategy.__name__, dst.name)

    logger.info(
        "\nDataset ready:\n  Genuine : %s (%d images)\n  Tampered: %s (%d images)",
        genuine_dir, len(genuine_paths), tampered_dir, args.count,
    )
    logger.info("Next step: python scripts/train_cnn.py")


# ── Tampering strategies ──────────────────────────────────────────────────────

def _date_swap(img: np.ndarray) -> np.ndarray:
    """Replace a random rectangular region with a fake date string."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    h, w = img.shape[:2]

    # Pick a region in the lower half (where dates usually appear)
    x1 = random.randint(w // 4, w // 2)
    y1 = random.randint(h // 2, int(h * 0.75))
    x2 = x1 + random.randint(80, 160)
    y2 = y1 + random.randint(18, 28)

    # Erase with white
    draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
    # Draw fake date
    date_str = random.choice(_FAKE_DATES)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((x1 + 2, y1 + 2), date_str, fill=(0, 0, 0), font=font)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _photo_blur(img: np.ndarray) -> np.ndarray:
    """Blur the upper-left region (typical photo location on ID cards)."""
    h, w = img.shape[:2]
    x1, y1 = int(w * 0.05), int(h * 0.10)
    x2, y2 = int(w * 0.30), int(h * 0.55)

    result = img.copy()
    roi = result[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (25, 25), 0)
    # Add slight brightness shift to simulate pasting
    blurred = np.clip(blurred.astype(np.int32) + random.randint(-20, 20), 0, 255).astype(np.uint8)
    result[y1:y2, x1:x2] = blurred
    return result


def _clone_stamp(img: np.ndarray) -> np.ndarray:
    """Copy-paste a patch from one location to another."""
    h, w = img.shape[:2]
    ph = random.randint(30, 80)
    pw = random.randint(60, 120)

    src_y = random.randint(0, h - ph - 1)
    src_x = random.randint(0, w - pw - 1)
    dst_y = random.randint(0, h - ph - 1)
    dst_x = random.randint(0, w - pw - 1)

    result = img.copy()
    result[dst_y:dst_y + ph, dst_x:dst_x + pw] = img[src_y:src_y + ph, src_x:src_x + pw]
    return result


def _text_insert(img: np.ndarray) -> np.ndarray:
    """Draw synthetic text over a random field position."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    h, w = img.shape[:2]

    x = random.randint(w // 3, int(w * 0.7))
    y = random.randint(h // 3, int(h * 0.7))
    text = f"MODIFIED {random.randint(1000, 9999)}"

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # White background behind text
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle(bbox, fill=(255, 255, 255))
    draw.text((x, y), text, fill=(10, 10, 10), font=font)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _brightness_patch(img: np.ndarray) -> np.ndarray:
    """Apply localised brightness shift — simulates editing software artefact."""
    h, w = img.shape[:2]
    ph = random.randint(20, 60)
    pw = random.randint(40, 100)
    y = random.randint(0, h - ph - 1)
    x = random.randint(0, w - pw - 1)

    result = img.copy().astype(np.int32)
    delta = random.choice([-40, -30, 30, 40, 50])
    result[y:y + ph, x:x + pw] = np.clip(result[y:y + ph, x:x + pw] + delta, 0, 255)
    return result.astype(np.uint8)


def _jpeg_ghost(img: np.ndarray) -> np.ndarray:
    """
    JPEG ghost: encode a patch at low quality, then paste it back.
    Creates a detectable compression mismatch (the key ELA signal).
    """
    h, w = img.shape[:2]
    ph = random.randint(40, 100)
    pw = random.randint(80, 180)
    y = random.randint(0, h - ph - 1)
    x = random.randint(0, w - pw - 1)

    patch = img[y:y + ph, x:x + pw]
    pil_patch = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_patch.save(buf, format="JPEG", quality=random.randint(30, 55))
    buf.seek(0)
    reloaded = np.array(Image.open(buf))
    reloaded_bgr = cv2.cvtColor(reloaded, cv2.COLOR_RGB2BGR)

    result = img.copy()
    result[y:y + ph, x:x + pw] = reloaded_bgr
    return result


def _augment_genuine(img: np.ndarray) -> np.ndarray:
    """Light augmentation for genuine image copies (brightness jitter only)."""
    delta = random.randint(-15, 15)
    return np.clip(img.astype(np.int32) + delta, 0, 255).astype(np.uint8)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
