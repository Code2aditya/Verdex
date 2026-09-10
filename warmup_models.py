"""Bake ML model weights into the Docker image at build time.

Railway containers have an ephemeral filesystem: anything downloaded at
request time (EasyOCR detector/recognizer, InsightFace buffalo_l pack)
is lost on every restart AND the download spikes RAM/CPU on the first
request. Pre-downloading here moves that cost to build time.

Respects EASYOCR_QUANTIZE / ENABLE_INSIGHTFACE so the baked weights
match what the runtime will actually load.

Failure is non-fatal (exit 0): the pipeline falls back to downloading
at runtime if warmup could not complete.
"""
import os
import traceback


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def main() -> None:
    quantize = _flag("EASYOCR_QUANTIZE", "0")
    want_insight = _flag("ENABLE_INSIGHTFACE", "1")

    try:
        import easyocr

        print(f"[warmup] downloading EasyOCR models (en, quantize={quantize})...", flush=True)
        easyocr.Reader(["en"], gpu=False, quantize=quantize)
        print(f"[warmup] downloading EasyOCR models (hi+en, quantize={quantize})...", flush=True)
        easyocr.Reader(["hi", "en"], gpu=False, quantize=quantize)
        print("[warmup] easyocr OK", flush=True)
    except Exception:
        print("[warmup] easyocr warmup FAILED, runtime fallback remains:", flush=True)
        traceback.print_exc()

    if not want_insight:
        print("[warmup] ENABLE_INSIGHTFACE=0 — skipping insightface download.", flush=True)
        return

    try:
        from insightface.app import FaceAnalysis

        print("[warmup] downloading InsightFace buffalo_l pack...", flush=True)
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("[warmup] insightface OK", flush=True)
    except Exception:
        print("[warmup] insightface warmup FAILED, runtime fallback remains:", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
