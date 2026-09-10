"""Bake ML model weights into the Docker image at build time.

Railway containers have an ephemeral filesystem: anything downloaded at
request time (EasyOCR detector/recognizer, InsightFace buffalo_l pack)
is lost on every restart AND the download spikes RAM/CPU on the first
request. Pre-downloading here moves that cost to build time.

Failure is non-fatal (exit 0): the pipeline falls back to downloading
at runtime if warmup could not complete.
"""
import traceback


def main() -> None:
    try:
        import easyocr

        print("[warmup] downloading EasyOCR models (en)...", flush=True)
        easyocr.Reader(["en"], gpu=False)
        print("[warmup] downloading EasyOCR models (hi+en)...", flush=True)
        easyocr.Reader(["hi", "en"], gpu=False)
        print("[warmup] easyocr OK", flush=True)
    except Exception:
        print("[warmup] easyocr warmup FAILED, runtime fallback remains:", flush=True)
        traceback.print_exc()

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
