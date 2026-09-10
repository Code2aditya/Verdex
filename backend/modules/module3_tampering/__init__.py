"""
module3_tampering — Document Tampering Detection
VERIDEX | Module 3

Sub-components:
  preprocessing   — image load, deskew, denoise, CNN tensor prep   (Anubhav)
  qr_check        — QR/barcode decode + OCR cross-check            (Anubhav)
  ela             — Error Level Analysis                            (Raj)
  cnn_classifier  — EfficientNet-B0 forgery classifier             (Raj)
  noise_analysis  — sensor noise / copy-move detection             (Raj)
  gradcam         — Grad-CAM explainability heatmap                (Raj)
  tamper_pipeline — orchestrates all sub-components → TamperResult (Raj+Anubhav)

The CNN classifier and Grad-CAM heatmap require torch. When torch is
absent the pipeline still runs (ELA + noise + QR), redistributing the
missing signal weights and noting it in the result warnings.

Primary entry point
-------------------
  from backend.modules.module3_tampering import run_tamper_detection

  result = run_tamper_detection(source, ocr_fields=ocr_dict)
  # result.tamper_score  → float [0-1]
  # result.heatmap_url   → "/static/heatmaps/<uuid>.jpg" or ""
  # result.flags         → list[str]
  # result.to_api_dict() → JSON-serialisable dict for FastAPI
"""

from .preprocessing   import preprocess, PreprocessResult, to_pil, from_pil, load_from_bytes
from .qr_check        import check_qr,   QRCheckResult,    BarcodeHit
from .ela             import run_ela,    ELAResult
from .noise_analysis  import analyze_noise, NoiseResult
from .tamper_pipeline import run_tamper_detection, TamperResult

try:
    from .cnn_classifier import classify, CNNResult, get_model, get_target_layer
    from .gradcam        import generate_heatmap, GradCAMResult
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    TORCH_AVAILABLE = False

__all__ = [
    "run_tamper_detection",
    "TamperResult",
    "preprocess",
    "PreprocessResult",
    "to_pil",
    "from_pil",
    "load_from_bytes",
    "check_qr",
    "QRCheckResult",
    "BarcodeHit",
    "run_ela",
    "ELAResult",
    "analyze_noise",
    "NoiseResult",
    "TORCH_AVAILABLE",
]