"""Module 4 face tests — safe degradation when no live photo / no models."""

from __future__ import annotations

from backend.modules.module4_face import run_face_analysis


def test_no_images_degrades_to_not_run() -> None:
    result = run_face_analysis(None, None)
    assert result["status"] == "not_checked"
    assert result["face_verification"]["status"] == "NOT_RUN"
    assert result["liveness"]["status"] == "NOT_RUN"
    assert result["morphing"]["status"] == "NOT_RUN"
    assert result["document_faces"] == 0


def test_document_only_no_crash(sample_document_bytes) -> None:
    result = run_face_analysis(sample_document_bytes, None)
    assert result["status"] == "not_checked"
    assert result["face_verification"]["status"] == "NOT_RUN"


def test_never_raises_on_garbage_bytes() -> None:
    result = run_face_analysis(b"not an image", b"also not an image")
    assert result["face_verification"]["status"] == "NOT_RUN"
    assert result["liveness"]["status"] == "NOT_RUN"