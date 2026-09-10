"""Module 4 — face matching (document portrait vs live capture)."""

from __future__ import annotations

from typing import Any

from backend.modules.module4_face.embedder import cosine_similarity, embed


def match_faces(document_bgr, live_bgr) -> dict[str, Any]:
    """Compare a document portrait against a live capture.

    Returns a dict with status MATCH/NOMATCH/NOT_RUN, match_score,
    passed, method and details.
    """
    if document_bgr is None or live_bgr is None:
        return {
            "status": "NOT_RUN",
            "match_score": None,
            "passed": None,
            "method": None,
            "details": "No live capture provided — face verification skipped.",
        }
    try:
        doc_emb, doc_method = embed(document_bgr)
        live_emb, live_method = embed(live_bgr)
        if doc_emb and live_emb:
            score = cosine_similarity(doc_emb, live_emb)
        else:
            score = 0.0
        method = f"{doc_method}+{live_method}"
        passed = score >= 0.55
        return {
            "status": "MATCH" if passed else "NOMATCH",
            "match_score": round(score, 4),
            "passed": bool(passed),
            "method": method,
            "details": (
                f"Portrait vs live similarity = {score:.3f} ({method}). "
                + ("Identities consistent." if passed else "Identities DO NOT match.")
            ),
        }
    except Exception as exc:  # pragma: no cover
        return {
            "status": "NOT_RUN",
            "match_score": None,
            "passed": None,
            "method": None,
            "details": f"Face matching error: {exc}",
        }