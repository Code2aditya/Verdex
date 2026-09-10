"""Shared rule/config loader for document-type YAML files.

The YAML documents live in Backend/rules/<document_type>.yaml so new
document formats can be added without a code deploy. This mirrors the
working loader from VerdexZip/app/config/loader.py.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

RULES_DIR = Path(__file__).resolve().parents[2] / "rules"


@lru_cache(maxsize=None)
def load_rule(document_type: str) -> dict:
    """Load and cache a document type's YAML rule config."""
    path = RULES_DIR / f"{document_type}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No rule config for document_type={document_type!r} at {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def supported_document_types() -> list[str]:
    return sorted(p.stem for p in RULES_DIR.glob("*.yaml"))