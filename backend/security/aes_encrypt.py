"""AES-256 / Fernet PII encryption helpers.

PII extracted from documents is never stored in plaintext. On the
edge node it is encrypted with Fernet (AES-128-CBC + HMAC-SHA256 over
a 256-bit URL-safe key) before entering the audit or offline queues.
The key is generated on first use and cached at backend/security/.fernet_key.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

KEY_PATH = Path(__file__).resolve().parent / ".fernet_key"


def _load_fernet() -> Fernet:
    env_key = os.environ.get("VERIDEX_FERNET_KEY")
    if env_key:
        return Fernet(env_key.encode() if isinstance(env_key, str) else env_key)
    if KEY_PATH.exists():
        return Fernet(KEY_PATH.read_bytes().strip())
    key = Fernet.generate_key()
    KEY_PATH.write_bytes(key)
    return Fernet(key)


FERNET = _load_fernet()


def encrypt_pii(plaintext: str) -> str:
    if plaintext is None:
        raise ValueError("plaintext is required")
    return FERNET.encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_pii(token: str) -> str:
    try:
        return FERNET.decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("PII token is invalid or was encrypted under a different key") from exc


def pii_roundtrip_demo(fields: dict[str, str]) -> dict[str, Any]:
    encrypted = {k: encrypt_pii(v) for k, v in fields.items()}
    decrypted = {k: decrypt_pii(v) for k, v in encrypted.items()}
    return {
        "algorithm": "Fernet (AES-CBC + HMAC-SHA256, 256-bit key)",
        "at_rest": encrypted,
        "decrypted": decrypted,
        "integrity_ok": decrypted == fields,
        "note": "Ciphertext is stored; plaintext is never written to the audit or offline tables.",
    }