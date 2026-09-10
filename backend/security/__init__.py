"""security — JWT auth, encrypted PII, offline queue, Indian checksums."""

from __future__ import annotations

from backend.security.aes_encrypt import decrypt_pii, encrypt_pii, pii_roundtrip_demo
from backend.security.checksums import (
    is_valid_aadhaar,
    is_valid_pan,
    pan_check_char,
    verhoeff_valid,
)
from backend.security.jwt_auth import (
    DEMO_OFFICERS,
    authenticate,
    create_token,
    decode_token,
    get_current_user,
    permissions_for,
    require_roles,
)
from backend.security.offline_queue import OfflineQueue

__all__ = [
    "authenticate",
    "create_token",
    "decode_token",
    "get_current_user",
    "require_roles",
    "permissions_for",
    "DEMO_OFFICERS",
    "encrypt_pii",
    "decrypt_pii",
    "pii_roundtrip_demo",
    "OfflineQueue",
    "verhoeff_valid",
    "is_valid_aadhaar",
    "is_valid_pan",
    "pan_check_char",
]