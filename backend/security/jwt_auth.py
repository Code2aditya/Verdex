"""JWT officer authentication and role-based access control.

Demo credentials (SSB Police II checkpoint officers):
    officer    / officer123   → SSB-OP-1042  Field Officer Mehta (Rupaidiha)
    supervisor / super123     → SSB-SV-2001  Supervisor Rao (Frontier HQ)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

JWT_SECRET = os.environ.get(
    "VERIDEX_JWT_SECRET", "veridex-sih26188-demo-secret-change-me"
)
JWT_ALG = "HS256"
TOKEN_HOURS = 12

DEMO_OFFICERS = {
    "officer": {
        "password": "officer123",
        "role": "officer",
        "officer_id": "SSB-OP-1042",
        "display_name": "Field Officer Mehta",
        "post": "Rupaidiha (Indo-Nepal)",
    },
    "supervisor": {
        "password": "super123",
        "role": "supervisor",
        "officer_id": "SSB-SV-2001",
        "display_name": "Supervisor Rao",
        "post": "Frontier HQ, SSB Police II",
    },
}

_bearer = HTTPBearer(auto_error=False)


def create_token(username: str, profile: dict[str, Any]) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "role": profile["role"],
        "officer_id": profile["officer_id"],
        "display_name": profile["display_name"],
        "post": profile["post"],
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=TOKEN_HOURS)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Session expired — log in again.") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid authentication token.") from exc


def authenticate(username: str, password: str) -> dict[str, Any]:
    profile = DEMO_OFFICERS.get((username or "").strip().lower())
    if not profile or profile["password"] != password:
        raise HTTPException(status_code=401, detail="Unknown officer or bad credentials.")
    token = create_token(username.strip().lower(), profile)
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": profile["role"],
        "officer_id": profile["officer_id"],
        "display_name": profile["display_name"],
        "post": profile["post"],
        "permissions": permissions_for(profile["role"]),
    }


def permissions_for(role: str) -> list[str]:
    if role == "supervisor":
        return [
            "screen", "view_audit", "verify_chain", "anchor_chain",
            "tamper_demo", "offline_sync", "pii_demo",
        ]
    return ["screen", "view_audit", "verify_chain", "offline_sync", "pii_demo"]


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any]:
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Bearer token required.")
    return decode_token(creds.credentials)


def require_roles(*roles: str) -> Callable[..., dict[str, Any]]:
    def _dep(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        if user.get("role") not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{user.get('role')}' cannot perform this action. Need: {', '.join(roles)}.",
            )
        return user

    return _dep