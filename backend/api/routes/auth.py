"""auth.py — officer login & current-session endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.security.jwt_auth import authenticate, get_current_user

router = APIRouter(prefix="/api", tags=["Auth"])


class LoginBody(BaseModel):
    username: str
    password: str


@router.post("/auth/login", summary="Log in as a checkpoint officer")
def login(body: LoginBody) -> dict[str, Any]:
    """Authenticate an officer and return a 12h JWT (HS256)."""
    return authenticate(body.username, body.password)


@router.get("/auth/me", summary="Current officer profile")
def me(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return user


@router.get("/me", summary="Current officer profile (shortcut)")
def me_shortcut(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return user