"""Helpers for resolving user identity (uid -> name/email)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .auth import get_auth_headers, is_logged_in
from .config import ReAlignConfig
from .logging_config import setup_logger

logger = setup_logger("realign.user_directory", "user_directory.log")


@dataclass
class UserProfile:
    uid: str
    user_name: Optional[str] = None
    user_email: Optional[str] = None


def fetch_remote_user_profile(uid: str, *, timeout: float = 2.5) -> Optional[UserProfile]:
    """Fetch user profile from server by UID (requires login)."""
    uid = str(uid or "").strip()
    if not uid or not HTTPX_AVAILABLE:
        return None
    if not is_logged_in():
        return None

    headers = get_auth_headers()
    if not headers:
        return None

    config = ReAlignConfig.load()
    backend_url = (config.share_backend_url or "https://realign-server.vercel.app").rstrip("/")
    url = f"{backend_url}/api/users/{quote(uid, safe='')}"

    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
    except Exception as e:
        logger.debug(f"Remote user lookup request failed for uid={uid}: {e}")
        return None

    if response.status_code in (401, 403, 404):
        return None

    try:
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        logger.debug(f"Failed to parse remote user lookup response for uid={uid}: {e}")
        return None

    user_obj = payload.get("user", payload) if isinstance(payload, dict) else {}
    user_name = str(user_obj.get("user_name") or "").strip() if isinstance(user_obj, dict) else ""
    user_email = str(user_obj.get("user_email") or "").strip() if isinstance(user_obj, dict) else ""

    if not user_name and not user_email:
        return None

    return UserProfile(
        uid=uid,
        user_name=user_name or None,
        user_email=user_email or None,
    )
