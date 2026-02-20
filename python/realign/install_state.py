"""Install ownership metadata for robust package upgrades."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

INSTALL_STATE_PATH = Path.home() / ".aline" / "install-state.json"
_VALID_OWNERS = {"pipx", "uv", "pip"}


def _normalize_owner(value: str | None) -> Optional[str]:
    owner = (value or "").strip().lower()
    return owner if owner in _VALID_OWNERS else None


def _owner_from_path(value: str | None) -> Optional[str]:
    path = (value or "").strip().lower()
    if not path:
        return None
    if "/pipx/" in path or "/pipx/venvs/" in path:
        return "pipx"
    if "/uv/tools/" in path or "/.local/share/uv/" in path:
        return "uv"
    if "site-packages" in path or "dist-packages" in path:
        return "pip"
    return None


def _read_shebang_target(script_path: Path) -> Optional[str]:
    try:
        first = script_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip()
    except Exception:
        return None
    if not first.startswith("#!"):
        return None
    return first[2:].strip() or None


def _which_aline() -> Optional[Path]:
    path = shutil.which("aline")
    if not path:
        return None
    try:
        return Path(path)
    except Exception:
        return None


def read_install_state() -> Optional[dict[str, Any]]:
    """Read persisted install owner metadata."""
    try:
        raw = json.loads(INSTALL_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    owner = _normalize_owner(str(raw.get("owner", "")))
    if not owner:
        return None
    return {
        "owner": owner,
        "source": str(raw.get("source", "")).strip(),
        "updated_at": str(raw.get("updated_at", "")).strip(),
        "executable": str(raw.get("executable", "")).strip(),
        "python_executable": str(raw.get("python_executable", "")).strip(),
    }


def write_install_state(
    owner: str,
    *,
    source: str,
    executable: str | None = None,
    python_executable: str | None = None,
) -> Path:
    """Persist install owner metadata for future upgrade routing."""
    normalized = _normalize_owner(owner)
    if not normalized:
        raise ValueError(f"Invalid install owner: {owner}")

    if executable is None:
        executable = shutil.which("aline") or ""
    if python_executable is None:
        python_executable = sys.executable or ""

    payload = {
        "owner": normalized,
        "source": source.strip(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "executable": executable,
        "python_executable": python_executable,
    }
    INSTALL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    INSTALL_STATE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return INSTALL_STATE_PATH


def detect_install_owner_from_environment() -> Optional[str]:
    """Best-effort owner detection from env/path/shebang."""
    env_owner = _normalize_owner(os.environ.get("ALINE_INSTALL_OWNER"))
    if env_owner:
        return env_owner

    owner = _owner_from_path(sys.executable)
    if owner:
        return owner

    aline_path = _which_aline()
    if not aline_path:
        return None

    owner = _owner_from_path(str(aline_path))
    if owner:
        return owner

    shebang_target = _read_shebang_target(aline_path)
    owner = _owner_from_path(shebang_target)
    if owner:
        return owner

    return None


def resolve_install_owner(*, persist_detected: bool = True) -> Optional[str]:
    """Resolve install owner, preferring persisted lock then runtime detection."""
    state = read_install_state()
    if state and state.get("owner"):
        return str(state["owner"])

    detected = detect_install_owner_from_environment()
    if detected and persist_detected:
        try:
            write_install_state(detected, source="runtime_detect")
        except Exception:
            pass
    return detected
