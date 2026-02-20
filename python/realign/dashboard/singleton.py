"""Singleton lock for the dashboard.

Goal: prevent multiple concurrent dashboard instances (or tmux attachments) which can
lead to mixed-version confusion during upgrades and a degraded UX.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from ..file_lock import FileLock
from .branding import BRANDING


@dataclass(frozen=True)
class DashboardLockPaths:
    lock_file: Path


@dataclass(frozen=True)
class DashboardLockResult:
    status: Literal["acquired", "skipped", "held"]
    lock: Optional[FileLock] = None


def get_dashboard_lock_paths() -> DashboardLockPaths:
    lock_dir = Path.home() / ".aline" / ".locks"
    return DashboardLockPaths(lock_file=lock_dir / "dashboard.lock")


def acquire_dashboard_lock(*, allow_multi: bool = False) -> DashboardLockResult:
    """Acquire the dashboard singleton lock (non-blocking).

    Returns:
        - status='acquired': lock is held by this process (caller must keep reference)
        - status='held': another process holds the lock (do not start another dashboard)
        - status='skipped': locking disabled by allow_multi/env
    """
    if allow_multi:
        return DashboardLockResult(status="skipped", lock=None)

    env_allow = os.environ.get("ALINE_DASHBOARD_ALLOW_MULTI", "").strip().lower()
    if env_allow in {"1", "true", "yes", "y", "on"}:
        return DashboardLockResult(status="skipped", lock=None)

    paths = get_dashboard_lock_paths()

    candidates: list[Path] = [paths.lock_file]
    try:
        tmpdir = Path(os.environ.get("TMPDIR", "/tmp")).expanduser()
        candidates.append(tmpdir / "aline" / "dashboard.lock")
    except Exception:
        pass

    last_error: Exception | None = None
    for path in candidates:
        lock = FileLock(path, timeout=0.0, inheritable=True)
        try:
            ok = lock.acquire(blocking=False)
        except (PermissionError, OSError) as e:
            last_error = e
            continue

        if ok:
            # Lock acquired for this candidate.
            break

        # Lock file is accessible but held by another process; respect singleton semantics.
        return DashboardLockResult(status="held", lock=None)
    else:
        lock = None

    if lock is None:
        # If we couldn't even create/open the lock file, don't crash the dashboard.
        # In constrained environments (e.g., sandboxed shells), falling back to "no lock"
        # is better than preventing the dashboard from starting.
        if last_error is not None:
            try:
                from .app import logger

                logger.debug(f"Dashboard lock unavailable; proceeding without lock: {last_error}")
            except Exception:
                pass
        return DashboardLockResult(status="skipped", lock=None)

    try:
        # Best-effort: record owner PID for debugging.
        if lock.fd is not None:
            os.ftruncate(lock.fd, 0)
            os.write(lock.fd, f"{os.getpid()}\n".encode("utf-8"))
    except Exception:
        pass

    return DashboardLockResult(status="acquired", lock=lock)


def dashboard_lock_message() -> str:
    return f"{BRANDING.product_name} is already running in another terminal."
