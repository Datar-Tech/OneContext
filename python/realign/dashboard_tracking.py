"""Helpers for dashboard-only tracking policy.

When dashboard-only mode is enabled, Claude/Codex sessions should be tracked
only if they were launched from the Aline dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def is_dashboard_only_enabled(config: object) -> bool:
    """Return whether dashboard-only tracking is enabled."""
    return bool(getattr(config, "dashboard_only", False))


def _normalize_path(path: str | Path) -> str:
    try:
        return str(Path(path).expanduser().resolve())
    except Exception:
        try:
            return str(Path(path).expanduser())
        except Exception:
            return str(path)


def _load_dashboard_claude_allowlist(db: object) -> tuple[set[str], set[str]]:
    """Load known Claude sessions/paths associated with dashboard terminals."""
    session_ids: set[str] = set()
    transcript_paths: set[str] = set()
    try:
        agents = db.list_agents(status=None, limit=10000)  # type: ignore[attr-defined]
    except Exception:
        agents = []

    for agent in agents:
        provider = str(getattr(agent, "provider", "") or "").strip().lower()
        session_type = str(getattr(agent, "session_type", "") or "").strip().lower()
        if provider not in {"claude", "claude_code"} and session_type not in {
            "claude",
            "claude_code",
        }:
            continue

        session_id = str(getattr(agent, "session_id", "") or "").strip()
        if session_id:
            session_ids.add(session_id)

        transcript_path = str(getattr(agent, "transcript_path", "") or "").strip()
        if transcript_path:
            transcript_paths.add(_normalize_path(transcript_path))

    return session_ids, transcript_paths


def is_dashboard_codex_session(session_file: Path) -> bool:
    """Return True if a Codex session file is under Aline-managed CODEX_HOME."""
    try:
        from .codex_home import codex_home_owner_from_session_file

        return codex_home_owner_from_session_file(session_file) is not None
    except Exception:
        return False


def filter_discovered_sessions_for_dashboard_only(
    *,
    session_type: str,
    sessions: Iterable[Path],
    db: object | None = None,
) -> list[Path]:
    """Filter discovered sessions according to dashboard-only policy."""
    stype = str(session_type or "").strip().lower()
    session_list = [Path(s) for s in sessions]

    if stype == "codex":
        return [s for s in session_list if is_dashboard_codex_session(s)]

    if stype != "claude":
        return session_list

    own_db = False
    if db is None:
        try:
            from .db import get_database

            db = get_database(read_only=True)
            own_db = True
        except Exception:
            db = None

    if db is None:
        return []

    try:
        allowed_ids, allowed_paths = _load_dashboard_claude_allowlist(db)
        out: list[Path] = []
        for session_file in session_list:
            session_id = str(session_file.stem or "").strip()
            if session_id and session_id in allowed_ids:
                out.append(session_file)
                continue
            if _normalize_path(session_file) in allowed_paths:
                out.append(session_file)
        return out
    finally:
        if own_db:
            try:
                db.close()  # type: ignore[attr-defined]
            except Exception:
                pass


def is_session_trackable_for_dashboard_only(
    *,
    session_type: str,
    session_file: Path,
    session_id: str = "",
    terminal_id: str = "",
    agent_id: str = "",
    db: object | None = None,
) -> bool:
    """Return whether this session is allowed under dashboard-only mode."""
    stype = str(session_type or "").strip().lower()
    if stype not in {"claude", "codex"}:
        return True

    terminal_id = str(terminal_id or "").strip()
    agent_id = str(agent_id or "").strip()
    if terminal_id or agent_id:
        return True

    if stype == "codex":
        return is_dashboard_codex_session(session_file)

    # Claude fallback: allow only sessions already mapped to dashboard terminals.
    own_db = False
    if db is None:
        try:
            from .db import get_database

            db = get_database(read_only=True)
            own_db = True
        except Exception:
            db = None

    if db is None:
        return False

    try:
        allowed_ids, allowed_paths = _load_dashboard_claude_allowlist(db)
        sid = str(session_id or "").strip() or str(session_file.stem or "").strip()
        if sid and sid in allowed_ids:
            return True
        return _normalize_path(session_file) in allowed_paths
    finally:
        if own_db:
            try:
                db.close()  # type: ignore[attr-defined]
            except Exception:
                pass
