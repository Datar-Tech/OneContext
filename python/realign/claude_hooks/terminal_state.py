"""Best-effort persistence for mapping a terminal tab to a Claude session.

This module is imported by hook scripts that are executed as standalone files,
so it must be dependency-light and robust to failures.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any, Optional


def _state_path() -> Path:
    override = os.environ.get("ALINE_TERMINAL_STATE_PATH", "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return Path.home() / ".aline" / "terminal.json"


def _lock_path() -> Path:
    return _state_path().with_suffix(".json.lock")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_db():
    """Get database connection (lazy import to avoid circular deps in hooks)."""
    try:
        from ..db import get_database

        return get_database(read_only=False)
    except Exception:
        try:
            import sys

            root = Path(__file__).resolve().parents[2]
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            from realign.db import get_database  # type: ignore

            return get_database(read_only=False)
        except Exception:
            return None


def _write_to_db(
    *,
    terminal_id: str,
    provider: str,
    session_type: str,
    session_id: str,
    transcript_path: str = "",
    cwd: str = "",
    project_dir: str = "",
    source: str = "",
    context_id: Optional[str] = None,
    attention: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> bool:
    """Write terminal mapping to database (best-effort).

    Returns True if successful, False otherwise.
    """
    try:
        db = _get_db()
        if not db:
            return False

        # Force source to agent mapping when agent_id is known
        if agent_id:
            source = f"agent:{agent_id}"

        # Check if agent exists
        existing = db.get_agent_by_id(terminal_id)
        if existing:
            # Update existing agent
            db.update_agent(
                terminal_id,
                provider=provider,
                session_type=session_type,
                session_id=session_id if session_id else None,
                transcript_path=transcript_path if transcript_path else None,
                cwd=cwd if cwd else None,
                project_dir=project_dir if project_dir else None,
                source=source if source else None,
                context_id=context_id,
                attention=attention,
            )
        else:
            # Create new agent
            db.get_or_create_agent(
                terminal_id,
                provider=provider,
                session_type=session_type,
                session_id=session_id if session_id else None,
                context_id=context_id,
                transcript_path=transcript_path if transcript_path else None,
                cwd=cwd if cwd else None,
                project_dir=project_dir if project_dir else None,
                source=source if source else None,
                attention=attention,
            )

        # Link session to agent if both are available (V19+)
        if session_id and agent_id:
            try:
                db.update_session_agent_id(session_id, agent_id)
            except Exception:
                pass

        # WindowLink: record terminal/session association (V23)
        try:
            db.insert_window_link(
                terminal_id=terminal_id,
                agent_id=agent_id,
                session_id=session_id,
                provider=provider,
                source=source,
                ts=time.time(),
            )
        except Exception:
            pass

        # Note: Don't close - get_database() returns a singleton
        return True
    except Exception:
        return False


def update_terminal_mapping(
    *,
    terminal_id: str,
    provider: str,
    session_type: str,
    session_id: str,
    transcript_path: str = "",
    cwd: str = "",
    project_dir: str = "",
    source: str = "",
    context_id: Optional[str] = None,
    attention: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> None:
    """Update terminal->session binding.

    Writes to both:
    1. SQLite database (primary storage, V15+)
    2. ~/.aline/terminal.json (backward compatibility fallback)

    Concurrency: uses a simple fcntl lock file for JSON; last writer wins, but updates are atomic.
    """
    # Phase 1: Write to database (best-effort, don't fail if DB unavailable)
    if agent_id:
        source = f"agent:{agent_id}"

    _write_to_db(
        terminal_id=terminal_id,
        provider=provider,
        session_type=session_type,
        session_id=session_id,
        transcript_path=transcript_path,
        cwd=cwd,
        project_dir=project_dir,
        source=source,
        context_id=context_id,
        attention=attention,
        agent_id=agent_id,
    )

    # Phase 2: Write to JSON (backward compatibility)
    state_path = _state_path()
    lock_path = _lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)

        payload = _read_json(state_path)
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("version", 1)
        terminals = payload.setdefault("terminals", {})
        if not isinstance(terminals, dict):
            terminals = {}
            payload["terminals"] = terminals

        terminals[terminal_id] = {
            "provider": provider,
            "session_type": session_type,
            "session_id": session_id,
            "transcript_path": transcript_path,
            "cwd": cwd,
            "project_dir": project_dir,
            "source": source,
            "context_id": context_id,
            "attention": attention,
            "updated_at": time.time(),
        }

        tmp_path = state_path.with_suffix(".json.tmp")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(state_path)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            os.close(fd)
        except Exception:
            pass
