"""Best-effort linking between Codex session files and Aline terminals.

Claude Code provides explicit hook callbacks with session identifiers. Codex CLI does not,
so we infer the binding by matching:
- session_meta.cwd (project/workspace path)
- session creation time vs. terminal creation time

This module is intentionally dependency-light so it can be used by the watcher.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Iterable, Optional, Protocol


@dataclass(frozen=True)
class CodexSessionMeta:
    session_file: Path
    cwd: str
    started_at: Optional[datetime] = None
    originator: Optional[str] = None
    source: Optional[str] = None


def _parse_iso8601(ts: str) -> Optional[datetime]:
    raw = (ts or "").strip()
    if not raw:
        return None
    # Common Codex format: 2025-12-23T09:14:28.152Z
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _normalize_cwd(cwd: str | None) -> str:
    raw = (cwd or "").strip()
    if not raw:
        return ""
    try:
        return os.path.normpath(raw)
    except Exception:
        return raw.rstrip("/\\")


def read_codex_session_meta(session_file: Path) -> Optional[CodexSessionMeta]:
    """Extract Codex session metadata from a session file (best-effort)."""
    try:
        with session_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 25:
                    break
                raw = (line or "").strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                # Typical format: {"type":"session_meta","payload":{...}}
                if data.get("type") == "session_meta":
                    payload = data.get("payload") or {}
                    cwd = str(payload.get("cwd") or "").strip()
                    if not cwd:
                        return None
                    started_at = _parse_iso8601(str(payload.get("timestamp") or ""))
                    originator = str(payload.get("originator") or "").strip() or None
                    source = str(payload.get("source") or "").strip() or None
                    return CodexSessionMeta(
                        session_file=session_file,
                        cwd=cwd,
                        started_at=started_at,
                        originator=originator,
                        source=source,
                    )

                # Newer Codex header: first line may have {id, timestamp, git} without "type"
                if i == 0 and "timestamp" in data and "type" not in data:
                    started_at = _parse_iso8601(str(data.get("timestamp") or ""))
                    git = data.get("git") if isinstance(data.get("git"), dict) else {}
                    cwd = ""
                    if isinstance(git, dict):
                        cwd = str(git.get("cwd") or "").strip()
                    if not cwd:
                        cwd = str(data.get("cwd") or "").strip()
                    if not cwd:
                        return None
                    return CodexSessionMeta(session_file=session_file, cwd=cwd, started_at=started_at)
    except OSError:
        return None
    except Exception:
        return None

    return None


class _AgentLike(Protocol):
    id: str
    provider: str
    status: str
    cwd: Optional[str]
    session_id: Optional[str]
    transcript_path: Optional[str]
    created_at: datetime


def select_agent_for_codex_session(
    agents: Iterable[_AgentLike],
    *,
    session: CodexSessionMeta,
    max_time_delta_seconds: Optional[int] = 6 * 60 * 60,
) -> Optional[str]:
    """Pick the best active Codex terminal for a Codex session file (best-effort)."""
    cwd = _normalize_cwd(session.cwd)
    if not cwd:
        return None

    candidates: list[_AgentLike] = []
    for a in agents:
        try:
            if getattr(a, "status", "") != "active":
                continue
            if getattr(a, "provider", "") != "codex":
                continue
            if _normalize_cwd(getattr(a, "cwd", None)) != cwd:
                continue
            # Avoid clobbering an existing binding to a different session.
            existing_sid = (getattr(a, "session_id", None) or "").strip()
            if existing_sid and existing_sid != session.session_file.stem:
                continue
            existing_path = (getattr(a, "transcript_path", None) or "").strip()
            if existing_path and existing_path != str(session.session_file):
                continue
            candidates.append(a)
        except Exception:
            continue

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0].id

    # Pick closest by creation time.
    if session.started_at is not None:
        ref = session.started_at
    else:
        try:
            ref = datetime.fromtimestamp(session.session_file.stat().st_mtime, tz=timezone.utc)
        except OSError:
            ref = datetime.now(tz=timezone.utc)

    best_id: Optional[str] = None
    best_delta: Optional[float] = None
    for a in candidates:
        try:
            created_at = a.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            delta = abs((ref - created_at).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_id = a.id
        except Exception:
            continue

    if best_id is None:
        return None
    if max_time_delta_seconds is not None and best_delta is not None:
        if best_delta > max_time_delta_seconds:
            # Ambiguous: don't bind if terminals are too far from the session start.
            return None
    return best_id
