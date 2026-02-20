"""Utility functions for detecting and integrating with Codex."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List


def _codex_session_roots() -> list[Path]:
    """Return all Codex session root directories to scan (best-effort)."""
    roots: list[Path] = []

    # Default Codex home: ~/.codex/sessions
    roots.append(Path.home() / ".codex" / "sessions")

    # Aline-managed per-terminal or per-agent CODEX_HOME isolation: ~/.aline/codex_homes/*/sessions
    try:
        from .codex_home import aline_codex_homes_dir, codex_sessions_dir_for_home

        homes = aline_codex_homes_dir()
        if homes.exists():
            for child in homes.iterdir():
                if not child.is_dir():
                    continue
                if child.name.startswith("agent-"):
                    # New layout: agent-<id>/<terminal_id>/sessions
                    try:
                        for grandchild in child.iterdir():
                            if grandchild.is_dir():
                                nested_sessions = codex_sessions_dir_for_home(grandchild)
                                if nested_sessions.exists():
                                    roots.append(nested_sessions)
                    except Exception:
                        continue
                else:
                    # Terminal layout: <terminal_id>/sessions
                    direct_sessions = codex_sessions_dir_for_home(child)
                    if direct_sessions.exists():
                        roots.append(direct_sessions)
    except Exception:
        pass

    # De-dup + only keep existing directories
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        try:
            rr = r.expanduser()
        except Exception:
            rr = r
        key = str(rr)
        if key in seen:
            continue
        seen.add(key)
        if rr.exists():
            out.append(rr)
    return out


def find_codex_sessions_for_project(project_path: Path, days_back: int = 7) -> List[Path]:
    """
    Find Codex sessions for a given project path.

    Codex stores sessions in ~/.codex/sessions/{YYYY}/{MM}/{DD}/
    with all projects mixed together. We need to search by date
    and filter by the 'cwd' field in session metadata.

    Args:
        project_path: The absolute path to the project
        days_back: Number of days to look back (default: 7)

    Returns:
        List of session file paths that match the project, sorted by timestamp (newest first)
    """
    codex_session_roots = _codex_session_roots()
    if not codex_session_roots:
        return []

    # Normalize project path for comparison
    abs_project_path = str(project_path.resolve())

    matching_sessions = []

    # Search through recent days in each root (YYYY/MM/DD layout).
    for root in codex_session_roots:
        for days_ago in range(days_back + 1):
            target_date = datetime.now() - timedelta(days=days_ago)
            date_path = (
                root
                / str(target_date.year)
                / f"{target_date.month:02d}"
                / f"{target_date.day:02d}"
            )

            if not date_path.exists():
                continue

            # Check all session files in this date directory
            for session_file in date_path.glob("rollout-*.jsonl"):
                try:
                    # Read first line to get session metadata
                    with open(session_file, "r", encoding="utf-8") as f:
                        first_line = f.readline()
                        if first_line:
                            data = json.loads(first_line)
                            if data.get("type") == "session_meta":
                                session_cwd = data.get("payload", {}).get("cwd", "")
                                # Match the project path
                                if session_cwd == abs_project_path:
                                    matching_sessions.append(session_file)
                except (json.JSONDecodeError, IOError):
                    # Skip malformed or unreadable files
                    continue

    # Sort by modification time, newest first
    matching_sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return matching_sessions


def get_latest_codex_session(project_path: Path, days_back: int = 7) -> Optional[Path]:
    """
    Get the most recent Codex session for a given project.

    Args:
        project_path: The absolute path to the project
        days_back: Number of days to look back (default: 7)

    Returns:
        Path to the most recent session file, or None if not found
    """
    sessions = find_codex_sessions_for_project(project_path, days_back)
    return sessions[0] if sessions else None


def get_codex_sessions_dir() -> Optional[Path]:
    """
    Get the Codex sessions base directory if it exists.

    Returns:
        Path to ~/.codex/sessions if it exists, None otherwise
    """
    # Preserve old API: return default location if present, else the first discovered root.
    codex_sessions = Path.home() / ".codex" / "sessions"
    if codex_sessions.exists():
        return codex_sessions
    roots = _codex_session_roots()
    return roots[0] if roots else None


def auto_detect_codex_sessions(
    project_path: Path, fallback_path: Optional[str] = None, days_back: int = 7
) -> Optional[Path]:
    """
    Auto-detect the most recent Codex session for a project.

    Priority:
    1. Most recent Codex session matching project path
    2. fallback_path parameter (if provided)
    3. Default: None (let caller decide fallback)

    Args:
        project_path: The absolute path to the project
        fallback_path: Optional fallback path if auto-detection fails
        days_back: Number of days to look back for sessions (default: 7)

    Returns:
        Path to use for session history, or None if not found
    """
    # Try to find the latest Codex session for this project
    latest_session = get_latest_codex_session(project_path, days_back)
    if latest_session:
        return latest_session

    # Use fallback if provided
    if fallback_path:
        return Path(os.path.expanduser(fallback_path))

    return None
