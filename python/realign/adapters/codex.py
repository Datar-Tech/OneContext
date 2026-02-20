"""
Codex Adapter

Handles session discovery and interaction for Codex CLI.
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import SessionAdapter
from ..triggers.codex_trigger import CodexTrigger
from ..codex_detector import find_codex_sessions_for_project


class CodexAdapter(SessionAdapter):
    """Adapter for Codex CLI sessions."""

    name = "codex"
    trigger_class = CodexTrigger

    def _discovery_days_back(self) -> int:
        """
        Limit Codex discovery to recent days to avoid scanning the entire history.

        Set via ALINE_CODEX_DISCOVERY_DAYS (default: 2).
        """
        raw = os.environ.get("ALINE_CODEX_DISCOVERY_DAYS", "2")
        try:
            return max(0, int(raw))
        except Exception:
            return 2

    def discover_sessions(self) -> List[Path]:
        """Find active/recent Codex sessions (bounded scan)."""
        sessions: list[Path] = []
        roots: list[Path] = []
        try:
            from ..codex_detector import _codex_session_roots  # type: ignore[attr-defined]

            roots = _codex_session_roots()
        except Exception:
            roots = [Path.home() / ".codex" / "sessions"]

        days_back = self._discovery_days_back()

        for root in roots:
            if not root.exists():
                continue
            try:
                # Prefer YYYY/MM/DD layout; only scan recent dates.
                if days_back > 0:
                    from datetime import datetime, timedelta

                    now = datetime.now()
                    for days_ago in range(days_back + 1):
                        dt = now - timedelta(days=days_ago)
                        date_path = root / str(dt.year) / f"{dt.month:02d}" / f"{dt.day:02d}"
                        if not date_path.exists():
                            continue
                        sessions.extend(date_path.glob("rollout-*.jsonl"))
                else:
                    # If explicitly disabled, only look at top-level.
                    sessions.extend(root.glob("rollout-*.jsonl"))
            except Exception:
                continue

        return sessions

    def discover_sessions_for_project(self, project_path: Path) -> List[Path]:
        """Find sessions for a specific project."""
        return find_codex_sessions_for_project(project_path)

    def extract_project_path(self, session_file: Path) -> Optional[Path]:
        """Extract project path from Codex session file metadata."""
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if not first_line:
                    return None
                data = json.loads(first_line)
                if data.get("type") == "session_meta":
                    cwd = data.get("payload", {}).get("cwd")
                    if cwd:
                        return Path(cwd)
        except Exception:
            pass
        return None

    def is_session_valid(self, session_file: Path) -> bool:
        """Check if this is a Codex session file."""
        if not session_file.name.startswith("rollout-") or not session_file.name.endswith(".jsonl"):
            return False

        # Check first line for Codex signature
        return super().is_session_valid(session_file)
