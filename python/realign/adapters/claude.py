"""
Claude Code Adapter

Handles session discovery and interaction for Claude Code CLI.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

from .base import SessionAdapter
from ..triggers.claude_trigger import ClaudeTrigger
from ..claude_detector import get_claude_project_name, find_claude_sessions_dir


class ClaudeAdapter(SessionAdapter):
    """Adapter for Claude Code CLI sessions."""

    name = "claude"
    trigger_class = ClaudeTrigger

    def discover_sessions(self) -> List[Path]:
        """
        Find all Claude Code sessions from ALL projects.

        Scans ~/.claude/projects/ and returns all session files.
        """
        sessions = []
        claude_base = Path.home() / ".claude" / "projects"

        if not claude_base.exists():
            return sessions

        try:
            for project_dir in claude_base.iterdir():
                if project_dir.is_dir():
                    # Find all sessions in this project directory
                    all_sessions = self.get_all_sessions_in_dir(project_dir)
                    sessions.extend(all_sessions)
        except Exception:
            pass

        return sessions

    def discover_sessions_for_project(self, project_path: Path) -> List[Path]:
        """Find sessions for a specific project."""
        claude_dir = find_claude_sessions_dir(project_path)
        if not claude_dir:
            return []

        # For Claude, we usually just care about the latest one
        latest = self.get_latest_session_in_dir(claude_dir)
        return [latest] if latest else []

    def extract_project_path(self, session_file: Path) -> Optional[Path]:
        """
        Extract project path from Claude session file.

        For Claude, the project path is encoded in the parent directory name.
        Example: -Users-name-project -> /Users/name/project
        """
        try:
            # Must be in .claude/projects/
            if ".claude" not in str(session_file):
                return None

            parent_name = session_file.parent.name
            if not parent_name.startswith("-"):
                return None

            # Simple decoding: replace leading '-' with '/' and all other '-' with '/'
            # This is a heuristic as we can't distinguish original '-' from '/'
            # But standard paths don't have '-' often in directory names, or at least this
            # matches the standard encoding for /Users/...
            project_path_str = "/" + parent_name[1:].replace("-", "/")
            project_path = Path(project_path_str)

            # If path exists locally, return it immediately
            if project_path.exists():
                return project_path

            # Fallback: read cwd from the session JSONL (more reliable when the encoded
            # directory name is ambiguous due to '-' in real paths).
            cwd_path: Optional[Path] = None
            try:
                with session_file.open("r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= 20:
                            break
                        raw = line.strip()
                        if not raw:
                            continue
                        try:
                            obj = json.loads(raw)
                        except Exception:
                            continue
                        cwd = obj.get("cwd")
                        if isinstance(cwd, str) and cwd.strip():
                            cwd_path = Path(cwd.strip())
                            # If cwd exists locally, prefer it
                            if cwd_path.exists():
                                return cwd_path
                            break  # Found cwd, stop searching
            except Exception:
                pass

            # Return the best available path even if it doesn't exist locally.
            # This allows importing sessions from other machines (e.g., SWEBench).
            # Prefer cwd from JSONL as it's more accurate than path decoding.
            if cwd_path is not None:
                return cwd_path
            return project_path

        except Exception:
            return None

    def get_all_sessions_in_dir(self, directory: Path) -> List[Path]:
        """Get all session files in a Claude project directory."""
        # Claude sessions are usually UUID.jsonl
        sessions = list(directory.glob("*.jsonl"))
        # Filter out agent-*.jsonl (subtasks)
        sessions = [s for s in sessions if not s.name.startswith("agent-")]
        return sessions

    def get_latest_session_in_dir(self, directory: Path) -> Optional[Path]:
        """Get the latest session file in a Claude project directory."""
        sessions = self.get_all_sessions_in_dir(directory)

        if not sessions:
            return None

        sessions.sort(key=lambda s: s.stat().st_mtime, reverse=True)
        return sessions[0]

    def is_session_valid(self, session_file: Path) -> bool:
        """Check if this is a Claude session file."""
        # Claude format check
        if not session_file.name.endswith(".jsonl"):
            return False

        # Check parent structure ~/.claude/projects/
        parts = session_file.parts
        if ".claude" in parts and "projects" in parts:
            return True

        return super().is_session_valid(session_file)
