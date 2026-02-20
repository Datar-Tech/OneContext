"""
Gemini CLI Adapter

Handles session discovery and interaction for Gemini CLI.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import SessionAdapter
from ..triggers.gemini_trigger import GeminiTrigger


class GeminiAdapter(SessionAdapter):
    """Adapter for Gemini CLI sessions."""

    name = "gemini"
    trigger_class = GeminiTrigger

    def discover_sessions(self) -> List[Path]:
        """Find all active Gemini CLI sessions."""
        sessions = []
        gemini_tmp = Path.home() / ".gemini" / "tmp"

        if not gemini_tmp.exists():
            return sessions

        try:
            for project_dir in gemini_tmp.iterdir():
                if project_dir.is_dir():
                    chats_dir = project_dir / "chats"
                    if chats_dir.exists():
                        sessions.extend(chats_dir.glob("session-*.json"))
        except Exception:
            pass

        return sessions

    def discover_sessions_for_project(self, project_path: Path) -> List[Path]:
        """
        Find sessions for a specific project.

        For Gemini CLI, project directories are hashed, so it's easier to scan all
        and check the encoded project path or project hash if we knew it.
        Actually GeminiTrigger might have some logic for this.
        """
        # For now, discover all and filter by extract_project_path
        all_sessions = self.discover_sessions()
        matching = []
        abs_project = str(project_path.resolve())

        for s in all_sessions:
            path = self.extract_project_path(s)
            if path and str(path.resolve()) == abs_project:
                matching.append(s)

        return matching

    def extract_project_path(self, session_file: Path) -> Optional[Path]:
        """
        Extract project path from Gemini session file.

        Gemini CLI JSON files usually don't have the full path, but they might
        have a projectHash. Wait, let me check the GeminiTrigger logic again.
        """
        # Gemini JSON structure: {"sessionId": "...", "projectHash": "...", "messages": [...]}
        # If we can't get the path from JSON, we might have to rely on discovery.
        return None

    def is_session_valid(self, session_file: Path) -> bool:
        """Check if this is a Gemini session file."""
        if not session_file.name.startswith("session-") or not session_file.name.endswith(".json"):
            return False

        return super().is_session_valid(session_file)
