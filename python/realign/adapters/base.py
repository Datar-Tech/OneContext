"""
Session Adapter Abstract Base Class

Defines the standard interface for CLI-specific session handling.
Each adapter encapsulates all logic for a specific AI coding CLI:
- Session discovery
- Project path extraction
- Turn detection (via composition with TurnTrigger)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any, Type
import logging

from ..triggers.base import TurnTrigger, TurnInfo

logger = logging.getLogger(__name__)


class SessionAdapter(ABC):
    """
    Abstract base class for CLI session adapters.

    Each adapter handles all CLI-specific logic:
    1. Discovering active sessions
    2. Extracting project paths from sessions
    3. Turn detection (delegated to TurnTrigger)
    4. Session metadata extraction

    Subclasses must implement:
    - name: Unique identifier for this adapter
    - trigger_class: The TurnTrigger class to use
    - discover_sessions(): Find all active sessions
    - extract_project_path(): Extract project path from a session
    """

    # Class attributes to be defined by subclasses
    name: str = ""  # e.g., "claude", "codex", "gemini"
    trigger_class: Optional[Type[TurnTrigger]] = None

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._trigger: Optional[TurnTrigger] = None

    @property
    def trigger(self) -> TurnTrigger:
        """Get or create the trigger instance (lazy initialization)."""
        if self._trigger is None:
            if self.trigger_class is None:
                raise NotImplementedError(f"{self.__class__.__name__} must define trigger_class")
            self._trigger = self.trigger_class(self.config)
        return self._trigger

    # ========================================================================
    # Abstract methods - must be implemented by subclasses
    # ========================================================================

    @abstractmethod
    def discover_sessions(self) -> List[Path]:
        """
        Discover all active sessions for this CLI.

        Returns:
            List of session file paths
        """
        pass

    @abstractmethod
    def discover_sessions_for_project(self, project_path: Path) -> List[Path]:
        """
        Discover sessions for a specific project.

        Args:
            project_path: Path to the project root

        Returns:
            List of session file paths for this project
        """
        pass

    @abstractmethod
    def extract_project_path(self, session_file: Path) -> Optional[Path]:
        """
        Extract the project path from a session file.

        Args:
            session_file: Path to the session file

        Returns:
            Path to the project, or None if cannot be determined
        """
        pass

    # ========================================================================
    # Delegated methods - use the trigger for turn detection
    # ========================================================================

    def count_turns(self, session_file: Path) -> int:
        """
        Count complete turns in the session.

        Delegates to the trigger's count_complete_turns method.

        Args:
            session_file: Path to session file

        Returns:
            Number of complete turns
        """
        return self.trigger.count_complete_turns(session_file)

    def is_turn_complete(self, session_file: Path, turn_number: int) -> bool:
        """
        Check if a specific turn is complete.

        Args:
            session_file: Path to session file
            turn_number: Turn number (1-based)

        Returns:
            True if the turn is complete
        """
        return self.trigger.is_turn_complete(session_file, turn_number)

    def extract_turn_info(self, session_file: Path, turn_number: int) -> Optional[TurnInfo]:
        """
        Extract information for a specific turn.

        Args:
            session_file: Path to session file
            turn_number: Turn number (1-based)

        Returns:
            TurnInfo object, or None if turn doesn't exist
        """
        return self.trigger.extract_turn_info(session_file, turn_number)

    def detect_session_format(self, session_file: Path) -> Optional[str]:
        """
        Detect the session file format.

        Args:
            session_file: Path to session file

        Returns:
            Format string, or None if unrecognized
        """
        return self.trigger.detect_session_format(session_file)

    # ========================================================================
    # Optional methods - can be overridden by subclasses
    # ========================================================================

    def get_session_metadata(self, session_file: Path) -> Dict[str, Any]:
        """
        Get metadata about a session file.

        Args:
            session_file: Path to session file

        Returns:
            Dictionary with metadata (project_path, format, turn_count, etc.)
        """
        try:
            project_path = self.extract_project_path(session_file)
            format_str = self.detect_session_format(session_file)
            turn_count = self.count_turns(session_file)

            return {
                "adapter": self.name,
                "session_file": str(session_file),
                "project_path": str(project_path) if project_path else None,
                "format": format_str,
                "turn_count": turn_count,
            }
        except Exception as e:
            logger.warning(f"Error getting session metadata: {e}")
            return {
                "adapter": self.name,
                "session_file": str(session_file),
                "error": str(e),
            }

    def is_session_valid(self, session_file: Path) -> bool:
        """
        Check if a session file is valid for this adapter.

        Args:
            session_file: Path to session file

        Returns:
            True if the session is valid
        """
        try:
            return self.detect_session_format(session_file) is not None
        except Exception:
            return False

    def get_latest_session(self, sessions: Optional[List[Path]] = None) -> Optional[Path]:
        """
        Get the most recently modified session.

        Args:
            sessions: Optional list of sessions (discovers if not provided)

        Returns:
            Path to the most recent session, or None
        """
        if sessions is None:
            sessions = self.discover_sessions()

        if not sessions:
            return None

        # Sort by modification time, most recent first
        try:
            sessions_with_mtime = [(s, s.stat().st_mtime) for s in sessions if s.exists()]
            if not sessions_with_mtime:
                return None

            sessions_with_mtime.sort(key=lambda x: x[1], reverse=True)
            return sessions_with_mtime[0][0]
        except Exception as e:
            logger.warning(f"Error finding latest session: {e}")
            return sessions[0] if sessions else None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"
