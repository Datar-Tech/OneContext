"""Terminal backend interface for native terminal support.

This module defines the protocol for terminal backends (iTerm2, Kitty)
that allow the Aline Dashboard to control native terminal windows
instead of using tmux for rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class TerminalInfo:
    """Information about a terminal tab/window managed by a backend."""

    terminal_id: str  # Aline internal ID (UUID)
    session_id: str  # Backend-specific session ID (iTerm2 session_id or Kitty window_id)
    name: str  # Display name
    active: bool = False  # Whether this terminal is currently focused
    claude_session_id: str | None = None  # Claude Code session ID if applicable
    context_id: str | None = None  # Aline context ID
    provider: str | None = None  # Terminal provider (claude, codex, etc.)
    attention: str | None = None  # Attention state (permission_request, stop, etc.)
    created_at: float | None = None  # Unix timestamp when terminal was created
    metadata: dict[str, str] = field(default_factory=dict)  # Additional metadata


@runtime_checkable
class TerminalBackend(Protocol):
    """Protocol for terminal backends.

    Implementations must provide async methods to:
    - Create new terminal tabs
    - Focus/switch to existing tabs
    - Close tabs
    - List all managed tabs
    """

    async def create_tab(
        self,
        command: str,
        terminal_id: str,
        *,
        name: str | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> str | None:
        """Create a new terminal tab.

        Args:
            command: The command to run in the new tab
            terminal_id: Aline internal terminal ID
            name: Optional display name for the tab
            env: Optional environment variables to set
            cwd: Optional working directory

        Returns:
            Backend-specific session ID, or None if creation failed
        """
        ...

    async def focus_tab(self, session_id: str, *, steal_focus: bool = False) -> bool:
        """Switch to/focus a terminal tab.

        Args:
            session_id: Backend-specific session ID
            steal_focus: If True, also bring the terminal window to front.
                         If False, switch tab but keep focus on Dashboard.

        Returns:
            True if successful, False otherwise
        """
        ...

    async def close_tab(self, session_id: str) -> bool:
        """Close a terminal tab.

        Args:
            session_id: Backend-specific session ID

        Returns:
            True if successful, False otherwise
        """
        ...

    async def list_tabs(self) -> list[TerminalInfo]:
        """List all terminal tabs managed by this backend.

        Returns:
            List of TerminalInfo objects for each managed tab
        """
        ...

    async def is_available(self) -> bool:
        """Check if this backend is available and usable.

        Returns:
            True if the backend can be used, False otherwise
        """
        ...

    def get_backend_name(self) -> str:
        """Get the human-readable name of this backend.

        Returns:
            Backend name (e.g., "iTerm2", "Kitty")
        """
        ...
