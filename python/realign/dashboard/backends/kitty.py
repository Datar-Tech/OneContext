"""Kitty terminal backend using the remote control protocol.

This backend allows the Aline Dashboard to create and manage terminal tabs
directly in Kitty, bypassing tmux for rendering. This provides native
terminal performance and features.

Requirements:
    Kitty must be configured with remote control enabled:

    In ~/.config/kitty/kitty.conf:
        allow_remote_control yes
        listen_on unix:/tmp/kitty_aline

    Or start kitty with:
        kitty --listen-on unix:/tmp/kitty_aline
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from typing import Any

from ..terminal_backend import TerminalBackend, TerminalInfo
from ...logging_config import setup_logger

logger = setup_logger("realign.dashboard.backends.kitty", "dashboard.log")

# Environment variable used to identify Aline-managed terminals
ENV_TERMINAL_ID = "ALINE_TERMINAL_ID"
ENV_TERMINAL_PROVIDER = "ALINE_TERMINAL_PROVIDER"

# Default socket path for Kitty remote control
DEFAULT_SOCKET = "/tmp/kitty_aline"


class KittyBackend:
    """Kitty terminal backend using remote control."""

    def __init__(self, socket: str = DEFAULT_SOCKET) -> None:
        self.socket = socket
        self._terminals: dict[str, TerminalInfo] = {}  # terminal_id -> TerminalInfo
        self._window_to_terminal: dict[str, str] = {}  # kitty window_id -> terminal_id

    def get_backend_name(self) -> str:
        return "Kitty"

    async def is_available(self) -> bool:
        """Check if Kitty and remote control are available."""
        # Check if kitty is installed
        if not shutil.which("kitty"):
            logger.debug("Kitty not found in PATH")
            return False

        # Check if kitten (Kitty's helper) is available
        if not shutil.which("kitten"):
            logger.debug("kitten command not found")
            return False

        # Check if the socket exists (Kitty is running with remote control)
        if not os.path.exists(self.socket):
            logger.debug(f"Kitty socket not found: {self.socket}")
            return False

        # Try to connect
        result = await self._run_kitten("ls")
        return result is not None

    async def _run_kitten(self, *args: str) -> dict[str, Any] | list[Any] | None:
        """Run a kitten @ command and return JSON output.

        Args:
            *args: Arguments to pass to kitten @

        Returns:
            Parsed JSON response, or None on error
        """
        cmd = ["kitten", "@", "--to", f"unix:{self.socket}", *args]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                if stderr:
                    logger.debug(f"kitten error: {stderr.decode()}")
                return None

            if stdout:
                try:
                    return json.loads(stdout.decode())
                except json.JSONDecodeError:
                    # Some commands return non-JSON output
                    return {"output": stdout.decode().strip()}

            return {}

        except Exception as e:
            logger.error(f"Failed to run kitten command: {e}")
            return None

    async def _run_kitten_simple(self, *args: str) -> str | None:
        """Run a kitten @ command and return raw output.

        Args:
            *args: Arguments to pass to kitten @

        Returns:
            Raw stdout, or None on error
        """
        cmd = ["kitten", "@", "--to", f"unix:{self.socket}", *args]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                if stderr:
                    logger.debug(f"kitten error: {stderr.decode()}")
                return None

            return stdout.decode().strip() if stdout else ""

        except Exception as e:
            logger.error(f"Failed to run kitten command: {e}")
            return None

    async def create_tab(
        self,
        command: str,
        terminal_id: str,
        *,
        name: str | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> str | None:
        """Create a new tab in Kitty.

        Args:
            command: The command to run in the new tab
            terminal_id: Aline internal terminal ID
            name: Optional display name for the tab
            env: Optional environment variables to set
            cwd: Optional working directory

        Returns:
            Kitty window ID (as string), or None if creation failed
        """
        created_at = time.time()

        # Build environment with Aline identifiers
        full_env = dict(env or {})
        full_env[ENV_TERMINAL_ID] = terminal_id

        # Build kitten launch arguments
        args = ["launch", "--type=tab"]

        # Add environment variables
        for key, value in full_env.items():
            args.extend(["--env", f"{key}={value}"])

        # Add working directory
        if cwd:
            args.extend(["--cwd", cwd])

        # Add tab title
        if name:
            args.extend(["--tab-title", name])

        # Add the command (as a shell command)
        import os as _os
        user_shell = _os.environ.get("SHELL", "/bin/sh")
        args.append(f"{user_shell} -lc {_shell_quote(command)}")

        # Run the launch command
        result = await self._run_kitten_simple(*args)

        if result is None:
            logger.error("Failed to create Kitty tab")
            return None

        # The launch command returns the window ID
        window_id = result.strip()
        if not window_id:
            logger.error("No window ID returned from Kitty launch")
            return None

        logger.info(
            f"Created Kitty tab: window_id={window_id}, terminal_id={terminal_id}"
        )

        # Track the terminal
        info = TerminalInfo(
            terminal_id=terminal_id,
            session_id=window_id,
            name=name or "Terminal",
            active=True,
            provider=full_env.get(ENV_TERMINAL_PROVIDER),
            created_at=created_at,
        )
        self._terminals[terminal_id] = info
        self._window_to_terminal[window_id] = terminal_id

        return window_id

    async def focus_tab(self, session_id: str, *, steal_focus: bool = False) -> bool:
        """Switch to a terminal tab.

        Args:
            session_id: Kitty window ID
            steal_focus: If True, also bring Kitty window to front

        Returns:
            True if successful
        """
        # Focus the window/tab
        result = await self._run_kitten_simple(
            "focus-window", "--match", f"id:{session_id}"
        )

        if result is None:
            logger.warning(f"Failed to focus Kitty window: {session_id}")
            return False

        logger.debug(f"Focused Kitty tab: window_id={session_id}")

        # If not stealing focus, try to return focus to the previous app
        # Note: Kitty doesn't have a direct way to do this, so we rely on
        # the window manager. On macOS, we could use AppleScript.
        if not steal_focus:
            # Best effort: the focus-window command in Kitty typically
            # doesn't steal app focus, just switches the active tab
            pass

        return True

    async def close_tab(self, session_id: str) -> bool:
        """Close a terminal tab.

        Args:
            session_id: Kitty window ID

        Returns:
            True if successful
        """
        result = await self._run_kitten_simple(
            "close-window", "--match", f"id:{session_id}"
        )

        if result is None:
            logger.warning(f"Failed to close Kitty window: {session_id}")
            return False

        logger.info(f"Closed Kitty window: {session_id}")

        # Clean up tracking
        if session_id in self._window_to_terminal:
            terminal_id = self._window_to_terminal.pop(session_id)
            self._terminals.pop(terminal_id, None)

        return True

    async def list_tabs(self) -> list[TerminalInfo]:
        """List all Aline-managed terminal tabs in Kitty.

        Returns:
            List of TerminalInfo for managed tabs
        """
        result = await self._run_kitten("ls")

        if result is None:
            return []

        tabs: list[TerminalInfo] = []

        # Parse the Kitty ls output
        # Structure: [{"id": os_window_id, "tabs": [{"id": tab_id, "windows": [...]}]}]
        try:
            os_windows = result if isinstance(result, list) else []

            for os_window in os_windows:
                if not isinstance(os_window, dict):
                    continue

                for tab in os_window.get("tabs", []):
                    if not isinstance(tab, dict):
                        continue

                    for window in tab.get("windows", []):
                        if not isinstance(window, dict):
                            continue

                        window_id = str(window.get("id", ""))
                        is_focused = window.get("is_focused", False)

                        # Check if this is an Aline-managed terminal
                        terminal_id = self._window_to_terminal.get(window_id)
                        if terminal_id and terminal_id in self._terminals:
                            info = self._terminals[terminal_id]
                            tabs.append(
                                TerminalInfo(
                                    terminal_id=info.terminal_id,
                                    session_id=window_id,
                                    name=info.name,
                                    active=is_focused,
                                    claude_session_id=info.claude_session_id,
                                    context_id=info.context_id,
                                    provider=info.provider,
                                    attention=info.attention,
                                    created_at=info.created_at,
                                    metadata=info.metadata,
                                )
                            )

        except Exception as e:
            logger.error(f"Failed to parse Kitty ls output: {e}")
            return []

        # Sort by creation time (newest first)
        tabs.sort(
            key=lambda t: t.created_at if t.created_at is not None else 0, reverse=True
        )
        return tabs

    def update_terminal_info(
        self, terminal_id: str, **kwargs: str | None
    ) -> bool:
        """Update metadata for a tracked terminal.

        Args:
            terminal_id: Aline internal terminal ID
            **kwargs: Fields to update

        Returns:
            True if terminal was found and updated
        """
        if terminal_id not in self._terminals:
            return False

        info = self._terminals[terminal_id]
        self._terminals[terminal_id] = TerminalInfo(
            terminal_id=info.terminal_id,
            session_id=info.session_id,
            name=info.name,
            active=info.active,
            claude_session_id=kwargs.get("claude_session_id") or info.claude_session_id,
            context_id=kwargs.get("context_id") or info.context_id,
            provider=kwargs.get("provider") or info.provider,
            attention=kwargs.get("attention"),  # Allow clearing attention
            created_at=info.created_at,
            metadata=info.metadata,
        )
        return True


def _shell_quote(s: str) -> str:
    """Quote a string for shell use."""
    import shlex

    return shlex.quote(s)
