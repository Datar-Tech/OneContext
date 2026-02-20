"""iTerm2 terminal backend using the iTerm2 Python API.

This backend allows the Aline Dashboard to create and manage terminal tabs
directly in iTerm2, bypassing tmux for rendering. This provides native
terminal performance and features (native scrolling, copy/paste, etc.).

Requirements:
    pip install iterm2

iTerm2 Setup:
    1. Enable Python API: Preferences > General > Magic > Enable Python API
    2. Grant automation permissions when prompted
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import shutil
import time
from typing import TYPE_CHECKING

from ..terminal_backend import TerminalBackend, TerminalInfo
from ...logging_config import setup_logger

if TYPE_CHECKING:
    pass

logger = setup_logger("realign.dashboard.backends.iterm2", "dashboard.log")

# Environment variable used to identify Aline-managed terminals
ENV_TERMINAL_ID = "ALINE_TERMINAL_ID"
ENV_TERMINAL_PROVIDER = "ALINE_TERMINAL_PROVIDER"

# Thread pool for running iterm2 operations (iterm2 has its own event loop)
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="iterm2")


def _run_iterm2_coroutine(coro_func):
    """Run an iterm2 coroutine in a separate thread.

    The iterm2 library uses its own event loop via run_until_complete(),
    which conflicts with an already-running asyncio loop (like Textual's).
    This function runs the iterm2 operation in a dedicated thread.

    Args:
        coro_func: An async function that takes (connection) as argument

    Returns:
        The result of the coroutine
    """
    import iterm2

    result = None
    exception = None

    async def wrapper(connection):
        nonlocal result, exception
        try:
            result = await coro_func(connection)
        except Exception as e:
            exception = e

    try:
        iterm2.run_until_complete(wrapper)
    except Exception as e:
        exception = e

    if exception:
        raise exception
    return result


class ITermBackend:
    """iTerm2 terminal backend using the Python API."""

    def __init__(self, right_pane_session_id: str | None = None) -> None:
        self._connection = None
        self._terminals: dict[str, TerminalInfo] = {}  # terminal_id -> TerminalInfo
        self._session_to_terminal: dict[str, str] = {}  # session_id -> terminal_id
        self._right_pane_session_id = right_pane_session_id  # Session ID for split pane mode
        self._pane_sessions: list[str] = []  # Track all sessions in right pane area

    def get_backend_name(self) -> str:
        return "iTerm2"

    async def is_available(self) -> bool:
        """Check if iTerm2 and its Python API are available."""
        # Check if iTerm2 is installed
        if not shutil.which("osascript"):
            return False

        # Check if iterm2 Python package is available
        try:
            import iterm2  # noqa: F401

            return True
        except ImportError:
            logger.debug("iterm2 Python package not installed")
            return False

    async def create_tab(
        self,
        command: str,
        terminal_id: str,
        *,
        name: str | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> str | None:
        """Create a new terminal in iTerm2.

        If split pane mode is active (right_pane_session_id is set), creates
        terminals in the right pane area using horizontal splits.
        Otherwise, creates new tabs.

        Args:
            command: The command to run in the new tab
            terminal_id: Aline internal terminal ID
            name: Optional display name for the tab
            env: Optional environment variables to set
            cwd: Optional working directory

        Returns:
            iTerm2 session ID, or None if creation failed
        """
        try:
            import iterm2
        except ImportError:
            logger.error("iterm2 package not installed")
            return None

        created_at = time.time()

        # Build environment with Aline identifiers
        full_env = dict(env or {})
        full_env[ENV_TERMINAL_ID] = terminal_id

        # Build the command with environment variables
        env_exports = " ".join(
            f"export {k}={_shell_quote(v)};" for k, v in full_env.items()
        )

        # Build full command with cd and env
        full_command = ""
        if cwd:
            full_command = f"cd {_shell_quote(cwd)} && "
        full_command += f"{env_exports} {command}"

        # Capture instance variables for closure
        right_pane_id = self._right_pane_session_id
        pane_sessions = self._pane_sessions

        async def _create(connection) -> str | None:
            app = await iterm2.async_get_app(connection)
            logger.debug(f"Got app, windows count: {len(app.terminal_windows)}")

            session_id = None

            # Split pane mode: create in right pane area
            if right_pane_id:
                logger.debug(f"Split pane mode, right_pane_id={right_pane_id}")

                # Find the right pane session
                target_session = None
                for window in app.terminal_windows:
                    for tab in window.tabs:
                        for session in tab.sessions:
                            if session.session_id == right_pane_id:
                                target_session = session
                                break
                        if target_session:
                            break
                    if target_session:
                        break

                if target_session:
                    # If this is the first terminal and right pane is empty, use it directly
                    if not pane_sessions:
                        logger.debug("Using existing right pane for first terminal")
                        session_id = target_session.session_id
                        # Send command to existing session
                        try:
                            await target_session.async_send_text(full_command + "\n")
                        except Exception as e:
                            logger.error(f"Exception sending command: {e}")
                    else:
                        # Create a new horizontal split in the right pane area
                        logger.debug("Creating horizontal split in right pane")
                        try:
                            profile = iterm2.LocalWriteOnlyProfile()
                            if name:
                                profile.set_name(name)
                                profile.set_allow_title_setting(False)

                            new_session = await target_session.async_split_pane(
                                vertical=False,  # Horizontal split (stack vertically)
                                profile_customizations=profile,
                            )
                            session_id = new_session.session_id

                            # Send command
                            await new_session.async_send_text(full_command + "\n")
                        except Exception as e:
                            logger.error(f"Exception creating split: {e}")
                            return None
                else:
                    logger.warning(f"Right pane session {right_pane_id} not found, falling back to tab mode")

            # Fallback: create new tab
            if session_id is None:
                window = app.current_terminal_window

                if window is None:
                    logger.debug("No current window, creating new one...")
                    try:
                        window = await iterm2.Window.async_create(connection)
                    except Exception as e:
                        logger.error(f"Exception creating window: {type(e).__name__}: {e}")
                        return None
                    if window is None:
                        logger.error("Failed to create iTerm2 window (returned None)")
                        return None

                # Create profile customizations for the tab name
                profile = iterm2.LocalWriteOnlyProfile()
                if name:
                    profile.set_name(name)
                    profile.set_allow_title_setting(False)

                # Create the tab
                logger.debug("Creating tab...")
                try:
                    tab = await window.async_create_tab(
                        profile_customizations=profile,
                    )
                except Exception as e:
                    logger.error(f"Exception creating tab: {type(e).__name__}: {e}")
                    return None

                if tab is None:
                    logger.error("Failed to create iTerm2 tab (returned None)")
                    return None

                session = tab.current_session
                if session is None:
                    logger.error("Tab created but no current session")
                    return None

                session_id = session.session_id

                # Send the command
                logger.debug(f"Sending command: {full_command[:100]}...")
                try:
                    await session.async_send_text(full_command + "\n")
                except Exception as e:
                    logger.error(f"Exception sending command: {type(e).__name__}: {e}")

            logger.info(
                f"Created iTerm2 terminal: session_id={session_id}, terminal_id={terminal_id}"
            )
            return session_id

        try:
            loop = asyncio.get_event_loop()
            session_id = await loop.run_in_executor(_executor, _run_iterm2_coroutine, _create)
        except Exception as e:
            import traceback
            logger.error(f"Failed to create iTerm2 tab: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            return None

        if session_id:
            # Track the terminal
            info = TerminalInfo(
                terminal_id=terminal_id,
                session_id=session_id,
                name=name or "Terminal",
                active=True,
                provider=env.get(ENV_TERMINAL_PROVIDER) if env else None,
                created_at=created_at,
            )
            self._terminals[terminal_id] = info
            self._session_to_terminal[session_id] = terminal_id
            self._pane_sessions.append(session_id)

        return session_id

    async def focus_tab(self, session_id: str, *, steal_focus: bool = False) -> bool:
        """Switch to a terminal tab without stealing window focus.

        Args:
            session_id: iTerm2 session ID
            steal_focus: If True, also bring iTerm2 window to front

        Returns:
            True if successful
        """
        try:
            import iterm2
        except ImportError:
            return False

        async def _focus(connection) -> bool:
            app = await iterm2.async_get_app(connection)

            # Find the session and its tab
            for window in app.terminal_windows:
                for tab in window.tabs:
                    if tab.current_session.session_id == session_id:
                        # Select the tab (this doesn't activate the window)
                        await tab.async_select()

                        # Only activate window if steal_focus is True
                        if steal_focus:
                            await window.async_activate()

                        logger.debug(
                            f"Focused iTerm2 tab: session_id={session_id}, steal_focus={steal_focus}"
                        )
                        return True

            logger.warning(f"Session not found: {session_id}")
            return False

        try:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(_executor, _run_iterm2_coroutine, _focus)
            return success or False
        except Exception as e:
            logger.error(f"Failed to focus iTerm2 tab: {e}")
            return False

    async def close_tab(self, session_id: str) -> bool:
        """Close a terminal tab.

        Args:
            session_id: iTerm2 session ID

        Returns:
            True if successful
        """
        try:
            import iterm2
        except ImportError:
            return False

        async def _close(connection) -> bool:
            app = await iterm2.async_get_app(connection)

            for window in app.terminal_windows:
                for tab in window.tabs:
                    session = tab.current_session
                    if session.session_id == session_id:
                        await session.async_close()
                        logger.info(f"Closed iTerm2 session: {session_id}")
                        return True

            logger.warning(f"Session not found for close: {session_id}")
            return False

        try:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(_executor, _run_iterm2_coroutine, _close)
        except Exception as e:
            logger.error(f"Failed to close iTerm2 tab: {e}")
            return False

        # Clean up tracking
        if success and session_id in self._session_to_terminal:
            terminal_id = self._session_to_terminal.pop(session_id)
            self._terminals.pop(terminal_id, None)

        return success or False

    async def list_tabs(self) -> list[TerminalInfo]:
        """List all Aline-managed terminal tabs in iTerm2.

        Returns:
            List of TerminalInfo for managed tabs
        """
        try:
            import iterm2
        except ImportError:
            return []

        # Capture self reference for closure
        session_to_terminal = self._session_to_terminal
        terminals = self._terminals

        async def _list(connection) -> list[TerminalInfo]:
            tabs: list[TerminalInfo] = []
            app = await iterm2.async_get_app(connection)

            # Get the currently focused session
            focused_session_id = None
            if app.current_terminal_window:
                current_tab = app.current_terminal_window.current_tab
                if current_tab:
                    focused_session_id = current_tab.current_session.session_id

            for window in app.terminal_windows:
                for tab in window.tabs:
                    session = tab.current_session
                    sess_id = session.session_id

                    # Check if this is an Aline-managed terminal
                    term_id = session_to_terminal.get(sess_id)
                    if term_id and term_id in terminals:
                        info = terminals[term_id]
                        # Update active state
                        tabs.append(
                            TerminalInfo(
                                terminal_id=info.terminal_id,
                                session_id=sess_id,
                                name=info.name,
                                active=(sess_id == focused_session_id),
                                claude_session_id=info.claude_session_id,
                                context_id=info.context_id,
                                provider=info.provider,
                                attention=info.attention,
                                created_at=info.created_at,
                                metadata=info.metadata,
                            )
                        )
            return tabs

        try:
            loop = asyncio.get_event_loop()
            tabs = await loop.run_in_executor(_executor, _run_iterm2_coroutine, _list)
            if tabs is None:
                tabs = []
        except Exception as e:
            logger.error(f"Failed to list iTerm2 tabs: {e}")
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
            **kwargs: Fields to update (claude_session_id, context_id, provider, attention, etc.)

        Returns:
            True if terminal was found and updated
        """
        if terminal_id not in self._terminals:
            return False

        info = self._terminals[terminal_id]

        # Update allowed fields
        if "claude_session_id" in kwargs:
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


async def setup_split_pane_layout(dashboard_width_percent: int = 40) -> str | None:
    """Set up a split pane layout with dashboard on left, terminals on right.

    This should be called BEFORE the dashboard starts. It will:
    1. Get the current session (where the command is running)
    2. Split it vertically, creating a right pane for terminals
    3. Return the right pane's session ID for later use

    Args:
        dashboard_width_percent: Percentage of width for dashboard (left pane)

    Returns:
        Session ID of the right pane (for terminals), or None if failed
    """
    try:
        import iterm2
    except ImportError:
        return None

    right_session_id: str | None = None

    async def _setup(connection) -> str | None:
        nonlocal right_session_id
        app = await iterm2.async_get_app(connection)

        window = app.current_terminal_window
        if not window:
            logger.error("No current window for split pane setup")
            return None

        tab = window.current_tab
        if not tab:
            logger.error("No current tab for split pane setup")
            return None

        session = tab.current_session
        if not session:
            logger.error("No current session for split pane setup")
            return None

        # Split vertically - new pane will be on the right
        try:
            right_session = await session.async_split_pane(vertical=True)
            right_session_id = right_session.session_id
            logger.info(f"Created right pane: {right_session_id}")

            # Activate the left pane (dashboard) so user sees it
            await session.async_activate()

            return right_session_id
        except Exception as e:
            logger.error(f"Failed to split pane: {e}")
            return None

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, _run_iterm2_coroutine, _setup)
        return result
    except Exception as e:
        logger.error(f"Failed to setup split pane layout: {e}")
        return None


def setup_split_pane_layout_sync(dashboard_width_percent: int = 40) -> str | None:
    """Synchronous version of setup_split_pane_layout for use before asyncio loop starts.

    Returns:
        Session ID of the right pane, or None if failed
    """
    try:
        import iterm2
    except ImportError:
        return None

    right_session_id: str | None = None

    async def _setup(connection) -> str | None:
        nonlocal right_session_id
        app = await iterm2.async_get_app(connection)

        window = app.current_terminal_window
        if not window:
            logger.error("No current window for split pane setup")
            return None

        tab = window.current_tab
        if not tab:
            logger.error("No current tab for split pane setup")
            return None

        session = tab.current_session
        if not session:
            logger.error("No current session for split pane setup")
            return None

        # Split vertically - new pane will be on the right
        try:
            right_session = await session.async_split_pane(vertical=True)
            right_session_id = right_session.session_id
            logger.info(f"Created right pane: {right_session_id}")

            # Activate the left pane (dashboard) so user sees it
            await session.async_activate()

            return right_session_id
        except Exception as e:
            logger.error(f"Failed to split pane: {e}")
            return None

    try:
        iterm2.run_until_complete(_setup)
        return right_session_id
    except Exception as e:
        logger.error(f"Failed to setup split pane layout: {e}")
        return None
