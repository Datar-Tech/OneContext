"""Window layout utilities for native terminal mode.

This module provides functions to arrange the Aline Dashboard and native
terminal windows (iTerm2/Kitty) in a side-by-side layout.
"""

from __future__ import annotations

import os
import subprocess
import sys

from ..logging_config import setup_logger

logger = setup_logger("realign.dashboard.layout", "dashboard.log")


def get_screen_size() -> tuple[int, int]:
    """Get the main screen size on macOS.

    Returns:
        (width, height) in pixels
    """
    if sys.platform != "darwin":
        # Default fallback for non-macOS
        return (1920, 1080)

    try:
        # Use system_profiler to get display resolution
        result = subprocess.run(
            [
                "system_profiler",
                "SPDisplaysDataType",
                "-json",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0 and result.stdout:
            import json

            data = json.loads(result.stdout)
            displays = data.get("SPDisplaysDataType", [])
            for display in displays:
                for monitor in display.get("spdisplays_ndrvs", []):
                    resolution = monitor.get("_spdisplays_resolution", "")
                    # Parse resolution like "2560 x 1440"
                    if " x " in resolution:
                        parts = resolution.split(" x ")
                        try:
                            width = int(parts[0].strip())
                            height = int(parts[1].split()[0].strip())
                            return (width, height)
                        except (ValueError, IndexError):
                            continue

    except Exception as e:
        logger.debug(f"Failed to get screen size via system_profiler: {e}")

    # Fallback: try AppKit
    try:
        import AppKit

        screen = AppKit.NSScreen.mainScreen()
        if screen:
            frame = screen.frame()
            return (int(frame.size.width), int(frame.size.height))
    except Exception as e:
        logger.debug(f"Failed to get screen size via AppKit: {e}")

    # Default fallback
    return (1920, 1080)


def setup_side_by_side_layout(
    terminal_app: str = "iTerm2",
    dashboard_on_left: bool = True,
    dashboard_width_percent: int = 50,
) -> bool:
    """Set up side-by-side layout for Dashboard and terminal windows.

    Arranges the Dashboard and terminal app windows to fill the screen
    side by side, without using full-screen mode.

    Args:
        terminal_app: Terminal application name ("iTerm2" or "Kitty")
        dashboard_on_left: If True, Dashboard on left, terminal on right
        dashboard_width_percent: Percentage of screen width for Dashboard (default 50)

    Returns:
        True if layout was set successfully
    """
    if sys.platform != "darwin":
        logger.warning("Side-by-side layout only supported on macOS")
        return False

    width, height = get_screen_size()
    menu_bar_height = 25  # Approximate macOS menu bar height

    dashboard_width = int(width * dashboard_width_percent / 100)
    terminal_width = width - dashboard_width

    if dashboard_on_left:
        dashboard_x = 0
        terminal_x = dashboard_width
    else:
        terminal_x = 0
        dashboard_x = terminal_width

    # Get the hosting terminal app (the one running the Dashboard)
    hosting_app = _detect_hosting_terminal()

    success = True

    # Position the hosting terminal (Dashboard)
    if hosting_app:
        if not _set_window_bounds(
            hosting_app,
            dashboard_x,
            menu_bar_height,
            dashboard_x + dashboard_width,
            height,
        ):
            success = False

    # Position the target terminal app (for the terminal tabs)
    if terminal_app and terminal_app != hosting_app:
        if not _set_window_bounds(
            terminal_app,
            terminal_x,
            menu_bar_height,
            terminal_x + terminal_width,
            height,
        ):
            success = False

    return success


def _detect_hosting_terminal() -> str | None:
    """Detect which terminal app is hosting the current process.

    Returns:
        Terminal app name ("Terminal", "iTerm2", "Kitty") or None
    """
    term_program = os.environ.get("TERM_PROGRAM", "").strip()

    if term_program in {"Apple_Terminal", "Terminal.app"}:
        return "Terminal"
    if term_program in {"iTerm.app", "iTerm2"} or term_program.startswith("iTerm"):
        return "iTerm2"
    if term_program == "kitty":
        return "Kitty"

    # Try to detect via parent process
    try:
        import psutil

        proc = psutil.Process()
        for parent in proc.parents():
            name = parent.name().lower()
            if "iterm" in name:
                return "iTerm2"
            if "terminal" in name:
                return "Terminal"
            if "kitty" in name:
                return "Kitty"
    except Exception:
        pass

    return None


def _set_window_bounds(
    app_name: str, x1: int, y1: int, x2: int, y2: int
) -> bool:
    """Set window bounds for an application using AppleScript.

    Args:
        app_name: Application name
        x1, y1: Top-left corner coordinates
        x2, y2: Bottom-right corner coordinates

    Returns:
        True if successful
    """
    if sys.platform != "darwin":
        return False

    # Map app names to their AppleScript names
    script_app_name = app_name
    if app_name == "Kitty":
        # Kitty doesn't support AppleScript well, use alternative method
        return _set_kitty_window_bounds(x1, y1, x2, y2)

    # Build AppleScript
    script = f'''
    tell application "{script_app_name}"
        if (count of windows) > 0 then
            set bounds of front window to {{{x1}, {y1}, {x2}, {y2}}}
        end if
    end tell
    '''

    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            logger.warning(
                f"Failed to set {app_name} window bounds: {result.stderr}"
            )
            return False

        logger.debug(f"Set {app_name} window bounds: ({x1}, {y1}, {x2}, {y2})")
        return True

    except Exception as e:
        logger.error(f"Failed to set {app_name} window bounds: {e}")
        return False


def _set_kitty_window_bounds(x1: int, y1: int, x2: int, y2: int) -> bool:
    """Set Kitty window bounds using its remote control.

    Kitty doesn't support AppleScript, so we use its native resize command.
    However, Kitty's remote control doesn't support window positioning,
    so we fall back to AppleScript via System Events.

    Args:
        x1, y1: Top-left corner coordinates
        x2, y2: Bottom-right corner coordinates

    Returns:
        True if successful
    """
    # Try using System Events (requires accessibility permissions)
    script = f'''
    tell application "System Events"
        tell process "kitty"
            if (count of windows) > 0 then
                set position of front window to {{{x1}, {y1}}}
                set size of front window to {{{x2 - x1}, {y2 - y1}}}
            end if
        end tell
    end tell
    '''

    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            logger.warning(f"Failed to set Kitty window bounds: {result.stderr}")
            return False

        logger.debug(f"Set Kitty window bounds: ({x1}, {y1}, {x2}, {y2})")
        return True

    except Exception as e:
        logger.error(f"Failed to set Kitty window bounds: {e}")
        return False


def bring_app_to_front(app_name: str) -> bool:
    """Bring an application to the front.

    Args:
        app_name: Application name

    Returns:
        True if successful
    """
    if sys.platform != "darwin":
        return False

    script = f'''
    tell application "{app_name}"
        activate
    end tell
    '''

    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.returncode == 0
    except Exception:
        return False


def focus_window_without_raising(app_name: str) -> bool:
    """Focus an application's window without bringing it to the front.

    This is tricky on macOS - we want to switch the active tab/window
    within the app without stealing focus from the current app.

    Args:
        app_name: Application name

    Returns:
        True if successful (best effort)
    """
    # This is inherently difficult on macOS. The best we can do is
    # use the terminal backend's focus_tab with steal_focus=False.
    # This function is here for API completeness.
    return True
