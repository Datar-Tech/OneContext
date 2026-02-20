#!/usr/bin/env python3
"""
Claude Code PermissionRequest Hook - Set attention state when permission is needed

When Claude Code needs user approval for a tool, this script is called.
It sets the @aline_attention tmux window option to notify the dashboard.

Usage:
    This script is called via Claude Code's PermissionRequest hook mechanism,
    receiving stdin JSON and environment variables.

Environment variables:
    ALINE_TERMINAL_ID - Terminal ID for the current window
    ALINE_INNER_TMUX_SOCKET - Inner tmux socket name
    ALINE_INNER_TMUX_SESSION - Inner tmux session name

stdin JSON format:
    {
        "session_id": "...",
        "tool_name": "...",
        "tool_input": {...}
    }
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path


def get_signal_dir() -> Path:
    """Get the signal directory for permission requests."""
    signal_dir = Path.home() / ".aline" / ".signals" / "permission_request"
    signal_dir.mkdir(parents=True, exist_ok=True)
    return signal_dir


def write_signal_file(terminal_id: str, tool_name: str = "") -> None:
    """Write a signal file to notify the dashboard of a permission request."""
    try:
        signal_dir = get_signal_dir()
        timestamp_ms = int(time.time() * 1000)
        signal_file = signal_dir / f"{terminal_id}_{timestamp_ms}.signal"
        tmp_file = signal_dir / f"{terminal_id}_{timestamp_ms}.signal.tmp"

        signal_data = {
            "terminal_id": terminal_id,
            "tool_name": tool_name,
            "timestamp": time.time(),
            "hook_event": "PermissionRequest",
        }

        tmp_file.write_text(json.dumps(signal_data, indent=2))
        tmp_file.replace(signal_file)
    except Exception:
        pass  # Best effort


def main():
    """Main function"""
    try:
        # Read stdin JSON (optional, we mainly care about setting attention)
        stdin_data = sys.stdin.read()
        try:
            data = json.loads(stdin_data) if stdin_data.strip() else {}
        except json.JSONDecodeError:
            data = {}

        # From environment
        terminal_id = os.environ.get("ALINE_TERMINAL_ID", "")
        inner_socket = os.environ.get("ALINE_INNER_TMUX_SOCKET", "")
        inner_session = os.environ.get("ALINE_INNER_TMUX_SESSION", "")

        # Try to get terminal_id from tmux if not in env
        if not terminal_id:
            try:
                terminal_id = (
                    subprocess.run(
                        ["tmux", "display-message", "-p", "#{@aline_terminal_id}"],
                        text=True,
                        capture_output=True,
                        check=False,
                    ).stdout
                    or ""
                ).strip()
            except Exception:
                terminal_id = ""

        # Set attention state on the tmux window
        try:
            if terminal_id and inner_socket and inner_session:
                # Find the window with matching terminal_id
                proc = subprocess.run(
                    [
                        "tmux",
                        "-L",
                        inner_socket,
                        "list-windows",
                        "-t",
                        inner_session,
                        "-F",
                        "#{window_id}\t#{@aline_terminal_id}",
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                for line in (proc.stdout or "").splitlines():
                    parts = line.split("\t", 1)
                    if len(parts) != 2:
                        continue
                    window_id, win_terminal_id = parts
                    if win_terminal_id == terminal_id:
                        subprocess.run(
                            [
                                "tmux",
                                "-L",
                                inner_socket,
                                "set-option",
                                "-w",
                                "-t",
                                window_id,
                                "@aline_attention",
                                "permission_request",
                            ],
                            check=False,
                        )
                        break
            else:
                # Fallback: try to set on current window
                window_id = (
                    subprocess.run(
                        ["tmux", "display-message", "-p", "#{window_id}"],
                        text=True,
                        capture_output=True,
                        check=False,
                    ).stdout
                    or ""
                ).strip()

                should_tag = bool(terminal_id)
                if not should_tag:
                    try:
                        context_id = (
                            subprocess.run(
                                ["tmux", "display-message", "-p", "#{@aline_context_id}"],
                                text=True,
                                capture_output=True,
                                check=False,
                            ).stdout
                            or ""
                        ).strip()
                        should_tag = bool(context_id)
                    except Exception:
                        should_tag = False

                if window_id and should_tag:
                    subprocess.run(
                        [
                            "tmux",
                            "set-option",
                            "-w",
                            "-t",
                            window_id,
                            "@aline_attention",
                            "permission_request",
                        ],
                        check=False,
                    )
        except Exception:
            pass

        # Write signal file to notify dashboard (triggers file watcher refresh)
        if terminal_id:
            tool_name = data.get("tool_name", "")
            write_signal_file(terminal_id, tool_name)

        # Exit 0 - don't block the permission request
        sys.exit(0)

    except Exception as e:
        # Fail silently to not affect Claude Code's normal operation
        sys.exit(0)


if __name__ == "__main__":
    main()
