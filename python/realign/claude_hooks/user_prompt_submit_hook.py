#!/usr/bin/env python3
"""
Claude Code UserPromptSubmit Hook - record user prompt for temp turn title.

Triggered immediately after the user submits a prompt (before Claude responds).
Writes a signal file for the watcher to process after a short delay.

stdin JSON (best-effort):
    {
        "session": {"id": "...", "transcript_path": "..."},
        "prompt": "...",
        "cwd": "...",
        "hook_event_name": "UserPromptSubmit"
    }
"""

from __future__ import annotations

import json
import os
import sys
import time
import subprocess
from pathlib import Path

try:
    from terminal_state import update_terminal_mapping  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    update_terminal_mapping = None


def get_signal_dir() -> Path:
    signal_dir = Path.home() / ".aline" / ".signals" / "user_prompt_submit"
    signal_dir.mkdir(parents=True, exist_ok=True)
    return signal_dir


def main() -> None:
    try:
        stdin_data = sys.stdin.read()
        try:
            data = json.loads(stdin_data) if stdin_data.strip() else {}
        except json.JSONDecodeError:
            data = {}

        session = data.get("session") or {}
        session_id = session.get("id") or data.get("session_id") or data.get("sessionId") or ""
        prompt = data.get("prompt") or data.get("user_prompt") or ""
        transcript_path = session.get("transcript_path") or data.get("transcript_path") or ""
        cwd = data.get("cwd") or ""
        project_dir = os.environ.get("CLAUDE_PROJECT_DIR", cwd)
        terminal_id = os.environ.get("ALINE_TERMINAL_ID", "")
        inner_socket = os.environ.get("ALINE_INNER_TMUX_SOCKET", "")
        inner_session = os.environ.get("ALINE_INNER_TMUX_SESSION", "")
        agent_id = os.environ.get("ALINE_AGENT_ID", "")

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
                terminal_id = terminal_id

        if not session_id and transcript_path:
            transcript_file = Path(transcript_path)
            if transcript_file.suffix == ".jsonl":
                session_id = transcript_file.stem

        if not session_id:
            session_id = f"unknown_{int(time.time() * 1000)}"

        signal_dir = get_signal_dir()
        timestamp_ms = int(time.time() * 1000)
        signal_file = signal_dir / f"{session_id}_{timestamp_ms}.signal"
        tmp_file = signal_dir / f"{session_id}_{timestamp_ms}.signal.tmp"

        # Check for no-track mode
        no_track = os.environ.get("ALINE_NO_TRACK", "") == "1"

        signal_data = {
            "session_id": session_id,
            "terminal_id": terminal_id,
            "agent_id": agent_id,
            "prompt": prompt,
            "transcript_path": transcript_path,
            "cwd": cwd,
            "project_dir": project_dir,
            "timestamp": time.time(),
            "hook_event": "UserPromptSubmit",
        }
        if no_track:
            signal_data["no_track"] = True

        tmp_file.write_text(json.dumps(signal_data, indent=2))
        tmp_file.replace(signal_file)

        # Best-effort: tag the tmux "terminal tab" early (as soon as session id is known).
        try:
            if terminal_id and inner_socket and inner_session and session_id:
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
                                "@aline_provider",
                                "claude",
                            ],
                            check=False,
                        )
                        subprocess.run(
                            [
                                "tmux",
                                "-L",
                                inner_socket,
                                "set-option",
                                "-w",
                                "-t",
                                window_id,
                                "@aline_session_type",
                                "claude",
                            ],
                            check=False,
                        )
                        subprocess.run(
                            [
                                "tmux",
                                "-L",
                                inner_socket,
                                "set-option",
                                "-w",
                                "-t",
                                window_id,
                                "@aline_session_id",
                                session_id,
                            ],
                            check=False,
                        )
                        if transcript_path:
                            subprocess.run(
                                [
                                    "tmux",
                                    "-L",
                                    inner_socket,
                                    "set-option",
                                    "-w",
                                    "-t",
                                    window_id,
                                    "@aline_transcript_path",
                                    transcript_path,
                                ],
                                check=False,
                            )
                        break
            elif session_id:
                # Fallback: when Claude runs hooks without our env vars / inner socket info,
                # tag the *current* tmux window (works when running inside tmux).
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
                            "@aline_provider",
                            "claude",
                        ],
                        check=False,
                    )
                    subprocess.run(
                        [
                            "tmux",
                            "set-option",
                            "-w",
                            "-t",
                            window_id,
                            "@aline_session_type",
                            "claude",
                        ],
                        check=False,
                    )
                    subprocess.run(
                        [
                            "tmux",
                            "set-option",
                            "-w",
                            "-t",
                            window_id,
                            "@aline_session_id",
                            session_id,
                        ],
                        check=False,
                    )
                    if transcript_path:
                        subprocess.run(
                            [
                                "tmux",
                                "set-option",
                                "-w",
                                "-t",
                                window_id,
                                "@aline_transcript_path",
                                transcript_path,
                            ],
                            check=False,
                        )
        except Exception:
            pass

        # Best-effort: persist mapping to ~/.aline/terminal.json to survive tmux restarts.
        try:
            if update_terminal_mapping and terminal_id and session_id:
                update_terminal_mapping(
                    terminal_id=terminal_id,
                    provider="claude",
                    session_type="claude",
                    session_id=session_id,
                    transcript_path=transcript_path,
                    cwd=cwd,
                    project_dir=project_dir,
                    source="UserPromptSubmit",
                    agent_id=agent_id if agent_id else None,
                )
        except Exception:
            pass

        sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
