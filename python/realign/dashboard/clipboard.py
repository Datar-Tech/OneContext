"""Clipboard helpers for the dashboard."""

from __future__ import annotations

import os
import shutil
import subprocess


def _run_copy(command: list[str], text: str) -> bool:
    try:
        return (
            subprocess.run(
                command,
                input=text,
                text=True,
                capture_output=False,
                check=False,
            ).returncode
            == 0
        )
    except Exception:
        return False


def copy_text(app, text: str) -> bool:
    if not text:
        return False

    if shutil.which("pbcopy"):
        if _run_copy(["pbcopy"], text):
            return True

    if os.name == "nt" and shutil.which("clip"):
        if _run_copy(["clip"], text):
            return True

    if shutil.which("wl-copy"):
        if _run_copy(["wl-copy"], text):
            return True

    if shutil.which("xclip"):
        if _run_copy(["xclip", "-selection", "clipboard"], text):
            return True

    if shutil.which("xsel"):
        if _run_copy(["xsel", "--clipboard", "--input"], text):
            return True

    try:
        app.copy_to_clipboard(text)
        return True
    except Exception:
        return False
