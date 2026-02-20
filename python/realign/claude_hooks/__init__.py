"""
Aline Hooks Module

This module contains hooks for integration with AI coding assistants:

- stop_hook: Claude Code Stop hook for turn completion detection
- stop_hook_installer: Automatic installation of hooks to Claude Code settings
"""

from pathlib import Path

# Signal directory for inter-process communication
SIGNAL_DIR = Path.home() / ".aline" / ".signals"


def get_signal_dir() -> Path:
    """Get the signal directory path, creating it if needed."""
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    return SIGNAL_DIR


__all__ = [
    "SIGNAL_DIR",
    "get_signal_dir",
]
