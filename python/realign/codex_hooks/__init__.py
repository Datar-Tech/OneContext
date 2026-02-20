"""Codex CLI hook integrations.

This package contains best-effort hooks/installers for integrating with the
OpenAI Codex CLI. Unlike Claude Code, Codex hooks are configured via Codex
configuration files (e.g. config.toml notify hook).
"""

from __future__ import annotations

from pathlib import Path


def codex_notify_signal_dir() -> Path:
    """Directory for fallback notify signals (when direct DB enqueue fails)."""
    return Path.home() / ".aline" / ".signals" / "codex_notify"

