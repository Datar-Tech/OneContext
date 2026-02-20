"""Backward-compatible import path for the watcher.

Deprecated: prefer `realign.watcher_core`.

This shim keeps tests and external callers working by ensuring monkeypatching
`realign.mcp_watcher.find_all_active_sessions` affects the watcher implementation.
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path

from .hooks import find_all_active_sessions as _default_find_all_active_sessions
from . import watcher_core as _watcher_core


def find_all_active_sessions(config, project_path: Optional[Path] = None):
    return _default_find_all_active_sessions(config, project_path=project_path)


def _find_all_active_sessions_proxy(config, project_path: Optional[Path] = None):
    # Indirection for compatibility: tests monkeypatch `realign.mcp_watcher.find_all_active_sessions`.
    return find_all_active_sessions(config, project_path=project_path)


# Ensure the core watcher uses the proxy (so patching this module works).
_watcher_core.find_all_active_sessions = _find_all_active_sessions_proxy


# Re-export public API for callers expecting `realign.mcp_watcher`.
ReAlignConfig = _watcher_core.ReAlignConfig
DialogueWatcher = _watcher_core.DialogueWatcher
is_path_blacklisted = _watcher_core.is_path_blacklisted
