"""
Database module for ReAlign.
"""

import os
from pathlib import Path
from .base import DatabaseInterface, MAX_SESSION_NUM, AgentSessionLimitExceededError
from .sqlite_db import SQLiteDatabase

_DB_INSTANCE = None
_MIGRATION_DONE = False


def _auto_migrate_agents_data(db: SQLiteDatabase) -> None:
    """Auto-migrate terminal.json and load.json data to SQLite (runs once).

    This is triggered automatically when the schema is upgraded to V15.
    Uses a marker file to avoid running multiple times.
    """
    global _MIGRATION_DONE
    if _MIGRATION_DONE:
        return

    # Check marker file
    marker_path = Path.home() / ".aline" / ".agents_migrated_v15"
    if marker_path.exists():
        _MIGRATION_DONE = True
        return

    # Skip during tests
    if os.getenv("PYTEST_CURRENT_TEST"):
        _MIGRATION_DONE = True
        return

    try:
        from .migrate_agents import migrate_terminal_json, migrate_load_json

        # Run migrations silently (no dry_run)
        agents_count = migrate_terminal_json(db, dry_run=False, silent=True)
        contexts_count = migrate_load_json(db, dry_run=False, silent=True)

        # Create marker file
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(f"Migrated: {agents_count} agents, {contexts_count} contexts\n")

        _MIGRATION_DONE = True
    except Exception:
        # Don't fail if migration fails - JSON fallback will still work
        _MIGRATION_DONE = True


def get_database(
    *, read_only: bool = False, connect_timeout_seconds: float | None = None
) -> DatabaseInterface:
    """Get a database instance.

    - `read_only=False` returns a per-process singleton initialized with migrations.
    - `read_only=True` returns a lightweight read-only instance (no migrations) to avoid
      blocking CLI commands under worker/watcher write load.
    """
    global _DB_INSTANCE

    # Resolution order:
    # 1) Env override (tests/ops): REALIGN_SQLITE_DB_PATH or REALIGN_DB_PATH (legacy)
    # 2) Config: ~/.aline/config.yaml (sqlite_db_path)
    # 3) Default: ~/.aline/db/aline.db
    env_db_path = os.getenv("REALIGN_SQLITE_DB_PATH") or os.getenv("REALIGN_DB_PATH")
    if env_db_path:
        db_path = env_db_path
    else:
        try:
            from ..config import ReAlignConfig

            config = ReAlignConfig.load()
            db_path = config.sqlite_db_path
        except Exception:
            db_path = str(Path.home() / ".aline" / "db" / "aline.db")

    if read_only:
        timeout = (
            float(connect_timeout_seconds)
            if connect_timeout_seconds is not None
            else float(os.getenv("REALIGN_SQLITE_READ_TIMEOUT_SECONDS", "0.2"))
        )
        return SQLiteDatabase(db_path, read_only=True, connect_timeout_seconds=timeout)

    if _DB_INSTANCE is None:
        timeout = (
            float(connect_timeout_seconds)
            if connect_timeout_seconds is not None
            else float(os.getenv("REALIGN_SQLITE_CONNECT_TIMEOUT_SECONDS", "5.0"))
        )
        _DB_INSTANCE = SQLiteDatabase(db_path, connect_timeout_seconds=timeout)
        _DB_INSTANCE.initialize()

        # Auto-migrate JSON data to SQLite (runs once after V15 upgrade)
        _auto_migrate_agents_data(_DB_INSTANCE)

    return _DB_INSTANCE
