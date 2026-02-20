"""Migration script to import terminal.json and load.json data into SQLite.

This script migrates:
- ~/.aline/terminal.json -> agents table
- ~/.aline/load.json -> agent_contexts, agent_context_sessions, agent_context_events tables

Usage:
    python -m realign.db.migrate_agents [--backup] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def migrate_terminal_json(db, dry_run: bool = False, silent: bool = False) -> int:
    """Migrate terminal.json to agents table.

    Args:
        db: Database instance
        dry_run: If True, don't actually migrate
        silent: If True, don't print any output

    Returns:
        Number of agents migrated
    """
    path = Path.home() / ".aline" / "terminal.json"
    if not path.exists():
        if not silent:
            print(f"[migrate] terminal.json not found at {path}, skipping")
        return 0

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        if not silent:
            print(f"[migrate] Failed to read terminal.json: {e}")
        return 0

    terminals = payload.get("terminals", {})
    if not isinstance(terminals, dict):
        if not silent:
            print("[migrate] terminal.json has no 'terminals' dict, skipping")
        return 0

    migrated = 0
    for terminal_id, data in terminals.items():
        if not isinstance(terminal_id, str) or not isinstance(data, dict):
            continue

        provider = data.get("provider", "unknown")
        session_type = data.get("session_type", provider)
        session_id = data.get("session_id") or None
        transcript_path = data.get("transcript_path") or None
        cwd = data.get("cwd") or None
        project_dir = data.get("project_dir") or None
        source = data.get("source") or None
        context_id = data.get("context_id") or None
        attention = data.get("attention") or None

        if dry_run:
            if not silent:
                print(f"[dry-run] Would migrate agent: {terminal_id[:8]}... ({provider})")
            migrated += 1
            continue

        try:
            existing = db.get_agent_by_id(terminal_id)
            if existing:
                if not silent:
                    print(f"[migrate] Agent {terminal_id[:8]}... already exists, skipping")
                continue

            db.get_or_create_agent(
                terminal_id,
                provider=provider,
                session_type=session_type,
                session_id=session_id,
                context_id=context_id,
                transcript_path=transcript_path,
                cwd=cwd,
                project_dir=project_dir,
                source=source,
                attention=attention,
            )
            if not silent:
                print(f"[migrate] Migrated agent: {terminal_id[:8]}... ({provider})")
            migrated += 1
        except Exception as e:
            if not silent:
                print(f"[migrate] Failed to migrate agent {terminal_id[:8]}...: {e}")

    return migrated


def migrate_load_json(db, dry_run: bool = False, silent: bool = False) -> int:
    """Migrate load.json to agent_contexts tables.

    Args:
        db: Database instance
        dry_run: If True, don't actually migrate
        silent: If True, don't print any output

    Returns:
        Number of contexts migrated
    """
    path = Path.home() / ".aline" / "load.json"
    if not path.exists():
        if not silent:
            print(f"[migrate] load.json not found at {path}, skipping")
        return 0

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        if not silent:
            print(f"[migrate] Failed to read load.json: {e}")
        return 0

    contexts = payload.get("contexts", [])
    if not isinstance(contexts, list):
        if not silent:
            print("[migrate] load.json has no 'contexts' list, skipping")
        return 0

    migrated = 0
    for ctx_data in contexts:
        if not isinstance(ctx_data, dict):
            continue

        context_id = ctx_data.get("context_id")
        if not context_id:
            # Generate context_id from workspace if not present
            workspace = ctx_data.get("workspace")
            if workspace:
                # Create a deterministic ID from workspace
                import hashlib

                context_id = f"ws-{hashlib.sha256(workspace.encode()).hexdigest()[:12]}"
            else:
                continue

        workspace = ctx_data.get("workspace")
        loaded_at = ctx_data.get("loaded_at")
        context_sessions = ctx_data.get("context_sessions", [])
        context_events = ctx_data.get("context_events", [])

        if dry_run:
            if not silent:
                print(
                    f"[dry-run] Would migrate context: {context_id} "
                    f"(sessions={len(context_sessions)}, events={len(context_events)})"
                )
            migrated += 1
            continue

        try:
            existing = db.get_agent_context_by_id(context_id)
            if existing:
                if not silent:
                    print(f"[migrate] Context {context_id} already exists, updating links")
            else:
                db.get_or_create_agent_context(
                    context_id,
                    workspace=workspace,
                    loaded_at=loaded_at,
                )
                if not silent:
                    print(f"[migrate] Created context: {context_id}")

            # Update session links (silently skips if session not in DB)
            if context_sessions:
                for session_id in context_sessions:
                    db.link_session_to_agent_context(context_id, session_id)
                if not silent:
                    print(f"[migrate] Linked {len(context_sessions)} sessions to {context_id}")

            # Update event links (silently skips if event not in DB)
            if context_events:
                for event_id in context_events:
                    db.link_event_to_agent_context(context_id, event_id)
                if not silent:
                    print(f"[migrate] Linked {len(context_events)} events to {context_id}")

            migrated += 1
        except Exception as e:
            if not silent:
                print(f"[migrate] Failed to migrate context {context_id}: {e}")

    return migrated


def backup_json_files() -> bool:
    """Create backup of JSON files.

    Returns:
        True if backup was successful or files don't exist
    """
    files = [
        Path.home() / ".aline" / "terminal.json",
        Path.home() / ".aline" / "load.json",
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for path in files:
        if not path.exists():
            continue

        backup_path = path.with_suffix(f".json.bak.{timestamp}")
        try:
            backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[backup] Created backup: {backup_path}")
        except Exception as e:
            print(f"[backup] Failed to backup {path}: {e}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate terminal.json and load.json to SQLite database"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of JSON files before migration",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually doing it",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database (uses config default if not specified)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Aline Agents Migration: JSON -> SQLite")
    print("=" * 60)

    if args.dry_run:
        print("[mode] DRY RUN - no changes will be made")

    # Backup if requested
    if args.backup and not args.dry_run:
        print("\n[step] Creating backups...")
        if not backup_json_files():
            print("[error] Backup failed, aborting migration")
            sys.exit(1)

    # Initialize database
    if not args.dry_run:
        if args.db_path:
            from .sqlite_db import SQLiteDatabase

            print(f"\n[step] Initializing database at {args.db_path}")
            db = SQLiteDatabase(args.db_path)
            if not db.initialize():
                print("[error] Database initialization failed")
                sys.exit(1)
        else:
            from . import get_database

            print("\n[step] Using configured database path")
            db = get_database(read_only=False)
    else:
        db = None

    # Migrate terminal.json
    print("\n[step] Migrating terminal.json -> agents table")
    agents_migrated = migrate_terminal_json(db, dry_run=args.dry_run)
    print(f"[result] Migrated {agents_migrated} agents")

    # Migrate load.json
    print("\n[step] Migrating load.json -> agent_contexts tables")
    contexts_migrated = migrate_load_json(db, dry_run=args.dry_run)
    print(f"[result] Migrated {contexts_migrated} contexts")

    # Cleanup
    if db:
        db.close()

    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"  Agents migrated: {agents_migrated}")
    print(f"  Contexts migrated: {contexts_migrated}")
    print("=" * 60)


if __name__ == "__main__":
    main()
