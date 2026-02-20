#!/usr/bin/env python3
"""Aline restore commands - Restore session data from database to original formats."""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from rich.console import Console
from rich.table import Table

from ..db.sqlite_db import SQLiteDatabase
from ..config import ReAlignConfig

console = Console()


def get_db() -> SQLiteDatabase:
    """Get database instance."""
    config = ReAlignConfig.load()
    db_path = Path(config.sqlite_db_path).expanduser()
    db = SQLiteDatabase(str(db_path))
    db.initialize()
    return db


def restore_claude_command(
    session_selector: Optional[str] = None,
    output_dir: Optional[Path] = None,
    list_sessions: bool = False,
) -> int:
    """
    Restore Claude sessions from database to original JSONL format.

    Args:
        session_selector: Session index, range, or UUID prefix. None for all.
        output_dir: Output directory (defaults to /tmp/aline_restore)
        list_sessions: If True, just list available sessions without restoring.

    Returns:
        Exit code (0 for success)
    """
    db = get_db()

    # Get all Claude sessions
    conn = db._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT id, session_file_path, session_title, started_at, last_activity_at, total_turns
           FROM sessions
           WHERE session_type = 'claude'
           ORDER BY last_activity_at DESC"""
    )
    sessions = cursor.fetchall()

    if not sessions:
        console.print("[yellow]No Claude sessions found in database.[/yellow]")
        return 0

    # List mode
    if list_sessions:
        table = Table(title="Claude Sessions in Database")
        table.add_column("#", style="dim")
        table.add_column("Session ID", style="cyan")
        table.add_column("Title", style="green")
        table.add_column("Turns", style="yellow")
        table.add_column("Last Activity", style="magenta")

        for idx, session in enumerate(sessions, 1):
            session_id = session["id"]
            title = session["session_title"] or "(no title)"
            if len(title) > 50:
                title = title[:47] + "..."
            turns = session["total_turns"] or 0
            last_activity = session["last_activity_at"] or ""
            if last_activity and len(last_activity) > 19:
                last_activity = last_activity[:19]

            table.add_row(
                str(idx),
                session_id[:12] + "...",
                title,
                str(turns),
                last_activity,
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(sessions)} sessions[/dim]")
        return 0

    # Parse session selector
    selected_sessions = []
    if session_selector is None:
        # All sessions
        selected_sessions = [dict(s) for s in sessions]
    else:
        # Try to parse as number/range first
        session_list = list(sessions)
        try:
            indices = _parse_selector(session_selector, len(session_list))
            for idx in indices:
                selected_sessions.append(dict(session_list[idx - 1]))
        except ValueError:
            # Try as UUID prefix
            for session in sessions:
                if session["id"].startswith(session_selector):
                    selected_sessions.append(dict(session))
                    break

    if not selected_sessions:
        console.print(f"[red]No sessions matching '{session_selector}' found.[/red]")
        return 1

    # Set up output directory
    if output_dir is None:
        output_dir = Path("/tmp/aline_restore")
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold]Restoring {len(selected_sessions)} Claude session(s) to {output_dir}[/bold]\n"
    )

    restored_count = 0
    for session in selected_sessions:
        session_id = session["id"]
        session_title = session["session_title"] or "untitled"

        # Get all turns for this session
        turns = db.get_turns_for_session(session_id)
        if not turns:
            console.print(f"  [yellow]Session {session_id[:12]}... has no turns, skipping[/yellow]")
            continue

        # Collect all JSONL content
        all_lines: List[str] = []
        for turn in turns:
            content = db.get_turn_content(turn.id)
            if content:
                # Content is stored as JSONL (each line is a JSON object)
                for line in content.strip().split("\n"):
                    if line.strip():
                        all_lines.append(line)

        if not all_lines:
            console.print(
                f"  [yellow]Session {session_id[:12]}... has no content, skipping[/yellow]"
            )
            continue

        # Write to output file
        output_file = output_dir / f"{session_id}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for line in all_lines:
                f.write(line + "\n")

        restored_count += 1
        console.print(
            f"  [green]✓[/green] {session_id[:12]}... → {output_file.name} "
            f"({len(turns)} turns, {len(all_lines)} lines)"
        )

    console.print(
        f"\n[bold green]Restored {restored_count} session(s) to {output_dir}[/bold green]"
    )
    return 0


def restore_codex_command(
    session_selector: Optional[str] = None,
    output_dir: Optional[Path] = None,
    list_sessions: bool = False,
) -> int:
    """
    Restore Codex sessions from database to original JSONL format.

    Args:
        session_selector: Session index, range, or UUID prefix. None for all.
        output_dir: Output directory (defaults to /tmp/aline_restore)
        list_sessions: If True, just list available sessions without restoring.

    Returns:
        Exit code (0 for success)
    """
    db = get_db()

    # Get all Codex sessions
    conn = db._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT id, session_file_path, session_title, started_at, last_activity_at, total_turns
           FROM sessions
           WHERE session_type = 'codex'
           ORDER BY last_activity_at DESC"""
    )
    sessions = cursor.fetchall()

    if not sessions:
        console.print("[yellow]No Codex sessions found in database.[/yellow]")
        return 0

    # List mode
    if list_sessions:
        table = Table(title="Codex Sessions in Database")
        table.add_column("#", style="dim")
        table.add_column("Session ID", style="cyan")
        table.add_column("Title", style="green")
        table.add_column("Turns", style="yellow")
        table.add_column("Last Activity", style="magenta")

        for idx, session in enumerate(sessions, 1):
            session_id = session["id"]
            title = session["session_title"] or "(no title)"
            if len(title) > 50:
                title = title[:47] + "..."
            turns = session["total_turns"] or 0
            last_activity = session["last_activity_at"] or ""
            if last_activity and len(last_activity) > 19:
                last_activity = last_activity[:19]

            table.add_row(
                str(idx),
                session_id[:12] + "...",
                title,
                str(turns),
                last_activity,
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(sessions)} sessions[/dim]")
        return 0

    # Parse session selector
    selected_sessions = []
    if session_selector is None:
        # All sessions
        selected_sessions = [dict(s) for s in sessions]
    else:
        # Try to parse as number/range first
        session_list = list(sessions)
        try:
            indices = _parse_selector(session_selector, len(session_list))
            for idx in indices:
                selected_sessions.append(dict(session_list[idx - 1]))
        except ValueError:
            # Try as UUID prefix
            for session in sessions:
                if session["id"].startswith(session_selector):
                    selected_sessions.append(dict(session))
                    break

    if not selected_sessions:
        console.print(f"[red]No sessions matching '{session_selector}' found.[/red]")
        return 1

    # Set up output directory
    if output_dir is None:
        output_dir = Path("/tmp/aline_restore")
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold]Restoring {len(selected_sessions)} Codex session(s) to {output_dir}[/bold]\n"
    )

    restored_count = 0
    for session in selected_sessions:
        session_id = session["id"]
        session_title = session["session_title"] or "untitled"

        # Get all turns for this session
        turns = db.get_turns_for_session(session_id)
        if not turns:
            console.print(f"  [yellow]Session {session_id[:12]}... has no turns, skipping[/yellow]")
            continue

        # Collect all JSONL content
        all_lines: List[str] = []
        for turn in turns:
            content = db.get_turn_content(turn.id)
            if content:
                # Content is stored as JSONL (each line is a JSON object)
                for line in content.strip().split("\n"):
                    if line.strip():
                        all_lines.append(line)

        if not all_lines:
            console.print(
                f"  [yellow]Session {session_id[:12]}... has no content, skipping[/yellow]"
            )
            continue

        # Write to output file
        output_file = output_dir / f"{session_id}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for line in all_lines:
                f.write(line + "\n")

        restored_count += 1
        console.print(
            f"  [green]✓[/green] {session_id[:12]}... → {output_file.name} "
            f"({len(turns)} turns, {len(all_lines)} lines)"
        )

    console.print(
        f"\n[bold green]Restored {restored_count} session(s) to {output_dir}[/bold green]"
    )
    return 0


def _parse_selector(selector: str, max_count: int) -> List[int]:
    """
    Parse a selector string like "1", "1-3", "1,3,5-7" into list of indices.

    Args:
        selector: Selector string
        max_count: Maximum valid index

    Returns:
        List of 1-based indices

    Raises:
        ValueError: If selector is invalid
    """
    indices = []
    parts = selector.split(",")

    for part in parts:
        part = part.strip()
        if "-" in part:
            # Range
            start, end = part.split("-", 1)
            start_idx = int(start.strip())
            end_idx = int(end.strip())
            if start_idx < 1 or end_idx > max_count or start_idx > end_idx:
                raise ValueError(f"Invalid range: {part}")
            indices.extend(range(start_idx, end_idx + 1))
        else:
            # Single number
            idx = int(part)
            if idx < 1 or idx > max_count:
                raise ValueError(f"Invalid index: {idx}")
            indices.append(idx)

    return sorted(set(indices))
