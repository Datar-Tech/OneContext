"""Agent management commands."""

import uuid
from pathlib import Path

from rich.console import Console

console = Console()


def _get_db():
    """Get database instance."""
    from ..config import ReAlignConfig
    from ..db.sqlite_db import SQLiteDatabase

    config = ReAlignConfig.load()
    db_path = Path(config.sqlite_db_path).expanduser()
    db = SQLiteDatabase(str(db_path))
    db.initialize()
    return db


def agent_new_command(name: str | None = None, desc: str = "") -> int:
    """Create a new agent profile.

    Args:
        name: Display name (random Docker-style name if None).
        desc: Agent description.

    Returns:
        Exit code (0 = success).
    """
    from ..agent_names import generate_agent_name

    agent_id = str(uuid.uuid4())
    display_name = name or generate_agent_name()

    db = _get_db()
    try:
        record = db.get_or_create_agent_info(agent_id, name=display_name)
        if desc:
            record = db.update_agent_info(agent_id, description=desc)

        console.print(f"[bold green]Agent created[/bold green]")
        console.print(f"  id:          {record.id}")
        console.print(f"  name:        {record.name}")
        console.print(f"  title:       {record.title or '(none)'}")
        console.print(f"  description: {record.description or '(none)'}")
        return 0
    finally:
        db.close()


def agent_list_command(*, include_invisible: bool = False) -> int:
    """List agent profiles.

    Returns:
        Exit code (0 = success).
    """
    from rich.table import Table

    db = _get_db()
    try:
        agents = db.list_agent_info(include_invisible=include_invisible)

        if not agents:
            console.print("[dim]No agents yet.[/dim]")
            return 0

        table = Table(title="Agents")
        table.add_column("ID", style="dim", width=8)
        table.add_column("Name", style="bold")
        table.add_column("Title")
        table.add_column("Description")
        table.add_column("Sessions", style="cyan")
        table.add_column("Created", style="dim")

        def _unique_prefixes(ids: list[str], min_len: int = 8) -> list[str]:
            if not ids:
                return []
            max_len = max(len(i) for i in ids)
            length = min_len
            while length <= max_len:
                prefixes = [i[:length] for i in ids]
                if len(set(prefixes)) == len(ids):
                    return prefixes
                length += 2
            return ids

        for agent in agents:
            created_str = agent.created_at.strftime("%Y-%m-%d %H:%M")
            sessions = db.get_sessions_by_agent_id(agent.id)
            if sessions:
                raw_ids = [s.id for s in sessions]
                short_ids = _unique_prefixes(raw_ids, min_len=8)
                session_ids = ", ".join(short_ids)
                sessions_display = f"{len(sessions)} ({session_ids})"
            else:
                sessions_display = "0"
            table.add_row(
                agent.id[:8],
                agent.name,
                agent.title or "",
                agent.description or "",
                sessions_display,
                created_str,
            )

        console.print(table)
        return 0
    finally:
        db.close()
