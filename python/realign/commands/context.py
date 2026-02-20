"""Context command for agent-scoped context inspection."""

import os

from rich.console import Console

from ..db import MAX_SESSION_NUM, get_database

console = Console()


def context_show_command() -> int:
    """Show minimal current agent context details from ALINE_AGENT_ID."""
    agent_id = (os.environ.get("ALINE_AGENT_ID") or "").strip()
    if not agent_id:
        console.print(
            "[red]Context failed:[/red] Context is unavailable outside the OneContext "
            "environment. Please run this command inside a OneContext dashboard session."
        )
        return 1

    try:
        db = get_database(read_only=True)

        agent_info = db.get_agent_info(agent_id)
        sessions = db.get_sessions_by_agent_id(agent_id, limit=MAX_SESSION_NUM)

        if agent_info is None and not sessions:
            console.print(f"[red]Agent not found:[/red] {agent_id}")
            return 1

        title = ((agent_info.title or "").strip() if agent_info else "") or "-"
        description = ((agent_info.description or "").strip() if agent_info else "") or "-"

        console.print(f"[bold]Context Title[/bold]: {title}")
        console.print(f"[bold]Description[/bold]: {description}")
        console.print("[bold]Sessions[/bold]:")

        if sessions:
            for row in sessions:
                session_id = (getattr(row, "id", "") or "").strip() or "-"
                session_title = (getattr(row, "session_title", "") or "").strip() or "-"
                console.print(f"{session_id}: {session_title}")
        else:
            console.print("[dim](none)[/dim]")

        return 0
    except Exception as e:
        console.print(f"[red]Error showing agent context:[/red] {e}")
        return 1
