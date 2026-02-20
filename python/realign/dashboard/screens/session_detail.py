"""Session detail modal for the dashboard."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Static

from ..widgets.copyable_text_area import CopyableTextArea


def _format_dt(dt: object) -> str:
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    if dt is None:
        return "-"
    return str(dt)


class SessionDetailScreen(ModalScreen):
    """Modal that shows session turns and per-turn details."""

    BINDINGS = [Binding("escape", "close", "Close", show=False)]

    DEFAULT_CSS = """
    SessionDetailScreen {
        align: center middle;
    }

    SessionDetailScreen #session-detail-root {
        width: 95%;
        height: 95%;
        padding: 1;
        background: $background;
        border: solid $accent;
    }

    SessionDetailScreen #session-meta {
        height: auto;
        margin-bottom: 1;
    }

    SessionDetailScreen #session-detail-body {
        height: 1fr;
    }

    SessionDetailScreen #turns-table {
        width: 100%;
        height: auto;
        max-height: 16;
    }

    SessionDetailScreen #turn-details {
        width: 100%;
        height: 1fr;
        margin-top: 1;
        border: round $accent;
    }

    SessionDetailScreen #session-close-bar {
        height: auto;
        width: 100%;
        align: center middle;
        margin-top: 1;
    }

    SessionDetailScreen #session-close-btn {
        min-width: 16;
    }
    """

    def __init__(self, session_id: str, *, initial_turn_id: Optional[str] = None) -> None:
        super().__init__()
        self.session_id = session_id
        self.initial_turn_id = initial_turn_id
        self._turn_by_id: dict[str, object] = {}

        self._load_error: Optional[str] = None

    def compose(self) -> ComposeResult:
        with Container(id="session-detail-root"):
            yield Static(id="session-meta")
            with Vertical(id="session-detail-body"):
                yield DataTable(id="turns-table")
                yield CopyableTextArea(
                    "",
                    id="turn-details",
                    read_only=True,
                    show_line_numbers=False,
                    soft_wrap=True,
                )
            with Horizontal(id="session-close-bar"):
                yield Button("Close", id="session-close-btn", variant="default")

    def on_mount(self) -> None:
        turns_table = self.query_one("#turns-table", DataTable)
        turns_table.add_columns("Turn", "Title", "Time")
        turns_table.cursor_type = "row"
        turns_table.show_vertical_scrollbar = True

        self._load_data()
        self._update_display()

        if turns_table.row_count <= 0:
            return

        target_turn_id: Optional[str] = None
        if self.initial_turn_id and self.initial_turn_id in self._turn_by_id:
            target_turn_id = self.initial_turn_id
        elif self._turns:
            target_turn_id = str(getattr(self._turns[0], "id", "")) or None

        if not target_turn_id:
            return

        target_row = 0
        for idx, turn in enumerate(self._turns):
            if str(getattr(turn, "id", "")) == target_turn_id:
                target_row = idx
                break

        turns_table.cursor_coordinate = (target_row, 0)
        self._update_turn_details(target_turn_id)
        turns_table.focus()

    def action_close(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "session-close-btn":
            self.action_close()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "turns-table":
            return
        self._update_turn_details(str(event.row_key.value))

    def _load_data(self) -> None:
        try:
            from ...db import get_database

            db = get_database()

            session = db.get_session_by_id(self.session_id)
            turns = db.get_turns_for_session(self.session_id)

            self._session = session
            self._turns = turns
            self._turn_by_id = {str(t.id): t for t in turns}
            self._load_error = None
        except Exception as e:
            self._session = None
            self._turns = []
            self._turn_by_id = {}
            self._load_error = str(e)

    def _update_display(self) -> None:
        meta = self.query_one("#session-meta", Static)
        if self._load_error:
            meta.update(f"[red]Failed to load session {self.session_id}:[/red] {self._load_error}")
            return

        session_title = getattr(self._session, "session_title", None) if self._session else None
        session_type = getattr(self._session, "session_type", None) if self._session else None
        last_activity_at = (
            getattr(self._session, "last_activity_at", None) if self._session else None
        )
        started_at = getattr(self._session, "started_at", None) if self._session else None

        source_map = {
            "claude": "Claude",
            "codex": "Codex",
            "gemini": "Gemini",
            "shell": "Shell",
            "zsh": "Shell",
        }
        source = source_map.get(session_type or "", session_type or "unknown")

        title_display = session_title or "(no title)"
        if len(title_display) > 120:
            title_display = title_display[:120] + "…"

        meta.update(
            "\n".join(
                [
                    f"[bold]Session[/bold] {self.session_id}",
                    f"[dim]Source:[/dim] {source}",
                    f"[dim]Started:[/dim] {_format_dt(started_at)}    [dim]Last Activity:[/dim] {_format_dt(last_activity_at)}",
                    f"[dim]Title:[/dim] {title_display}",
                ]
            )
        )

        turns_table = self.query_one("#turns-table", DataTable)
        turns_table.clear()

        for turn in self._turns:
            title = (getattr(turn, "llm_title", None) or "").strip() or "(no title)"
            if len(title) > 60:
                title = title[:60] + "…"

            time_str = _format_dt(getattr(turn, "timestamp", None))

            turns_table.add_row(
                str(getattr(turn, "turn_number", "")),
                title,
                time_str,
                key=str(getattr(turn, "id", "")),
            )

    def _update_turn_details(self, turn_id: str) -> None:
        details = self.query_one("#turn-details", CopyableTextArea)
        turn = self._turn_by_id.get(turn_id)
        if not turn:
            details.text = ""
            return

        llm_title = getattr(turn, "llm_title", None) or ""
        temp_title = getattr(turn, "temp_title", None) or ""
        llm_description = getattr(turn, "llm_description", None) or ""
        git_commit_hash = getattr(turn, "git_commit_hash", None) or ""

        lines: list[str] = []
        lines.append(f"Turn #{getattr(turn, 'turn_number', '-')}")
        lines.append(f"Time: {_format_dt(getattr(turn, 'timestamp', None))}")
        if git_commit_hash:
            lines.append(f"Git commit: {git_commit_hash}")

        lines.append("")
        if llm_title:
            lines.append("Title:")
            lines.append(llm_title)
            lines.append("")
        if temp_title and temp_title != llm_title:
            lines.append("Temp title:")
            lines.append(temp_title)
            lines.append("")
        if llm_description:
            lines.append("Description:")
            lines.append(llm_description)
            lines.append("")

        details.text = "\n".join(lines)
