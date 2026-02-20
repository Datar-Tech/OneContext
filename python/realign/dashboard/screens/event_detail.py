"""Event detail modal for the dashboard."""

from __future__ import annotations

import contextlib
from datetime import datetime
import io
import json
from typing import Optional

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.worker import Worker, WorkerState
from textual.widgets import DataTable, Static

from ..widgets.copyable_text_area import CopyableTextArea
from ..widgets.openable_table import OpenableDataTable
from ..clipboard import copy_text


def _format_dt(dt: object) -> str:
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    if dt is None:
        return "-"
    return str(dt)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n… (truncated)"


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _truncate_single_line(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _format_relative_time(dt: datetime) -> str:
    now = datetime.now()
    diff = now - dt
    seconds = diff.total_seconds()

    if seconds < 60:
        return "just now"
    if seconds < 3600:
        mins = int(seconds / 60)
        return f"{mins}m ago"
    if seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours}h ago"
    days = int(seconds / 86400)
    return f"{days}d ago"


def _format_iso_relative(iso_str: Optional[str]) -> str:
    if not iso_str:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_str)
        return _format_relative_time(dt)
    except Exception:
        return iso_str


class EventDetailScreen(ModalScreen):
    """Modal that shows an event and the sessions linked to it."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
        Binding("x", "share_export", "Share Export", show=False),
    ]

    DEFAULT_CSS = """
    EventDetailScreen {
        align: center middle;
    }

    EventDetailScreen #event-detail-root {
        width: 95%;
        height: 95%;
        padding: 1;
        background: $background;
        border: solid $accent;
    }

    EventDetailScreen #event-meta {
        height: auto;
        margin-bottom: 1;
    }

    EventDetailScreen #event-detail-body {
        height: 1fr;
    }

    EventDetailScreen #event-sessions-table {
        width: 1fr;
        height: 1fr;
    }

    EventDetailScreen #event-details {
        width: 1fr;
        height: 1fr;
        margin-top: 1;
    }

    EventDetailScreen #event-hint {
        height: 1;
        margin-top: 1;
        color: $text-muted;
        text-align: right;
    }
    """

    def __init__(self, event_id: str) -> None:
        super().__init__()
        self.event_id = event_id
        self._load_error: Optional[str] = None
        self._sessions: list[dict] = []
        self._session_record_cache: dict[str, object] = {}
        self._initialized: bool = False
        self._share_export_worker: Optional[Worker[dict]] = None

    def compose(self) -> ComposeResult:
        with Container(id="event-detail-root"):
            yield Static(id="event-meta")
            with Vertical(id="event-detail-body"):
                sessions_table = OpenableDataTable(id="event-sessions-table")
                sessions_table.add_columns(
                    "#",
                    "Session ID",
                    "Source",
                    "Project",
                    "Turns",
                    "Title",
                    "Last Activity",
                )
                sessions_table.cursor_type = "row"
                yield sessions_table
                yield CopyableTextArea(
                    "",
                    id="event-details",
                    read_only=True,
                    show_line_numbers=False,
                    soft_wrap=True,
                )
            yield Static(
                "Click: select   Enter/dblclick: open   x: share export (no password)   Esc: close",
                id="event-hint",
            )

    def on_show(self) -> None:
        self.call_later(self._ensure_initialized)

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        sessions_table = self.query_one("#event-sessions-table", DataTable)
        self._load_data()
        self._update_display()

        if sessions_table.row_count > 0:
            sessions_table.focus()

    def action_close(self) -> None:
        self.app.pop_screen()

    def action_share_export(self) -> None:
        """Generate/reuse a share link for this event (runs in background)."""
        if not self.event_id:
            return

        if self._share_export_worker is not None and self._share_export_worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        ):
            return

        def work() -> dict:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                from ...commands import export_shares

                exit_code = export_shares.export_shares_interactive_command(
                    indices=self.event_id,
                    password=None,  # default: no password (unencrypted)
                    enable_preview=False,  # don't use Rich preview/editor in TUI
                    json_output=True,  # suppress prompts; defaults to no password
                    compact=True,
                )

            output = stdout.getvalue().strip()
            error_text = stderr.getvalue().strip()
            result: dict = {"exit_code": exit_code, "output": output, "stderr": error_text}

            if output:
                try:
                    result["json"] = json.loads(output)
                except Exception:
                    result["json"] = None
            else:
                result["json"] = None

            return result

        self.app.notify("Exporting share link...", title="Share", timeout=2)
        self._share_export_worker = self.run_worker(work, thread=True, exit_on_error=False)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if self._share_export_worker is None or event.worker is not self._share_export_worker:
            return

        if event.state == WorkerState.ERROR:
            err = self._share_export_worker.error
            self.app.notify(f"Share export failed: {err}", title="Share", severity="error")
            return

        if event.state != WorkerState.SUCCESS:
            return

        result = self._share_export_worker.result or {}
        raw_exit_code = result.get("exit_code", None)
        exit_code = 1 if raw_exit_code is None else int(raw_exit_code)
        payload = result.get("json") or {}
        share_link = payload.get("share_link") or payload.get("share_url")
        slack_message = payload.get("slack_message") if isinstance(payload, dict) else None

        # Refresh share metadata (share_url, slack_message, etc.) after export.
        try:
            self._load_data()
            self._update_display()
        except Exception:
            pass

        if exit_code == 0 and share_link:
            from ..clipboard import copy_text

            if not slack_message:
                slack_message = getattr(self._event, "slack_message", None) if self._event else None

            if slack_message:
                text_to_copy = str(slack_message) + "\n\n" + str(share_link)
            else:
                text_to_copy = str(share_link)

            copied = copy_text(self.app, text_to_copy)

            suffix = " (copied)" if copied else ""
            self.app.notify(f"Share link: {share_link}{suffix}", title="Share", timeout=6)
        elif exit_code == 0:
            self.app.notify("Share export completed", title="Share", timeout=3)
        else:
            extra = result.get("stderr") or ""
            suffix = f": {extra}" if extra else ""
            self.app.notify(
                f"Share export failed (exit {exit_code}){suffix}", title="Share", timeout=6
            )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "event-sessions-table":
            return
        self._update_session_preview(str(event.row_key.value))

    def on_openable_data_table_row_activated(self, event: OpenableDataTable.RowActivated) -> None:
        if event.data_table.id != "event-sessions-table":
            return

        session_id = str(event.row_key.value)
        if not session_id:
            return

        from .session_detail import SessionDetailScreen

        self.app.push_screen(SessionDetailScreen(session_id))

    def _get_session_record(self, session_id: str) -> Optional[object]:
        if session_id in self._session_record_cache:
            return self._session_record_cache[session_id]
        try:
            from ...db import get_database

            db = get_database()
            record = db.get_session_by_id(session_id)
            self._session_record_cache[session_id] = record
            return record
        except Exception:
            self._session_record_cache[session_id] = None
            return None

    def _load_data(self) -> None:
        try:
            from ...db import get_database
            from ...db.sqlite_db import SQLiteDatabase

            db = get_database()
            event = db.get_event_by_id(self.event_id)
            self._event = event

            self._sessions = []
            if isinstance(db, SQLiteDatabase):
                conn = db._get_connection()
                rows = conn.execute(
                    """
                    SELECT
                        s.id,
                        s.session_type,
                        s.workspace_path,
                        s.session_title,
                        s.session_summary,
                        s.last_activity_at,
                        s.total_turns,
                        (SELECT COUNT(*) FROM turns WHERE session_id = s.id) AS turn_count
                    FROM sessions s
                    JOIN event_sessions es ON s.id = es.session_id
                    WHERE es.event_id = ?
                    ORDER BY s.last_activity_at DESC
                    """,
                    (self.event_id,),
                ).fetchall()

                for row in rows:
                    session_id = row[0]
                    session_type = row[1] or "unknown"
                    workspace = row[2]
                    title = row[3] or "(no title)"
                    session_summary = row[4] or ""
                    last_activity = row[5]
                    total_turns = row[6]
                    turn_count = row[7]

                    source_map = {
                        "claude": "Claude",
                        "codex": "Codex",
                        "gemini": "Gemini",
                        "shell": "Shell",
                        "zsh": "Shell",
                    }
                    source = source_map.get(session_type, session_type)
                    project = workspace.split("/")[-1] if workspace else "-"

                    self._sessions.append(
                        {
                            "id": session_id,
                            "short_id": self._shorten_id(session_id),
                            "source": source,
                            "project": project,
                            "turns": int(turn_count or 0),
                            "total_turns": (
                                int(total_turns or 0) if total_turns is not None else None
                            ),
                            "title": title,
                            "session_summary": session_summary,
                            "last_activity": last_activity,
                        }
                    )
            else:
                for s in db.get_sessions_for_event(self.event_id):
                    session_id = str(s.id)
                    session_type = getattr(s, "session_type", None) or "unknown"
                    workspace = getattr(s, "workspace_path", None)
                    title = getattr(s, "session_title", None) or "(no title)"
                    session_summary = getattr(s, "session_summary", None) or ""
                    last_activity = getattr(s, "last_activity_at", None)
                    turns = int(getattr(s, "total_turns", 0) or 0)

                    source_map = {
                        "claude": "Claude",
                        "codex": "Codex",
                        "gemini": "Gemini",
                        "shell": "Shell",
                        "zsh": "Shell",
                    }
                    source = source_map.get(session_type, session_type)
                    project = str(workspace).split("/")[-1] if workspace else "-"

                    self._sessions.append(
                        {
                            "id": session_id,
                            "short_id": self._shorten_id(session_id),
                            "source": source,
                            "project": project,
                            "turns": turns,
                            "total_turns": turns,
                            "title": title,
                            "session_summary": session_summary,
                            "last_activity": _format_dt(last_activity),
                        }
                    )

            self._load_error = None
        except Exception as e:
            self._event = None
            self._sessions = []
            self._session_record_cache = {}
            self._load_error = str(e)

    def _update_display(self) -> None:
        meta = self.query_one("#event-meta", Static)
        details = self.query_one("#event-details", CopyableTextArea)

        if self._load_error:
            meta.update(f"[red]Failed to load event {self.event_id}:[/red] {self._load_error}")
            details.text = ""
            return

        title = getattr(self._event, "title", None) if self._event else None
        description = getattr(self._event, "description", None) if self._event else None
        event_type = getattr(self._event, "event_type", None) if self._event else None
        status = getattr(self._event, "status", None) if self._event else None
        created_at = getattr(self._event, "created_at", None) if self._event else None
        updated_at = getattr(self._event, "updated_at", None) if self._event else None
        share_url = getattr(self._event, "share_url", None) if self._event else None
        share_id = getattr(self._event, "share_id", None) if self._event else None
        share_expiry_at = getattr(self._event, "share_expiry_at", None) if self._event else None
        slack_message = getattr(self._event, "slack_message", None) if self._event else None

        description_display = _truncate(escape(description or "(no description)"), 2_000)

        meta_lines = [
            f"[bold]Event[/bold] {self.event_id}",
            f"[dim]Title:[/dim] {escape(title) if title else '(no title)'}",
            f"[dim]Description:[/dim] {description_display}",
            f"[dim]Type:[/dim] {event_type or '-'}    [dim]Status:[/dim] {status or '-'}",
            f"[dim]Created:[/dim] {_format_dt(created_at)}    [dim]Updated:[/dim] {_format_dt(updated_at)}",
            f"[dim]Sessions:[/dim] {len(self._sessions)}",
        ]

        if share_url:
            meta_lines.append(f"[dim]Share URL:[/dim] {escape(str(share_url))}")
        if share_id:
            meta_lines.append(f"[dim]Share ID:[/dim] {escape(str(share_id))}")
        if share_expiry_at:
            meta_lines.append(f"[dim]Share Expiry:[/dim] {_format_dt(share_expiry_at)}")
        if slack_message:
            slack_preview = _truncate_single_line(_collapse_whitespace(str(slack_message)), 400)
            meta_lines.append(f"[dim]Slack Message:[/dim] {escape(slack_preview)}")

        meta.update("\n".join(meta_lines))

        table = self.query_one("#event-sessions-table", DataTable)
        selected_session_id: Optional[str] = None
        try:
            if table.row_count > 0:
                selected_session_id = str(
                    table.coordinate_to_cell_key(table.cursor_coordinate)[0].value
                )
        except Exception:
            selected_session_id = None
        table.clear()

        for idx, s in enumerate(self._sessions, 1):
            title_cell = s["title"]
            if len(title_cell) > 40:
                title_cell = title_cell[:40] + "..."

            last_activity = s["last_activity"]
            if isinstance(last_activity, str):
                last_activity_str = _format_iso_relative(last_activity)
            else:
                last_activity_str = _format_dt(last_activity)

            table.add_row(
                str(idx),
                s["short_id"],
                s["source"],
                s["project"],
                str(s["turns"]),
                title_cell,
                last_activity_str,
                key=s["id"],
            )

        if table.row_count > 0:
            if selected_session_id:
                try:
                    table.cursor_coordinate = (table.get_row_index(selected_session_id), 0)
                except Exception:
                    table.cursor_coordinate = (0, 0)
            else:
                table.cursor_coordinate = (0, 0)

            row_key = table.coordinate_to_cell_key(table.cursor_coordinate)[0]
            self._update_session_preview(str(row_key.value))
        else:
            details.text = "No sessions linked to this event."

    def _update_session_preview(self, session_id: str) -> None:
        details = self.query_one("#event-details", CopyableTextArea)
        session = next((s for s in self._sessions if s["id"] == session_id), None)
        if not session:
            details.text = ""
            return

        record = self._get_session_record(session_id)

        record_title = getattr(record, "session_title", None) if record else None
        record_summary = getattr(record, "session_summary", None) if record else None
        record_type = getattr(record, "session_type", None) if record else None
        record_started = getattr(record, "started_at", None) if record else None
        record_last = getattr(record, "last_activity_at", None) if record else None

        source_map = {
            "claude": "Claude",
            "codex": "Codex",
            "gemini": "Gemini",
            "shell": "Shell",
            "zsh": "Shell",
        }
        source = source_map.get(
            record_type or "", record_type or session.get("source") or "unknown"
        )

        title = record_title or session.get("title") or "(no title)"
        summary = record_summary or session.get("session_summary") or ""
        turns = session.get("turns")
        total_turns = session.get("total_turns")

        lines: list[str] = []
        lines.append(f"Session: {session_id}")
        lines.append(f"Source: {source}")
        lines.append(f"Project: {session.get('project') or '-'}")

        if record_started is not None or record_last is not None:
            lines.append(f"Started: {_format_dt(record_started)}")
            lines.append(f"Last activity: {_format_dt(record_last)}")
        else:
            lines.append(f"Last activity: {_format_iso_relative(session.get('last_activity'))}")

        if turns is not None:
            if total_turns is not None and total_turns != turns:
                lines.append(f"Turns: {turns} (cached: {total_turns})")
            else:
                lines.append(f"Turns: {turns}")

        lines.append("")
        lines.append("Title:")
        lines.append(_truncate(str(title), 5_000))
        lines.append("")
        lines.append("Session summary:")
        lines.append(_truncate(str(summary) if summary else "(no summary)", 50_000))

        details.text = "\n".join(lines)

    def _shorten_id(self, id_str: str) -> str:
        if len(id_str) > 20:
            return id_str[:8] + "..." + id_str[-8:]
        return id_str
