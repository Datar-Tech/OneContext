"""Sessions Table Widget with keyboard pagination."""

import contextlib
import io
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.worker import Worker, WorkerState
from textual.widgets import Button, DataTable, Select, Static

from ...logging_config import setup_logger
from ..clipboard import copy_text
from .openable_table import OpenableDataTable

logger = setup_logger("realign.dashboard.sessions", "dashboard.log")


class SessionsListTable(OpenableDataTable):
    """Sessions list table with multi-select behavior."""

    BINDINGS = [
        Binding("space", "toggle_mark", "Toggle", show=False),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.owner: Optional["SessionsTable"] = None

    def action_toggle_mark(self) -> None:
        if self.owner is not None:
            self.owner.toggle_selection_at_cursor()

    async def _on_click(self, event: events.Click) -> None:
        style = event.style
        meta = style.meta if style else {}
        row_index = meta.get("row")
        is_data_row = isinstance(row_index, int) and row_index >= 0

        await super()._on_click(event)

        if self.owner is None or not is_data_row:
            return

        self.owner.apply_mouse_selection(row_index, shift=event.shift, meta=event.meta)


class SessionsTable(Container):
    """Table displaying sessions with keyboard pagination support."""

    DEFAULT_CSS = """
    SessionsTable {
        height: 100%;
        padding: 1;
        overflow: hidden;
    }

    SessionsTable .action-section {
        height: auto;
        margin-bottom: 1;
    }

    SessionsTable .action-section Button {
        width: auto;
        margin-bottom: 1;
    }

    SessionsTable .section-header {
        height: auto;
        margin-bottom: 1;
    }

    SessionsTable .table-container {
        height: 1fr;
        overflow-x: auto;
        overflow-y: auto;
    }

    SessionsTable DataTable {
        height: auto;
        max-height: 100%;
        overflow-x: auto;
        overflow-y: auto;
    }

    SessionsTable .stats-info {
        height: 1;
        color: $text-muted;
        text-align: center;
    }
    """

    # Reactive properties
    wrap_mode: reactive[bool] = reactive(False)
    # Maximum sessions to load (for scrollable list)
    MAX_SESSIONS: int = 500

    def __init__(self) -> None:
        super().__init__()
        self._sessions: list = []
        self._sessions_by_id: dict = {}  # Index sessions by id for quick lookup
        self._total_sessions: int = 0
        self._stats: dict = {}
        self._selected_session_ids: Set[str] = set()
        self._selection_anchor_row: Optional[int] = None
        self._last_wrap_mode: bool = bool(self.wrap_mode)
        self._refresh_worker: Optional[Worker] = None
        self._share_export_worker: Optional[Worker] = None
        self._refresh_timer = None
        self._is_refreshing: bool = False
        self._pending_refresh: bool = False
        self._saved_cursor_session_id: Optional[str] = None

    def compose(self) -> ComposeResult:
        """Compose the sessions table layout."""
        with Vertical(classes="action-section"):
            yield Button(
                "Load selected context to current agent",
                id="load-context-btn",
                variant="primary",
                disabled=True,
            )
            yield Button(
                "Share selected contexts to others",
                id="share-context-btn",
                variant="primary",
                disabled=True,
            )
        yield Static(id="section-header", classes="section-header")
        with Container(classes="table-container"):
            yield SessionsListTable(id="sessions-table")
        yield Static(id="stats-info", classes="stats-info")

    def on_mount(self) -> None:
        """Set up the table on mount."""
        logger.debug("SessionsTable.on_mount() started")
        try:
            table = self.query_one("#sessions-table", SessionsListTable)
            table.owner = self

            self._setup_table_columns(table)
            logger.debug("SessionsTable.on_mount() completed")
        except Exception as e:
            logger.error(f"SessionsTable.on_mount() failed: {e}\n{traceback.format_exc()}")
            raise

    def on_resize(self) -> None:
        """Handle window resize."""
        pass  # No longer need to recalculate pagination

    def on_show(self) -> None:
        """Refresh data when the tab becomes visible."""
        if self._refresh_timer is None:
            self._refresh_timer = self.set_interval(60.0, self._on_refresh_timer)
        else:
            try:
                self._refresh_timer.resume()
            except Exception:
                pass
        self.call_later(self._on_became_visible)

    def on_hide(self) -> None:
        """Pause background refresh when hidden."""
        if self._refresh_timer is None:
            return
        try:
            self._refresh_timer.pause()
        except Exception:
            pass

    def _on_became_visible(self) -> None:
        self._load_sessions()
        try:
            self.query_one("#sessions-table", DataTable).focus()
        except Exception:
            pass

    def on_openable_data_table_row_activated(
        self, event: OpenableDataTable.RowActivated
    ) -> None:
        if event.data_table.id != "sessions-table":
            return

        session_id = str(event.row_key.value)
        if not session_id:
            return

        from ..screens import SessionDetailScreen

        self.app.push_screen(SessionDetailScreen(session_id))

    def action_switch_view(self) -> None:
        """Toggle between compact vs expanded cell display."""
        self.wrap_mode = not self.wrap_mode
        self._update_display()

    def _setup_table_columns(self, table: DataTable) -> None:
        table.clear(columns=True)
        table.add_column("✓", key="sel", width=2)
        table.add_column("#", key="index", width=3)
        table.add_column("Title", key="title")  # Auto width for full title
        table.add_column("Project", key="project", width=15)
        table.add_column("Source", key="source", width=10)
        table.add_column("Turns", key="turns", width=6)
        table.add_column("Created By", key="created_by", width=10)
        table.add_column("Shared By", key="shared_by", width=10)
        table.add_column("Session ID", key="session_id", width=20)
        table.add_column("Last Activity", key="last_activity", width=12)
        table.cursor_type = "row"

    def get_selected_session_ids(self) -> list[str]:
        return sorted(self._selected_session_ids)

    def clear_selection(self) -> None:
        self._selected_session_ids.clear()
        self._selection_anchor_row = None
        self._refresh_checkboxes_only()

    def toggle_selection_at_cursor(self) -> None:
        table = self.query_one("#sessions-table", DataTable)
        if table.row_count == 0:
            return
        try:
            session_id = str(
                table.coordinate_to_cell_key(table.cursor_coordinate)[0].value
            )
        except Exception:
            return

        if not session_id:
            return

        if session_id in self._selected_session_ids:
            self._selected_session_ids.remove(session_id)
        else:
            self._selected_session_ids.add(session_id)

        self._selection_anchor_row = table.cursor_coordinate.row
        self._refresh_checkboxes_only()

    def apply_mouse_selection(self, row_index: int, *, shift: bool, meta: bool) -> None:
        table = self.query_one("#sessions-table", DataTable)
        if table.row_count == 0:
            return
        if row_index < 0 or row_index >= table.row_count:
            return

        try:
            clicked_id = str(table.coordinate_to_cell_key((row_index, 0))[0].value)
        except Exception:
            return

        if not clicked_id:
            return

        if shift:
            anchor = self._selection_anchor_row
            if anchor is None:
                anchor = row_index
            start = min(anchor, row_index)
            end = max(anchor, row_index)
            ids_in_range: list[str] = []
            for r in range(start, end + 1):
                try:
                    sid = str(table.coordinate_to_cell_key((r, 0))[0].value)
                except Exception:
                    continue
                if sid:
                    ids_in_range.append(sid)

            if meta:
                self._selected_session_ids.update(ids_in_range)
            else:
                self._selected_session_ids = set(ids_in_range)
        else:
            # Toggle selection on click (no modifier keys needed)
            if clicked_id in self._selected_session_ids:
                self._selected_session_ids.remove(clicked_id)
            else:
                self._selected_session_ids.add(clicked_id)

        self._selection_anchor_row = row_index
        self._refresh_checkboxes_only()

    def _checkbox_cell(self, session_id: str) -> str:
        return (
            "[bold green]●[/bold green]"
            if session_id in self._selected_session_ids
            else "○"
        )

    def _format_cell(self, value: str, session_id: str) -> str:
        """Format cell value with bold if selected."""
        if session_id in self._selected_session_ids:
            return f"[bold]{value}[/bold]"
        return value

    def _refresh_checkboxes_only(self) -> None:
        table = self.query_one("#sessions-table", DataTable)
        if table.row_count == 0:
            self._update_summary_widget()
            return

        for row in range(table.row_count):
            try:
                sid = str(table.coordinate_to_cell_key((row, 0))[0].value)
            except Exception:
                continue
            if not sid:
                continue
            session = self._sessions_by_id.get(sid)
            if not session:
                continue
            try:
                # Update all cells in the row with proper formatting
                table.update_cell(sid, "sel", self._checkbox_cell(sid))
                table.update_cell(
                    sid, "index", self._format_cell(str(session["index"]), sid)
                )
                table.update_cell(
                    sid, "title", self._format_cell(session["title"], sid)
                )
                table.update_cell(
                    sid, "project", self._format_cell(session["project"], sid)
                )
                table.update_cell(
                    sid, "source", self._format_cell(session["source"], sid)
                )
                table.update_cell(
                    sid, "turns", self._format_cell(str(session["turns"]), sid)
                )
                table.update_cell(
                    sid, "created_by", self._format_cell(session.get("created_by", "-"), sid)
                )
                table.update_cell(
                    sid, "shared_by", self._format_cell(session.get("shared_by", "-"), sid)
                )
                table.update_cell(
                    sid, "session_id", self._format_cell(session["short_id"], sid)
                )
                table.update_cell(
                    sid,
                    "last_activity",
                    self._format_cell(session["last_activity"], sid),
                )
            except Exception:
                continue

        self._update_summary_widget()

    def _update_summary_widget(self) -> None:
        # Update stats at the bottom
        stats_widget = self.query_one("#stats-info", Static)
        total = self._stats.get("total", 0)
        claude = self._stats.get("claude", 0)
        codex = self._stats.get("codex", 0)
        gemini = self._stats.get("gemini", 0)

        stats_parts = [f"Total: {total}"]
        if claude > 0:
            stats_parts.append(f"Claude: {claude}")
        if codex > 0:
            stats_parts.append(f"Codex: {codex}")
        if gemini > 0:
            stats_parts.append(f"Gemini: {gemini}")

        selected_count = len(self._selected_session_ids)
        if selected_count:
            stats_parts.append(f"Selected: {selected_count}")

        stats_widget.update(f"[dim]{' | '.join(stats_parts)}[/dim]")

        # Enable/disable buttons based on selection
        load_btn = self.query_one("#load-context-btn", Button)
        load_btn.disabled = selected_count == 0
        share_btn = self.query_one("#share-context-btn", Button)
        share_btn.disabled = selected_count == 0

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""

        if button_id == "load-context-btn":
            action = getattr(self.app, "action_load_context", None)
            if callable(action):
                await action()
            return

        if button_id == "share-context-btn":
            self._start_share_export()
            return

    def _on_refresh_timer(self) -> None:
        self.refresh_data(force=False)

    def refresh_data(self, *, force: bool = True) -> None:
        """Refresh sessions data without blocking the UI."""
        if not self.display:
            return

        if self._refresh_worker is not None and self._refresh_worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        ):
            if force:
                self._pending_refresh = True
            return

        # Save current cursor position before refresh
        self._save_cursor_position()

        def work() -> dict:
            return self._collect_all_sessions()

        self._is_refreshing = True
        self._pending_refresh = False
        self._refresh_worker = self.run_worker(work, thread=True, exit_on_error=False)

    def _save_cursor_position(self) -> None:
        """Save the current cursor session ID to restore after refresh."""
        try:
            table = self.query_one("#sessions-table", DataTable)
            if table.row_count > 0:
                self._saved_cursor_session_id = str(
                    table.coordinate_to_cell_key(table.cursor_coordinate)[0].value
                )
            else:
                self._saved_cursor_session_id = None
        except Exception:
            self._saved_cursor_session_id = None

    def _load_sessions(self) -> None:
        """Compatibility hook (tests stub this); default triggers async refresh."""
        if not self.is_mounted:
            return
        self.refresh_data()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        # Handle share export worker
        if (
            self._share_export_worker is not None
            and event.worker is self._share_export_worker
        ):
            self._on_share_export_worker_changed(event)
            return

        # Handle refresh worker
        if self._refresh_worker is None or event.worker is not self._refresh_worker:
            return

        if event.state == WorkerState.ERROR:
            result = {
                "total_sessions": 0,
                "stats": {},
                "sessions": [],
            }
            self._is_refreshing = False
        elif event.state != WorkerState.SUCCESS:
            return
        else:
            result = self._refresh_worker.result or {}
            self._is_refreshing = False

        try:
            self._total_sessions = int(result.get("total_sessions") or 0)
        except Exception:
            self._total_sessions = 0
        try:
            self._stats = dict(result.get("stats") or {})
        except Exception:
            self._stats = {}
        try:
            self._sessions = list(result.get("sessions") or [])
            self._sessions_by_id = {s["id"]: s for s in self._sessions}
        except Exception:
            self._sessions = []
            self._sessions_by_id = {}
        self._update_display()

        if self._pending_refresh:
            self.refresh_data()

    def _start_share_export(self) -> None:
        """Generate an event from selected sessions and export as share link."""
        selected_ids = list(self._selected_session_ids)
        if not selected_ids:
            return

        if (
            self._share_export_worker is not None
            and self._share_export_worker.state
            in (
                WorkerState.PENDING,
                WorkerState.RUNNING,
            )
        ):
            return

        def work() -> dict:
            # Step 1: Generate event from selected sessions
            session_selector = ",".join(selected_ids)
            stdout_gen = io.StringIO()
            stderr_gen = io.StringIO()

            with (
                contextlib.redirect_stdout(stdout_gen),
                contextlib.redirect_stderr(stderr_gen),
            ):
                from ...commands.watcher import watcher_event_generate_command

                gen_exit_code = watcher_event_generate_command(
                    session_selector=session_selector,
                    show_sessions=False,
                )

            if gen_exit_code != 0:
                return {
                    "step": "generate",
                    "exit_code": gen_exit_code,
                    "stderr": stderr_gen.getvalue().strip(),
                }

            # Extract event_id from output
            gen_output = stdout_gen.getvalue().strip()
            event_id = None
            for line in gen_output.split("\n"):
                if line.startswith("Event ID:"):
                    event_id = line.split(":", 1)[1].strip()
                    break

            if not event_id:
                # Try to find the most recent event
                try:
                    from ...db import get_database

                    db = get_database()
                    conn = db._get_connection()
                    row = conn.execute(
                        "SELECT id FROM events ORDER BY created_at DESC LIMIT 1"
                    ).fetchone()
                    if row:
                        event_id = row[0]
                except Exception:
                    pass

            if not event_id:
                return {
                    "step": "generate",
                    "exit_code": 1,
                    "stderr": "Could not determine event ID",
                }

            # Step 2: Export the event as a share link
            stdout_exp = io.StringIO()
            stderr_exp = io.StringIO()

            with (
                contextlib.redirect_stdout(stdout_exp),
                contextlib.redirect_stderr(stderr_exp),
            ):
                from ...commands import export_shares

                exp_exit_code = export_shares.export_shares_interactive_command(
                    indices=event_id,
                    password=None,
                    enable_preview=False,
                    json_output=True,
                    compact=True,
                )

            exp_output = stdout_exp.getvalue().strip()
            result: dict = {
                "step": "export",
                "exit_code": exp_exit_code,
                "output": exp_output,
                "stderr": stderr_exp.getvalue().strip(),
                "event_id": event_id,
            }

            if exp_output:
                # Try to extract JSON from output (may contain other text)
                try:
                    result["json"] = json.loads(exp_output)
                except Exception:
                    # Try to find JSON object in output
                    json_start = exp_output.find("{")
                    json_end = exp_output.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        try:
                            result["json"] = json.loads(exp_output[json_start:json_end])
                        except Exception:
                            result["json"] = None
                    else:
                        result["json"] = None
            else:
                result["json"] = None

            return result

        self.app.notify("Creating share link...", title="Share", timeout=2)
        self._share_export_worker = self.run_worker(
            work, thread=True, exit_on_error=False
        )

    def _on_share_export_worker_changed(self, event: Worker.StateChanged) -> None:
        """Handle share export worker state changes."""
        if event.state == WorkerState.ERROR:
            err = (
                self._share_export_worker.error
                if self._share_export_worker
                else "Unknown error"
            )
            self.app.notify(
                f"Share export failed: {err}", title="Share", severity="error"
            )
            return

        if event.state != WorkerState.SUCCESS:
            return

        result = self._share_export_worker.result if self._share_export_worker else {}
        if not result:
            result = {}

        step = result.get("step", "")
        exit_code = int(result.get("exit_code", 1))

        if step == "generate" and exit_code != 0:
            stderr = result.get("stderr", "")
            self.app.notify(
                f"Failed to generate event: {stderr}"
                if stderr
                else "Failed to generate event",
                title="Share",
                severity="error",
            )
            return

        if exit_code != 0:
            stderr = result.get("stderr", "")
            self.app.notify(
                f"Share export failed: {stderr}" if stderr else "Share export failed",
                title="Share",
                severity="error",
            )
            return

        payload = result.get("json") or {}
        share_link = payload.get("share_link") or payload.get("share_url")
        slack_message = (
            payload.get("slack_message") if isinstance(payload, dict) else None
        )
        event_id = result.get("event_id")

        # Try to fetch share_link and slack_message from database
        if event_id:
            try:
                from ...db import get_database

                db = get_database()
                event = db.get_event_by_id(event_id)
                if event:
                    if not share_link:
                        share_link = getattr(event, "share_url", None)
                    if not slack_message:
                        slack_message = getattr(event, "slack_message", None)
            except Exception:
                pass

        if not share_link:
            self.app.notify(
                "Share export completed but no link generated",
                title="Share",
                severity="warning",
            )
            return

        # Build copy text
        if slack_message:
            text_to_copy = str(slack_message) + "\n\n" + str(share_link)
        else:
            text_to_copy = str(share_link)

        # Copy to clipboard
        copied = copy_text(self.app, text_to_copy)

        suffix = " (copied to clipboard)" if copied else ""
        self.app.notify(f"Share link created{suffix}", title="Share", timeout=4)

    def _collect_all_sessions(self) -> dict:
        """Collect all sessions + stats (background thread)."""
        total_sessions: int = 0
        stats: dict = {}
        sessions: list[dict] = []

        try:
            from ...db import get_database
            from ...db.sqlite_db import SQLiteDatabase

            db = get_database()

            if isinstance(db, SQLiteDatabase):
                conn = db._get_connection()

                # Get total count
                row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
                total_sessions = int(row[0]) if row else 0

                # Get stats
                stats_row = conn.execute(
                    """
                    SELECT
                        COUNT(*) AS total,
                        COUNT(CASE WHEN session_type = 'claude' THEN 1 END) AS claude,
                        COUNT(CASE WHEN session_type = 'codex' THEN 1 END) AS codex,
                        COUNT(CASE WHEN session_type = 'gemini' THEN 1 END) AS gemini
                    FROM sessions
                """
                ).fetchone()

                stats = {
                    "total": stats_row[0] if stats_row else 0,
                    "claude": stats_row[1] if stats_row else 0,
                    "codex": stats_row[2] if stats_row else 0,
                    "gemini": stats_row[3] if stats_row else 0,
                }

                # Get all sessions (up to MAX_SESSIONS)
                try:
                    rows = conn.execute(
                        """
                        SELECT
                            s.id,
                            s.session_type,
                            s.workspace_path,
                            s.session_title,
                            s.last_activity_at,
                            s.total_turns,
                            s.created_by,
                            s.shared_by
                        FROM sessions s
                        ORDER BY s.last_activity_at DESC
                        LIMIT ?
                    """,
                        (self.MAX_SESSIONS,),
                    ).fetchall()
                    has_new_columns = True
                except Exception:
                    # Fallback for older schema without created_by/shared_by
                    rows = conn.execute(
                        """
                        SELECT
                            s.id,
                            s.session_type,
                            s.workspace_path,
                            s.session_title,
                            s.last_activity_at,
                            s.total_turns
                        FROM sessions s
                        ORDER BY s.last_activity_at DESC
                        LIMIT ?
                    """,
                        (self.MAX_SESSIONS,),
                    ).fetchall()
                    has_new_columns = False

                for i, row in enumerate(rows):
                    session_id = row[0]
                    session_type = row[1] or "unknown"
                    workspace = row[2]
                    title = row[3] or "(no title)"
                    last_activity = row[4]
                    turn_count = row[5]
                    if has_new_columns:
                        created_by = row[6]
                        shared_by = row[7]
                    else:
                        created_by = None
                        shared_by = None

                    source_map = {
                        "claude": "Claude",
                        "codex": "Codex",
                        "gemini": "Gemini",
                        "shell": "Shell",
                        "zsh": "Shell",
                    }
                    source = source_map.get(session_type, session_type)
                    project = Path(workspace).name if workspace else "-"

                    activity_str = "-"
                    if last_activity:
                        try:
                            dt = datetime.fromisoformat(last_activity)
                            activity_str = self._format_relative_time(dt)
                        except Exception:
                            activity_str = last_activity

                    # Look up user names from users table
                    created_by_display = "-"
                    shared_by_display = "-"
                    if created_by:
                        try:
                            user_row = conn.execute(
                                "SELECT user_name FROM users WHERE uid = ?", (created_by,)
                            ).fetchone()
                            created_by_display = user_row[0] if user_row and user_row[0] else created_by[:8] + "..."
                        except Exception:
                            created_by_display = created_by[:8] + "..." if len(created_by) > 8 else created_by
                    if shared_by:
                        try:
                            user_row = conn.execute(
                                "SELECT user_name FROM users WHERE uid = ?", (shared_by,)
                            ).fetchone()
                            shared_by_display = user_row[0] if user_row and user_row[0] else shared_by[:8] + "..."
                        except Exception:
                            shared_by_display = shared_by[:8] + "..." if len(shared_by) > 8 else shared_by

                    sessions.append(
                        {
                            "index": i + 1,
                            "id": session_id,
                            "short_id": self._shorten_session_id(session_id),
                            "source": source,
                            "project": project,
                            "turns": turn_count,
                            "title": title,
                            "created_by": created_by_display,
                            "shared_by": shared_by_display,
                            "last_activity": activity_str,
                        }
                    )
        except Exception as e:
            logger.error(f"_collect_all_sessions failed: {e}\n{traceback.format_exc()}")
            total_sessions = 0
            stats = {}
            sessions = []

        return {
            "total_sessions": total_sessions,
            "stats": stats,
            "sessions": sessions,
        }

    def _update_display(self) -> None:
        """Update the display with current data."""
        self._update_summary_widget()

        # Update section header
        header_widget = self.query_one("#section-header", Static)
        header_widget.update("[bold]Sessions[/bold]")

        # Update table
        table = self.query_one("#sessions-table", DataTable)

        # Use saved cursor position if available, otherwise try to get current
        restore_session_id = self._saved_cursor_session_id
        if restore_session_id is None:
            try:
                if table.row_count > 0:
                    restore_session_id = str(
                        table.coordinate_to_cell_key(table.cursor_coordinate)[0].value
                    )
            except Exception:
                restore_session_id = None

        table.clear()

        # Always enable scrollbars
        table.styles.overflow_x = "auto"
        table.styles.overflow_y = "auto"
        table.show_horizontal_scrollbar = True

        for session in self._sessions:
            sid = session["id"]
            title = session["title"]
            if not self.wrap_mode and len(title) > 60:
                title = title[:57] + "..."

            display_id = session["short_id"]
            if self.wrap_mode:
                display_id = sid

            # Column order: ✓, #, Title, Project, Source, Turns, Created By, Shared By, Session ID, Last Activity
            table.add_row(
                self._checkbox_cell(sid),
                self._format_cell(str(session["index"]), sid),
                self._format_cell(title, sid),
                self._format_cell(session["project"], sid),
                self._format_cell(session["source"], sid),
                self._format_cell(str(session["turns"]), sid),
                self._format_cell(session.get("created_by", "-"), sid),
                self._format_cell(session.get("shared_by", "-"), sid),
                self._format_cell(display_id, sid),
                self._format_cell(session["last_activity"], sid),
                key=sid,
            )

        if table.row_count > 0:
            if restore_session_id:
                try:
                    table.cursor_coordinate = (
                        table.get_row_index(restore_session_id),
                        0,
                    )
                except Exception:
                    table.cursor_coordinate = (0, 0)
            else:
                table.cursor_coordinate = (0, 0)

        # Clear saved cursor position after restore
        self._saved_cursor_session_id = None

    def _shorten_session_id(self, session_id: str) -> str:
        """Shorten a session ID for display."""
        if len(session_id) > 20:
            return session_id[:8] + "..." + session_id[-8:]
        return session_id

    def _format_relative_time(self, dt: datetime) -> str:
        """Format a datetime as relative time."""
        now = datetime.now()
        diff = now - dt
        seconds = diff.total_seconds()

        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"{mins}m ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours}h ago"
        else:
            days = int(seconds / 86400)
            return f"{days}d ago"
