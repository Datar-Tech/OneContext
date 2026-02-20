"""Events Table Widget with keyboard pagination."""

import contextlib
import io
import json
import traceback
from datetime import datetime
from typing import Optional, Set
from urllib.parse import urlparse

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.worker import Worker, WorkerState
from textual.widgets import Button, DataTable, Static

from ...logging_config import setup_logger
from ..clipboard import copy_text
from .openable_table import OpenableDataTable

logger = setup_logger("realign.dashboard.events", "dashboard.log")


class EventsListTable(OpenableDataTable):
    """Events list table with multi-select behavior."""

    BINDINGS = [
        Binding("space", "toggle_mark", "Toggle", show=False),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.owner: Optional["EventsTable"] = None

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


class EventsTable(Container):
    """Table displaying events with keyboard pagination support."""

    DEFAULT_CSS = """
    EventsTable {
        height: 100%;
        padding: 1;
        overflow: hidden;
    }

    EventsTable .action-section {
        height: auto;
        margin-bottom: 1;
    }

    EventsTable .action-section Button {
        width: auto;
        margin-bottom: 1;
    }

    EventsTable .section-header {
        height: auto;
        margin-bottom: 1;
    }

    EventsTable .table-container {
        height: 1fr;
        overflow-x: auto;
        overflow-y: auto;
    }

    EventsTable DataTable {
        height: auto;
        max-height: 100%;
        overflow-x: auto;
        overflow-y: auto;
    }

    EventsTable .stats-info {
        height: 1;
        color: $text-muted;
        text-align: center;
    }
    """

    # Reactive properties
    wrap_mode: reactive[bool] = reactive(False)
    # Maximum events to load (for scrollable list)
    MAX_EVENTS: int = 500

    def __init__(self) -> None:
        super().__init__()
        self._events: list = []
        self._events_by_id: dict = {}  # Index events by id for quick lookup
        self._total_events: int = 0
        self._selected_event_ids: Set[str] = set()
        self._selection_anchor_row: Optional[int] = None
        self._last_wrap_mode: bool = bool(self.wrap_mode)
        self._refresh_worker: Optional[Worker] = None
        self._share_export_worker: Optional[Worker] = None
        self._refresh_timer = None
        self._is_refreshing: bool = False
        self._pending_refresh: bool = False
        self._saved_cursor_event_id: Optional[str] = None

    def compose(self) -> ComposeResult:
        """Compose the events table layout."""
        logger.debug("EventsTable.compose() started")
        try:
            with Vertical(classes="action-section"):
                yield Button(
                    "Load selected context to current agent",
                    id="load-context-btn",
                    variant="primary",
                    disabled=True,
                )
                yield Button(
                    "Share selected events to others",
                    id="share-event-btn",
                    variant="primary",
                    disabled=True,
                )
                yield Button(
                    "Import context from others",
                    id="share-import-btn",
                    variant="primary",
                )
            yield Static(id="section-header", classes="section-header")
            with Container(classes="table-container"):
                yield EventsListTable(id="events-table")
            yield Static(id="stats-info", classes="stats-info")
            logger.debug("EventsTable.compose() completed")
        except Exception as e:
            logger.error(f"EventsTable.compose() failed: {e}\n{traceback.format_exc()}")
            raise

    def on_mount(self) -> None:
        """Set up the table on mount."""
        logger.debug("EventsTable.on_mount() started")
        try:
            table = self.query_one("#events-table", EventsListTable)
            table.owner = self
            self._setup_table_columns(table)
            logger.debug("EventsTable.on_mount() completed")
        except Exception as e:
            logger.error(f"EventsTable.on_mount() failed: {e}\n{traceback.format_exc()}")
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
        self._load_events()
        try:
            self.query_one("#events-table", DataTable).focus()
        except Exception:
            pass

    def on_openable_data_table_row_activated(
        self, event: OpenableDataTable.RowActivated
    ) -> None:
        if event.data_table.id != "events-table":
            return

        event_id = str(event.row_key.value)
        if not event_id:
            return

        from ..screens import EventDetailScreen

        self.app.push_screen(EventDetailScreen(event_id))

    def action_switch_view(self) -> None:
        """Toggle between compact vs wrapped cell display."""
        self.wrap_mode = not self.wrap_mode
        self._update_display()

    def _setup_table_columns(self, table: DataTable) -> None:
        table.clear(columns=True)
        table.add_column("✓", key="sel", width=2)
        table.add_column("#", key="index", width=4)
        table.add_column("Title", key="title")  # Auto width for full title
        table.add_column("Share", key="share", width=12)
        table.add_column("Type", key="type", width=8)
        table.add_column("Sessions", key="sessions", width=8)
        table.add_column("Created By", key="created_by", width=10)
        table.add_column("Shared By", key="shared_by", width=10)
        table.add_column("Event ID", key="event_id", width=12)
        table.add_column("Created", key="created", width=10)
        table.cursor_type = "row"

    def get_selected_event_ids(self) -> list[str]:
        return sorted(self._selected_event_ids)

    def clear_selection(self) -> None:
        self._selected_event_ids.clear()
        self._selection_anchor_row = None
        self._refresh_checkboxes_only()

    def toggle_selection_at_cursor(self) -> None:
        table = self.query_one("#events-table", DataTable)
        if table.row_count == 0:
            return
        try:
            event_id = str(
                table.coordinate_to_cell_key(table.cursor_coordinate)[0].value
            )
        except Exception:
            return

        if not event_id:
            return

        if event_id in self._selected_event_ids:
            self._selected_event_ids.remove(event_id)
        else:
            self._selected_event_ids.add(event_id)

        self._selection_anchor_row = table.cursor_coordinate.row
        self._refresh_checkboxes_only()

    def apply_mouse_selection(self, row_index: int, *, shift: bool, meta: bool) -> None:
        table = self.query_one("#events-table", DataTable)
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
                    eid = str(table.coordinate_to_cell_key((r, 0))[0].value)
                except Exception:
                    continue
                if eid:
                    ids_in_range.append(eid)

            if meta:
                self._selected_event_ids.update(ids_in_range)
            else:
                self._selected_event_ids = set(ids_in_range)
        else:
            # Toggle selection on click (no modifier keys needed)
            if clicked_id in self._selected_event_ids:
                self._selected_event_ids.remove(clicked_id)
            else:
                self._selected_event_ids.add(clicked_id)

        self._selection_anchor_row = row_index
        self._refresh_checkboxes_only()

    def _checkbox_cell(self, event_id: str) -> str:
        return (
            "[bold green]●[/bold green]"
            if event_id in self._selected_event_ids
            else "○"
        )

    def _format_cell(self, value: str, event_id: str) -> str:
        """Format cell value with bold if selected."""
        if event_id in self._selected_event_ids:
            return f"[bold]{value}[/bold]"
        return value

    def _refresh_checkboxes_only(self) -> None:
        table = self.query_one("#events-table", DataTable)
        if table.row_count == 0:
            self._update_summary_widget()
            return

        for row in range(table.row_count):
            try:
                eid = str(table.coordinate_to_cell_key((row, 0))[0].value)
            except Exception:
                continue
            if not eid:
                continue
            event = self._events_by_id.get(eid)
            if not event:
                continue
            try:
                # Update all cells in the row with proper formatting
                share_id = event.get("share_id") or "-"
                table.update_cell(eid, "sel", self._checkbox_cell(eid))
                table.update_cell(
                    eid, "index", self._format_cell(str(event["index"]), eid)
                )
                table.update_cell(eid, "title", self._format_cell(event["title"], eid))
                table.update_cell(eid, "share", self._format_cell(share_id, eid))
                table.update_cell(eid, "type", self._format_cell(event["type"], eid))
                table.update_cell(
                    eid, "sessions", self._format_cell(str(event["sessions"]), eid)
                )
                table.update_cell(
                    eid, "created_by", self._format_cell(event.get("created_by", "-"), eid)
                )
                table.update_cell(
                    eid, "shared_by", self._format_cell(event.get("shared_by", "-"), eid)
                )
                table.update_cell(
                    eid, "event_id", self._format_cell(event["short_id"], eid)
                )
                table.update_cell(
                    eid, "created", self._format_cell(event["created"], eid)
                )
            except Exception:
                continue

        self._update_summary_widget()

    def _update_summary_widget(self) -> None:
        # Update stats at the bottom
        stats_widget = self.query_one("#stats-info", Static)
        selected_count = len(self._selected_event_ids)

        stats_parts = [f"Total: {self._total_events}"]
        if selected_count:
            stats_parts.append(f"Selected: {selected_count}")

        stats_widget.update(f"[dim]{' | '.join(stats_parts)}[/dim]")

        # Enable/disable buttons based on selection
        load_btn = self.query_one("#load-context-btn", Button)
        load_btn.disabled = selected_count == 0
        share_btn = self.query_one("#share-event-btn", Button)
        share_btn.disabled = selected_count == 0

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "share-import-btn":
            self.action_share_import()
            return
        if button_id == "load-context-btn":
            action = getattr(self.app, "action_load_context", None)
            if callable(action):
                await action()
            return
        if button_id == "share-event-btn":
            self._start_share_export()
            return

    def action_share_import(self) -> None:
        from ..screens import ShareImportScreen

        self.app.push_screen(ShareImportScreen())

    def _on_refresh_timer(self) -> None:
        self.refresh_data(force=False)

    def refresh_data(self, *, force: bool = True) -> None:
        """Refresh events data without blocking the UI."""
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
            return self._collect_all_events()

        self._is_refreshing = True
        self._pending_refresh = False
        self._refresh_worker = self.run_worker(work, thread=True, exit_on_error=False)

    def _save_cursor_position(self) -> None:
        """Save the current cursor event ID to restore after refresh."""
        try:
            table = self.query_one("#events-table", DataTable)
            if table.row_count > 0:
                self._saved_cursor_event_id = str(
                    table.coordinate_to_cell_key(table.cursor_coordinate)[0].value
                )
            else:
                self._saved_cursor_event_id = None
        except Exception:
            self._saved_cursor_event_id = None

    def _load_events(self) -> None:
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
                "total_events": 0,
                "events": [],
            }
            self._is_refreshing = False
        elif event.state != WorkerState.SUCCESS:
            return
        else:
            result = self._refresh_worker.result or {}
            self._is_refreshing = False

        try:
            self._total_events = int(result.get("total_events") or 0)
        except Exception:
            self._total_events = 0
        try:
            self._events = list(result.get("events") or [])
            self._events_by_id = {e["id"]: e for e in self._events}
        except Exception:
            self._events = []
            self._events_by_id = {}
        self._update_display()

        if self._pending_refresh:
            self.refresh_data()

    def _start_share_export(self) -> None:
        """Export selected events as share links."""
        selected_ids = list(self._selected_event_ids)
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
            # Export the first selected event (for now, support single event)
            event_id = selected_ids[0]

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

        exit_code = int(result.get("exit_code", 1))

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
                db_event = db.get_event_by_id(event_id)
                if db_event:
                    if not share_link:
                        share_link = getattr(db_event, "share_url", None)
                    if not slack_message:
                        slack_message = getattr(db_event, "slack_message", None)
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

        # Refresh to show updated share info
        self.refresh_data()

    def _extract_share_id(self, share_url: Optional[str]) -> str:
        if not share_url:
            return ""
        try:
            url = str(share_url).rstrip("/")
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split("/") if p]
            if "share" in path_parts:
                share_idx = path_parts.index("share")
                if share_idx + 1 < len(path_parts):
                    return path_parts[share_idx + 1]
        except Exception:
            return ""
        return ""

    def _collect_all_events(self) -> dict:
        """Collect all events (background thread)."""
        total_events: int = 0
        events: list[dict] = []

        try:
            from ...db import get_database

            db = get_database()
            conn = db._get_connection()

            # Get total count
            row = conn.execute("SELECT COUNT(*) FROM events").fetchone()
            total_events = int(row[0]) if row else 0

            # Get all events (up to MAX_EVENTS)
            try:
                rows = conn.execute(
                    """
                    SELECT
                        e.id,
                        e.title,
                        e.event_type,
                        e.created_at,
                        e.share_url,
                        e.created_by,
                        e.shared_by,
                        (SELECT COUNT(*) FROM event_sessions WHERE event_id = e.id) AS session_count
                    FROM events e
                    ORDER BY e.created_at DESC
                    LIMIT ?
                    """,
                    (self.MAX_EVENTS,),
                ).fetchall()
                has_new_columns = True
            except Exception:
                # Fallback for older schema without created_by/shared_by
                rows = conn.execute(
                    """
                    SELECT
                        e.id,
                        e.title,
                        e.event_type,
                        e.created_at,
                        (SELECT COUNT(*) FROM event_sessions WHERE event_id = e.id) AS session_count
                    FROM events e
                    ORDER BY e.created_at DESC
                    LIMIT ?
                    """,
                    (self.MAX_EVENTS,),
                ).fetchall()
                has_new_columns = False

            for i, row in enumerate(rows):
                event_id = row[0]
                title = row[1] or "(no title)"
                event_type = row[2] or "unknown"
                created_at = row[3]
                if has_new_columns:
                    share_url = row[4]
                    created_by = row[5]
                    shared_by = row[6]
                    session_count = row[7]
                else:
                    share_url = None
                    created_by = None
                    shared_by = None
                    session_count = row[4]

                # Format event type
                type_map = {
                    "task": "User",
                    "preset_day": "Daily",
                    "preset_week": "Weekly",
                }
                event_type_display = type_map.get(event_type, event_type)

                created_str = "-"
                if created_at:
                    try:
                        dt = datetime.fromisoformat(created_at)
                        created_str = self._format_relative_time(dt)
                    except Exception:
                        created_str = created_at

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

                events.append(
                    {
                        "index": i + 1,
                        "id": event_id,
                        "short_id": self._shorten_id(event_id),
                        "title": title,
                        "type": event_type_display,
                        "sessions": session_count,
                        "share_url": share_url,
                        "share_id": self._extract_share_id(share_url),
                        "created_by": created_by_display,
                        "shared_by": shared_by_display,
                        "created": created_str,
                    }
                )
        except Exception:
            total_events = 0
            events = []

        return {
            "total_events": total_events,
            "events": events,
        }

    def _update_display(self) -> None:
        """Update the display with current data."""
        # Update summary
        self._update_summary_widget()

        # Update section header
        header_widget = self.query_one("#section-header", Static)
        header_widget.update("[bold]Events[/bold]")

        # Update table
        table = self.query_one("#events-table", DataTable)

        # Use saved cursor position if available, otherwise try to get current
        restore_event_id = self._saved_cursor_event_id
        if restore_event_id is None:
            try:
                if table.row_count > 0:
                    restore_event_id = str(
                        table.coordinate_to_cell_key(table.cursor_coordinate)[0].value
                    )
            except Exception:
                restore_event_id = None

        table.clear()

        # Always enable scrollbars
        table.styles.overflow_x = "auto"
        table.styles.overflow_y = "auto"
        table.show_horizontal_scrollbar = True

        for event in self._events:
            eid = event["id"]
            share_id = event.get("share_id") or "-"
            title = event["title"]

            if not self.wrap_mode and len(title) > 60:
                title = title[:57] + "..."

            share_val = share_id
            if self.wrap_mode and event.get("share_url"):
                share_val = event.get("share_url")

            # Column order: ✓, #, Title, Share, Type, Sessions, Created By, Shared By, Event ID, Created
            table.add_row(
                self._checkbox_cell(eid),
                self._format_cell(str(event["index"]), eid),
                self._format_cell(title, eid),
                self._format_cell(share_val, eid),
                self._format_cell(event["type"], eid),
                self._format_cell(str(event["sessions"]), eid),
                self._format_cell(event.get("created_by", "-"), eid),
                self._format_cell(event.get("shared_by", "-"), eid),
                self._format_cell(event["short_id"], eid),
                self._format_cell(event["created"], eid),
                key=eid,
            )

        if table.row_count > 0:
            if restore_event_id:
                try:
                    table.cursor_coordinate = (
                        table.get_row_index(restore_event_id),
                        0,
                    )
                except Exception:
                    table.cursor_coordinate = (0, 0)
            else:
                table.cursor_coordinate = (0, 0)

        # Clear saved cursor position after restore
        self._saved_cursor_event_id = None

    def _shorten_id(self, event_id: str) -> str:
        """Shorten an event ID for display."""
        if len(event_id) > 12:
            return event_id[:8] + "..."
        return event_id

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
