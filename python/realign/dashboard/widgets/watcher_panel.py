"""Watcher Status Panel Widget with keyboard navigation."""

from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.reactive import reactive
from textual.worker import Worker, WorkerState
from textual.widgets import DataTable, Static


class WatcherPanel(Container, can_focus=True):
    """Panel displaying Watcher status with keyboard navigation."""

    BINDINGS = [
        Binding("s", "switch_view", "Switch View"),
        Binding("p", "prev_page", "Prev Page"),
        Binding("n", "next_page", "Next Page"),
    ]

    DEFAULT_CSS = """
    WatcherPanel {
        height: 100%;
        padding: 1;
        overflow: hidden;
    }

    WatcherPanel:focus {
        border: solid $accent;
    }

    WatcherPanel .status-section {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary-darken-2;
    }

    WatcherPanel .monitored-paths {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary-darken-2;
    }

    WatcherPanel .section-header {
        height: auto;
        margin-bottom: 1;
    }

    WatcherPanel .section-title {
        text-style: bold;
        color: $primary-lighten-2;
    }

    WatcherPanel .view-hint {
        color: $text-muted;
        text-style: italic;
    }

    WatcherPanel .table-container {
        height: auto;
        overflow: hidden;
    }

    WatcherPanel DataTable {
        height: auto;
        max-height: 100%;
        overflow: hidden;
    }

    WatcherPanel .pagination-info {
        height: 1;
        margin-top: 1;
        color: $text-muted;
        text-align: center;
    }
    """

    # Reactive properties
    current_view: reactive[str] = reactive("active")  # "active" or "recent"
    current_page: reactive[int] = reactive(1)
    rows_per_page: reactive[int] = reactive(10)

    def __init__(self) -> None:
        super().__init__()
        self._status_data: dict = {}
        self._monitored_paths: List[Tuple[str, str, bool]] = []
        self._all_active_sessions: list[dict] = []
        self._recent_sessions_page: list[dict] = []
        self._recent_total_sessions: int = 0
        self._refresh_worker: Worker | None = None
        self._refresh_timer = None
        self._active_refresh_snapshot: tuple[str, int, int] | None = None
        self._pending_refresh_snapshot: tuple[str, int, int] | None = None

    def compose(self) -> ComposeResult:
        """Compose the watcher panel layout."""
        yield Static(id="watcher-status", classes="status-section")
        yield Static(id="monitored-paths", classes="monitored-paths")
        yield Static(id="section-header", classes="section-header")
        with Container(classes="table-container"):
            yield DataTable(id="sessions-table")
        yield Static(id="pagination-info", classes="pagination-info")

    def on_mount(self) -> None:
        """Set up the panel on mount."""
        # Initialize table
        table = self.query_one("#sessions-table", DataTable)
        table.add_columns("#", "Session", "Source", "Project", "Last Activity")
        table.cursor_type = "row"

        # Calculate initial rows per page
        self._calculate_rows_per_page()

    def on_show(self) -> None:
        """Start auto-refresh when visible."""
        if self._refresh_timer is None:
            self._refresh_timer = self.set_interval(5.0, self._on_refresh_timer)
        else:
            try:
                self._refresh_timer.resume()
            except Exception:
                pass
        self.refresh_data(force=True)
        try:
            self.query_one("#sessions-table", DataTable).focus()
        except Exception:
            pass

    def on_hide(self) -> None:
        """Pause auto-refresh when hidden."""
        if self._refresh_timer is None:
            return
        try:
            self._refresh_timer.pause()
        except Exception:
            pass

    def on_resize(self) -> None:
        """Handle window resize to adjust rows per page."""
        old_rows = int(self.rows_per_page)
        self._calculate_rows_per_page()
        if int(self.rows_per_page) != old_rows:
            total_pages = self._get_total_pages()
            if self.current_page > total_pages:
                self.current_page = total_pages
            if self.current_view == "recent":
                self.refresh_data()
                return
        self._update_display()

    def _calculate_rows_per_page(self) -> None:
        """Calculate rows per page based on available height, ensuring no scrollbar."""
        try:
            panel_height = self.size.height

            # Calculate exact space needed:
            # - Status section: 1 line content + 2 border + 2 padding + 1 margin = 6
            # - Monitored paths: 5 lines (header + 4 paths) + 2 border + 2 padding + 1 margin = 10
            # - Section header: 1 line + 1 margin = 2
            # - Table header row: 1
            # - Pagination info: 1 line + 1 margin = 2
            # - Panel padding: 2 (top + bottom)
            # - Extra buffer: 3 (for borders, focus indicators, etc.)

            fixed_height = 6 + 10 + 2 + 1 + 2 + 2 + 3  # = 26

            available_for_rows = panel_height - fixed_height
            rows = max(available_for_rows, 3)  # At least 3 rows

            self.rows_per_page = rows
        except Exception:
            self.rows_per_page = 5

    def action_switch_view(self) -> None:
        """Switch between active and recent sessions view."""
        if self.current_view == "active":
            self.current_view = "recent"
        else:
            self.current_view = "active"
        self.current_page = 1
        self.refresh_data()

    def action_prev_page(self) -> None:
        """Go to previous page."""
        if self.current_page > 1:
            self.current_page -= 1
            if self.current_view == "recent":
                self.refresh_data()
            else:
                self._update_display()

    def action_next_page(self) -> None:
        """Go to next page."""
        total_pages = self._get_total_pages()
        if self.current_page < total_pages:
            self.current_page += 1
            if self.current_view == "recent":
                self.refresh_data()
            else:
                self._update_display()

    def _get_total_items(self) -> int:
        if self.current_view == "active":
            return len(self._all_active_sessions)
        return int(self._recent_total_sessions)

    def _get_total_pages(self) -> int:
        """Calculate total pages for current view."""
        total_items = self._get_total_items()
        if total_items == 0:
            return 1
        return (total_items + self.rows_per_page - 1) // self.rows_per_page

    def _on_refresh_timer(self) -> None:
        self.refresh_data(force=False)

    def refresh_data(self, *, force: bool = True) -> None:
        """Refresh watcher data without blocking the UI."""
        if not self.display:
            return

        snapshot = (str(self.current_view), int(self.current_page), int(self.rows_per_page))

        if self._refresh_worker is not None and self._refresh_worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        ):
            if force:
                self._pending_refresh_snapshot = snapshot
            return

        if force and self._pending_refresh_snapshot is not None:
            snapshot = self._pending_refresh_snapshot
            self._pending_refresh_snapshot = None

        def work() -> dict:
            return self._collect_snapshot(*snapshot)

        self._active_refresh_snapshot = snapshot
        self._pending_refresh_snapshot = None
        self._refresh_worker = self.run_worker(work, thread=True, exit_on_error=False)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if self._refresh_worker is None or event.worker is not self._refresh_worker:
            return

        if event.state == WorkerState.ERROR:
            result = {
                "snapshot": self._active_refresh_snapshot,
                "status_data": {
                    "is_running": False,
                    "pid": None,
                    "mode": "error",
                    "error": str(self._refresh_worker.error),
                },
                "monitored_paths": [],
            }
        elif event.state != WorkerState.SUCCESS:
            return
        else:
            result = self._refresh_worker.result or {}

        try:
            self._status_data = dict(result.get("status_data") or {})
        except Exception:
            self._status_data = {}
        try:
            self._monitored_paths = list(result.get("monitored_paths") or [])
        except Exception:
            self._monitored_paths = []

        snapshot = result.get("snapshot")
        if snapshot == (str(self.current_view), int(self.current_page), int(self.rows_per_page)):
            if self.current_view == "active":
                try:
                    self._all_active_sessions = list(result.get("active_sessions") or [])
                except Exception:
                    self._all_active_sessions = []
            else:
                try:
                    self._recent_sessions_page = list(result.get("recent_sessions_page") or [])
                except Exception:
                    self._recent_sessions_page = []
                try:
                    self._recent_total_sessions = int(result.get("recent_total_sessions") or 0)
                except Exception:
                    self._recent_total_sessions = 0

            self._update_display()

        if self._pending_refresh_snapshot is not None:
            self.refresh_data()

    def _collect_snapshot(self, view: str, page: int, rows_per_page: int) -> dict:
        """Collect all display data (safe to run in a background thread)."""
        status_data: dict
        try:
            from ...commands.watcher import detect_watcher_process

            is_running, pid, mode = detect_watcher_process()
            status_data = {"is_running": is_running, "pid": pid, "mode": mode}
        except Exception as e:
            status_data = {"is_running": False, "pid": None, "mode": "error", "error": str(e)}

        monitored_paths: List[Tuple[str, str, bool]]
        try:
            from ...config import ReAlignConfig

            config = ReAlignConfig.load()
            monitored_paths = [
                ("Claude", "~/.claude/projects/", config.auto_detect_claude),
                ("Codex", "~/.codex/sessions/", config.auto_detect_codex),
                ("Gemini", "~/.gemini/tmp/", config.auto_detect_gemini),
            ]
        except Exception:
            monitored_paths = []

        active_sessions: list[dict] = []
        recent_sessions_page: list[dict] = []
        recent_total_sessions: int = 0

        if view == "active":
            active_sessions = self._collect_active_sessions()
        else:
            recent_sessions_page, recent_total_sessions = self._collect_recent_sessions_page(
                page=page, rows_per_page=rows_per_page
            )

        return {
            "snapshot": (view, int(page), int(rows_per_page)),
            "status_data": status_data,
            "monitored_paths": monitored_paths,
            "active_sessions": active_sessions,
            "recent_sessions_page": recent_sessions_page,
            "recent_total_sessions": recent_total_sessions,
        }

    def _collect_active_sessions(self) -> list[dict]:
        """Collect all active sessions being monitored, sorted by time (newest first)."""
        try:
            from ...config import ReAlignConfig
            from ...hooks import find_all_active_sessions
            from ...adapters import get_adapter_registry

            config = ReAlignConfig.load()
            active_session_paths = find_all_active_sessions(config, project_path=None)
            registry = get_adapter_registry()

            sessions_with_time: list[dict] = []
            for session_path in active_session_paths:
                try:
                    adapter = registry.auto_detect_adapter(session_path)
                    source = adapter.name.capitalize() if adapter else "Unknown"

                    project_name = "-"
                    if adapter:
                        try:
                            p_path = adapter.extract_project_path(session_path)
                            if p_path:
                                project_name = p_path.name
                        except Exception:
                            pass

                    mtime = session_path.stat().st_mtime
                    mtime_dt = datetime.fromtimestamp(mtime)

                    sessions_with_time.append(
                        {
                            "name": self._shorten_session_id(session_path.name),
                            "source": source,
                            "project": project_name,
                            "last_activity": self._format_relative_time(mtime_dt),
                            "mtime": mtime,
                        }
                    )
                except Exception:
                    continue

            sessions_with_time.sort(key=lambda x: x.get("mtime", 0), reverse=True)
            return sessions_with_time
        except Exception:
            return []

    def _collect_recent_sessions_page(
        self, *, page: int, rows_per_page: int
    ) -> tuple[list[dict], int]:
        """Collect one page of recent sessions from the database."""
        try:
            from ...db import get_database

            db = get_database()
            conn = db._get_connection()

            count_row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
            total = int(count_row[0]) if count_row else 0

            offset = max(0, (int(page) - 1) * int(rows_per_page))
            sessions = list(
                conn.execute(
                    """
                    SELECT id, session_type, workspace_path, last_activity_at
                    FROM sessions
                    ORDER BY last_activity_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (int(rows_per_page), int(offset)),
                )
            )

            out: list[dict] = []
            for s in sessions:
                session_id = s[0]
                session_type = s[1] or "unknown"
                workspace = s[2]
                last_activity = s[3]

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

                out.append(
                    {
                        "name": self._shorten_session_id(str(session_id)),
                        "source": source,
                        "project": project,
                        "last_activity": activity_str,
                    }
                )

            return out, total
        except Exception:
            return [], 0

    def _update_display(self) -> None:
        """Update the display with current data."""
        # Update status
        status_widget = self.query_one("#watcher-status", Static)
        is_running = self._status_data.get("is_running", False)
        pid = self._status_data.get("pid")

        if is_running:
            status_text = f"[bold]Status:[/bold] [green]● Running[/green]"
            if pid:
                status_text += f" (PID: {pid})"
        else:
            status_text = f"[bold]Status:[/bold] [red]● Stopped[/red]"

        status_text += f"  [bold]Mode:[/bold] Standalone (SQLite)"
        status_widget.update(status_text)

        # Update monitored paths
        paths_widget = self.query_one("#monitored-paths", Static)
        paths_lines = ["[bold]Monitored Paths:[/bold]"]
        for name, path, enabled in self._monitored_paths:
            if enabled:
                expanded_path = Path(path.replace("~", str(Path.home())))
                exists = expanded_path.exists()
                if exists:
                    paths_lines.append(f"  [green]●[/green] {name}: {path}")
                else:
                    paths_lines.append(
                        f"  [yellow]○[/yellow] {name}: {path} [dim](not found)[/dim]"
                    )
            else:
                paths_lines.append(f"  [dim]○ {name}: {path} (disabled)[/dim]")
        paths_widget.update("\n".join(paths_lines))

        # Update section header
        header_widget = self.query_one("#section-header", Static)
        if self.current_view == "active":
            title = "[bold]Active Sessions (File System)[/bold]"
            other = "Database"
        else:
            title = "[bold]Recent Sessions (Database)[/bold]"
            other = "File System"
        header_widget.update(f"{title}  [dim]│ (s) switch to {other}[/dim]")

        # Update table
        table = self.query_one("#sessions-table", DataTable)
        table.clear()

        start_idx = (self.current_page - 1) * self.rows_per_page
        if self.current_view == "active":
            data = self._all_active_sessions
            page_data = data[start_idx : start_idx + self.rows_per_page]
            total_items = len(data)
        else:
            page_data = self._recent_sessions_page
            total_items = self._recent_total_sessions

        for i, session in enumerate(page_data, start=start_idx + 1):
            table.add_row(
                str(i),
                session["name"],
                session["source"],
                session["project"],
                session["last_activity"],
            )

        # Update pagination info
        total_pages = self._get_total_pages()
        pagination_widget = self.query_one("#pagination-info", Static)
        pagination_widget.update(
            f"[dim]Page {self.current_page}/{total_pages} ({total_items} total)  │  (p) prev  (n) next  (s) switch view[/dim]"
        )

    def _shorten_session_id(self, session_id: str) -> str:
        """Shorten a session ID for display."""
        if len(session_id) > 23:
            return session_id[:10] + "..." + session_id[-10:]
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
