"""Worker Status Panel Widget with view switching and pagination."""

from pathlib import Path
from typing import Dict, Optional

from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.worker import Worker, WorkerState
from textual.widgets import DataTable, Static


class WorkerPanel(Container, can_focus=True):
    """Panel displaying Worker status with view switching and pagination."""

    DEFAULT_CSS = """
    WorkerPanel {
        height: 100%;
        padding: 1;
        overflow: hidden;
    }

    WorkerPanel:focus {
        border: solid $accent;
    }

    WorkerPanel .status-section {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary-darken-2;
    }

    WorkerPanel .stats-section {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary-darken-2;
    }

    WorkerPanel .section-header {
        height: auto;
        margin-bottom: 1;
    }

    WorkerPanel .table-container {
        height: auto;
        overflow: hidden;
    }

    WorkerPanel DataTable {
        height: auto;
        max-height: 100%;
        overflow: hidden;
    }

    WorkerPanel .pagination-info {
        height: 1;
        margin-top: 1;
        color: $text-muted;
        text-align: center;
    }
    """

    # Reactive properties
    current_view: reactive[str] = reactive("turn")  # "turn" or "session"
    current_page: reactive[int] = reactive(1)
    rows_per_page: reactive[int] = reactive(10)

    def __init__(self) -> None:
        super().__init__()
        self._status_data: dict = {}
        self._job_counts: dict = {}
        self._jobs: list[dict] = []
        self._total_jobs: int = 0
        self._db_path: Optional[str] = None
        self._refresh_worker: Optional[Worker] = None
        self._refresh_timer = None
        self._active_refresh_snapshot: Optional[tuple[str, int, int]] = None
        self._pending_refresh_snapshot: Optional[tuple[str, int, int]] = None

    def compose(self) -> ComposeResult:
        """Compose the worker panel layout."""
        yield Static(id="worker-status", classes="status-section")
        yield Static(id="queue-stats", classes="stats-section")
        yield Static(id="section-header", classes="section-header")
        with Container(classes="table-container"):
            yield DataTable(id="jobs-table")
        yield Static(id="pagination-info", classes="pagination-info")

    def on_mount(self) -> None:
        """Set up the panel on mount."""
        # Initialize table
        table = self.query_one("#jobs-table", DataTable)
        self._setup_table_columns(table)
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
            self.refresh_data()
            return
        self._update_display()

    def _calculate_rows_per_page(self) -> None:
        """Calculate rows per page based on available height."""
        try:
            panel_height = self.size.height

            # Calculate exact space needed:
            # - Status section: ~3 lines content + 2 border + 2 padding + 1 margin = 8
            # - Stats section: ~5 lines content + 2 border + 2 padding + 1 margin = 10
            # - Section header: 1 line + 1 margin = 2
            # - Table header row: 1
            # - Pagination info: 1 line + 1 margin = 2
            # - Panel padding: 2 (top + bottom)
            # - Extra buffer: 3

            fixed_height = 8 + 10 + 2 + 1 + 2 + 2 + 3  # = 28

            available_for_rows = panel_height - fixed_height
            rows = max(available_for_rows, 3)  # At least 3 rows

            self.rows_per_page = rows
        except Exception:
            self.rows_per_page = 5

    def _setup_table_columns(self, table: DataTable) -> None:
        """Set up table columns based on current view."""
        table.clear(columns=True)
        if self.current_view == "turn":
            table.add_columns("#", "Status", "Source", "Session", "Turn", "Attempts")
        else:
            table.add_columns("#", "Status", "Source", "Session", "Attempts")

    def action_switch_view(self) -> None:
        """Switch between turn jobs and session jobs view."""
        if self.current_view == "turn":
            self.current_view = "session"
        else:
            self.current_view = "turn"
        self.current_page = 1

        # Re-setup table columns
        table = self.query_one("#jobs-table", DataTable)
        self._setup_table_columns(table)

        self.refresh_data()

    def action_prev_page(self) -> None:
        """Go to previous page."""
        if self.current_page > 1:
            self.current_page -= 1
            self.refresh_data()

    def action_next_page(self) -> None:
        """Go to next page."""
        total_pages = self._get_total_pages()
        if self.current_page < total_pages:
            self.current_page += 1
            self.refresh_data()

    def _get_current_data(self) -> list:
        """Get data for current view."""
        return self._jobs

    def _get_total_pages(self) -> int:
        """Calculate total pages for current view."""
        if self._total_jobs == 0:
            return 1
        return (self._total_jobs + self.rows_per_page - 1) // self.rows_per_page

    def _on_refresh_timer(self) -> None:
        self.refresh_data(force=False)

    def refresh_data(self, *, force: bool = True) -> None:
        """Refresh all worker data without blocking the UI."""
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
                "status_data": {
                    "is_running": False,
                    "pid": None,
                    "mode": "error",
                    "error": str(self._refresh_worker.error),
                }
            }
        elif event.state != WorkerState.SUCCESS:
            return
        else:
            result = self._refresh_worker.result or {}

        # Always update status/stats; jobs update is snapshot-aware (view/page could have changed).
        try:
            self._status_data = dict(result.get("status_data") or {})
        except Exception:
            self._status_data = {}
        self._db_path = result.get("db_path") or None
        try:
            self._job_counts = dict(result.get("job_counts") or {})
        except Exception:
            self._job_counts = {}

        snapshot = result.get("snapshot")
        if snapshot == (str(self.current_view), int(self.current_page), int(self.rows_per_page)):
            try:
                self._jobs = list(result.get("jobs") or [])
            except Exception:
                self._jobs = []
            try:
                self._total_jobs = int(result.get("total_jobs") or 0)
            except Exception:
                self._total_jobs = 0
            self._update_display()

        if self._pending_refresh_snapshot is not None:
            self.refresh_data()

    def _collect_snapshot(self, view: str, page: int, rows_per_page: int) -> dict:
        """Collect all display data (safe to run in a background thread)."""
        status_data: dict
        try:
            from ...commands.worker import detect_worker_process

            is_running, pid, mode = detect_worker_process()
            status_data = {"is_running": is_running, "pid": pid, "mode": mode}
        except Exception as e:
            status_data = {"is_running": False, "pid": None, "mode": "error", "error": str(e)}

        db_path: Optional[str] = None
        job_counts: dict = {}
        total_jobs: int = 0
        jobs: list[dict] = []

        try:
            from ...db import get_database
            from ...db.sqlite_db import SQLiteDatabase

            dbi = get_database()
            if isinstance(dbi, SQLiteDatabase):
                db_path = str(Path(str(getattr(dbi, "db_path", ""))).expanduser())
                job_counts = dbi.get_job_counts()

                kinds = ["turn_summary"] if view == "turn" else ["session_summary"]
                total_jobs = dbi.count_jobs(kinds=kinds)

                offset = max(0, (int(page) - 1) * int(rows_per_page))
                raw_jobs = dbi.list_jobs(limit=int(rows_per_page), offset=offset, kinds=kinds)

                session_ids: set[str] = set()
                for job in raw_jobs:
                    sid = self._extract_session_id(job)
                    if sid:
                        session_ids.add(sid)

                session_types: Dict[str, str] = {}
                if session_ids:
                    for s in dbi.get_sessions_by_ids(list(session_ids)):
                        session_types[str(s.id)] = str(getattr(s, "session_type", "") or "")

                for job in raw_jobs:
                    if view == "turn":
                        jobs.append(self._process_turn_job(job, session_types))
                    else:
                        jobs.append(self._process_session_job(job, session_types))
        except Exception:
            db_path = None
            job_counts = {}
            total_jobs = 0
            jobs = []

        return {
            "snapshot": (view, int(page), int(rows_per_page)),
            "status_data": status_data,
            "db_path": db_path,
            "job_counts": job_counts,
            "total_jobs": total_jobs,
            "jobs": jobs,
        }

    def _process_turn_job(self, job: dict, session_types: Dict[str, str]) -> dict:
        """Process a turn job for display."""
        status, color = self._format_status(job.get("status", ""))
        source = self._get_source_from_job(job, session_types)
        session_id = self._shorten_session_id(self._extract_session_id(job))
        turn_number = self._extract_turn_number(job)
        attempts = str(job.get("attempts", 0))

        return {
            "status": status,
            "color": color,
            "source": source,
            "session": session_id,
            "turn": turn_number,
            "attempts": attempts,
        }

    def _process_session_job(self, job: dict, session_types: Dict[str, str]) -> dict:
        """Process a session job for display."""
        status, color = self._format_status(job.get("status", ""))
        source = self._get_source_from_job(job, session_types)
        session_id = self._shorten_session_id(self._extract_session_id(job))
        attempts = str(job.get("attempts", 0))

        return {
            "status": status,
            "color": color,
            "source": source,
            "session": session_id,
            "attempts": attempts,
        }

    def _extract_session_id(self, job: dict) -> str:
        """Extract session ID from a job."""
        payload = job.get("payload") or {}
        session_id = str(payload.get("session_id") or "").strip()
        if session_id:
            return session_id

        dedupe_key = str(job.get("dedupe_key") or "")
        if dedupe_key.startswith("turn:"):
            parts = dedupe_key.split(":")
            if len(parts) >= 3:
                return parts[1].strip()
        if dedupe_key.startswith("session:"):
            return dedupe_key.split(":", 1)[1].strip()
        return ""

    def _get_source_from_job(self, job: dict, session_types: Dict[str, str]) -> str:
        """Get source type from a job."""
        payload = job.get("payload") or {}
        raw = str(payload.get("session_type") or "").strip()
        if raw:
            return raw.capitalize()

        sid = self._extract_session_id(job)
        source = str(session_types.get(sid, "") or "").lower()
        return source.capitalize() if source else "-"

    def _update_display(self) -> None:
        """Update the display with current data."""
        # Update status
        status_widget = self.query_one("#worker-status", Static)
        is_running = self._status_data.get("is_running", False)
        pid = self._status_data.get("pid")

        if is_running:
            status_text = f"[bold]Status:[/bold] [green]● Running[/green]"
            if pid:
                status_text += f" (PID: {pid})"
        else:
            status_text = f"[bold]Status:[/bold] [red]● Stopped[/red]"

        if self._db_path:
            status_text += f"\n[bold]DB:[/bold] {self._db_path}"

        status_widget.update(status_text)

        # Update queue stats with ASCII bar chart
        stats_widget = self.query_one("#queue-stats", Static)
        stats_text = self._render_queue_stats()
        stats_widget.update(stats_text)

        # Update section header
        header_widget = self.query_one("#section-header", Static)
        if self.current_view == "turn":
            title = "[bold]Turn Jobs[/bold]"
            other = "Session Jobs"
        else:
            title = "[bold]Session Jobs[/bold]"
            other = "Turn Jobs"
        header_widget.update(f"{title}  [dim]│ (s) switch to {other}[/dim]")

        # Update table
        table = self.query_one("#jobs-table", DataTable)
        table.clear()

        data = self._get_current_data()
        start_idx = (self.current_page - 1) * self.rows_per_page
        for i, job in enumerate(data, start=start_idx + 1):
            if self.current_view == "turn":
                table.add_row(
                    str(i),
                    f"[{job['color']}]{job['status']}[/{job['color']}]",
                    job["source"],
                    job["session"],
                    job["turn"],
                    job["attempts"],
                )
            else:
                table.add_row(
                    str(i),
                    f"[{job['color']}]{job['status']}[/{job['color']}]",
                    job["source"],
                    job["session"],
                    job["attempts"],
                )

        # Update pagination info
        total_pages = self._get_total_pages()
        total_items = self._total_jobs
        pagination_widget = self.query_one("#pagination-info", Static)
        pagination_widget.update(
            f"[dim]Page {self.current_page}/{total_pages} ({total_items} total)  │  (p) prev  (n) next  (s) switch view[/dim]"
        )

    def _render_queue_stats(self) -> str:
        """Render queue statistics as ASCII bar chart."""
        queued = self._job_counts.get("queued", 0)
        processing = self._job_counts.get("processing", 0)
        done = self._job_counts.get("done", 0)
        failed = self._job_counts.get("failed", 0)

        total = queued + processing + done + failed
        if total == 0:
            return "[bold]Queue Statistics:[/bold] (empty)"

        # Create bar chart
        max_width = 40
        lines = ["[bold]Queue Statistics:[/bold]"]

        for label, count, color in [
            ("Queued", queued, "cyan"),
            ("Processing", processing, "yellow"),
            ("Done", done, "green"),
            ("Failed", failed, "red"),
        ]:
            bar_width = int((count / total) * max_width) if total > 0 else 0
            bar = (
                "["
                + color
                + "]"
                + ("█" * bar_width)
                + "[/"
                + color
                + "]"
                + ("░" * (max_width - bar_width))
            )
            lines.append(f"  {label:<12} {bar}  {count}")

        return "\n".join(lines)

    def _format_status(self, status: str) -> tuple:
        """Format job status with color."""
        status = (status or "").lower()
        if status == "processing":
            return "running", "yellow"
        if status in ("queued", "retry"):
            return "queued", "cyan"
        if status == "done":
            return "done", "green"
        if status == "failed":
            return "failed", "red"
        return status or "unknown", "white"

    def _extract_turn_number(self, job: dict) -> str:
        """Extract turn number from a job."""
        payload = job.get("payload") or {}
        turn_number = payload.get("turn_number")
        if turn_number is not None:
            return str(turn_number)

        dedupe_key = str(job.get("dedupe_key") or "")
        if dedupe_key.startswith("turn:"):
            parts = dedupe_key.split(":")
            if len(parts) >= 3:
                return parts[2]
        return "-"

    def _shorten_session_id(self, session_id: str) -> str:
        """Shorten a session ID for display."""
        if not session_id:
            return "-"
        if len(session_id) > 18:
            return session_id[:6] + "..." + session_id[-6:]
        return session_id
