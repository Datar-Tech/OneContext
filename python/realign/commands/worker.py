"""Aline worker commands - Manage background job worker daemon."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from rich.console import Console
from rich.table import Table

from ..db import get_database
from ..db.sqlite_db import SQLiteDatabase
from ..logging_config import setup_logger

logger = setup_logger("realign.worker", "worker.log")
console = Console()


def get_worker_pid_file() -> Path:
    return Path.home() / ".aline/.logs/worker.pid"


def detect_worker_process() -> tuple[bool, int | None, str]:
    """
    Detect whether worker daemon is running.

    Returns:
        (is_running, pid, mode)
    """
    pid_file = get_worker_pid_file()
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            return True, pid, "pidfile"
        except Exception:
            pass

    try:
        output = subprocess.check_output(["ps", "aux"], text=True)
        for line in output.splitlines():
            if "worker_daemon.py" in line and "grep" not in line:
                parts = line.split()
                if len(parts) >= 2:
                    return True, int(parts[1]), "ps"
    except Exception:
        pass

    return False, None, "none"


def detect_all_worker_processes() -> list[tuple[int, str]]:
    processes: list[tuple[int, str]] = []
    try:
        output = subprocess.check_output(["ps", "aux"], text=True)
        for line in output.splitlines():
            if "worker_daemon.py" in line and "grep" not in line:
                parts = line.split()
                if len(parts) >= 2:
                    processes.append((int(parts[1]), "ps"))
    except Exception:
        pass

    # If pidfile exists and isn't in list, add it.
    pid_file = get_worker_pid_file()
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if all(p != pid for p, _ in processes):
                processes.append((pid, "pidfile"))
        except Exception:
            pass

    return processes


def worker_status_command(verbose: bool = False, *, json_output: bool = False) -> int:
    try:
        is_running, pid, mode = detect_worker_process()
        status = "Running" if is_running else "Stopped"

        # Show job counts if DB available
        try:
            dbi = get_database()
            db_path = (
                Path(str(getattr(dbi, "db_path", ""))).expanduser()
                if isinstance(dbi, SQLiteDatabase)
                else None
            )
            if isinstance(dbi, SQLiteDatabase):
                counts = dbi.get_job_counts()
                turn_jobs = dbi.list_jobs(limit=30, kinds=["turn_summary"])
                session_jobs = dbi.list_jobs(limit=30, kinds=["session_summary"])
            else:
                counts = {}
                turn_jobs = []
                session_jobs = []
            job_kinds = {}
            if isinstance(dbi, SQLiteDatabase):
                try:
                    conn = dbi._get_connection()
                    rows = conn.execute(
                        "SELECT kind, COUNT(*) AS c FROM jobs GROUP BY kind ORDER BY c DESC"
                    ).fetchall()
                    for row in rows or []:
                        job_kinds[str(row["kind"])] = int(row["c"])
                except Exception:
                    pass
        except Exception:
            dbi = None
            db_path = None
            counts = {}
            turn_jobs = []
            session_jobs = []
            job_kinds = {}

        if json_output:

            def _session_id_from_job(job: dict) -> str:
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

            session_type_by_id: dict[str, str] = {}
            if isinstance(dbi, SQLiteDatabase):
                session_ids = sorted(
                    {
                        sid
                        for sid in (_session_id_from_job(j) for j in (turn_jobs + session_jobs))
                        if sid
                    }
                )
                if session_ids:
                    for s in dbi.get_sessions_by_ids(session_ids):
                        session_type_by_id[str(s.id)] = str(getattr(s, "session_type", "") or "")

            def _source_from_job(job: dict) -> str:
                payload = job.get("payload") or {}
                raw = str(payload.get("session_type") or "").strip()
                if raw:
                    return raw.lower()
                sid = _session_id_from_job(job)
                return str(session_type_by_id.get(sid, "") or "").lower() or ""

            def _parse_dedupe_key_turn(key: str) -> tuple[str, Optional[int]]:
                parts = (key or "").split(":")
                if len(parts) >= 3 and parts[0] == "turn":
                    try:
                        return parts[1], int(parts[2])
                    except Exception:
                        return parts[1], None
                return "", None

            def _normalize_job(job: dict) -> dict:
                payload = job.get("payload") or {}
                dedupe_key = str(job.get("dedupe_key") or "")
                session_id = _session_id_from_job(job) or None
                turn_number = None
                turn_id = None
                if str(job.get("kind") or "") == "turn_summary":
                    turn_number = payload.get("turn_number")
                    if turn_number is None:
                        _, turn_number = _parse_dedupe_key_turn(dedupe_key)
                    try:
                        if turn_number is not None:
                            turn_number = int(turn_number)
                    except Exception:
                        turn_number = None
                    if dedupe_key.startswith("turn:"):
                        turn_id = dedupe_key

                return {
                    "id": job.get("id"),
                    "kind": job.get("kind"),
                    "dedupe_key": dedupe_key,
                    "status": job.get("status"),
                    "priority": job.get("priority"),
                    "attempts": job.get("attempts"),
                    "next_run_at": job.get("next_run_at"),
                    "locked_until": job.get("locked_until"),
                    "locked_by": job.get("locked_by"),
                    "reschedule": job.get("reschedule"),
                    "last_error": job.get("last_error"),
                    "created_at": job.get("created_at"),
                    "updated_at": job.get("updated_at"),
                    "payload": payload,
                    "session_id": session_id,
                    "session_type": _source_from_job(job) or None,
                    "turn_number": turn_number,
                    "turn_id": turn_id,
                }

            data = {
                "worker": {
                    "status": status.lower(),
                    "running": is_running,
                    "pid": pid,
                    "mode": mode,
                },
                "db_path": str(db_path) if db_path else None,
                "job_counts": counts,
                "job_kinds": job_kinds,
                "turn_jobs": [_normalize_job(j) for j in turn_jobs],
                "session_jobs": [_normalize_job(j) for j in session_jobs],
            }
            print(json.dumps(data, ensure_ascii=True))
            return 0

        console.print(
            f"[bold]Worker:[/bold] {status}" + (f" (PID: {pid}, mode: {mode})" if pid else "")
        )
        if db_path:
            console.print(f"[dim]DB:[/dim] {db_path}")
        if counts:
            pairs = " | ".join([f"{k}: {v}" for k, v in sorted(counts.items())])
            console.print(f"[dim]Jobs:[/dim] {pairs}")
        if isinstance(dbi, SQLiteDatabase):
            max_attempts = os.getenv("REALIGN_JOB_MAX_ATTEMPTS", "10")
            console.print(f"[dim]Max attempts:[/dim] {max_attempts}")
            if os.getenv("REALIGN_DISABLE_AUTO_SUMMARIES"):
                console.print(
                    "[yellow]Note:[/yellow] REALIGN_DISABLE_AUTO_SUMMARIES is set; "
                    "session_summary jobs may not be enqueued automatically."
                )
            if job_kinds:
                pairs = " | ".join([f"{k}: {v}" for k, v in sorted(job_kinds.items())])
                console.print(f"[dim]Job kinds:[/dim] {pairs}")

        try:

            def fmt_status(s: str) -> tuple[str, str]:
                s = (s or "").lower()
                if s == "processing":
                    return "running", "yellow"
                if s in ("queued", "retry"):
                    return "queued", "cyan"
                if s == "done":
                    return "done", "green"
                if s == "failed":
                    return "failed", "red"
                return s or "unknown", "white"

            def _session_id_from_job(job: dict) -> str:
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

            session_type_by_id: dict[str, str] = {}
            if isinstance(dbi, SQLiteDatabase):
                session_ids = sorted(
                    {
                        sid
                        for sid in (_session_id_from_job(j) for j in (turn_jobs + session_jobs))
                        if sid
                    }
                )
                if session_ids:
                    for s in dbi.get_sessions_by_ids(session_ids):
                        session_type_by_id[str(s.id)] = str(getattr(s, "session_type", "") or "")

            def _source_from_job(job: dict) -> str:
                payload = job.get("payload") or {}
                raw = str(payload.get("session_type") or "").strip()
                if raw:
                    return raw.lower()
                sid = _session_id_from_job(job)
                return str(session_type_by_id.get(sid, "") or "").lower() or "-"

            def render_table(title: str, jobs: list[dict], *, show_turn: bool) -> None:
                if not jobs:
                    console.print(f"[dim]{title}:[/dim] (none)")
                    return

                table = Table(title=title, show_lines=False)
                table.add_column("Type", style="bold")
                table.add_column("Status")
                table.add_column("Source", style="dim", no_wrap=True)
                table.add_column("Session", no_wrap=True)
                if show_turn:
                    table.add_column("Turn", no_wrap=True)
                table.add_column("Att.", justify="right")
                table.add_column("Updated", style="dim")

                for job in jobs:
                    kind = str(job.get("kind") or "")
                    payload = job.get("payload") or {}
                    dedupe_key = str(job.get("dedupe_key") or "")

                    def _parse_dedupe_key_turn(key: str) -> tuple[str, str]:
                        # Expected: "turn:<session_id>:<turn_number>"
                        parts = (key or "").split(":")
                        if len(parts) >= 3 and parts[0] == "turn":
                            return parts[1], parts[2]
                        return "", ""

                    def _short_session_id(session_id: str) -> str:
                        s = (session_id or "").strip()
                        if not s:
                            return ""
                        if len(s) <= 12:
                            return s
                        return f"{s[:6]}...{s[-6:]}"

                    if kind == "turn_summary":
                        job_type = "[cyan]TURN[/cyan]"
                        session_id = str(payload.get("session_id") or "")
                        turn_number = str(payload.get("turn_number") or "")
                        if not session_id or not turn_number:
                            session_id, turn_number = _parse_dedupe_key_turn(dedupe_key)
                        sid = _short_session_id(session_id)
                        session_display = sid or (session_id or "-")
                        turn_display = turn_number or "-"
                    elif kind == "session_summary":
                        job_type = "[magenta]SESSION[/magenta]"
                        session_id = str(payload.get("session_id") or "")
                        if not session_id and dedupe_key.startswith("session:"):
                            session_id = dedupe_key.split(":", 1)[1]
                        session_display = _short_session_id(session_id) or (session_id or "-")
                        turn_display = "-"
                    else:
                        job_type = "[white]JOB[/white]"
                        session_display = (dedupe_key or "-")[:80]
                        turn_display = "-"

                    status_label, color = fmt_status(str(job.get("status") or ""))
                    source = _source_from_job(job)
                    attempts = str(job.get("attempts") or 0)
                    updated = str(job.get("updated_at") or "") or "-"

                    if show_turn:
                        table.add_row(
                            job_type,
                            f"[{color}]{status_label}[/{color}]",
                            source,
                            session_display,
                            turn_display,
                            attempts,
                            updated,
                        )
                    else:
                        table.add_row(
                            job_type,
                            f"[{color}]{status_label}[/{color}]",
                            source,
                            session_display,
                            attempts,
                            updated,
                        )

                console.print(table)
                if show_turn:
                    console.print(
                        "[dim]Legend: Session=session id (shortened); Turn=turn number (TURN jobs only); "
                        "Source=session type (claude/codex/etc); Updated=last job row update.[/dim]"
                    )
                else:
                    console.print(
                        "[dim]Legend: Session=session id (shortened); "
                        "Source=session type (claude/codex/etc); Updated=last job row update.[/dim]"
                    )

            render_table("Turn Jobs (top 30)", turn_jobs, show_turn=True)
            render_table("Session Jobs (top 30)", session_jobs, show_turn=False)

            if not session_jobs and turn_jobs:
                try:
                    if isinstance(dbi, SQLiteDatabase):
                        conn = dbi._get_connection()
                        row = conn.execute(
                            "SELECT COUNT(*) AS c FROM sessions WHERE summary_updated_at IS NOT NULL"
                        ).fetchone()
                        has_summary = int(row["c"]) if row else 0
                        row = conn.execute(
                            "SELECT COUNT(*) AS c FROM sessions WHERE summary_updated_at IS NULL"
                        ).fetchone()
                        missing_summary = int(row["c"]) if row else 0
                        console.print(
                            f"[dim]Hint: no session_summary jobs found in queue; "
                            f"sessions with summary: {has_summary}, missing: {missing_summary}. "
                            "This usually means session summaries were generated via import/refresh, "
                            "or auto-enqueue was disabled earlier (REALIGN_DISABLE_AUTO_SUMMARIES), "
                            "or an older version created turns without enqueueing session_summary jobs. "
                            "Check watcher/worker logs for 'Failed to enqueue session summary job'.[/dim]"
                        )
                    else:
                        console.print(
                            "[dim]Hint: session_summary jobs are typically enqueued after a completed "
                            "turn_summary, or when a completed turn is created (unless auto summaries are disabled).[/dim]"
                        )
                except Exception:
                    console.print(
                        "[dim]Hint: session_summary jobs are typically enqueued after a completed "
                        "turn_summary, or when a completed turn is created (unless auto summaries are disabled).[/dim]"
                    )
            if verbose and not turn_jobs and not session_jobs:
                console.print("[dim]Jobs: (no data)[/dim]")
        except Exception as e:
            if verbose:
                console.print(f"[dim]Jobs: unavailable ({e})[/dim]")

        if not is_running:
            console.print("[dim]Run 'aline worker start' to start the worker[/dim]")
        return 0
    except Exception as e:
        logger.error(f"Error in worker status: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1


def worker_repair_command(*, force: bool = False) -> int:
    """
    Repair the durable jobs queue by requeueing orphaned processing jobs.

    This is useful after a worker crash/kill where jobs can remain stuck as 'processing'.
    """
    try:
        dbi = get_database()
        if not isinstance(dbi, SQLiteDatabase):
            console.print("[red]Error:[/red] repair is only supported for SQLite backend")
            return 1

        n = dbi.requeue_stale_processing_jobs(force=force)
        if force:
            console.print(f"[yellow]Requeued processing jobs (force):[/yellow] {n}")
            console.print(
                "[dim]Note: --force will requeue even non-expired leases; avoid running it while a worker is active.[/dim]"
            )
        else:
            console.print(f"[green]Requeued stale processing jobs:[/green] {n}")
            console.print("[dim]Only jobs with expired/missing leases were requeued.[/dim]")

        return 0
    except Exception as e:
        logger.error(f"Error in worker repair: {e}", exc_info=True)
        console.print(f"[red]Error:[/red] {e}")
        return 1


def worker_start_command() -> int:
    try:
        # Check login status first
        from ..auth import is_logged_in

        if not is_logged_in():
            console.print("[red]✗ Not logged in. Worker requires authentication.[/red]")
            console.print("[dim]Run 'aline login' first.[/dim]")
            return 1

        is_running, pid, mode = detect_worker_process()
        if is_running:
            console.print(f"[yellow]Worker is already running (PID: {pid}, mode: {mode})[/yellow]")
            return 0

        console.print("[cyan]Starting worker daemon...[/cyan]")

        spec = importlib.util.find_spec("realign.worker_daemon")
        if not spec or not spec.origin:
            console.print("[red]✗ Could not find worker daemon module[/red]")
            return 1

        daemon_script = spec.origin
        subprocess.Popen(
            [sys.executable, daemon_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=Path.cwd(),
            env=os.environ.copy(),
        )

        import time

        time.sleep(1)
        is_running, pid, mode = detect_worker_process()
        if is_running:
            console.print(f"[green]✓ Worker started successfully (PID: {pid})[/green]")
            console.print("[dim]File logs are disabled (except dashboard.log)[/dim]")
            return 0

        console.print("[red]✗ Failed to start worker[/red]")
        console.print("[dim]No worker log file is generated in this mode.[/dim]")
        return 1

    except Exception as e:
        logger.error(f"Error in worker start: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1


def worker_stop_command() -> int:
    import time

    try:
        all_processes = detect_all_worker_processes()
        if not all_processes:
            console.print("[yellow]No worker processes found[/yellow]")
            console.print("[dim]Use 'aline worker start' to start it[/dim]")
            return 1

        if len(all_processes) == 1:
            pid, mode = all_processes[0]
            console.print(f"[cyan]Stopping worker (PID: {pid}, mode: {mode})...[/cyan]")
        else:
            console.print(
                f"[cyan]Found {len(all_processes)} worker processes, stopping all...[/cyan]"
            )

        for pid, _mode in all_processes:
            try:
                os.kill(pid, 15)  # SIGTERM
            except ProcessLookupError:
                pass
            except Exception as e:
                console.print(f"[yellow]Warning: failed to stop PID {pid}: {e}[/yellow]")

        time.sleep(1)
        is_running, pid, mode = detect_worker_process()
        if not is_running:
            console.print("[green]✓ Worker stopped[/green]")
            return 0

        console.print(f"[yellow]Worker still running (PID: {pid}); sending SIGKILL...[/yellow]")
        for pid, _ in all_processes:
            try:
                os.kill(pid, 9)
            except Exception:
                pass

        time.sleep(1)
        is_running, pid, mode = detect_worker_process()
        if not is_running:
            console.print("[green]✓ Worker stopped (SIGKILL)[/green]")
            return 0

        console.print("[red]✗ Failed to stop worker[/red]")
        return 1

    except Exception as e:
        logger.error(f"Error in worker stop: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1


def worker_fresh_command() -> int:
    stop_code = worker_stop_command()
    start_code = worker_start_command()
    return 0 if stop_code == 0 and start_code == 0 else 1
