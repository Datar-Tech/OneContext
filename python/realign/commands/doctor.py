"""Aline doctor command - Repair common issues after updates."""

from __future__ import annotations

import contextlib
import io
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich.console import Console
from rich.table import Table

from ..config import ReAlignConfig, get_default_config_content
from ..install_state import (
    INSTALL_STATE_PATH,
    detect_install_owner_from_environment,
    read_install_state,
    write_install_state,
)

console = Console()


def _is_log_artifact(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".log") or ".log." in name or name.endswith(".jsonl")


def _cleanup_non_dashboard_logs(*, verbose: bool) -> tuple[int, int]:
    """Remove log artifacts except dashboard.log."""
    from ..logging_config import get_log_directory, is_primary_log_file

    log_dir = get_log_directory()
    if not log_dir.exists():
        return 0, 0

    removed = 0
    kept = 0

    for entry in log_dir.iterdir():
        try:
            if entry.is_dir() and entry.name == "dashboard_diagnostics":
                shutil.rmtree(entry)
                removed += 1
                if verbose:
                    console.print(f"  [dim]Removed legacy diagnostics dir: {entry}[/dim]")
                continue

            if not entry.is_file():
                continue
            if not _is_log_artifact(entry):
                continue

            if is_primary_log_file(entry):
                kept += 1
                continue

            entry.unlink(missing_ok=True)
            removed += 1
            if verbose:
                console.print(f"  [dim]Removed log: {entry}[/dim]")
        except Exception as e:
            if verbose:
                console.print(f"  [yellow]Failed to remove {entry}: {e}[/yellow]")

    return removed, kept


def _clear_python_cache(root: Path, *, verbose: bool) -> Tuple[int, int]:
    pyc_count = 0
    pycache_count = 0

    for pyc_file in root.rglob("*.pyc"):
        try:
            pyc_file.unlink()
            pyc_count += 1
            if verbose:
                console.print(f"  [dim]Removed: {pyc_file}[/dim]")
        except Exception as e:
            if verbose:
                console.print(f"  [yellow]Failed to remove {pyc_file}: {e}[/yellow]")

    for pycache_dir in root.rglob("__pycache__"):
        if not pycache_dir.is_dir():
            continue
        try:
            shutil.rmtree(pycache_dir)
            pycache_count += 1
            if verbose:
                console.print(f"  [dim]Removed: {pycache_dir}[/dim]")
        except Exception as e:
            if verbose:
                console.print(f"  [yellow]Failed to remove {pycache_dir}: {e}[/yellow]")

    return pyc_count, pycache_count


def _ensure_global_config(*, force: bool, verbose: bool) -> Path:
    config_path = Path.home() / ".aline" / "config.yaml"

    if force or not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(get_default_config_content(), encoding="utf-8")
        if verbose:
            console.print(f"  [dim]Wrote config: {config_path}[/dim]")

    return config_path


def _ensure_database_initialized(config: ReAlignConfig, *, verbose: bool) -> Path:
    db_path = Path(config.sqlite_db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    from ..db.sqlite_db import SQLiteDatabase

    db = SQLiteDatabase(str(db_path))
    ok = db.initialize()
    db.close()
    if verbose:
        console.print(f"  [dim]Database init: {'ok' if ok else 'failed'}[/dim]")

    return db_path


def _update_claude_hooks(*, verbose: bool) -> Tuple[list[str], list[str]]:
    hooks_updated: list[str] = []
    hooks_failed: list[str] = []

    # Stop hook
    try:
        from ..claude_hooks.stop_hook_installer import install_stop_hook, get_settings_path

        if install_stop_hook(get_settings_path(), quiet=True, force=True):
            hooks_updated.append("Stop")
            if verbose:
                console.print("  [dim]Stop hook updated[/dim]")
        else:
            hooks_failed.append("Stop")
    except Exception as e:
        hooks_failed.append("Stop")
        if verbose:
            console.print(f"  [yellow]Stop hook failed: {e}[/yellow]")

    # UserPromptSubmit hook
    try:
        from ..claude_hooks.user_prompt_submit_hook_installer import (
            install_user_prompt_submit_hook,
            get_settings_path as get_submit_settings_path,
        )

        if install_user_prompt_submit_hook(get_submit_settings_path(), quiet=True, force=True):
            hooks_updated.append("UserPromptSubmit")
            if verbose:
                console.print("  [dim]UserPromptSubmit hook updated[/dim]")
        else:
            hooks_failed.append("UserPromptSubmit")
    except Exception as e:
        hooks_failed.append("UserPromptSubmit")
        if verbose:
            console.print(f"  [yellow]UserPromptSubmit hook failed: {e}[/yellow]")

    # PermissionRequest hook
    try:
        from ..claude_hooks.permission_request_hook_installer import (
            install_permission_request_hook,
            get_settings_path as get_permission_settings_path,
        )

        if install_permission_request_hook(get_permission_settings_path(), quiet=True, force=True):
            hooks_updated.append("PermissionRequest")
            if verbose:
                console.print("  [dim]PermissionRequest hook updated[/dim]")
        else:
            hooks_failed.append("PermissionRequest")
    except Exception as e:
        hooks_failed.append("PermissionRequest")
        if verbose:
            console.print(f"  [yellow]PermissionRequest hook failed: {e}[/yellow]")

    return hooks_updated, hooks_failed


def _update_codex_notify_hook(*, verbose: bool) -> Tuple[int, int]:
    """
    Ensure Codex notify hook is installed:
    - global ~/.codex
    - all Aline-managed ~/.aline/codex_homes/*
    """
    try:
        from ..codex_hooks.notify_hook_installer import (
            ensure_all_aline_codex_homes_notify_hook_installed,
            ensure_global_codex_notify_hook_installed,
        )

        ok_global = 1 if ensure_global_codex_notify_hook_installed(quiet=not verbose) else 0
        ok_homes = int(ensure_all_aline_codex_homes_notify_hook_installed(quiet=not verbose))
        return ok_global, ok_homes
    except Exception as e:
        if verbose:
            console.print(f"  [yellow]Codex notify hook update failed: {e}[/yellow]")
        return 0, 0


def _remove_deprecated_skills(*, verbose: bool) -> list[str]:
    """Remove deprecated skills that are no longer shipped by default."""
    deprecated = ("aline-share", "aline-import-history-sessions")
    removed: list[str] = []

    aline_skill_root = Path.home() / ".aline" / "skills"
    targets = [
        aline_skill_root,
        Path.home() / ".claude" / "skills",
        Path.home() / ".codex" / "skills",
        Path.home() / ".config" / "opencode" / "skill",
    ]

    for skill_name in deprecated:
        for root in targets:
            skill_dir = root / skill_name
            if skill_dir.is_dir() or skill_dir.is_symlink():
                try:
                    if skill_dir.is_symlink():
                        skill_dir.unlink()
                    else:
                        shutil.rmtree(skill_dir)
                    removed.append(f"{root.parent.name}/{skill_name}")
                    if verbose:
                        console.print(f"  [dim]Removed deprecated: {skill_dir}[/dim]")
                except Exception as e:
                    if verbose:
                        console.print(f"  [yellow]Failed to remove {skill_dir}: {e}[/yellow]")

    return removed


def _update_instruction_blocks(*, verbose: bool) -> int:
    """Inject/update proactive OneContext instruction blocks in global config files."""
    from .add import inject_onecontext_instructions

    results = inject_onecontext_instructions(force=True)
    updated = sum(1 for v in results.values() if v)
    for name, written in results.items():
        if written:
            if verbose:
                console.print(f"  [dim]Updated instruction block in {name} config[/dim]")
    return updated


def _update_skills(*, verbose: bool) -> int:
    from .add import add_skills_command

    # Clean up deprecated skills first
    removed = _remove_deprecated_skills(verbose=verbose)
    if removed:
        console.print(f"  [green]✓[/green] Removed deprecated skill(s): {', '.join(removed)}")

    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture):
        add_skills_command(force=True, include_extras=False)

    output = stdout_capture.getvalue()
    updated_count = output.count("✓")
    if verbose and output.strip():
        for line in output.strip().split("\n"):
            console.print(f"  [dim]{line}[/dim]")
    return updated_count


def _manual_upgrade_command_for_owner(owner: str) -> str:
    owner_key = (owner or "").strip().lower()
    if owner_key == "uv":
        return "uv tool upgrade aline-ai"
    if owner_key == "pipx":
        return "pipx upgrade aline-ai"
    return "python -m pip install --upgrade aline-ai"


def _repair_upgrade_routing_metadata(*, verbose: bool) -> tuple[Optional[str], Path]:
    """Ensure install owner metadata exists for stable auto-upgrade routing."""
    state = read_install_state() or {}
    detected = detect_install_owner_from_environment()
    owner = detected or str(state.get("owner", "")).strip().lower() or None

    if not owner:
        return None, INSTALL_STATE_PATH

    source = "doctor_fix_upgrade_detected" if detected else "doctor_fix_upgrade_existing"
    write_install_state(owner, source=source)
    if verbose:
        console.print(f"  [dim]Upgrade owner: {owner} ({source})[/dim]")
    return owner, INSTALL_STATE_PATH


def _check_failed_jobs(
    config: ReAlignConfig,
    *,
    verbose: bool,
    fix: bool,
) -> Tuple[int, int]:
    """
    Check for failed turn_summary and session_summary jobs.

    Returns:
        (failed_count, requeued_count)
    """
    from ..db.sqlite_db import SQLiteDatabase

    db_path = Path(config.sqlite_db_path).expanduser()
    if not db_path.exists():
        return 0, 0

    db = SQLiteDatabase(str(db_path))

    try:
        # Count failed jobs in queue
        failed_turn_count = db.count_jobs(kinds=["turn_summary"], statuses=["failed"])
        failed_session_count = db.count_jobs(kinds=["session_summary"], statuses=["failed"])
        total_failed = failed_turn_count + failed_session_count

        if verbose and total_failed > 0:
            # Show details of failed jobs
            failed_jobs = db.list_jobs(statuses=["failed"], limit=20)
            if failed_jobs:
                table = Table(title="Failed Jobs in Queue", show_header=True, header_style="bold")
                table.add_column("Kind", style="cyan")
                table.add_column("Session", style="dim")
                table.add_column("Turn", style="dim")
                table.add_column("Error", style="red", max_width=40)
                table.add_column("Attempts", justify="right")

                for job in failed_jobs:
                    payload = job.get("payload", {})
                    session_id = str(payload.get("session_id", ""))[:8]
                    turn_num = str(payload.get("turn_number", "-"))
                    error = (job.get("last_error") or "")[:40]
                    attempts = str(job.get("attempts", 0))
                    table.add_row(job["kind"], session_id, turn_num, error, attempts)

                console.print(table)

        requeued = 0
        if fix and total_failed > 0:
            # Requeue failed jobs
            requeued, _ = db.requeue_failed_jobs(kinds=["turn_summary", "session_summary"])

        return total_failed, requeued

    finally:
        db.close()


def _check_llm_error_turns(
    config: ReAlignConfig,
    *,
    verbose: bool,
    fix: bool,
) -> Tuple[int, int]:
    """
    Check for turns with LLM API errors (llm_title contains 'LLM API Error').

    Returns:
        (error_count, requeued_count)
    """
    from ..db.sqlite_db import SQLiteDatabase

    db_path = Path(config.sqlite_db_path).expanduser()
    if not db_path.exists():
        return 0, 0

    db = SQLiteDatabase(str(db_path))

    try:
        conn = db._get_connection()

        # Find turns with LLM API Error marker (exact prefix match)
        rows = conn.execute(
            """
            SELECT t.session_id, t.turn_number, t.llm_title, s.session_file_path, s.workspace_path
            FROM turns t
            JOIN sessions s ON t.session_id = s.id
            WHERE t.llm_title LIKE '⚠ LLM API Error%'
            ORDER BY t.timestamp DESC
            """
        ).fetchall()

        if not rows:
            return 0, 0

        if verbose:
            table = Table(title="Turns with LLM API Error", show_header=True, header_style="bold")
            table.add_column("Session", style="dim")
            table.add_column("Turn", justify="right")
            table.add_column("Title", style="yellow", max_width=50)

            for row in rows[:20]:  # Show max 20
                session_id = str(row["session_id"])[:12]
                turn_num = str(row["turn_number"])
                title = (row["llm_title"] or "")[:50]
                table.add_row(session_id, turn_num, title)

            if len(rows) > 20:
                table.add_row("...", "...", f"({len(rows) - 20} more)")

            console.print(table)

        if not fix:
            return len(rows), 0

        # Requeue turn_summary jobs for these turns
        requeued = 0
        skipped = 0
        for row in rows:
            session_file_path_str = row["session_file_path"] or ""

            # Skip invalid session file paths
            if not session_file_path_str or session_file_path_str in (".", ".."):
                if verbose:
                    console.print(
                        f"  [dim]Skip: invalid session path for {row['session_id'][:8]} #{row['turn_number']}[/dim]"
                    )
                skipped += 1
                continue

            session_file_path = Path(session_file_path_str)
            workspace_path = Path(row["workspace_path"]) if row["workspace_path"] else None

            if not session_file_path.exists():
                if verbose:
                    console.print(f"  [dim]Skip: session file not found: {session_file_path}[/dim]")
                skipped += 1
                continue

            try:
                db.enqueue_turn_summary_job(
                    session_file_path=session_file_path,
                    workspace_path=workspace_path,
                    turn_number=row["turn_number"],
                    skip_dedup=True,  # Force regeneration
                )
                requeued += 1
            except Exception as e:
                if verbose:
                    console.print(
                        f"  [yellow]Failed to enqueue {row['session_id'][:8]} #{row['turn_number']}: {e}[/yellow]"
                    )
                skipped += 1

        return len(rows), requeued

    finally:
        db.close()


def _check_watcher_backlog(
    config: ReAlignConfig,
    *,
    verbose: bool,
    fix: bool,
) -> Tuple[int, int]:
    """
    Check for backlog sessions that changed since the watcher last run.

    Uses the same 2-phase startup scan logic as the watcher:
    1) stat previously persisted session paths (fast)
    2) full scan of watch paths (complete)
    """
    from ..db.sqlite_db import SQLiteDatabase
    from ..watcher_core import DialogueWatcher

    db_path = Path(config.sqlite_db_path).expanduser()
    if not db_path.exists():
        return 0, 0

    watcher = DialogueWatcher()
    candidates, _sizes, _mtimes, report = watcher._startup_scan_collect_candidates()

    if verbose:
        console.print(
            f"  [dim]Tracked paths: {report.prev_paths} (missing: {report.prev_missing}, changed: {report.prev_changed})[/dim]"
        )
        console.print(
            f"  [dim]Scan paths: {report.scan_paths} (new: {report.scan_new}, changed: {report.scan_changed})[/dim]"
        )

    if not candidates:
        return 0, 0

    if not fix:
        return len(candidates), 0

    db = SQLiteDatabase(str(db_path))
    try:
        enqueued = 0
        for session_file in candidates:
            try:
                db.enqueue_session_process_job(
                    session_file_path=session_file,
                    session_id=session_file.stem,
                    workspace_path=None,
                    session_type=watcher._detect_session_type(session_file),
                    source_event="doctor_backlog_scan",
                    priority=getattr(watcher, "_startup_scan_priority", 5),
                )
                enqueued += 1
            except Exception as e:
                if verbose:
                    console.print(f"  [yellow]Failed to enqueue {session_file}: {e}[/yellow]")
                continue
        return len(candidates), enqueued
    finally:
        db.close()


def run_doctor(
    *,
    restart_daemons: bool,
    start_if_not_running: bool,
    verbose: bool,
    clear_cache: bool,
    auto_fix: bool = False,
    skip_ensure_env: bool = False,
    fix_upgrade: bool = False,
) -> int:
    """
    Core doctor logic.

    Args:
        restart_daemons: Restart/ensure daemons at the end.
        start_if_not_running: If True and restart_daemons is True, start daemons even if not running.
        verbose: Print details.
        clear_cache: Clear Python bytecode cache for the installed package directory.
        auto_fix: If True, automatically fix failed jobs without prompting.
        skip_ensure_env: If True, skip steps 2-4b (config/db/tmux/hooks/skills/instructions)
            because the caller (e.g. init) already performed them.
        fix_upgrade: If True, repair install owner metadata and print upgrade command.
    """
    from ..auth import is_logged_in
    from . import watcher as watcher_cmd
    from . import worker as worker_cmd
    from . import init as init_cmd

    console.print("\n[bold blue]═══ Aline Doctor ═══[/bold blue]\n")

    watcher_running, _watcher_pid, _watcher_mode = watcher_cmd.detect_watcher_process()
    worker_running, _worker_pid, _worker_mode = worker_cmd.detect_worker_process()

    can_start_daemons = is_logged_in()
    if restart_daemons and not can_start_daemons:
        console.print("[yellow]![/yellow] Not logged in; skipping daemon restart/start.")
        console.print("[dim]Run 'aline login' then re-run 'aline doctor'.[/dim]")
        restart_daemons = False

    if fix_upgrade:
        console.print("[bold]0. Repairing upgrade routing...[/bold]")
        try:
            owner, state_path = _repair_upgrade_routing_metadata(verbose=verbose)
            if owner:
                console.print(f"  [green]✓[/green] Upgrade manager: {owner}")
                console.print(f"  [green]✓[/green] State file: {state_path}")
                console.print("  [green]✓[/green] Recommended command: onecontext update")
                console.print(
                    f"  [dim]Manual fallback: {_manual_upgrade_command_for_owner(owner)}[/dim]"
                )
            else:
                console.print(
                    "  [yellow]![/yellow] Could not detect install manager automatically."
                )
                console.print(
                    "  [dim]Install owner will be re-detected on next successful upgrade.[/dim]"
                )
                console.print("  [dim]Try: onecontext update[/dim]")
        except Exception as e:
            console.print(f"  [yellow]![/yellow] Failed to repair upgrade routing: {e}")

    # Stop daemons early to avoid DB lock during migrations.
    if restart_daemons:
        if watcher_running:
            if verbose:
                console.print("[dim]Stopping watcher...[/dim]")
            try:
                watcher_cmd.watcher_stop_command()
            except Exception:
                pass
        if worker_running:
            if verbose:
                console.print("[dim]Stopping worker...[/dim]")
            try:
                worker_cmd.worker_stop_command()
            except Exception:
                pass

    # 1. Clear Python cache (package scope)
    if clear_cache:
        console.print("[bold]1. Clearing Python cache...[/bold]")
        package_root = Path(__file__).resolve().parents[1]
        pyc_count, pycache_count = _clear_python_cache(package_root, verbose=verbose)
        console.print(
            f"  [green]✓[/green] Cleared {pyc_count} .pyc files, {pycache_count} __pycache__ directories"
        )

    # Load config (needed by later steps even when skipping ensure-env).
    try:
        config_path = (
            _ensure_global_config(force=False, verbose=verbose)
            if not skip_ensure_env
            else (Path.home() / ".aline" / "config.yaml")
        )
        config = ReAlignConfig.load(config_path)
    except Exception as e:
        console.print(f"  [red]✗[/red] Failed to load config: {e}")
        return 1

    if not skip_ensure_env:
        # 2. Ensure global environment (config/db/tmux)
        console.print("\n[bold]2. Ensuring global environment...[/bold]")
        try:
            # Ensure early session title is enabled (default-on since rename)
            if not config.enable_early_session_title:
                config.enable_early_session_title = True
                config.save()
                if verbose:
                    console.print("  [dim]Enabled early_session_title[/dim]")

            db_path = _ensure_database_initialized(config, verbose=verbose)

            # tmux auto-updates Aline-managed config.
            tmux_conf = init_cmd._initialize_tmux_config()

            console.print(f"  [green]✓[/green] Config: {config_path}")
            console.print(f"  [green]✓[/green] Database: {db_path}")
            console.print(f"  [green]✓[/green] Tmux: {tmux_conf}")
        except Exception as e:
            console.print(f"  [red]✗[/red] Failed to ensure global environment: {e}")
            return 1

        # 3. Update Claude Code hooks
        console.print("\n[bold]3. Updating Claude Code hooks...[/bold]")
        hooks_updated, hooks_failed = _update_claude_hooks(verbose=verbose)
        if hooks_updated:
            console.print(f"  [green]✓[/green] Updated hooks: {', '.join(hooks_updated)}")
        if hooks_failed:
            console.print(f"  [yellow]![/yellow] Failed hooks: {', '.join(hooks_failed)}")

        # 3b. Update Codex notify hook
        console.print("\n[bold]3b. Updating Codex notify hook...[/bold]")
        try:
            ok_global, ok_homes = _update_codex_notify_hook(verbose=verbose)
            if ok_global:
                console.print("  [green]✓[/green] Updated global Codex config (~/.codex)")
            else:
                console.print(
                    "  [dim]Global Codex config not updated (may be missing or unwritable)[/dim]"
                )
            if ok_homes:
                console.print(f"  [green]✓[/green] Updated {ok_homes} Aline CODEX_HOME(s)")
            else:
                console.print("  [dim]No Aline CODEX_HOME(s) updated[/dim]")

            # If Codex exists but is legacy, warn: Aline expects the Rust notify hook.
            try:
                from ..codex_hooks.notify_hook_installer import codex_cli_supports_notify_hook

                supported = codex_cli_supports_notify_hook()
                if supported is False:
                    console.print("  [yellow]![/yellow] Codex CLI does not support notify hook.")
                    console.print(
                        "  [dim]Tip: update to the Rust Codex CLI to enable reliable, event-driven Codex imports.[/dim]"
                    )
            except Exception:
                pass
        except Exception as e:
            console.print(f"  [yellow]![/yellow] Codex notify hook update failed: {e}")

        # 4. Update skills
        console.print("\n[bold]4. Updating skills...[/bold]")
        try:
            updated_count = _update_skills(verbose=verbose)
            if updated_count > 0:
                console.print(f"  [green]✓[/green] Updated {updated_count} skill(s)")
            else:
                console.print("  [green]✓[/green] Skills are up to date")
        except Exception as e:
            console.print(f"  [yellow]![/yellow] Failed to update skills: {e}")

        # 4b. Update proactive instruction blocks
        console.print("\n[bold]4b. Updating instruction blocks...[/bold]")
        try:
            instr_updated = _update_instruction_blocks(verbose=verbose)
            if instr_updated > 0:
                console.print(f"  [green]✓[/green] Updated {instr_updated} instruction block(s)")
            else:
                console.print("  [green]✓[/green] Instruction blocks are up to date")
        except Exception as e:
            console.print(f"  [yellow]![/yellow] Failed to update instruction blocks: {e}")

    # 4c. Remove non-dashboard logs in ~/.aline/.logs
    console.print("\n[bold]4c. Cleaning legacy logs...[/bold]")
    try:
        removed_logs, kept_logs = _cleanup_non_dashboard_logs(verbose=verbose)
        if removed_logs > 0:
            console.print(
                f"  [green]✓[/green] Removed {removed_logs} non-dashboard log artifact(s)"
            )
        else:
            console.print("  [green]✓[/green] No non-dashboard logs found")
        if verbose:
            console.print(f"  [dim]Retained dashboard log files: {kept_logs}[/dim]")
    except Exception as e:
        console.print(f"  [yellow]![/yellow] Failed to clean logs: {e}")

    # 5. Check/fix failed jobs and LLM error turns
    console.print("\n[bold]5. Checking failed summary jobs...[/bold]")
    try:
        # First pass: check counts without fixing
        failed_count, _ = _check_failed_jobs(config, verbose=verbose, fix=False)
        llm_error_count, _ = _check_llm_error_turns(config, verbose=verbose, fix=False)

        total_issues = failed_count + llm_error_count

        if total_issues == 0:
            console.print("  [green]✓[/green] No failed jobs or LLM errors found")
        else:
            # Show what was found
            if failed_count > 0:
                console.print(f"  [yellow]![/yellow] Found {failed_count} failed job(s) in queue")
            if llm_error_count > 0:
                console.print(
                    f"  [yellow]![/yellow] Found {llm_error_count} turn(s) with LLM API errors"
                )

            # Ask user if they want to fix
            if auto_fix or typer.confirm(
                "\n  Do you want to requeue these for regeneration?", default=True
            ):
                requeued_jobs = 0
                requeued_turns = 0

                if failed_count > 0:
                    _, requeued_jobs = _check_failed_jobs(config, verbose=verbose, fix=True)
                if llm_error_count > 0:
                    _, requeued_turns = _check_llm_error_turns(config, verbose=verbose, fix=True)

                total_requeued = requeued_jobs + requeued_turns
                console.print(
                    f"  [green]✓[/green] Requeued {total_requeued} item(s) for regeneration"
                )
            else:
                console.print("  [dim]Skipped fixing failed jobs[/dim]")
    except Exception as e:
        console.print(f"  [yellow]![/yellow] Failed to check jobs: {e}")

    # 5b. Repair agent/session associations
    console.print("\n[bold]5b. Repairing agent/session associations...[/bold]")
    try:
        repaired = _repair_agent_session_links(verbose=verbose)
        if repaired > 0:
            console.print(f"  [green]✓[/green] Repaired {repaired} session(s)")
        else:
            console.print("  [green]✓[/green] No missing associations found")
    except Exception as e:
        console.print(f"  [yellow]![/yellow] Failed to repair associations: {e}")

    # 5c. Check backlog sessions (watcher startup scan semantics)
    console.print("\n[bold]5c. Checking watcher backlog sessions...[/bold]")
    try:
        backlog_count, _ = _check_watcher_backlog(config, verbose=verbose, fix=False)
        if backlog_count == 0:
            console.print("  [green]✓[/green] No backlog sessions detected")
        else:
            console.print(
                f"  [yellow]![/yellow] Found {backlog_count} backlog session(s) to process"
            )
            if auto_fix or typer.confirm(
                "\n  Do you want to enqueue these for processing now?", default=True
            ):
                _, enqueued = _check_watcher_backlog(config, verbose=verbose, fix=True)
                console.print(f"  [green]✓[/green] Enqueued {enqueued} session_process job(s)")
            else:
                console.print("  [dim]Skipped enqueueing backlog sessions[/dim]")
    except Exception as e:
        console.print(f"  [yellow]![/yellow] Failed to check watcher backlog: {e}")

    # 6. Restart/ensure daemons
    if restart_daemons:
        console.print("\n[bold]6. Checking daemons...[/bold]")

        should_start_watcher = watcher_running or start_if_not_running
        should_start_worker = worker_running or start_if_not_running

        if should_start_watcher:
            try:
                exit_code = watcher_cmd.watcher_start_command()
                if exit_code == 0:
                    console.print("  [green]✓[/green] Watcher is running")
                else:
                    console.print("  [yellow]![/yellow] Failed to start watcher")
            except Exception as e:
                console.print(f"  [yellow]![/yellow] Failed to start watcher: {e}")
        else:
            console.print("  [dim]Watcher was not running; leaving it stopped.[/dim]")

        if should_start_worker:
            try:
                exit_code = worker_cmd.worker_start_command()
                if exit_code == 0:
                    console.print("  [green]✓[/green] Worker is running")
                else:
                    console.print("  [yellow]![/yellow] Failed to start worker")
            except Exception as e:
                console.print(f"  [yellow]![/yellow] Failed to start worker: {e}")
        else:
            console.print("  [dim]Worker was not running; leaving it stopped.[/dim]")
    else:
        console.print("\n[dim]Skipping daemon restart (--no-restart)[/dim]")

    console.print("\n[green]Done![/green] Aline is ready with the latest code.")
    return 0


def _repair_agent_session_links(*, verbose: bool = False) -> int:
    """Repair agent-session associations using windowlink and agents mappings."""
    def _agent_id_from_source(raw_source: object) -> str:
        source = str(raw_source or "").strip()
        if not source.startswith("agent:"):
            return ""
        return source[6:].strip()

    def _is_valid_agent_info_id(db_obj: object, candidate: object) -> bool:
        agent_id = str(candidate or "").strip()
        if not agent_id:
            return False
        try:
            info = db_obj.get_agent_info(agent_id)  # type: ignore[attr-defined]
            return info is not None
        except Exception:
            try:
                conn = db_obj._get_connection()  # type: ignore[attr-defined]
                row = conn.execute(
                    "SELECT 1 FROM agent_info WHERE id = ? LIMIT 1",
                    (agent_id,),
                ).fetchone()
                return bool(row)
            except Exception:
                return False

    try:
        from ..db import get_database

        db = get_database(read_only=False)
    except Exception:
        return 0

    repaired = 0

    def _is_already_linked(db_obj: object, agent_id: str, session_id: str) -> bool:
        try:
            return bool(
                db_obj.is_session_linked_to_agent(agent_id, session_id)  # type: ignore[attr-defined]
            )
        except Exception:
            try:
                session = db_obj.get_session_by_id(session_id)  # type: ignore[attr-defined]
                return bool(session and getattr(session, "agent_id", None) == agent_id)
            except Exception:
                return False

    def _link_session(db_obj: object, agent_id: str, session_id: str) -> bool:
        try:
            return bool(
                db_obj.link_session_to_agent(agent_id, session_id)  # type: ignore[attr-defined]
            )
        except Exception:
            try:
                db_obj.update_session_agent_id(session_id, agent_id)  # type: ignore[attr-defined]
                return True
            except Exception:
                return False

    try:
        conn = db._get_connection()  # type: ignore[attr-defined]
        # Canonical source uses "agent:<agent_info_id>". If agent_id is missing or
        # mismatched (including polluted terminal_id writes), heal it in-place.
        conn.execute(
            """
            UPDATE windowlink
            SET agent_id = substr(source, 7)
            WHERE source LIKE 'agent:%'
              AND length(substr(source, 7)) > 0
              AND EXISTS (
                  SELECT 1 FROM agent_info ai WHERE ai.id = substr(source, 7)
              )
              AND (agent_id IS NULL OR agent_id = '' OR agent_id != substr(source, 7))
            """
        )
        conn.commit()
    except Exception:
        pass

    # 1) Use windowlink latest records
    try:
        links = db.list_latest_window_links(limit=5000)
    except Exception:
        links = []

    for link in links:
        session_id = (link.session_id or "").strip()
        agent_id = (link.agent_id or "").strip()
        if not _is_valid_agent_info_id(db, agent_id):
            agent_id = _agent_id_from_source(link.source)
            if not _is_valid_agent_info_id(db, agent_id):
                continue
        if not session_id or not agent_id:
            continue
        try:
            session = db.get_session_by_id(session_id)
            if not session:
                continue
        except Exception:
            continue
        if _is_already_linked(db, agent_id, session_id):
            continue
        if _link_session(db, agent_id, session_id):
            repaired += 1

    # 2) Fallback: agents table mapping (session_id -> agent source)
    try:
        agents = db.list_agents(status=None, limit=5000)
    except Exception:
        agents = []
    for agent in agents:
        session_id = (agent.session_id or "").strip()
        source = (agent.source or "").strip()
        if not session_id or not source.startswith("agent:"):
            continue
        agent_id = source[6:]
        if not _is_valid_agent_info_id(db, agent_id):
            continue
        try:
            session = db.get_session_by_id(session_id)
            if not session:
                continue
        except Exception:
            continue
        if _is_already_linked(db, agent_id, session_id):
            continue
        if _link_session(db, agent_id, session_id):
            repaired += 1

    if verbose and repaired:
        console.print(f"  [dim]Repaired session links: {repaired}[/dim]")

    return repaired


def doctor_command(
    no_restart: bool = typer.Option(
        False, "--no-restart", help="Only repair files, don't restart daemons"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    fix_upgrade: bool = typer.Option(
        False,
        "--fix-upgrade",
        help="Repair upgrade routing metadata for the unified 'onecontext update' flow",
    ),
):
    """
    Fix common issues after code updates.

    This command:
    - Clears Python bytecode cache for the installed Aline package
    - Ensures global config/DB/tmux are present and up to date
    - Updates Claude Code hooks (Stop, UserPromptSubmit, PermissionRequest)
    - Updates Claude Code skills to the latest version
    - Cleans non-dashboard logs in ~/.aline/.logs (keeps dashboard.log only)
    - Checks for failed summary jobs (prompts to fix if found)
    - Restarts watcher/worker (default) so long-running processes use the latest code
    """
    exit_code = run_doctor(
        restart_daemons=not no_restart,
        start_if_not_running=True,
        verbose=verbose,
        clear_cache=True,
        fix_upgrade=fix_upgrade,
    )
    raise typer.Exit(code=exit_code)
