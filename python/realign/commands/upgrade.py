"""ReAlign upgrade command - Upgrade database schema to latest version."""

from pathlib import Path
import os
import subprocess
import sys
import typer
from rich.console import Console
from rich.table import Table
from typing import Optional, Tuple

from ..config import ReAlignConfig
from ..db.schema import SCHEMA_VERSION, get_migration_scripts
from ..install_state import resolve_install_owner, write_install_state

console = Console()


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _stop_daemons_best_effort_for_package_upgrade() -> None:
    """Stop watcher/worker daemons (best-effort) before upgrading the package.

    Upgrading the installed package while old daemons are running can leave the system
    in a mixed-version state (old processes continue running old code).
    """
    try:
        from . import watcher as watcher_cmd

        try:
            watcher_running = bool(watcher_cmd._is_watcher_running())
        except Exception:
            watcher_running = False
        if watcher_running:
            console.print("[dim]Stopping watcher for upgrade...[/dim]")
            try:
                watcher_cmd.watcher_stop_command()
            except Exception:
                pass
    except Exception:
        pass

    try:
        from . import worker as worker_cmd

        try:
            worker_running, _pid, _mode = worker_cmd.detect_worker_process()
        except Exception:
            worker_running = False
        if worker_running:
            console.print("[dim]Stopping worker for upgrade...[/dim]")
            try:
                worker_cmd.worker_stop_command()
            except Exception:
                pass
    except Exception:
        pass


def _detect_running_tmux_dashboard() -> tuple[bool, str, str]:
    """Detect an Aline-managed tmux dashboard session (best-effort)."""
    try:
        from ..dashboard.tmux_manager import OUTER_SESSION, OUTER_SOCKET  # type: ignore

        outer_session = str(OUTER_SESSION)
        outer_socket = str(OUTER_SOCKET)
    except Exception:
        # Keep defaults stable for older versions / import failures.
        outer_session = "aline"
        outer_socket = "aline_dash"

    try:
        proc = subprocess.run(
            ["tmux", "-L", outer_socket, "has-session", "-t", outer_session],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        return proc.returncode == 0, outer_socket, outer_session
    except Exception:
        return False, outer_socket, outer_session


def _stop_running_tmux_dashboard_best_effort(socket_name: str, session_name: str) -> bool:
    """Stop a running dashboard tmux session so new launches use upgraded code."""
    try:
        proc = subprocess.run(
            ["tmux", "-L", socket_name, "kill-session", "-t", session_name],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _run_doctor_after_upgrade() -> None:
    if _env_truthy("ALINE_SKIP_DOCTOR_AFTER_UPGRADE"):
        console.print("[dim]Skipping 'aline doctor' (ALINE_SKIP_DOCTOR_AFTER_UPGRADE=1).[/dim]\n")
        return

    console.print("\n[dim]Running 'aline doctor' to finalize the update...[/dim]\n")
    try:
        result = subprocess.run(["aline", "doctor"], capture_output=False)
        if result.returncode == 0:
            console.print("\n[green]✓ 'aline doctor' completed.[/green]\n")
        else:
            console.print(
                f"\n[yellow]⚠ 'aline doctor' exited with code {result.returncode}.[/yellow]\n"
            )
    except FileNotFoundError:
        console.print(
            "\n[yellow]⚠ Could not find 'aline' on PATH to run doctor. Please run: aline doctor[/yellow]\n"
        )
    except Exception as e:
        console.print(f"\n[yellow]⚠ Failed to run 'aline doctor': {e}[/yellow]\n")


def _get_installed_version_via_pipx() -> Optional[str]:
    """Return the version of aline-ai currently installed by pipx.

    Uses ``pipx list --json`` so we get a machine-readable answer that is
    independent of the importlib.metadata cache in the *current* process.
    """
    import json as _json

    try:
        proc = subprocess.run(
            ["pipx", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return None
        data = _json.loads(proc.stdout)
        venv = data.get("venvs", {}).get("aline-ai", {})
        return venv.get("metadata", {}).get("main_package", {}).get("package_version")
    except Exception:
        return None


def _get_installed_version_via_current_python() -> Optional[str]:
    """Return installed aline-ai version using the currently running Python interpreter."""
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from importlib.metadata import version; print(version('aline-ai'))",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode != 0:
            return None
        value = (proc.stdout or "").strip()
        return value or None
    except Exception:
        return None


def _upgrade_attempts(owner: Optional[str] = None) -> list[tuple[str, list[str]]]:
    """Build ordered upgrade attempts for common installation methods."""
    attempts = {
        "pipx": ("pipx", ["pipx", "upgrade", "aline-ai"]),
        "uv": ("uv", ["uv", "tool", "upgrade", "aline-ai"]),
        "pip": ("pip", [sys.executable, "-m", "pip", "install", "--upgrade", "aline-ai"]),
    }
    default_order = ["pipx", "uv", "pip"]
    if owner in attempts:
        order = [owner] + [name for name in default_order if name != owner]
    else:
        order = default_order
    return [attempts[name] for name in order]


def _perform_package_upgrade(current_version: str, latest_version: str) -> bool:
    """Upgrade package via best matching manager and verify resulting version."""
    install_owner = resolve_install_owner(persist_detected=True)
    if install_owner:
        console.print(f"[dim]Detected install manager: {install_owner}[/dim]")

    # Perform update using the first available package manager that succeeds.
    console.print("\n[bold]Updating aline-ai package...[/bold]\n")

    try:
        result = None
        manager_used = None
        attempted_managers: list[str] = []
        for manager, cmd in _upgrade_attempts(install_owner):
            attempted_managers.append(manager)
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=False,
                )
            except FileNotFoundError:
                continue
            if result.returncode == 0:
                manager_used = manager
                break

        if manager_used and result is not None and result.returncode == 0:
            # Verify the version actually changed.  pipx returns rc=0 even
            # when "already at latest version", so we cannot just trust the
            # exit code.  Query the *new* aline binary (the current process
            # still has the old importlib.metadata cache).
            if manager_used == "pipx":
                actual_new = (
                    _get_installed_version_via_pipx() or _get_installed_version_via_current_python()
                )
            else:
                actual_new = _get_installed_version_via_current_python()

            if actual_new and compare_versions(actual_new, latest_version) >= 0:
                try:
                    write_install_state(manager_used, source="upgrade_success")
                except Exception:
                    pass
                console.print(f"\n[bold green]✓ Successfully updated to {actual_new}![/bold green]")
                _run_doctor_after_upgrade()
                console.print("[dim]Please restart 'aline' to use the new version.[/dim]\n")
                return True
            else:
                # Upgrade command said OK but the installed version is still old.
                actual_label = actual_new or current_version
                console.print(
                    f"\n[yellow]⚠ Upgrade completed but version is still "
                    f"[cyan]{actual_label}[/cyan] (expected [green]{latest_version}[/green]).[/yellow]"
                )
                console.print(
                    "[dim]This can happen when PyPI mirrors haven't synced yet. "
                    "Launching normally...[/dim]\n"
                )
                console.print(
                    "[dim]Try this recovery flow:[/dim]\n"
                    "[dim]1) onecontext doctor --fix-upgrade[/dim]\n"
                    "[dim]2) onecontext update[/dim]\n"
                )
                return False
        else:
            console.print("\n[bold red]✗ Update failed.[/bold red]")
            if attempted_managers:
                tried = ", ".join(attempted_managers)
                console.print(f"[dim]Tried package managers: {tried}[/dim]")
            console.print(
                "[dim]Try this recovery flow:[/dim]\n"
                "[dim]1) onecontext doctor --fix-upgrade[/dim]\n"
                "[dim]2) onecontext update[/dim]\n"
            )
            return False

    except Exception as e:
        console.print(f"\n[bold red]✗ Update failed: {e}[/bold red]\n")
        console.print(
            "[dim]Try this recovery flow:[/dim]\n"
            "[dim]1) onecontext doctor --fix-upgrade[/dim]\n"
            "[dim]2) onecontext update[/dim]\n"
        )
        return False


def get_latest_pypi_version() -> Optional[str]:
    """Fetch the latest version of aline-ai from PyPI.

    Returns:
        The latest version string, or None if unable to fetch.
    """
    import urllib.request
    import json

    try:
        url = "https://pypi.org/pypi/aline-ai/json"
        with urllib.request.urlopen(url, timeout=3) as response:
            data = json.loads(response.read().decode())
            return data.get("info", {}).get("version")
    except Exception:
        return None


def compare_versions(current: str, latest: str) -> int:
    """Compare two version strings.

    Returns:
        -1 if current < latest (update available)
         0 if current == latest
         1 if current > latest
    """

    def parse_version(v: str) -> Tuple[int, ...]:
        """Parse version string to tuple of integers."""
        parts = []
        for part in v.split("."):
            # Handle pre-release versions like "0.5.5a1"
            num_part = ""
            for char in part:
                if char.isdigit():
                    num_part += char
                else:
                    break
            if num_part:
                parts.append(int(num_part))
            else:
                parts.append(0)
        return tuple(parts)

    current_tuple = parse_version(current)
    latest_tuple = parse_version(latest)

    # Pad shorter tuple with zeros
    max_len = max(len(current_tuple), len(latest_tuple))
    current_tuple = current_tuple + (0,) * (max_len - len(current_tuple))
    latest_tuple = latest_tuple + (0,) * (max_len - len(latest_tuple))

    if current_tuple < latest_tuple:
        return -1
    elif current_tuple > latest_tuple:
        return 1
    return 0


def check_and_prompt_update() -> bool:
    """Check for updates and prompt user to update if available.

    Returns:
        True if update was performed (should restart), False otherwise.
    """
    from importlib.metadata import version

    try:
        current_version = version("aline-ai")
    except Exception:
        return False

    latest_version = get_latest_pypi_version()
    if latest_version is None:
        return False

    if compare_versions(current_version, latest_version) >= 0:
        # Current version is up to date or newer
        return False

    # New version available - prompt user
    console.print(
        f"\n[bold yellow]⬆ Update available:[/bold yellow] "
        f"[cyan]{current_version}[/cyan] → [green]{latest_version}[/green]"
    )

    try:
        response = (
            console.input(
                "[dim]Do you want to run 'onecontext update' now? ([/dim][green]y[/green][dim]/[/dim][yellow]n[/yellow][dim]):[/dim] "
            )
            .strip()
            .lower()
        )
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Update skipped.[/dim]\n")
        return False

    if response not in ("y", "yes"):
        console.print("[dim]Update skipped.[/dim]\n")
        return False

    # Best practice: stop daemons and close dashboards before upgrading.
    # Otherwise users can end up with old processes still running old code.
    dash_running, dash_socket, dash_session = _detect_running_tmux_dashboard()
    if dash_running:
        console.print(
            "\n[bold yellow]⚠ Detected a running Aline dashboard tmux session.[/bold yellow]"
        )
        console.print(
            f"[dim]Aline will try to close it before upgrading to avoid mixed versions. "
            f"(Command: tmux -L {dash_socket} kill-session -t {dash_session})[/dim]\n"
        )
        try:
            proceed = (
                console.input(
                    "[dim]Continue upgrade and close dashboard now? ([/dim][green]y[/green][dim]/[/dim][yellow]n[/yellow][dim]):[/dim] "
                )
                .strip()
                .lower()
            )
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Update skipped.[/dim]\n")
            return False
        if proceed not in ("y", "yes"):
            console.print("[dim]Update skipped.[/dim]\n")
            return False

        console.print("[dim]Stopping running dashboard session before upgrade...[/dim]")
        if not _stop_running_tmux_dashboard_best_effort(dash_socket, dash_session):
            console.print(
                "[yellow]⚠ Could not stop dashboard tmux session automatically. "
                "Proceeding anyway may keep old code running until you restart it.[/yellow]"
            )

    _stop_daemons_best_effort_for_package_upgrade()

    return _perform_package_upgrade(current_version, latest_version)


def update_package_command() -> int:
    """Non-interactive self-update entrypoint for end users."""
    from importlib.metadata import version

    try:
        current_version = version("aline-ai")
    except Exception:
        console.print("[red]✗ Could not detect installed aline-ai version.[/red]")
        console.print("[dim]Try: onecontext doctor --fix-upgrade[/dim]")
        console.print("[dim]Then rerun: onecontext update[/dim]")
        return 1

    latest_version = get_latest_pypi_version()
    if latest_version is None:
        console.print("[red]✗ Could not fetch latest version from PyPI.[/red]")
        console.print("[dim]Please check network and try again.[/dim]")
        return 1

    if compare_versions(current_version, latest_version) >= 0:
        console.print(f"[green]✓ Already up to date ({current_version}).[/green]")
        return 0

    console.print(
        f"[bold yellow]⬆ Upgrading Aline:[/bold yellow] "
        f"[cyan]{current_version}[/cyan] → [green]{latest_version}[/green]"
    )

    dash_running, dash_socket, dash_session = _detect_running_tmux_dashboard()
    if dash_running:
        console.print("[dim]Detected running dashboard session; closing it before upgrade...[/dim]")
        if not _stop_running_tmux_dashboard_best_effort(dash_socket, dash_session):
            console.print(
                "[yellow]⚠ Could not stop dashboard automatically. "
                "Upgrade continues, but you may need to restart it manually.[/yellow]"
            )

    _stop_daemons_best_effort_for_package_upgrade()

    if _perform_package_upgrade(current_version, latest_version):
        return 0
    return 1


def get_current_db_version(db_path: Path) -> int:
    """Get current schema version from database without triggering migration."""
    import sqlite3

    if not db_path.exists():
        return -1  # Database doesn't exist

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(version) FROM schema_version")
        result = cursor.fetchone()
        conn.close()
        return result[0] if result and result[0] is not None else 0
    except sqlite3.OperationalError:
        return 0  # Table doesn't exist, fresh database


def upgrade_command(
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Show what would be done without making changes"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force upgrade even if versions match"),
    restart_watcher: bool = typer.Option(
        True, "--restart-watcher/--no-restart-watcher", help="Restart watcher after upgrade"
    ),
):
    """Upgrade aline database schema to the latest version.

    This command is useful after updating aline via pip install to ensure
    the database schema is compatible with the new code version.

    Examples:
        aline upgrade                    # Upgrade database
        aline upgrade --dry-run          # Preview changes
        aline upgrade --no-restart-watcher  # Don't restart watcher
    """
    console.print("\n[bold blue]═══ Aline Upgrade ═══[/bold blue]\n")

    # Load config
    try:
        config = ReAlignConfig.load()
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        console.print("[dim]Run 'aline init' first to initialize aline.[/dim]")
        raise typer.Exit(1)

    db_path = Path(config.sqlite_db_path).expanduser()

    # Check current version
    current_version = get_current_db_version(db_path)
    target_version = SCHEMA_VERSION

    # Display version info
    table = Table(title="Schema Version", show_header=False, box=None)
    table.add_column("Label", style="bold")
    table.add_column("Value", style="cyan")

    if current_version == -1:
        table.add_row("Database", "[red]Not found[/red]")
        table.add_row("Path", str(db_path))
        console.print(table)
        console.print("\n[yellow]Database does not exist. Run 'aline init' first.[/yellow]")
        raise typer.Exit(1)

    table.add_row("Current version", f"V{current_version}")
    table.add_row("Target version", f"V{target_version}")
    table.add_row("Database path", str(db_path))
    console.print(table)

    # Check if upgrade is needed
    if current_version >= target_version and not force:
        console.print("\n[green]✓ Database is already up to date![/green]")
        raise typer.Exit(0)

    if current_version == target_version and force:
        console.print(
            "\n[yellow]Versions match, but --force specified. Re-running migrations...[/yellow]"
        )

    # Get migration scripts
    migrations = get_migration_scripts(current_version, target_version)

    if not migrations:
        console.print("\n[green]✓ No migrations needed.[/green]")
        raise typer.Exit(0)

    # Display migrations
    console.print(f"\n[bold]Migrations to apply ({len(migrations)} scripts):[/bold]")
    for i, script in enumerate(migrations, 1):
        # Show truncated script preview
        preview = script.strip().replace("\n", " ")[:60]
        if len(script.strip()) > 60:
            preview += "..."
        console.print(f"  {i}. [dim]{preview}[/dim]")

    if dry_run:
        console.print("\n[yellow]Dry run - no changes made.[/yellow]")
        raise typer.Exit(0)

    # Confirm upgrade
    console.print("")

    # Check if watcher/worker are running (to restart later)
    watcher_was_running = False
    worker_was_running = False
    if restart_watcher:
        try:
            from . import watcher as watcher_cmd

            watcher_was_running = watcher_cmd._is_watcher_running()
            if watcher_was_running:
                console.print("[dim]Stopping watcher for upgrade...[/dim]")
                watcher_cmd.watcher_stop_command()
        except Exception:
            pass
        try:
            from . import worker as worker_cmd

            is_running, _pid, _mode = worker_cmd.detect_worker_process()
            worker_was_running = bool(is_running)
            if worker_was_running:
                console.print("[dim]Stopping worker for upgrade...[/dim]")
                worker_cmd.worker_stop_command()
        except Exception:
            pass

    # Perform upgrade
    console.print("[bold]Upgrading database...[/bold]")

    try:
        from ..db.sqlite_db import SQLiteDatabase

        db = SQLiteDatabase(str(db_path))
        db.initialize()  # This handles migrations automatically
        db.close()

        # Verify upgrade
        new_version = get_current_db_version(db_path)

        if new_version >= target_version:
            console.print(
                f"\n[bold green]✓ Upgrade successful![/bold green] V{current_version} → V{new_version}"
            )
        else:
            console.print(
                f"\n[yellow]⚠ Partial upgrade: V{current_version} → V{new_version} (target: V{target_version})[/yellow]"
            )

    except Exception as e:
        console.print(f"\n[bold red]✗ Upgrade failed: {e}[/bold red]")
        raise typer.Exit(1)

    # Restart watcher/worker if they were running
    if watcher_was_running and restart_watcher:
        console.print("\n[dim]Restarting watcher...[/dim]")
        try:
            from . import watcher as watcher_cmd

            exit_code = watcher_cmd.watcher_start_command()
            if exit_code == 0:
                console.print("[green]✓ Watcher restarted[/green]")
            else:
                console.print(
                    "[yellow]⚠ Failed to restart watcher. Run 'aline watcher start' manually.[/yellow]"
                )
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to restart watcher: {e}[/yellow]")

    if worker_was_running and restart_watcher:
        console.print("[dim]Restarting worker...[/dim]")
        try:
            from . import worker as worker_cmd

            exit_code = worker_cmd.worker_start_command()
            if exit_code == 0:
                console.print("[green]✓ Worker restarted[/green]")
            else:
                console.print(
                    "[yellow]⚠ Failed to restart worker. Run 'aline worker start' manually.[/yellow]"
                )
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to restart worker: {e}[/yellow]")

    console.print("\n[bold]Done![/bold]\n")


def version_command():
    """Show aline version and database schema version."""
    console.print("\n[bold blue]═══ Aline Version Info ═══[/bold blue]\n")

    # Package version
    try:
        from importlib.metadata import version

        pkg_version = version("aline-ai")
    except Exception:
        pkg_version = "unknown"

    # Load config and check DB version
    try:
        config = ReAlignConfig.load()
        db_path = Path(config.sqlite_db_path).expanduser()
        current_version = get_current_db_version(db_path)
    except Exception:
        db_path = Path("~/.aline/db/aline.db").expanduser()
        current_version = -1

    table = Table(show_header=False, box=None)
    table.add_column("Label", style="bold")
    table.add_column("Value", style="cyan")

    table.add_row("Package version", pkg_version)
    table.add_row("Schema version (code)", f"V{SCHEMA_VERSION}")

    if current_version == -1:
        table.add_row("Schema version (db)", "[red]Not found[/red]")
    else:
        if current_version < SCHEMA_VERSION:
            table.add_row(
                "Schema version (db)", f"[yellow]V{current_version}[/yellow] (upgrade available)"
            )
        else:
            table.add_row("Schema version (db)", f"[green]V{current_version}[/green]")

    table.add_row("Database path", str(db_path))

    console.print(table)

    if current_version >= 0 and current_version < SCHEMA_VERSION:
        console.print(
            f"\n[yellow]Tip: Run 'aline upgrade' to update database schema V{current_version} → V{SCHEMA_VERSION}[/yellow]"
        )

    console.print("")


if __name__ == "__main__":
    typer.run(upgrade_command)
