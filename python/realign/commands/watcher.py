#!/usr/bin/env python3
"""Aline watcher commands - Manage watcher daemon process."""

import json
import sqlite3
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..config import ReAlignConfig
from ..hooks import find_all_active_sessions
from ..logging_config import setup_logger

# Initialize logger
logger = setup_logger("realign.watcher", "watcher.log")
console = Console()
_warned_permission_pids: set[int] = set()


def get_watcher_pid_file() -> Path:
    """Get path to the watcher PID file."""
    return Path.home() / ".aline/.logs/watcher.pid"


def detect_watcher_process() -> tuple[bool, int | None, str]:
    """
    Detect if watcher is running.

    Returns:
        tuple: (is_running, pid, mode)
        mode can be: 'standalone' or 'unknown'
    """
    # First check for standalone daemon via PID file
    pid_file = get_watcher_pid_file()
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            # Verify process is still running
            try:
                import os

                os.kill(pid, 0)  # Signal 0 just checks if process exists
                return True, pid, "standalone"
            except PermissionError:
                if pid not in _warned_permission_pids:
                    _warned_permission_pids.add(pid)
                    logger.warning(
                        "Insufficient permission to check watcher PID %s – assuming it is running",
                        pid,
                    )
                else:
                    logger.debug(
                        "Insufficient permission to check watcher PID %s – assuming it is running",
                        pid,
                    )
                return True, pid, "standalone"
            except (OSError, ProcessLookupError):
                # PID file exists but process is dead - clean it up
                pid_file.unlink(missing_ok=True)
        except (ValueError, Exception) as e:
            logger.warning(f"Failed to read PID file: {e}")

    # Then check for orphaned daemon process (PID file missing)
    limited_detection = False
    try:
        ps_output = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=2)
        if ps_output.returncode == 0:
            for line in ps_output.stdout.split("\n"):
                if "watcher_daemon.py" in line and "grep" not in line:
                    # Extract PID (second column)
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pid = int(parts[1])
                            return True, pid, "standalone"
                        except ValueError:
                            return True, None, "standalone"
            return False, None, "unknown"
    except subprocess.TimeoutExpired:
        logger.warning("Process check timed out")
        return False, None, "unknown"
    except PermissionError as e:
        logger.warning(f"Process check not permitted: {e}")
        limited_detection = True
    except Exception as e:
        logger.warning(f"Failed to detect watcher process: {e}")
        return False, None, "unknown"

    if limited_detection:
        is_active, _ = check_watcher_log_activity()
        if is_active:
            logger.info("Falling back to log timestamps – watcher appears active")
            return True, None, "standalone"

    return False, None, "unknown"


def detect_all_watcher_processes() -> list[tuple[int, str]]:
    """
    Detect ALL running watcher daemon processes.

    Returns:
        list of tuples: [(pid, mode), ...]
        mode is always: 'standalone'
    """
    processes = []

    try:
        # Use ps to find all watcher_daemon.py processes
        ps_output = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=2)

        if ps_output.returncode == 0:
            for line in ps_output.stdout.split("\n"):
                # Look for watcher_daemon.py processes
                if "watcher_daemon.py" in line and "grep" not in line:
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pid = int(parts[1])
                            processes.append((pid, "standalone"))
                        except ValueError:
                            continue

    except subprocess.TimeoutExpired:
        logger.warning("Process check timed out")
    except Exception as e:
        logger.warning(f"Failed to detect watcher processes: {e}")

    return processes


def check_watcher_log_activity() -> tuple[bool, datetime | None]:
    """
    Check if watcher log has recent activity.

    Returns:
        tuple: (is_active, last_modified)
    """
    log_path = Path.home() / ".aline/.logs/watcher_core.log"
    if log_path.exists():
        try:
            last_modified = datetime.fromtimestamp(log_path.stat().st_mtime)
            seconds_since_modified = (datetime.now() - last_modified).total_seconds()
            is_active = seconds_since_modified < 300  # 5 mins
            return is_active, last_modified
        except Exception as e:
            logger.warning(f"Failed to check log timestamp: {e}")
            return False, None
    return False, None


def get_watched_projects() -> list[Path]:
    """
    Get list of projects being watched by checking ~/.aline directory.

    Returns:
        List of project paths that have Aline initialized
    """
    aline_dir = Path.home() / ".aline"
    if not aline_dir.exists():
        return []

    watched = []
    try:
        for project_dir in aline_dir.iterdir():
            if project_dir.is_dir() and project_dir.name not in [".logs", ".cache"]:
                # Skip test/temporary directories
                if project_dir.name.startswith(("tmp", "test_")):
                    continue
                # Check if it has .git (shadow git repo)
                if (project_dir / ".git").exists():
                    watched.append(project_dir)
    except Exception as e:
        logger.warning(f"Failed to scan watched projects: {e}")

    return watched


def extract_project_name_from_session(session_file: Path) -> str:
    """
    Extract project name from session file path.

    Supports:
        - Claude Code format: ~/.claude/projects/-Users-foo-Projects-MyApp/abc.jsonl → MyApp
        - .aline format: ~/.aline/MyProject/sessions/abc.jsonl → MyProject

    Args:
        session_file: Path to session file

    Returns:
        Project name, or "unknown" if cannot determine
    """
    try:
        # Method 1: Claude Code format
        if ".claude/projects/" in str(session_file):
            project_dir = session_file.parent.name
            if project_dir.startswith("-"):
                # Decode: -Users-foo-Projects-MyApp → MyApp
                parts = project_dir[1:].split("-")
                return parts[-1] if parts else "unknown"

        # Method 2: .aline format
        if ".aline/" in str(session_file):
            # Find the project directory (parent of 'sessions')
            path_parts = session_file.parts
            try:
                aline_idx = path_parts.index(".aline")
                if aline_idx + 1 < len(path_parts):
                    return path_parts[aline_idx + 1]
            except ValueError:
                pass

        return "unknown"
    except Exception as e:
        logger.debug(f"Error extracting project name from {session_file}: {e}")
        return "unknown"


def _detect_session_type(session_file: Path) -> str:
    """
    Detect the type of session file.

    Returns:
        "claude" for Claude Code sessions
        "codex" for Codex/GPT sessions
        "unknown" if cannot determine
    """
    try:
        with open(session_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 20:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("type") in ("assistant", "user") and "message" in data:
                        return "claude"
                    if data.get("type") == "session_meta":
                        payload = data.get("payload", {})
                        if "codex" in payload.get("originator", "").lower():
                            return "codex"
                    if data.get("type") == "response_item":
                        payload = data.get("payload", {})
                        if "message" not in data and "role" in payload:
                            return "codex"
                except json.JSONDecodeError:
                    continue
        return "unknown"
    except Exception as e:
        logger.debug(f"Error detecting session type: {e}")
        return "unknown"


def _count_complete_turns(session_file: Path) -> int:
    """
    Count complete dialogue turns in a session file.

    Returns:
        Number of complete turns
    """
    session_type = _detect_session_type(session_file)

    if session_type == "claude":
        return _count_claude_turns(session_file)
    elif session_type == "codex":
        return _count_codex_turns(session_file)
    else:
        return 0


def _count_claude_turns(session_file: Path) -> int:
    """Count complete dialogue turns for Claude Code sessions."""
    user_message_ids = set()
    try:
        with open(session_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    msg_type = data.get("type")

                    if msg_type == "user":
                        message = data.get("message", {})
                        content = message.get("content", [])

                        is_tool_result = False
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "tool_result":
                                    is_tool_result = True
                                    break

                        if not is_tool_result:
                            uuid = data.get("uuid")
                            if uuid:
                                user_message_ids.add(uuid)
                except json.JSONDecodeError:
                    continue
        return len(user_message_ids)
    except Exception as e:
        logger.debug(f"Error counting Claude turns: {e}")
        return 0


def _count_codex_turns(session_file: Path) -> int:
    """Count complete dialogue turns for Codex sessions."""
    count = 0
    try:
        with open(session_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("type") == "event_msg":
                        payload = data.get("payload", {})
                        if payload.get("type") == "token_count":
                            count += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.debug(f"Error counting Codex turns: {e}")
    return count


def get_session_details(session_file: Path, idle_timeout: float = 300.0) -> Dict:
    """
    Get detailed information about a session file.

    Args:
        session_file: Path to session file
        idle_timeout: Idle timeout threshold in seconds

    Returns:
        dict with session details including:
        - name: session filename
        - path: session file path
        - project_name: project name extracted from path
        - type: claude/codex/unknown
        - turns: number of complete turns
        - mtime: last modified time
        - idle_seconds: seconds since last modification
        - is_idle: whether session exceeds idle timeout
        - size_kb: file size in KB
    """
    try:
        stat = session_file.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime)
        current_time = time.time()
        idle_seconds = current_time - stat.st_mtime

        return {
            "name": session_file.name,
            "path": session_file,
            "project_name": extract_project_name_from_session(session_file),
            "type": _detect_session_type(session_file),
            "turns": _count_complete_turns(session_file),
            "mtime": mtime,
            "idle_seconds": idle_seconds,
            "is_idle": idle_seconds >= idle_timeout,
            "size_kb": stat.st_size / 1024,
        }
    except Exception as e:
        logger.debug(f"Error getting session details for {session_file}: {e}")
        return {
            "name": session_file.name,
            "path": session_file,
            "project_name": "unknown",
            "type": "error",
            "turns": 0,
            "mtime": None,
            "idle_seconds": 0,
            "is_idle": False,
            "size_kb": 0,
        }


def get_all_tracked_sessions() -> List[Dict]:
    """
    Get detailed information for all active sessions being tracked.

    Returns:
        List of session detail dictionaries
    """
    try:
        config = ReAlignConfig.load()

        # Find all active sessions across ALL projects (multi-project mode)
        all_sessions = find_all_active_sessions(config, project_path=None)

        # Get details for each session
        session_details = []
        for session_file in all_sessions:
            if session_file.exists():
                details = get_session_details(session_file)
                session_details.append(details)

        # Sort by mtime (most recent first)
        session_details.sort(key=lambda x: x["mtime"] if x["mtime"] else datetime.min, reverse=True)

        return session_details
    except Exception as e:
        logger.error(f"Error getting tracked sessions: {e}")
        return []


def watcher_status_command(verbose: bool = False) -> int:
    """
    Display watcher status.

    Args:
        verbose: Show detailed session tracking information

    Returns:
        int: Exit code (0 = success, 1 = error)
    """
    try:
        config = ReAlignConfig.load()

        # Check process
        is_running, pid, mode = detect_watcher_process()

        # Determine status
        if is_running:
            status = "Running"
            color = "green"
        else:
            status = "Stopped"
            color = "red"

        # Display status header
        console.print(f"\nWatcher Status: [{color}]{status}[/{color}]")

        if pid:
            console.print(f"PID: {pid}")

        # Show mode
        console.print(f"Mode: Standalone (SQLite)")

        # Active Sessions (New!)
        console.print(f"\n[bold cyan]Active Sessions (Currently Monitoring)[/bold cyan]")
        try:
            from ..hooks import find_all_active_sessions
            from ..adapters import get_adapter_registry

            active_sessions = find_all_active_sessions(config, project_path=None)
            registry = get_adapter_registry()

            if active_sessions:
                for s in active_sessions[:5]:  # Show top 5
                    adapter = registry.auto_detect_adapter(s)
                    source = adapter.name.capitalize() if adapter else "Unknown"
                    mtime = datetime.fromtimestamp(s.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

                    # Try to get project name
                    project_name = "-"
                    if adapter:
                        try:
                            p_path = adapter.extract_project_path(s)
                            if p_path:
                                project_name = p_path.name
                        except:
                            pass

                    console.print(
                        f"  {s.name[:20]:<20} | {source:<12} | {project_name:<15} | {mtime}"
                    )

                if len(active_sessions) > 5:
                    console.print(
                        f"  [dim]... and {len(active_sessions) - 5} more active sessions[/dim]"
                    )
            else:
                console.print(f"  [dim](no active sessions found)[/dim]")
        except Exception as e:
            logger.debug(f"Error finding active sessions for watcher status: {e}")

        # Get recent sessions and conversations from SQLite
        try:
            from ..db import get_database

            db = get_database()
            conn = db._get_connection()

            # Recent Sessions (3)
            console.print(f"\n[bold]Recent Sessions[/bold]")
            sessions = list(
                conn.execute(
                    """
                SELECT id, session_type, workspace_path, last_activity_at
                FROM sessions
                ORDER BY last_activity_at DESC
                LIMIT 3
            """
                )
            )

            if sessions:
                for s in sessions:
                    # Format session name: first 10 + ... + last 10 (including extension)
                    session_id = s[0]
                    if len(session_id) > 23:
                        session_name = session_id[:10] + "..." + session_id[-10:]
                    else:
                        session_name = session_id

                    # Format source type
                    session_type = s[1] or "unknown"
                    source_map = {
                        "claude": "Claude Code",
                        "codex": "Codex",
                        "gemini": "Gemini",
                    }
                    source = source_map.get(session_type, session_type)

                    # Format workspace/project
                    workspace = s[2]
                    if workspace:
                        workspace = Path(workspace).name
                    else:
                        workspace = "-"

                    # Format last activity as absolute time
                    last_activity = s[3] or "-"
                    if last_activity and last_activity != "-":
                        try:
                            dt = datetime.fromisoformat(last_activity)
                            last_activity = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            pass

                    console.print(f"  {session_name} | {source} | {workspace} | {last_activity}")
            else:
                console.print(f"  [dim](no sessions yet)[/dim]")

            # Recent Conversations (5)
            console.print(f"\n[bold]Recent Conversations[/bold]")
            turns = list(
                conn.execute(
                    """
                SELECT t.turn_number, t.llm_title, t.timestamp,
                       s.id, s.session_type, s.workspace_path
                FROM turns t
                JOIN sessions s ON t.session_id = s.id
                ORDER BY t.timestamp DESC
                LIMIT 5
            """
                )
            )

            if turns:
                for t in turns:
                    turn_num = t[0]
                    title = t[1] or "(no title)"
                    timestamp = t[2] or "-"
                    session_id = t[3]
                    session_type = t[4] or "unknown"
                    workspace = t[5]

                    # Format session name
                    if len(session_id) > 23:
                        session_name = session_id[:10] + "..." + session_id[-10:]
                    else:
                        session_name = session_id

                    # Format source type
                    source_map = {
                        "claude": "Claude Code",
                        "codex": "Codex",
                        "gemini": "Gemini",
                    }
                    source = source_map.get(session_type, session_type)

                    # Format workspace
                    if workspace:
                        workspace = Path(workspace).name
                    else:
                        workspace = "-"

                    # Format timestamp as absolute time
                    time_str = "-"
                    if timestamp and timestamp != "-":
                        try:
                            dt = datetime.fromisoformat(timestamp)
                            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            time_str = timestamp

                    # First line: session info with turn number
                    console.print(
                        f"  {session_name} | {source} | {workspace} | Turn#{turn_num} | {time_str}"
                    )

                    # Second line: full llm_title with indentation and proper wrapping
                    indent = "    "
                    wrapped = textwrap.fill(
                        title, width=80, initial_indent=indent, subsequent_indent=indent
                    )
                    console.print(f"[dim]{wrapped}[/dim]")
                    console.print()  # Empty line between conversations
            else:
                console.print(f"  [dim](no conversations yet)[/dim]")

            # Database statistics
            console.print(f"\n[bold]Database Statistics[/bold]")
            stats = {}
            for table in ["sessions", "turns", "projects", "events"]:
                try:
                    count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                    stats[table] = count
                except:
                    stats[table] = 0
            console.print(
                f"  Sessions: {stats['sessions']} | Turns: {stats['turns']} | Projects: {stats['projects']} | Events: {stats['events']}"
            )

        except Exception as e:
            logger.debug(f"Error reading from database: {e}")
            console.print(f"\n[dim]Database not available or empty[/dim]")

        # Suggestions
        if status == "Stopped":
            console.print(f"\n[dim]Run 'aline watcher start' to start the watcher[/dim]")

        console.print()
        return 0

    except Exception as e:
        logger.error(f"Error in watcher status: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1


def watcher_start_command() -> int:
    """
    Start the watcher in standalone mode.

    Launches a background daemon process that monitors session files
    and auto-commits changes.

    Returns:
        int: Exit code (0 = success, 1 = error)
    """
    try:
        # Check login status first
        from ..auth import is_logged_in

        if not is_logged_in():
            console.print("[red]✗ Not logged in. Watcher requires authentication.[/red]")
            console.print("[dim]Run 'aline login' first.[/dim]")
            return 1

        # Check if already running
        is_running, pid, mode = detect_watcher_process()

        if is_running:
            console.print(f"[yellow]Watcher is already running (PID: {pid}, mode: {mode})[/yellow]")
            console.print(f"[dim]Use 'aline watcher stop' to stop it first[/dim]")
            return 0

        console.print(f"[cyan]Starting standalone watcher daemon...[/cyan]")

        # Launch the daemon as a background process
        import os
        import importlib.util

        # Get the path to the daemon script
        spec = importlib.util.find_spec("realign.watcher_daemon")
        if not spec or not spec.origin:
            console.print(f"[red]✗ Could not find watcher daemon module[/red]")
            return 1

        daemon_script = spec.origin

        # Launch daemon using python with nohup-like behavior
        # Using start_new_session=True to detach from terminal
        log_dir = Path.home() / ".aline/.logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Explicitly pass environment to ensure ACME_USER_DATA_DIR is inherited.
        subprocess.Popen(
            [sys.executable, daemon_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=Path.cwd(),
            env=os.environ.copy(),  # Explicitly inherit all env vars including ACME_USER_DATA_DIR
        )

        # Give it a moment to start
        import time

        time.sleep(1)

        # Verify it started
        is_running, pid, mode = detect_watcher_process()

        if is_running:
            console.print(f"[green]✓ Watcher started successfully (PID: {pid})[/green]")
            console.print("[dim]File logs are disabled (except dashboard.log)[/dim]")

            # Ensure worker is running (turn/session summaries are processed by the worker).
            try:
                from . import worker as worker_cmd

                worker_cmd.worker_start_command()
            except Exception:
                pass

            return 0
        else:
            console.print(f"[red]✗ Failed to start watcher[/red]")
            console.print("[dim]No watcher log file is generated in this mode.[/dim]")
            return 1

    except Exception as e:
        logger.error(f"Error in watcher start: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1


def watcher_stop_command() -> int:
    """
    Stop ALL watcher daemon processes.

    Returns:
        int: Exit code (0 = success, 1 = error)
    """
    import time

    try:
        # Detect ALL running watcher processes
        all_processes = detect_all_watcher_processes()

        if not all_processes:
            console.print(f"[yellow]No watcher processes found[/yellow]")
            console.print(f"[dim]Use 'aline watcher start' to start it[/dim]")
            return 1

        # Display all processes that will be stopped
        if len(all_processes) == 1:
            pid, mode = all_processes[0]
            console.print(f"[cyan]Stopping watcher (PID: {pid}, mode: {mode})...[/cyan]")
        else:
            console.print(
                f"[cyan]Found {len(all_processes)} watcher processes, stopping all...[/cyan]"
            )
            for pid, mode in all_processes:
                console.print(f"  • PID: {pid} (mode: {mode})")

        # Send SIGTERM to all processes
        failed_pids = []
        for pid, mode in all_processes:
            try:
                subprocess.run(["kill", str(pid)], check=True, timeout=2)
            except subprocess.CalledProcessError:
                failed_pids.append((pid, mode))

        # Wait a moment for graceful shutdown
        time.sleep(1)

        # Check if any processes are still running
        still_running = detect_all_watcher_processes()

        if still_running:
            # Force kill remaining processes
            console.print(
                f"[yellow]{len(still_running)} process(es) still running, forcing stop...[/yellow]"
            )
            for pid, mode in still_running:
                try:
                    subprocess.run(["kill", "-9", str(pid)], check=True, timeout=2)
                except subprocess.CalledProcessError as e:
                    console.print(f"[red]✗ Failed to force-stop PID {pid}: {e}[/red]")
                    failed_pids.append((pid, mode))

        # Clean up PID file
        get_watcher_pid_file().unlink(missing_ok=True)

        # Final verification
        time.sleep(0.5)
        final_check = detect_all_watcher_processes()

        if not final_check:
            if len(all_processes) == 1:
                console.print(f"[green]✓ Watcher stopped successfully[/green]")
            else:
                console.print(
                    f"[green]✓ All {len(all_processes)} watcher processes stopped successfully[/green]"
                )
            return 0
        else:
            console.print(f"[red]✗ Failed to stop {len(final_check)} process(es)[/red]")
            for pid, mode in final_check:
                console.print(f"  • PID {pid} ({mode}) is still running")
            return 1

    except Exception as e:
        logger.error(f"Error stopping watcher: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1


def watcher_fresh_command() -> int:
    """
    Restart the watcher daemon (stop + start).

    This command stops the watcher if it's running, then starts it again.
    It's equivalent to running 'aline watcher stop' followed by 'aline watcher start'.

    Returns:
        int: Exit code (0 = success, 1 = error)
    """
    try:
        console.print(f"[cyan]Restarting watcher daemon...[/cyan]\n")

        # Check if watcher is running
        is_running, pid, mode = detect_watcher_process()

        if is_running:
            # Stop the watcher
            stop_exit_code = watcher_stop_command()

            # If stop failed, don't try to start
            if stop_exit_code != 0:
                console.print(f"\n[red]✗ Failed to stop watcher, aborting restart[/red]")
                return stop_exit_code

            console.print()  # Add blank line for readability
        else:
            console.print(f"[dim]Watcher is not running, starting it...[/dim]\n")

        # Start the watcher
        start_exit_code = watcher_start_command()

        if start_exit_code == 0:
            console.print(f"\n[green]✓ Watcher restarted successfully[/green]")
        else:
            console.print(f"\n[red]✗ Failed to start watcher[/red]")

        return start_exit_code

    except Exception as e:
        logger.error(f"Error restarting watcher: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1


def watcher_command(
    action: str = "status",
) -> int:
    """
    Main watcher command dispatcher.

    Args:
        action: Action to perform (status, start, stop)

    Returns:
        int: Exit code
    """
    if action == "status":
        return watcher_status_command()
    elif action == "start":
        return watcher_start_command()
    elif action == "stop":
        return watcher_stop_command()
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print(f"[dim]Available actions: status, start, stop[/dim]")
        return 1


# =============================================================================
# Session management commands
# =============================================================================


def _format_relative_time(dt: datetime) -> str:
    """Format datetime as relative time string (e.g., '2h ago', '3d ago')."""
    now = datetime.now()
    diff = now - dt

    seconds = int(diff.total_seconds())
    if seconds < 0:
        return "just now"
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months < 12:
        return f"{months}mo ago"
    years = days // 365
    return f"{years}y ago"


def _get_imported_sessions(db, exclude_session_ids: set) -> list:
    """
    Get imported sessions from database (sessions without corresponding files).

    Args:
        db: Database interface
        exclude_session_ids: Set of session IDs to exclude (already tracked)

    Returns:
        List of session info dicts compatible with _get_session_tracking_status output
    """
    imported_infos = []

    try:
        # Query all sessions with session_type='imported' or empty session_file_path
        all_sessions = db.list_sessions(limit=1000)

        for session in all_sessions:
            # Skip if already in tracked sessions
            if session.id in exclude_session_ids:
                continue

            # Only include sessions that are imported via import_shares command
            # Check for: 1) metadata.source == 'share_import' OR 2) session_type == 'imported'
            # AND file_path is '.' (from Path(''))
            file_path_str = str(session.session_file_path) if session.session_file_path else ""

            # Check if file path indicates imported session
            has_no_file = not session.session_file_path or file_path_str in ("", ".")

            # Check metadata for share_import source
            is_from_share_import = False
            if session.metadata:
                try:
                    import json

                    metadata = (
                        json.loads(session.metadata)
                        if isinstance(session.metadata, str)
                        else session.metadata
                    )
                    is_from_share_import = metadata.get("source") == "share_import"
                except Exception:
                    pass

            # Session is imported if it has no file AND (is marked as imported OR from share_import)
            is_imported = has_no_file and (
                session.session_type == "imported" or is_from_share_import
            )

            if not is_imported:
                continue

            # Get turn count for this session
            try:
                committed_turns = int(db.get_max_turn_number(session.id))
            except Exception:
                committed_turns = 0

            # For imported sessions, committed_turns == total_turns (all turns are in DB)
            total_turns = committed_turns

            # Determine status (imported sessions are always "tracked" if they have turns)
            status = "tracked" if committed_turns > 0 else "new"

            # Extract project name from workspace_path or metadata
            project_name = "-"
            if session.workspace_path:
                project_name = Path(session.workspace_path).name
            elif session.metadata:
                import json

                try:
                    metadata = (
                        json.loads(session.metadata)
                        if isinstance(session.metadata, str)
                        else session.metadata
                    )
                    project_name = metadata.get("project_name", "-")
                except Exception:
                    pass

            imported_infos.append(
                {
                    "status": status,
                    "committed_turns": committed_turns,
                    "total_turns": total_turns,
                    "session_id": session.id,
                    "source": session.session_type or "imported",
                    "project_name": project_name,
                    "project_path": (
                        Path(session.workspace_path) if session.workspace_path else None
                    ),
                    "progress_source": "db",
                    "created_at": session.created_at,  # Use import time, not original session start time
                    "last_activity": session.last_activity_at,
                    "session_file": None,  # No file for imported sessions
                    "session_title": session.session_title,
                    "session_summary": session.session_summary,
                    "created_by": session.created_by,
                }
            )

    except Exception as e:
        logger.warning(f"Error fetching imported sessions: {e}")

    return imported_infos


def _get_session_tracking_status_batch(
    session_files: List[Path],
    config: ReAlignConfig,
    db=None,
    detect_turns: bool = False,
) -> List[dict]:
    """
    Batch version of _get_session_tracking_status - processes multiple sessions at once.

    This optimizes performance by:
    1. Fetching session records for all sessions in one query
    2. Fetching committed_turns (max turn numbers) for all sessions in one query
    3. Using mtime-based caching for total_turns (V12) - only recompute when file changed
    4. Only computing total_turns when detect_turns=True (expensive operation)

    Args:
        session_files: List of session file paths
        config: ReAlignConfig instance
        db: Optional database connection
        detect_turns: If True, compute total_turns from files (slow). Default False.

    Returns:
        List of session info dicts (same format as _get_session_tracking_status)
    """
    # Avoid instantiating DialogueWatcher unless we truly need expensive turn detection.
    # Session listing can become slow with many sessions if we repeatedly auto-detect adapters
    # and parse session files just to infer the project name.
    watcher = None
    if detect_turns:
        from realign.watcher_core import DialogueWatcher

        watcher = DialogueWatcher()
    from realign.adapters.registry import get_adapter_registry

    registry = get_adapter_registry()
    adapters = {
        "claude": registry.get_adapter("claude"),
        "codex": registry.get_adapter("codex"),
        "gemini": registry.get_adapter("gemini"),
    }

    def _infer_adapter_name(session_file: Path) -> str:
        # Fast-path by filesystem layout (no file I/O).
        parts = session_file.parts
        if ".claude" in parts and "projects" in parts:
            return "claude"
        if ".codex" in parts and "sessions" in parts:
            return "codex"
        if ".gemini" in parts:
            return "gemini"
        return "unknown"

    session_infos = []

    if not session_files:
        return session_infos

    # Build lookup dicts from session files with metadata (stat only once).
    session_data_by_id: Dict[str, Dict] = {}
    for session_file in session_files:
        session_id = session_file.stem
        try:
            stat = session_file.stat()
            mtime = stat.st_mtime
            last_activity = datetime.fromtimestamp(stat.st_mtime)
            created_at = datetime.fromtimestamp(getattr(stat, "st_birthtime", stat.st_ctime))
        except Exception:
            mtime = None
            last_activity = datetime.now()
            created_at = datetime.now()
        session_data_by_id[session_id] = {
            "session_file": session_file,
            "session_id": session_id,
            "mtime": mtime,
            "created_at": created_at,
            "last_activity": last_activity,
        }

    # Fetch session records from DB (for title, summary, total_turns, etc.)
    session_records: Dict[str, Any] = {}
    tracked_session_ids: set[str] = set()  # Sessions that exist in DB
    if db is not None:
        try:
            session_ids = list(session_data_by_id.keys())
            if session_ids:
                # Batch fetch session records (includes total_turns, total_turns_mtime, title, summary)
                records = db.get_sessions_by_ids(session_ids)
                for record in records:
                    session_records[record.id] = record
                    tracked_session_ids.add(record.id)
        except Exception:
            pass

    # Only fetch committed_turns when detect_turns is enabled
    committed_turns_map: dict[str, int] = {}
    if detect_turns and db is not None:
        try:
            session_ids = list(session_data_by_id.keys())
            if session_ids:
                committed_turns_map = db.get_max_turn_numbers_batch(session_ids)
        except Exception:
            pass

    # Track sessions that need cache updates
    cache_updates: List[tuple] = []  # [(session_id, total_turns, mtime), ...]

    # Now process each session without additional DB calls
    for session_file in session_files:
        session_id = session_file.stem
        session_data = session_data_by_id[session_id]
        file_mtime = session_data["mtime"]

        record = session_records.get(session_id)

        # Prefer DB metadata when available (fast, avoids parsing session files).
        session_type = None
        project_path: Optional[Path] = None
        if record:
            try:
                session_type = str(getattr(record, "session_type", "") or "") or None
            except Exception:
                session_type = None
            try:
                workspace_path = getattr(record, "workspace_path", None)
                if isinstance(workspace_path, str) and workspace_path.strip():
                    project_path = Path(workspace_path.strip())
            except Exception:
                project_path = None

        if not session_type:
            session_type = _infer_adapter_name(session_file)
        if project_path is None:
            adapter = adapters.get(session_type or "")
            if adapter:
                try:
                    project_path = adapter.extract_project_path(session_file)
                except Exception:
                    project_path = None

        # Determine status based on whether session exists in DB
        # (simplified logic - no longer based on turn counts)
        is_in_db = session_id in tracked_session_ids
        status = "tracked" if is_in_db else "new"

        # Get committed_turns and total_turns only if detect_turns is enabled
        committed_turns = None
        total_turns = None
        if detect_turns:
            committed_turns = committed_turns_map.get(session_id, 0)

            # V12: Use mtime-based caching for total_turns
            record = session_records.get(session_id)
            cache_valid = False
            if record and record.total_turns is not None and file_mtime is not None:
                # Check if cached mtime matches current file mtime
                if record.total_turns_mtime is not None:
                    # Allow small tolerance for float comparison
                    if abs(record.total_turns_mtime - file_mtime) < 0.001:
                        cache_valid = True
                        total_turns = record.total_turns

            if not cache_valid:
                # Cache miss or invalid - recompute from file
                # Use _get_total_turn_count which uses get_detailed_analysis()['total_turns']
                # instead of count_complete_turns() which excludes the last turn
                assert watcher is not None
                total_turns = watcher._get_total_turn_count(session_file)
                # Schedule cache update if we have DB and valid mtime
                if db is not None and file_mtime is not None:
                    cache_updates.append((session_id, total_turns, file_mtime))

            # Update status based on turn counts when detect_turns is enabled
            if committed_turns == 0:
                status = "new"
            elif committed_turns >= total_turns:
                status = "tracked"
            else:
                status = "partial"

        created_at = session_data["created_at"]
        last_activity = session_data["last_activity"]

        info = {
            "status": status,
            "committed_turns": committed_turns,
            "total_turns": total_turns,
            "session_id": session_id,
            "source": session_type,
            "project_name": project_path.name if project_path else "-",
            "project_path": project_path,
            "progress_source": "db" if db is not None else "unknown",
            "created_at": created_at,
            "last_activity": last_activity,
            "session_file": session_file,
        }

        # Add session record fields if available
        if record and status in ("partial", "tracked"):
            info["session_title"] = record.session_title
            info["session_summary"] = record.session_summary
            info["created_by"] = record.created_by

        session_infos.append(info)

    # Apply cache updates to DB (batch after processing)
    if db is not None and cache_updates:
        for session_id, total_turns, mtime in cache_updates:
            try:
                db.update_session_total_turns_with_mtime(session_id, total_turns, mtime)
            except Exception as e:
                logger.debug(f"Failed to update cache for {session_id}: {e}")

    return session_infos


def _get_session_tracking_status(session_file: Path, config: ReAlignConfig, db=None) -> dict:
    """
    Get tracking status for a session file.

    Returns dict with keys:
        - status: "tracked" | "partial" | "new"
        - committed_turns: int
        - total_turns: int
        - session_id: str
        - source: str (claude/codex/gemini)
        - project_name: str | None
        - last_activity: datetime
        - session_file: Path
    """
    from realign.watcher_core import DialogueWatcher

    session_id = session_file.stem

    # Create a temporary watcher instance for helper methods
    watcher = DialogueWatcher()

    # V10: Try to get cached total_turns from database first (avoids reading file)
    total_turns = None
    if db is not None:
        try:
            session_record = db.get_session_by_id(session_id)
            if (
                session_record
                and session_record.total_turns is not None
                and session_record.total_turns > 0
            ):
                total_turns = session_record.total_turns
        except Exception:
            pass

    # Fall back to reading file if not cached
    # Use _get_total_turn_count which uses get_detailed_analysis()['total_turns']
    # instead of count_complete_turns() which excludes the last turn
    if total_turns is None:
        total_turns = watcher._get_total_turn_count(session_file)

    # Get session type and project
    session_type = watcher._detect_session_type(session_file)
    project_path = watcher._extract_project_path(session_file)

    # SQLite is the source of truth:
    # committed_turns is derived from max(turn_number) stored for this session.
    committed_turns = 0
    if db is not None:
        try:
            committed_turns = int(db.get_max_turn_number(session_id))
        except Exception:
            committed_turns = 0

    # Determine status
    if committed_turns == 0:
        status = "new"
    elif committed_turns >= total_turns:
        status = "tracked"
    else:
        status = "partial"

    # Get file times
    try:
        stat = session_file.stat()
        last_activity = datetime.fromtimestamp(stat.st_mtime)
        # Use birthtime on macOS, fallback to ctime
        created_at = datetime.fromtimestamp(getattr(stat, "st_birthtime", stat.st_ctime))
    except Exception:
        last_activity = datetime.now()
        created_at = datetime.now()

    return {
        "status": status,
        "committed_turns": committed_turns,
        "total_turns": total_turns,
        "session_id": session_id,
        "source": session_type,
        "project_name": project_path.name if project_path else "-",
        "project_path": project_path,
        "progress_source": "db" if db is not None else "unknown",
        "created_at": created_at,
        "last_activity": last_activity,
        "session_file": session_file,
    }


def _get_sorted_session_infos(detect_turns: bool = False, include_empty: bool = True) -> List[dict]:
    """
    Get sorted session infos list - shared logic between session list and event generate.

    This function provides the canonical session ordering used by both
    'aline watcher session list' and 'aline watcher event generate' to ensure
    index consistency.

    Args:
        detect_turns: If True, compute total_turns from files (uses mtime cache). Default False.
        include_empty: If True, include sessions with 0 turns. Default True.
                      When False and detect_turns=True, filters out sessions with total_turns=0.

    Returns:
        List of session info dicts, sorted by created_at descending (newest first).
        Returns empty list if no sessions found or on error.
    """
    config = ReAlignConfig.load()
    sessions = find_all_active_sessions(config, project_path=None)

    # Get status for each session (DB is optional; if missing, treat as empty).
    db = None
    try:
        env_db_path = os.getenv("REALIGN_SQLITE_DB_PATH") or os.getenv("REALIGN_DB_PATH")
        resolved_db_path = Path(env_db_path or config.sqlite_db_path).expanduser()
        if resolved_db_path.exists():
            from ..db import get_database

            # Use a read-only DB connection for listing to avoid blocking on worker writes.
            db = get_database(read_only=True)
    except Exception as e:
        logger.debug(f"Failed to load database for session titles: {e}")

    session_infos = []
    tracked_session_ids = set()

    # Process file-based sessions in batch (optimized)
    try:
        batch_infos = _get_session_tracking_status_batch(
            sessions, config, db=db, detect_turns=detect_turns
        )
        for info in batch_infos:
            session_infos.append(info)
            tracked_session_ids.add(info["session_id"])
    except Exception as e:
        logger.warning(f"Failed to get batch session status: {e}")
        # Fall back to individual processing
        for s in sessions:
            try:
                info = _get_session_tracking_status(s, config, db=db)
                session_infos.append(info)
                tracked_session_ids.add(info["session_id"])
            except Exception as e2:
                logger.warning(f"Failed to get status for {s.name}: {e2}")

    # Process imported sessions from database (sessions without files)
    if db is not None:
        try:
            imported_sessions = _get_imported_sessions(db, tracked_session_ids)
            session_infos.extend(imported_sessions)
        except Exception as e:
            logger.warning(f"Failed to get imported sessions: {e}")

    # Sort by created_at descending (newest first)
    session_infos.sort(key=lambda x: x["created_at"], reverse=True)

    # Filter out empty sessions if requested (only effective when detect_turns=True)
    if not include_empty and detect_turns:
        session_infos = [
            info
            for info in session_infos
            if info.get("total_turns") is not None and info["total_turns"] > 0
        ]

    return session_infos


def watcher_session_list_command(
    verbose: bool = False,
    page: int = 1,
    per_page: int = 30,
    json_output: bool = False,
    detect_turns: bool = False,
    records: bool = False,
    include_all: bool = False,
) -> int:
    """
    List discovered sessions with tracking status.

    Args:
        verbose: Show detailed information
        json_output: Output results in JSON format
        page: Page number (1-based)
        per_page: Number of sessions per page
        detect_turns: If True, compute total_turns from files (uses mtime cache). Default False.
        records: If True, include turn titles for each session (JSON mode only).
        include_all: If True, include sessions with 0 turns. Default False (hides empty sessions
                    when detect_turns is enabled).

    Returns:
        int: Exit code
    """
    try:
        # Use shared helper to get sorted session list (ensures index consistency)
        # When detect_turns=True and include_all=False, filter out empty sessions
        session_infos = _get_sorted_session_infos(
            detect_turns=detect_turns, include_empty=include_all or not detect_turns
        )

        if not session_infos:
            console.print("[yellow]No sessions discovered.[/yellow]")
            console.print("[dim]Sessions are discovered from:[/dim]")
            console.print("[dim]  • Claude Code: ~/.claude/projects/[/dim]")
            console.print("[dim]  • Codex (legacy): ~/.codex/sessions/[/dim]")
            console.print("[dim]  • Codex (isolated): ~/.aline/codex_homes/*/sessions/[/dim]")
            console.print("[dim]  • Gemini: ~/.gemini/tmp/*/chats/[/dim]")
            console.print("[dim]  • Imported shares: Database[/dim]")
            return 0

        # Get DB connection for records feature (turn titles)
        db = None
        if records:
            try:
                config = ReAlignConfig.load()
                env_db_path = os.getenv("REALIGN_SQLITE_DB_PATH") or os.getenv("REALIGN_DB_PATH")
                resolved_db_path = Path(env_db_path or config.sqlite_db_path).expanduser()
                if resolved_db_path.exists():
                    from ..db import get_database

                    # Read-only is sufficient for listing turn titles.
                    db = get_database(read_only=True)
            except Exception as e:
                logger.debug(f"Failed to load database for turn titles: {e}")

        # JSON output mode
        if json_output:
            import json
            import math

            total_sessions = len(session_infos)
            total_pages = max(1, math.ceil(total_sessions / per_page))

            if page < 1:
                page = 1

            start_index = (page - 1) * per_page
            end_index = min(total_sessions, start_index + per_page)
            page_session_infos = session_infos[start_index:end_index]

            # Pre-fetch turn titles if records=True
            session_turns_map: Dict[str, list] = {}
            if records and db is not None:
                try:
                    session_ids = [info["session_id"] for info in page_session_infos]
                    conn = db._get_connection()
                    placeholders = ",".join("?" * len(session_ids))
                    cursor = conn.execute(
                        f"""
                        SELECT session_id, turn_number, llm_title, created_at
                        FROM turns
                        WHERE session_id IN ({placeholders})
                        ORDER BY session_id, turn_number ASC
                        """,
                        session_ids,
                    )
                    for row in cursor:
                        sid, turn_num, title, created_at = row
                        if sid not in session_turns_map:
                            session_turns_map[sid] = []
                        # Convert datetime to ISO format string if needed
                        if created_at and isinstance(created_at, datetime):
                            created_at_str = created_at.isoformat()
                        else:
                            created_at_str = created_at
                        session_turns_map[sid].append(
                            {
                                "turn_number": turn_num,
                                "title": title or "(no title)",
                                "created_at": created_at_str,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to fetch turn titles: {e}")

            # Build JSON output
            json_sessions = []
            for idx, info in enumerate(page_session_infos, start_index + 1):
                session_data = {
                    "index": idx,
                    "status": info["status"],
                    "source": info["source"],
                    "project_name": info["project_name"],
                    "session_id": info["session_id"],
                    "committed_turns": info["committed_turns"],
                    "total_turns": info["total_turns"],
                    "created_at": info["created_at"].isoformat(),
                    "last_activity": info["last_activity"].isoformat(),
                    "session_title": info.get("session_title"),
                    "session_summary": info.get("session_summary"),
                    "created_by": info.get("created_by"),
                    "session_file": (
                        str(info.get("session_file")) if info.get("session_file") else None
                    ),
                }
                # Add turn records if requested
                if records:
                    session_data["turns"] = session_turns_map.get(info["session_id"], [])
                json_sessions.append(session_data)

            # Count by status
            tracked_count = sum(1 for info in session_infos if info["status"] == "tracked")
            partial_count = sum(1 for info in session_infos if info["status"] == "partial")
            new_count = sum(1 for info in session_infos if info["status"] == "new")

            output = {
                "total": total_sessions,
                "page": page,
                "per_page": per_page,
                "total_pages": total_pages,
                "tracked_count": tracked_count,
                "partial_count": partial_count,
                "new_count": new_count,
                "sessions": json_sessions,
            }
            print(json.dumps(output, indent=2))
            return 0

        # Count by status
        tracked_count = sum(1 for info in session_infos if info["status"] == "tracked")
        partial_count = sum(1 for info in session_infos if info["status"] == "partial")
        new_count = sum(1 for info in session_infos if info["status"] == "new")

        console.print(f"\n[bold]Discovered Sessions ({len(session_infos)} total):[/bold]")
        if detect_turns:
            console.print(
                f"  [green]{tracked_count} tracked[/green], [yellow]{partial_count} partial[/yellow], [dim]{new_count} new[/dim]\n"
            )
        else:
            console.print(f"  [green]{tracked_count} tracked[/green], [dim]{new_count} new[/dim]\n")

        import math

        total_sessions = len(session_infos)
        total_pages = max(1, math.ceil(total_sessions / per_page))

        if page < 1:
            page = 1

        start_index = (page - 1) * per_page
        if start_index >= total_sessions:
            console.print(f"[red]Error: page {page} is out of range (1-{total_pages})[/red]")
            console.print("[dim]Tip: use --page 1 to start from the beginning[/dim]")
            return 1

        end_index = min(total_sessions, start_index + per_page)
        page_session_infos = session_infos[start_index:end_index]

        console.print(
            f"[dim]Showing {start_index + 1}-{end_index} of {total_sessions} (page {page}/{total_pages})[/dim]\n"
        )

        # Display table
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("STATUS", no_wrap=True)
        table.add_column("CREATOR", no_wrap=True)
        table.add_column("SOURCE", no_wrap=True)
        table.add_column("PROJECT")
        table.add_column("SESSION ID")
        if detect_turns:
            table.add_column("TURNS", justify="right", no_wrap=True)
        table.add_column("CREATED", no_wrap=True)
        table.add_column("ACTIVITY", no_wrap=True)
        table.add_column("TITLE", overflow="fold")

        for idx, info in enumerate(page_session_infos, start_index + 1):
            status_display = {
                "tracked": "[green]tracked[/green]",
                "partial": "[yellow]partial[/yellow]",
                "new": "[dim]new[/dim]",
            }.get(info["status"], info["status"])

            created_str = info["created_at"].strftime("%m-%d %H:%M")  # Shorter date format
            activity_str = _format_relative_time(info["last_activity"])

            # Truncate session ID if too long (show enough for UUID matching)
            session_id = info["session_id"]
            if len(session_id) > 36:
                session_id = session_id[:33] + "..."

            title_str = ""
            if info["status"] in ("partial", "tracked"):
                title_str = info.get("session_title") or "-"
                title_str = title_str.strip()

            # V18: Display created_by UID (truncate if too long)
            creator_display = "-"
            if info["status"] in ("partial", "tracked"):
                created_by = info.get("created_by")
                if created_by:
                    creator_display = created_by[:8] + "..."

            # Truncate project name
            project_name = info["project_name"]
            if len(project_name) > 12:
                project_name = project_name[:10] + ".."

            # Build row data
            row_data = [
                str(idx),
                status_display,
                creator_display,
                info["source"],
                project_name,
                session_id,
            ]

            if detect_turns:
                total_turns = info.get("total_turns") or 0
                committed_turns = info.get("committed_turns") or 0
                turns_str = f"{committed_turns}/{total_turns}"
                row_data.append(turns_str)

            row_data.extend(
                [
                    created_str,
                    activity_str,
                    title_str,
                ]
            )

            table.add_row(*row_data)

        console.print(table)
        console.print()
        console.print(
            "[dim]Legend: tracked=all imported, partial=some imported, new=not tracked[/dim]"
        )
        console.print("[dim]Title: shown for tracked/partial sessions (from DB)[/dim]")
        if verbose:
            console.print(
                "[dim]Progress: tracking status is derived from SQLite turns (max turn_number)[/dim]"
            )
            # Show where the state is coming from so "tracked" isn't mysterious after cleanup.
            for idx, info in enumerate(page_session_infos, start_index + 1):
                if info.get("status") in ("tracked", "partial"):
                    console.print(
                        f"[dim]  #{idx} {info.get('session_id', '-')} → {info.get('progress_source', '-')}[/dim]"
                    )
        if total_pages > 1:
            if page > 1:
                console.print(f"[dim]Prev page: aline watcher session list --page {page - 1}[/dim]")
            if page < total_pages:
                console.print(f"[dim]Next page: aline watcher session list --page {page + 1}[/dim]")
            console.print("[dim]Tip: adjust page size with --per-page[/dim]")

        console.print(
            "[dim]Import is available from OneContext Dashboard: Config -> Import history[/dim]"
        )
        console.print()

        return 0

    except Exception as e:
        logger.error(f"Error listing sessions: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1


def watcher_event_generate_command(session_selector: str, show_sessions: bool = False) -> int:
    """
    Generate an event from selected sessions.

    Args:
        session_selector: Session selector - can be:
            - Single number: "1"
            - Range: "1-5"
            - Multiple numbers: "1,3,5"
            - UUID or prefix: "abc123de" (at least 4 hex chars)
            - Multiple UUIDs: "abc123de,def456gh,xyz789"
            - "list" to show available sessions
        show_sessions: If True, show available sessions and exit

    Returns:
        int: Exit code
    """
    import uuid
    from datetime import datetime
    from ..db.base import EventRecord

    try:
        from ..db import get_database
        from ..events.event_summarizer import _generate_event_summary_llm
        from ..config import ReAlignConfig

        db = get_database()

        # V9: Get user identity for creator tracking
        config = ReAlignConfig.load()

        # Use shared helper to get sorted session list (ensures index consistency with session list)
        session_infos = _get_sorted_session_infos(detect_turns=False)
        if not session_infos:
            console.print("[yellow]No sessions discovered.[/yellow]")
            console.print(
                "[dim]Use OneContext Dashboard -> Config -> Import history to import sessions first.[/dim]"
            )
            return 1

        # If selector is "list", show available sessions
        if session_selector.lower() == "list" or show_sessions:
            console.print(f"\n[bold]Sessions ({len(session_infos)} available):[/bold]\n")
            table = Table(show_header=True, header_style="bold", box=None)
            table.add_column("#", justify="right", style="cyan", no_wrap=True)
            table.add_column("SESSION ID", no_wrap=True)
            table.add_column("STATUS", no_wrap=True)
            table.add_column("TITLE", overflow="fold")
            table.add_column("ACTIVITY", no_wrap=True)

            for idx, info in enumerate(session_infos, 1):
                # Show enough of session ID for UUID matching (first 16 chars)
                sid = info["session_id"]
                session_id_display = sid[:16] + "..." if len(sid) > 16 else sid
                status = info["status"]
                status_style = {"tracked": "green", "partial": "yellow", "new": "dim"}.get(
                    status, ""
                )
                title = info.get("session_title") or "-"
                if len(title) > 50:
                    title = title[:47] + "..."
                activity = _format_relative_time(info["last_activity"])
                table.add_row(
                    str(idx),
                    session_id_display,
                    f"[{status_style}]{status}[/{status_style}]",
                    title,
                    activity,
                )

            console.print(table)
            console.print()
            console.print("[dim]Usage: aline watcher event generate <selector>[/dim]")
            console.print("[dim]Examples: 1-3, abc123de, abc123de,def456gh[/dim]")
            console.print(
                "[dim]Note: Only 'tracked' sessions can be used to generate events.[/dim]"
            )
            return 0

        # Check if selector is a UUID (or UUID prefix) for sessions
        indices = _find_session_info_by_uuid(session_selector, session_infos)
        if not indices:
            # Fall back to numeric selector
            indices = _parse_session_selector(session_selector, len(session_infos))

        if not indices:
            console.print(f"[red]Invalid session selector: {session_selector}[/red]")
            console.print(f"[dim]Valid range: 1-{len(session_infos)}, or session UUID/prefix[/dim]")
            return 1

        # Validate indices
        invalid_indices = [i for i in indices if i < 1 or i > len(session_infos)]
        if invalid_indices:
            console.print(f"[red]Invalid session indices: {invalid_indices}[/red]")
            console.print(f"[dim]Valid range: 1-{len(session_infos)}[/dim]")
            return 1

        # Get selected session_ids (convert 1-based to 0-based)
        selected_infos = [session_infos[i - 1] for i in sorted(indices)]

        # Check that all selected sessions are tracked (exist in database)
        non_tracked = [info for info in selected_infos if info["status"] == "new"]
        if non_tracked:
            console.print("[red]Cannot generate event: some sessions are not tracked.[/red]")
            for info in non_tracked:
                console.print(f"  • {info['session_id'][:16]}... (status: new)")
            console.print(
                "[dim]Use OneContext Dashboard -> Config -> Import history to import these sessions first.[/dim]"
            )
            return 1

        # Get SessionRecord objects from database for the selected sessions
        selected_session_ids = [info["session_id"] for info in selected_infos]
        selected_sessions = db.get_sessions_by_ids(selected_session_ids)

        if not selected_sessions:
            console.print("[red]Failed to retrieve session details from database.[/red]")
            return 1

        console.print(f"\n[bold]Creating event from {len(selected_sessions)} session(s):[/bold]")
        for s in selected_sessions:
            title = s.session_title or s.id[:16]
            console.print(f"  • {title}")

        # Generate event title and description using LLM
        console.print("\n[dim]Generating event summary with LLM...[/dim]")
        title, description = _generate_event_summary_llm(selected_sessions)

        # Create event
        event_id = str(uuid.uuid4())
        now = datetime.now()

        # Calculate time range from sessions
        start_times = [s.started_at for s in selected_sessions if s.started_at]
        end_times = [s.last_activity_at for s in selected_sessions if s.last_activity_at]

        event = EventRecord(
            id=event_id,
            title=title,
            description=description,
            event_type="user",
            status="active",
            start_timestamp=min(start_times) if start_times else None,
            end_timestamp=max(end_times) if end_times else None,
            created_at=now,
            updated_at=now,
            metadata={},
            commit_hashes=[],
            created_by=config.uid,
        )

        # Save to database
        db.sync_events([event])

        # Link all selected sessions to this event
        for session in selected_sessions:
            db.link_session_to_event(event_id, session.id)

        console.print(f"\n[green]✓ Event created:[/green]")
        console.print(f"  [bold]Title:[/bold] {title}")
        console.print(
            f"  [bold]Description:[/bold] {description[:200]}{'...' if len(description) > 200 else ''}"
        )
        console.print(f"  [bold]Sessions:[/bold] {len(selected_sessions)}")
        console.print(f"  [bold]Event ID:[/bold] {event_id[:16]}...")

        return 0

    except Exception as e:
        logger.error(f"Error generating event: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1


def watcher_event_delete_command(event_selector: str) -> int:
    """
    Delete events by number or UUID.

    Args:
        event_selector: Event selector - can be:
            - Single number: "1"
            - Range: "1-5"
            - Multiple: "1,3,5"
            - UUID or prefix: "abc123de" (at least 4 hex chars)
            - "all" to delete all events

    Returns:
        int: Exit code
    """
    try:
        from ..db import get_database

        db = get_database()

        # Get all events
        all_events = db.list_events(limit=1000)
        if not all_events:
            console.print("[yellow]No events found.[/yellow]")
            return 0

        # Handle "all" selector
        if event_selector.lower() == "all":
            confirm = input(f"Delete ALL {len(all_events)} events? [y/N]: ").strip().lower()
            if confirm != "y":
                console.print("Cancelled.")
                return 0
            indices = list(range(1, len(all_events) + 1))
        else:
            # Check if selector is a UUID (or UUID prefix)
            indices = _find_event_by_uuid(event_selector, all_events)
            if not indices:
                # Fall back to numeric selector
                indices = _parse_session_selector(event_selector, len(all_events))

            if not indices:
                console.print(f"[red]Invalid event selector: {event_selector}[/red]")
                console.print(f"[dim]Valid range: 1-{len(all_events)}, or event UUID/prefix[/dim]")
                return 1

        # Validate indices
        invalid_indices = [i for i in indices if i < 1 or i > len(all_events)]
        if invalid_indices:
            console.print(f"[red]Invalid event indices: {invalid_indices}[/red]")
            console.print(f"[dim]Valid range: 1-{len(all_events)}[/dim]")
            return 1

        # Delete events
        deleted_count = 0
        for idx in sorted(indices, reverse=True):  # Delete from end to avoid index shift
            event = all_events[idx - 1]
            if db.delete_event(event.id):
                deleted_count += 1
                console.print(f"[red]Deleted:[/red] {event.title[:50]}")

        console.print(f"\n[bold]Done:[/bold] {deleted_count} event(s) deleted")
        return 0

    except Exception as e:
        logger.error(f"Error deleting events: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1


def watcher_event_show_command(event_selector: str) -> int:
    """
    Show details of a specific event, including its sessions.

    Args:
        event_selector: Event selector - can be:
            - Index number: "1"
            - UUID or prefix: "abc123de" (at least 4 hex chars)

    Returns:
        int: Exit code
    """
    try:
        from ..db import get_database

        db = get_database()

        # Get all events
        all_events = db.list_events(limit=1000)
        if not all_events:
            console.print("[yellow]No events found.[/yellow]")
            return 1

        # Check if selector is a UUID (or UUID prefix)
        indices = _find_event_by_uuid(event_selector, all_events)
        if not indices:
            # Fall back to numeric selector
            indices = _parse_session_selector(event_selector, len(all_events))

        if not indices:
            console.print(f"[red]Invalid event selector: {event_selector}[/red]")
            console.print(f"[dim]Valid range: 1-{len(all_events)}, or event UUID/prefix[/dim]")
            console.print("[dim]Use 'aline watcher event list' to see available events[/dim]")
            return 1

        if len(indices) > 1:
            console.print(
                f"[yellow]Multiple events match '{event_selector}'. Using first match.[/yellow]"
            )

        event_index = indices[0]

        # Get the selected event
        event = all_events[event_index - 1]

        generated_by = event.event_type
        if generated_by == "task":
            generated_by = "user"
        generated_by_display = {
            "user": "user",
            "preset_day": "preset(day)",
            "preset_week": "preset(week)",
        }.get(generated_by, generated_by)

        # Display event header
        console.print(f"\n[bold cyan]Event #{event_index}[/bold cyan]")
        console.print(f"  [bold]ID:[/bold] {event.id}")
        console.print(f"  [bold]Title:[/bold] {event.title}")
        console.print(f"  [bold]Generated by:[/bold] {generated_by_display}")
        console.print(f"  [bold]Created:[/bold] {event.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"  [bold]Updated:[/bold] {event.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")

        if event.description:
            console.print(f"\n[bold]Description:[/bold]")
            console.print(f"  {event.description}")

        # Display share link if available
        if event.share_url:
            console.print(f"\n[bold]Share Link:[/bold]")
            console.print(f"  {event.share_url}")

        # Display Slack message if available
        if event.slack_message:
            console.print(f"\n[bold]Slack Message:[/bold]")
            console.print(f"  {event.slack_message}")

        # Get sessions for this event
        sessions = db.get_sessions_for_event(event.id)

        if not sessions:
            console.print(f"\n[yellow]No sessions linked to this event.[/yellow]")
            return 0

        console.print(f"\n[bold]Sessions ({len(sessions)}):[/bold]")

        # Display each session with full title and summary
        for idx, s in enumerate(sessions, 1):
            # Get turn count
            turns = db.get_turns_for_session(s.id)
            turn_count = len(turns) if turns else 0
            activity = _format_relative_time(s.last_activity_at) if s.last_activity_at else "-"

            console.print(
                f"\n[bold cyan]Session #{idx}[/bold cyan]  [dim]({s.session_type or '-'}, {turn_count} turns, {activity})[/dim]"
            )

            # Full title
            title = s.session_title or "(no title)"
            console.print(f"  [bold]Title:[/bold] {title}")

            # Full summary
            if s.session_summary:
                console.print(f"  [bold]Summary:[/bold]")
                # Indent each line of summary
                for line in s.session_summary.split("\n"):
                    console.print(f"    {line}")
            else:
                console.print(f"  [bold]Summary:[/bold] [dim](no summary)[/dim]")

        console.print()

        return 0

    except Exception as e:
        logger.error(f"Error showing event: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1


def watcher_event_list_command(
    limit: int = 50, verbose: bool = True, json_output: bool = False
) -> int:
    """
    List all events from the database (verbose by default).

    Args:
        limit: Maximum number of events to show
        verbose: Show detailed information (default: True)
        json_output: Output results in JSON format

    Returns:
        int: Exit code
    """
    try:
        from ..db import get_database

        db = get_database()
        events = db.list_events(limit=limit)

        if not events:
            if json_output:
                import json

                print(json.dumps({"total": 0, "events": []}))
                return 0
            console.print("[yellow]No events found.[/yellow]")
            console.print(
                "[dim]Use 'aline watcher event generate <sessions>' to create events.[/dim]"
            )
            return 0

        # JSON output mode
        if json_output:
            import json

            json_events = []
            for idx, event in enumerate(events, 1):
                sessions = db.get_sessions_for_event(event.id)
                session_count = len(sessions)
                session_ids = [s.id for s in sessions]

                generated_by = event.event_type
                if generated_by == "task":
                    generated_by = "user"

                event_data = {
                    "index": idx,
                    "id": event.id,
                    "title": event.title,
                    "description": event.description,
                    "created_by": event.created_by,
                    "generated_by": generated_by,
                    "session_count": session_count,
                    "session_ids": session_ids,
                    "created_at": event.created_at.isoformat(),
                    "updated_at": event.updated_at.isoformat(),
                    "share_url": event.share_url,
                    "slack_message": event.slack_message,
                }
                json_events.append(event_data)

            output = {
                "total": len(events),
                "events": json_events,
            }
            print(json.dumps(output, indent=2))
            return 0

        console.print(f"\n[bold]Events ({len(events)} shown):[/bold]\n")

        # Display table (V9: includes CREATOR column)
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("CREATOR", no_wrap=True)
        table.add_column("GENERATED BY", no_wrap=True)
        table.add_column("TITLE", overflow="fold")
        table.add_column("SESSIONS", justify="right", no_wrap=True)
        table.add_column("CREATED", no_wrap=True)
        table.add_column("UPDATED", no_wrap=True)
        table.add_column("SHARE LINK", overflow="fold")

        for idx, event in enumerate(events, 1):
            # Get session count for this event
            sessions = db.get_sessions_for_event(event.id)
            session_count = len(sessions)

            created_str = event.created_at.strftime("%m-%d %H:%M")
            updated_str = event.updated_at.strftime("%m-%d %H:%M")

            # Truncate title if too long
            title = event.title or "(untitled)"
            if len(title) > 50 and not verbose:
                title = title[:47] + "..."

            generated_by = event.event_type
            if generated_by == "task":
                generated_by = "user"
            generated_by_display = {
                "user": "user",
                "preset_day": "preset(day)",
                "preset_week": "preset(week)",
            }.get(generated_by, generated_by)

            # Display share link if available
            share_link_display = ""
            if event.share_url:
                share_link_display = event.share_url
            else:
                share_link_display = "[dim]-[/dim]"

            # V18: Display created_by UID (truncate)
            creator_display = (event.created_by[:8] + "...") if event.created_by else "-"

            table.add_row(
                str(idx),
                creator_display,
                generated_by_display,
                title,
                str(session_count),
                created_str,
                updated_str,
                share_link_display,
            )

        console.print(table)
        console.print()
        console.print("[dim]Create events: aline watcher event generate <sessions>[/dim]")
        console.print()

        return 0

    except Exception as e:
        logger.error(f"Error listing events: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1


def watcher_event_revise_slack_command(
    input_json: dict,
    instruction: str,
    json_output: bool = False,
) -> int:
    """
    Revise the Slack message for an event based on user instructions.

    Args:
        input_json: JSON data from 'aline share export --json --no-preview' containing:
            - event_id: Event UUID
            - share_link: Share URL
            - event_title: Event title
            - event_summary: Event description/summary
            - slack_message: Current Slack message
            - password: Optional password
        instruction: User's revision instructions
        json_output: If True, output JSON (same format as input)

    Returns:
        int: Exit code
    """
    import json as json_module

    try:
        from ..db import get_database
        from ..llm_client import call_llm_cloud

        # Extract data from input JSON
        event_id = input_json.get("event_id")
        share_link = input_json.get("share_link")
        event_title = input_json.get("event_title")
        event_summary = input_json.get("event_summary")
        slack_message = input_json.get("slack_message")
        password = input_json.get("password")

        # Validate required fields
        if not event_id:
            if not json_output:
                console.print("[red]Error: Missing 'event_id' in input JSON[/red]")
            return 1

        if not slack_message:
            if not json_output:
                console.print("[red]Error: Missing 'slack_message' in input JSON[/red]")
            return 1

        # Load optional custom prompt
        custom_prompt = None
        prompt_path = Path.home() / ".aline" / "prompts" / "slack_share_revise.md"
        try:
            if prompt_path.exists():
                custom_prompt = prompt_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass

        # Build event context from input JSON
        context_parts = []
        if event_title:
            context_parts.append(f"Title: {event_title}")
        if event_summary:
            context_parts.append(f"Summary: {event_summary}")

        event_context = "\n".join(context_parts) if context_parts else "No additional context"

        # Call cloud LLM to revise the message
        if not json_output:
            console.print(f"→ Revising Slack message for event: [cyan]{event_title}[/cyan]")
        logger.info(f"Calling cloud LLM to revise Slack message for event {event_id}")

        try:
            model_name, result = call_llm_cloud(
                task="revise_slack_message",
                payload={
                    "event_context": event_context,
                    "current_message": slack_message,
                    "revision_instruction": instruction,
                },
                custom_prompt=custom_prompt,
                silent=True,
            )
        except Exception as e:
            if not json_output:
                console.print(f"[red]LLM call failed: {e}[/red]")
            logger.error(f"LLM invocation failed: {e}", exc_info=True)
            return 1

        if not result:
            if not json_output:
                console.print("[red]LLM did not return a response[/red]")
            logger.error("LLM returned empty response")
            return 1

        revised_message = result.get("message", "")
        if not revised_message:
            if not json_output:
                console.print("[red]LLM response missing 'message' field[/red]")
            logger.error("LLM response missing 'message' field")
            return 1

        # Update the event in the database
        db = get_database()
        db.update_event_share_metadata(event_id=event_id, slack_message=revised_message)

        # Output results
        if json_output:
            # Output same format as input, with revised slack_message
            output_data = {
                "event_id": event_id,
                "share_link": share_link,
                "event_title": event_title,
                "event_summary": event_summary,
                "slack_message": revised_message,
                "password": password,
            }
            print(json_module.dumps(output_data, ensure_ascii=False, indent=2))
        else:
            # Display the revised message
            console.print("\n" + "=" * 60)
            console.print("[bold green]REVISED SLACK MESSAGE[/bold green]")
            console.print("=" * 60)
            console.print(revised_message)
            console.print("=" * 60)
            console.print()
            console.print(f"[dim]Updated event: {event_id}[/dim]")
            console.print()

        logger.info(
            f"Successfully revised Slack message for event {event_id} ({len(revised_message)} chars)"
        )
        return 0

    except Exception as e:
        logger.error(f"Error revising Slack message: {e}", exc_info=True)
        if not json_output:
            console.print(f"[red]Error: {e}[/red]")
        return 1


def _is_session_id_like(selector: str) -> bool:
    """
    Check if a string looks like a session ID or session ID prefix.

    Supports:
        - UUID format: "abc123de-..." (hex chars and dashes)
        - Codex format: "rollout-2026-01-16T23-30-47-..." (alphanumeric with dashes)

    Args:
        selector: String to check

    Returns:
        True if it looks like a session ID (at least 4 chars, contains letters, valid chars only)
    """
    selector = selector.strip()

    # Must have at least 4 chars
    if len(selector) < 4:
        return False

    # Check if it contains only valid session ID characters (alphanumeric, dash, underscore)
    valid_chars = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_")
    if not all(c in valid_chars for c in selector):
        return False

    # Must contain at least one letter to distinguish from pure numbers like "1,2,3"
    if not any(c.isalpha() for c in selector):
        return False

    return True


def _is_uuid_like(selector: str) -> bool:
    """
    Check if a string looks like a UUID or UUID prefix.

    Deprecated: Use _is_session_id_like for broader session ID support.

    Args:
        selector: String to check

    Returns:
        True if it looks like a UUID (hex chars and dashes, at least 4 chars, contains letters)
    """
    selector = selector.strip().lower()

    # Must have at least 4 chars
    if len(selector) < 4:
        return False

    # Check if it looks like a UUID (hex chars and dashes only)
    valid_chars = set("0123456789abcdef-")
    if not all(c in valid_chars for c in selector):
        return False

    # Must contain at least one letter to distinguish from pure numbers
    if not any(c in "abcdef" for c in selector):
        return False

    return True


def _find_session_by_uuid(selector: str, session_infos: List[Dict]) -> List[int]:
    """
    Find session by session ID or session ID prefix.

    Args:
        selector: Session ID string (full or prefix, at least 4 characters)
            Supports UUID format and Codex format (rollout-...)
        session_infos: List of session info dicts with 'session_id' key

    Returns:
        List of 1-based indices matching the session ID, or empty list if not found/invalid
    """
    selector = selector.strip().lower()

    if not _is_session_id_like(selector):
        return []

    indices = []
    for i, info in enumerate(session_infos, 1):
        session_id = info.get("session_id", "").lower()
        if session_id.startswith(selector) or session_id == selector:
            indices.append(i)

    return indices


def _find_event_by_uuid(selector: str, events: List) -> List[int]:
    """
    Find event by UUID or UUID prefix.

    Args:
        selector: UUID string (full or prefix, at least 4 characters)
        events: List of event records with 'id' attribute

    Returns:
        List of 1-based indices matching the UUID, or empty list if not found/invalid
    """
    selector = selector.strip().lower()

    if not _is_uuid_like(selector):
        return []

    indices = []
    for i, event in enumerate(events, 1):
        event_id = getattr(event, "id", "").lower()
        if event_id.startswith(selector) or event_id == selector:
            indices.append(i)

    return indices


def _find_db_session_by_uuid(selector: str, sessions: List) -> List[int]:
    """
    Find database session by session ID or session ID prefix.

    Args:
        selector: Session ID string (full or prefix, at least 4 characters)
            Supports UUID format and Codex format (rollout-...)
        sessions: List of session records with 'id' attribute

    Returns:
        List of 1-based indices matching the session ID, or empty list if not found/invalid
    """
    selector = selector.strip().lower()

    if not _is_session_id_like(selector):
        return []

    indices = []
    for i, session in enumerate(sessions, 1):
        session_id = getattr(session, "id", "").lower()
        if session_id.startswith(selector) or session_id == selector:
            indices.append(i)

    return indices


def _find_session_info_by_uuid(selector: str, session_infos: List[dict]) -> List[int]:
    """
    Find session info by session ID or session ID prefix.

    Args:
        selector: Session ID string (full or prefix, at least 4 characters)
            - Single ID: "abc123de" or "rollout-2026-01-16T23"
            - Multiple IDs: "abc123de,rollout-2026-01-16T23"
        session_infos: List of session info dicts with 'session_id' key

    Returns:
        List of 1-based indices matching the session ID(s), or empty list if not found/invalid
    """
    indices = []

    # Support comma-separated multiple session IDs
    parts = [p.strip() for p in selector.split(",")]

    for part in parts:
        if not _is_session_id_like(part):
            # If any part is not session-ID-like, return empty (fall back to numeric parsing)
            if len(parts) == 1:
                return []
            # For multi-part selectors, skip invalid parts but continue
            continue

        # Case-insensitive matching for flexibility
        part_lower = part.lower()
        for i, info in enumerate(session_infos, 1):
            session_id = info.get("session_id", "").lower()
            if session_id.startswith(part_lower) or session_id == part_lower:
                if i not in indices:
                    indices.append(i)

    return sorted(indices)


def _parse_session_selector(selector: str, total_sessions: int) -> List[int]:
    """
    Parse session selector into list of indices (1-based).

    Supports:
    - Single number: "1" -> [1]
    - Range: "1-10" -> [1, 2, ..., 10]
    - Multiple: "1,3,5" -> [1, 3, 5]
    - Mixed: "1-3,5,7-9" -> [1, 2, 3, 5, 7, 8, 9]

    Returns empty list if selector doesn't look like numbers.
    """
    indices = []

    # Check if it looks like a number pattern
    if not any(c.isdigit() for c in selector):
        return []

    try:
        for part in selector.split(","):
            part = part.strip()
            if "-" in part:
                # Range: "1-10"
                start, end = part.split("-", 1)
                start_idx = int(start.strip())
                end_idx = int(end.strip())
                for i in range(start_idx, end_idx + 1):
                    if 1 <= i <= total_sessions:
                        indices.append(i)
            else:
                # Single number
                idx = int(part)
                if 1 <= idx <= total_sessions:
                    indices.append(idx)
    except ValueError:
        return []

    return sorted(set(indices))


def _create_debug_callback(debug_file: Path) -> Callable[[Dict[str, Any]], None]:
    """
    Create a debug callback that logs events to a file in pretty-printed JSON format.
    """

    def callback(payload: Dict[str, Any]) -> None:
        try:
            payload_with_ts = {"timestamp": datetime.now().isoformat(), **payload}
            with open(debug_file, "a", encoding="utf-8") as f:
                # Pretty-print JSON for readability
                f.write(json.dumps(payload_with_ts, ensure_ascii=False, default=str, indent=2))
                f.write("\n---\n")  # Separator between events
        except Exception as e:
            logger.debug(f"Debug callback error: {e}")

    return callback


def _import_single_session(
    session_file: Path,
    config: ReAlignConfig,
    force: bool = False,
    show_header: bool = True,
    debug_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    regenerate: bool = False,
    queue: bool = True,
) -> bool:
    """
    Import a single session. Returns True if successful.
    """
    import asyncio
    from realign.watcher_core import DialogueWatcher
    from ..db import get_database

    db = get_database()
    status_info = _get_session_tracking_status(session_file, config, db=db)

    if status_info["status"] == "tracked" and not force:
        if show_header:
            console.print(
                f"[yellow]Session already fully tracked ({status_info['committed_turns']} turns)[/yellow]"
            )
        return True

    if show_header:
        console.print(f"\n[bold]Importing: {session_file.name}[/bold]")
        console.print(f"  Source: {status_info['source']}, Project: {status_info['project_name']}")

    watcher = DialogueWatcher()
    project_path = watcher._extract_project_path(session_file)

    if not project_path:
        console.print("[red]  Could not determine project path[/red]")
        return False

    async def do_import():
        last_committed = db.get_max_turn_number(session_file.stem)
        # Use _get_total_turn_count which uses get_detailed_analysis()['total_turns']
        # to get actual turn count (count_complete_turns excludes last turn)
        current_count = watcher._get_total_turn_count(session_file)

        if current_count <= last_committed and not force:
            console.print("  [yellow]No new turns to import[/yellow]")
            return True

        start_turn = 1 if force else (last_committed + 1)
        imported_count = 0

        expected_turns: Optional[int] = None
        if current_count > 1000000000:
            # Timestamp-based tracking (Antigravity): Just commit the latest state/timestamp
            # Do not iterate from 1 to 1.7 billion!
            to_import = [current_count]
        else:
            to_import = range(start_turn, current_count + 1)
            expected_turns = current_count

        session_type = watcher._detect_session_type(session_file)
        for turn in to_import:
            # Log turn info to debug callback for each turn
            if debug_callback:
                debug_callback(
                    {
                        "event": "turn_info",
                        "session_id": session_file.stem,
                        "session_file": str(session_file),
                        "source": status_info["source"],
                        "project_path": str(project_path) if project_path else None,
                        "turn_number": turn,
                        "total_turns": current_count,
                    }
                )

            console.print(f"  Turn {turn}/{current_count}...", end="")
            try:
                if queue:
                    db.enqueue_turn_summary_job(
                        session_file_path=session_file,
                        workspace_path=project_path,
                        turn_number=turn,
                        session_type=session_type,
                        skip_session_summary=expected_turns is not None,
                        expected_turns=expected_turns,
                        skip_dedup=regenerate,
                    )
                    imported_count += 1
                    console.print(" [green]queued[/green]")
                else:
                    success = await watcher._do_commit(
                        project_path,
                        session_file,
                        target_turn=turn,
                        from_catchup=True,
                        debug_callback=debug_callback,
                        skip_dedup=regenerate,
                    )
                    if success:
                        imported_count += 1
                        console.print(" [green]done[/green]")
                    else:
                        console.print(" [yellow]skipped[/yellow]")
            except Exception as e:
                console.print(f" [red]failed: {e}[/red]")
                return False

        if imported_count > 0:
            label = "Queued" if queue else "Imported"
            console.print(f"  [green]{label} {imported_count} turn(s)[/green]")
        return True

    return asyncio.run(do_import())


def watcher_session_import_command(
    session_id: str,
    force: bool = False,
    debug: Optional[str] = None,
    regenerate: bool = False,
    queue: bool = True,
) -> int:
    """
    Import sessions by number, range, or session ID.

    Args:
        session_id: Session selector - can be:
            - Number: "1" (from list)
            - Range: "1-10"
            - Multiple: "1,3,5"
            - Session ID or prefix (at least 6 chars)
        force: Re-import already tracked sessions
        debug: Path to debug log file (optional)

    Returns:
        int: Exit code
    """
    try:
        config = ReAlignConfig.load()
        db = None
        try:
            from ..db import get_database

            db = get_database()
        except Exception:
            db = None

        # Create debug callback if --debug is specified
        debug_callback = None
        if debug:
            debug_path = Path(debug)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_callback = _create_debug_callback(debug_path)
            console.print(f"[dim]Debug logging to: {debug_path}[/dim]")

        sessions = find_all_active_sessions(config, project_path=None)

        # Get session infos and sort by created_at (same order as list command)
        # Use batch processing for performance (detect_turns=False for speed)
        session_infos = []
        tracked_session_ids = set()

        try:
            batch_infos = _get_session_tracking_status_batch(
                sessions, config, db=db, detect_turns=False
            )
            for info in batch_infos:
                session_infos.append(info)
                tracked_session_ids.add(info["session_id"])
        except Exception as e:
            logger.warning(f"Failed to get batch session status: {e}")

        # Include imported sessions (same as session list command)
        if db is not None:
            try:
                imported_sessions = _get_imported_sessions(db, tracked_session_ids)
                session_infos.extend(imported_sessions)
            except Exception as e:
                logger.warning(f"Failed to get imported sessions: {e}")

        if not session_infos:
            console.print("[yellow]No sessions discovered.[/yellow]")
            return 1

        session_infos.sort(key=lambda x: x["created_at"], reverse=True)

        # Check if selector is a UUID (or UUID prefix)
        indices = _find_session_by_uuid(session_id, session_infos)
        if not indices:
            # Fall back to numeric selector
            indices = _parse_session_selector(session_id, len(session_infos))

        if not indices:
            console.print(f"[red]Session not found: {session_id}[/red]")
            console.print(f"[dim]Valid range: 1-{len(session_infos)}, or session UUID/prefix[/dim]")
            console.print("[dim]Use 'aline watcher session list' to see available sessions[/dim]")
            return 1

        # Import selected sessions
        console.print(f"\n[bold]Importing {len(indices)} session(s)...[/bold]")

        success_count = 0
        fail_count = 0

        for i, idx in enumerate(indices, 1):
            info = session_infos[idx - 1]  # Convert to 0-based
            session_file = info.get("session_file")

            if not session_file:
                console.print(
                    f"\n[cyan]({i}/{len(indices)})[/cyan] #{idx} [yellow]Skipped (imported session)[/yellow]"
                )
                continue

            console.print(f"\n[cyan]({i}/{len(indices)})[/cyan] #{idx} {session_file.name}")

            if _import_single_session(
                session_file,
                config,
                force,
                show_header=False,
                debug_callback=debug_callback,
                regenerate=regenerate,
                queue=queue,
            ):
                success_count += 1
            else:
                fail_count += 1

        console.print(
            f"\n[bold]Import complete:[/bold] {success_count} succeeded, {fail_count} failed"
        )
        if queue:
            console.print(
                "[dim]Queued turn_summary jobs; run 'aline worker start' if the worker is not running.[/dim]"
            )
        return 0 if fail_count == 0 else 1

    except Exception as e:
        logger.error(f"Error importing session: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1


def _refresh_single_session(
    session_index: int,
    info: Dict,
    db,
    watcher,
    show_header: bool = True,
) -> tuple[int, int]:
    """
    Refresh a single session's turn summaries.

    Returns:
        Tuple of (success_count, fail_count)
    """
    session_id = info["session_id"]
    session_file = info["session_file"]

    if show_header:
        console.print(f"\n[bold cyan]Refreshing Session #{session_index}[/bold cyan]")
        console.print(f"  ID: {session_id}")
        console.print(f"  Source: {info['source']}")
        console.print(f"  Project: {info['project_name']}")

    # Get all turns for this session
    turns = db.get_turns_for_session(session_id)

    if not turns:
        console.print(f"  [yellow]No turns found in database, skipped[/yellow]")
        return 0, 0

    console.print(f"  Regenerating {len(turns)} turn(s)...")

    success_count = 0
    fail_count = 0

    for turn in turns:
        turn_number = turn.turn_number
        console.print(f"    Turn {turn_number}...", end="")

        try:
            # Get turn content from database
            turn_content = db.get_turn_content(turn.id)
            if not turn_content:
                console.print(" [yellow]no content, skipped[/yellow]")
                continue

            # Regenerate LLM summary
            llm_result = watcher._generate_llm_summary(
                session_file,
                turn_number=turn_number,
                turn_content=turn_content,
                user_message=turn.user_message,
                session_id=session_id,
            )

            if llm_result:
                title, model_name, description, if_last_task, satisfaction = llm_result

                # Update turn in database
                updated = db.update_turn_summary(
                    turn_id=turn.id,
                    llm_title=title,
                    llm_description=description,
                    assistant_summary=description,
                    model_name=model_name,
                    if_last_task=if_last_task,
                    satisfaction=satisfaction,
                )

                if updated:
                    console.print(f" [green]done[/green] ({model_name})")
                    success_count += 1
                else:
                    console.print(" [red]db update failed[/red]")
                    fail_count += 1
            else:
                console.print(" [red]LLM failed[/red]")
                fail_count += 1

        except Exception as e:
            console.print(f" [red]error: {e}[/red]")
            fail_count += 1
            logger.error(f"Error refreshing turn {turn_number}: {e}", exc_info=True)

    # Now regenerate session summary (force, bypass debounce)
    if success_count > 0:
        console.print(f"  Regenerating session summary...", end="")
        try:
            from ..events.session_summarizer import force_update_session_summary

            force_update_session_summary(db, session_id)
            console.print(f" [green]done[/green]")
        except Exception as e:
            console.print(f" [red]failed: {e}[/red]")
            logger.error(f"Error updating session summary: {e}", exc_info=True)

    return success_count, fail_count


def watcher_session_refresh_command(session_selector: str) -> int:
    """
    Refresh (regenerate) all turn summaries for session(s).

    This command regenerates the LLM summary for each turn in the session(s),
    then regenerates the session-level summary. It has highest priority and
    ignores debounce settings.

    Args:
        session_selector: Session selector - can be:
            - Single number: "1"
            - Range: "1-5"
            - Multiple: "1,3,5"
            - Mixed: "1-3,5,7-9"
            - UUID or prefix: "abc123de" (at least 4 hex chars)

    Returns:
        int: Exit code
    """
    try:
        config = ReAlignConfig.load()
        sessions = find_all_active_sessions(config, project_path=None)

        # Get database connection
        db = None
        try:
            env_db_path = os.getenv("REALIGN_SQLITE_DB_PATH") or os.getenv("REALIGN_DB_PATH")
            resolved_db_path = Path(env_db_path or config.sqlite_db_path).expanduser()
            if resolved_db_path.exists():
                from ..db import get_database

                db = get_database()
        except Exception as e:
            logger.debug(f"Failed to load database: {e}")

        # Get session infos and sort by created_at (same order as list command)
        # Use batch processing for performance (detect_turns=False for speed)
        session_infos = []
        tracked_session_ids = set()

        try:
            batch_infos = _get_session_tracking_status_batch(
                sessions, config, db=db, detect_turns=False
            )
            for info in batch_infos:
                session_infos.append(info)
                tracked_session_ids.add(info["session_id"])
        except Exception as e:
            logger.warning(f"Failed to get batch session status: {e}")

        # Include imported sessions (same as session list command)
        if db is not None:
            try:
                imported_sessions = _get_imported_sessions(db, tracked_session_ids)
                session_infos.extend(imported_sessions)
            except Exception as e:
                logger.warning(f"Failed to get imported sessions: {e}")

        if not session_infos:
            console.print("[yellow]No sessions discovered.[/yellow]")
            return 1

        session_infos.sort(key=lambda x: x["created_at"], reverse=True)

        # Check if selector is a UUID (or UUID prefix)
        indices = _find_session_by_uuid(session_selector, session_infos)
        if not indices:
            # Fall back to numeric selector
            indices = _parse_session_selector(session_selector, len(session_infos))

        if not indices:
            console.print(f"[red]Invalid session selector: {session_selector}[/red]")
            console.print(f"[dim]Valid range: 1-{len(session_infos)}, or session UUID/prefix[/dim]")
            console.print("[dim]Examples: 1, 1-5, 1,3,5-7, abc123de, abc123de,def456gh[/dim]")
            return 1

        # Create watcher instance for LLM summary generation
        from realign.watcher_core import DialogueWatcher

        watcher = DialogueWatcher()

        console.print(f"\n[bold]Refreshing {len(indices)} session(s)...[/bold]")

        total_success = 0
        total_fail = 0
        sessions_processed = 0

        for i, idx in enumerate(indices, 1):
            info = session_infos[idx - 1]  # Convert to 0-based

            if len(indices) > 1:
                console.print(f"\n[cyan]({i}/{len(indices)})[/cyan] Session #{idx}")

            success, fail = _refresh_single_session(
                session_index=idx,
                info=info,
                db=db,
                watcher=watcher,
                show_header=(len(indices) == 1),
            )

            total_success += success
            total_fail += fail
            sessions_processed += 1

        console.print(
            f"\n[bold]Refresh complete:[/bold] {sessions_processed} session(s), {total_success} turns succeeded, {total_fail} failed"
        )
        console.print()
        return 0 if total_fail == 0 else 1

    except Exception as e:
        logger.error(f"Error refreshing session: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1


def watcher_session_show_command(
    session_selector: str,
    json_output: bool = False,
) -> int:
    """
    Show all turns for a specific session by index or UUID.

    Args:
        session_selector: Session selector - can be:
            - Index number: "1"
            - UUID or prefix: "abc123de" (at least 4 hex chars)

    Returns:
        int: Exit code
    """
    try:
        config = ReAlignConfig.load()
        sessions = find_all_active_sessions(config, project_path=None)

        # Get database connection
        db = None
        try:
            env_db_path = os.getenv("REALIGN_SQLITE_DB_PATH") or os.getenv("REALIGN_DB_PATH")
            resolved_db_path = Path(env_db_path or config.sqlite_db_path).expanduser()
            if resolved_db_path.exists():
                from ..db import get_database

                db = get_database()
        except Exception as e:
            logger.debug(f"Failed to load database: {e}")

        if db is None:
            console.print(f"[red]Database not available[/red]")
            return 1

        # Get session infos and sort by created_at (same order as list command)
        # Use batch processing for performance (detect_turns=False for speed)
        session_infos = []
        tracked_session_ids = set()

        try:
            batch_infos = _get_session_tracking_status_batch(
                sessions, config, db=db, detect_turns=False
            )
            for info in batch_infos:
                session_infos.append(info)
                tracked_session_ids.add(info["session_id"])
        except Exception as e:
            logger.warning(f"Failed to get batch session status: {e}")

        # Include imported sessions (same as session list command)
        try:
            imported_sessions = _get_imported_sessions(db, tracked_session_ids)
            session_infos.extend(imported_sessions)
        except Exception as e:
            logger.warning(f"Failed to get imported sessions: {e}")

        if not session_infos:
            console.print("[yellow]No sessions discovered.[/yellow]")
            return 1

        session_infos.sort(key=lambda x: x["created_at"], reverse=True)

        # Check if selector is a UUID (or UUID prefix)
        indices = _find_session_by_uuid(session_selector, session_infos)
        if not indices:
            # Fall back to numeric selector
            indices = _parse_session_selector(session_selector, len(session_infos))

        if not indices:
            console.print(f"[red]Invalid session selector: {session_selector}[/red]")
            console.print(f"[dim]Valid range: 1-{len(session_infos)}, or session UUID/prefix[/dim]")
            console.print("[dim]Use 'aline watcher session list' to see available sessions[/dim]")
            return 1

        if len(indices) > 1:
            console.print(
                f"[yellow]Multiple sessions match '{session_selector}'. Using first match.[/yellow]"
            )

        session_index = indices[0]

        # Get the selected session
        info = session_infos[session_index - 1]
        session_id = info["session_id"]

        # Get session info from database
        session_record = db.get_session_by_id(session_id)
        conn = db._get_connection()

        # Get all turns for this session
        try:
            turns = list(
                conn.execute(
                    """
                SELECT turn_number, llm_title, temp_title, timestamp, user_message, assistant_summary
                FROM turns
                WHERE session_id = ?
                ORDER BY turn_number ASC
            """,
                    (session_id,),
                )
            )
        except sqlite3.OperationalError:
            turns = list(
                conn.execute(
                    """
                SELECT turn_number, llm_title, timestamp, user_message, assistant_summary
                FROM turns
                WHERE session_id = ?
                ORDER BY turn_number ASC
            """,
                    (session_id,),
                )
            )

        # JSON output mode
        if json_output:
            import json

            # Format source type
            source_map = {
                "claude": "Claude Code",
                "codex": "Codex",
                "gemini": "Gemini",
                }
            source = source_map.get(info["source"], info["source"])

            # Build turns data
            turns_data = []
            for t in turns:
                keys = t.keys() if hasattr(t, "keys") else []
                turn_num = t["turn_number"] if "turn_number" in keys else t[0]
                title = (t["llm_title"] if "llm_title" in keys else t[1]) or "(no title)"
                temp_title = t["temp_title"] if "temp_title" in keys else None
                timestamp = (t["timestamp"] if "timestamp" in keys else t[2]) or "-"

                # Format timestamp as ISO format
                time_str = "-"
                if timestamp and timestamp != "-":
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.isoformat()
                    except:
                        time_str = timestamp

                turns_data.append(
                    {
                        "turn_number": turn_num,
                        "title": title,
                        "temp_title": temp_title or "",
                        "timestamp": time_str,
                        "user_message": (t["user_message"] if "user_message" in keys else t[3])
                        or "",
                        "assistant_summary": (
                            t["assistant_summary"] if "assistant_summary" in keys else t[4]
                        )
                        or "",
                    }
                )

            # Build session data
            session_data = {
                "index": session_index,
                "session_id": session_id,
                "source": source,
                "project_name": info["project_name"],
                "created_at": info["created_at"].isoformat(),
                "last_activity": info["last_activity"].isoformat(),
                "total_turns": info["total_turns"],
                "committed_turns": info["committed_turns"],
                "session_title": session_record.session_title if session_record else None,
                "session_summary": session_record.session_summary if session_record else None,
                "summary_status": (
                    getattr(session_record, "summary_status", None) if session_record else None
                ),
                "turns": turns_data,
            }

            print(json.dumps(session_data, indent=2, ensure_ascii=False))
            return 0

        # Display session header
        console.print(f"\n[bold cyan]Session #{session_index}[/bold cyan]")
        console.print(f"  ID: {session_id}")
        console.print(f"  Source: {info['source']}")
        console.print(f"  Project: {info['project_name']}")
        console.print(f"  Created: {info['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"  Last Activity: {info['last_activity'].strftime('%Y-%m-%d %H:%M:%S')}")

        if session_record:
            summary_status = getattr(session_record, "summary_status", None)
            summary_locked_until = getattr(session_record, "summary_locked_until", None)
            summary_error = getattr(session_record, "summary_error", None)
            if summary_status:
                console.print(f"  Summary status: {summary_status}")
            if summary_locked_until:
                console.print(f"  Summary locked until: {summary_locked_until}")
            if summary_error:
                console.print(f"  Summary error: {str(summary_error)[:200]}")

        if session_record and session_record.session_title:
            console.print(f"  Title: {session_record.session_title}")
        if session_record and session_record.session_summary:
            console.print(f"  Summary: {session_record.session_summary}")

        if not turns:
            console.print(f"\n[yellow]No turns found for this session in database.[/yellow]")
            console.print(
                f"[dim]Total turns in file: {info['total_turns']}, Committed: {info['committed_turns']}[/dim]"
            )
            if info["committed_turns"] == 0:
                console.print(
                    "[dim]Import this session from OneContext Dashboard -> Config -> Import history[/dim]"
                )
            return 0

        console.print(f"\n[bold]Turns ({len(turns)} total)[/bold]\n")

        # Format source type
        source_map = {
            "claude": "Claude Code",
            "codex": "Codex",
            "gemini": "Gemini",
        }
        source = source_map.get(info["source"], info["source"])

        # Format session name for display
        if len(session_id) > 23:
            session_name = session_id[:10] + "..." + session_id[-10:]
        else:
            session_name = session_id

        workspace = info["project_name"]

        for t in turns:
            keys = t.keys() if hasattr(t, "keys") else []
            turn_num = t["turn_number"] if "turn_number" in keys else t[0]
            title = (t["llm_title"] if "llm_title" in keys else t[1]) or "(no title)"
            temp_title = t["temp_title"] if "temp_title" in keys else None
            timestamp = (t["timestamp"] if "timestamp" in keys else t[2]) or "-"

            # Format timestamp as absolute time
            time_str = "-"
            if timestamp and timestamp != "-":
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    time_str = timestamp

            # First line: session info with turn number
            console.print(
                f"  {session_name} | {source} | {workspace} | Turn#{turn_num} | {time_str}"
            )

            # Second line: full llm_title with indentation and proper wrapping
            indent = "    "
            wrapped = textwrap.fill(
                title, width=80, initial_indent=indent, subsequent_indent=indent
            )
            console.print(f"[dim]{wrapped}[/dim]")
            if temp_title and temp_title != title:
                wrapped_temp = textwrap.fill(
                    f"temp: {temp_title}",
                    width=80,
                    initial_indent=indent,
                    subsequent_indent=indent,
                )
                console.print(f"[dim]{wrapped_temp}[/dim]")
            console.print()  # Empty line between turns

        return 0

    except Exception as e:
        logger.error(f"Error showing session: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1


def watcher_llm_command(
    watch: bool = False,
    include_expired: bool = False,
    verbose: bool = False,
) -> int:
    """
    Monitor lock operations in real-time.

    This command displays the most recent lock operations from the lock log file.
    The log file (~/.aline/.logs/watcher_lock.log) records all lock acquire/release
    operations performed by the watcher.

    Args:
        watch: If True, refresh display every 1 second (default mode)
        include_expired: Deprecated - kept for compatibility
        verbose: Deprecated - kept for compatibility

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        log_file = Path.home() / ".aline/.logs/watcher_lock.log"

        def display_log():
            """Display recent lock operations from log file."""
            # Clear screen if in watch mode
            if watch:
                console.clear()

            # Header
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            console.print(f"\n[bold cyan]Lock Operations Monitor[/bold cyan] - {timestamp}")
            console.print(f"[dim]Log file: {log_file}[/dim]\n")

            # Check if log file exists
            if not log_file.exists():
                console.print("[yellow]No lock operations recorded yet.[/yellow]")
                console.print(
                    "[dim]The log file will be created when the watcher acquires or releases locks.[/dim]"
                )
                return

            # Read last 10 lines from log file
            try:
                with open(log_file, "r") as f:
                    lines = f.readlines()

                if not lines:
                    console.print("[yellow]No lock operations recorded yet.[/yellow]")
                    return

                # Get last 10 lines
                recent_lines = lines[-10:]

                console.print(
                    f"[bold]Recent Lock Operations[/bold] (showing last {len(recent_lines)} of {len(lines)} total)\n"
                )

                # Display each line with simple formatting
                for line in recent_lines:
                    line = line.strip()
                    if not line or not line.startswith("["):
                        continue

                    # Parse log line
                    # Format: [timestamp] OPERATION STATUS TYPE LOCK_DISPLAY owner=... ttl=...
                    try:
                        # Extract timestamp
                        ts_end = line.index("]")
                        timestamp_str = line[1:ts_end]
                        rest = line[ts_end + 2 :]  # Skip "] "

                        # Split first 3 parts (operation, status, type)
                        parts = rest.split(None, 3)  # Split on whitespace, max 4 parts
                        if len(parts) < 4:
                            continue

                        operation = parts[0]
                        status = parts[1]
                        lock_type = parts[2]
                        remainder = parts[3]  # Everything else

                        # Split lock display and details (owner, ttl)
                        if "owner=" in remainder:
                            detail_start = remainder.index("owner=")
                            lock_display = remainder[:detail_start].strip()
                            details = remainder[detail_start:]
                        else:
                            lock_display = remainder.strip()
                            details = ""

                        # Colorize operation
                        if operation == "ACQUIRE":
                            op_colored = "[green]ACQ[/green]"
                        elif operation == "RELEASE":
                            op_colored = "[blue]REL[/blue]"
                        else:
                            op_colored = operation[:3]

                        # Colorize status
                        if status == "SUCCESS":
                            status_colored = "[green]✓[/green]"
                        else:
                            status_colored = "[red]✗[/red]"

                        # Extract time only from timestamp (HH:MM:SS)
                        time_only = (
                            timestamp_str.split()[-1] if " " in timestamp_str else timestamp_str
                        )

                        # Format and print line
                        console.print(
                            f"[dim]{time_only}[/dim] {op_colored} {status_colored} "
                            f"[cyan]{lock_type:<16}[/cyan] {lock_display:<30} [dim]{details}[/dim]"
                        )
                    except (ValueError, IndexError) as e:
                        # Skip malformed lines
                        logger.debug(f"Failed to parse log line: {line}, error: {e}")
                        continue

            except Exception as e:
                console.print(f"[red]Error reading log file: {e}[/red]")
                return

        # Main loop
        if watch:
            try:
                while True:
                    display_log()
                    console.print(
                        f"\n[dim]Auto-refreshing every 1 second. Press Ctrl+C to exit[/dim]"
                    )
                    time.sleep(1)
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopped monitoring.[/yellow]")
                return 0
        else:
            display_log()
            return 0

    except Exception as e:
        logger.error(f"Error monitoring locks: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 0:
        return "0s"

    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def _format_lock_key(lock_key: str) -> str:
    """Format lock key for better readability."""
    parts = lock_key.split(":")
    if len(parts) < 2:
        return lock_key

    lock_type = parts[0]

    if lock_type == "turn_process":
        # turn_process:session_id:turn_number
        if len(parts) >= 3:
            session_id = parts[1][:8] + "..." if len(parts[1]) > 8 else parts[1]
            turn_num = parts[2]
            return f"Turn #{turn_num} (session: {session_id})"
    elif lock_type == "commit_pipeline":
        # commit_pipeline:/path/to/project
        project_path = ":".join(parts[1:])
        project_name = Path(project_path).name if project_path else "unknown"
        return f"Commit: {project_name}"
    elif lock_type == "session_summary":
        # session_summary:session_id
        if len(parts) >= 2:
            session_id = parts[1][:8] + "..." if len(parts[1]) > 8 else parts[1]
            return f"Summary: {session_id}"
    elif lock_type == "event_summary":
        # event_summary:event_id
        if len(parts) >= 2:
            event_id = parts[1][:8] + "..." if len(parts[1]) > 8 else parts[1]
            return f"Event: {event_id}"

    return lock_key


def _shorten_owner(owner: str) -> str:
    """Shorten owner string for display."""
    # Owner format: prefix:hostname:pid:uuid
    parts = owner.split(":")
    if len(parts) >= 3:
        prefix = parts[0]
        hostname = parts[1]
        pid = parts[2]
        return f"{prefix}@{hostname}:{pid}"
    return owner


def watcher_session_delete_command(
    session_selector: str,
    force: bool = False,
) -> int:
    """
    Delete a session from the database.

    Args:
        session_selector: Session selector - can be:
            - Index number: "1"
            - UUID or prefix: "abc123de" (at least 4 hex chars)
        force: Skip confirmation prompt

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        config = ReAlignConfig.load()
        sessions = find_all_active_sessions(config, project_path=None)

        # Get database connection
        db = None
        try:
            env_db_path = os.getenv("REALIGN_SQLITE_DB_PATH") or os.getenv("REALIGN_DB_PATH")
            resolved_db_path = Path(env_db_path or config.sqlite_db_path).expanduser()
            if resolved_db_path.exists():
                from ..db import get_database

                db = get_database()
        except Exception as e:
            logger.debug(f"Failed to load database: {e}")

        if db is None:
            console.print("[red]Database not available[/red]")
            return 1

        # Get session infos and sort by created_at (same order as list command)
        session_infos = []
        tracked_session_ids = set()

        try:
            batch_infos = _get_session_tracking_status_batch(
                sessions, config, db=db, detect_turns=False
            )
            for info in batch_infos:
                session_infos.append(info)
                tracked_session_ids.add(info["session_id"])
        except Exception as e:
            logger.warning(f"Failed to get batch session status: {e}")

        # Include imported sessions (same as session list command)
        try:
            imported_sessions = _get_imported_sessions(db, tracked_session_ids)
            session_infos.extend(imported_sessions)
        except Exception as e:
            logger.warning(f"Failed to get imported sessions: {e}")

        if not session_infos:
            console.print("[yellow]No sessions discovered.[/yellow]")
            return 1

        session_infos.sort(key=lambda x: x["created_at"], reverse=True)

        # Check if selector is a UUID (or UUID prefix)
        indices = _find_session_by_uuid(session_selector, session_infos)
        if not indices:
            # Fall back to numeric selector
            indices = _parse_session_selector(session_selector, len(session_infos))

        if not indices:
            console.print(f"[red]Invalid session selector: {session_selector}[/red]")
            console.print(f"[dim]Valid range: 1-{len(session_infos)}, or session UUID/prefix[/dim]")
            console.print("[dim]Use 'aline watcher session list' to see available sessions[/dim]")
            return 1

        if len(indices) > 1:
            console.print(
                f"[yellow]Multiple sessions match '{session_selector}'. Please be more specific.[/yellow]"
            )
            return 1

        session_index = indices[0]

        # Get the selected session
        info = session_infos[session_index - 1]
        session_id = info["session_id"]

        # Get session info from database
        session_record = db.get_session_by_id(session_id)

        # Display session info
        session_title = info.get("session_title") or "(untitled)"
        source_map = {
            "claude": "Claude Code",
            "codex": "Codex",
            "gemini": "Gemini",
        }
        source = source_map.get(info.get("source", ""), info.get("source", "unknown"))

        console.print(f"\n[bold]Session to delete:[/bold]")
        console.print(f"  ID: [cyan]{session_id[:12]}...[/cyan]")
        console.print(f"  Title: {session_title}")
        console.print(f"  Source: {source}")
        if session_record:
            # Get turn count
            conn = db._get_connection()
            turn_count = conn.execute(
                "SELECT COUNT(*) FROM turns WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
            console.print(f"  Turns in DB: {turn_count}")

        # Confirm deletion
        if not force:
            console.print(
                "\n[yellow]This will permanently delete the session and all its turns from the database.[/yellow]"
            )
            confirm = input("Are you sure? (y/N): ").strip().lower()
            if confirm != "y":
                console.print("[dim]Deletion cancelled.[/dim]")
                return 0

        # Delete the session
        success = db.delete_session(session_id)

        if success:
            console.print(f"[green]✓ Session deleted successfully[/green]")
            return 0
        else:
            console.print(f"[red]Failed to delete session (not found in database)[/red]")
            return 1

    except Exception as e:
        logger.error(f"Error deleting session: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1
