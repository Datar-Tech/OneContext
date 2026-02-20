"""tmux integration for the Aline dashboard.

This is an optional runtime dependency:
- If tmux isn't installed, the dashboard runs normally without terminal controls.
- If tmux is installed, `aline dashboard` can bootstrap into a managed tmux session.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

from ..logging_config import setup_logger
from .branding import BRANDING

logger = setup_logger("realign.dashboard.tmux", "dashboard.log")

OUTER_SESSION = "aline"
OUTER_WINDOW = "dashboard"
OUTER_SOCKET = "aline_dash"
INNER_SOCKET = "aline_term"
INNER_SESSION = "term"
MANAGED_ENV = "ALINE_TMUX_MANAGED"
ENV_TERMINAL_ID = "ALINE_TERMINAL_ID"
ENV_TERMINAL_PROVIDER = "ALINE_TERMINAL_PROVIDER"
ENV_INNER_SOCKET = "ALINE_INNER_TMUX_SOCKET"
ENV_INNER_SESSION = "ALINE_INNER_TMUX_SESSION"

OPT_TERMINAL_ID = "@aline_terminal_id"
OPT_PROVIDER = "@aline_provider"
OPT_SESSION_TYPE = "@aline_session_type"
OPT_SESSION_ID = "@aline_session_id"
OPT_TRANSCRIPT_PATH = "@aline_transcript_path"
OPT_CONTEXT_ID = "@aline_context_id"
OPT_ATTENTION = "@aline_attention"
OPT_CREATED_AT = "@aline_created_at"
OPT_NO_TRACK = "@aline_no_track"

# Default outer layout preferences (tmux mode).
# Note: tmux "pixels" are terminal cell columns; Textual has no notion of pixels.
DEFAULT_DASHBOARD_PANE_WIDTH_COLS = 80
MIN_DASHBOARD_PANE_WIDTH_COLS = 68
_STATE_BORDER_RESIZE_ENABLED_KEY = "tmux_border_resize_enabled"
_STATE_DASHBOARD_PANE_WIDTH_COLS_KEY = "tmux_dashboard_pane_width_cols"


@dataclass(frozen=True)
class InnerWindow:
    window_id: str
    window_name: str
    active: bool
    terminal_id: str | None = None
    provider: str | None = None
    session_type: str | None = None
    session_id: str | None = None
    transcript_path: str | None = None
    context_id: str | None = None
    attention: str | None = None  # "permission_request", "stop", or None
    created_at: float | None = None  # Unix timestamp when window was created
    no_track: bool = False  # Whether tracking is disabled for this terminal
    pane_pid: int | None = None  # PID of the initial process in the pane
    pane_current_command: str | None = None  # Foreground process in the pane
    pane_tty: str | None = None  # Controlling TTY for processes in the pane


def tmux_available() -> bool:
    return shutil.which("tmux") is not None


def in_tmux() -> bool:
    return bool(os.environ.get("TMUX"))


def managed_env_enabled() -> bool:
    return os.environ.get(MANAGED_ENV) == "1"


_TMUX_VERSION_RE = re.compile(r"tmux\s+(\d+)\.(\d+)")
_TMUX_NO_SERVER_RE = re.compile(r"no server running on\s+(.+)$", re.MULTILINE)


def tmux_version() -> tuple[int, int] | None:
    if not tmux_available():
        return None
    try:
        proc = subprocess.run(["tmux", "-V"], text=True, capture_output=True, check=False)
    except OSError:
        return None
    match = _TMUX_VERSION_RE.search((proc.stdout or "") + " " + (proc.stderr or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _run_tmux(
    args: Sequence[str], *, capture: bool = False, timeout_s: float | None = None
) -> subprocess.CompletedProcess[str]:
    # Always capture stdout/stderr so tmux errors never corrupt the Textual TUI.
    # Callers that need output already read .stdout/.stderr; callers that don't can ignore it.
    kwargs: dict[str, object] = {
        "text": True,
        "capture_output": True,
        "check": False,
    }
    # Keep keyword list minimal for test fakes that don't accept extra kwargs.
    if timeout_s is not None:
        kwargs["timeout"] = timeout_s
    return subprocess.run(["tmux", *args], **kwargs)  # type: ignore[arg-type]


def _run_outer_tmux(
    args: Sequence[str], *, capture: bool = False, timeout_s: float | None = None
) -> subprocess.CompletedProcess[str]:
    """Run tmux commands against the dedicated outer server socket."""
    return _run_tmux(["-L", OUTER_SOCKET, *args], capture=capture, timeout_s=timeout_s)


def _run_inner_tmux(
    args: Sequence[str], *, capture: bool = False, timeout_s: float | None = None
) -> subprocess.CompletedProcess[str]:
    return _run_tmux(["-L", INNER_SOCKET, *args], capture=capture, timeout_s=timeout_s)


def _python_dashboard_command() -> str:
    # Use the current interpreter for predictable environments (venv, editable installs).
    python_cmd = shlex.join(
        [
            sys.executable,
            "-c",
            "from realign.dashboard.app import run_dashboard; run_dashboard()",
        ]
    )
    # After the dashboard process exits (normal exit, crash, Ctrl+C, logout), kill the
    # outer tmux session.  This cascades: session dies → `tmux attach` in CLI returns →
    # Terminal.app closes.  The inner tmux sessions (aline_term) are on a separate
    # server and are NOT affected.
    kill_cmd = f"tmux -L {OUTER_SOCKET} kill-session -t {OUTER_SESSION} 2>/dev/null"
    return f"{MANAGED_ENV}=1 {python_cmd} ; {kill_cmd}"


def _parse_lines(output: str) -> list[str]:
    return [line.strip() for line in output.splitlines() if line.strip()]


def _unique_name(existing: Iterable[str], base: str) -> str:
    if base not in existing:
        return base
    idx = 2
    while f"{base}-{idx}" in existing:
        idx += 1
    return f"{base}-{idx}"


def _user_shell_name() -> str:
    """Return a display-friendly name for the user's default shell."""
    shell = os.environ.get("SHELL", "/bin/sh")
    return os.path.basename(shell)


def shell_run_and_keep_open(command: str) -> str:
    """Run a command via the user's login shell, then keep an interactive shell open."""
    shell = os.environ.get("SHELL", "/bin/sh")
    script = f"{command}; exec {shell} -l"
    return shlex.join([shell, "-lc", script])


# Backward-compatible alias
zsh_run_and_keep_open = shell_run_and_keep_open


def new_terminal_id() -> str:
    return str(uuid.uuid4())


def shell_command_with_env(command: str, env: dict[str, str]) -> str:
    if not env:
        return command
    # Important: callers often pass compound shell commands like `cd ... && zsh -lc ...`.
    # `VAR=... cd ... && ...` only applies VAR to the first command (`cd`) in POSIX sh.
    # Wrap in a subshell so env vars apply to the entire script.
    assignments = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    return f"env {assignments} sh -lc {shlex.quote(command)}"


_SESSION_ID_FROM_TRANSCRIPT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,}$")


def _session_id_from_transcript_path(transcript_path: str | None) -> str | None:
    raw = (transcript_path or "").strip()
    if not raw:
        return None
    try:
        path = Path(raw)
    except Exception:
        return None
    if path.suffix != ".jsonl":
        return None
    stem = (path.stem or "").strip()
    if not stem:
        return None
    # Heuristic guard: avoid overwriting with generic filenames like "transcript.jsonl".
    if not _SESSION_ID_FROM_TRANSCRIPT_RE.fullmatch(stem):
        return None
    if not any(ch.isdigit() for ch in stem):
        return None
    return stem


def _load_terminal_state_from_db() -> dict[str, dict[str, str]]:
    """Load terminal state from database (best-effort)."""
    import time as _time

    t0 = _time.time()
    try:
        from ..db import get_database

        t1 = _time.time()
        db = get_database(read_only=True)
        try:
            logger.debug(
                f"[PERF] _load_terminal_state_from_db get_database: {_time.time() - t1:.3f}s"
            )
            t2 = _time.time()
            agents = db.list_agents(status="active", limit=100)
            logger.debug(
                f"[PERF] _load_terminal_state_from_db list_agents: {_time.time() - t2:.3f}s"
            )
        finally:
            try:
                db.close()
            except Exception:
                pass

        out: dict[str, dict[str, str]] = {}
        for agent in agents:
            data: dict[str, str] = {}
            if agent.provider:
                data["provider"] = agent.provider
            if agent.session_type:
                data["session_type"] = agent.session_type
            if agent.session_id:
                data["session_id"] = agent.session_id
            if agent.transcript_path:
                data["transcript_path"] = agent.transcript_path
            if agent.cwd:
                data["cwd"] = agent.cwd
            if agent.project_dir:
                data["project_dir"] = agent.project_dir
            if agent.source:
                data["source"] = agent.source
            if agent.context_id:
                data["context_id"] = agent.context_id
            if agent.attention:
                data["attention"] = agent.attention
            out[agent.id] = data
        return out
    except Exception:
        return {}


def _load_terminal_state_from_json() -> dict[str, dict[str, str]]:
    """Load terminal state from JSON file (fallback)."""
    try:
        path = Path.home() / ".aline" / "terminal.json"
        if not path.exists():
            return {}
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        terminals = payload.get("terminals", {}) if isinstance(payload, dict) else {}
        if not isinstance(terminals, dict):
            return {}
        out: dict[str, dict[str, str]] = {}
        for terminal_id, data in terminals.items():
            if isinstance(terminal_id, str) and isinstance(data, dict):
                out[terminal_id] = {str(k): str(v) for k, v in data.items() if v is not None}
        return out
    except Exception:
        return {}


_TERMINAL_STATE_CACHE_LOCK = threading.Lock()
_TERMINAL_STATE_CACHE: dict[str, dict[str, str]] | None = None
_TERMINAL_STATE_CACHE_AT: float = 0.0
_TERMINAL_STATE_CACHE_TTL_S: float = 1.5


def _load_terminal_state() -> dict[str, dict[str, str]]:
    """Load terminal state.

    Priority:
    1. SQLite database (primary storage, V15+)
    2. ~/.aline/terminal.json (fallback for backward compatibility)

    Merges both sources, with DB taking precedence.
    """
    global _TERMINAL_STATE_CACHE, _TERMINAL_STATE_CACHE_AT
    now = time.monotonic()
    with _TERMINAL_STATE_CACHE_LOCK:
        cache = _TERMINAL_STATE_CACHE
        if cache is not None and (now - _TERMINAL_STATE_CACHE_AT) <= _TERMINAL_STATE_CACHE_TTL_S:
            return dict(cache)

    # Phase 1: Load from database
    db_state = _load_terminal_state_from_db()

    # Phase 2: Load from JSON as fallback
    json_state = _load_terminal_state_from_json()

    # Merge: DB takes precedence, JSON provides fallback for entries not in DB
    result = dict(json_state)
    result.update(db_state)

    with _TERMINAL_STATE_CACHE_LOCK:
        _TERMINAL_STATE_CACHE = dict(result)
        _TERMINAL_STATE_CACHE_AT = time.monotonic()

    return result


def _aline_tmux_conf_path() -> Path:
    return Path.home() / ".aline" / "tmux" / "tmux.conf"


def _source_aline_tmux_config(run_fn) -> None:  # type: ignore[no-untyped-def]
    """Best-effort source ~/.aline/tmux/tmux.conf if present."""
    try:
        # Ensure the config exists and is parseable.
        # Users may run `aline dashboard` before `aline init`, or have older auto-generated configs
        # that included unquoted `#` bindings (tmux treats `#` as a comment delimiter).
        try:
            from ..commands.init import _initialize_tmux_config

            conf = _initialize_tmux_config()
        except Exception:
            conf = _aline_tmux_conf_path()

        if conf.exists():
            run_fn(["source-file", str(conf)])
    except Exception:
        return


_DID_WARN_AUTOMATION = False


def _warn_automation_blocked(*, terminal_app: str, detail: str | None = None) -> None:
    """Best-effort hint when macOS Automation blocks AppleScript."""
    global _DID_WARN_AUTOMATION
    if _DID_WARN_AUTOMATION:
        return
    _DID_WARN_AUTOMATION = True

    msg = (
        "[OneContext] Unable to auto-maximize terminal window: macOS Automation permission likely "
        f"blocked AppleScript for {terminal_app}.\n"
        "Enable it at System Settings → Privacy & Security → Automation, then allow "
        f"your terminal (or osascript) to control {terminal_app}.\n"
    )
    if detail:
        msg += f"Detail: {detail.strip()}\n"
    try:
        sys.stderr.write(msg)
    except Exception:
        pass


def _terminal_app_from_env() -> str | None:
    """Detect the hosting terminal via common environment variables (no Automation needed)."""
    term_program = (os.environ.get("TERM_PROGRAM") or "").strip()
    if term_program in {"Apple_Terminal", "Terminal.app"}:
        return "Terminal"
    if term_program in {"iTerm.app", "iTerm2"} or term_program.startswith("iTerm"):
        return "iTerm2"
    return None


def _terminal_app_from_system_events() -> str | None:
    """Detect the frontmost app via System Events (may require Automation permission)."""
    try:
        detect_result = subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to name of first application process whose frontmost is true',
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None

    stderr = (detect_result.stderr or "").strip()
    if detect_result.returncode != 0 and stderr:
        _warn_automation_blocked(terminal_app="System Events", detail=stderr)
        return None

    front_app = (detect_result.stdout or "").strip()
    return front_app or None


def _maximize_terminal_window() -> None:
    """Maximize the current terminal window.

    On macOS, uses AppleScript to set the window to zoomed state.
    On Linux, tries wmctrl, xdotool, then ANSI escape sequence as fallbacks.
    """
    if sys.platform == "darwin":
        _maximize_terminal_window_macos()
    else:
        _maximize_terminal_window_linux()


def _maximize_terminal_window_macos() -> None:
    """Maximize on macOS via AppleScript."""
    try:
        front_app = _terminal_app_from_env() or _terminal_app_from_system_events() or ""

        if front_app == "Terminal":
            proc = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "Terminal" to set zoomed of front window to true',
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if proc.returncode != 0 and (proc.stderr or "").strip():
                _warn_automation_blocked(terminal_app="Terminal", detail=proc.stderr)
        elif front_app == "iTerm2":
            # iTerm2: get screen size and set window bounds
            script = (
                'tell application "iTerm2" to tell current window to '
                'set bounds to {0, 25, (do shell script "system_profiler SPDisplaysDataType | '
                "awk '/Resolution/{print $2; exit}'\") as integer, "
                '(do shell script "system_profiler SPDisplaysDataType | '
                "awk '/Resolution/{print $4; exit}'\") as integer}"
            )
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if proc.returncode != 0 and (proc.stderr or "").strip():
                _warn_automation_blocked(terminal_app="iTerm2", detail=proc.stderr)
    except Exception:
        pass  # Best-effort; don't fail if this doesn't work


def _maximize_terminal_window_linux() -> None:
    """Maximize on Linux via wmctrl, xdotool, or ANSI escape sequence."""
    import shutil

    try:
        # Method 1: wmctrl (most reliable on X11)
        if shutil.which("wmctrl"):
            proc = subprocess.run(
                ["wmctrl", "-r", ":ACTIVE:", "-b", "add,maximized_vert,maximized_horz"],
                capture_output=True,
                timeout=2,
                check=False,
            )
            if proc.returncode == 0:
                return

        # Method 2: xdotool (common on X11)
        if shutil.which("xdotool"):
            # Get active window id, then maximize it
            proc = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            wid = (proc.stdout or "").strip()
            if proc.returncode == 0 and wid:
                subprocess.run(
                    ["xdotool", "windowsize", "--sync", wid, "100%", "100%"],
                    capture_output=True,
                    timeout=2,
                    check=False,
                )
                return

        # Method 3: ANSI escape sequence (xterm-compatible terminals)
        # \e[9;1t = maximize window; works in many terminals without extra tools
        sys.stdout.write("\033[9;1t")
        sys.stdout.flush()
    except Exception:
        pass  # Best-effort; don't fail if this doesn't work


def _cleanup_stale_tmux_socket(stderr: str) -> bool:
    """Remove a stale tmux socket when the server is gone (best-effort)."""
    match = _TMUX_NO_SERVER_RE.search(stderr or "")
    if not match:
        return False
    path = match.group(1).strip()
    if not path:
        return False
    try:
        st = os.stat(path)
        if not stat.S_ISSOCK(st.st_mode):
            return False
        os.unlink(path)
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False


def bootstrap_dashboard_into_tmux(*, attach: Literal["exec", "subprocess"] = "exec") -> int | None:
    """Ensure a managed tmux session exists, then attach to it.

    This is intended to be called when *not* already inside tmux.
    """
    logger.debug("bootstrap_dashboard_into_tmux() started")
    if in_tmux():
        logger.debug("Already in tmux, skipping bootstrap")
        return None
    if not tmux_available():
        logger.debug("tmux not available, skipping bootstrap")
        return None

    # Maximize terminal window before attaching to tmux
    _maximize_terminal_window()
    logger.debug("Terminal window maximized")

    # Ensure session exists.
    has = _run_outer_tmux(["has-session", "-t", OUTER_SESSION], capture=True)
    if has.returncode != 0:
        created = _run_outer_tmux(
            ["new-session", "-d", "-s", OUTER_SESSION, "-n", OUTER_WINDOW],
            capture=True,
        )
        if created.returncode != 0 and _cleanup_stale_tmux_socket(
            (created.stderr or "") + "\n" + (has.stderr or "")
        ):
            created = _run_outer_tmux(
                ["new-session", "-d", "-s", OUTER_SESSION, "-n", OUTER_WINDOW],
                capture=True,
            )
        if created.returncode != 0:
            detail = (created.stderr or created.stdout or "").strip()
            if detail:
                sys.stderr.write(f"[OneContext] tmux bootstrap failed: {detail}\n")
            return None

    # Load Aline tmux config (clipboard bindings, etc.) into this dedicated server.
    _source_aline_tmux_config(_run_outer_tmux)

    # Enable mouse for the managed session only.
    _run_outer_tmux(["set-option", "-t", OUTER_SESSION, "mouse", "on"])
    _disable_outer_border_resize()
    try:
        from .state import set_dashboard_state_value

        set_dashboard_state_value(_STATE_BORDER_RESIZE_ENABLED_KEY, False)
    except Exception:
        pass

    # Set terminal window title so Terminal.app shows the dashboard title.
    _run_outer_tmux(["set-option", "-t", OUTER_SESSION, "set-titles", "on"])
    _run_outer_tmux(
        ["set-option", "-t", OUTER_SESSION, "set-titles-string", BRANDING.dashboard_title]
    )

    # Disable status bar for cleaner UI (Aline sessions only).
    _run_outer_tmux(["set-option", "-t", OUTER_SESSION, "status", "off"])

    # Pane border styling - use double lines for wider, more visible borders.
    # This helps users identify the resizable border area more easily and reduces
    # accidental drag-to-resize when trying to select text near the border.
    _run_outer_tmux(["set-option", "-t", OUTER_SESSION, "pane-border-lines", "double"])
    _run_outer_tmux(["set-option", "-t", OUTER_SESSION, "pane-border-style", "fg=brightblack"])
    _run_outer_tmux(["set-option", "-t", OUTER_SESSION, "pane-active-border-style", "fg=blue"])
    _run_outer_tmux(["set-option", "-t", OUTER_SESSION, "pane-border-indicators", "arrows"])

    # Ensure dashboard window exists.
    windows_out = (
        _run_outer_tmux(
            ["list-windows", "-t", OUTER_SESSION, "-F", "#{window_name}"], capture=True
        ).stdout
        or ""
    )
    windows = set(_parse_lines(windows_out))
    if OUTER_WINDOW not in windows:
        _run_outer_tmux(["new-window", "-t", OUTER_SESSION, "-n", OUTER_WINDOW])

    # Keep an already-running dashboard pane intact when attaching from another terminal.
    # Otherwise, every extra `aline` launch would kill and restart the active dashboard.
    pane_target = f"{OUTER_SESSION}:{OUTER_WINDOW}.0"
    pane_cmd_proc = _run_outer_tmux(
        ["display-message", "-p", "-t", pane_target, "#{pane_current_command}"],
        capture=True,
    )
    pane_cmd = (pane_cmd_proc.stdout or "").strip().lower() if pane_cmd_proc.returncode == 0 else ""
    if not pane_cmd.startswith("python"):
        _run_outer_tmux(
            [
                "respawn-pane",
                "-k",
                "-t",
                pane_target,
                _python_dashboard_command(),
            ]
        )
    else:
        logger.debug(
            "bootstrap_dashboard_into_tmux: dashboard pane already active (pane_current_command=%s), "
            "skip respawn",
            pane_cmd,
        )
    _run_outer_tmux(["select-window", "-t", f"{OUTER_SESSION}:{OUTER_WINDOW}"])

    # Ensure the right pane exists and is attached to the inner tmux session.
    attach_cmd = shlex.join(["tmux", "-L", INNER_SOCKET, "attach", "-t", INNER_SESSION])
    pane1_target = f"{OUTER_SESSION}:{OUTER_WINDOW}.1"
    panes_out = (
        _run_outer_tmux(
            ["list-panes", "-t", f"{OUTER_SESSION}:{OUTER_WINDOW}", "-F", "#{pane_index}"],
            capture=True,
        ).stdout
        or ""
    )
    panes = set(_parse_lines(panes_out))
    if "1" not in panes:
        split = _run_outer_tmux(
            [
                "split-window",
                "-h",
                "-p",
                "50",
                "-t",
                pane_target,
                "-d",
                attach_cmd,
            ],
            capture=True,
        )
        if split.returncode != 0:
            detail = (split.stderr or split.stdout or "").strip()
            if detail:
                logger.warning(f"bootstrap_dashboard_into_tmux split-window failed: {detail}")
    else:
        pane1_cmd_proc = _run_outer_tmux(
            ["display-message", "-p", "-t", pane1_target, "#{pane_current_command}"],
            capture=True,
        )
        pane1_cmd = (
            (pane1_cmd_proc.stdout or "").strip().lower() if pane1_cmd_proc.returncode == 0 else ""
        )
        if pane1_cmd != "tmux":
            respawn = _run_outer_tmux(
                [
                    "respawn-pane",
                    "-k",
                    "-t",
                    pane1_target,
                    attach_cmd,
                ],
                capture=True,
            )
            if respawn.returncode != 0:
                detail = (respawn.stderr or respawn.stdout or "").strip()
                if detail:
                    logger.warning(f"bootstrap_dashboard_into_tmux pane1 respawn failed: {detail}")

    # Best-effort: enforce a stable dashboard pane width if the terminal pane already exists.
    enforce_outer_dashboard_pane_width(DEFAULT_DASHBOARD_PANE_WIDTH_COLS)

    # Sanity-check before exec'ing into tmux attach. If this fails, fall back to non-tmux mode.
    ready = _run_outer_tmux(["has-session", "-t", OUTER_SESSION], capture=True)
    if ready.returncode != 0:
        detail = (ready.stderr or ready.stdout or "").strip()
        if detail:
            sys.stderr.write(f"[OneContext] tmux attach skipped: {detail}\n")
        return None

    if attach == "subprocess":
        # Important: keep the current process alive so higher-level callers can hold
        # cross-process resources (e.g. singleton locks) while tmux is attached.
        proc = subprocess.run(
            ["tmux", "-L", OUTER_SOCKET, "attach", "-t", OUTER_SESSION],
            text=True,
            capture_output=True,
            check=False,
        )
        if int(proc.returncode) != 0:
            # If attach fails (permissions, missing socket, etc.), fall back to running the
            # dashboard in the current terminal instead of exiting the CLI immediately.
            return None
        return 0

    os.execvp("tmux", ["tmux", "-L", OUTER_SOCKET, "attach", "-t", OUTER_SESSION])
    return None  # unreachable


_inner_session_configured = False


def ensure_inner_session() -> bool:
    """Ensure the inner tmux server/session exists (returns True on success).

    The full configuration (mouse, status bar, border styles, home window setup) is
    only applied once per process lifetime.  Subsequent calls just verify the session
    is still alive via a cheap ``has-session`` check.
    """
    global _inner_session_configured

    if not (tmux_available() and in_tmux() and managed_env_enabled()):
        return False

    if _run_inner_tmux(["has-session", "-t", INNER_SESSION]).returncode != 0:
        # Create a stable "home" window so user-created terminals can use names like "zsh"
        # without always becoming "zsh-2".
        if (
            _run_inner_tmux(["new-session", "-d", "-s", INNER_SESSION, "-n", "home"]).returncode
            != 0
        ):
            return False
        # Force re-configuration after creating a new session.
        _inner_session_configured = False

    if _inner_session_configured:
        return True

    # --- One-time configuration below ---

    # Ensure the default/home window stays named "home" (tmux auto-rename would otherwise
    # change it to "zsh"/"opencode" depending on the last foreground command).
    try:
        _ensure_inner_home_window()
    except Exception:
        pass

    # Dedicated inner server; safe to enable mouse globally there.
    _run_inner_tmux(["set-option", "-g", "mouse", "on"])

    # Disable status bar for cleaner UI.
    _run_inner_tmux(["set-option", "-t", INNER_SESSION, "status", "off"])

    # Pane border styling - use double lines for wider, more visible borders.
    # This helps users identify the resizable border area more easily and reduces
    # accidental drag-to-resize when trying to select text near the border.
    _run_inner_tmux(["set-option", "-g", "pane-border-lines", "double"])
    _run_inner_tmux(["set-option", "-g", "pane-border-style", "fg=brightblack"])
    _run_inner_tmux(["set-option", "-g", "pane-active-border-style", "fg=blue"])
    _run_inner_tmux(["set-option", "-g", "pane-border-indicators", "arrows"])

    _source_aline_tmux_config(_run_inner_tmux)

    _inner_session_configured = True
    return True


def _ensure_inner_home_window() -> None:
    """Ensure the inner session has a reserved, non-renaming 'home' window (best-effort)."""
    if _run_inner_tmux(["has-session", "-t", INNER_SESSION]).returncode != 0:
        return

    out = (
        _run_inner_tmux(
            [
                "list-windows",
                "-t",
                INNER_SESSION,
                "-F",
                "#{window_id}\t#{window_index}\t#{window_name}\t#{"
                + OPT_TERMINAL_ID
                + "}\t#{"
                + OPT_PROVIDER
                + "}\t#{"
                + OPT_SESSION_TYPE
                + "}\t#{"
                + OPT_CONTEXT_ID
                + "}\t#{"
                + OPT_CREATED_AT
                + "}\t#{"
                + OPT_NO_TRACK
                + "}",
            ],
            capture=True,
        ).stdout
        or ""
    )

    candidates: list[tuple[str, int, str, str, str, str, str, str, str]] = []
    for line in _parse_lines(out):
        parts = (line.split("\t", 8) + [""] * 9)[:9]
        window_id = parts[0]
        try:
            window_index = int(parts[1])
        except Exception:
            window_index = 9999
        window_name = parts[2]
        terminal_id = parts[3]
        provider = parts[4]
        session_type = parts[5]
        context_id = parts[6]
        created_at = parts[7]
        no_track = parts[8]

        # Pick an unmanaged window (the default one created by `new-session`) as "home".
        unmanaged = (
            not (terminal_id or "").strip()
            and not (provider or "").strip()
            and not (session_type or "").strip()
            and not (context_id or "").strip()
            and not (created_at or "").strip()
        )
        if unmanaged:
            candidates.append(
                (
                    window_id,
                    window_index,
                    window_name,
                    terminal_id,
                    provider,
                    session_type,
                    context_id,
                    created_at,
                    no_track,
                )
            )

    if not candidates:
        return

    # Prefer the first window (index 0) if present.
    candidates.sort(key=lambda t: t[1])
    window_id = candidates[0][0]

    # Rename to "home" and prevent tmux auto-renaming it based on foreground command.
    _run_inner_tmux(["rename-window", "-t", window_id, "home"])
    _run_inner_tmux(["set-option", "-w", "-t", window_id, "automatic-rename", "off"])
    _run_inner_tmux(["set-option", "-w", "-t", window_id, "allow-rename", "off"])

    # Mark as internal/no-track so UI can hide it.
    # NOTE: We use _run_inner_tmux directly here instead of set_inner_window_options
    # to avoid recursion: set_inner_window_options → ensure_inner_session →
    # _ensure_inner_home_window → set_inner_window_options.
    try:
        _run_inner_tmux(["set-option", "-w", "-t", window_id, OPT_NO_TRACK, "1"])
        _run_inner_tmux(["set-option", "-w", "-t", window_id, OPT_CREATED_AT, str(time.time())])
    except Exception:
        pass

    # Keep the home window alive even if its shell exits.  Without this, the home
    # window can disappear (e.g. user accidentally switches to it and types "exit"),
    # and when the last user terminal is later closed the inner session is destroyed,
    # which cascades into killing the outer right pane and crashing the dashboard.
    try:
        _run_inner_tmux(["set-option", "-w", "-t", window_id, "remain-on-exit", "on"])
    except Exception:
        pass


def ensure_right_pane(width_percent: int = 50) -> bool:
    """Create the right-side pane (terminal area) if it doesn't exist.

    Returns True if the pane exists/was created successfully.
    """
    if not ensure_inner_session():
        return False

    panes_out = (
        _run_tmux(
            [
                "list-panes",
                "-t",
                f"{OUTER_SESSION}:{OUTER_WINDOW}",
                "-F",
                "#{pane_index}",
            ],
            capture=True,
        ).stdout
        or ""
    )
    panes = _parse_lines(panes_out)
    if len(panes) >= 2:
        enforce_outer_dashboard_pane_width(DEFAULT_DASHBOARD_PANE_WIDTH_COLS)
        return True

    # Split from the dashboard pane to keep it on the left.
    attach_cmd = shlex.join(["tmux", "-L", INNER_SOCKET, "attach", "-t", INNER_SESSION])
    split = _run_tmux(
        [
            "split-window",
            "-h",
            "-p",
            str(int(width_percent)),
            "-t",
            f"{OUTER_SESSION}:{OUTER_WINDOW}.0",
            "-d",
            attach_cmd,
        ],
        capture=True,
    )
    if split.returncode != 0:
        detail = (split.stderr or split.stdout or "").strip()
        if detail:
            logger.warning(f"ensure_right_pane split-window failed: {detail}")
    if split.returncode == 0:
        enforce_outer_dashboard_pane_width(DEFAULT_DASHBOARD_PANE_WIDTH_COLS)
        return True
    return False


def ensure_right_pane_ready(width_percent: int = 50) -> bool:
    """Ensure the right pane exists and is attached to the inner tmux session."""
    try:
        ok = ensure_right_pane(width_percent)
    except TypeError:
        # Tests and some callers monkeypatch ensure_right_pane() as a no-arg lambda.
        ok = ensure_right_pane()
    if not ok:
        return False

    # Best-effort: enforce a stable dashboard pane width whenever we touch the outer layout
    # (unless user enabled drag-to-resize).
    enforce_outer_dashboard_pane_width(DEFAULT_DASHBOARD_PANE_WIDTH_COLS)

    attach_cmd = shlex.join(["tmux", "-L", INNER_SOCKET, "attach", "-t", INNER_SESSION])

    # If the right pane exists but isn't running `tmux attach`, it may look "blank" or stale.
    try:
        proc = _run_outer_tmux(
            [
                "display-message",
                "-p",
                "-t",
                f"{OUTER_SESSION}:{OUTER_WINDOW}.1",
                "#{pane_current_command}",
            ],
            capture=True,
        )
        current_cmd = (proc.stdout or "").strip()
    except Exception:
        current_cmd = ""

    if current_cmd and current_cmd != "tmux":
        respawn = _run_outer_tmux(
            [
                "respawn-pane",
                "-k",
                "-t",
                f"{OUTER_SESSION}:{OUTER_WINDOW}.1",
                attach_cmd,
            ],
            capture=True,
        )
        if respawn.returncode != 0:
            detail = (respawn.stderr or respawn.stdout or "").strip()
            if detail:
                logger.warning(f"ensure_right_pane_ready respawn failed: {detail}")
            return False

    return True


def _disable_outer_border_resize() -> None:
    """Disable mouse drag-to-resize on pane borders (outer dashboard tmux server only)."""
    try:
        # tmux enables border resizing via MouseDrag1Border. Unbind it on our dedicated server
        # to avoid accidental resizing when selecting text near the pane divider.
        for key in (
            "MouseDrag1Border",
            "MouseDown1Border",
            "MouseDragEnd1Border",
            "MouseUp1Border",
        ):
            _run_outer_tmux(["unbind-key", "-T", "root", key], capture=True)
    except Exception:
        return


def outer_border_resize_enabled() -> bool:
    """Return True when user has enabled tmux border drag-to-resize in the dashboard."""
    try:
        from .state import get_dashboard_state_value

        return bool(get_dashboard_state_value(_STATE_BORDER_RESIZE_ENABLED_KEY, False))
    except Exception:
        return False


def _get_leftmost_outer_pane_width_cols() -> int | None:
    """Return current width (cols) of the leftmost outer pane (best-effort)."""
    try:
        panes_out = (
            _run_outer_tmux(
                [
                    "list-panes",
                    "-t",
                    f"{OUTER_SESSION}:{OUTER_WINDOW}",
                    "-F",
                    "#{pane_id}\t#{pane_left}\t#{pane_index}\t#{pane_width}",
                ],
                capture=True,
                timeout_s=0.2,
            ).stdout
            or ""
        )
        panes: list[tuple[int, int, int]] = []
        for line in _parse_lines(panes_out):
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            pane_id = (parts[0] or "").strip()
            if not pane_id:
                continue
            try:
                pane_left = int((parts[1] or "0").strip())
                pane_index = int((parts[2] or "0").strip() or "0")
                pane_width = int((parts[3] or "0").strip())
            except Exception:
                continue
            if pane_width <= 0:
                continue
            panes.append((pane_left, pane_index, pane_width))

        # Only meaningful when there are two panes (dashboard + terminal).
        if len(panes) < 2:
            return None

        panes.sort(key=lambda t: (t[0], t[1]))
        return int(panes[0][2])
    except Exception:
        return None


def remember_outer_dashboard_pane_width_cols() -> None:
    """Persist the current outer dashboard pane width (best-effort).

    Intended to be called when users disable border resize after manually adjusting the width.
    """
    width = _get_leftmost_outer_pane_width_cols()
    if width is None:
        return
    # Guardrails: avoid persisting obviously bogus sizes.
    if width < 20 or width > 500:
        return
    try:
        from .state import set_dashboard_state_value

        set_dashboard_state_value(_STATE_DASHBOARD_PANE_WIDTH_COLS_KEY, int(width))
    except Exception:
        return


def _preferred_outer_dashboard_pane_width_cols(default: int) -> int:
    """Preferred width to enforce when border resize is disabled."""
    try:
        from .state import get_dashboard_state_value

        raw = get_dashboard_state_value(_STATE_DASHBOARD_PANE_WIDTH_COLS_KEY, None)
        if raw is None:
            return int(default)
        try:
            w = int(raw)
        except Exception:
            return int(default)
        if w > 500:
            return int(default)
        if w < MIN_DASHBOARD_PANE_WIDTH_COLS:
            return MIN_DASHBOARD_PANE_WIDTH_COLS
        return w
    except Exception:
        return int(default)


def _get_outer_dashboard_pane_width() -> int | None:
    """Return the current width (columns) of the leftmost dashboard pane, or *None*."""
    try:
        panes_out = (
            _run_outer_tmux(
                [
                    "list-panes",
                    "-t",
                    f"{OUTER_SESSION}:{OUTER_WINDOW}",
                    "-F",
                    "#{pane_id}\t#{pane_left}\t#{pane_index}\t#{pane_width}",
                ],
                capture=True,
                timeout_s=0.2,
            ).stdout
            or ""
        )
        panes: list[tuple[int, int, str, int]] = []
        for line in _parse_lines(panes_out):
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            pane_id = (parts[0] or "").strip()
            if not pane_id:
                continue
            try:
                pane_left = int((parts[1] or "0").strip())
            except Exception:
                continue
            pane_index = 0
            if (parts[2] or "").strip():
                try:
                    pane_index = int(parts[2].strip())
                except Exception:
                    pane_index = 0
            try:
                pane_width = int((parts[3] or "0").strip())
            except Exception:
                continue
            panes.append((pane_left, pane_index, pane_id, pane_width))

        if len(panes) < 2:
            return None

        panes.sort(key=lambda t: (t[0], t[1]))
        return panes[0][3]
    except Exception:
        return None


def _set_outer_dashboard_pane_width(width_cols: int) -> None:
    """Best-effort: size the left dashboard pane to a fixed width (in terminal columns)."""
    try:
        w = int(width_cols)
        if w <= 0:
            return
    except Exception:
        return

    try:
        # Avoid assuming pane indexes are 0/1. Determine the leftmost pane by geometry.
        panes_out = (
            _run_outer_tmux(
                [
                    "list-panes",
                    "-t",
                    f"{OUTER_SESSION}:{OUTER_WINDOW}",
                    "-F",
                    "#{pane_id}\t#{pane_left}\t#{pane_index}",
                ],
                capture=True,
                timeout_s=0.2,
            ).stdout
            or ""
        )
        panes: list[tuple[int, int, str]] = []
        for line in _parse_lines(panes_out):
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            pane_id = (parts[0] or "").strip()
            if not pane_id:
                continue
            try:
                pane_left = int((parts[1] or "0").strip())
            except Exception:
                continue
            pane_index = 0
            if len(parts) > 2 and (parts[2] or "").strip():
                try:
                    pane_index = int(parts[2].strip())
                except Exception:
                    pane_index = 0
            panes.append((pane_left, pane_index, pane_id))

        # Only enforce widths when there are two panes (dashboard + terminal).
        if len(panes) < 2:
            return

        panes.sort(key=lambda t: (t[0], t[1]))
        leftmost_pane_id = panes[0][2]
        _run_outer_tmux(
            ["resize-pane", "-t", leftmost_pane_id, "-x", str(w)],
            capture=True,
            timeout_s=0.2,
        )
    except Exception:
        return


def enforce_outer_dashboard_pane_width(width_cols: int = DEFAULT_DASHBOARD_PANE_WIDTH_COLS) -> None:
    """Best-effort: enforce the fixed dashboard pane width for the outer tmux layout."""
    if outer_border_resize_enabled():
        # Still enforce a minimum so the dashboard never disappears on resize.
        current = _get_outer_dashboard_pane_width()
        if current is not None and current < MIN_DASHBOARD_PANE_WIDTH_COLS:
            _set_outer_dashboard_pane_width(MIN_DASHBOARD_PANE_WIDTH_COLS)
        return
    # When the user has ever resized the border (and then disabled it), use their last
    # chosen width as the default for future sessions.
    effective = (
        _preferred_outer_dashboard_pane_width_cols(DEFAULT_DASHBOARD_PANE_WIDTH_COLS)
        if int(width_cols) == int(DEFAULT_DASHBOARD_PANE_WIDTH_COLS)
        else int(width_cols)
    )
    _set_outer_dashboard_pane_width(effective)


def list_inner_windows() -> list[InnerWindow]:
    import time as _time

    t0 = _time.time()
    if not ensure_inner_session():
        return []
    logger.debug(f"[PERF] list_inner_windows ensure_inner_session: {_time.time() - t0:.3f}s")
    t1 = _time.time()
    state = _load_terminal_state()
    logger.debug(f"[PERF] list_inner_windows _load_terminal_state: {_time.time() - t1:.3f}s")
    out = (
        _run_inner_tmux(
            [
                "list-windows",
                "-t",
                INNER_SESSION,
                "-F",
                "#{window_id}\t#{window_name}\t#{window_active}\t#{"
                + OPT_TERMINAL_ID
                + "}\t#{"
                + OPT_PROVIDER
                + "}\t#{"
                + OPT_SESSION_TYPE
                + "}\t#{"
                + OPT_SESSION_ID
                + "}\t#{"
                + OPT_TRANSCRIPT_PATH
                + "}\t#{"
                + OPT_CONTEXT_ID
                + "}\t#{"
                + OPT_ATTENTION
                + "}\t#{"
                + OPT_CREATED_AT
                + "}\t#{"
                + OPT_NO_TRACK
                + "}\t#{pane_pid}\t#{pane_current_command}\t#{pane_tty}",
            ],
            capture=True,
        ).stdout
        or ""
    )
    windows: list[InnerWindow] = []
    for line in _parse_lines(out):
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        window_id = parts[0]
        window_name = parts[1]
        active = parts[2] == "1"
        terminal_id = parts[3] if len(parts) > 3 and parts[3] else None
        provider = parts[4] if len(parts) > 4 and parts[4] else None
        session_type = parts[5] if len(parts) > 5 and parts[5] else None
        session_id = parts[6] if len(parts) > 6 and parts[6] else None
        transcript_path = parts[7] if len(parts) > 7 and parts[7] else None
        context_id = parts[8] if len(parts) > 8 and parts[8] else None
        attention = parts[9] if len(parts) > 9 and parts[9] else None
        created_at_str = parts[10] if len(parts) > 10 and parts[10] else None
        created_at: float | None = None
        if created_at_str:
            try:
                created_at = float(created_at_str)
            except ValueError:
                pass
        no_track_str = parts[11] if len(parts) > 11 and parts[11] else None
        no_track = no_track_str == "1"
        pane_pid_str = parts[12] if len(parts) > 12 and parts[12] else None
        pane_pid: int | None = None
        if pane_pid_str:
            try:
                pane_pid = int(pane_pid_str)
            except ValueError:
                pass
        pane_current_command = parts[13] if len(parts) > 13 and parts[13] else None
        pane_tty = parts[14] if len(parts) > 14 and parts[14] else None

        if terminal_id:
            persisted = state.get(terminal_id) or {}
            updates: dict[str, str] = {}
            persisted_provider = (persisted.get("provider") or "").strip()
            if persisted_provider and persisted_provider != (provider or "").strip():
                provider = persisted_provider
                updates[OPT_PROVIDER] = persisted_provider
            if not provider and persisted_provider:
                provider = persisted_provider
            persisted_session_type = (persisted.get("session_type") or "").strip()
            if persisted_session_type and persisted_session_type != (session_type or "").strip():
                session_type = persisted_session_type
                updates[OPT_SESSION_TYPE] = persisted_session_type
            if not session_type and persisted_session_type:
                session_type = persisted_session_type
            persisted_session_id = (persisted.get("session_id") or "").strip()
            if persisted_session_id and persisted_session_id != (session_id or "").strip():
                session_id = persisted_session_id
                updates[OPT_SESSION_ID] = persisted_session_id
            if not session_id and persisted_session_id:
                session_id = persisted_session_id
            persisted_transcript = (persisted.get("transcript_path") or "").strip()
            if persisted_transcript and persisted_transcript != (transcript_path or "").strip():
                transcript_path = persisted_transcript
                updates[OPT_TRANSCRIPT_PATH] = persisted_transcript
            if not transcript_path and persisted_transcript:
                transcript_path = persisted_transcript
            persisted_context = (persisted.get("context_id") or "").strip()
            if persisted_context and persisted_context != (context_id or "").strip():
                context_id = persisted_context
                updates[OPT_CONTEXT_ID] = persisted_context
            if not context_id and persisted_context:
                context_id = persisted_context
            if updates:
                try:
                    set_inner_window_options(window_id, updates)
                except Exception:
                    pass

        transcript_session_id = _session_id_from_transcript_path(transcript_path)
        if transcript_session_id:
            session_id = transcript_session_id

        windows.append(
            InnerWindow(
                window_id=window_id,
                window_name=window_name,
                active=active,
                terminal_id=terminal_id,
                provider=provider,
                session_type=session_type,
                session_id=session_id,
                transcript_path=transcript_path,
                context_id=context_id,
                attention=attention,
                created_at=created_at,
                no_track=no_track,
                pane_pid=pane_pid,
                pane_current_command=pane_current_command,
                pane_tty=pane_tty,
            )
        )
    # Sort by creation time (newest first). Windows without created_at go to the bottom.
    windows.sort(key=lambda w: w.created_at if w.created_at is not None else 0, reverse=True)
    return windows


def list_outer_panes(*, timeout_s: float = 0.2) -> list[str]:
    """List panes in the outer dashboard window (best-effort).

    Intended for lightweight watchdog checks; returns tab-delimited lines in the same
    format as `collect_tmux_debug_state()["outer_panes"]["stdout"]`.
    """
    if not tmux_available():
        return []
    try:
        proc = _run_outer_tmux(
            [
                "list-panes",
                "-t",
                f"{OUTER_SESSION}:{OUTER_WINDOW}",
                "-F",
                "#{pane_index}\t#{pane_active}\t#{pane_current_command}\t#{pane_pid}\t#{pane_tty}",
            ],
            capture=True,
            timeout_s=timeout_s,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    return [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]


def list_outer_messages_tail(limit: int = 120, *, timeout_s: float = 1.5) -> list[str]:
    """Return recent outer tmux server messages (best-effort)."""
    if not (tmux_available() and in_tmux() and managed_env_enabled()):
        logger.debug(
            "list_outer_messages_tail skipped (tmux_available=%s in_tmux=%s managed_env=%s)",
            tmux_available(),
            in_tmux(),
            managed_env_enabled(),
        )
        return []
    try:
        proc = _run_outer_tmux(["show-messages"], capture=True, timeout_s=timeout_s)
    except Exception as e:
        logger.debug(f"list_outer_messages_tail failed to run show-messages: {e}")
        return []
    if proc.returncode != 0:
        logger.debug(
            "list_outer_messages_tail show-messages rc=%s stderr=%s",
            proc.returncode,
            (proc.stderr or "").strip(),
        )
        return []
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if limit <= 0:
        return lines
    return lines[-limit:]


def set_inner_window_options(window_id: str, options: dict[str, str]) -> bool:
    import time as _time

    if not ensure_inner_session():
        return False
    ok = True
    for key, value in options.items():
        t0 = _time.time()
        # Important: these are per-window (not session-wide) to avoid cross-tab clobbering.
        if _run_inner_tmux(["set-option", "-w", "-t", window_id, key, value]).returncode != 0:
            ok = False
        logger.debug(f"[PERF] set_inner_window_options {key}: {_time.time() - t0:.3f}s")
    return ok


def kill_inner_window(window_id: str) -> bool:
    if not ensure_inner_session():
        return False
    return _run_inner_tmux(["kill-window", "-t", window_id]).returncode == 0


def create_inner_window(
    base_name: str,
    command: str,
    *,
    terminal_id: str | None = None,
    provider: str | None = None,
    context_id: str | None = None,
    no_track: bool = False,
) -> InnerWindow | None:
    import time as _time

    t0 = _time.time()
    logger.debug("[PERF] create_inner_window START")
    if not ensure_right_pane_ready():
        logger.warning("create_inner_window: right pane unavailable")
        return None
    logger.debug(f"[PERF] create_inner_window ensure_right_pane: {_time.time() - t0:.3f}s")

    t1 = _time.time()
    existing = list_inner_windows()
    logger.debug(f"[PERF] create_inner_window list_inner_windows: {_time.time() - t1:.3f}s")
    name = _unique_name((w.window_name for w in existing), base_name)

    # Record creation time before creating the window
    created_at = time.time()

    t2 = _time.time()
    proc = _run_inner_tmux(
        [
            "new-window",
            "-P",
            "-F",
            "#{window_id}\t#{window_name}",
            "-t",
            INNER_SESSION,
            "-n",
            name,
            command,
        ],
        capture=True,
    )
    logger.debug(f"[PERF] create_inner_window new-window: {_time.time() - t2:.3f}s")
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        if detail:
            logger.warning(f"create_inner_window new-window failed: {detail}")
        return None

    created = _parse_lines(proc.stdout or "")
    if not created:
        return None
    window_id, window_name = (created[0].split("\t", 1) + [""])[:2]

    # Always set options including the creation timestamp
    opts: dict[str, str] = {OPT_CREATED_AT: str(created_at)}
    if terminal_id:
        opts[OPT_TERMINAL_ID] = terminal_id
    if provider:
        opts[OPT_PROVIDER] = provider
    if context_id:
        opts[OPT_CONTEXT_ID] = context_id
    opts.setdefault(OPT_SESSION_TYPE, "")
    opts.setdefault(OPT_SESSION_ID, "")
    opts.setdefault(OPT_TRANSCRIPT_PATH, "")
    if no_track:
        opts[OPT_NO_TRACK] = "1"
    else:
        opts.setdefault(OPT_NO_TRACK, "")
    t3 = _time.time()
    set_inner_window_options(window_id, opts)
    logger.debug(f"[PERF] create_inner_window set_options: {_time.time() - t3:.3f}s")

    _run_inner_tmux(["select-window", "-t", window_id])

    return InnerWindow(
        window_id=window_id,
        window_name=window_name or name,
        active=True,
        terminal_id=terminal_id,
        provider=provider,
        context_id=context_id,
        created_at=created_at,
    )


def select_inner_window(window_id: str) -> bool:
    if not ensure_right_pane_ready():
        return False
    proc = _run_inner_tmux(["select-window", "-t", window_id], capture=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        if detail:
            logger.warning(f"select_inner_window failed ({window_id}): {detail}")
        return False
    return True


def focus_right_pane() -> bool:
    """Focus the right pane (terminal area) in the outer tmux layout."""
    proc = _run_outer_tmux(["select-pane", "-t", f"{OUTER_SESSION}:{OUTER_WINDOW}.1"], capture=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        if detail:
            logger.warning(f"focus_right_pane failed: {detail}")
        return False
    return True


def clear_attention(window_id: str) -> bool:
    """Clear the attention state for a window (e.g., after user acknowledges permission request)."""
    if not ensure_inner_session():
        return False
    return _run_inner_tmux(["set-option", "-w", "-t", window_id, OPT_ATTENTION, ""]).returncode == 0


def collect_tmux_debug_state() -> dict[str, object]:
    """Best-effort snapshot of tmux state for diagnosing blank/stuck panes.

    This must be non-intrusive: it should not create sessions or change state.
    """

    def _cap(proc: subprocess.CompletedProcess[str] | None) -> dict[str, object]:
        if proc is None:
            return {}

        def _trim(s: str | None) -> str:
            text = (s or "").strip()
            if len(text) > 4000:
                return text[:4000] + "…(truncated)"
            return text

        return {
            "rc": int(getattr(proc, "returncode", -1)),
            "stdout": _trim(getattr(proc, "stdout", "")),
            "stderr": _trim(getattr(proc, "stderr", "")),
        }

    state: dict[str, object] = {
        "tmux_available": tmux_available(),
        "in_tmux": in_tmux(),
        "managed_env": managed_env_enabled(),
        "outer_socket": OUTER_SOCKET,
        "inner_socket": INNER_SOCKET,
        "outer_session": OUTER_SESSION,
        "outer_window": OUTER_WINDOW,
        "inner_session": INNER_SESSION,
    }

    if not tmux_available():
        return state

    try:
        state["outer_has_session"] = _cap(
            _run_outer_tmux(["has-session", "-t", OUTER_SESSION], capture=True, timeout_s=0.5)
        )
        state["outer_panes"] = _cap(
            _run_outer_tmux(
                [
                    "list-panes",
                    "-t",
                    f"{OUTER_SESSION}:{OUTER_WINDOW}",
                    "-F",
                    "#{pane_index}\t#{pane_active}\t#{pane_current_command}\t#{pane_pid}\t#{pane_tty}",
                ],
                capture=True,
                timeout_s=0.5,
            )
        )
    except Exception:
        pass

    try:
        state["inner_has_session"] = _cap(
            _run_inner_tmux(["has-session", "-t", INNER_SESSION], capture=True, timeout_s=0.5)
        )
        state["inner_windows"] = _cap(
            _run_inner_tmux(
                [
                    "list-windows",
                    "-t",
                    INNER_SESSION,
                    "-F",
                    "#{window_id}\t#{window_index}\t#{window_name}\t#{window_active}\t#{"
                    + OPT_TERMINAL_ID
                    + "}\t#{"
                    + OPT_PROVIDER
                    + "}\t#{"
                    + OPT_SESSION_TYPE
                    + "}\t#{"
                    + OPT_SESSION_ID
                    + "}\t#{"
                    + OPT_CONTEXT_ID
                    + "}\t#{"
                    + OPT_ATTENTION
                    + "}",
                ],
                capture=True,
                timeout_s=0.5,
            )
        )
    except Exception:
        pass

    return state
