"""ReAlign init command - Initialize ReAlign tracking system."""

import shutil
import sys
from typing import Annotated, Any, Dict, Optional, Tuple
from pathlib import Path
import re
import typer
from rich.console import Console

from ..config import (
    ReAlignConfig,
    get_default_config_content,
    generate_random_username,
)

console = Console()


# tmux config template for Aline-managed dashboard sessions.
# Stored at ~/.aline/tmux/tmux.conf and sourced by the dashboard tmux bootstrap.
# Bump this version when the tmux config changes to trigger auto-update on `aline init`.
_TMUX_CONFIG_VERSION = 11


def _get_tmux_config() -> str:
    """Generate tmux config with Type-to-Exit bindings."""
    conf = f"# Aline tmux config (v{_TMUX_CONFIG_VERSION})\n"
    conf += r"""#
# Goal: make mouse selection copy to the system clipboard (macOS Terminal friendly).
# - Drag-select text with the mouse; when you release, it is copied to the clipboard.
# - Paste anywhere with Cmd+V.
#
# Note: Cmd+C inside tmux is still SIGINT (Terminal behavior).

set -g mouse on

# Pane border styling - use double lines for wider, more visible borders (tmux 3.2+).
# This helps users identify the resizable border area more easily.
set -g pane-border-lines double
set -g pane-border-style "fg=brightblack"
set -g pane-active-border-style "fg=blue"

# Add a small indicator showing where the border is (tmux 3.2+).
# This creates a visual "dead zone" that's more obvious for resizing.
set -g pane-border-indicators arrows

# Disable paste-time detection so key bindings work during paste.
set -g assume-paste-time 0

# Fast escape time so ESC is processed immediately (helps with paste detection).
set -s escape-time 0

# Better scrolling: enter copy-mode with -e so scrolling to bottom exits it.
# Important: lock focus to the pane under the mouse first (`select-pane -t =`), then
# forward wheel events to that same pane (`-t =`). This avoids fast-scroll momentum
# leaking into the previously active pane's input history.
bind-key -n WheelUpPane select-pane -t = \; if-shell -F -t = "#{mouse_any_flag}" "send-keys -M -t =" "if -Ft= '#{pane_in_mode}' 'send-keys -M -t =' 'copy-mode -e -t ='"
bind-key -n WheelDownPane select-pane -t = \; if-shell -F -t = "#{mouse_any_flag}" "send-keys -M -t =" "send-keys -M -t ="

# macOS clipboard: copy selection to clipboard when drag ends.
# Use copy-pipe-no-clear to preserve selection highlight after copying.
bind -T copy-mode-vi MouseDragEnd1Pane send -X copy-pipe-no-clear "pbcopy"
bind -T copy-mode    MouseDragEnd1Pane send -X copy-pipe-no-clear "pbcopy"

# MouseDrag1Pane: Clear old selection and start new one when dragging begins.
# This ensures selection only clears when starting a NEW drag, not on click.
bind -T copy-mode-vi MouseDrag1Pane select-pane \; send -X clear-selection \; send -X begin-selection
bind -T copy-mode    MouseDrag1Pane select-pane \; send -X clear-selection \; send -X begin-selection

# MouseDown1Pane: Click clears selection but stays in copy-mode (no scroll).
# To exit copy-mode: scroll to bottom (auto-exit) or press q/Escape.
bind -T copy-mode-vi MouseDown1Pane select-pane \; send -X clear-selection
bind -T copy-mode    MouseDown1Pane select-pane \; send -X clear-selection

# Escape key: exit copy-mode (also use this before Cmd+V paste in copy-mode).
bind -T copy-mode-vi Escape send -X cancel
bind -T copy-mode    Escape send -X cancel

# Type-to-Exit: Typing any alphanumeric character exits copy-mode and sends the key.
"""
    def _tmux_quote(value: str) -> str:
        # tmux config treats `#` as a comment delimiter; quote args so keys like `#` don't disappear.
        # Note: `~` is special in tmux config parsing; use an escaped form instead of quotes.
        if value == "~":
            return r"\~"
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def _bind_cancel_and_send(key: str) -> str:
        key_token = _tmux_quote(key)
        return (
            f"bind -T copy-mode-vi {key_token} send -X cancel \\; send-keys -- {key_token}\n"
            f"bind -T copy-mode    {key_token} send -X cancel \\; send-keys -- {key_token}\n"
        )

    # Generate bindings for common characters.
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.@/!#$%^&*()+=,<>?[]{}|~`;\\\"'"
    for c in chars:
        conf += _bind_cancel_and_send(c)

    # Space: use tmux key name.
    conf += "bind -T copy-mode-vi Space send -X cancel \\; send-keys Space\n"
    conf += "bind -T copy-mode    Space send -X cancel \\; send-keys Space\n"

    # Cmd+V (paste): exit copy-mode when paste is detected.
    # Different terminals handle Cmd+V differently:
    # - Some send M-v (Meta+V)
    # - Some use bracketed paste mode and send the content directly
    # Since we bind all printable chars above, pasting text starting with a letter/number will exit.
    # For other cases, bind Enter to also exit (multiline paste often starts with newline).
    conf += "bind -T copy-mode-vi M-v send -X cancel \\; run 'sleep 0.05' \\; send-keys M-v\n"
    conf += "bind -T copy-mode    M-v send -X cancel \\; run 'sleep 0.05' \\; send-keys M-v\n"

    return conf


_TMUX_CONF_REPAIR_TILDE_KEY_RE = re.compile(
    r'^(bind(?:-key)?\s+-T\s+copy-mode(?:-vi)?\s+)(?:"~"|~)(\s+send\s+-X\s+cancel\s+\\;\s+send-keys\s+)(?:"~"|~)\s*$',
    re.MULTILINE,
)

_TMUX_CONF_REPAIR_KEY_NEEDS_QUOTE_RE = re.compile(
    r"^(bind(?:-key)?\s+-T\s+copy-mode(?:-vi)?\s+)([#{}])(\s+send\s+-X\s+cancel\s+\\;\s+send-keys\s+)\2\s*$",
    re.MULTILINE,
)

_TMUX_CONF_REPAIR_SEND_KEYS_DASHDASH_RE = re.compile(
    r"^(bind(?:-key)?\s+-T\s+copy-mode(?:-vi)?\s+.+\s+send\s+-X\s+cancel\s+\\;\s+send-keys\s+)(?!--\s)(.+)$",
    re.MULTILINE,
)


def _get_tmux_config_version(content: str) -> int:
    """Extract version number from tmux config content. Returns 0 if not found."""
    # Look for "# Aline tmux config (vN)" pattern
    match = re.search(r"# Aline tmux config \(v(\d+)\)", content)
    if match:
        return int(match.group(1))
    # Old configs without version marker are version 1
    if "# Aline tmux config" in content:
        return 1
    return 0


def _initialize_tmux_config() -> Path:
    """Initialize ~/.aline/tmux/tmux.conf with auto-update on version change."""
    tmux_conf_path = Path.home() / ".aline" / "tmux" / "tmux.conf"
    tmux_conf_path.parent.mkdir(parents=True, exist_ok=True)

    if not tmux_conf_path.exists():
        tmux_conf_path.write_text(_get_tmux_config(), encoding="utf-8")
        return tmux_conf_path

    # Check existing config
    try:
        existing = tmux_conf_path.read_text(encoding="utf-8")
    except Exception:
        return tmux_conf_path

    # Only manage Aline-generated configs
    if "# Aline tmux config" not in existing:
        return tmux_conf_path

    # Check version and update if outdated
    existing_version = _get_tmux_config_version(existing)
    if existing_version < _TMUX_CONFIG_VERSION:
        # Auto-update to latest config
        tmux_conf_path.write_text(_get_tmux_config(), encoding="utf-8")
        return tmux_conf_path

    # Best-effort repair for older Aline-generated configs that used unquoted `#` keys.
    # tmux parses `#` as a comment delimiter, turning `bind ... # ...` into `bind ...` (invalid).
    repaired = existing
    repaired = _TMUX_CONF_REPAIR_TILDE_KEY_RE.sub(r"\1\\~\2\\~", repaired)
    repaired = _TMUX_CONF_REPAIR_KEY_NEEDS_QUOTE_RE.sub(r'\1"\2"\3"\2"', repaired)
    repaired = _TMUX_CONF_REPAIR_SEND_KEYS_DASHDASH_RE.sub(r"\1-- \2", repaired)
    if repaired != existing:
        tmux_conf_path.write_text(repaired, encoding="utf-8")
    return tmux_conf_path


def _detect_existing_global_state() -> Tuple[bool, bool]:
    """
    Detect whether global config and DB existed *before* `aline init`.

    Returns:
        (config_existed, db_existed)
    """
    config_path = Path.home() / ".aline" / "config.yaml"
    config_existed = config_path.exists()

    # Best-effort: determine DB path without requiring config to exist.
    try:
        if config_existed:
            db_path = Path(ReAlignConfig.load(config_path).sqlite_db_path).expanduser()
        else:
            db_path = Path(ReAlignConfig().sqlite_db_path).expanduser()
    except Exception:
        db_path = Path(ReAlignConfig().sqlite_db_path).expanduser()

    return config_existed, db_path.exists()


def _should_start_watcher(
    start_watcher: Optional[bool],
    *,
    config_existed: bool,
    db_existed: bool,
) -> bool:
    """
    Decide whether `aline init` should start the watcher.

    Default behavior (start_watcher=None):
      - Auto-start only on first init (when config/db didn't exist yet)
    """
    if start_watcher is not None:
        return bool(start_watcher)
    return (not config_existed) or (not db_existed)


def _initialize_skills() -> Path:
    """Initialize Claude Code skills (best-effort, no overwrite).

    Returns:
        Path to the skills root directory
    """
    from .add import add_skills_command

    add_skills_command(force=False, include_extras=False)
    return Path.home() / ".claude" / "skills"


def _initialize_claude_hooks() -> Tuple[bool, list]:
    """Initialize Claude Code hooks (Stop, UserPromptSubmit, PermissionRequest).

    Installs all Aline hooks to the global Claude Code settings.
    Does not overwrite existing hooks.

    Returns:
        (all_success, list of installed hook names)
    """
    installed_hooks = []
    all_success = True

    try:
        from ..claude_hooks.stop_hook_installer import ensure_stop_hook_installed

        if ensure_stop_hook_installed(quiet=True):
            installed_hooks.append("Stop")
        else:
            all_success = False
    except Exception:
        all_success = False

    try:
        from ..claude_hooks.user_prompt_submit_hook_installer import (
            ensure_user_prompt_submit_hook_installed,
        )

        if ensure_user_prompt_submit_hook_installed(quiet=True):
            installed_hooks.append("UserPromptSubmit")
        else:
            all_success = False
    except Exception:
        all_success = False

    try:
        from ..claude_hooks.permission_request_hook_installer import (
            ensure_permission_request_hook_installed,
        )

        if ensure_permission_request_hook_installed(quiet=True):
            installed_hooks.append("PermissionRequest")
        else:
            all_success = False
    except Exception:
        all_success = False

    return all_success, installed_hooks


def init_global(
    force: bool = False,
) -> Dict[str, Any]:
    """
    Core global initialization logic (non-interactive).

    Args:
        force: Overwrite the global config with defaults

    Returns:
        Dictionary with initialization results and metadata
    """
    result = {
        "success": False,
        "config_path": None,
        "db_path": None,
        "tmux_conf": None,
        "skills_path": None,
        "hooks_installed": None,
        "message": "",
        "errors": [],
    }

    try:
        # Initialize global config if not exists
        global_config_path = Path.home() / ".aline" / "config.yaml"
        if force or not global_config_path.exists():
            global_config_path.parent.mkdir(parents=True, exist_ok=True)
            global_config_path.write_text(
                get_default_config_content(), encoding="utf-8"
            )
        result["config_path"] = str(global_config_path)

        # Load config
        config = ReAlignConfig.load()

        # Ensure early session title is enabled (default-on since rename)
        if not config.enable_early_session_title:
            config.enable_early_session_title = True
            config.save()

        # User identity setup (V17: uid from Supabase login)
        if not config.uid:
            console.print("\n[bold blue]═══ User Identity Setup ═══[/bold blue]")
            console.print(
                "Aline requires login for user identification.\n"
            )
            console.print(
                "[yellow]Run 'aline login' to authenticate with your account.[/yellow]\n"
            )
            # If user_name is also not set, generate a temporary one
            if not config.user_name:
                config.user_name = generate_random_username()
                config.save()
                console.print(
                    f"Auto-generated username: [yellow]{config.user_name}[/yellow] (will update on login)\n"
                )

        # Initialize database
        db_path = Path(config.sqlite_db_path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        result["db_path"] = str(db_path)

        # Create/upgrade database schema
        from ..db.sqlite_db import SQLiteDatabase

        db = SQLiteDatabase(str(db_path))
        db.initialize()
        db.close()

        tmux_conf = _initialize_tmux_config()
        result["tmux_conf"] = str(tmux_conf)

        # Initialize Claude Code skills
        skills_path = _initialize_skills()
        result["skills_path"] = str(skills_path)

        # Inject proactive OneContext instructions into global config files
        from .add import inject_onecontext_instructions

        instruction_results = inject_onecontext_instructions(force=False)
        result["instruction_blocks"] = instruction_results

        # Initialize Claude Code hooks (Stop, UserPromptSubmit, PermissionRequest)
        hooks_success, hooks_installed = _initialize_claude_hooks()
        result["hooks_installed"] = hooks_installed
        if not hooks_success:
            result["errors"].append("Some Claude Code hooks failed to install")

        result["success"] = True
        result["message"] = (
            "Aline initialized successfully (global config + database + tmux + skills + hooks ready)"
        )

    except Exception as e:
        result["errors"].append(f"Initialization failed: {e}")
        result["message"] = f"Failed to initialize: {e}"

    return result


def init_command(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite global config with defaults"),
    doctor: Annotated[
        bool,
        typer.Option(
            "--doctor/--no-doctor",
            help="Run 'aline doctor' after init (best for upgrades)",
        ),
    ] = False,
    install_tmux: Annotated[
        bool,
        typer.Option(
            "--install-tmux/--no-install-tmux",
            help="Auto-install tmux via Homebrew if missing (macOS only)",
        ),
    ] = True,
    start_watcher: Optional[bool] = typer.Option(
        None,
        "--start-watcher/--no-start-watcher",
        help="Start watcher daemon after init (default: auto on first init)",
    ),
):
    """Initialize Aline global config and SQLite database.

    Initializes the global config/database (in `~/.aline/`). Project/workspace context
    is inferred automatically when sessions are imported/processed.
    """
    config_existed, db_existed = _detect_existing_global_state()

    result = init_global(
        force=force,
    )

    # First-time UX: tmux is required for the default dashboard experience (tmux mode).
    # Only attempt on macOS; on other platforms, leave it to user.
    if (
        result.get("success")
        and install_tmux
        and result.get("tmux_conf")
        and sys.platform == "darwin"
        and shutil.which("tmux") is None
    ):
        console.print("\n[bold]tmux not found. Installing via Homebrew...[/bold]")
        try:
            from . import add as add_cmd

            rc = add_cmd.add_tmux_command(install_brew=True)
            if rc != 0:
                result["errors"] = (result.get("errors") or []) + [
                    "tmux install failed (required for the default tmux dashboard)",
                    "Tip: set ALINE_TERMINAL_MODE=native to run without tmux",
                ]
        except Exception as e:
            result["errors"] = (result.get("errors") or []) + [
                f"tmux install failed: {e}",
                "Tip: set ALINE_TERMINAL_MODE=native to run without tmux",
            ]

    if doctor and result.get("success"):
        # Run doctor in "safe" mode: restart only if already running, and keep init fast.
        try:
            from . import doctor as doctor_cmd

            restart_daemons = start_watcher is not False
            doctor_exit = doctor_cmd.run_doctor(
                restart_daemons=restart_daemons,
                start_if_not_running=False,
                verbose=False,
                clear_cache=False,
                skip_ensure_env=True,
            )
            if doctor_exit != 0:
                result["success"] = False
                result["errors"] = (result.get("errors") or []) + [
                    "aline doctor failed (see output above)"
                ]
                result["message"] = f"{result.get('message', '').strip()} (doctor failed)".strip()
        except Exception as e:
            result["success"] = False
            result["errors"] = (result.get("errors") or []) + [f"aline doctor failed: {e}"]
            result["message"] = f"{result.get('message', '').strip()} (doctor failed)".strip()

    watcher_started: Optional[bool] = None
    watcher_start_exit: Optional[int] = None
    should_start = False

    if result.get("success"):
        should_start = _should_start_watcher(
            start_watcher,
            config_existed=config_existed,
            db_existed=db_existed,
        )
        if should_start:
            # watcher_start_command() auto-starts the worker daemon too.
            try:
                from . import watcher as watcher_cmd

                watcher_start_exit = watcher_cmd.watcher_start_command()
                watcher_started = watcher_start_exit == 0
            except Exception:
                watcher_started = False
                watcher_start_exit = 1

    # Print detailed results
    console.print("\n[bold blue]═══ Aline Initialization ═══[/bold blue]\n")

    if result["success"]:
        console.print("[bold green]✓ Status: SUCCESS[/bold green]\n")
    else:
        console.print("[bold red]✗ Status: FAILED[/bold red]\n")

    # Print all parameters and results
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Config: [cyan]{result.get('config_path', 'N/A')}[/cyan]")
    console.print(f"  Database: [cyan]{result.get('db_path', 'N/A')}[/cyan]")
    console.print(f"  Tmux: [cyan]{result.get('tmux_conf', 'N/A')}[/cyan]")
    console.print(f"  Skills: [cyan]{result.get('skills_path', 'N/A')}[/cyan]")

    # Codex compatibility note (best-effort).
    # We rely on the Rust Codex CLI notify hook to avoid expensive polling. If the installed
    # Codex binary is legacy/unsupported, warn and suggest upgrading.
    try:
        from ..codex_hooks.notify_hook_installer import codex_cli_supports_notify_hook

        supported = codex_cli_supports_notify_hook()
        if supported is False:
            console.print("\n[yellow]![/yellow] Codex CLI detected but does not support notify hook.")
            console.print(
                "[dim]Tip: update to the Rust Codex CLI to enable reliable, event-driven Codex imports (no polling).[/dim]"
            )
        # If Codex isn't installed (None), stay silent.
    except Exception:
        pass

    hooks_installed = result.get("hooks_installed") or []
    if hooks_installed:
        console.print(f"  Hooks: [cyan]{', '.join(hooks_installed)}[/cyan]")
    else:
        console.print("  Hooks: [yellow]None installed[/yellow]")

    if result.get("success") and should_start:
        console.print("\n[bold]Daemons:[/bold]")
        if watcher_started:
            console.print("  Status: [green]STARTED[/green] (watcher + worker)")
            console.print("  Check: [cyan]aline watcher status[/cyan]", style="dim")
        else:
            console.print("  Status: [red]FAILED TO START[/red]")
            console.print("  Try: [cyan]aline watcher start[/cyan]", style="dim")
            console.print("  Logs: [dim]disabled (except dashboard.log)[/dim]", style="dim")

    if result.get("errors"):
        console.print("\n[bold red]Errors:[/bold red]")
        for error in result["errors"]:
            console.print(f"  • {error}", style="red")

    console.print(f"\n[bold]Result:[/bold] {result['message']}\n")

    if result["success"]:
        console.print("[bold]Next steps:[/bold]")
        console.print(
            "  1. Start Claude Code or Codex - sessions are tracked automatically",
            style="dim",
        )
        console.print(
            "  2. Search history with: [cyan]aline search <query>[/cyan]", style="dim"
        )

        # If the user explicitly asked to start the watcher, failing to do so should fail init too.
        if start_watcher is True and watcher_start_exit not in (0, None):
            raise typer.Exit(1)
    else:
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(init_command)
