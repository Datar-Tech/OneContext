"""
Claude Code PermissionRequest Hook Auto-installer

Automatically installs Aline's PermissionRequest hook into Claude Code's configuration.
Uses append mode - does not overwrite user's existing hooks configuration.
"""

import json
import sys
from pathlib import Path
from typing import Optional

from ..logging_config import setup_logger

logger = setup_logger("realign.hooks.permission_request_installer", "hooks_installer.log")

# Marker to identify Aline hook for later uninstallation
ALINE_HOOK_MARKER = "aline-permission-request-hook"


def get_permission_request_hook_script_path() -> Path:
    """Get path to permission_request_hook.py script"""
    return Path(__file__).parent / "permission_request_hook.py"


def get_permission_request_hook_command() -> str:
    """
    Get the PermissionRequest hook execution command.

    Uses direct script path to avoid module import issues.
    """
    script_path = get_permission_request_hook_script_path()
    return f"{sys.executable} {script_path}"


def get_settings_path(project_path: Optional[Path] = None) -> Path:
    """
    Get Claude Code settings.json path.

    Args:
        project_path: If provided, returns project-level config path; otherwise global config.

    Returns:
        Path to settings.json
    """
    if project_path:
        return project_path / ".claude" / "settings.local.json"
    return Path.home() / ".claude" / "settings.json"


def is_hook_installed(settings_path: Path) -> bool:
    """
    Check if Aline PermissionRequest hook is installed.

    Args:
        settings_path: Path to settings.json

    Returns:
        True if installed
    """
    if not settings_path.exists():
        return False

    try:
        settings = json.loads(settings_path.read_text())
        permission_hooks = settings.get("hooks", {}).get("PermissionRequest", [])

        for hook_config in permission_hooks:
            hooks_list = hook_config.get("hooks", [])
            for h in hooks_list:
                command = h.get("command", "")
                if ALINE_HOOK_MARKER in command or "permission_request_hook" in command:
                    return True

        return False

    except Exception:
        return False


def install_permission_request_hook(
    settings_path: Optional[Path] = None,
    force: bool = False,
    quiet: bool = False,
) -> bool:
    """
    Install Claude Code PermissionRequest hook.

    Uses append mode: does not overwrite user's existing hooks configuration.

    Args:
        settings_path: Path to settings.json (defaults to global config)
        force: If True, reinstall even if already installed
        quiet: If True, don't output any messages

    Returns:
        True if installation successful or already installed
    """
    if settings_path is None:
        settings_path = get_settings_path()

    try:
        # Check if already installed
        if not force and is_hook_installed(settings_path):
            logger.debug("Aline PermissionRequest hook already installed")
            return True

        # Read existing settings
        settings = {}
        if settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text())
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {settings_path}, creating new settings")
                settings = {}

        # Ensure hooks structure exists
        if "hooks" not in settings:
            settings["hooks"] = {}
        if "PermissionRequest" not in settings["hooks"]:
            settings["hooks"]["PermissionRequest"] = []

        # If force install, remove old Aline hook first
        if force:
            permission_hooks = settings["hooks"]["PermissionRequest"]
            new_hooks = []
            for hook_config in permission_hooks:
                hooks_list = hook_config.get("hooks", [])
                filtered = [
                    h
                    for h in hooks_list
                    if ALINE_HOOK_MARKER not in h.get("command", "")
                    and "permission_request_hook" not in h.get("command", "")
                ]
                if filtered:
                    hook_config["hooks"] = filtered
                    new_hooks.append(hook_config)
            settings["hooks"]["PermissionRequest"] = new_hooks

        # Append Aline hook with matcher for all tools
        hook_command = get_permission_request_hook_command()
        aline_hook = {
            "matcher": "*",  # Match all tools
            "hooks": [{"type": "command", "command": f"{hook_command}  # {ALINE_HOOK_MARKER}"}],
        }
        settings["hooks"]["PermissionRequest"].append(aline_hook)

        # Write back settings
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings, indent=2))

        logger.info(f"Aline PermissionRequest hook installed to {settings_path}")
        if not quiet:
            print(f"[Aline] PermissionRequest hook installed to {settings_path}", file=sys.stderr)

        return True

    except Exception as e:
        logger.error(f"Failed to install PermissionRequest hook: {e}")
        if not quiet:
            print(f"[Aline] Failed to install PermissionRequest hook: {e}", file=sys.stderr)
        return False


def uninstall_permission_request_hook(
    settings_path: Optional[Path] = None,
    quiet: bool = False,
) -> bool:
    """
    Uninstall Aline PermissionRequest hook.

    Args:
        settings_path: Path to settings.json (defaults to global config)
        quiet: If True, don't output any messages

    Returns:
        True if uninstallation successful
    """
    if settings_path is None:
        settings_path = get_settings_path()

    try:
        if not settings_path.exists():
            return True

        settings = json.loads(settings_path.read_text())
        permission_hooks = settings.get("hooks", {}).get("PermissionRequest", [])

        # Filter out Aline hook
        new_hooks = []
        removed = False
        for hook_config in permission_hooks:
            hooks_list = hook_config.get("hooks", [])
            filtered = [
                h
                for h in hooks_list
                if ALINE_HOOK_MARKER not in h.get("command", "")
                and "permission_request_hook" not in h.get("command", "")
            ]
            if len(filtered) < len(hooks_list):
                removed = True
            if filtered:
                hook_config["hooks"] = filtered
                new_hooks.append(hook_config)

        if removed:
            settings["hooks"]["PermissionRequest"] = new_hooks
            settings_path.write_text(json.dumps(settings, indent=2))
            logger.info("Aline PermissionRequest hook uninstalled")
            if not quiet:
                print("[Aline] PermissionRequest hook uninstalled", file=sys.stderr)

        return True

    except Exception as e:
        logger.error(f"Failed to uninstall PermissionRequest hook: {e}")
        if not quiet:
            print(f"[Aline] Failed to uninstall PermissionRequest hook: {e}", file=sys.stderr)
        return False


def ensure_permission_request_hook_installed(quiet: bool = True) -> bool:
    """
    Ensure PermissionRequest hook is installed (called at watcher startup).

    This is an idempotent operation - will not reinstall if already installed.

    Args:
        quiet: If True, only output message on first install

    Returns:
        True if hook is available
    """
    settings_path = get_settings_path()

    if is_hook_installed(settings_path):
        logger.debug("PermissionRequest hook already installed, skipping")
        return True

    return install_permission_request_hook(settings_path, quiet=quiet)
