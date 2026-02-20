"""
Claude Code UserPromptSubmit Hook installer.

Adds Aline hook to Claude Code settings without overwriting existing hooks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from ..logging_config import setup_logger

logger = setup_logger("realign.hooks.installer", "hooks_installer.log")

ALINE_HOOK_MARKER = "aline-user-prompt-submit-hook"


def get_user_prompt_submit_hook_script_path() -> Path:
    return Path(__file__).parent / "user_prompt_submit_hook.py"


def get_user_prompt_submit_hook_command() -> str:
    script_path = get_user_prompt_submit_hook_script_path()
    return f"{sys.executable} {script_path}"


def get_settings_path(project_path: Optional[Path] = None) -> Path:
    if project_path:
        return project_path / ".claude" / "settings.local.json"
    return Path.home() / ".claude" / "settings.json"


def is_hook_installed(settings_path: Path) -> bool:
    if not settings_path.exists():
        return False
    try:
        settings = json.loads(settings_path.read_text())
        hooks = settings.get("hooks", {}).get("UserPromptSubmit", [])
        for hook_config in hooks:
            hooks_list = hook_config.get("hooks", [])
            for h in hooks_list:
                command = h.get("command", "")
                if ALINE_HOOK_MARKER in command or "realign.hooks.user_prompt_submit" in command:
                    return True
        return False
    except Exception:
        return False


def install_user_prompt_submit_hook(
    settings_path: Optional[Path] = None,
    force: bool = False,
    quiet: bool = False,
) -> bool:
    if settings_path is None:
        settings_path = get_settings_path()

    try:
        if not force and is_hook_installed(settings_path):
            logger.debug("Aline UserPromptSubmit hook already installed")
            return True

        settings = {}
        if settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text())
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {settings_path}, creating new settings")
                settings = {}

        if "hooks" not in settings:
            settings["hooks"] = {}
        if "UserPromptSubmit" not in settings["hooks"]:
            settings["hooks"]["UserPromptSubmit"] = []

        if force:
            new_hooks = []
            for hook_config in settings["hooks"]["UserPromptSubmit"]:
                hooks_list = hook_config.get("hooks", [])
                filtered = [
                    h
                    for h in hooks_list
                    if ALINE_HOOK_MARKER not in h.get("command", "")
                    and "realign.hooks.user_prompt_submit" not in h.get("command", "")
                ]
                if filtered:
                    hook_config["hooks"] = filtered
                    new_hooks.append(hook_config)
            settings["hooks"]["UserPromptSubmit"] = new_hooks

        hook_command = get_user_prompt_submit_hook_command()
        aline_hook = {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{hook_command}  # {ALINE_HOOK_MARKER}",
                }
            ]
        }
        settings["hooks"]["UserPromptSubmit"].append(aline_hook)

        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings, indent=2))

        logger.info(f"Aline UserPromptSubmit hook installed to {settings_path}")
        if not quiet:
            print(f"[Aline] UserPromptSubmit hook installed to {settings_path}", file=sys.stderr)
        return True
    except Exception as e:
        logger.error(f"Failed to install UserPromptSubmit hook: {e}")
        if not quiet:
            print(f"[Aline] Failed to install UserPromptSubmit hook: {e}", file=sys.stderr)
        return False


def uninstall_user_prompt_submit_hook(settings_path: Optional[Path] = None) -> bool:
    if settings_path is None:
        settings_path = get_settings_path()
    if not settings_path.exists():
        return True
    try:
        settings = json.loads(settings_path.read_text())
        hooks = settings.get("hooks", {}).get("UserPromptSubmit", [])
        new_hooks = []
        for hook_config in hooks:
            hooks_list = hook_config.get("hooks", [])
            filtered = [
                h
                for h in hooks_list
                if ALINE_HOOK_MARKER not in h.get("command", "")
                and "realign.hooks.user_prompt_submit" not in h.get("command", "")
            ]
            if filtered:
                hook_config["hooks"] = filtered
                new_hooks.append(hook_config)
        settings.setdefault("hooks", {})["UserPromptSubmit"] = new_hooks
        settings_path.write_text(json.dumps(settings, indent=2))
        logger.info("Aline UserPromptSubmit hook uninstalled")
        return True
    except Exception as e:
        logger.error(f"Failed to uninstall UserPromptSubmit hook: {e}")
        return False


def ensure_user_prompt_submit_hook_installed(quiet: bool = True) -> bool:
    if is_hook_installed(get_settings_path()):
        logger.debug("UserPromptSubmit hook already installed, skipping")
        return True
    return install_user_prompt_submit_hook(quiet=quiet)
