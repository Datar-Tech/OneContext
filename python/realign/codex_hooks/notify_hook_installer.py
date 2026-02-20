"""
Codex notify hook installer (best-effort).

We primarily support the Rust Codex CLI which reads CODEX_HOME/config.toml and
supports `notify = "command args..."` to run a script when a turn finishes.

For legacy Codex config.yaml/config.json formats, we can only set `notify: true`
to enable built-in notifications; there is no guaranteed script hook in that format.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..logging_config import setup_logger

logger = setup_logger("realign.codex_hooks.installer", "codex_hooks_installer.log")

ALINE_HOOK_MARKER = "aline-codex-notify-hook"


def get_notify_hook_script_path() -> Path:
    return Path(__file__).parent / "notify_hook.py"


def get_notify_hook_command_parts() -> list[str]:
    script_path = get_notify_hook_script_path()
    return [sys.executable, str(script_path)]


def _toml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _format_notify_toml(cmd: list[str]) -> str:
    # Codex CLI expects notify as a string (shell command), not an array.
    # NOTE: we intentionally do not attempt complex quoting here; the common case is
    # paths without spaces (e.g. /opt/homebrew/...).
    command_str = " ".join(cmd)
    return f"notify = \"{_toml_escape(command_str)}\"  # {ALINE_HOOK_MARKER}\n"


def _update_toml_linewise(path: Path, *, cmd: list[str]) -> bool:
    """
    Update config.toml in a minimal, formatting-preserving way (line-based).

    Returns True if the file was written/updated.
    """
    desired = _format_notify_toml(cmd)
    existing = ""
    try:
        existing = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = ""
    except Exception:
        return False

    lines = existing.splitlines(keepends=True) if existing else []
    out: list[str] = []
    replaced = False

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("notify ="):
            if not replaced:
                out.append(desired)
                replaced = True
            else:
                # Drop duplicate notify lines.
                continue
        else:
            out.append(line)

    if not replaced:
        if out and not out[-1].endswith("\n"):
            out[-1] = out[-1] + "\n"
        if out and out[-1].strip():
            out.append("\n")
        out.append(desired)

    new_text = "".join(out)
    if new_text == existing:
        return True

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_text, encoding="utf-8")
        return True
    except Exception:
        return False


def _ensure_legacy_notify_enabled(codex_home: Path) -> list[Path]:
    """
    Best-effort for legacy Codex config formats (YAML/JSON):
    set `notify: true` / `"notify": true` if the file exists.
    """
    updated: list[Path] = []
    yaml_path = codex_home / "config.yaml"
    json_path = codex_home / "config.json"

    if yaml_path.exists():
        try:
            raw = yaml_path.read_text(encoding="utf-8")
            if "notify:" in raw:
                # Minimal replace: notify: <anything> -> notify: true
                out_lines: list[str] = []
                for line in raw.splitlines():
                    if line.strip().startswith("notify:"):
                        out_lines.append("notify: true")
                    else:
                        out_lines.append(line)
                new_raw = "\n".join(out_lines) + ("\n" if raw.endswith("\n") else "")
            else:
                new_raw = raw + ("\n" if raw and not raw.endswith("\n") else "") + "notify: true\n"
            if new_raw != raw:
                yaml_path.write_text(new_raw, encoding="utf-8")
            updated.append(yaml_path)
        except Exception:
            pass

    if json_path.exists():
        try:
            obj = json.loads(json_path.read_text(encoding="utf-8") or "{}")
            if isinstance(obj, dict):
                if obj.get("notify") is not True:
                    obj["notify"] = True
                    json_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                updated.append(json_path)
        except Exception:
            pass

    return updated


def ensure_notify_hook_installed_for_codex_home(
    codex_home: Path, *, quiet: bool = True
) -> bool:
    """
    Ensure the notify hook is installed for a given CODEX_HOME.

    - Rust CLI: writes/updates CODEX_HOME/config.toml notify command.
    - Legacy: enables notify=true if config.yaml/config.json exist.
    """
    codex_home = Path(codex_home).expanduser()
    cmd = get_notify_hook_command_parts()

    ok = False
    toml_path = codex_home / "config.toml"
    if _update_toml_linewise(toml_path, cmd=cmd):
        ok = True
        if not quiet:
            print(f"[Aline] Codex notify hook installed: {toml_path}", file=sys.stderr)
    _ensure_legacy_notify_enabled(codex_home)
    return ok


def ensure_global_codex_notify_hook_installed(*, quiet: bool = True) -> bool:
    """Best-effort: install notify hook into default global CODEX_HOME (~/.codex)."""
    return ensure_notify_hook_installed_for_codex_home(Path.home() / ".codex", quiet=quiet)


def ensure_all_aline_codex_homes_notify_hook_installed(*, quiet: bool = True) -> int:
    """
    Best-effort: install notify hook into every Aline-managed CODEX_HOME under ~/.aline/codex_homes.
    Returns number of homes updated.
    """
    try:
        from ..codex_home import aline_codex_homes_dir

        root = aline_codex_homes_dir()
    except Exception:
        root = Path.home() / ".aline" / "codex_homes"

    if not root.exists():
        return 0

    updated = 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        # Layouts:
        # - <terminal_id>/
        # - agent-<id>/<terminal_id>/
        if child.name.startswith("agent-"):
            try:
                for grandchild in child.iterdir():
                    if grandchild.is_dir():
                        if ensure_notify_hook_installed_for_codex_home(grandchild, quiet=quiet):
                            updated += 1
            except Exception:
                continue
        else:
            if ensure_notify_hook_installed_for_codex_home(child, quiet=quiet):
                updated += 1
    return updated


def codex_cli_supports_notify_hook(*, timeout_seconds: float = 0.5) -> Optional[bool]:
    """
    Best-effort detect whether the installed `codex` binary supports the Rust notify hook.

    Returns:
      - True: looks like Rust Codex CLI (supports config.toml + notify command)
      - False: looks like legacy Codex (no reliable script notify hook)
      - None: codex binary not found
    """
    if shutil.which("codex") is None:
        return None

    def run(args: list[str]) -> str:
        try:
            proc = subprocess.run(
                args,
                text=True,
                capture_output=True,
                check=False,
                timeout=float(timeout_seconds),
            )
            return f"{proc.stdout}\n{proc.stderr}".strip().lower()
        except Exception:
            return ""

    help_out = run(["codex", "--help"])
    ver_out = run(["codex", "--version"])
    out = (help_out + "\n" + ver_out).lower()

    # Strong positive indicators for Rust CLI.
    if "config.toml" in out and "notify" in out:
        return True
    # Avoid false positives (e.g. "trusted" contains the substring "rust").
    if "codex-rs" in out or re.search(r"\brust\b", out):
        return True

    # Legacy indicators (YAML/JSON config keys from older docs/CLI).
    if "config.yaml" in out or "approvalmode" in out or "fullautoerrormode" in out:
        return False

    # Conservative default: if we cannot confirm, treat as unsupported so we don't
    # claim Codex integration works when it won't.
    return False
