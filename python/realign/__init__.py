"""Aline - AI Agent Chat Session Tracker."""

import hashlib
from pathlib import Path

__version__ = "0.8.3"


def get_realign_dir(project_root: Path) -> Path:
    """
    Get the per-project Aline data directory.

    Resolution order:
    1) `<project_root>/.aline-config` (preferred; optional override)
    2) `<project_root>/.realign-config` (legacy override)
    3) `<project_root>/.realign` (legacy on-disk directory)
    4) Default: `~/.aline/projects/<project_name>-<hash>/`

    Args:
        project_root: Root directory of the project.

    Returns:
        Path to the project's Aline data directory.
    """
    # Use absolute path but avoid resolving symlinks (macOS /private vs /var)
    project_root = Path(project_root).absolute()

    aline_config = project_root / ".aline-config"
    legacy_config = project_root / ".realign-config"

    for marker in (aline_config, legacy_config):
        try:
            if marker.exists():
                configured_path = marker.read_text(encoding="utf-8").strip()
                if configured_path:
                    return Path(configured_path).expanduser()
        except Exception:
            # Ignore unreadable marker files and fall back to defaults.
            pass

    legacy_local_dir = project_root / ".realign"
    if legacy_local_dir.exists():
        return legacy_local_dir

    # Default: deterministic per-project dir under ~/.aline/projects/
    digest = hashlib.sha1(str(project_root).encode("utf-8")).hexdigest()[:10]
    project_dir_name = f"{project_root.name}-{digest}"
    return Path.home() / ".aline" / "projects" / project_dir_name
