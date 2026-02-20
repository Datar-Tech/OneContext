"""Utility functions for detecting and integrating with Claude Code."""

import os
from pathlib import Path
from typing import Optional


def get_claude_project_name(project_path: Path) -> str:
    """
    Convert a project path to Claude Code's project directory name format.

    Claude Code transforms project paths by replacing '/' with '-' (excluding root '/').
    For example: /Users/alice/Projects/MyApp -> -Users-alice-Projects-MyApp

    Args:
        project_path: The absolute path to the project

    Returns:
        The transformed project name used by Claude Code
    """
    # Convert to absolute path and normalize
    abs_path = project_path.resolve()

    # Convert path to string and replace separators
    # Remove leading '/' and replace all '/' and '_' with '-'
    # Claude Code replaces both slashes and underscores with dashes
    path_str = str(abs_path)
    if path_str.startswith("/"):
        path_str = path_str[1:]

    return "-" + path_str.replace("/", "-").replace("_", "-")


def find_claude_sessions_dir(project_path: Path) -> Optional[Path]:
    """
    Find the Claude Code sessions directory for a given project path.

    Claude Code stores project sessions in:
    ~/.claude/projects/{project_name}/

    where project_name is the project path with '/' replaced by '-'.

    Args:
        project_path: The absolute path to the project

    Returns:
        Path to the Claude sessions directory if it exists, None otherwise
    """
    # Get Claude base directory
    claude_base = Path.home() / ".claude" / "projects"

    if not claude_base.exists():
        return None

    # Get the project name in Claude's format
    project_name = get_claude_project_name(project_path)

    # Check if the project directory exists
    claude_project_dir = claude_base / project_name

    if claude_project_dir.exists() and claude_project_dir.is_dir():
        return claude_project_dir

    return None


def auto_detect_sessions_path(project_path: Path, fallback_path: Optional[str] = None) -> Path:
    """
    Auto-detect the best sessions path to use.

    Priority:
    1. Claude Code sessions directory (if exists)
    2. fallback_path parameter (if provided)
    3. Default: ~/.local/share/realign/histories

    Args:
        project_path: The absolute path to the project
        fallback_path: Optional fallback path if auto-detection fails

    Returns:
        Path to use for session history
    """
    # Try to find Claude sessions directory
    claude_dir = find_claude_sessions_dir(project_path)
    if claude_dir:
        return claude_dir

    # Use fallback if provided
    if fallback_path:
        return Path(os.path.expanduser(fallback_path))

    # Default path
    return Path.home() / ".local" / "share" / "realign" / "histories"
