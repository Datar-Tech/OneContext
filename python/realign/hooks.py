#!/usr/bin/env python3
"""
ReAlign Git Hooks - Entry points for git hook commands.

This module provides the hook functionality as Python commands that can be
invoked directly from git hooks without copying any Python files to the target repository.
"""

import os
import re
import sys
import json
import time
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Callable

from .config import ReAlignConfig
from .adapters import get_adapter_registry
from .claude_detector import find_claude_sessions_dir
from .dashboard_tracking import (
    filter_discovered_sessions_for_dashboard_only,
    is_dashboard_only_enabled,
)
from .logging_config import setup_logger
from .llm_client import extract_json, call_llm_cloud

try:
    from .redactor import check_and_redact_session, save_original_session

    REDACTOR_AVAILABLE = True
except ImportError:
    REDACTOR_AVAILABLE = False

# Initialize logger for hooks
logger = setup_logger("realign.hooks", "hooks.log")

DEFAULT_METADATA_PROMPT_TEXT = """You are a metadata classifier for AI-assisted git commits.
Determine two fields for the current turn:
1. "if_last_task": "yes" if this turn continues, debugs, or fixes the previous task. "no" if it is a new task.
2. "satisfaction": "good" when the user is clearly happy or moving on, "fine" when progress is partial/mixed, "bad" when the user reports failure or dissatisfaction.

Rules:
- Prioritize the latest user request/feedback for satisfaction.
- Use the assistant recap + recent commit context to decide if this turn continues prior work.
- If uncertain, default to "no" and "fine".
- Respond with JSON ONLY."""

_METADATA_PROMPT_CACHE: Optional[str] = None
_COMMIT_MESSAGE_PROMPT_CACHE: Optional[str] = None

# Global variable to store the last LLM error
_last_llm_error: Optional[str] = None


def set_last_llm_error(error: str) -> None:
    """Record the last LLM API error."""
    global _last_llm_error
    _last_llm_error = error


def get_last_llm_error() -> Optional[str]:
    """Get the last LLM API error."""
    return _last_llm_error


def _emit_llm_debug(
    callback: Optional[Callable[[Dict[str, Any]], None]], payload: Dict[str, Any]
) -> None:
    if not callback:
        return
    try:
        callback(payload)
    except Exception:
        logger.debug("LLM debug callback failed for payload=%s", payload, exc_info=True)


def _normalize_if_last_task(raw_value: Any) -> str:
    """Normalize if_last_task values from LLM output."""
    if isinstance(raw_value, bool):
        return "yes" if raw_value else "no"
    if isinstance(raw_value, str):
        lower = raw_value.lower().strip()
        if lower in ("yes", "true", "1", "continue", "continued"):
            return "yes"
        if lower in ("no", "false", "0", "new", "fresh"):
            return "no"
        if any(
            keyword in lower for keyword in ["still", "again", "continue", "follow-up", "follow up"]
        ):
            return "yes"
    return "no"


def _normalize_satisfaction(raw_value: Any) -> str:
    """Normalize satisfaction values from LLM output."""
    if isinstance(raw_value, (int, float)):
        logger.warning("LLM returned numeric satisfaction (%s), defaulting to 'fine'", raw_value)
        return "fine"
    if isinstance(raw_value, str):
        lower = raw_value.lower().strip()
        if lower in ("good", "fine", "bad"):
            return lower
        positive = [
            "great",
            "perfect",
            "excellent",
            "works",
            "thanks",
            "done",
            "awesome",
            "success",
            "completed",
        ]
        negative = [
            "bad",
            "fail",
            "error",
            "broken",
            "didn't work",
            "not working",
            "no,",
            "no ",
            "still broken",
        ]
        mixed = [
            "but",
            "still",
            "however",
            "almost",
            "not quite",
            "partial",
            "some",
            "kinda",
            "kind of",
        ]
        if any(keyword in lower for keyword in positive):
            return "good"
        if any(keyword in lower for keyword in negative):
            return "bad"
        if any(keyword in lower for keyword in mixed):
            return "fine"
    if isinstance(raw_value, bool):
        return "good" if raw_value else "bad"
    return "fine"


def _classify_task_metadata(
    *,
    provider: str,
    user_messages: str,
    assistant_replies: str,
    code_changes: str,
    summary_title: Optional[str],
    summary_description: Optional[str],
    debug_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    system_prompt: Optional[str] = None,
    previous_commit_title: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Run a dedicated LLM classification pass for if_last_task and satisfaction tags.

    NOTE: LLM-based metadata classification is disabled. Always returns defaults.
    """
    # Metadata LLM classification disabled - always return defaults
    return ("no", "fine")

    # =========================================================================
    # LOCAL LLM FALLBACK DISABLED - Code kept for reference
    # =========================================================================
    # def _get_metadata_prompt() -> str:
    #     global _METADATA_PROMPT_CACHE
    #     if system_prompt is not None:
    #         return system_prompt
    #     if _METADATA_PROMPT_CACHE is not None:
    #         return _METADATA_PROMPT_CACHE
    #
    #     # Try user-customized prompt first (~/.aline/prompts/metadata.md)
    #     user_prompt_path = Path.home() / ".aline" / "prompts" / "metadata.md"
    #     try:
    #         if user_prompt_path.exists():
    #             text = user_prompt_path.read_text(encoding="utf-8").strip()
    #             if text:
    #                 _METADATA_PROMPT_CACHE = text
    #                 logger.debug(f"Loaded user-customized metadata prompt from {user_prompt_path}")
    #                 return text
    #     except Exception:
    #         logger.debug(
    #             "Failed to load user-customized metadata prompt, falling back", exc_info=True
    #         )
    #
    #     # Fall back to built-in prompt (tools/commit_message_prompts/metadata_default.md)
    #     candidate = (
    #         Path(__file__).resolve().parents[2]
    #         / "tools"
    #         / "commit_message_prompts"
    #         / "metadata_default.md"
    #     )
    #     try:
    #         text = candidate.read_text(encoding="utf-8").strip()
    #         if text:
    #             _METADATA_PROMPT_CACHE = text
    #             return text
    #     except Exception:
    #         logger.debug("Falling back to built-in metadata prompt", exc_info=True)
    #     _METADATA_PROMPT_CACHE = DEFAULT_METADATA_PROMPT_TEXT
    #     return _METADATA_PROMPT_CACHE
    #
    # classification_system_prompt = _get_metadata_prompt()
    #
    # prompt_parts: List[str] = [
    #     f"Previous commit title: {previous_title}",
    #     "User request:\n" + clipped_user,
    #     f"Current commit title: {current_title}",
    #     'Return strict JSON with exactly these fields:\n{"if_last_task": "yes|no", "satisfaction": "good|fine|bad"}',
    # ]
    # user_prompt = "\n\n".join(prompt_parts)
    #
    # model_name, response_text = _invoke_llm(
    #     provider=provider,
    #     system_prompt=classification_system_prompt,
    #     user_prompt=user_prompt,
    #     debug_callback=debug_callback,
    #     purpose="metadata",
    # )
    # if not response_text:
    #     return defaults
    #
    # try:
    #     metadata = _extract_json_object(response_text)
    # except json.JSONDecodeError as exc:
    #     logger.warning("Failed to parse metadata JSON: %s", exc)
    #     logger.debug("Raw metadata response: %s", response_text)
    #     return defaults
    #
    # if_last_task = _normalize_if_last_task(metadata.get("if_last_task"))
    # satisfaction = _normalize_satisfaction(metadata.get("satisfaction"))
    # logger.info("LLM metadata response: %s", json.dumps(metadata, ensure_ascii=False))
    # print(
    #     f"   🔍 Metadata classification: if_last_task={metadata.get('if_last_task')}→{if_last_task}, "
    #     f"satisfaction={metadata.get('satisfaction')}→{satisfaction}",
    #     file=sys.stderr,
    # )
    # if model_name:
    #     print(f"   ✅ LLM metadata classification successful ({model_name})", file=sys.stderr)
    # else:
    #     print("   ✅ LLM metadata classification successful", file=sys.stderr)
    # return if_last_task, satisfaction


# ============================================================================
# Message Cleaning Utilities
# ============================================================================


def clean_user_message(text: str) -> str:
    """
    Clean user message by removing IDE context tags and other system noise.

    This function removes IDE-generated context that's not part of the actual
    user intent, making commit messages and session logs cleaner.

    Removes:
    - <ide_opened_file>...</ide_opened_file> tags
    - <ide_selection>...</ide_selection> tags
    - System interrupt messages like "[Request interrupted by user for tool use]"
    - Other system-generated context tags

    Args:
        text: Raw user message text

    Returns:
        Cleaned message text with system tags removed, or empty string if message is purely system-generated
    """
    if not text:
        return text

    # Check for system interrupt messages first (return empty for these)
    # These are generated when user stops the AI mid-execution
    if text.strip() == "[Request interrupted by user for tool use]":
        return ""

    # Remove IDE opened file tags
    text = re.sub(r"<ide_opened_file>.*?</ide_opened_file>\s*", "", text, flags=re.DOTALL)

    # Remove IDE selection tags
    text = re.sub(r"<ide_selection>.*?</ide_selection>\s*", "", text, flags=re.DOTALL)

    # Remove other common system tags if needed
    # text = re.sub(r'<system_context>.*?</system_context>\s*', '', text, flags=re.DOTALL)

    # Clean up extra whitespace
    text = re.sub(
        r"\n\s*\n\s*\n+", "\n\n", text
    )  # Replace multiple blank lines with double newline
    text = text.strip()

    return text


def get_new_content_from_git_diff(repo_root: Path, session_relpath: str) -> str:
    """
    Extract new content added in this commit by using git diff.
    Returns the raw text of all added lines, without parsing.

    Args:
        repo_root: Path to git repository root
        session_relpath: Relative path to session file in repo (e.g. ".realign/sessions/xxx.jsonl")

    Returns:
        String containing all new content added in this commit
    """
    logger.debug(f"Extracting new content from git diff for: {session_relpath}")

    try:
        # Use --cached to check staged changes (what will be committed)
        # This compares the staging area with HEAD
        result = subprocess.run(
            ["git", "diff", "--cached", "--", session_relpath],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            logger.warning(f"Git diff command failed with return code {result.returncode}")
            return ""

        # If no diff output, this file has no staged changes in this commit
        if not result.stdout.strip():
            logger.debug(f"No staged changes found for: {session_relpath}")
            return ""

        # Parse diff output to extract added lines
        new_lines = []
        for line in result.stdout.split("\n"):
            # Lines starting with '+' (but not '+++') are additions
            if line.startswith("+") and not line.startswith("+++"):
                # Remove the '+' prefix
                new_lines.append(line[1:])

        content = "\n".join(new_lines)
        logger.info(
            f"Extracted {len(new_lines)} new lines ({len(content)} bytes) from {session_relpath}"
        )
        return content

    except subprocess.TimeoutExpired:
        logger.error(f"Git diff command timed out for: {session_relpath}")
        print("Warning: git diff command timed out", file=sys.stderr)
        return ""
    except Exception as e:
        logger.error(f"Failed to extract new content from git diff: {e}", exc_info=True)
        print(f"Warning: Could not extract new content from git diff: {e}", file=sys.stderr)
        return ""


def get_claude_project_name(project_path: Path) -> str:
    """
    Convert a project path to Claude Code's project directory name format.

    Claude Code transforms project paths by replacing '/' with '-' (excluding root '/').
    For example: /Users/alice/Projects/MyApp -> -Users-alice-Projects-MyApp
    """
    from .claude_detector import get_claude_project_name as _get_name

    return _get_name(project_path)


def find_codex_latest_session(project_path: Path, days_back: int = 7) -> Optional[Path]:
    """
    Find the most recent Codex session for a given project path.

    Codex stores sessions in ~/.codex/sessions/{YYYY}/{MM}/{DD}/
    with all projects mixed together. We need to search by date
    and filter by the 'cwd' field in session metadata.

    Args:
        project_path: The absolute path to the project
        days_back: Number of days to look back (default: 7)

    Returns:
        Path to the most recent session file, or None if not found
    """
    from .codex_detector import get_latest_codex_session

    logger.debug(f"Searching for Codex sessions for project: {project_path}")
    return get_latest_codex_session(project_path, days_back=days_back)


def find_all_claude_sessions() -> List[Path]:
    """
    Find all active Claude Code sessions from ALL projects.

    (Legacy wrapper for ClaudeAdapter)
    """
    adapter = get_adapter_registry().get_adapter("claude")
    if adapter:
        return adapter.discover_sessions()
    return []


def find_all_codex_sessions(days_back: int = 1) -> List[Path]:
    """
    Find all active Codex sessions from recent days.

    (Legacy wrapper for CodexAdapter)
    """
    adapter = get_adapter_registry().get_adapter("codex")
    if adapter:
        # Note: adapters currently use their internal logic for discovery
        return adapter.discover_sessions()
    return []


def find_all_gemini_cli_sessions() -> List[Path]:
    """
    Find all active Gemini CLI sessions.

    (Legacy wrapper for GeminiAdapter)
    """
    adapter = get_adapter_registry().get_adapter("gemini")
    if adapter:
        return adapter.discover_sessions()
    return []


def find_all_active_sessions(
    config: ReAlignConfig, project_path: Optional[Path] = None
) -> List[Path]:
    """
    Find all active session files based on enabled auto-detection options.

    Uses current adapter-based architecture.
    """
    logger.info("Searching for active AI sessions")
    logger.debug(
        f"Config: auto_detect_codex={config.auto_detect_codex}, auto_detect_claude={config.auto_detect_claude}"
    )
    logger.debug(f"Project path: {project_path}")

    sessions = []
    dashboard_only = is_dashboard_only_enabled(config)
    dashboard_db = None

    if dashboard_only:
        try:
            from .db import get_database

            dashboard_db = get_database(read_only=True)
        except Exception:
            dashboard_db = None

    # 1. Use AdapterRegistry for discovery
    registry = get_adapter_registry()

    # Map config options to adapter names
    enabled_adapters = []
    if config.auto_detect_claude:
        enabled_adapters.append("claude")
    if config.auto_detect_codex:
        enabled_adapters.append("codex")
    if config.auto_detect_gemini:
        enabled_adapters.append("gemini")

    try:
        for name in enabled_adapters:
            adapter = registry.get_adapter(name)
            if not adapter:
                continue

            try:
                if project_path:
                    # Single-project mode
                    project_sessions = adapter.discover_sessions_for_project(project_path)
                    if dashboard_only and name in {"claude", "codex"}:
                        project_sessions = filter_discovered_sessions_for_dashboard_only(
                            session_type=name,
                            sessions=project_sessions,
                            db=dashboard_db,
                        )
                    sessions.extend(project_sessions)
                else:
                    # Multi-project mode
                    all_project_sessions = adapter.discover_sessions()
                    if dashboard_only and name in {"claude", "codex"}:
                        all_project_sessions = filter_discovered_sessions_for_dashboard_only(
                            session_type=name,
                            sessions=all_project_sessions,
                            db=dashboard_db,
                        )
                    sessions.extend(all_project_sessions)
            except Exception as e:
                logger.warning(f"Error in adapter {name} discovery: {e}")
    finally:
        if dashboard_db is not None:
            try:
                dashboard_db.close()
            except Exception:
                pass

    logger.info(f"Session discovery complete: found {len(sessions)} session(s)")
    return sessions


def find_latest_session(history_path: Path, explicit_path: Optional[str] = None) -> Optional[Path]:
    """
    Find the most recent session file.

    Filters out Claude Code agent sessions (agent-*.jsonl) since they are
    sub-tasks of main sessions and their results are already incorporated
    into the main session files.

    Args:
        history_path: Path to history directory or a specific session file (for Codex)
        explicit_path: Explicit path to a session file (overrides history_path)

    Returns:
        Path to the session file, or None if not found
    """
    if explicit_path:
        session_file = Path(explicit_path)
        if session_file.exists():
            return session_file
        return None

    # Expand user path
    history_path = (
        Path(os.path.expanduser(history_path)) if isinstance(history_path, str) else history_path
    )

    if not history_path.exists():
        return None

    # If history_path is already a file (e.g., Codex session), return it directly
    if history_path.is_file():
        return history_path

    # Otherwise, search directory for session files
    session_files = []
    for pattern in ["*.json", "*.jsonl"]:
        for file in history_path.glob(pattern):
            # Filter out Claude Code agent sessions (agent-*.jsonl)
            # These are sub-tasks whose results are already in main sessions
            if file.name.startswith("agent-"):
                logger.debug(f"Skipping agent session: {file.name}")
                continue
            session_files.append(file)

    if not session_files:
        return None

    # Return most recently modified
    return max(session_files, key=lambda p: p.stat().st_mtime)


def filter_session_content(content: str) -> Tuple[str, str, str]:
    """
    Filter session content to extract meaningful information for LLM summarization.

    Filters out exploratory operations (Read, Grep, Glob) and technical details,
    keeping only user requests, ALL AI responses (thinking + text), and code changes.

    Supports both Claude Code and Codex session formats.

    Args:
        content: Raw text content of new session additions

    Returns:
        Tuple of (user_messages, assistant_replies, code_changes)
    """
    if not content or not content.strip():
        return "", "", ""

    user_messages = []
    assistant_replies = []
    code_changes = []

    lines = content.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)

            # Extract user messages and tool results
            if obj.get("type") == "user":
                msg = obj.get("message", {})
                if isinstance(msg, dict):
                    content_data = msg.get("content", "")
                    if isinstance(content_data, str) and content_data.strip():
                        user_messages.append(content_data.strip())
                    elif isinstance(content_data, list):
                        # Extract text from content list
                        for item in content_data:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    text = item.get("text", "").strip()
                                    if text:
                                        user_messages.append(text)
                                # Extract code changes from tool results
                                elif item.get("type") == "tool_result":
                                    tool_use_result = obj.get("toolUseResult", {})
                                    if (
                                        "oldString" in tool_use_result
                                        and "newString" in tool_use_result
                                    ):
                                        # This is an Edit operation
                                        new_string = tool_use_result.get("newString", "")
                                        if new_string:
                                            code_changes.append(f"Edit: {new_string[:300]}")
                                    elif (
                                        "content" in tool_use_result
                                        and "filePath" in tool_use_result
                                    ):
                                        # This is a Write operation
                                        new_content = tool_use_result.get("content", "")
                                        if new_content:
                                            code_changes.append(f"Write: {new_content[:300]}")

            # Extract assistant text replies (including thinking, but not tool use)
            elif obj.get("type") == "assistant":
                msg = obj.get("message", {})
                if isinstance(msg, dict):
                    content_data = msg.get("content", [])
                    if isinstance(content_data, list):
                        for item in content_data:
                            if isinstance(item, dict):
                                # Extract text blocks
                                if item.get("type") == "text":
                                    text = item.get("text", "").strip()
                                    if text:
                                        assistant_replies.append(text)
                                # Extract thinking blocks
                                elif item.get("type") == "thinking":
                                    thinking_text = item.get("thinking", "").strip()
                                    if thinking_text:
                                        # Prefix thinking with a marker for clarity
                                        assistant_replies.append(f"[Thinking] {thinking_text}")
                                # Extract code changes from Edit/Write tool uses
                                elif item.get("type") == "tool_use":
                                    tool_name = item.get("name", "")
                                    if tool_name in ("Edit", "Write"):
                                        params = item.get("input", {})
                                        if tool_name == "Edit":
                                            new_string = params.get("new_string", "")
                                            if new_string:
                                                code_changes.append(f"Edit: {new_string[:200]}")
                                        elif tool_name == "Write":
                                            new_content = params.get("content", "")
                                            if new_content:
                                                code_changes.append(f"Write: {new_content[:200]}")

            # Also handle simple role/content format (for compatibility)
            elif obj.get("role") == "user":
                content_text = obj.get("content", "")
                if isinstance(content_text, str) and content_text.strip():
                    user_messages.append(content_text.strip())

            elif obj.get("role") == "assistant":
                content_text = obj.get("content", "")
                if isinstance(content_text, str) and content_text.strip():
                    assistant_replies.append(content_text.strip())

            # Handle Codex format
            elif obj.get("type") == "response_item":
                payload = obj.get("payload", {})
                payload_type = payload.get("type", "")

                # Codex user messages
                if payload_type == "message":
                    role = payload.get("role", "")
                    if role == "user":
                        content_list = payload.get("content", [])
                        if isinstance(content_list, list):
                            for content_item in content_list:
                                if (
                                    isinstance(content_item, dict)
                                    and content_item.get("type") == "input_text"
                                ):
                                    text = content_item.get("text", "").strip()
                                    if text:
                                        user_messages.append(text)

                # Codex reasoning (encrypted - skip, we'll use agent_reasoning instead)
                # elif payload_type == "reasoning":
                #     pass  # Skip encrypted reasoning

            # Handle Codex event_msg with agent_reasoning (visible thinking)
            elif obj.get("type") == "event_msg":
                payload = obj.get("payload", {})
                payload_type = payload.get("type", "")

                if payload_type == "agent_reasoning":
                    text = payload.get("text", "").strip()
                    if text:
                        # Prefix with [Thinking] for consistency
                        assistant_replies.append(f"[Thinking] {text}")

        except (json.JSONDecodeError, KeyError, TypeError):
            # Not JSON or doesn't have expected structure, skip
            continue

    # Keep ALL thinking and responses (not filtered to first+last)
    # This provides complete context of the assistant's reasoning process
    # Join with newlines for better readability
    user_str = "\n".join(user_messages) if user_messages else ""
    assistant_str = "\n\n".join(assistant_replies) if assistant_replies else ""
    code_str = "\n".join(code_changes) if code_changes else ""

    return user_str, assistant_str, code_str


def simple_summarize(content: str, max_chars: int = 500) -> str:
    """
    Generate a simple summary from new session content.
    Extracts key information without LLM.

    Args:
        content: Raw text content of new session additions
        max_chars: Maximum characters in summary
    """
    if not content or not content.strip():
        return "No new content in this session"

    lines = content.strip().split("\n")

    # Try to extract meaningful content from JSONL format
    summaries = []
    for line in lines[:10]:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            # Extract summary from special summary lines
            if obj.get("type") == "summary" and obj.get("summary"):
                summaries.append(f"Summary: {obj.get('summary')}")
            # Extract message content from user/assistant messages (complex format)
            elif obj.get("type") in ("user", "assistant") and obj.get("message"):
                msg = obj.get("message")
                if isinstance(msg, dict) and msg.get("content"):
                    content_text = msg.get("content")
                    if isinstance(content_text, str):
                        summaries.append(content_text[:100])
                    elif isinstance(content_text, list):
                        for item in content_text:
                            if isinstance(item, dict) and item.get("type") == "text":
                                summaries.append(item.get("text", "")[:100])
                                break
            # Also handle simple role/content format (for compatibility)
            elif obj.get("role") in ("user", "assistant") and obj.get("content"):
                content_text = obj.get("content")
                if isinstance(content_text, str):
                    summaries.append(content_text[:100])
        except (json.JSONDecodeError, KeyError, TypeError):
            # Not JSON or doesn't have expected structure, try raw text
            if len(line) > 20:
                summaries.append(line[:100])

    if summaries:
        summary = " | ".join(summaries[:3])
        return summary[:max_chars]

    # Fallback: surface the first few non-empty raw lines to give context
    fallback_lines = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        # Skip noisy JSON braces-only lines
        if stripped in ("{", "}", "[", "]"):
            continue
        fallback_lines.append(stripped[:120])
        if len(fallback_lines) == 3:
            break

    if fallback_lines:
        summary = " | ".join(fallback_lines)
        return summary[:max_chars]

    return f"Session updated with {len(lines)} new lines"


def detect_agent_from_session_path(session_relpath: str) -> str:
    """Infer agent type based on session filename."""
    lower_path = session_relpath.lower()

    if "codex" in lower_path or "rollout-" in lower_path:
        return "Codex"
    if "claude" in lower_path or "agent-" in lower_path:
        return "Claude"
    if lower_path.endswith(".jsonl"):
        # Default to Unknown to avoid mislabeling generic files
        return "Unknown"
    return "Unknown"


def generate_summary_with_llm(
    content: str,
    max_chars: int = 500,
    provider: str = "auto",
    system_prompt: Optional[str] = None,
    debug_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    metadata_system_prompt: Optional[str] = None,
    previous_commit_title: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str], str, str]:
    """
    Generate a structured summary for session content using an LLM.

    The summary call only produces title + description. A second LLM call
    classifies if_last_task/satisfaction so that each prompt can stay focused.
    """
    logger.info("Attempting to generate LLM summary (provider: %s)", provider)

    if not content or not content.strip():
        logger.debug("No content provided for summarization")
        return "No new content in this session", None, "", "no", "fine"

    user_messages, assistant_replies, code_changes = filter_session_content(content)

    if not user_messages and not assistant_replies and not code_changes:
        logger.debug("No meaningful content after filtering")
        return (
            "Session update with no significant changes",
            None,
            "No significant changes detected in this session",
            "no",
            "fine",
        )

    # Cloud LLM is the supported path. Historically, `provider` selected a direct
    # provider ("claude" / "openai") but local fallbacks are disabled. Treat
    # those values as cloud-compatible so existing configs keep working.
    cloud_compatible_providers = {"auto", "cloud", "claude", "openai"}
    if provider in cloud_compatible_providers:
        try:
            from .auth import is_logged_in

            if is_logged_in():
                logger.debug("Attempting cloud LLM for summary generation")
                # Load user custom prompt if available
                custom_prompt = None
                if system_prompt is not None:
                    custom_prompt = system_prompt
                else:
                    user_prompt_path = Path.home() / ".aline" / "prompts" / "commit_message.md"
                    try:
                        if user_prompt_path.exists():
                            custom_prompt = user_prompt_path.read_text(encoding="utf-8").strip()
                    except Exception:
                        pass

                model_name, result = call_llm_cloud(
                    task="summary",
                    payload={
                        "user_messages": user_messages[:4000],
                        "assistant_replies": assistant_replies[:8000],
                    },
                    custom_prompt=custom_prompt,
                    silent=False,
                )

                if result:
                    title = result.get("title", "")
                    description = result.get("description", "")
                    logger.info("Cloud LLM summary success: title=%s", title[:50] if title else "")

                    # Now classify metadata using cloud
                    if_last_task, satisfaction = _classify_task_metadata(
                        provider=provider,
                        user_messages=user_messages,
                        assistant_replies=assistant_replies,
                        code_changes=code_changes,
                        summary_title=title,
                        summary_description=description,
                        debug_callback=debug_callback,
                        system_prompt=metadata_system_prompt,
                        previous_commit_title=previous_commit_title,
                    )

                    return title, model_name, description, if_last_task, satisfaction
                else:
                    # Cloud LLM failed, return None (local fallback disabled)
                    logger.warning("Cloud LLM summary failed, returning None")
                    print("   ⚠️  Cloud LLM summary failed", file=sys.stderr)
                    set_last_llm_error("Cloud LLM summary call failed")
                    return None, None, None, None, None
        except ImportError:
            logger.debug("Auth module not available, skipping cloud LLM")
            set_last_llm_error("Cloud LLM unavailable: auth module not available")
            return None, None, None, None, None

        # If we got here, auth is available but user isn't logged in.
        logger.warning("Not logged in, cannot use cloud LLM for summary")
        print("   ⚠️  Please login with 'aline login' to use LLM features", file=sys.stderr)
        set_last_llm_error("Not logged in to Aline cloud")
        return None, None, None, None, None

    logger.warning(
        "LLM provider '%s' is not cloud-compatible; local LLM providers are disabled",
        provider,
    )
    print(
        "   ⚠️  Local LLM providers are disabled. Set REALIGN_LLM_PROVIDER=auto (or 'cloud')",
        file=sys.stderr,
    )
    set_last_llm_error(f"Unsupported LLM provider in cloud-only mode: {provider}")
    return None, None, None, None, None

    # =========================================================================
    # LOCAL LLM FALLBACK DISABLED - Code kept for reference
    # =========================================================================
    # # Fall back to local LLM call
    # # Try to load system prompt from default.md file
    # def _get_commit_message_prompt() -> str:
    #     global _COMMIT_MESSAGE_PROMPT_CACHE
    #     if system_prompt is not None:
    #         return system_prompt
    #     if _COMMIT_MESSAGE_PROMPT_CACHE is not None:
    #         return _COMMIT_MESSAGE_PROMPT_CACHE
    #
    #     # Try user-customized prompt first (~/.aline/prompts/commit_message.md)
    #     user_prompt_path = Path.home() / ".aline" / "prompts" / "commit_message.md"
    #     try:
    #         if user_prompt_path.exists():
    #             text = user_prompt_path.read_text(encoding="utf-8").strip()
    #             if text:
    #                 _COMMIT_MESSAGE_PROMPT_CACHE = text
    #                 logger.debug(
    #                     f"Loaded user-customized commit message prompt from {user_prompt_path}"
    #                 )
    #                 return text
    #     except Exception:
    #         logger.debug(
    #             "Failed to load user-customized commit message prompt, falling back", exc_info=True
    #         )
    #
    #     # Fall back to built-in prompt (tools/commit_message_prompts/default.md)
    #     candidate = (
    #         Path(__file__).resolve().parents[2] / "tools" / "commit_message_prompts" / "default.md"
    #     )
    #     try:
    #         text = candidate.read_text(encoding="utf-8").strip()
    #         if text:
    #             _COMMIT_MESSAGE_PROMPT_CACHE = text
    #             logger.debug(f"Loaded commit message prompt from {candidate}")
    #             return text
    #     except Exception:
    #         logger.debug("Falling back to built-in commit message prompt", exc_info=True)
    #
    #     # Fallback to built-in prompt
    #     default_system_prompt = """You are a git commit message generator for AI chat sessions.
    # You will receive content for ONE dialogue turn (user request, assistant recap, optional recent commit context).
    #
    # Guidelines:
    # - Prefer the assistant recap for factual details.
    # - If the assistant recap includes "Turn status: ...", mirror it exactly in the description's first line.
    # - Keep continuity with any provided recent commit context, but avoid repeating unchanged background.
    #
    # Return JSON with EXACTLY two fields:
    # {
    #   "title": "One-line summary (imperative mood, 25-60 chars preferred, max 80).",
    #   "description": "Status line + 3-7 concise bullets describing what changed in THIS turn."
    # }
    #
    # Rules for title:
    # - Imperative, concrete, no vague fillers like "Update session".
    # - Mention if the turn continues work (e.g., "Continue fixing ...") or is blocked.
    #
    # Rules for description:
    # - First line MUST be "Status: <completed|user_interrupted|rate_limited|compacted|unknown>".
    # - Follow with short "- " bullets explaining WHAT changed and WHY it matters.
    # - Include concrete technical anchors (files, functions) when available.
    # - If continuing prior work, dedicate one bullet to explain the relationship.
    #
    # Respond with JSON only."""
    #     _COMMIT_MESSAGE_PROMPT_CACHE = default_system_prompt
    #     return _COMMIT_MESSAGE_PROMPT_CACHE
    #
    # system_prompt_to_use = _get_commit_message_prompt()
    #
    # user_prompt_parts = ["Summarize this AI chat session:\n"]
    # if user_messages:
    #     user_prompt_parts.append(f"User requests:\n{user_messages[:4000]}\n")
    # if assistant_replies:
    #     user_prompt_parts.append(f"Assistant recap / responses:\n{assistant_replies[:8000]}\n")
    # # Note: code_changes are excluded from LLM input per user preference
    # # if code_changes:
    # #     user_prompt_parts.append(f"Code changes:\n{code_changes[:4000]}\n")
    # # The output format instruction is now in the system prompt (default.md)
    # # user_prompt_parts.append("\nReturn JSON with exactly two fields: title and description. No other text.")
    # user_prompt = "\n".join(user_prompt_parts)
    #
    # model_name, response_text = _invoke_llm(
    #     provider=provider,
    #     system_prompt=system_prompt_to_use,
    #     user_prompt=user_prompt,
    #     debug_callback=debug_callback,
    #     purpose="summary",
    # )
    # if not response_text:
    #     return None, None, None, None, None
    #
    # try:
    #     summary_data = _extract_json_object(response_text)
    #     # New format: single "record" field instead of title+description
    #     record = (summary_data.get("record") or "").strip()
    #
    #     # Fallback to old format for backwards compatibility
    #     if not record:
    #         title = (summary_data.get("title") or "").strip()
    #         description = summary_data.get("description") or ""
    #         if title:
    #             record = title  # Use title as record if present
    #     else:
    #         title = record  # Use record as title
    #         description = ""
    #
    #     if not record or len(record) < 2:
    #         raise json.JSONDecodeError("Record validation failed", response_text, 0)
    # except json.JSONDecodeError as exc:
    #     # Construct detailed error information for debugging
    #     error_type = type(exc).__name__
    #     error_msg = str(exc)
    #
    #     logger.warning("Failed to parse JSON from LLM summary: %s", exc)
    #     logger.debug("Raw summary response: %s", response_text)
    #
    #     # Try to extract partial information from the broken JSON
    #     import re
    #
    #     record_match = re.search(r'"(?:record|title)"\s*:\s*"([^"]{10,})"', response_text)
    #     extracted_content = record_match.group(1)[:80] if record_match else None
    #
    #     # Construct informative error title and description
    #     if "control character" in error_msg.lower():
    #         error_title = "⚠ JSON Parse Error: Invalid control character"
    #         error_detail = f"LLM response contained unescaped control characters. Error at {error_msg.split('at:')[-1].strip() if 'at:' in error_msg else 'unknown position'}"
    #     elif "expecting" in error_msg.lower():
    #         error_title = "⚠ JSON Parse Error: Malformed JSON"
    #         error_detail = f"LLM response had invalid JSON syntax: {error_msg[:100]}"
    #     elif "Record validation failed" in error_msg:
    #         error_title = "⚠ LLM Error: Empty or invalid record"
    #         error_detail = "LLM returned JSON but record/title field was empty or too short"
    #     else:
    #         error_title = f"⚠ JSON Parse Error: {error_type}"
    #         error_detail = f"Failed to parse LLM response: {error_msg[:200]}"
    #
    #     # Add extracted content if available
    #     if extracted_content:
    #         error_detail += f"\n\nPartial content extracted: {extracted_content}..."
    #
    #     # Add response preview for debugging
    #     response_preview = (
    #         response_text[:200].replace("\n", "\\n")
    #         if len(response_text) > 200
    #         else response_text.replace("\n", "\\n")
    #     )
    #     error_detail += f"\n\nResponse preview: {response_preview}"
    #     if len(response_text) > 200:
    #         error_detail += "..."
    #
    #     # Print to stderr for immediate visibility
    #     print(f"   ⚠️  {error_title}", file=sys.stderr)
    #     print(f"   ⚠️  {error_detail.split(chr(10))[0]}", file=sys.stderr)
    #
    #     # Try simple fallback: use first non-JSON line as title
    #     first_line = response_text.split("\n")[0][:150].strip()
    #     if first_line and len(first_line) >= 2 and not first_line.startswith("{"):
    #         print("   ⚠️  Using first line as fallback title", file=sys.stderr)
    #         return first_line, model_name, error_detail, "no", "fine"
    #
    #     # Return structured error information instead of None
    #     logger.error("JSON parse error with no fallback: %s", error_title)
    #     return error_title, model_name, error_detail, "no", "fine"
    #
    # logger.info("LLM summary response: %s", json.dumps(summary_data, ensure_ascii=False))
    # if model_name:
    #     print(f"   ✅ LLM summary successful ({model_name})", file=sys.stderr)
    # else:
    #     print("   ✅ LLM summary successful", file=sys.stderr)
    #
    # if_last_task, satisfaction = _classify_task_metadata(
    #     provider=provider,
    #     user_messages=user_messages,
    #     assistant_replies=assistant_replies,
    #     code_changes=code_changes,
    #     summary_title=title,
    #     summary_description=description,
    #     debug_callback=debug_callback,
    #     system_prompt=metadata_system_prompt,
    #     previous_commit_title=previous_commit_title,
    # )
    #
    # # Return record as title, keep description for backwards compatibility
    # return title or record, model_name, description, if_last_task, satisfaction


def generate_session_filename(user: str, agent: str = "claude") -> str:
    """Generate a unique session filename."""
    timestamp = int(time.time())
    user_short = user.split()[0].lower() if user else "unknown"
    short_id = uuid.uuid4().hex[:8]
    return f"{timestamp}_{user_short}_{agent}_{short_id}.jsonl"


def extract_codex_rollout_hash(filename: str) -> Optional[str]:
    """
    Extract stable hash from Codex rollout filename.

    Primary Codex rollout format:
        rollout-YYYY-MM-DDTHH-MM-SS-<uuid>.jsonl
        Example: rollout-2025-11-16T18-10-42-019a8ddc-b4b3-7942-9a4f-fac74d1580c9.jsonl
                 -> 019a8ddc-b4b3-7942-9a4f-fac74d1580c9

    Legacy format (still supported):
        rollout-<timestamp>-<hash>.jsonl
        Example: rollout-1763315655-abc123def.jsonl -> abc123def

    Args:
        filename: Original Codex rollout filename

    Returns:
        Hash string, or None if parsing fails
    """
    if not filename.startswith("rollout-"):
        return None

    # Normalize filename (strip extension) and remove prefix
    stem = Path(filename).stem
    if stem.startswith("rollout-"):
        stem = stem[len("rollout-") :]

    if not stem:
        return None

    def looks_like_uuid(value: str) -> bool:
        """Return True if value matches canonical UUID format."""
        parts = value.split("-")
        expected_lengths = [8, 4, 4, 4, 12]
        if len(parts) != 5:
            return False
        hex_digits = set("0123456789abcdefABCDEF")
        for part, length in zip(parts, expected_lengths):
            if len(part) != length or not set(part).issubset(hex_digits):
                return False
        return True

    # Newer Codex exports append a full UUID after the human-readable timestamp.
    uuid_candidate_parts = stem.rsplit("-", 5)
    if len(uuid_candidate_parts) == 6:
        candidate_uuid = "-".join(uuid_candidate_parts[1:])
        if looks_like_uuid(candidate_uuid):
            return candidate_uuid.lower()

    # Fallback for legacy rollout names: everything after first '-' is the hash.
    legacy_parts = stem.split("-", 1)
    if len(legacy_parts) == 2 and legacy_parts[1]:
        return legacy_parts[1]

    return None


def get_git_user() -> str:
    """Get git user name."""
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return os.getenv("USER", "unknown")


def get_username(session_relpath: str = "") -> str:
    """
    Get username for commit message.

    Tries to get from git config first, then falls back to extracting
    from session filename.

    Args:
        session_relpath: Relative path to session file (used for fallback)

    Returns:
        Username string
    """
    # Try git config first
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
            text=True,
            check=True,
        )
        username = result.stdout.strip()
        if username:
            return username
    except subprocess.CalledProcessError:
        pass

    # Fallback: extract from session filename
    # Format: username_agent_hash.jsonl
    if session_relpath:
        filename = Path(session_relpath).name
        parts = filename.split("_")
        if len(parts) >= 3:
            # First part is username
            return parts[0]

    # Final fallback
    return os.getenv("USER", "unknown")


def copy_session_to_repo(
    session_file: Path, repo_root: Path, user: str, config: Optional[ReAlignConfig] = None
) -> Tuple[Path, str, bool, int]:
    """
    Copy session file to repository sessions/ directory (in ~/.aline/{project_name}/).
    Optionally redacts sensitive information if configured.
    If the source filename is in UUID format, renames it to include username for better identification.
    Returns (absolute_path, relative_path, was_redacted, content_size).
    """
    repo_root = Path(repo_root).absolute()
    logger.info(f"Copying session to repo: {session_file.name}")
    logger.debug(f"Source: {session_file}, Repo root: {repo_root}, User: {user}")

    from realign import get_realign_dir

    realign_dir = get_realign_dir(repo_root)
    sessions_dir = realign_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    original_filename = session_file.name

    # Check if filename is in UUID format (no underscores, only hyphens and hex chars)
    # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.jsonl
    stem = session_file.stem  # filename without extension
    is_uuid_format = (
        "-" in stem and "_" not in stem and len(stem) == 36  # UUID is 36 chars including hyphens
    )
    # Codex rollout exports always start with rollout-<timestamp>-
    is_codex_rollout = original_filename.startswith("rollout-")

    # Read session content first to detect agent type
    try:
        with open(session_file, "r", encoding="utf-8") as f:
            content = f.read()
        logger.debug(f"Session file read: {len(content)} bytes")
    except Exception as e:
        logger.error(f"Failed to read session file: {e}", exc_info=True)
        print(f"Warning: Could not read session file: {e}", file=sys.stderr)
        # Fallback to simple copy with unknown agent
        if is_uuid_format:
            short_id = stem.split("-")[0]
            user_short = user.split()[0].lower() if user else "unknown"
            new_filename = f"{user_short}_unknown_{short_id}.jsonl"
            dest_path = sessions_dir / new_filename
        elif is_codex_rollout:
            # Extract stable hash from rollout filename
            rollout_hash = extract_codex_rollout_hash(original_filename)
            user_short = user.split()[0].lower() if user else "unknown"
            if rollout_hash:
                new_filename = f"{user_short}_codex_{rollout_hash}.jsonl"
            else:
                # Fallback if hash extraction fails
                new_filename = generate_session_filename(user, "codex")
            dest_path = sessions_dir / new_filename
        else:
            dest_path = sessions_dir / original_filename
        temp_path = dest_path.with_suffix(".tmp")
        shutil.copy2(session_file, temp_path)
        temp_path.rename(dest_path)
        try:
            rel_path = dest_path.relative_to(repo_root)
        except ValueError:
            rel_path = dest_path
        logger.warning(f"Copied session with fallback (no agent detection): {rel_path}")
        # Get file size for the fallback case
        try:
            fallback_size = dest_path.stat().st_size
        except Exception:
            fallback_size = 0
        return dest_path, str(rel_path), False, fallback_size

    # Detect agent type from session content
    agent_type = "unknown"
    try:
        import json

        for line in content.split("\n")[:10]:  # Check first 10 lines
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # Claude Code format
                if data.get("type") in ("user", "assistant"):
                    agent_type = "claude"
                    break
                # Codex format
                elif data.get("type") == "response_item":
                    agent_type = "codex"
                    break
                elif data.get("type") == "session_meta":
                    payload = data.get("payload", {})
                    if "codex" in payload.get("originator", "").lower():
                        agent_type = "codex"
                        break
            except json.JSONDecodeError:
                continue
        logger.debug(f"Detected agent type: {agent_type}")
    except Exception as e:
        logger.warning(f"Agent type detection failed: {e}")

    # If it's UUID format, rename to include username and agent type
    if is_uuid_format:
        # Extract short ID from UUID (first 8 chars)
        short_id = stem.split("-")[0]
        user_short = user.split()[0].lower() if user else "unknown"
        # Format: username_agent_shortid.jsonl (no timestamp for consistency)
        new_filename = f"{user_short}_{agent_type}_{short_id}.jsonl"
        dest_path = sessions_dir / new_filename
    elif is_codex_rollout:
        # Extract stable hash from rollout filename
        codex_agent = agent_type if agent_type != "unknown" else "codex"
        rollout_hash = extract_codex_rollout_hash(original_filename)
        user_short = user.split()[0].lower() if user else "unknown"
        if rollout_hash:
            # Format: username_codex_hash.jsonl (stable naming)
            new_filename = f"{user_short}_{codex_agent}_{rollout_hash}.jsonl"
        else:
            # Fallback if hash extraction fails
            new_filename = generate_session_filename(user, codex_agent)
        dest_path = sessions_dir / new_filename
    else:
        # Keep original filename (could be timestamp_user_agent_id format or other)
        dest_path = sessions_dir / original_filename

    # Check if redaction is enabled
    was_redacted = False
    if config and config.redact_on_match and REDACTOR_AVAILABLE:
        logger.info("Redaction enabled, checking for secrets")
        # Backup original before redaction
        backup_path = save_original_session(dest_path, repo_root)
        if backup_path:
            logger.info(f"Original session backed up to: {backup_path}")
            try:
                backup_rel = backup_path.relative_to(repo_root)
            except ValueError:
                backup_rel = backup_path
            print(f"   💾 Original session backed up to: {backup_rel}", file=sys.stderr)

        # Perform redaction
        redacted_content, has_secrets, secrets = check_and_redact_session(
            content, redact_mode="auto"
        )

        if has_secrets:
            logger.warning(f"Secrets detected and redacted: {len(secrets)} secret(s)")
            content = redacted_content
            was_redacted = True
        else:
            logger.info("No secrets detected")
    elif config and config.redact_on_match and not REDACTOR_AVAILABLE:
        logger.warning("Redaction enabled but detect-secrets not installed")
        print("⚠️  Redaction enabled but detect-secrets not installed", file=sys.stderr)
        print("   Install with: pip install 'realign-git[redact]'", file=sys.stderr)

    # Write content to destination (redacted or original)
    temp_path = dest_path.with_suffix(".tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)
        temp_path.rename(dest_path)
        try:
            log_path = dest_path.absolute().relative_to(repo_root)
        except ValueError:
            log_path = dest_path.absolute()
        logger.info(f"Session written to: {log_path}")
    except Exception as e:
        logger.error(f"Failed to write session file: {e}", exc_info=True)
        print(f"Warning: Could not write session file: {e}", file=sys.stderr)
        # Fallback to simple copy
        if temp_path.exists():
            temp_path.unlink()
        shutil.copy2(session_file, dest_path)
        logger.warning("Fallback to simple copy")

    # Return both absolute and relative paths, plus redaction status and content size
    try:
        rel_path = dest_path.absolute().relative_to(repo_root)
    except ValueError:
        rel_path = dest_path.absolute()
    content_size = len(content)
    return dest_path, str(rel_path), was_redacted, content_size


def save_session_metadata(repo_root: Path, session_relpath: str, content_size: int):
    """
    Save metadata about a processed session to avoid reprocessing.

    Args:
        repo_root: Path to repository root
        session_relpath: Relative path to session file
        content_size: Size of session content when processed
    """
    from realign import get_realign_dir

    realign_dir = get_realign_dir(repo_root)
    metadata_dir = realign_dir / ".metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Use session filename as metadata key
    session_name = Path(session_relpath).name
    metadata_file = metadata_dir / f"{session_name}.meta"

    metadata = {
        "processed_at": time.time(),
        "content_size": content_size,
        "session_relpath": session_relpath,
    }

    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        logger.debug(f"Saved metadata for {session_relpath}: {content_size} bytes")
    except Exception as e:
        logger.warning(f"Failed to save metadata for {session_relpath}: {e}")


def get_session_metadata(repo_root: Path, session_relpath: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata about a previously processed session.

    Args:
        repo_root: Path to repository root
        session_relpath: Relative path to session file

    Returns:
        Metadata dictionary or None if not found
    """
    from realign import get_realign_dir

    realign_dir = get_realign_dir(repo_root)
    metadata_dir = realign_dir / ".metadata"
    session_name = Path(session_relpath).name
    metadata_file = metadata_dir / f"{session_name}.meta"

    if not metadata_file.exists():
        return None

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.debug(f"Loaded metadata for {session_relpath}: {metadata.get('content_size')} bytes")
        return metadata
    except Exception as e:
        logger.warning(f"Failed to load metadata for {session_relpath}: {e}")
        return None


def generate_summary_with_llm_from_turn_context(
    user_message: str,
    assistant_summary: str,
    turn_status: str,
    recent_commit_context: str = "",
    provider: str = "auto",
    system_prompt: Optional[str] = None,
    debug_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    metadata_system_prompt: Optional[str] = None,
    previous_commit_title: Optional[str] = None,
    full_turn_content: Optional[str] = None,
    previous_records: Optional[list] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str], str, str]:
    """
    Generate summary using trigger-derived turn context (preferred for watchers).

    This wraps the existing JSONL-based summarizer. If full_turn_content is provided,
    it uses that directly (including all messages, thinking, etc., but excluding tool use).
    Otherwise, it constructs a minimal JSONL payload from user_message and assistant_summary.

    Args:
        user_message: User's message for this turn
        assistant_summary: Assistant's summary/response
        turn_status: Status of the turn (completed, interrupted, etc.)
        recent_commit_context: Recent commit history context
        provider: LLM provider to use
        system_prompt: Optional custom system prompt
        debug_callback: Optional debug callback
        metadata_system_prompt: Optional metadata system prompt
        previous_commit_title: Previous commit title for context
        full_turn_content: Optional full JSONL content for the turn (all messages)

    Returns:
        Tuple of (title, model_name, description, if_last_task, satisfaction)
    """
    # If full turn content is provided, extract ALL assistant messages
    # (thinking + text responses) and combine with the final conclusion
    if full_turn_content:
        full_turn_content = full_turn_content.strip()
        if full_turn_content:
            # Extract ALL assistant messages (thinking + responses) from full_turn_content
            _, all_assistant_msgs, _ = filter_session_content(full_turn_content)

            _emit_llm_debug(
                debug_callback,
                {
                    "event": "turn_context",
                    "mode": "full_content_all_messages",
                    "turn_status": turn_status or "unknown",
                    "recent_commit_context": recent_commit_context or "",
                    "all_messages_length": len(all_assistant_msgs),
                    "assistant_summary_length": len(assistant_summary or ""),
                },
            )

            # Build payload with ALL assistant messages + conclusion + previous records
            user_message = (user_message or "").strip()
            assistant_summary = (assistant_summary or "").strip()
            turn_status = (turn_status or "").strip() or "unknown"

            payload_lines = []
            if user_message:
                payload_lines.append(
                    json.dumps(
                        {"type": "user", "message": {"content": user_message}}, ensure_ascii=False
                    )
                )

            # Build a single assistant message with all components
            assistant_parts = []

            # Include ALL thinking and responses from the turn
            if all_assistant_msgs:
                assistant_parts.append(f"Assistant thinking and responses:\n{all_assistant_msgs}")

            # Add the final conclusion from trigger (which is the authoritative summary)
            if assistant_summary:
                assistant_parts.append(
                    f"Turn status: {turn_status}\n\nFinal conclusion:\n{assistant_summary}"
                )

            # Add previous records context (new format) or fallback to recent_commit_context
            if previous_records is not None:
                if len(previous_records) > 0:
                    records_text = "\n".join(f"- {rec}" for rec in previous_records[-5:])
                    context_text = f"Last {len(previous_records[-5:])} records:\n{records_text}"
                else:
                    context_text = "No previous record"
                assistant_parts.append(context_text)
            elif recent_commit_context:
                assistant_parts.append(f"Recent commit context:\n{recent_commit_context}")

            # Combine all parts into a single assistant message
            if assistant_parts:
                combined_text = "\n\n".join(assistant_parts)
                payload_lines.append(
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": combined_text,
                                    }
                                ]
                            },
                        },
                        ensure_ascii=False,
                    )
                )

            synthetic_jsonl = "\n".join(payload_lines)

            return generate_summary_with_llm(
                synthetic_jsonl,
                max_chars=500,
                provider=provider,
                system_prompt=system_prompt,
                debug_callback=debug_callback,
                metadata_system_prompt=metadata_system_prompt,
                previous_commit_title=previous_commit_title,
            )

    # Fallback to legacy mode: construct minimal JSONL from user_message and assistant_summary
    user_message = (user_message or "").strip()
    assistant_summary = (assistant_summary or "").strip()
    turn_status = (turn_status or "").strip() or "unknown"

    if not user_message and not assistant_summary:
        return None, None, None, "no", "fine"

    recent_commit_context = (recent_commit_context or "").strip()

    payload_lines = [
        json.dumps({"type": "user", "message": {"content": user_message}}, ensure_ascii=False),
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Turn status: {turn_status}\n\nAssistant recap / final response:\n{assistant_summary}",
                        }
                    ]
                },
            },
            ensure_ascii=False,
        ),
    ]

    # Add previous records context (new format) or fallback to recent_commit_context
    # Note: previous_records check uses "is not None" to handle empty list case
    if previous_records is not None:
        if len(previous_records) > 0:
            records_text = "\n".join(f"- {rec}" for rec in previous_records[-5:])
            context_text = f"Last {len(previous_records[-5:])} records:\n{records_text}"
        else:
            context_text = "No previous record"

        payload_lines.append(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": context_text,
                            }
                        ]
                    },
                },
                ensure_ascii=False,
            )
        )
    elif recent_commit_context:
        payload_lines.append(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Recent commit context:\n{recent_commit_context}",
                            }
                        ]
                    },
                },
                ensure_ascii=False,
            )
        )

    synthetic_jsonl = "\n".join(payload_lines)
    _emit_llm_debug(
        debug_callback,
        {
            "event": "turn_context",
            "mode": "legacy",
            "user_message": user_message,
            "assistant_summary": assistant_summary,
            "turn_status": turn_status,
            "recent_commit_context": recent_commit_context,
            "synthetic_jsonl": synthetic_jsonl,
        },
    )

    return generate_summary_with_llm(
        synthetic_jsonl,
        max_chars=500,
        provider=provider,
        system_prompt=system_prompt,
        debug_callback=debug_callback,
        metadata_system_prompt=metadata_system_prompt,
        previous_commit_title=previous_commit_title,
    )
