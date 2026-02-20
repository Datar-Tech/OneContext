#!/usr/bin/env python3
"""
Export shares command - Export selected sessions' chat history to JSON files.

This allows users to select specific sessions and extract their chat history
into standalone JSON files for sharing.
"""

import json
import os
import re
import subprocess
import sys
import secrets
import hashlib
import base64
import threading
import shutil
from urllib.parse import urlparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple, Set, Callable

from ..logging_config import setup_logger
from ..db.base import SessionRecord, TurnRecord
from ..llm_client import extract_json, call_llm_cloud
from ..auth import get_auth_headers, is_logged_in

logger = setup_logger("realign.commands.export_shares", "export_shares.log")

# Prompt cache for share UI metadata
_SHARE_UI_METADATA_PROMPT_CACHE: Optional[str] = None


# ============================================================================
# Utility functions (previously from review.py and hide.py)
# ============================================================================


def parse_indices(indices_str: str) -> List[int]:
    """
    Parse user input of indices (sessions or turns).

    Supports:
    - Single: "3" -> [3]
    - Multiple: "1,3,5" -> [1, 3, 5]
    - Range: "2-4" -> [2, 3, 4]
    - Combined: "1,3,5-7" -> [1, 3, 5, 6, 7]

    Args:
        indices_str: User input string

    Returns:
        Sorted list of unique indices

    Raises:
        ValueError: If input format is invalid
    """
    if not indices_str or not indices_str.strip():
        raise ValueError("Empty input")

    result: Set[int] = set()

    for part in indices_str.split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            # Range: "2-4"
            range_parts = part.split("-", 1)
            if len(range_parts) != 2:
                raise ValueError(f"Invalid range format: {part}")

            try:
                start = int(range_parts[0].strip())
                end = int(range_parts[1].strip())
            except ValueError:
                raise ValueError(f"Invalid range format: {part}")

            if start > end:
                raise ValueError(f"Invalid range (start > end): {part}")

            result.update(range(start, end + 1))
        else:
            # Single number
            try:
                num = int(part)
            except ValueError:
                raise ValueError(f"Invalid number: {part}")

            if num < 1:
                raise ValueError(f"Index must be >= 1: {num}")

            result.add(num)

    return sorted(result)


def _is_uuid_like(selector: str) -> bool:
    """
    Check if a string looks like a UUID or UUID prefix.

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


def _find_events_by_uuid(selector: str, events: List) -> List[int]:
    """
    Find events by UUID or UUID prefix.

    Args:
        selector: UUID string (full or prefix, at least 4 characters)
            - Single UUID: "abc123de"
            - Multiple UUIDs: "abc123de,def456gh,xyz789"
        events: List of event objects with 'id' attribute

    Returns:
        List of 1-based indices matching the UUID(s), or empty list if not found/invalid
    """
    indices = []

    # Support comma-separated multiple UUIDs
    parts = [p.strip().lower() for p in selector.split(",")]

    for part in parts:
        if not _is_uuid_like(part):
            # If any part is not UUID-like, return empty (fall back to numeric parsing)
            if len(parts) == 1:
                return []
            # For multi-part selectors, skip invalid parts but continue
            continue

        for i, event in enumerate(events, 1):
            # Support both 'id' and 'event_id' attributes
            eid = getattr(event, "event_id", None) or getattr(event, "id", None) or ""
            eid = eid.lower()
            if eid.startswith(part) or eid == part:
                if i not in indices:
                    indices.append(i)

    return sorted(indices)


@dataclass
class ExportableSession:
    """Represents a session available for export."""

    index: int  # User-visible index (1-based)
    session_id: str  # Session ID
    session_type: str  # Session type (claude, codex, etc.)
    session_title: Optional[str]  # Session title (from LLM summary)
    session_summary: Optional[str]  # Session summary (from LLM)
    workspace_path: Optional[str]  # Project workspace path
    started_at: Optional[datetime]  # Session start time
    last_activity_at: Optional[datetime]  # Last activity time
    turn_count: int  # Number of turns in session
    turns: List[TurnRecord]  # List of turns
    # V18: user identity
    created_by: Optional[str] = None  # Creator UID
    shared_by: Optional[str] = None  # Sharer UID


def get_sessions_for_export(
    workspace_path: Optional[str] = None, limit: int = 100
) -> List[ExportableSession]:
    """
    Get sessions available for export from database.

    Args:
        workspace_path: Optional filter by workspace path
        limit: Maximum number of sessions to return

    Returns:
        List of ExportableSession objects, ordered by last activity (most recent first)
    """
    from ..db import get_database

    db = get_database()
    sessions = db.list_sessions(limit=limit, workspace_path=workspace_path)

    result = []
    for idx, session in enumerate(sessions, 1):
        turns = db.get_turns_for_session(session.id)
        result.append(
            ExportableSession(
                index=idx,
                session_id=session.id,
                session_type=session.session_type or "unknown",
                session_title=session.session_title,
                session_summary=session.session_summary,
                workspace_path=session.workspace_path,
                started_at=session.started_at,
                last_activity_at=session.last_activity_at,
                turn_count=len(turns),
                turns=turns,
                created_by=session.created_by,
                shared_by=session.shared_by,
            )
        )

    return result


@dataclass
class ExportableEvent:
    """Represents an event available for export."""

    index: int  # User-visible index (1-based)
    event_id: str  # Event ID
    title: str  # Event title
    description: Optional[str]  # Event description
    event_type: str  # 'task', 'temporal', etc.
    status: str  # 'active', 'frozen', 'archived'
    updated_at: Optional[datetime]  # Last updated time
    sessions: List[SessionRecord]  # Sessions linked to this event (workspace-filtered)


def get_events_for_export(
    workspace_path: Optional[str] = None, limit: int = 100
) -> List[ExportableEvent]:
    """
    Get events available for export from the database.

    Events are filtered by workspace_path via their linked sessions.
    """
    from ..db import get_database

    db = get_database()
    events = db.list_events(limit=limit, offset=0)

    result: List[ExportableEvent] = []
    next_index = 1
    for event in events:
        sessions = db.get_sessions_for_event(event.id)
        if workspace_path:
            sessions = [s for s in sessions if s.workspace_path == workspace_path]
        if not sessions:
            continue

        result.append(
            ExportableEvent(
                index=next_index,
                event_id=event.id,
                title=event.title,
                description=event.description,
                event_type=event.event_type,
                status=event.status,
                updated_at=event.updated_at,
                sessions=sessions,
            )
        )
        next_index += 1

    return result


def build_exportable_sessions_from_records(
    session_records: List[SessionRecord],
) -> List[ExportableSession]:
    """Convert SessionRecord(s) into ExportableSession(s) with turns preloaded."""
    from ..db import get_database

    db = get_database()
    exportable: List[ExportableSession] = []
    for idx, session in enumerate(session_records, 1):
        turns = db.get_turns_for_session(session.id)
        exportable.append(
            ExportableSession(
                index=idx,
                session_id=session.id,
                session_type=session.session_type or "unknown",
                session_title=session.session_title,
                session_summary=session.session_summary,
                workspace_path=session.workspace_path,
                started_at=session.started_at,
                last_activity_at=session.last_activity_at,
                turn_count=len(turns),
                turns=turns,
                created_by=session.created_by,
                shared_by=session.shared_by,
            )
        )
    return exportable


def get_messages_from_session(
    session: ExportableSession,
) -> List[Tuple[datetime, dict]]:
    """
    Extract messages from a session's turns.

    Args:
        session: ExportableSession object

    Returns:
        List of (timestamp, message_dict) tuples
    """
    from ..db import get_database

    db = get_database()
    messages = []

    for turn in session.turns:
        content = db.get_turn_content(turn.id)
        if content:
            # Parse JSONL content
            for line in content.strip().split("\n"):
                if line.strip():
                    try:
                        msg = json.loads(line)
                        ts = turn.timestamp or datetime.now()
                        messages.append((ts, msg))
                    except json.JSONDecodeError:
                        continue

    return messages


@dataclass(frozen=True)
class ExportCompactionConfig:
    """
    Controls how much data is included in share exports.

    The default export format mirrors the local JSONL records, which can be very large
    (tool commands, tool results, request metadata, token usage, etc.). Compaction
    strips most metadata and optionally truncates tool I/O to keep uploads small.
    """

    enabled: bool = False
    max_tool_result_chars: int = 8_000
    max_tool_command_chars: int = 2_000
    omit_tool_messages: bool = True
    omit_thinking_blocks: bool = True
    strip_thinking_signatures: bool = True
    max_message_text_chars: int = 12_000
    max_turn_user_message_chars: int = 2_000
    max_turn_assistant_summary_chars: int = 2_000
    collapse_high_entropy_strings: bool = True
    min_entropy_span_chars: int = 200


def _truncate_string(value: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}…[truncated {len(value) - max_chars} chars]"


_HEX_RUN_RE = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{200,}(?![0-9a-f])")
_BASE64ISH_RUN_RE = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/=_-]{200,}(?![A-Za-z0-9+/=_-])")


def _collapse_high_entropy_runs(text: str, config: ExportCompactionConfig) -> str:
    """
    Replace very long base64/hex-like runs that are almost always low-signal (signatures, hashes,
    embedded binaries, etc.).
    """
    if not config.collapse_high_entropy_strings:
        return text

    min_len = max(50, int(config.min_entropy_span_chars))

    def _replace(kind: str, s: str) -> str:
        digest = hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]
        return f"[omitted {kind} span len={len(s)} sha256={digest}]"

    def _sub(pattern: re.Pattern[str], kind: str, value: str) -> str:
        def repl(m: re.Match[str]) -> str:
            s = m.group(0)
            if len(s) < min_len:
                return s
            return _replace(kind, s)

        return pattern.sub(repl, value)

    text = _sub(_HEX_RUN_RE, "hex", text)
    text = _sub(_BASE64ISH_RUN_RE, "base64", text)
    return text


def _compact_text(text: str, config: ExportCompactionConfig) -> str:
    text = _collapse_high_entropy_runs(text, config)
    return _truncate_string(text, config.max_message_text_chars)


def _compact_message_content(content: Any, config: ExportCompactionConfig) -> Any:
    if not config.enabled:
        return content

    if isinstance(content, list):
        compacted: List[Any] = []
        for block in content:
            if not isinstance(block, dict):
                compacted.append(block)
                continue

            block_type = block.get("type")
            if config.omit_tool_messages and block_type in (
                "tool_use",
                "tool_result",
                "tool_call",
                "tool_call_output",
                "function_call",
                "function_call_output",
            ):
                continue

            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str):
                    block = dict(block)
                    block["text"] = _compact_text(text, config)
                compacted.append(block)
                continue

            if block_type == "thinking":
                if config.omit_thinking_blocks:
                    continue
                if config.strip_thinking_signatures and "signature" in block:
                    block = dict(block)
                    block.pop("signature", None)
                compacted.append(block)
                continue

            if block_type == "tool_use":
                # Keep tool name + description; truncate long shell commands/scripts.
                input_obj = block.get("input")
                if isinstance(input_obj, dict):
                    cmd = input_obj.get("command")
                    if isinstance(cmd, str):
                        input_obj = dict(input_obj)
                        input_obj["command"] = _truncate_string(cmd, config.max_tool_command_chars)
                    block = dict(block)
                    block["input"] = input_obj
                compacted.append(block)
                continue

            if block_type == "tool_result":
                result_content = block.get("content")
                if isinstance(result_content, str):
                    block = dict(block)
                    block["content"] = _truncate_string(
                        result_content, config.max_tool_result_chars
                    )
                compacted.append(block)
                continue

            compacted.append(block)
        return compacted

    if isinstance(content, str):
        return _compact_text(content, config)

    return content


def _compact_jsonl_record(
    record: dict,
    config: ExportCompactionConfig,
    *,
    fallback_session_id: Optional[str] = None,
) -> Optional[dict]:
    """
    Reduce a raw JSONL record to a minimal envelope used by the server-side tools.

    Keeps:
      - sessionId (for stats + grouping)
      - timestamp (for ordering)
      - message.role + message.content (for conversation display/search)
    Drops everything else (cwd, gitBranch, token usage, request ids, duplicated toolUseResult, etc.).
    """
    if not config.enabled:
        return record

    msg = record.get("message")
    if not isinstance(msg, dict):
        # Codex format: { type: "response_item", payload: {...} }
        rec_type = record.get("type")
        if rec_type == "response_item":
            payload = record.get("payload")
            if not isinstance(payload, dict):
                return None

            payload_type = payload.get("type")

            # Drop tool/function call traffic in compact mode (high volume, low value for share Q&A)
            if payload_type in (
                "function_call",
                "function_call_output",
                "tool_call",
                "tool_call_output",
            ):
                return None

            # Keep user/assistant visible messages
            if payload_type == "message":
                role = payload.get("role")
                content_items = payload.get("content", [])
                texts: List[str] = []
                if isinstance(content_items, list):
                    for item in content_items:
                        if not isinstance(item, dict):
                            continue
                        t = item.get("type")
                        if t in ("input_text", "output_text") and isinstance(item.get("text"), str):
                            texts.append(item["text"])
                compact_message: Dict[str, Any] = {}
                if isinstance(role, str):
                    compact_message["role"] = role
                combined = "\n".join(texts).strip()
                if config.omit_thinking_blocks and combined.startswith("[Thinking]"):
                    return None
                compact_message["content"] = _compact_text(combined, config)
                return {
                    "sessionId": record.get("sessionId") or fallback_session_id,
                    "timestamp": record.get("timestamp"),
                    "type": rec_type,
                    "message": compact_message,
                }

        # Codex visible thinking: { type: "event_msg", payload: {type:"agent_reasoning", text:"..."} }
        if record.get("type") == "event_msg":
            payload = record.get("payload")
            if isinstance(payload, dict) and payload.get("type") == "agent_reasoning":
                if config.omit_thinking_blocks:
                    return None
                text = payload.get("text")
                if isinstance(text, str) and text.strip():
                    return {
                        "sessionId": record.get("sessionId") or fallback_session_id,
                        "timestamp": record.get("timestamp"),
                        "type": "event_msg",
                        "message": {
                            "role": "assistant",
                            "content": _compact_text(f"[Thinking] {text.strip()}", config),
                        },
                    }

        # Drop other non-message records (snapshots, queue operations, etc.)
        return None

    role = msg.get("role")
    content = msg.get("content")

    compact_message: Dict[str, Any] = {}
    if isinstance(role, str):
        compact_message["role"] = role
    compacted_content = _compact_message_content(content, config)
    if (
        isinstance(compacted_content, str)
        and config.omit_thinking_blocks
        and compacted_content.startswith("[Thinking]")
    ):
        return None
    compact_message["content"] = compacted_content

    # Retain a stable identifier when present (helps debugging + "get_message_by_id").
    if isinstance(msg.get("id"), str):
        compact_message["id"] = msg["id"]

    # Minimal envelope expected by VirtualFileSystem stats: {sessionId, message:{role,...}}
    out: Dict[str, Any] = {
        "sessionId": record.get("sessionId") or fallback_session_id,
        "timestamp": record.get("timestamp"),
        "type": record.get("type"),
        "message": compact_message,
    }

    # Keep uuid linkage when present; useful for context chaining without much size cost.
    if isinstance(record.get("uuid"), str):
        out["uuid"] = record["uuid"]
    if isinstance(record.get("parentUuid"), str):
        out["parentUuid"] = record["parentUuid"]

    return out


def _build_export_size_report(
    selected_sessions: List["ExportableSession"],
    db,
    *,
    top_n: int = 10,
) -> dict:
    """
    Analyze raw JSONL turn content sizes to explain why share exports become large.

    Returns a dict safe to print (no raw content included).
    """
    top_lines: List[dict] = []
    total_lines = 0
    total_bytes = 0

    tool_use_index: Dict[str, dict] = {}

    def _push_candidate(candidate: dict) -> None:
        nonlocal top_lines
        top_lines.append(candidate)
        top_lines.sort(key=lambda x: x["size_bytes"], reverse=True)
        if len(top_lines) > top_n:
            top_lines = top_lines[:top_n]

    for session in selected_sessions:
        for turn in session.turns:
            turn_content = db.get_turn_content(turn.id)
            if not turn_content:
                continue

            for line in turn_content.split("\n"):
                if not line.strip():
                    continue

                size_bytes = len(line.encode("utf-8"))
                total_lines += 1
                total_bytes += size_bytes

                # Keep parse-free info by default; attempt parse to classify record type.
                kind = "unknown"
                role = None
                block_types: List[str] = []
                tool_use_ids: List[str] = []

                try:
                    rec = json.loads(line)
                    msg = rec.get("message")
                    if isinstance(msg, dict):
                        role = msg.get("role")
                        kind = rec.get("type") or msg.get("type") or "message"
                        content = msg.get("content")
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and isinstance(block.get("type"), str):
                                    block_types.append(block["type"])
                                    if block.get("type") == "tool_use":
                                        tool_use_id = block.get("id")
                                        if isinstance(tool_use_id, str) and tool_use_id:
                                            input_obj = block.get("input") or {}
                                            input_keys: List[str] = []
                                            input_string_lengths: Dict[str, int] = {}
                                            input_path_hint = None
                                            command = None
                                            if isinstance(input_obj, dict) and isinstance(
                                                input_obj.get("command"), str
                                            ):
                                                command = input_obj["command"]
                                            if isinstance(input_obj, dict):
                                                input_keys = sorted(
                                                    [
                                                        k
                                                        for k in input_obj.keys()
                                                        if isinstance(k, str)
                                                    ]
                                                )
                                                for k, v in input_obj.items():
                                                    if isinstance(k, str) and isinstance(v, str):
                                                        input_string_lengths[k] = len(v)
                                                for path_key in ("path", "file_path", "filename"):
                                                    pv = input_obj.get(path_key)
                                                    if isinstance(pv, str) and pv:
                                                        input_path_hint = _truncate_string(pv, 120)
                                                        break
                                            tool_use_index[tool_use_id] = {
                                                "tool_name": block.get("name"),
                                                "description": (
                                                    input_obj.get("description")
                                                    if isinstance(input_obj, dict)
                                                    else None
                                                ),
                                                "input_keys": input_keys,
                                                "input_string_lengths": input_string_lengths,
                                                "path_hint": input_path_hint,
                                                "command_snippet": (
                                                    _truncate_string(command, 200)
                                                    if isinstance(command, str)
                                                    else None
                                                ),
                                                "command_chars": (
                                                    len(command)
                                                    if isinstance(command, str)
                                                    else None
                                                ),
                                            }
                                    if block.get("type") == "tool_result":
                                        tool_use_id = block.get("tool_use_id")
                                        if isinstance(tool_use_id, str) and tool_use_id:
                                            tool_use_ids.append(tool_use_id)
                except Exception:
                    pass

                if len(top_lines) < top_n or size_bytes > top_lines[-1]["size_bytes"]:
                    tools: List[dict] = []
                    for tool_use_id in tool_use_ids[:3]:
                        info = tool_use_index.get(tool_use_id)
                        if info:
                            tools.append({"tool_use_id": tool_use_id, **info})
                        else:
                            tools.append({"tool_use_id": tool_use_id})

                    _push_candidate(
                        {
                            "size_bytes": size_bytes,
                            "session_id": session.session_id,
                            "turn_id": turn.id,
                            "turn_number": turn.turn_number,
                            "timestamp": (turn.timestamp.isoformat() if turn.timestamp else None),
                            "record_type": kind,
                            "role": role,
                            "content_block_types": sorted(set(block_types))[:6],
                            "tools": tools,
                        }
                    )

    return {
        "total_lines": total_lines,
        "total_bytes": total_bytes,
        "top_lines": top_lines,
    }


def _top_string_fields(data: Any, top_n: int = 10) -> List[dict]:
    results: List[Tuple[int, str]] = []

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str):
                    walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
        elif isinstance(obj, str):
            results.append((len(obj), path))

    walk(data, "$")
    results.sort(reverse=True, key=lambda x: x[0])
    return [{"chars": n, "path": p} for n, p in results[:top_n]]


def build_enhanced_conversation_data(
    selected_event: ExportableEvent,
    selected_sessions: List[ExportableSession],
    username: str,
    db,
    *,
    compaction: Optional[ExportCompactionConfig] = None,
) -> dict:
    """
    Build conversation data with full Event/Session/Turn structure (v2.0 format).

    This enhanced format preserves the complete hierarchy of Event → Sessions → Turns
    for proper import/reconstruction on the receiving end.

    Args:
        selected_event: The event being exported
        selected_sessions: List of sessions with preloaded turns
        username: Username for the export
        db: Database instance for fetching turn content

    Returns:
        Dictionary with v2.0 structure including event metadata, session metadata,
        and structured turns with messages
    """
    logger.info(f"Building enhanced conversation data (v2.0) for event: {selected_event.title}")

    # Build event data
    event_data = {
        "event_id": selected_event.event_id,
        "title": selected_event.title,
        "description": selected_event.description or "",
        "event_type": selected_event.event_type,
        "status": selected_event.status,
        "created_at": None,  # Will be populated if available from EventRecord
        "updated_at": selected_event.updated_at.isoformat() if selected_event.updated_at else None,
        "metadata": {},
    }

    # Get full EventRecord to access created_at and other fields (including V9 creator info)
    full_event = db.get_event_by_id(selected_event.event_id)
    if full_event:
        event_data["created_at"] = (
            full_event.created_at.isoformat() if full_event.created_at else None
        )
        event_data["metadata"] = full_event.metadata or {}
        event_data["created_by"] = full_event.created_by
        event_data["shared_by"] = full_event.shared_by

    # Build sessions data with turn structure
    sessions_data = []
    for session in selected_sessions:
        turns_data = []

        for turn in session.turns:
            # Get turn content (JSONL)
            turn_content = db.get_turn_content(turn.id)
            messages = []

            if turn_content:
                # Parse JSONL into list of message objects
                for line in turn_content.strip().split("\n"):
                    if line.strip():
                        try:
                            msg = json.loads(line)
                            if compaction and compaction.enabled:
                                msg = _compact_jsonl_record(
                                    msg,
                                    compaction,
                                    fallback_session_id=session.session_id,
                                )
                                if msg is None:
                                    continue
                            messages.append(msg)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSONL line in turn {turn.id}")
                            continue

            # Build turn data (V9: includes creator fields)
            turn_data = {
                "turn_id": turn.id,
                "turn_number": turn.turn_number,
                "content_hash": turn.content_hash,
                "timestamp": (
                    turn.timestamp.isoformat() if turn.timestamp else datetime.now().isoformat()
                ),
                "llm_title": turn.llm_title or "",
                "llm_description": turn.llm_description,
                "user_message": (
                    _truncate_string(
                        _collapse_high_entropy_runs(turn.user_message, compaction),
                        compaction.max_turn_user_message_chars,
                    )
                    if (compaction and compaction.enabled and isinstance(turn.user_message, str))
                    else turn.user_message
                ),
                "assistant_summary": (
                    _truncate_string(
                        _collapse_high_entropy_runs(turn.assistant_summary, compaction),
                        compaction.max_turn_assistant_summary_chars,
                    )
                    if (
                        compaction
                        and compaction.enabled
                        and isinstance(turn.assistant_summary, str)
                    )
                    else turn.assistant_summary
                ),
                "model_name": turn.model_name,
                "git_commit_hash": turn.git_commit_hash,
                "messages": messages,  # Structured messages for this turn
            }
            turns_data.append(turn_data)

        # Build session data (V18: created_by/shared_by)
        session_data = {
            "session_id": session.session_id,
            "session_type": session.session_type,
            "workspace_path": session.workspace_path,
            "session_title": session.session_title,
            "session_summary": session.session_summary,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "last_activity_at": (
                session.last_activity_at.isoformat() if session.last_activity_at else None
            ),
            "created_by": session.created_by,
            "shared_by": session.shared_by,
            "turns": turns_data,
        }
        sessions_data.append(session_data)

    logger.info(
        f"Built v2.0 data: {len(sessions_data)} sessions, {sum(len(s['turns']) for s in sessions_data)} turns"
    )

    # Return v2.0 structure
    return {
        "version": "2.0",
        "username": username,
        "time": datetime.now().isoformat(),
        "event": event_data,
        "sessions": sessions_data,
        "ui_metadata": {},  # Will be populated later with LLM-generated content
    }


def build_agent_share_conversation_data_metadata_only(
    *,
    agent_id: str,
    agent_name: str,
    agent_title: str = "",
    agent_description: str,
    selected_sessions: List[ExportableSession],
    username: str,
) -> dict:
    """
    Build agent share payload in v2.0 format, but metadata-only (no turn content/messages).

    Field policy (Phase 0 / Team mode):
    - Do not upload: turn messages/content, user_message, model_name, git_commit_hash
    - Do not upload: session workspace_path, shared_by
    - Do not upload: agent visibility (local-only)
    - Upload: created_by (session, read-only), and turn llm_description (usually empty)
    """
    now = datetime.now(timezone.utc).isoformat()

    event_data = {
        "event_id": f"agent-{agent_id}",
        "title": agent_name or "Agent Sessions",
        "context_title": agent_title or "",
        "description": agent_description or "",
        "event_type": "agent",
        "status": "active",
        "created_at": now,
        "updated_at": now,
        "metadata": {},
    }

    sessions_data: List[dict] = []
    for session in selected_sessions:
        turns_data: List[dict] = []
        for turn in session.turns:
            turns_data.append(
                {
                    "turn_id": turn.id,
                    "turn_number": turn.turn_number,
                    "content_hash": turn.content_hash,
                    "timestamp": (
                        turn.timestamp.isoformat()
                        if turn.timestamp
                        else datetime.now(timezone.utc).isoformat()
                    ),
                    "llm_title": turn.llm_title or "",
                    "llm_description": turn.llm_description,
                    "assistant_summary": turn.assistant_summary,
                    "messages": [],  # placeholder (content not uploaded in Phase 0)
                    "content_available": False,
                }
            )

        sessions_data.append(
            {
                "session_id": session.session_id,
                "session_type": session.session_type,
                "session_title": session.session_title,
                "session_summary": session.session_summary,
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "last_activity_at": (
                    session.last_activity_at.isoformat() if session.last_activity_at else None
                ),
                "created_by": session.created_by,
                "turns": turns_data,
            }
        )

    return {
        "version": "2.0",
        "username": username,
        "time": now,
        "agent": {
            "agent_id": agent_id,
            "agent_name": agent_name or "Agent Sessions",
            "context_title": agent_title or "",
            "description": agent_description or "",
        },
        "event": event_data,
        "sessions": sessions_data,
        "ui_metadata": {},
    }


# Legacy compatibility - alias for old function name
def parse_commit_indices(indices_str: str) -> List[int]:
    """Legacy alias for parse_indices."""
    return parse_indices(indices_str)


@dataclass
class UnpushedCommit:
    """
    Legacy: Represents an unpushed commit with session information.
    Kept for backward compatibility with existing code.
    """

    index: int  # User-visible index (1-based)
    hash: str  # Short commit hash
    full_hash: str  # Full commit hash
    message: str  # First line of commit message
    timestamp: datetime  # Commit timestamp
    llm_summary: str  # Extracted LLM summary
    user_request: Optional[str]  # User's request text
    session_files: List[str]  # Session files modified
    session_additions: Dict[str, List[Tuple[int, int]]]  # {file: [(start, end), ...]}
    has_sensitive: bool = False  # Whether sensitive content detected


def get_unpushed_commits(repo_root: Path) -> List[UnpushedCommit]:
    """
    Legacy: Get all unpushed commits from the shadow git.
    Returns empty list since git tracking is no longer used.
    Use get_sessions_for_export() instead.
    """
    logger.warning("get_unpushed_commits is deprecated, use get_sessions_for_export instead")
    return []


def get_commits_by_hashes(repo_root: Path, hashes: List[str]) -> List[UnpushedCommit]:
    """
    Legacy: Load commits by their hashes.
    Returns empty list since git tracking is no longer used.
    """
    logger.warning("get_commits_by_hashes is deprecated")
    return []


# ============================================================================
# End utility functions
# ============================================================================


# Try to import cryptography
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography package not available, interactive mode disabled")

# Try to import httpx
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx package not available, interactive mode disabled")


def get_line_timestamp(line_json: dict) -> datetime:
    """
    从 JSON 对象中提取时间戳。

    Args:
        line_json: JSON 对象

    Returns:
        datetime 对象，如果没有 timestamp 则返回 datetime.min
    """
    min_ts = datetime.min.replace(tzinfo=timezone.utc)

    if "timestamp" in line_json:
        ts_str = line_json["timestamp"]
        # 处理 ISO 格式: "2025-12-07T17:54:42.618Z"
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            return min_ts
        # Normalize to offset-aware UTC to avoid comparing naive/aware datetimes.
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    return min_ts  # 没有时间戳的放在最前面


def get_session_id(line_json: dict) -> Optional[str]:
    """
    从 JSON 对象中提取 session ID。

    Args:
        line_json: JSON 对象

    Returns:
        session ID 字符串，如果没有则返回 None
    """
    return line_json.get("sessionId")


def extract_messages_from_commit(
    commit: UnpushedCommit, repo_root: Path
) -> Dict[str, List[Tuple[datetime, dict]]]:
    """
    从单个 commit 提取所有新增消息，按 session ID 分组。

    Args:
        commit: UnpushedCommit 对象
        repo_root: shadow git 仓库路径

    Returns:
        字典: {session_id: [(timestamp, json_object), ...]}
    """
    logger.info(f"Extracting messages from commit {commit.hash}")
    session_messages = defaultdict(list)

    for session_file, line_ranges in commit.session_additions.items():
        logger.debug(f"Processing session file: {session_file}")

        # 获取文件内容
        result = subprocess.run(
            ["git", "show", f"{commit.full_hash}:{session_file}"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.warning(f"Failed to get content for {session_file}")
            continue

        lines = result.stdout.split("\n")

        # 提取新增行
        for start, end in line_ranges:
            for line_num in range(start, end + 1):
                if line_num <= len(lines):
                    line = lines[line_num - 1].strip()

                    if not line or "[REDACTED]" in line:
                        continue

                    try:
                        json_obj = json.loads(line)

                        # 跳过已被标记为 redacted 的内容
                        if json_obj.get("redacted"):
                            continue

                        session_id = get_session_id(json_obj)
                        if session_id:
                            timestamp = get_line_timestamp(json_obj)
                            session_messages[session_id].append((timestamp, json_obj))
                            logger.debug(f"Extracted message from session {session_id}")

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON at line {line_num}: {e}")
                        continue

    logger.info(
        f"Extracted {sum(len(msgs) for msgs in session_messages.values())} messages from {len(session_messages)} sessions"
    )
    return session_messages


def merge_messages_from_commits(
    selected_commits: List[UnpushedCommit], repo_root: Path
) -> Dict[str, List[dict]]:
    """
    合并所有选中 commits 的消息，按 session ID 分组并排序。

    Args:
        selected_commits: 选中的 commits 列表
        repo_root: shadow git 仓库路径

    Returns:
        字典: {session_id: [sorted_json_objects]}
    """
    logger.info(f"Merging messages from {len(selected_commits)} commits")
    all_session_messages = defaultdict(list)

    # 按时间顺序处理 commits（从旧到新）
    for commit in reversed(selected_commits):
        commit_messages = extract_messages_from_commit(commit, repo_root)

        for session_id, messages in commit_messages.items():
            all_session_messages[session_id].extend(messages)

    # 排序和去重
    result = {}
    for session_id, messages_with_ts in all_session_messages.items():
        # 按时间戳排序
        sorted_messages = sorted(messages_with_ts, key=lambda x: x[0])

        # 去重（基于 JSON 字符串）
        seen = set()
        unique_messages = []
        for ts, msg in sorted_messages:
            msg_str = json.dumps(msg, sort_keys=True, ensure_ascii=False)
            if msg_str not in seen:
                seen.add(msg_str)
                unique_messages.append(msg)

        result[session_id] = unique_messages
        logger.info(f"Session {session_id}: {len(unique_messages)} unique messages")

    return result


def save_export_file(
    session_messages: Dict[str, List[dict]], output_dir: Path, username: str
) -> Path:
    """
    保存导出文件。

    Args:
        session_messages: {session_id: [messages]}
        output_dir: 输出目录
        username: 用户名

    Returns:
        导出文件路径
    """
    logger.info(f"Saving export file to {output_dir}")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成文件名: username_timestamp.json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{username}_{timestamp}.json"
    output_path = output_dir / filename

    # 构建导出数据
    export_data = {
        "username": username,
        "time": datetime.now().isoformat(),
        "sessions": [
            {"session_id": session_id, "messages": messages}
            for session_id, messages in session_messages.items()
        ],
    }

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Export file saved to {output_path}")
    return output_path


def display_commits_for_selection(commits: List[UnpushedCommit]) -> None:
    """
    显示 commits 供用户选择。

    Args:
        commits: UnpushedCommit 列表
    """
    print(f"\n📋 Available commits ({len(commits)}):\n")

    for commit in commits:
        # Display format: [index] hash - message
        print(f"  [{commit.index}] {commit.hash} - {commit.message}")

        # Show user request if available
        if commit.user_request:
            request_preview = commit.user_request[:60]
            if len(commit.user_request) > 60:
                request_preview += "..."
            print(f"      └─ {request_preview}")

    print()


def display_sessions_for_selection(sessions: List[ExportableSession]) -> None:
    """
    Display sessions for user selection (1-based indices).

    Args:
        sessions: ExportableSession list
    """
    print(f"\n📋 Available sessions ({len(sessions)}):\n")

    for session in sessions:
        session_id_short = session.session_id[:8]
        title = (session.session_title or "").strip()
        if not title:
            title = (session.session_summary or "").strip()
        title = title.replace("\n", " ").strip()
        if title:
            title = title[:80] + ("..." if len(title) > 80 else "")

        last_ts = session.last_activity_at.isoformat() if session.last_activity_at else "unknown"
        turns = session.turn_count
        meta = f"{session.session_type} | turns={turns} | last={last_ts}"
        if title:
            print(f"  [{session.index}] {session_id_short}... - {title}")
            print(f"      └─ {meta}")
        else:
            print(f"  [{session.index}] {session_id_short}... ({meta})")

    print()


def display_events_for_selection(events: List[ExportableEvent]) -> None:
    """
    Display events for user selection (1-based indices).
    New format with improved display and single-selection only.

    Args:
        events: ExportableEvent list
    """
    try:
        from rich.console import Console

        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False
        console = None

    if use_rich:
        console.print(f"\n📋 [bold]Available events ({len(events)}):[/bold]\n")
    else:
        print(f"\n📋 Available events ({len(events)}):\n")

    for event in events:
        # First line: [索引] event_id... | 会话数 | 更新时间
        event_id_short = event.event_id[:8]
        session_count = len(event.sessions)
        updated = event.updated_at.strftime("%Y-%m-%d %H:%M") if event.updated_at else "unknown"

        first_line = f"  [{event.index}] {event_id_short}... | {session_count} sessions | {updated}"
        if use_rich:
            console.print(first_line)
        else:
            print(first_line)

        # Second line: 标题（蓝色，缩进）
        title = (event.title or "(untitled)").replace("\n", " ").strip()
        if title:
            # Truncate title if too long
            if len(title) > 80:
                title = title[:77] + "..."
            if use_rich:
                console.print(f"      [cyan]{title}[/cyan]")
            else:
                print(f"      {title}")

        # Third line: 描述（最多3行，缩进与标题对齐）
        desc = (event.description or "").strip()
        if desc:
            # Split by newlines and process
            desc_lines = desc.replace("\r\n", "\n").split("\n")
            display_lines = []

            for line in desc_lines[:3]:  # 最多3行
                clean_line = line.strip()
                if clean_line:
                    display_lines.append(clean_line)

            # Join and truncate if needed
            desc_text = " ".join(display_lines)

            # Calculate max characters for 3 lines (assuming ~80 chars per line)
            max_chars = 240
            if len(desc_text) > max_chars:
                desc_text = desc_text[: max_chars - 3] + "..."

            # Word wrap for proper display
            import textwrap

            wrapped_lines = textwrap.fill(
                desc_text, width=74, initial_indent="      ", subsequent_indent="      "
            )
            if use_rich:
                console.print(f"[dim]{wrapped_lines}[/dim]")
            else:
                print(wrapped_lines)

        # Empty line between events
        if use_rich:
            console.print()
        else:
            print()

    if use_rich:
        console.print()
    else:
        print()


def merge_messages_from_sessions(
    selected_sessions: List[ExportableSession],
) -> Dict[str, List[dict]]:
    """
    Merge messages from selected sessions (from SQLite turn content).

    Returns:
        {session_id: [sorted_unique_json_objects]}
    """
    logger.info(f"Merging messages from {len(selected_sessions)} session(s)")
    all_session_messages: Dict[str, List[Tuple[datetime, dict]]] = defaultdict(list)
    min_ts = datetime.min.replace(tzinfo=timezone.utc)

    for session in selected_sessions:
        for fallback_ts, msg in get_messages_from_session(session):
            if not msg or msg.get("redacted"):
                continue
            ts = get_line_timestamp(msg)
            if ts == min_ts and fallback_ts:
                if fallback_ts.tzinfo is None:
                    ts = fallback_ts.replace(tzinfo=timezone.utc)
                else:
                    ts = fallback_ts.astimezone(timezone.utc)
            all_session_messages[session.session_id].append((ts, msg))

    result: Dict[str, List[dict]] = {}
    for session_id, messages_with_ts in all_session_messages.items():
        sorted_messages = sorted(messages_with_ts, key=lambda x: x[0])

        seen: Set[str] = set()
        unique_messages: List[dict] = []
        for _, msg in sorted_messages:
            msg_str = json.dumps(msg, sort_keys=True, ensure_ascii=False)
            if msg_str in seen:
                continue
            seen.add(msg_str)
            unique_messages.append(msg)

        result[session_id] = unique_messages
        logger.info(f"Session {session_id}: {len(unique_messages)} unique messages")

    logger.info(
        f"Merged {sum(len(msgs) for msgs in result.values())} messages from {len(result)} session(s)"
    )
    return result


def export_shares_command(
    indices: Optional[str] = None,
    username: Optional[str] = None,
    repo_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> int:
    """
    Main entry point for export shares command.

    Allows users to select events and export their chat history to JSON files.

    Args:
        indices: Event selector to export. Supports:
            - Numeric indices: "1", "1,3,5-7", "all"
            - UUID or prefix: "abc123de" (at least 4 hex chars)
            - Multiple UUIDs: "abc123de,def456gh,xyz789"
            If None, prompts user.
        username: Username for the export. If None, uses system username.
        repo_root: Path to user's project root (used to filter sessions by workspace path)
        output_dir: Custom output directory. If None, uses `<get_realign_dir(repo_root)>/share/`

    Returns:
        0 on success, 1 on error
    """
    logger.info("======== Export shares command started ========")

    if repo_root is None:
        repo_root = Path.cwd().resolve()
    else:
        repo_root = Path(repo_root).resolve()

    from .. import get_realign_dir

    project_data_dir = get_realign_dir(repo_root)

    # Get username
    if username is None:
        username = os.environ.get("USER") or os.environ.get("USERNAME") or "user"

    logger.debug(f"Using username: {username}")

    # Set output directory
    if output_dir is None:
        output_dir = project_data_dir / "share"

    logger.debug(f"Output directory: {output_dir}")

    events = get_events_for_export(workspace_path=str(repo_root), limit=200)
    if not events:
        # Common case: user runs from a subdirectory; fallback to showing all events.
        events = get_events_for_export(workspace_path=None, limit=200)

    if not events:
        print(
            "No events found in database. Run 'aline watcher start' and/or import sessions first.",
            file=sys.stderr,
        )
        return 1

    # Get event selection
    if indices is None:
        display_events_for_selection(events)
        print("Enter event selector (e.g., '1,3,5-7', 'all', or UUID like 'abc123de'):")
        indices_input = input("Selector: ").strip()

        if not indices_input:
            print("No events selected. Exiting.")
            logger.info("No events selected by user")
            return 0
    else:
        indices_input = indices

    # Parse indices - supports numeric indices, UUID, or comma-separated UUIDs
    try:
        if indices_input.lower() == "all":
            indices_list = [e.index for e in events]
        else:
            # First try UUID matching
            indices_list = _find_events_by_uuid(indices_input, events)
            if not indices_list:
                # Fall back to numeric indices
                indices_list = parse_indices(indices_input)
    except ValueError as e:
        print(f"Error: Invalid selector format: {e}", file=sys.stderr)
        print(
            f"Valid formats: 1-{len(events)}, abc123de (UUID prefix), or abc123de,def456gh (multiple UUIDs)",
            file=sys.stderr,
        )
        logger.error(f"Invalid selector format: {e}")
        return 1

    # Validate indices
    max_index = len(events)
    invalid_indices = [i for i in indices_list if i < 1 or i > max_index]
    if invalid_indices:
        print(
            f"Error: Invalid indices (out of range 1-{max_index}): {invalid_indices}",
            file=sys.stderr,
        )
        logger.error(f"Invalid indices: {invalid_indices}")
        return 1

    selected_events = [e for e in events if e.index in indices_list]
    if not selected_events:
        print("No events selected. Exiting.")
        logger.info("No events selected after filtering indices")
        return 0

    selected_session_records_by_id: Dict[str, SessionRecord] = {}
    for event in selected_events:
        for session in event.sessions:
            selected_session_records_by_id[session.id] = session

    min_ts = datetime.min.replace(tzinfo=timezone.utc)
    selected_session_records = sorted(
        selected_session_records_by_id.values(),
        key=lambda s: (
            (
                s.last_activity_at.replace(tzinfo=timezone.utc)
                if s.last_activity_at and s.last_activity_at.tzinfo is None
                else s.last_activity_at.astimezone(timezone.utc)
            )
            if s.last_activity_at
            else min_ts
        ),
        reverse=True,
    )
    selected_sessions = build_exportable_sessions_from_records(selected_session_records)

    logger.info(
        f"Selected {len(selected_events)} event(s) containing {len(selected_sessions)} session(s) to export"
    )

    try:
        session_messages = merge_messages_from_sessions(selected_sessions)
    except Exception as e:
        print(f"\nError: Failed to extract messages: {e}", file=sys.stderr)
        logger.error(f"Failed to extract messages: {e}", exc_info=True)
        return 1

    if not session_messages:
        print("\nWarning: No chat history found in selected sessions.", file=sys.stderr)
        logger.warning("No chat history found in selected sessions")
        return 1

    # Save export file
    try:
        output_path = save_export_file(session_messages, output_dir, username)
    except Exception as e:
        print(f"\nError: Failed to save export file: {e}", file=sys.stderr)
        logger.error(f"Failed to save export file: {e}", exc_info=True)
        return 1

    # Success message
    total_messages = sum(len(msgs) for msgs in session_messages.values())
    print(f"\n✅ Successfully exported {len(session_messages)} session(s)")
    print(f"📁 Export file: {output_path}")
    print(f"📊 Total messages: {total_messages}")
    print()

    logger.info(f"======== Export shares command completed: {output_path} ========")
    return 0


def encrypt_conversation_data(data: dict, password: str) -> dict:
    """
    使用 AES-256-GCM 加密对话数据

    Args:
        data: 要加密的数据字典
        password: 加密密码

    Returns:
        包含加密数据的字典: {encrypted_data, salt, nonce, password_hash}
    """
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography package not installed. Run: pip install cryptography")

    # 生成盐值和随机数
    salt = os.urandom(32)
    nonce = os.urandom(12)

    # 密钥派生
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=1000,
        backend=default_backend(),
    )
    key = kdf.derive(password.encode())

    # 加密数据
    cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
    encryptor = cipher.encryptor()

    json_data = json.dumps(data, ensure_ascii=False).encode("utf-8")
    ciphertext = encryptor.update(json_data) + encryptor.finalize()

    # 添加认证标签
    ciphertext_with_tag = ciphertext + encryptor.tag

    # 计算密码 hash
    password_hash = hashlib.sha256(password.encode()).hexdigest()

    return {
        "encrypted_data": base64.b64encode(ciphertext_with_tag).decode("ascii"),
        "salt": base64.b64encode(salt).decode("ascii"),
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "password_hash": password_hash,
    }


# Chunk size for chunked uploads (2MB - base64 encoded becomes ~2.7MB, safe under 4.5MB limit)
CHUNK_SIZE = 2 * 1024 * 1024  # 2MB
# Threshold to trigger chunked upload (3MB)
CHUNKED_UPLOAD_THRESHOLD = 3 * 1024 * 1024  # 3MB


def upload_to_backend(
    encrypted_payload: dict,
    metadata: dict,
    backend_url: str,
    ui_metadata: Optional[dict] = None,
    progress_callback: Optional[Callable] = None,
    background: bool = False,
) -> dict:
    """
    上传加密数据到后端服务器，自动选择普通上传或分块上传

    Args:
        encrypted_payload: 加密后的数据
        metadata: 元数据
        backend_url: 后端 URL
        ui_metadata: UI 元数据（用于 Open Graph）
        progress_callback: 上传进度回调函数 (current, total, message)
        background: 如果为 True，立即返回 share_url，上传在后台继续（仅对分块上传有效）

    Returns:
        包含 share_id 和 share_url 的字典
    """
    if not HTTPX_AVAILABLE:
        raise RuntimeError("httpx package not installed. Run: pip install httpx")

    # Calculate full payload size (not just encrypted_data)
    # The actual HTTP request includes encrypted_payload + metadata + ui_metadata
    full_payload = {
        "encrypted_payload": encrypted_payload,
        "metadata": metadata,
    }
    if ui_metadata:
        full_payload["ui_metadata"] = ui_metadata

    payload_json = json.dumps(full_payload)
    payload_size = len(payload_json.encode("utf-8"))

    # Also check encrypted_data separately for chunking
    encrypted_data = encrypted_payload.get("encrypted_data", "")
    encrypted_data_size = len(encrypted_data.encode("utf-8"))

    # Always print payload size for debugging
    print(
        f"📊 Payload size: {payload_size / 1024 / 1024:.2f}MB (threshold: {CHUNKED_UPLOAD_THRESHOLD / 1024 / 1024:.2f}MB)"
    )

    # Decide upload method based on size
    # Use chunked upload if either the full payload or encrypted_data exceeds threshold
    if payload_size > CHUNKED_UPLOAD_THRESHOLD or encrypted_data_size > CHUNKED_UPLOAD_THRESHOLD:
        logger.info(
            f"Payload size ({payload_size / 1024 / 1024:.2f}MB) exceeds threshold, using chunked upload"
        )
        print(f"📦 Using chunked upload...")
        return _chunked_upload(
            encrypted_payload, metadata, backend_url, ui_metadata, progress_callback, background
        )
    else:
        logger.info(f"Payload size ({payload_size / 1024:.2f}KB), using standard upload")
        print(f"📤 Using standard upload...")
        return _standard_upload(encrypted_payload, metadata, backend_url, ui_metadata)


def _extract_share_id_from_url(share_url: str) -> Optional[str]:
    try:
        parsed = urlparse(share_url)
        path = parsed.path or ""
        # Expected: /share/<id>
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[-2] == "share":
            return parts[-1]
        # Fallback: last segment
        return parts[-1] if parts else None
    except Exception:
        return None


def _extend_share_expiry(
    backend_url: str,
    share_id: str,
    admin_token: str,
    expiry_days: int,
) -> Optional[datetime]:
    if not HTTPX_AVAILABLE:
        raise RuntimeError("httpx package not installed. Run: pip install httpx")

    response = httpx.post(
        f"{backend_url}/api/share/{share_id}/admin/update",
        headers={"x-admin-token": admin_token, "content-type": "application/json"},
        json={"expiry_days": expiry_days},
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()
    expiry_at = data.get("expiry_at")
    if isinstance(expiry_at, str) and expiry_at:
        try:
            return datetime.fromisoformat(expiry_at.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


def _update_share_content(
    backend_url: str,
    share_id: str,
    token: str,
    conversation_data: dict,
    expected_version: int = 0,
) -> dict:
    """
    Push updated content to an existing share via PUT /api/share/{id}.

    Args:
        backend_url: Backend server URL
        share_id: Share ID on the server
        token: Admin or contributor token for auth
        conversation_data: Full conversation data to replace current content
        expected_version: Optimistic locking version (409 on mismatch)

    Returns:
        Response dict with success, version fields

    Raises:
        RuntimeError on upload failure
        httpx.HTTPStatusError with 409 status on version conflict
    """
    if not HTTPX_AVAILABLE:
        raise RuntimeError("httpx package not installed. Run: pip install httpx")

    # Large payloads can exceed serverless request limits; fall back to chunked update.
    update_payload = {"conversation_data": conversation_data}
    payload_size = len(json.dumps(update_payload).encode("utf-8"))
    if payload_size > CHUNKED_UPLOAD_THRESHOLD:
        logger.info(
            f"Update payload size ({payload_size / 1024 / 1024:.2f}MB) exceeds threshold, using chunked update"
        )
        return _chunked_update_share_content(
            backend_url=backend_url,
            share_id=share_id,
            token=token,
            conversation_data=conversation_data,
            expected_version=expected_version,
        )

    headers = {
        "X-Token": token,
        "X-Expected-Version": str(expected_version),
        "Content-Type": "application/json",
    }
    headers.update(get_auth_headers())

    response = httpx.put(
        f"{backend_url}/api/share/{share_id}",
        headers=headers,
        json=update_payload,
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def _standard_upload(
    encrypted_payload: dict,
    metadata: dict,
    backend_url: str,
    ui_metadata: Optional[dict] = None,
) -> dict:
    """标准单次上传"""
    try:
        payload = {"encrypted_payload": encrypted_payload, "metadata": metadata}
        if ui_metadata:
            payload["ui_metadata"] = ui_metadata

        # Include auth headers for Bearer token authentication
        headers = get_auth_headers()

        response = httpx.post(
            f"{backend_url}/api/share/create",
            json=payload,
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Upload failed: {e}")
        raise RuntimeError(f"Failed to upload to server: {e}")


def _upload_chunks_and_complete(
    chunks: List[str],
    upload_id: str,
    backend_url: str,
    progress_callback: Optional[Callable] = None,
    headers_provider: Optional[Callable[[], Dict[str, str]]] = None,
) -> Optional[dict]:
    """
    Helper function to upload chunks and complete the upload.
    Can be run in background thread.
    """
    import time

    total_chunks = len(chunks)

    def _headers() -> Dict[str, str]:
        try:
            return dict(headers_provider()) if headers_provider else {}
        except Exception:
            return {}

    def _post_with_retries(url: str, payload: dict, timeout: float) -> Optional[dict]:
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = httpx.post(url, json=payload, headers=_headers(), timeout=timeout)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                status = getattr(e.response, "status_code", None)
                retriable = status in (401, 403, 409, 429, 500, 502, 503, 504)
                if retriable and attempt < max_attempts - 1:
                    time.sleep(0.5 * (2**attempt))
                    continue
                logger.error(f"POST failed: {e}")
                return None
            except httpx.HTTPError as e:
                if attempt < max_attempts - 1:
                    time.sleep(0.5 * (2**attempt))
                    continue
                logger.error(f"POST failed: {e}")
                return None

    # Upload each chunk
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i + 1, total_chunks + 2, f"Uploading chunk {i + 1}/{total_chunks}...")

        chunk_payload = {
            "upload_id": upload_id,
            "chunk_index": i,
            "data": chunk,
        }

        result = _post_with_retries(
            f"{backend_url}/api/share/chunk/upload",
            chunk_payload,
            timeout=60.0,  # Longer timeout for chunk uploads
        )
        if not result:
            logger.error(f"Failed to upload chunk {i}")
            return None

        logger.debug(
            f"Chunk {i + 1}/{total_chunks} uploaded, received: {result.get('received_chunks')}"
        )

    # Complete upload
    if progress_callback:
        progress_callback(total_chunks + 1, total_chunks + 2, "Finalizing upload...")

    result = _post_with_retries(
        f"{backend_url}/api/share/chunk/complete",
        {"upload_id": upload_id},
        timeout=60.0,
    )
    if not result:
        logger.error("Failed to complete chunked upload")
        return None

    logger.info(f"Chunked upload completed: {result.get('share_url')}")

    if progress_callback:
        progress_callback(total_chunks + 2, total_chunks + 2, "Upload complete!")

    return result


def _chunked_update_share_content(
    backend_url: str,
    share_id: str,
    token: str,
    conversation_data: dict,
    expected_version: int,
) -> dict:
    data_str = json.dumps(conversation_data)
    data_bytes = data_str.encode("utf-8")
    total_size = len(data_bytes)

    chunks: List[str] = []
    for i in range(0, total_size, CHUNK_SIZE):
        chunk_data = data_bytes[i : i + CHUNK_SIZE]
        chunks.append(base64.b64encode(chunk_data).decode("ascii"))

    total_chunks = len(chunks)
    logger.info(f"Splitting update into {total_chunks} chunks")

    init_payload = {
        "total_chunks": total_chunks,
        "total_size": total_size,
        "metadata": {},
        "encrypted_info": None,
        "ui_metadata": conversation_data.get("ui_metadata"),
        "share_id": share_id,
        "operation": "update",
    }

    def headers_provider() -> Dict[str, str]:
        # Keep bearer auth when available; required for create, optional for update.
        headers = get_auth_headers()
        headers["X-Token"] = token
        headers["X-Expected-Version"] = str(expected_version)
        return headers

    init_headers = headers_provider()
    init_headers["Content-Type"] = "application/json"

    response = httpx.post(
        f"{backend_url}/api/share/chunk/init",
        json=init_payload,
        headers=init_headers,
        timeout=30.0,
    )
    response.raise_for_status()
    init_result = response.json()
    upload_id = init_result["upload_id"]

    result = _upload_chunks_and_complete(
        chunks,
        upload_id,
        backend_url,
        progress_callback=None,
        headers_provider=headers_provider,
    )
    if not result:
        raise RuntimeError("Failed to complete chunked update")
    return result


def _chunked_upload(
    encrypted_payload: dict,
    metadata: dict,
    backend_url: str,
    ui_metadata: Optional[dict] = None,
    progress_callback: Optional[Callable] = None,
    background: bool = False,
) -> dict:
    """
    分块上传大文件

    流程：
    1. 初始化上传会话 (POST /api/share/chunk/init)
    2. 逐块上传数据 (POST /api/share/chunk/upload)
    3. 完成上传 (POST /api/share/chunk/complete)

    Args:
        encrypted_payload: 加密后的数据
        metadata: 元数据
        backend_url: 后端 URL
        ui_metadata: UI 元数据（用于 Open Graph）
        progress_callback: 上传进度回调函数
        background: 如果为 True，在 init 后立即返回 share_url，上传在后台继续

    Returns:
        包含 share_id 和 share_url 的字典。
        如果 background=True，返回时上传可能还未完成。
    """
    encrypted_data = encrypted_payload.get("encrypted_data", "")
    data_bytes = encrypted_data.encode("utf-8")
    total_size = len(data_bytes)

    # Calculate chunks
    chunks = []
    for i in range(0, total_size, CHUNK_SIZE):
        chunk_data = data_bytes[i : i + CHUNK_SIZE]
        # Encode chunk as base64 for safe JSON transmission
        chunks.append(base64.b64encode(chunk_data).decode("ascii"))

    total_chunks = len(chunks)
    logger.info(f"Splitting data into {total_chunks} chunks")

    if progress_callback:
        progress_callback(0, total_chunks + 2, "Initializing chunked upload...")

    def headers_provider() -> Dict[str, str]:
        # Refresh token if needed between chunks.
        headers = get_auth_headers()
        headers["Content-Type"] = "application/json"
        return headers

    # Step 1: Initialize upload session (now returns share_url immediately)
    try:
        init_payload = {
            "total_chunks": total_chunks,
            "total_size": total_size,
            "metadata": metadata,
            "encrypted_info": {
                "salt": encrypted_payload.get("salt", ""),
                "nonce": encrypted_payload.get("nonce", ""),
                "password_hash": encrypted_payload.get("password_hash", ""),
            },
            "ui_metadata": ui_metadata,
        }

        response = httpx.post(
            f"{backend_url}/api/share/chunk/init",
            json=init_payload,
            headers=headers_provider(),
            timeout=30.0,
        )
        response.raise_for_status()
        init_result = response.json()
        upload_id = init_result["upload_id"]
        # New: init now returns share_url immediately
        share_url = init_result.get("share_url")
        share_id = init_result.get("share_id")
        admin_token = init_result.get("admin_token")
        expiry_at = init_result.get("expiry_at")
        logger.info(f"Chunked upload initialized: {upload_id}, share_url: {share_url}")

    except httpx.HTTPError as e:
        logger.error(f"Failed to initialize chunked upload: {e}")
        raise RuntimeError(f"Failed to initialize chunked upload: {e}")

    # If background mode, return immediately with share_url and upload in background
    if background and share_url:
        # Start background thread for chunk upload
        # Note: NOT a daemon thread - we want it to complete even if main exits
        # The CLI will wait for this thread to finish before exiting,
        # but user already has the share URL displayed
        thread = threading.Thread(
            target=_upload_chunks_and_complete,
            args=(chunks, upload_id, backend_url, None, headers_provider),  # No callback in background
            daemon=False,  # Important: let thread complete before process exits
        )
        thread.start()
        logger.info(f"Background upload started for {upload_id}")

        return {
            "share_id": share_id,
            "share_url": share_url,
            "admin_token": admin_token,
            "expiry_at": expiry_at,
            "upload_pending": True,
        }

    # Foreground mode: upload chunks synchronously
    result = _upload_chunks_and_complete(
        chunks, upload_id, backend_url, progress_callback, headers_provider
    )
    if not result:
        raise RuntimeError("Failed to complete chunked upload")
    # Preserve init fields if server didn't echo them
    result.setdefault("share_id", share_id)
    result.setdefault("share_url", share_url)
    result.setdefault("admin_token", admin_token)
    result.setdefault("expiry_at", expiry_at)
    return result


def upload_to_backend_unencrypted(
    conversation_data: dict,
    metadata: dict,
    backend_url: str,
    progress_callback: Optional[Callable] = None,
    background: bool = False,
) -> dict:
    """
    上传非加密数据到后端服务器，自动选择普通上传或分块上传

    Args:
        conversation_data: 对话数据
        metadata: 元数据
        backend_url: 后端 URL
        progress_callback: 上传进度回调函数
        background: 如果为 True，立即返回 share_url，上传在后台继续（仅对分块上传有效）

    Returns:
        包含 share_id 和 share_url 的字典
    """
    if not HTTPX_AVAILABLE:
        raise RuntimeError("httpx package not installed. Run: pip install httpx")

    # Calculate full payload size
    full_payload = {"conversation_data": conversation_data, "metadata": metadata}
    payload_json = json.dumps(full_payload)
    payload_size = len(payload_json.encode("utf-8"))

    # Always print payload size for debugging
    print(
        f"📊 Payload size: {payload_size / 1024 / 1024:.2f}MB (threshold: {CHUNKED_UPLOAD_THRESHOLD / 1024 / 1024:.2f}MB)"
    )

    # Decide upload method based on size
    if payload_size > CHUNKED_UPLOAD_THRESHOLD:
        logger.info(
            f"Payload size ({payload_size / 1024 / 1024:.2f}MB) exceeds threshold, using chunked upload"
        )
        print(f"📦 Using chunked upload...")
        return _chunked_upload_unencrypted(
            conversation_data, metadata, backend_url, progress_callback, background
        )
    else:
        logger.info(f"Payload size ({payload_size / 1024:.2f}KB), using standard upload")
        print(f"📤 Using standard upload...")
        # Standard upload with auth headers
        try:
            headers = get_auth_headers()
            response = httpx.post(
                f"{backend_url}/api/share/create",
                json=full_payload,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Upload failed: {e}")
            raise RuntimeError(f"Failed to upload to server: {e}")


def _chunked_upload_unencrypted(
    conversation_data: dict,
    metadata: dict,
    backend_url: str,
    progress_callback: Optional[Callable] = None,
    background: bool = False,
) -> dict:
    """
    分块上传非加密大文件

    流程：
    1. 初始化上传会话 (POST /api/share/chunk/init)
    2. 逐块上传数据 (POST /api/share/chunk/upload)
    3. 完成上传 (POST /api/share/chunk/complete)

    Args:
        conversation_data: 对话数据
        metadata: 元数据
        backend_url: 后端 URL
        progress_callback: 上传进度回调函数
        background: 如果为 True，在 init 后立即返回 share_url，上传在后台继续

    Returns:
        包含 share_id 和 share_url 的字典。
        如果 background=True，返回时上传可能还未完成。
    """
    # Serialize conversation data to JSON string
    data_str = json.dumps(conversation_data)
    data_bytes = data_str.encode("utf-8")
    total_size = len(data_bytes)

    # Calculate chunks
    chunks = []
    for i in range(0, total_size, CHUNK_SIZE):
        chunk_data = data_bytes[i : i + CHUNK_SIZE]
        # Encode chunk as base64 for safe JSON transmission
        chunks.append(base64.b64encode(chunk_data).decode("ascii"))

    total_chunks = len(chunks)
    logger.info(f"Splitting unencrypted data into {total_chunks} chunks")

    if progress_callback:
        progress_callback(0, total_chunks + 2, "Initializing chunked upload...")

    def headers_provider() -> Dict[str, str]:
        headers = get_auth_headers()
        headers["Content-Type"] = "application/json"
        return headers

    # Step 1: Initialize upload session (now returns share_url immediately)
    try:
        init_payload = {
            "total_chunks": total_chunks,
            "total_size": total_size,
            "metadata": metadata,
            "encrypted_info": None,  # No encryption
            "ui_metadata": conversation_data.get("ui_metadata"),
        }

        response = httpx.post(
            f"{backend_url}/api/share/chunk/init",
            json=init_payload,
            headers=headers_provider(),
            timeout=30.0,
        )
        response.raise_for_status()
        init_result = response.json()
        upload_id = init_result["upload_id"]
        # New: init now returns share_url immediately
        share_url = init_result.get("share_url")
        share_id = init_result.get("share_id")
        admin_token = init_result.get("admin_token")
        expiry_at = init_result.get("expiry_at")
        logger.info(f"Chunked upload initialized: {upload_id}, share_url: {share_url}")

    except httpx.HTTPError as e:
        logger.error(f"Failed to initialize chunked upload: {e}")
        raise RuntimeError(f"Failed to initialize chunked upload: {e}")

    # If background mode, return immediately with share_url and upload in background
    if background and share_url:
        # Start background thread for chunk upload
        # Note: NOT a daemon thread - we want it to complete even if main exits
        # The CLI will wait for this thread to finish before exiting,
        # but user already has the share URL displayed
        thread = threading.Thread(
            target=_upload_chunks_and_complete,
            args=(chunks, upload_id, backend_url, None, headers_provider),  # No callback in background
            daemon=False,  # Important: let thread complete before process exits
        )
        thread.start()
        logger.info(f"Background upload started for {upload_id}")

        return {
            "share_id": share_id,
            "share_url": share_url,
            "admin_token": admin_token,
            "expiry_at": expiry_at,
            "upload_pending": True,
        }

    # Foreground mode: upload chunks synchronously
    result = _upload_chunks_and_complete(
        chunks, upload_id, backend_url, progress_callback, headers_provider
    )
    if not result:
        raise RuntimeError("Failed to complete chunked upload")
    result.setdefault("share_id", share_id)
    result.setdefault("share_url", share_url)
    result.setdefault("admin_token", admin_token)
    result.setdefault("expiry_at", expiry_at)
    return result


def clean_text_for_prompt(text: str) -> str:
    """
    清理文本中的控制字符，使其适合在 LLM prompt 中使用

    Args:
        text: 原始文本

    Returns:
        清理后的文本
    """
    # 替换控制字符为空格或删除
    import re

    # 保留换行符和制表符，删除其他控制字符
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    # 将多个连续空格合并为一个
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _get_share_ui_metadata_prompt_template() -> Optional[str]:
    """
    Load share UI metadata prompt template with user customization support.

    Priority:
    1. User custom prompt (~/.aline/prompts/share_ui_metadata.md)
    2. None (will use hardcoded default in generate_ui_metadata_with_llm)

    Returns:
        Custom prompt template or None
    """
    global _SHARE_UI_METADATA_PROMPT_CACHE
    if _SHARE_UI_METADATA_PROMPT_CACHE is not None:
        return _SHARE_UI_METADATA_PROMPT_CACHE

    # Try user-customized prompt (~/.aline/prompts/share_ui_metadata.md)
    user_prompt_path = Path.home() / ".aline" / "prompts" / "share_ui_metadata.md"
    try:
        if user_prompt_path.exists():
            text = user_prompt_path.read_text(encoding="utf-8").strip()
            if text:
                _SHARE_UI_METADATA_PROMPT_CACHE = text
                logger.debug(
                    f"Loaded user-customized share UI metadata prompt from {user_prompt_path}"
                )
                return text
    except Exception:
        logger.debug(
            "Failed to load user-customized share UI metadata prompt, using default",
            exc_info=True,
        )

    # No custom prompt, will use default
    return None


def generate_ui_metadata_with_llm(
    conversation_data: dict,
    selected_commits: List,
    event_title: str,
    event_description: str,
    provider: str = "auto",
    preset_id: str = "default",
    silent: bool = False,
) -> Tuple[Optional[dict], Optional[dict]]:
    """
    使用 LLM 根据对话内容生成个性化的 UI 元数据

    Args:
        conversation_data: 对话数据字典 {username, time, sessions}
        selected_commits: 选中的 UnpushedCommit 列表
        event_title: Event 的标题（直接使用，不由 LLM 生成）
        event_description: Event 的描述（直接使用，不由 LLM 生成）
        provider: LLM provider ("auto", "claude", "openai")
        preset_id: Prompt preset ID，用于调整生成风格
        silent: 是否静默输出 (默认: False)

    Returns:
        Tuple[ui_metadata, debug_info]
        - ui_metadata: UI 元数据字典，包含 title (from event), description (from event), welcome, agent_sections, slack_message
        - debug_info: {system_prompt, user_prompt, response_text, provider} 或 None
        如果生成失败则都返回 None
    """
    if not silent:
        logger.info(f"Generating UI metadata with LLM (preset: {preset_id})")

    # 构建对话内容摘要

    sessions = conversation_data.get("sessions", [])
    total_messages = sum(len(s.get("messages", [])) for s in sessions)

    # Prefer a structured outline (sessions + session title/description + turn titles),
    # so ui_metadata generation stays robust even when message content is absent.
    outline_sessions: list[dict] = []
    total_turns_added = 0
    max_total_turns = 5000  # safety guard; titles-only but can still be huge
    try:
        for i, session in enumerate(sessions, start=1):
            session_title = session.get("session_title")
            session_desc = session.get("session_summary")
            session_type = session.get("session_type")

            title = clean_text_for_prompt(str(session_title)) if session_title else ""
            description = clean_text_for_prompt(str(session_desc)) if session_desc else ""
            if len(title) > 200:
                title = title[:200].rstrip()
            if len(description) > 800:
                description = description[:800].rstrip()

            turns_outline: list[dict] = []
            for turn in (session.get("turns") or []):
                if total_turns_added >= max_total_turns:
                    break
                turn_number = (turn or {}).get("turn_number")
                turn_title_raw = (turn or {}).get("llm_title") or ""
                turn_title = clean_text_for_prompt(str(turn_title_raw)) if turn_title_raw else ""
                if not turn_title:
                    # Keep structure stable even when titles are missing.
                    turn_title = "(untitled)"
                if len(turn_title) > 220:
                    turn_title = turn_title[:220].rstrip()
                try:
                    tn = int(turn_number) if turn_number is not None else len(turns_outline) + 1
                except Exception:
                    tn = len(turns_outline) + 1
                turns_outline.append({"turn_number": tn, "title": turn_title})
                total_turns_added += 1

            outline_sessions.append(
                {
                    "session_number": i,
                    "session_type": str(session_type) if session_type else "",
                    "title": title,
                    "description": description,
                    "turns": turns_outline,
                }
            )
            if total_turns_added >= max_total_turns:
                break
    except Exception:
        outline_sessions = []

    # Try cloud provider first if logged in
    if provider in ("auto", "cloud") and is_logged_in():
        logger.debug("Attempting cloud LLM for UI metadata generation")
        # Load user custom prompt if available
        custom_prompt = _get_share_ui_metadata_prompt_template()

        # Build payload
        cloud_payload = {
            "event_title": event_title,
            "event_description": event_description,
            "sessions_count": len(sessions),
            "total_messages": total_messages,
        }
        if outline_sessions:
            cloud_payload["sessions"] = outline_sessions

        model_name, result = call_llm_cloud(
            task="ui_metadata",
            payload=cloud_payload,
            custom_prompt=custom_prompt,
            preset_id=preset_id,
            silent=silent,
        )

        if result:
            # Normalize agent_sections so front-end sidebar/TOC can render safely.
            normalized_agent_sections = []
            raw_sections = result.get("agent_sections")
            if isinstance(raw_sections, list):
                for raw_section in raw_sections[:4]:
                    if not isinstance(raw_section, dict):
                        continue
                    heading = str(raw_section.get("heading") or "").strip()
                    content = str(raw_section.get("content") or "").strip()
                    if not heading:
                        continue

                    normalized_subsections = []
                    raw_subsections = raw_section.get("subsections")
                    if isinstance(raw_subsections, list):
                        for raw_sub in raw_subsections[:8]:
                            if not isinstance(raw_sub, dict):
                                continue
                            sub_heading = str(raw_sub.get("heading") or "").strip()
                            sub_content = str(raw_sub.get("content") or "").strip()
                            if not sub_heading:
                                continue
                            raw_items = raw_sub.get("items")
                            items = []
                            if isinstance(raw_items, list):
                                items = [
                                    str(item).strip()
                                    for item in raw_items[:10]
                                    if isinstance(item, str) and str(item).strip()
                                ]
                            sub_payload = {"heading": sub_heading, "content": sub_content}
                            if items:
                                sub_payload["items"] = items
                            normalized_subsections.append(sub_payload)

                    section_payload = {"heading": heading, "content": content}
                    if normalized_subsections:
                        section_payload["subsections"] = normalized_subsections
                    normalized_agent_sections.append(section_payload)

            if not normalized_agent_sections:
                fallback_heading = (event_title or "").strip() or "Context Overview"
                fallback_content = (event_description or "").strip()
                if not fallback_content:
                    fallback_content = (
                        "This shared context includes key conversations, decisions, and outcomes. "
                        "Use the chat to explore details and ask follow-up questions."
                    )
                normalized_agent_sections = [
                    {
                        "heading": fallback_heading,
                        "content": fallback_content,
                    }
                ]

            # Build ui_metadata from result
            ui_metadata = {
                "title": event_title,
                "description": event_description,
                "welcome": "",  # No longer generated by LLM
                "agent_sections": normalized_agent_sections,
                "slack_message": result.get("slack_message", ""),
            }
            debug_info = {
                "system_prompt": "(cloud)",
                "user_prompt": "(cloud)",
                "response_text": str(result),
                "provider": model_name or "cloud",
            }
            logger.info(f"Cloud LLM UI metadata generation success ({model_name})")
            return ui_metadata, debug_info
        else:
            # Cloud LLM failed, return None (local fallback disabled)
            logger.warning("Cloud LLM UI metadata failed")
            if not silent:
                print("   ⚠️  Cloud LLM UI metadata failed", file=sys.stderr)
            return None, None

    # User not logged in, return None (local fallback disabled)
    logger.warning("Not logged in, cannot use cloud LLM for UI metadata")
    if not silent:
        print("   ⚠️  Please login with 'aline login' to use LLM features", file=sys.stderr)
    return None, None

    # =========================================================================
    # LOCAL LLM FALLBACK DISABLED - Code kept for reference
    # =========================================================================
    # # 根据 preset_id 定制 system_prompt
    # preset_configs = {
    #     "default": {
    #         "role_description": "a general-purpose conversation assistant",
    #         "title_style": "a neutral, descriptive summary of the topic",
    #         "welcome_tone": "friendly and informative, with a brief overview of the conversation",
    #         "description_focus": "what information can be found and how the assistant can help",
    #         "question_angles": [
    #             "high-level summary",
    #             "technical or implementation details",
    #             "decision-making or reasoning",
    #             "results, impact, or follow-up",
    #         ],
    #     },
    #     "work-report": {
    #         "role_description": "a professional work report agent representing the user to colleagues/managers",
    #         "title_style": "a professional, achievement-oriented summary",
    #         "welcome_tone": "professional and confident, highlighting accomplishments and progress",
    #         "description_focus": "what work was done, what value was created",
    #         "question_angles": [
    #             "overall progress and achievements",
    #             "technical solutions implemented",
    #             "challenges overcome and decisions made",
    #             "next steps and impact on project goals",
    #         ],
    #     },
    #     "knowledge-agent": {
    #         "role_description": "a knowledge-sharing agent representing the user's deep thinking",
    #         "title_style": "a thought-provoking, conceptual title",
    #         "welcome_tone": "insightful and educational",
    #         "description_focus": "the knowledge and insights shared",
    #         "question_angles": [
    #             "core concepts and philosophy",
    #             "design rationale and trade-offs",
    #             "key insights and learning",
    #             "practical implications and applications",
    #         ],
    #     },
    #     "personality-analyzer": {
    #         "role_description": "a personality analysis assistant",
    #         "title_style": "an analytical, personality-focused title",
    #         "welcome_tone": "analytical yet friendly",
    #         "description_focus": "personality traits, working styles, and communication patterns",
    #         "question_angles": [
    #             "overall personality traits and characteristics",
    #             "working style and approach to problem-solving",
    #             "communication patterns and preferences",
    #             "strengths, growth areas, and unique qualities",
    #         ],
    #     },
    # }
    #
    # # Check for user-customized prompt first
    # custom_prompt = _get_share_ui_metadata_prompt_template()
    # if custom_prompt:
    #     system_prompt = custom_prompt
    #     logger.info("Using user-customized share UI metadata prompt")
    # else:
    #     preset_config = preset_configs.get(preset_id, preset_configs["default"])
    #     system_prompt = f"""You are a conversation interface copy generator for {preset_config["role_description"]}..."""
    #
    # # 构建 user prompt - 优先使用 commit summaries
    # if commit_summaries:
    #     user_prompt = f"""Analyze the following conversation..."""
    # else:
    #     user_prompt = f"""Analyze the following conversation..."""
    #
    # # Use unified LLM client
    # try:
    #     model_name, response_text = call_llm(
    #         system_prompt=system_prompt,
    #         user_prompt=user_prompt,
    #         provider=provider,
    #         max_tokens=1000,
    #         temperature=0.7,
    #         purpose="ui_metadata",
    #         silent=silent,
    #     )
    #
    #     if not response_text:
    #         return None, None
    #
    #     ui_metadata = extract_json(response_text)
    #     ui_metadata["title"] = event_title
    #     ui_metadata["description"] = event_description
    #     ui_metadata["welcome"] = ""
    #
    #     debug_info = {
    #         "system_prompt": system_prompt,
    #         "user_prompt": user_prompt,
    #         "response_text": response_text,
    #         "provider": model_name or provider,
    #     }
    #     return ui_metadata, debug_info
    #
    # except Exception as e:
    #     logger.error(f"LLM UI metadata generation failed: {e}", exc_info=True)
    #     return None, None


def display_selection_statistics(
    selected_commits: List[UnpushedCommit], session_messages: Dict[str, List[dict]]
) -> None:
    """
    显示选择的统计信息

    Args:
        selected_commits: 选中的commits
        session_messages: 合并后的session消息
    """
    try:
        from rich.console import Console
        from rich.panel import Panel
    except ImportError:
        # Fallback to plain text if rich is not available
        print(f"\n📊 Selection Summary:")
        print(f"  Commits: {len(selected_commits)}")
        print(f"  Sessions: {len(session_messages)}")
        total_messages = sum(len(msgs) for msgs in session_messages.values())
        print(f"  Messages: {total_messages}")
        return

    console = Console()

    # 计算统计信息
    total_sessions = len(session_messages)
    total_messages = sum(len(msgs) for msgs in session_messages.values())

    # 消息角色分布
    user_messages = 0
    assistant_messages = 0
    other_messages = 0

    for messages in session_messages.values():
        for msg in messages:
            role = msg.get("role", "unknown")
            if role == "user":
                user_messages += 1
            elif role == "assistant":
                assistant_messages += 1
            else:
                other_messages += 1

    # 时间范围
    all_timestamps = []
    for messages in session_messages.values():
        for msg in messages:
            if "timestamp" in msg:
                ts_str = msg["timestamp"]
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    all_timestamps.append(ts)
                except:
                    pass

    time_info = ""
    if all_timestamps:
        earliest = min(all_timestamps)
        latest = max(all_timestamps)
        duration = latest - earliest

        time_info = f"""
[bold]Time Range:[/bold]
  {earliest.strftime("%Y-%m-%d %H:%M")} → {latest.strftime("%Y-%m-%d %H:%M")}
  Duration: {duration.days}d {duration.seconds // 3600}h {(duration.seconds % 3600) // 60}m
"""

    # 构建统计文本
    stats_text = f"""[bold cyan]Commits Selected:[/bold cyan] {len(selected_commits)}

[bold green]Sessions:[/bold green] {total_sessions}

[bold yellow]Messages:[/bold yellow] {total_messages}
  ├─ User:      {user_messages}
  ├─ Assistant: {assistant_messages}"""

    if other_messages > 0:
        stats_text += f"\n  └─ Other:     {other_messages}"

    if time_info:
        stats_text += "\n" + time_info

    panel = Panel(
        stats_text,
        title="[bold]📊 Selection Summary[/bold]",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)


def display_session_preview(
    session_messages: Dict[str, List[dict]], max_sessions: int = 10
) -> None:
    """
    显示session预览（简化版）

    Args:
        session_messages: {session_id: [messages]}
        max_sessions: 最多显示的session数量
    """
    total_sessions = len(session_messages)
    total_messages = sum(len(msgs) for msgs in session_messages.values())

    print(f"\n📝 {total_sessions} sessions, {total_messages} messages")


def display_ui_metadata_preview(ui_metadata: dict) -> None:
    """
    显示UI metadata预览

    Args:
        ui_metadata: UI metadata字典
    """
    # Simple text output
    print("\n--- UI Content ---")

    # Title
    title = ui_metadata.get("title", "[No title]")
    print(f"Title: {title}")

    # Description
    desc = ui_metadata.get("description", "[No description]")
    print(f"Description: {desc}")

    # Agent Sections
    sections = ui_metadata.get("agent_sections", [])
    if isinstance(sections, list) and sections:
        print(f"\nAgent Sections:")
        for i, section in enumerate(sections, 1):
            heading = section.get("heading") if isinstance(section, dict) else ""
            content = section.get("content") if isinstance(section, dict) else ""
            print(f"  {i}. {heading}")
            if content:
                print(f"     {content}")

    # Slack Message
    slack_msg = ui_metadata.get("slack_message", "")
    if slack_msg:
        print(f"\nSlack Message:")
        print(f"  {slack_msg}")


def display_share_result(
    share_url: str,
    password: str,
    expiry_days: int,
    max_views: int,
    admin_token: Optional[str] = None,
) -> None:
    """
    显示分享创建结果

    Args:
        share_url: 分享URL
        password: 加密密码
        expiry_days: 过期天数
        max_views: 最大浏览次数
        admin_token: 管理员token(可选)
    """
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        # Fallback to plain text
        print("\n╭────────────────────────────────────────╮")
        print("│  ✅ Share Created Successfully!        │")
        print("╰────────────────────────────────────────╯\n")
        print(f"🔗 Share URL: {share_url}")
        print(f"🔑 Password: {password}")
        print(f"📅 Expires: {expiry_days} days")
        print(f"👁️  Max views: {max_views}")
        if admin_token:
            print(f"\n📊 Admin token: {admin_token}")
            print(f"   View stats at: {share_url}/stats?token={admin_token}")
        print("\n💡 Share this URL and password with your team!")
        return

    console = Console()

    # Simplified output without panels
    console.print("\n[bold green]✅ Share Created[/bold green]")
    console.print(f"\n[bold]🔗 URL:[/bold] {share_url}")
    console.print(f"[bold]🔑 Password:[/bold] {password}")
    console.print(f"[bold]📅 Expires:[/bold] {expiry_days} days")
    console.print(f"[bold]👁️  Max Views:[/bold] {max_views}\n")


def _run_clipboard_command(command: list[str], text: str) -> bool:
    try:
        return (
            subprocess.run(
                command,
                input=text,
                text=True,
                capture_output=False,
                check=False,
            ).returncode
            == 0
        )
    except Exception:
        return False


def _copy_text_to_clipboard(text: str) -> bool:
    if not text:
        return False

    if shutil.which("pbcopy"):
        if _run_clipboard_command(["pbcopy"], text):
            return True

    if os.name == "nt" and shutil.which("clip"):
        if _run_clipboard_command(["clip"], text):
            return True

    if shutil.which("wl-copy"):
        if _run_clipboard_command(["wl-copy"], text):
            return True

    if shutil.which("xclip"):
        if _run_clipboard_command(["xclip", "-selection", "clipboard"], text):
            return True

    if shutil.which("xsel"):
        if _run_clipboard_command(["xsel", "--clipboard", "--input"], text):
            return True

    return False


def _copy_share_to_clipboard(share_url: Optional[str], slack_message: Optional[str]) -> bool:
    if not share_url:
        return False
    if slack_message:
        text_to_copy = f"{slack_message}\n\n{share_url}"
    else:
        text_to_copy = share_url
    return _copy_text_to_clipboard(text_to_copy)


def _export_by_events_interactive(
    all_commits: List,
    shadow_git: Path,
    password: Optional[str] = None,
    expiry_days: int = 7,
    max_views: int = 100,
    enable_preview: bool = True,
    backend_url: Optional[str] = None,
    repo_root: Optional[Path] = None,
    preset: Optional[str] = None,
) -> int:
    """
    Interactive export using event selection.

    Args:
        all_commits: List of all unpushed commits
        shadow_git: Path to shadow git directory
        password: Encryption password
        expiry_days: Expiry days
        max_views: Maximum views
        enable_preview: Enable UI preview
        backend_url: Backend URL
        repo_root: Repository root
        preset: Prompt preset ID

    Returns:
        0 on success, 1 on error
    """
    from ..storage.event_storage import EventStorage
    from ..models.event import EventType

    # Load events from storage (auto-generated by session_summarizer)
    storage = EventStorage(shadow_git)
    collection = storage.load_events()
    events = [e for e in collection.events if e.event_type == EventType.TASK]

    if not events:
        print("No events available yet.")
        print("Tip: events are auto-generated by session_summarizer during watcher sessions")
        return 1

    # Default: show 3-7 most recent events, but allow showing all (tree view).
    events_sorted = sorted(
        events,
        key=lambda e: (e.updated_at or e.end_timestamp or e.created_at),
        reverse=True,
    )
    default_limit = 7
    show_choice = (
        input(f"Show [r]ecent events ({min(default_limit, len(events_sorted))}) or [a]ll? [r]: ")
        .strip()
        .lower()
    )
    events_to_show = events_sorted if show_choice.startswith("a") else events_sorted[:default_limit]

    def _event_path(ev) -> List[str]:
        ui = ev.ui_metadata or {}
        p = ui.get("taxonomy_path")
        if isinstance(p, list) and all(isinstance(x, str) and x.strip() for x in p):
            return [x.strip() for x in p][:5]
        title = (ev.title or "").strip()
        if " / " in title:
            return [x.strip() for x in title.split(" / ") if x.strip()][:5]
        return [title] if title else ["(untitled)"]

    # Build a pruned tree containing only selected nodes + ancestors.
    from dataclasses import dataclass

    @dataclass
    class _Node:
        name: str
        children: dict
        event: Optional[object] = None
        path: Optional[List[str]] = None

    root = _Node(name="ROOT", children={}, event=None, path=[])

    def ensure_path(path: List[str]) -> _Node:
        node = root
        current_path: List[str] = []
        for part in path:
            current_path.append(part)
            if part not in node.children:
                node.children[part] = _Node(
                    name=part, children={}, event=None, path=list(current_path)
                )
            node = node.children[part]
        return node

    for ev in events_to_show:
        path = _event_path(ev)
        node = ensure_path(path)
        node.event = ev

    # Render tree and assign indices to selectable nodes (only nodes that are actual events).
    # This keeps the default UX simple: users pick 3-7 events, then confirm included commits.
    index_to_event: Dict[int, object] = {}
    next_index = 1
    tree_items: List[Dict[str, Any]] = []

    def walk(node: _Node, indent: int) -> None:
        nonlocal next_index
        for key in sorted(node.children.keys(), key=lambda s: s.lower()):
            child: _Node = node.children[key]
            if child.event is not None and getattr(child.event, "commit_hashes", None):
                idx = next_index
                next_index += 1
                index_to_event[idx] = child.event
                commit_count = len(getattr(child.event, "commit_hashes", []) or [])
                desc = (getattr(child.event, "description", "") or "").strip()
                tree_items.append(
                    {
                        "kind": "event",
                        "indent": indent,
                        "idx": idx,
                        "label": key,
                        "commit_count": commit_count,
                        "desc": desc if (desc and not desc.startswith("Auto-added")) else "",
                    }
                )
            else:
                tree_items.append({"kind": "node", "indent": indent, "label": key})
            walk(child, indent + 1)

    walk(root, 0)

    from rich.console import Console
    from rich.padding import Padding
    from rich.text import Text

    console = Console()

    def _render_markdown_lite(s: str) -> Text:
        """
        Render a small Markdown subset while preserving whitespace/newlines.

        Supported:
        - **bold**  -> styled as "bold yellow"
        - `code`    -> styled as "bold cyan"
        """
        s = s or ""
        out = Text()
        i = 0
        n = len(s)
        while i < n:
            if s.startswith("**", i):
                j = s.find("**", i + 2)
                if j != -1:
                    out.append(s[i + 2 : j], style="bold yellow")
                    i = j + 2
                    continue
            if s[i] == "`":
                j = s.find("`", i + 1)
                if j != -1:
                    out.append(s[i + 1 : j], style="bold cyan")
                    i = j + 1
                    continue
            out.append(s[i])
            i += 1
        return out

    console.print(
        f"\n📦 Available event taxonomy ({len(index_to_event)} selectable events shown / {len(events)} total):\n"
    )
    for item in tree_items:
        indent = int(item.get("indent") or 0)
        if item.get("kind") == "event":
            idx = int(item["idx"])
            label = str(item["label"])
            commit_count = int(item["commit_count"])
            line = f"[{idx}] {label} ({commit_count} commits)"
            console.print(Padding(_render_markdown_lite(line), (0, 0, 0, 2 * indent)))
            desc = (item.get("desc") or "").strip()
            if desc:
                console.print(Padding(_render_markdown_lite(desc), (0, 0, 1, 2 * (indent + 1))))
        else:
            console.print(Padding(_render_markdown_lite(str(item["label"])), (0, 0, 0, 2 * indent)))
    console.print()

    # Get user selection
    print("Enter event selector (e.g., '1,3,5-7', 'all', or UUID like 'abc123de'):")
    selection = input("Selector: ").strip()

    if not selection:
        print("No events selected. Exiting.")
        return 0

    # Parse selection - supports numeric indices, UUID, or comma-separated UUIDs
    try:
        if selection.lower() == "all":
            selected_events = list(index_to_event.values())
        else:
            # Build a list for UUID matching
            events_list = [(idx, ev) for idx, ev in index_to_event.items()]
            # First try UUID matching
            uuid_indices = _find_events_by_uuid(selection, [ev for _, ev in events_list])
            if uuid_indices:
                # Map back to original indices
                indices_list = [events_list[i - 1][0] for i in uuid_indices]
            else:
                # Fall back to numeric indices
                indices_list = parse_commit_indices(selection)
            selected_events = [index_to_event[i] for i in indices_list if i in index_to_event]

        if not selected_events:
            print("No valid events selected.")
            return 0

    except ValueError as e:
        print(f"Error: Invalid selector format: {e}", file=sys.stderr)
        print(
            "Valid formats: 1,3,5-7, all, abc123de (UUID prefix), or abc123de,def456gh (multiple UUIDs)"
        )
        return 1

    all_commit_hashes = set()
    for ev in selected_events:
        all_commit_hashes.update(getattr(ev, "commit_hashes", []) or [])

    # Get corresponding commits (events may include pushed commits; load by hashes if missing)
    selected_commits = [c for c in all_commits if c.full_hash in all_commit_hashes]
    missing_hashes = list(all_commit_hashes - {c.full_hash for c in selected_commits})
    if missing_hashes:
        # Note: get_commits_by_hashes returns empty list since git tracking is removed
        try:
            loaded = get_commits_by_hashes(shadow_git, missing_hashes)
            selected_commits.extend(loaded)
        except Exception as e:
            logger.error(f"Failed to load commits by hashes for events: {e}", exc_info=True)
            print(
                f"Error: Failed to load commits for selected events: {e}",
                file=sys.stderr,
            )
            return 1

    selected_commits = sorted(selected_commits, key=lambda c: c.timestamp, reverse=True)

    if not selected_commits:
        print("Error: No commits found for selected events.", file=sys.stderr)
        return 1

    print(
        f"\n✓ Selected {len(selected_events)} event(s) covering {len(all_commit_hashes)} commit(s)"
    )

    # Show selected event nodes (tree view hides commits until after selection).
    console.print("\n🗂️ Selected events:\n")
    max_events_to_list = 20
    for ev in selected_events[:max_events_to_list]:
        title = getattr(ev, "title", "(untitled)")
        console.print(_render_markdown_lite(f"- {title}"))
        desc = (getattr(ev, "description", "") or "").strip()
        if desc and not desc.startswith("Auto-added"):
            console.print(Padding(_render_markdown_lite(desc), (0, 0, 1, 2)))
    if len(selected_events) > max_events_to_list:
        console.print(f"  ... ({len(selected_events) - max_events_to_list} more)")

    # Show commits for user confirmation (tree view does not show commits).
    selected_commits = sorted(selected_commits, key=lambda c: c.timestamp, reverse=True)
    print("\n📋 Commits included:\n")
    for i, c in enumerate(selected_commits, 1):
        summary = c.llm_summary if c.llm_summary and c.llm_summary != "(No summary)" else c.message
        print(f"  [{i}] {c.hash} - {summary}")
    print()

    confirm = input("Proceed to export these commits? [Y/n]: ").strip().lower()
    if confirm == "n":
        subset = input(
            "Enter commit indices to export (e.g., '1,3,5-7' or 'all'), or blank to cancel: "
        ).strip()
        if not subset:
            print("Cancelled.")
            return 0
        if subset.lower() != "all":
            try:
                keep = parse_commit_indices(subset)
                selected_commits = [
                    selected_commits[i - 1] for i in keep if 0 < i <= len(selected_commits)
                ]
            except Exception as e:
                print(f"Error: Invalid indices format: {e}", file=sys.stderr)
                return 1
        if not selected_commits:
            print("No commits selected. Exiting.")
            return 0

    # Extract messages (same as commit-based export)
    try:
        session_messages = merge_messages_from_commits(selected_commits, shadow_git)
    except Exception as e:
        print(f"\nError: Failed to extract messages: {e}", file=sys.stderr)
        return 1

    if not session_messages:
        print("\nWarning: No chat history found in selected commits.", file=sys.stderr)
        return 1

    # Build conversation data
    username = os.environ.get("USER") or os.environ.get("USERNAME") or "anonymous"
    conversation_data = {
        "username": username,
        "time": datetime.now().isoformat(),
        "sessions": [
            {"session_id": session_id, "messages": messages}
            for session_id, messages in session_messages.items()
        ],
    }

    # Display session preview (Selection Summary removed)
    print()
    display_session_preview(session_messages)

    # Use default preset (no user selection needed)
    preset_id = "default"

    # Generate UI metadata with LLM (enhanced with event context)
    ui_metadata = None
    debug_info = None

    # Add event context to UI metadata generation
    event_context = {
        "event_count": len(selected_events),
        "event_titles": [e.title for e in selected_events],
        "event_types": list(set(e.event_type.value for e in selected_events)),
        "commit_count": len(selected_commits),
    }

    if enable_preview:
        ui_metadata, debug_info = generate_ui_metadata_with_llm(
            conversation_data,
            selected_commits,
            event_title="",  # Will be filled inside
            event_description="",  # Will be filled inside
            provider="auto",
            preset_id=preset_id,
            silent=False,  # Always interactive in this function
        )

        # Enhance UI metadata with event information
        if ui_metadata and selected_events:
            if "welcome" in ui_metadata:
                # Prepend event info to welcome message
                event_summary = f"This share contains {len(selected_events)} event(s) with {len(selected_commits)} commits. "
                ui_metadata["welcome"] = event_summary + ui_metadata["welcome"]

            # Add event metadata to ui_metadata for frontend display
            ui_metadata["event_info"] = event_context

    # Continue with encryption and upload...
    # Generate or use provided password
    if password is None:
        import secrets
        import string

        alphabet = string.ascii_letters + string.digits
        password = "".join(secrets.choice(alphabet) for _ in range(16))
        password_generated = True
    else:
        password_generated = False

    # Encrypt conversation data
    try:
        encrypted_payload = encrypt_conversation_data(conversation_data, password)
    except Exception as e:
        print(f"\n❌ Encryption failed: {e}", file=sys.stderr)
        logger.error(f"Encryption failed: {e}", exc_info=True)
        return 1

    # Prepare metadata
    metadata = {
        "username": username,
        "expiry_days": expiry_days,
        "max_views": max_views,
        "ui_metadata": ui_metadata,
    }

    # Upload to backend
    try:
        response = upload_to_backend(
            encrypted_payload, metadata, backend_url, ui_metadata=ui_metadata
        )
        share_url = response.get("share_url")
        admin_token = response.get("admin_token")

        # Display success message
        display_share_result(
            share_url=share_url,
            password=password,
            expiry_days=expiry_days,
            max_views=max_views,
            admin_token=admin_token,
        )

        return 0

    except Exception as e:
        print(f"\n❌ Upload failed: {e}", file=sys.stderr)
        logger.error(f"Upload failed: {e}", exc_info=True)
        return 1


def export_shares_interactive_command(
    indices: Optional[str] = None,
    password: Optional[str] = None,
    expiry_days: int = 7,
    max_views: int = 100,
    enable_preview: bool = True,
    backend_url: Optional[str] = None,
    repo_root: Optional[Path] = None,
    json_output: bool = False,
    size_report: bool = False,
    compact: bool = False,
    max_tool_result_chars: int = 8_000,
    max_tool_command_chars: int = 2_000,
    dump_payload_dir: Optional[Path] = None,
    force_new_link: bool = False,
) -> int:
    """
    交互式导出对话历史并生成分享链接

    Args:
        indices: Event index to export (single number, e.g. "1")
        password: 加密密码 (如果为 None 则自动生成)
        expiry_days: 过期天数
        max_views: 最大访问次数
        enable_preview: 是否启用UI预览和编辑 (默认: True)
        backend_url: 后端服务器 URL
        repo_root: 项目根目录
        json_output: 是否以 JSON 格式输出结果 (默认: False)

    Returns:
        0 on success, 1 on error
    """
    if not json_output:
        logger.info("======== Interactive export shares command started ========")

    # Check dependencies
    if not CRYPTO_AVAILABLE:
        if not json_output:
            print("Error: cryptography package not installed", file=sys.stderr)
            print("Install it with: pip install cryptography", file=sys.stderr)
        return 1

    if not HTTPX_AVAILABLE:
        if not json_output:
            print("Error: httpx package not installed", file=sys.stderr)
            print("Install it with: pip install httpx", file=sys.stderr)
        return 1

    # Check authentication - require login to create shares
    if not is_logged_in():
        if not json_output:
            print("Error: Not logged in. Please run 'aline login' first.", file=sys.stderr)
        return 1
    if not get_auth_headers():
        if not json_output:
            print("Error: Login expired. Please run 'aline login' again.", file=sys.stderr)
        return 1

    # Get backend URL
    if backend_url is None:
        # Try to load from config
        from ..config import ReAlignConfig

        if repo_root is None:
            repo_root = Path.cwd().resolve()

        config = ReAlignConfig.load()
        backend_url = config.share_backend_url

    if not json_output:
        print("\n=== ReAlign Share Export ===")

    # Step 1: Select events (SQLite-based, no git required)
    if repo_root is None:
        repo_root = Path.cwd().resolve()
    else:
        repo_root = Path(repo_root).resolve()

    from .. import get_realign_dir

    shadow_git = get_realign_dir(repo_root)  # used for log/output placement (not a git repo)

    # Always show all events, don't filter by workspace
    events = get_events_for_export(workspace_path=None, limit=200)

    if not events:
        if not json_output:
            print(
                "No events found in database. Run 'aline watcher start' and/or import sessions first.",
                file=sys.stderr,
            )
        return 1

    if indices is None:
        display_events_for_selection(events)
        try:
            from rich.console import Console

            console = Console()
            console.print("[yellow]Enter event number to export (single event only):[/yellow]")
        except ImportError:
            print("Enter event number to export (single event only):")
        indices = input("Event #: ").strip()
        if not indices:
            if not json_output:
                print("No event selected. Exiting.")
            return 0

    # Parse as UUID or integer
    event_index: Optional[int] = None

    # First check if input looks like a UUID prefix
    if _is_uuid_like(indices):
        uuid_matches = _find_events_by_uuid(indices, events)
        if len(uuid_matches) == 1:
            event_index = uuid_matches[0]
        elif len(uuid_matches) > 1:
            if not json_output:
                print(
                    f"Error: UUID prefix '{indices}' matches multiple events. Please be more specific.",
                    file=sys.stderr,
                )
            return 1
        else:
            if not json_output:
                print(
                    f"Error: No event found matching UUID prefix '{indices}'",
                    file=sys.stderr,
                )
            return 1
    else:
        # Try to parse as integer
        try:
            event_index = int(indices)
        except ValueError:
            if not json_output:
                print(
                    f"Error: Please enter a number (1-{len(events)}) or UUID prefix (e.g., 'ea48983b')",
                    file=sys.stderr,
                )
            return 1

    # Validate index
    if event_index < 1 or event_index > len(events):
        if not json_output:
            print(
                f"Error: Invalid event number. Please enter 1-{len(events)}",
                file=sys.stderr,
            )
        return 1

    # Get single selected event
    selected_events = [e for e in events if e.index == event_index]
    if not selected_events:
        if not json_output:
            print("No event selected. Exiting.")
        return 0

    selected_session_records_by_id: Dict[str, SessionRecord] = {}
    for event in selected_events:
        for session in event.sessions:
            selected_session_records_by_id[session.id] = session

    min_ts = datetime.min.replace(tzinfo=timezone.utc)
    selected_session_records = sorted(
        selected_session_records_by_id.values(),
        key=lambda s: (
            (
                s.last_activity_at.replace(tzinfo=timezone.utc)
                if s.last_activity_at and s.last_activity_at.tzinfo is None
                else s.last_activity_at.astimezone(timezone.utc)
            )
            if s.last_activity_at
            else min_ts
        ),
        reverse=True,
    )
    selected_sessions = build_exportable_sessions_from_records(selected_session_records)
    if not selected_sessions:
        if not json_output:
            print("No sessions found for selected events. Exiting.")
        return 0

    # Keep variable for downstream UI metadata logic (falls back to message samples).
    selected_commits: List[UnpushedCommit] = []

    # Step 2: Build enhanced conversation data (v2.0 format with Event/Session/Turn structure)
    from ..db import get_database

    db = get_database()
    username = os.environ.get("USER") or os.environ.get("USERNAME") or "anonymous"
    selected_event = selected_events[0]

    # Reuse existing share link for the same event_id (avoid duplicate uploads).
    # If a prior share exists and we have its admin token, extend expiry and return early.
    try:
        full_event = db.get_event_by_id(selected_event.event_id)
    except Exception:
        full_event = None

    reuse_share_url: Optional[str] = None
    reuse_share_id: Optional[str] = None
    reuse_admin_token: Optional[str] = None

    if (
        not force_new_link
        and full_event
        and isinstance(getattr(full_event, "share_url", None), str)
        and full_event.share_url
    ):
        reuse_share_url = full_event.share_url
        reuse_share_id = getattr(full_event, "share_id", None) or _extract_share_id_from_url(
            reuse_share_url
        )
        reuse_admin_token = getattr(full_event, "share_admin_token", None)

    # Fast path: if we're not debugging, don't rebuild payload—just extend expiry.
    if (
        reuse_share_url
        and reuse_share_id
        and reuse_admin_token
        and not size_report
        and dump_payload_dir is None
    ):
        try:
            new_expiry_at = _extend_share_expiry(
                backend_url=backend_url,
                share_id=reuse_share_id,
                admin_token=reuse_admin_token,
                expiry_days=expiry_days,
            )
            try:
                db.update_event_share_metadata(
                    event_id=selected_event.event_id,
                    share_url=reuse_share_url,
                    share_id=reuse_share_id,
                    share_admin_token=reuse_admin_token,
                    share_expiry_at=new_expiry_at,
                )
            except Exception:
                pass

            if json_output:
                print(
                    json.dumps(
                        {
                            "event_id": selected_event.event_id,
                            "share_link": reuse_share_url,
                            "reused": True,
                            "expiry_at": new_expiry_at.isoformat() if new_expiry_at else None,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print("\n🔁 Reusing existing share link for this event (expiry updated).")
                display_share_result(
                    share_url=reuse_share_url,
                    password=None,
                    expiry_days=expiry_days,
                    max_views=max_views,
                    admin_token=reuse_admin_token,
                )
            return 0
        except Exception as e:
            if not json_output:
                print(f"\n⚠️  Failed to extend existing share expiry, creating a new link: {e}")

    conversation_data_full_for_debug: Optional[dict] = None
    conversation_data_compact_for_debug: Optional[dict] = None

    if size_report and not json_output:
        report = _build_export_size_report(selected_sessions, db, top_n=10)
        mb = report["total_bytes"] / 1024 / 1024 if report["total_bytes"] else 0
        print("\n=== Share Export Size Report (raw JSONL) ===")
        print(f"Total JSONL lines: {report['total_lines']}")
        print(f"Total JSONL bytes: {mb:.2f}MB")

        # Compare payload sizes before/after compaction (pre-encryption)
        try:
            conversation_data_full = build_enhanced_conversation_data(
                selected_event=selected_events[0],
                selected_sessions=selected_sessions,
                username=username,
                db=db,
                compaction=None,
            )
            conversation_data_full_for_debug = conversation_data_full
            full_bytes = len(json.dumps(conversation_data_full, ensure_ascii=False).encode("utf-8"))

            conversation_data_compact = build_enhanced_conversation_data(
                selected_event=selected_events[0],
                selected_sessions=selected_sessions,
                username=username,
                db=db,
                compaction=ExportCompactionConfig(
                    enabled=True,
                    max_tool_result_chars=max_tool_result_chars,
                    max_tool_command_chars=max_tool_command_chars,
                ),
            )
            conversation_data_compact_for_debug = conversation_data_compact
            compact_bytes = len(
                json.dumps(conversation_data_compact, ensure_ascii=False).encode("utf-8")
            )

            full_mb = full_bytes / 1024 / 1024
            compact_mb = compact_bytes / 1024 / 1024
            reduction = (1 - (compact_bytes / full_bytes)) * 100 if full_bytes else 0.0
            print("\n=== Export Payload Size (pre-encryption) ===")
            print(f"No compact: {full_mb:.2f}MB")
            print(f"Compact:    {compact_mb:.2f}MB ({reduction:.1f}% smaller)")

            print("\n=== Largest Fields In Compact Payload (chars) ===")
            for item in _top_string_fields(conversation_data_compact, top_n=10):
                print(f"- {item['chars']:>6}  {item['path']}")
        except Exception as e:
            logger.debug(f"Failed to compute compact size comparison: {e}")

        if report["top_lines"]:
            print("\nLargest lines (no content):")
            for item in report["top_lines"]:
                size_kb = item["size_bytes"] / 1024
                blocks = (
                    ",".join(item["content_block_types"]) if item["content_block_types"] else "-"
                )
                tools = item.get("tools") or []
                tool_hint = ""
                if tools:
                    parts = []
                    for t in tools:
                        name = t.get("tool_name") or "tool"
                        snippet = t.get("command_snippet")
                        if snippet:
                            parts.append(f"{name}:{snippet}")
                        else:
                            path_hint = t.get("path_hint")
                            keys = t.get("input_keys") or []
                            if path_hint:
                                parts.append(f"{name} path={path_hint}")
                            elif keys:
                                parts.append(f"{name} keys={','.join(keys[:8])}")
                            else:
                                parts.append(f"{name}:{t.get('tool_use_id')}")
                    tool_hint = " | " + " ; ".join(parts)
                print(
                    f"- {size_kb:8.1f}KB | session={item['session_id']} turn={item['turn_number']} "
                    f"type={item['record_type']} role={item['role'] or '-'} blocks={blocks}{tool_hint}"
                )
        print()

    try:
        compaction_cfg = (
            ExportCompactionConfig(
                enabled=True,
                max_tool_result_chars=max_tool_result_chars,
                max_tool_command_chars=max_tool_command_chars,
            )
            if compact
            else None
        )

        if dump_payload_dir and not json_output:
            # Save both full and compact payloads for dev inspection.
            dump_payload_dir = Path(dump_payload_dir).expanduser()
            dump_payload_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%dT%H%M%S")
            event_id = selected_events[0].event_id

            if conversation_data_full_for_debug is None:
                conversation_data_full_for_debug = build_enhanced_conversation_data(
                    selected_event=selected_events[0],
                    selected_sessions=selected_sessions,
                    username=username,
                    db=db,
                    compaction=None,
                )
            if conversation_data_compact_for_debug is None:
                conversation_data_compact_for_debug = build_enhanced_conversation_data(
                    selected_event=selected_events[0],
                    selected_sessions=selected_sessions,
                    username=username,
                    db=db,
                    compaction=ExportCompactionConfig(
                        enabled=True,
                        max_tool_result_chars=max_tool_result_chars,
                        max_tool_command_chars=max_tool_command_chars,
                    ),
                )

            full_path = dump_payload_dir / f"share_export_{event_id}_{ts}_full.json"
            compact_path = dump_payload_dir / f"share_export_{event_id}_{ts}_compact.json"

            full_path.write_text(
                json.dumps(conversation_data_full_for_debug, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            compact_path.write_text(
                json.dumps(conversation_data_compact_for_debug, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"🧪 Saved payloads:\n- {full_path}\n- {compact_path}")

        # Debug path: even if we generate size reports/dumps, still avoid duplicate uploads.
        if reuse_share_url and reuse_share_id and reuse_admin_token and not force_new_link:
            new_expiry_at = _extend_share_expiry(
                backend_url=backend_url,
                share_id=reuse_share_id,
                admin_token=reuse_admin_token,
                expiry_days=expiry_days,
            )
            try:
                db.update_event_share_metadata(
                    event_id=selected_event.event_id,
                    share_url=reuse_share_url,
                    share_id=reuse_share_id,
                    share_admin_token=reuse_admin_token,
                    share_expiry_at=new_expiry_at,
                )
            except Exception:
                pass

            if json_output:
                print(
                    json.dumps(
                        {
                            "event_id": selected_event.event_id,
                            "share_link": reuse_share_url,
                            "reused": True,
                            "expiry_at": new_expiry_at.isoformat() if new_expiry_at else None,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print("\n🔁 Reusing existing share link for this event (expiry updated).")
                display_share_result(
                    share_url=reuse_share_url,
                    password=None,
                    expiry_days=expiry_days,
                    max_views=max_views,
                    admin_token=reuse_admin_token,
                )
            return 0

        conversation_data = build_enhanced_conversation_data(
            selected_event=selected_events[0],
            selected_sessions=selected_sessions,
            username=username,
            db=db,
            compaction=compaction_cfg,
        )
        if not json_output:
            logger.info(
                f"Built enhanced conversation data v{conversation_data.get('version', '2.0')}"
            )
    except Exception as e:
        if not json_output:
            print(f"\nError: Failed to build conversation data: {e}", file=sys.stderr)
        logger.error(f"Failed to build conversation data: {e}", exc_info=True)
        return 1

    # Validate that we have data to export
    if not conversation_data.get("sessions"):
        if not json_output:
            print("\nWarning: No sessions found in conversation data.", file=sys.stderr)
        return 1

    # Build session_messages for display_session_preview (backward compatibility)
    # This is used only for preview display, not for export
    try:
        session_messages = merge_messages_from_sessions(selected_sessions)
    except Exception as e:
        logger.warning(f"Failed to build session preview: {e}")
        # Create fallback preview from v2.0 data
        session_messages = {
            s["session_id"]: [msg for turn in s["turns"] for msg in turn["messages"]]
            for s in conversation_data["sessions"]
        }

    # Display session preview (Selection Summary removed)
    if not json_output:
        print()
        display_session_preview(session_messages)

    # Step 3: Use default preset (no user selection needed)
    preset_id = "default"  # Always use default preset

    # Step 4: Generate UI metadata with LLM
    # Get event title and description (we now only select one event)
    selected_event = selected_events[0]
    event_title = selected_event.title or "Untitled Event"
    event_description = selected_event.description or ""

    from ..config import ReAlignConfig

    config = ReAlignConfig.load()
    ui_metadata, llm_debug_info = generate_ui_metadata_with_llm(
        conversation_data,
        selected_commits,
        event_title=event_title,
        event_description=event_description,
        provider=config.llm_provider,
        preset_id=preset_id,
        silent=json_output,
    )

    # Display and add UI metadata to conversation data
    if ui_metadata:
        if not json_output:
            display_ui_metadata_preview(ui_metadata)
        conversation_data["ui_metadata"] = ui_metadata
    else:
        # Use default UI metadata if LLM generation failed
        if not json_output:
            try:
                from rich.console import Console

                console = Console()
                console.print("[yellow]⚠️  Using default UI content[/yellow]")
            except ImportError:
                print("⚠️  Using default UI content")
        conversation_data["ui_metadata"] = {}

    # Step 4: Ask if user wants password protection
    use_password = True  # Default
    if password is None:
        if json_output:
            # In JSON mode, default to no password if not provided
            use_password = False
        else:
            try:
                from rich.prompt import Confirm

                print()
                use_password = Confirm.ask(
                    "[yellow]🔐 Would you like to protect this share with a password?[/yellow]",
                    default=False,
                )
            except ImportError:
                # Fallback: ask with plain input
                print(
                    "\n🔐 Would you like to protect this share with a password? (y/N): ",
                    end="",
                )
                response = input().strip().lower()
                use_password = response == "y" or response == "yes"

    # Step 5: Generate password or skip encryption
    encrypted_payload = None
    if use_password:
        if password is None:
            # Generate a random password
            password = secrets.token_urlsafe(16)
            if not json_output:
                print(f"\nPassword: {password}")

        # Encrypt data
        try:
            encrypted_payload = encrypt_conversation_data(conversation_data, password)
        except Exception as e:
            if not json_output:
                print(f"\nError: Failed to encrypt data: {e}", file=sys.stderr)
            logger.error(f"Encryption failed: {e}", exc_info=True)
            return 1

    else:
        password = None

    # Step 6: Upload to backend

    metadata = {
        "username": username,
        "expiry_days": expiry_days,
        "max_views": max_views,
    }

    try:
        # Extract ui_metadata for Open Graph tags
        # This allows link previews to work even for encrypted shares
        ui_metadata_for_og = conversation_data.get("ui_metadata")

        # Upload to backend (with automatic chunked upload for large payloads)
        # Use background=True for faster UX - share URL is returned immediately
        if encrypted_payload:
            # Use upload_to_backend which handles chunked uploads automatically
            result = upload_to_backend(
                encrypted_payload=encrypted_payload,
                metadata=metadata,
                backend_url=backend_url,
                ui_metadata=ui_metadata_for_og,
                background=True,  # Return share URL immediately, upload in background
            )
        else:
            # No encryption - use upload_to_backend_unencrypted for chunked support
            result = upload_to_backend_unencrypted(
                conversation_data=conversation_data,
                metadata=metadata,
                backend_url=backend_url,
                background=True,  # Return share URL immediately, upload in background
            )
    except Exception as e:
        if not json_output:
            print(f"\n❌ Upload failed: {e}", file=sys.stderr)
        logger.error(f"Upload failed: {e}", exc_info=True)
        return 1

    # Check if upload is pending (background mode for chunked uploads)
    upload_pending = result.get("upload_pending", False)
    if upload_pending and not json_output:
        print("\n🔗 Share link ready! Uploading data in background...")
        print("   You can share this link now - recipients will see the content")
        print("   once the upload completes (usually within a few seconds).")

    # Save generated metadata and share URL to database
    try:
        from ..db import get_database

        db = get_database()
        db.update_event_share_metadata(
            event_id=selected_event.event_id,
            slack_message=ui_metadata.get("slack_message") if ui_metadata else None,
            share_url=result.get("share_url"),
            share_id=result.get("share_id")
            or _extract_share_id_from_url(result.get("share_url") or ""),
            share_admin_token=result.get("admin_token"),
            share_expiry_at=(
                datetime.fromisoformat(result["expiry_at"].replace("Z", "+00:00"))
                if isinstance(result.get("expiry_at"), str)
                else None
            ),
        )
        logger.info(f"Saved share metadata and URL to event {selected_event.event_id}")
    except Exception as e:
        logger.warning(f"Failed to save share metadata to database: {e}")
        # Don't fail the export if database save fails

    # Step 7: Output results
    if json_output:
        # Output strictly JSON
        output_data = {
            "event_id": selected_event.event_id,
            "share_link": result.get("share_url"),
            "event_title": event_title,
            "event_summary": event_description,
            "slack_message": ui_metadata.get("slack_message") if ui_metadata else None,
            "password": password,
            "upload_pending": upload_pending,
        }
        print(json.dumps(output_data, ensure_ascii=False, indent=2))
    else:
        # Display success with beautiful formatting
        display_share_result(
            share_url=result["share_url"],
            password=password,
            expiry_days=expiry_days,
            max_views=max_views,
            admin_token=result.get("admin_token"),
        )
        copied = _copy_share_to_clipboard(
            result.get("share_url"),
            ui_metadata.get("slack_message") if ui_metadata else None,
        )
        if copied:
            print("📋 Copied Slack message and share link to clipboard.")

    if not json_output:
        logger.info(f"======== Interactive export completed: {result['share_url']} ========")
    return 0


def export_agent_shares_command(
    agent_id: str,
    password: Optional[str] = None,
    expiry_days: int = 7,
    max_views: int = 100,
    backend_url: Optional[str] = None,
    json_output: bool = False,
    compact: bool = True,
    max_tool_result_chars: int = 8_000,
    max_tool_command_chars: int = 2_000,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> int:
    """
    Export all sessions associated with an agent and generate a share link.

    This function creates a synthetic event structure from agent sessions,
    generates UI metadata (Slack message), uploads to backend, and returns
    the share link.

    Args:
        agent_id: The agent_info ID to export sessions for
        password: Encryption password (if None, no encryption)
        expiry_days: Share expiry in days
        max_views: Maximum number of views
        backend_url: Backend server URL (uses config default if None)
        json_output: If True, output JSON format
        compact: Whether to compact the export data
        max_tool_result_chars: Max chars for tool results (with compact)
        max_tool_command_chars: Max chars for tool commands (with compact)
        progress_callback: Optional callback for progress updates (message: str) -> None

    Returns:
        0 on success, 1 on error
    """
    def _default_agent_slack_message(*, agent_name: str, synced: bool) -> str:
        verb = "Synced" if synced else "Shared"
        return f"{verb} Aline agent “{agent_name}”.\nTake a look when you have a moment."

    def _progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    if not json_output:
        logger.info(f"======== Export agent shares command started for agent {agent_id} ========")

    # Check dependencies
    if not CRYPTO_AVAILABLE:
        if not json_output:
            print("Error: cryptography package not installed", file=sys.stderr)
        return 1

    if not HTTPX_AVAILABLE:
        if not json_output:
            print("Error: httpx package not installed", file=sys.stderr)
        return 1

    # Check authentication
    if not is_logged_in():
        if not json_output:
            print("Error: Not logged in. Please run 'aline login' first.", file=sys.stderr)
        return 1
    # is_logged_in() can be true with an expired token + refresh_token; ensure we actually have an access token
    # before attempting uploads (otherwise the server returns 401/403 and the UX is confusing).
    if not get_auth_headers():
        if not json_output:
            print("Error: Login expired. Please run 'aline login' again.", file=sys.stderr)
        return 1

    _progress("Fetching agent info...")

    # Get backend URL
    if backend_url is None:
        from ..config import ReAlignConfig

        config = ReAlignConfig.load()
        backend_url = config.share_backend_url

    # Get database
    from ..db import get_database

    db = get_database()

    # Get agent info
    agent_info = db.get_agent_info(agent_id)
    if not agent_info:
        if not json_output:
            print(f"Error: Agent not found: {agent_id}", file=sys.stderr)
        return 1

    # Check for existing share (re-share → sync instead of new link)
    if (
        not password
        and agent_info.share_url
        and agent_info.share_id
        and (agent_info.share_admin_token or agent_info.share_contributor_token)
    ):
        _progress("Agent already shared, syncing...")
        try:
            from .sync_agent import sync_agent_command

            sync_result = sync_agent_command(
                agent_id=agent_id,
                backend_url=backend_url,
                progress_callback=progress_callback,
            )
            if sync_result.get("success"):
                # Extend expiry if we have admin token
                if agent_info.share_admin_token:
                    try:
                        _extend_share_expiry(
                            backend_url=backend_url,
                            share_id=agent_info.share_id,
                            admin_token=agent_info.share_admin_token,
                            expiry_days=expiry_days,
                        )
                    except Exception as ext_err:
                        logger.warning(f"Failed to extend share expiry: {ext_err}")

                if json_output:
                    try:
                        refreshed = db.get_agent_info(agent_id)
                    except Exception:
                        refreshed = None
                    stored_msg = getattr(refreshed, "slack_message", None) if refreshed else None
                    slack_message = (
                        stored_msg
                        if isinstance(stored_msg, str) and stored_msg.strip()
                        else _default_agent_slack_message(agent_name=agent_info.name, synced=True)
                    )
                    output_data = {
                        "agent_id": agent_id,
                        "agent_name": agent_info.name,
                        "share_link": agent_info.share_url,
                        "slack_message": slack_message,
                        "synced": True,
                        "sessions_pulled": sync_result.get("sessions_pulled", 0),
                        "sessions_pushed": sync_result.get("sessions_pushed", 0),
                    }
                    print(json.dumps(output_data, ensure_ascii=False, indent=2))
                else:
                    pulled = sync_result.get("sessions_pulled", 0)
                    pushed = sync_result.get("sessions_pushed", 0)
                    print(f"\n🔄 Synced agent: {agent_info.name}")
                    print(f"   Pulled {pulled} session(s), pushed {pushed} session(s)")
                    print(f"🔗 Share link: {agent_info.share_url}")
                    copied = _copy_share_to_clipboard(agent_info.share_url, None)
                    if copied:
                        print("📋 Copied share link to clipboard.")
                return 0
            else:
                err = sync_result.get("error", "Unknown sync error")
                logger.warning(f"Sync failed, falling through to new share: {err}")
                if not json_output:
                    print(f"⚠️  Sync failed ({err}), creating new share link...", file=sys.stderr)
        except ImportError:
            logger.warning("sync_agent module not available, creating new share")
        except Exception as e:
            logger.warning(f"Sync failed, falling through to new share: {e}")
            if not json_output:
                print(f"⚠️  Sync failed ({e}), creating new share link...", file=sys.stderr)

    # Get sessions for this agent
    session_records = db.get_sessions_by_agent_id(agent_id)
    if not session_records:
        if not json_output:
            print(f"Error: Agent has no sessions to share", file=sys.stderr)
        return 1

    _progress(f"Found {len(session_records)} session(s)")

    # Build exportable sessions
    selected_sessions = build_exportable_sessions_from_records(session_records)
    if not selected_sessions:
        if not json_output:
            print("Error: No sessions found for agent", file=sys.stderr)
        return 1

    # Create a synthetic event structure for the agent
    # We use the agent name/description as event title/description
    event_title = agent_info.title or agent_info.name or "Agent Sessions"
    event_description = agent_info.description or f"Sessions from agent: {agent_info.name}"

    # Create a synthetic ExportableEvent
    synthetic_event = ExportableEvent(
        index=1,
        event_id=f"agent-{agent_id}",
        title=event_title,
        description=event_description,
        event_type="agent",
        status="active",
        updated_at=datetime.now(timezone.utc),
        sessions=session_records,
    )

    # Build conversation data
    _progress("Building conversation data...")

    username = os.environ.get("USER") or os.environ.get("USERNAME") or "anonymous"

    compaction = None
    if compact:
        compaction = ExportCompactionConfig(
            enabled=True,
            max_tool_result_chars=max_tool_result_chars,
            max_tool_command_chars=max_tool_command_chars,
        )

    try:
        conversation_data = build_agent_share_conversation_data_metadata_only(
            agent_id=agent_id,
            agent_name=agent_info.name,
            agent_title=agent_info.title or "",
            agent_description=agent_info.description or "",
            selected_sessions=selected_sessions,
            username=username,
        )
    except Exception as e:
        if not json_output:
            print(f"Error: Failed to build conversation data: {e}", file=sys.stderr)
        logger.error(f"Failed to build conversation data: {e}", exc_info=True)
        return 1

    if not conversation_data.get("sessions"):
        if not json_output:
            print("Error: No sessions found in conversation data", file=sys.stderr)
        return 1

    # Generate UI metadata with LLM
    _progress("Generating share message...")

    from ..config import ReAlignConfig

    config = ReAlignConfig.load()
    ui_metadata, _ = generate_ui_metadata_with_llm(
        conversation_data,
        [],  # No commits
        event_title=event_title,
        event_description=event_description,
        provider=config.llm_provider,
        preset_id="default",
        silent=json_output,
    )

    if ui_metadata:
        conversation_data["ui_metadata"] = ui_metadata
    else:
        conversation_data["ui_metadata"] = {
            "title": event_title,
            "description": event_description,
        }

    # Add agent name and context title to ui_metadata for display
    conversation_data["ui_metadata"]["agent_name"] = agent_info.name
    conversation_data["ui_metadata"]["share_kind"] = "agent"
    if agent_info.title:
        conversation_data["ui_metadata"]["context_title"] = agent_info.title

    # Include sync_metadata placeholder (contributor_token will be added after upload for unencrypted shares)
    if not password:
        conversation_data["sync_metadata"] = {
            "contributor_token": None,  # Will be populated after upload
            "sync_version": 0,
        }

    # Upload to backend (no encryption for agent shares by default)
    _progress("Uploading to cloud...")

    metadata = {
        "username": username,
        "expiry_days": expiry_days,
        "max_views": max_views,
    }

    try:
        if password:
            encrypted_payload = encrypt_conversation_data(conversation_data, password)
            result = upload_to_backend(
                encrypted_payload=encrypted_payload,
                metadata=metadata,
                backend_url=backend_url,
                ui_metadata=conversation_data.get("ui_metadata"),
                background=False,
            )
        else:
            def upload_progress(current: int, total: int, message: str) -> None:
                _progress(f"{message} ({current}/{total})")

            result = upload_to_backend_unencrypted(
                conversation_data=conversation_data,
                metadata=metadata,
                backend_url=backend_url,
                progress_callback=upload_progress,
                background=False,
            )
    except Exception as e:
        if not json_output:
            print(f"Error: Upload failed: {e}", file=sys.stderr)
        logger.error(f"Upload failed: {e}", exc_info=True)
        return 1

    share_url = result.get("share_url")
    slack_message = ui_metadata.get("slack_message") if ui_metadata else None
    if not (isinstance(slack_message, str) and slack_message.strip()):
        slack_message = _default_agent_slack_message(
            agent_name=agent_info.name,
            synced=False,
        )

    # Store sync metadata for unencrypted shares
    if not password and share_url:
        share_id_result = result.get("share_id") or _extract_share_id_from_url(share_url)
        admin_token = result.get("admin_token")
        contributor_token = result.get("contributor_token")
        expiry_at = result.get("expiry_at")

        if share_id_result:
            try:
                db.update_agent_sync_metadata(
                    agent_id,
                    share_id=share_id_result,
                    share_url=share_url,
                    share_admin_token=admin_token,
                    share_contributor_token=contributor_token,
                    share_expiry_at=expiry_at,
                    slack_message=slack_message,
                    last_synced_at=datetime.now(timezone.utc).isoformat(),
                    sync_version=0,
                )
            except Exception as e:
                logger.warning(f"Failed to store sync metadata: {e}")

    # Output results
    if json_output:
        output_data = {
            "agent_id": agent_id,
            "agent_name": agent_info.name,
            "share_link": share_url,
            "slack_message": slack_message,
            "session_count": len(selected_sessions),
            "password": password,
        }
        print(json.dumps(output_data, ensure_ascii=False, indent=2))
    else:
        print(f"\n✅ Shared {len(selected_sessions)} session(s) from agent: {agent_info.name}")
        print(f"🔗 Share link: {share_url}")
        if slack_message:
            print(f"\n📝 Slack message:\n{slack_message}")
        copied = _copy_share_to_clipboard(share_url, slack_message)
        if copied:
            print("📋 Copied Slack message and share link to clipboard.")

    if not json_output:
        logger.info(f"======== Agent export completed: {share_url} ========")
    return 0
