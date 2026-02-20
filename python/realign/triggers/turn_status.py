"""
Turn end-status detection utilities.

These helpers classify how a turn ended based on artifacts in the session JSONL:
- user interrupt markers (e.g., "[Request interrupted by user for tool use]")
- rate limit / API error synthetic assistant messages
- automatic conversation compaction boundaries
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class TurnEndStatus:
    """
    Classification for a turn's end state.

    `status` values:
    - "completed"
    - "user_interrupted"
    - "rate_limited"
    - "compacted"
    - "unknown"
    """

    status: str
    line: Optional[int] = None
    message: Optional[str] = None


def preview_text(text: Optional[str], max_chars: int = 150) -> str:
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _iter_text_blobs(value: Any) -> list[str]:
    """
    Extract human-readable text blobs from common Claude Code JSONL shapes.
    """
    blobs: list[str] = []

    if value is None:
        return blobs

    if isinstance(value, str):
        if value.strip():
            blobs.append(value)
        return blobs

    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                blobs.append(item)
            elif isinstance(item, dict):
                t = item.get("type")
                if t == "text":
                    txt = item.get("text")
                    if isinstance(txt, str) and txt.strip():
                        blobs.append(txt)
                elif t == "tool_result":
                    content = item.get("content")
                    blobs.extend(_iter_text_blobs(content))
        return blobs

    if isinstance(value, dict):
        # Common: {"role":"user","content":...}
        if "content" in value:
            blobs.extend(_iter_text_blobs(value.get("content")))
        if "text" in value and isinstance(value.get("text"), str):
            txt = value.get("text")
            if txt and txt.strip():
                blobs.append(txt)
        if "toolUseResult" in value:
            blobs.extend(_iter_text_blobs(value.get("toolUseResult")))
        return blobs

    return blobs


def detect_turn_end_status(session_file: Path, start_line: int, end_line: int) -> TurnEndStatus:
    """
    Detect how the turn ended by scanning lines in [start_line, end_line].

    Priority (highest first):
    1) user_interrupted
    2) rate_limited
    3) compacted
    4) completed
    """
    if start_line <= 0 or end_line <= 0 or end_line < start_line:
        return TurnEndStatus(status="unknown")

    # Track best match by priority then later line.
    best: TurnEndStatus = TurnEndStatus(status="completed")
    best_priority = 0

    def consider(status: str, line: int, message: Optional[str], priority: int) -> None:
        nonlocal best, best_priority
        if priority > best_priority or (priority == best_priority and (best.line or 0) < line):
            best = TurnEndStatus(status=status, line=line, message=message)
            best_priority = priority

    try:
        with session_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line_no < start_line:
                    continue
                if line_no > end_line:
                    break

                raw = line.strip()
                if not raw:
                    continue

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")

                # 1) Conversation compaction boundary
                if msg_type == "system":
                    subtype = data.get("subtype") or ""
                    content = (data.get("content") or "").strip()
                    if subtype == "compact_boundary" or "compact" in str(subtype):
                        consider(
                            "compacted", line_no, content or "Conversation compacted", priority=1
                        )
                        continue

                # 2) User interrupt markers (user stopped the agent)
                if msg_type == "user":
                    msg = data.get("message", {})
                    content = msg.get("content")
                    blobs = _iter_text_blobs(content)
                    # Some sessions store toolUseResult at top-level string
                    blobs.extend(_iter_text_blobs(data.get("toolUseResult")))

                    joined = "\n".join(blobs)
                    if "[Request interrupted by user" in joined:
                        consider("user_interrupted", line_no, joined, priority=3)
                        continue

                    if "The user doesn't want to proceed with this tool use" in joined:
                        consider("user_interrupted", line_no, joined, priority=3)
                        continue

                # 3) Rate limit / API error synthetic assistant
                if msg_type == "assistant":
                    if data.get("isApiErrorMessage") or data.get("error") == "rate_limit":
                        blobs = _iter_text_blobs(data.get("message", {}).get("content"))
                        msg = "\n".join(blobs) if blobs else "rate_limit"
                        consider("rate_limited", line_no, msg, priority=2)
                        continue

                    # Also detect via text content if present
                    blobs = _iter_text_blobs(data.get("message", {}).get("content"))
                    joined = "\n".join(blobs)
                    if "You've hit your limit" in joined:
                        consider("rate_limited", line_no, joined, priority=2)
                        continue

        return best

    except Exception:
        return TurnEndStatus(status="unknown")
