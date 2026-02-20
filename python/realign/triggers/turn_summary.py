"""
Turn summary extraction utilities.

We intentionally do NOT use Claude Code's file-level `type="summary"` records here.
Instead, we derive a per-turn recap by scanning the assistant messages between
the user turn start and the next user message.
"""

import json
from pathlib import Path
from typing import Optional, Tuple


def extract_turn_summary(
    session_file: Path,
    start_line: int,
    end_line: int,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Extract a per-turn “summary” by looking at the last assistant message within
    the line range (start_line, end_line], preferring text blocks, and falling
    back to thinking blocks if no text exists.

    Returns:
        (summary_line, summary_text)
    """
    if start_line <= 0 or end_line <= 0 or end_line < start_line:
        return None, None

    last_text: Optional[str] = None
    last_text_line: Optional[int] = None
    last_thinking: Optional[str] = None
    last_thinking_line: Optional[int] = None

    try:
        with session_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line_no <= start_line:
                    continue
                if line_no > end_line:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text_parts = []
                thinking_parts = []

                # Legacy format (Claude Code sessions)
                if data.get("type") == "assistant":
                    # Ignore synthetic/system error assistant messages (rate limit, API error retries, etc.)
                    if data.get("isApiErrorMessage") or data.get("error"):
                        continue

                    msg_model = (data.get("message", {}) or {}).get("model")
                    if msg_model == "<synthetic>":
                        continue

                    content = data.get("message", {}).get("content", [])
                    if not isinstance(content, list):
                        continue

                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "text":
                            txt = (item.get("text") or "").strip()
                            if txt:
                                text_parts.append(txt)
                        elif item.get("type") == "thinking":
                            th = (item.get("thinking") or "").strip()
                            if th:
                                thinking_parts.append(th)

                # Codex CLI sessions (OpenAI Responses JSONL)
                elif data.get("type") == "response_item":
                    payload = data.get("payload") or {}
                    if not isinstance(payload, dict):
                        continue

                    if payload.get("type") != "message" or payload.get("role") != "assistant":
                        continue

                    content = payload.get("content", [])
                    if not isinstance(content, list):
                        continue

                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") in ("output_text", "text"):
                            txt = (item.get("text") or "").strip()
                            if txt:
                                text_parts.append(txt)
                        elif item.get("type") == "thinking":
                            th = (item.get("thinking") or "").strip()
                            if th:
                                thinking_parts.append(th)
                else:
                    continue

                if text_parts:
                    last_text = "\n".join(text_parts).strip()
                    last_text_line = line_no
                elif thinking_parts:
                    last_thinking = "\n".join(thinking_parts).strip()
                    last_thinking_line = line_no

        if last_text:
            return last_text_line, last_text
        if last_thinking:
            return last_thinking_line, last_thinking
        return None, None

    except Exception:
        return None, None
