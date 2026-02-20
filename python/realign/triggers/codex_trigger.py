"""
Codex trigger - turn detection for Codex/Codex CLI JSONL sessions.

Codex sessions are event streams (session_meta / response_item / event_msg / turn_context),
which differ significantly from Claude Code's {type:"user"/"assistant", message:{...}} JSONL.

This trigger detects completed turns as:
  (user request) -> (assistant final message)

It also exposes the same enriched fields as the latest Claude triggers so the watcher
can generate commit summaries via trigger-derived turn context:
- user_message
- summary_message (+ summary_line)
- turn_status
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import TurnInfo, TurnTrigger


def _preview(text: Optional[str], max_chars: int = 150) -> str:
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().split())


def _extract_text_from_codex_content(content: Any) -> str:
    """
    Extract human-readable text from Codex response_item.message content blocks.

    Known shapes:
    - [{"type":"input_text","text":"..."}]
    - [{"type":"output_text","text":"..."}]
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t in ("input_text", "output_text", "text"):
                txt = item.get("text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt)
        return "\n".join(parts).strip()

    if isinstance(content, dict):
        txt = content.get("text")
        if isinstance(txt, str):
            return txt.strip()
        return ""

    return ""


def _clean_codex_user_request(text: str) -> Optional[str]:
    """
    Convert Codex "IDE context wrapper" prompts into the actual user request.

    Typical wrapper:
      "# Context from my IDE setup: ...\n\n## My request for Codex:\n<REQUEST>"
    """
    raw = (text or "").strip()
    if not raw:
        return None

    # Pure environment context stanzas are not user requests.
    if raw.startswith("<environment_context>") or raw.startswith("<environment_context"):
        return None

    marker = "## My request for Codex:"
    if marker in raw:
        req = raw.split(marker, 1)[1].strip()
        return req or None

    # Sometimes the user prompt is already a plain instruction.
    return raw


def _extract_codex_user_candidate(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (timestamp, cleaned_user_request) if this line represents a user request.
    """
    msg_type = data.get("type")

    # Preferred: event_msg user_message
    if msg_type == "event_msg":
        payload = data.get("payload") or {}
        if payload.get("type") == "user_message":
            ts = data.get("timestamp")
            msg = payload.get("message") or ""
            return ts, _clean_codex_user_request(str(msg))

    # Fallback: response_item message role=user
    if msg_type == "response_item":
        payload = data.get("payload") or {}
        if payload.get("type") == "message" and payload.get("role") == "user":
            ts = data.get("timestamp")
            content = payload.get("content")
            text = _extract_text_from_codex_content(content)
            return ts, _clean_codex_user_request(text)

    # Direct message format (newer Codex CLI): type="message" with role="user"
    if msg_type == "message" and data.get("role") == "user":
        ts = data.get("timestamp")
        content = data.get("content")
        text = _extract_text_from_codex_content(content)
        return ts, _clean_codex_user_request(text)

    return None, None


def _extract_codex_assistant_candidate(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (timestamp, assistant_text) if this line represents an assistant final message.
    """
    msg_type = data.get("type")

    # Newer: event_msg agent_message
    if msg_type == "event_msg":
        payload = data.get("payload") or {}
        if payload.get("type") == "agent_message":
            ts = data.get("timestamp")
            msg = payload.get("message") or ""
            msg = str(msg).strip()
            return ts, (msg or None)

    # Older: response_item message role=assistant
    if msg_type == "response_item":
        payload = data.get("payload") or {}
        if payload.get("type") == "message" and payload.get("role") == "assistant":
            ts = data.get("timestamp")
            content = payload.get("content")
            text = _extract_text_from_codex_content(content).strip()
            return ts, (text or None)

    # Direct message format (newer Codex CLI): type="message" with role="assistant"
    if msg_type == "message" and data.get("role") == "assistant":
        ts = data.get("timestamp")
        content = data.get("content")
        text = _extract_text_from_codex_content(content).strip()
        return ts, (text or None)

    return None, None


def _is_turn_aborted(data: Dict[str, Any]) -> bool:
    if data.get("type") != "event_msg":
        return False
    payload = data.get("payload") or {}
    return payload.get("type") == "turn_aborted"


@dataclass
class _CodexTurn:
    start_line: int
    start_timestamp: Optional[str]
    user_message: str
    end_line: Optional[int] = None
    assistant_line: Optional[int] = None
    assistant_timestamp: Optional[str] = None
    assistant_message: Optional[str] = None


class CodexTrigger(TurnTrigger):
    """
    Turn trigger for Codex session JSONL.

    Completed turn definition:
    - A "real" user request (after cleaning) appears, and
    - A subsequent assistant final message appears (agent_message or assistant message)
    """

    def get_supported_formats(self) -> List[str]:
        return ["codex"]

    def detect_session_format(self, session_file: Path) -> Optional[str]:
        try:
            with session_file.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 20:
                        break
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    # Format 1: session_meta with originator containing "codex"
                    if data.get("type") == "session_meta":
                        payload = data.get("payload") or {}
                        originator = str(payload.get("originator") or "").lower()
                        if "codex" in originator:
                            return "codex"

                    # Format 2: response_item wrapper around messages
                    if data.get("type") == "response_item":
                        payload = data.get("payload") or {}
                        if payload.get("type") == "message" and payload.get("role") in (
                            "user",
                            "assistant",
                        ):
                            return "codex"

                    # Format 3: event_msg with user_message/agent_message
                    if data.get("type") == "event_msg":
                        payload = data.get("payload") or {}
                        if payload.get("type") in ("user_message", "agent_message"):
                            return "codex"

                    # Format 4: Direct message format (newer Codex CLI)
                    # First line may have {id, timestamp, git} without type field
                    if i == 0 and "id" in data and "timestamp" in data and "type" not in data:
                        # This looks like a Codex session header
                        return "codex"

                    # Direct type: "message" with role (newer format)
                    if data.get("type") == "message" and data.get("role") in ("user", "assistant"):
                        return "codex"

            return None
        except Exception:
            return None

    def _extract_completed_turns(self, session_file: Path) -> List[_CodexTurn]:
        turns: list[_CodexTurn] = []
        current: Optional[_CodexTurn] = None
        total_lines = 0

        try:
            with session_file.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    total_lines = line_no
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    # 1) User request markers (turn starts)
                    user_ts, user_msg = _extract_codex_user_candidate(data)
                    if user_msg:
                        if current is None:
                            current = _CodexTurn(
                                start_line=line_no,
                                start_timestamp=user_ts,
                                user_message=user_msg,
                            )
                        else:
                            # Deduplicate the common pair:
                            # response_item(role=user) + event_msg(user_message) with same content.
                            if current.assistant_line is None and _normalize(
                                user_msg
                            ) == _normalize(current.user_message):
                                current.start_line = min(current.start_line, line_no)
                                if current.start_timestamp is None:
                                    current.start_timestamp = user_ts
                                continue

                            # If we already have an assistant message, previous turn is complete.
                            if current.assistant_line is not None:
                                current.end_line = line_no - 1
                                turns.append(current)

                            # Start new turn (drop incomplete previous turn if any).
                            current = _CodexTurn(
                                start_line=line_no,
                                start_timestamp=user_ts,
                                user_message=user_msg,
                            )
                        continue

                    # 2) Assistant final message markers
                    asst_ts, asst_msg = _extract_codex_assistant_candidate(data)
                    if asst_msg and current is not None:
                        current.assistant_line = line_no
                        current.assistant_timestamp = asst_ts
                        current.assistant_message = asst_msg
                        continue

                    # 3) Aborted turns (user interrupted / cancelled)
                    if (
                        current is not None
                        and current.assistant_line is None
                        and _is_turn_aborted(data)
                    ):
                        current = None

            if current is not None and current.assistant_line is not None:
                current.end_line = total_lines
                turns.append(current)

        except Exception:
            return []

        # Renumber as completed turns only; important for watcher turn_number alignment.
        return turns

    def count_complete_turns(self, session_file: Path) -> int:
        return len(self._extract_completed_turns(session_file))

    def extract_turn_info(self, session_file: Path, turn_number: int) -> Optional[TurnInfo]:
        turns = self._extract_completed_turns(session_file)
        if not (1 <= turn_number <= len(turns)):
            return None
        t = turns[turn_number - 1]
        return TurnInfo(
            turn_number=turn_number,
            user_message=t.user_message,
            start_line=t.start_line,
            end_line=int(t.end_line or t.start_line),
            timestamp=t.start_timestamp,
        )

    def is_turn_complete(self, session_file: Path, turn_number: int) -> bool:
        return 1 <= turn_number <= self.count_complete_turns(session_file)

    def get_detailed_analysis(self, session_file: Path) -> Dict[str, Any]:
        turns = self._extract_completed_turns(session_file)

        groups: list[dict[str, Any]] = []
        for idx, t in enumerate(turns, 1):
            start_line = int(t.start_line)
            end_line = int(t.end_line or t.start_line)

            group: dict[str, Any] = {
                "turn_number": idx,
                "root_timestamp": t.start_timestamp,
                "message_count": 1,
                "parent_chains": 1,
                "retry_count": 0,
                "start_line": start_line,
                "end_line": end_line,
                "line_range": (
                    f"{start_line}-{end_line}" if start_line != end_line else str(start_line)
                ),
                "user_message": t.user_message,
                "summary_line": t.assistant_line,
                "summary_message": t.assistant_message,
                "turn_status": "completed",
                "interrupted": False,
                "turn_status_line": None,
                "turn_status_message_preview": "",
            }
            groups.append(group)

        return {
            "total_turns": len(groups),
            "total_messages": len(groups),
            "total_retries": 0,
            "groups": groups,
        }
