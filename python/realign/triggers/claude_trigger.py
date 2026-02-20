"""
ClaudeTrigger - Claude Code session turn detection

Combines ParentChainTrigger logic (correct turn grouping) with retry detection.

Features:
1. Traces parentUUID chains to find root messages
2. Groups messages by root timestamp (handles session compact correctly)
3. Detects and merges API retry patterns (same content + short interval)
4. Enriches turn data with status and summary information
"""

import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from collections import defaultdict

from .base import TurnTrigger, TurnInfo


class ClaudeTrigger(TurnTrigger):
    """
    Turn trigger for Claude Code JSONL sessions.

    Strategy:
    1. Extract all user messages (excluding tool_result)
    2. Trace parentUUID chain to find each message's root
    3. Group by root timestamp (same root timestamp = same logical turn)
    4. Detect and merge API retries (same content within time window)
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        # Retry detection config
        self.retry_time_window = config.get("retry_time_window", 120) if config else 120

    def get_supported_formats(self) -> List[str]:
        return ["claude_code"]

    def detect_session_format(self, session_file: Path) -> Optional[str]:
        """Detect if this is a Claude Code session file."""
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Claude Code format: {type: "assistant"/"user", message: {...}}
                        if data.get("type") in ("assistant", "user") and "message" in data:
                            return "claude_code"
                    except json.JSONDecodeError:
                        continue
            return None
        except Exception:
            return None

    # =========================================================================
    # Message Extraction
    # =========================================================================

    def _extract_messages(self, session_file: Path) -> List[Dict]:
        """Extract all user messages (excluding tool_result and system placeholders)."""
        messages = []

        def _is_command_wrapper(content: str) -> bool:
            if not content:
                return False
            stripped = content.strip()
            return (
                stripped.startswith("<command-name>")
                or stripped.startswith("<local-command-stdout>")
                or stripped.startswith("<local-command-stderr>")
            )

        def _is_interrupt_placeholder(text: str) -> bool:
            if not text:
                return False
            return "request interrupted by user" in text.strip().lower()

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        # Skip meta/snapshot lines
                        if data.get("isMeta"):
                            continue

                        if data.get("type") == "user":
                            message = data.get("message", {})
                            content = message.get("content", [])

                            # Skip tool results
                            is_tool_result = False
                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict) and item.get("type") == "tool_result":
                                        is_tool_result = True
                                        break

                            # Skip command wrappers
                            if isinstance(content, str) and _is_command_wrapper(content):
                                continue

                            # Skip interrupt placeholders
                            if isinstance(content, str) and _is_interrupt_placeholder(content):
                                continue

                            if isinstance(content, list):
                                text_items = [
                                    item.get("text", "")
                                    for item in content
                                    if isinstance(item, dict) and item.get("type") == "text"
                                ]
                                if text_items and all(
                                    _is_interrupt_placeholder(t) for t in text_items
                                ):
                                    continue

                            if not is_tool_result:
                                messages.append(
                                    {
                                        "line_no": line_no,
                                        "uuid": data.get("uuid"),
                                        "parent_uuid": data.get("parentUuid"),
                                        "timestamp": data.get("timestamp"),
                                        "content": content,
                                    }
                                )

                    except json.JSONDecodeError:
                        continue

        except Exception:
            pass

        return messages

    # =========================================================================
    # Parent Chain Grouping
    # =========================================================================

    def _find_root(self, msg: Dict, uuid_to_msg: Dict[str, Dict]) -> Dict:
        """Trace parentUUID chain to find root message."""
        visited = set()
        current = msg

        while current["parent_uuid"] and current["parent_uuid"] in uuid_to_msg:
            if current["uuid"] in visited:
                break
            visited.add(current["uuid"])
            current = uuid_to_msg[current["parent_uuid"]]

        return current

    def _group_by_root_timestamp(self, messages: List[Dict]) -> Dict[str, List[Dict]]:
        """Group messages by their root's timestamp."""
        uuid_to_msg = {m["uuid"]: m for m in messages}

        for msg in messages:
            root = self._find_root(msg, uuid_to_msg)
            msg["root_uuid"] = root["uuid"]
            msg["root_timestamp"] = root["timestamp"]

        groups = defaultdict(list)
        for msg in messages:
            if msg["root_timestamp"]:
                groups[msg["root_timestamp"]].append(msg)

        return dict(groups)

    # =========================================================================
    # Retry Detection & Merging
    # =========================================================================

    def _get_content_hash(self, messages: List[Dict]) -> str:
        """Compute hash of message content for duplicate detection."""
        texts = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))

        combined = "|".join(texts)
        return hashlib.md5(combined.encode()).hexdigest()

    def _parse_timestamp(self, timestamp: str) -> Optional[datetime]:
        if not timestamp:
            return None
        try:
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except Exception:
            return None

    def _time_diff_seconds(self, ts1: str, ts2: str) -> float:
        dt1 = self._parse_timestamp(ts1)
        dt2 = self._parse_timestamp(ts2)
        if not dt1 or not dt2:
            return float("inf")
        return abs((dt2 - dt1).total_seconds())

    def _is_retry(self, group1: Dict, group2: Dict) -> bool:
        """Check if group2 is a retry of group1 (same content within time window)."""
        time_diff = self._time_diff_seconds(group1["timestamp"], group2["timestamp"])
        if time_diff > self.retry_time_window:
            return False

        hash1 = self._get_content_hash(group1["messages"])
        hash2 = self._get_content_hash(group2["messages"])

        return hash1 == hash2

    def _merge_retry_groups(self, groups: List[Dict]) -> List[Dict]:
        """Merge consecutive retry groups into single logical turns."""
        if not groups:
            return []

        sorted_groups = sorted(groups, key=lambda g: g["timestamp"] or "")
        merged = []
        i = 0

        while i < len(sorted_groups):
            current = sorted_groups[i]
            retry_groups = [current]
            j = i + 1

            while j < len(sorted_groups):
                if self._is_retry(current, sorted_groups[j]):
                    retry_groups.append(sorted_groups[j])
                    j += 1
                else:
                    break

            if len(retry_groups) > 1:
                merged_group = {
                    "timestamp": retry_groups[0]["timestamp"],
                    "messages": [],
                    "lines": [],
                    "retry_count": len(retry_groups),
                    "parent_chains": sum(g.get("parent_chains", 1) for g in retry_groups),
                }
                for g in retry_groups:
                    merged_group["messages"].extend(g["messages"])
                    merged_group["lines"].extend(g["lines"])
                merged_group["lines"] = sorted(set(merged_group["lines"]))
                merged.append(merged_group)
            else:
                current["retry_count"] = 0
                merged.append(current)

            i = j

        return merged

    # =========================================================================
    # Core Interface Methods
    # =========================================================================

    def _get_file_line_count(self, session_file: Path) -> int:
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def count_complete_turns(self, session_file: Path) -> int:
        """
        Count complete turns (with retry merging).

        Turn completion detection strategy (designed to work with Stop Hook):

        1. For non-last turns (has next user message): Always considered complete.
           Rationale: The existence of a subsequent user message proves the previous
           turn finished (user can't send a new message while Claude is responding).

        2. For the last turn: Only count as complete if it has an explicit end marker
           (user_interrupted, rate_limited, compacted). Do NOT use has_summary here
           because assistant content appears immediately when Claude starts responding,
           which would cause false positives.

        The actual completion of the last turn is detected by:
        - Stop Hook (primary, immediate) - fires when Claude truly finishes
        - Idle timeout (fallback) - 60s of no file changes

        This prevents the watcher from triggering commits while Claude is still
        responding (the old bug where summary was extracted from incomplete responses).
        """
        messages = self._extract_messages(session_file)
        groups_dict = self._group_by_root_timestamp(messages)

        groups = []
        for timestamp, msgs in groups_dict.items():
            root_uuids = set(msg.get("root_uuid") for msg in msgs if msg.get("root_uuid"))
            groups.append(
                {
                    "timestamp": timestamp,
                    "messages": msgs,
                    "lines": sorted([msg["line_no"] for msg in msgs]),
                    "parent_chains": len(root_uuids),
                }
            )

        merged_groups = self._merge_retry_groups(groups)
        if not merged_groups:
            return 0

        total_lines = self._get_file_line_count(session_file)

        # Optional enricher for detecting explicit end markers
        try:
            from .turn_status import detect_turn_end_status
        except Exception:
            detect_turn_end_status = None  # type: ignore[assignment]

        complete = 0
        for idx, group in enumerate(merged_groups, 1):
            lines = group["lines"]
            start_line = int(lines[0]) if lines else 0

            # Non-last turn: has a subsequent user message = definitely complete
            if idx < len(merged_groups):
                complete += 1
                continue

            # Last turn: only count if it has an explicit end marker
            # (Do NOT check has_summary - it becomes true as soon as Claude starts responding)
            if idx < len(merged_groups):
                next_start = merged_groups[idx]["lines"][0]
                end_line = max(int(lines[-1]), int(next_start) - 1)
            else:
                end_line = max(int(lines[-1]) if lines else 0, int(total_lines))

            has_explicit_end_marker = False
            if detect_turn_end_status and start_line > 0 and end_line >= start_line:
                status = detect_turn_end_status(session_file, start_line, end_line)
                if status.line is not None and status.status in (
                    "user_interrupted",
                    "rate_limited",
                    "compacted",
                ):
                    has_explicit_end_marker = True

            if has_explicit_end_marker:
                complete += 1

        return complete

    def extract_turn_info(self, session_file: Path, turn_number: int) -> Optional[TurnInfo]:
        """Extract info for a specific turn."""
        from ..hooks import clean_user_message

        analysis = self.get_detailed_analysis(session_file)

        if 0 < turn_number <= len(analysis["groups"]):
            group = analysis["groups"][turn_number - 1]

            messages = self._extract_messages(session_file)
            group_lines = set(group["lines"])
            first_msg = None

            for msg in messages:
                if msg["line_no"] in group_lines:
                    first_msg = msg
                    break

            if first_msg:
                content = first_msg["content"]
                extracted_text = None

                if isinstance(content, str):
                    extracted_text = content
                elif isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    if text_parts:
                        extracted_text = "\n".join(text_parts)

                if extracted_text:
                    cleaned_text = clean_user_message(extracted_text)
                    return TurnInfo(
                        turn_number=turn_number,
                        user_message=cleaned_text,
                        start_line=group["start_line"],
                        end_line=group["end_line"],
                        timestamp=group["root_timestamp"],
                    )

        return None

    def is_turn_complete(self, session_file: Path, turn_number: int) -> bool:
        return turn_number <= self.count_complete_turns(session_file)

    def get_detailed_analysis(self, session_file: Path) -> Dict[str, Any]:
        """Get detailed analysis with turn status and summary enrichment."""
        from ..hooks import clean_user_message

        messages = self._extract_messages(session_file)
        groups_dict = self._group_by_root_timestamp(messages)

        groups = []
        for timestamp, msgs in groups_dict.items():
            root_uuids = set(msg.get("root_uuid") for msg in msgs if msg.get("root_uuid"))
            groups.append(
                {
                    "timestamp": timestamp,
                    "messages": msgs,
                    "lines": sorted([msg["line_no"] for msg in msgs]),
                    "parent_chains": len(root_uuids),
                }
            )

        merged_groups = self._merge_retry_groups(groups)
        total_lines = self._get_file_line_count(session_file)

        detailed_groups = []
        for turn_num, group in enumerate(merged_groups, 1):
            lines = group["lines"]
            start_line = lines[0]

            # Extend end_line to next turn's start - 1, or file end
            if turn_num < len(merged_groups):
                next_start = merged_groups[turn_num]["lines"][0]
                end_line = max(lines[-1], next_start - 1)
            else:
                end_line = max(lines[-1], total_lines)

            # Extract user message text
            extracted_text = ""
            if group.get("messages"):
                first_msg = group["messages"][0]
                content = first_msg.get("content", [])
                if isinstance(content, str):
                    extracted_text = content
                elif isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    if text_parts:
                        extracted_text = "\n".join(text_parts)

            user_message = clean_user_message(extracted_text) if extracted_text else ""

            detailed_groups.append(
                {
                    "turn_number": turn_num,
                    "root_timestamp": group["timestamp"],
                    "message_count": len(group["messages"]),
                    "parent_chains": group.get("parent_chains", 1),
                    "retry_count": group.get("retry_count", 0),
                    "lines": lines,
                    "start_line": start_line,
                    "end_line": end_line,
                    "line_range": (
                        f"{start_line}-{end_line}" if start_line != end_line else str(start_line)
                    ),
                    "user_message": user_message,
                }
            )

        # Enrich with turn status
        try:
            from .turn_status import detect_turn_end_status, preview_text

            for g in detailed_groups:
                status = detect_turn_end_status(session_file, g["start_line"], g["end_line"])
                g["turn_status"] = status.status
                g["interrupted"] = status.status in ("user_interrupted", "rate_limited")
                g["turn_status_line"] = status.line
                g["turn_status_message_preview"] = preview_text(status.message)
        except Exception:
            pass

        # Enrich with assistant summary
        try:
            from .turn_summary import extract_turn_summary

            for g in detailed_groups:
                summary_line, summary_msg = extract_turn_summary(
                    session_file, g["start_line"], g["end_line"]
                )
                g["summary_line"] = summary_line
                g["summary_message"] = summary_msg
        except Exception:
            pass

        return {
            "total_turns": len(merged_groups),
            "total_messages": len(messages),
            "total_retries": sum(g.get("retry_count", 0) for g in merged_groups),
            "groups": detailed_groups,
            "format": "claude_code",
        }
