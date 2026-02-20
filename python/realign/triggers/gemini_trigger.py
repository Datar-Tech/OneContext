"""
Gemini CLI Trigger

Handles turn detection for Gemini CLI sessions (.json format).
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from .base import TurnTrigger, TurnInfo

logger = logging.getLogger(__name__)


class GeminiTrigger(TurnTrigger):
    """
    Trigger for Gemini CLI sessions.

    Gemini CLI stores sessions as JSON files with structure:
    {
        "sessionId": "...",
        "projectHash": "...",
        "messages": [
            {"id": "...", "timestamp": "...", "type": "user", "content": "..."},
            {"id": "...", "timestamp": "...", "type": "gemini", "content": "..."},
            {"id": "...", "timestamp": "...", "type": "info", "content": "..."},
        ]
    }

    A turn is defined as a user message followed by gemini response(s).
    """

    def get_supported_formats(self) -> List[str]:
        """Return supported session formats."""
        return ["gemini_json", "gemini"]

    def detect_session_format(self, session_file: Path) -> Optional[str]:
        """Detect if this is a Gemini session file."""
        try:
            # Check file extension and path
            if session_file.suffix != ".json":
                return None
            if ".gemini/" not in str(session_file) or "/chats/" not in str(session_file):
                return None

            # Verify it's a valid Gemini session
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "messages" in data and "sessionId" in data:
                    return "gemini_json"

            return None
        except Exception as e:
            logger.debug(f"Error detecting Gemini format: {e}")
            return None

    def count_complete_turns(self, session_file: Path) -> int:
        """
        Count complete turns in a Gemini session.

        A turn is complete when there's a user message.
        """
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            messages = data.get("messages", [])
            user_count = sum(1 for m in messages if m.get("type") == "user")

            return user_count
        except Exception as e:
            logger.error(f"Error counting Gemini turns: {e}")
            return 0

    def is_turn_complete(self, session_file: Path, turn_number: int) -> bool:
        """Check if a specific turn is complete."""
        total_turns = self.count_complete_turns(session_file)
        return turn_number <= total_turns

    def extract_turn_info(self, session_file: Path, turn_number: int) -> Optional[TurnInfo]:
        """
        Extract information for a specific turn.

        Returns TurnInfo with user message and metadata.
        """
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            messages = data.get("messages", [])

            # Find the nth user message
            user_messages = [(i, m) for i, m in enumerate(messages) if m.get("type") == "user"]

            if turn_number < 1 or turn_number > len(user_messages):
                return None

            user_idx, user_msg = user_messages[turn_number - 1]

            return TurnInfo(
                turn_number=turn_number,
                user_message=user_msg.get("content", ""),
                start_line=1,  # JSON file doesn't use line numbers
                end_line=1,
                timestamp=user_msg.get("timestamp"),
            )

        except Exception as e:
            logger.error(f"Error extracting Gemini turn info: {e}")
            return None

    def extract_turn_content(
        self, session_file: Path, turn_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Extract full content for a specific turn (for commit generation).

        Returns dict with:
        - user_message: The user's input
        - assistant_response: Gemini's response(s)
        - turn_content: Full JSON content for this turn
        - timestamp: Turn timestamp
        """
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            messages = data.get("messages", [])

            # Find the nth user message
            user_messages = [(i, m) for i, m in enumerate(messages) if m.get("type") == "user"]

            if turn_number < 1 or turn_number > len(user_messages):
                return None

            user_idx, user_msg = user_messages[turn_number - 1]

            # Collect gemini responses after this user message until next user message
            next_user_idx = len(messages)
            if turn_number < len(user_messages):
                next_user_idx = user_messages[turn_number][0]

            gemini_responses = []
            for i in range(user_idx + 1, next_user_idx):
                msg = messages[i]
                if msg.get("type") == "gemini":
                    gemini_responses.append(msg.get("content", ""))

            # Build turn content
            turn_content = json.dumps(
                {
                    "turn_number": turn_number,
                    "user": user_msg,
                    "responses": [messages[i] for i in range(user_idx, next_user_idx)],
                },
                ensure_ascii=False,
                indent=2,
            )

            return {
                "user_message": user_msg.get("content", ""),
                "assistant_response": "\n".join(gemini_responses),
                "turn_content": turn_content,
                "timestamp": user_msg.get("timestamp"),
                "turn_number": turn_number,
            }

        except Exception as e:
            logger.error(f"Error extracting Gemini turn content: {e}")
            return None

    def get_detailed_analysis(self, session_file: Path) -> Dict[str, Any]:
        """
        Get detailed analysis of the session for watcher integration.

        Returns dict with:
        - groups: List of turn groups with metadata
        - total_turns: Total number of turns
        """
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            messages = data.get("messages", [])
            user_messages = [(i, m) for i, m in enumerate(messages) if m.get("type") == "user"]

            groups = []
            for turn_idx, (msg_idx, user_msg) in enumerate(user_messages, 1):
                # Find next user message index
                next_user_idx = len(messages)
                if turn_idx < len(user_messages):
                    next_user_idx = user_messages[turn_idx][0]

                # Collect gemini responses
                gemini_responses = []
                for i in range(msg_idx + 1, next_user_idx):
                    if messages[i].get("type") == "gemini":
                        gemini_responses.append(messages[i].get("content", ""))

                groups.append(
                    {
                        "turn_number": turn_idx,
                        "user_message": user_msg.get("content", ""),
                        "summary_message": (
                            "\n".join(gemini_responses)[:500] if gemini_responses else ""
                        ),
                        "turn_status": "completed",
                        "start_line": 1,  # JSON file doesn't use line numbers
                        "end_line": 1,
                        "lines": [1],
                    }
                )

            return {
                "groups": groups,
                "total_turns": len(groups),
                "format": "gemini_json",
            }

        except Exception as e:
            logger.error(f"Error in get_detailed_analysis for Gemini: {e}")
            return {"groups": [], "total_turns": 0, "format": "gemini_json"}
