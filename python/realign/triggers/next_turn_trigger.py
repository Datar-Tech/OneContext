"""
NextTurnTrigger

A generic delegating trigger that auto-detects the session format and dispatches
to the appropriate concrete trigger (Claude/Codex/Gemini).

This is used as a stable default trigger name ("next_turn") across the codebase.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import TurnInfo, TurnTrigger
from .claude_trigger import ClaudeTrigger
from .codex_trigger import CodexTrigger
from .gemini_trigger import GeminiTrigger


class NextTurnTrigger(TurnTrigger):
    """Auto-detect trigger for any supported session type."""

    def select_trigger(self, session_file: Path) -> Optional[TurnTrigger]:
        # Skip directories (not a supported session format)
        if session_file.is_dir():
            return None

        # Heuristics based on path
        path_str = str(session_file).lower()
        if ".gemini" in path_str and "/tmp/" in path_str and session_file.suffix == ".json":
            return GeminiTrigger(self.config)
        if ".claude" in path_str and "/projects/" in path_str and session_file.suffix == ".jsonl":
            return ClaudeTrigger(self.config)

        # Sniff content (first ~50 lines)
        try:
            with session_file.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 50:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    # Claude Code: {"type":"user"|"assistant", "message": {...}}
                    if obj.get("type") in ("user", "assistant") and isinstance(
                        obj.get("message"), dict
                    ):
                        return ClaudeTrigger(self.config)

                    # Codex: session_meta/response_item patterns
                    if obj.get("type") == "session_meta":
                        payload = obj.get("payload") or {}
                        originator = str(payload.get("originator", "")).lower()
                        if "codex" in originator:
                            return CodexTrigger(self.config)
                    if obj.get("type") == "response_item":
                        return CodexTrigger(self.config)
        except Exception:
            pass

        # Fallback: codex rollout naming
        if session_file.name.startswith("rollout-") and session_file.suffix == ".jsonl":
            return CodexTrigger(self.config)

        return None

    def count_complete_turns(self, session_file: Path) -> int:
        trigger = self.select_trigger(session_file)
        return trigger.count_complete_turns(session_file) if trigger else 0

    def extract_turn_info(self, session_file: Path, turn_number: int) -> Optional[TurnInfo]:
        trigger = self.select_trigger(session_file)
        return trigger.extract_turn_info(session_file, turn_number) if trigger else None

    def is_turn_complete(self, session_file: Path, turn_number: int) -> bool:
        trigger = self.select_trigger(session_file)
        return trigger.is_turn_complete(session_file, turn_number) if trigger else False

    def get_supported_formats(self) -> List[str]:
        # Union of underlying triggers.
        formats: List[str] = []
        for cls in (ClaudeTrigger, CodexTrigger, GeminiTrigger):
            try:
                formats.extend(cls(self.config).get_supported_formats())
            except Exception:
                continue
        # Preserve order while deduping
        seen = set()
        deduped: List[str] = []
        for f in formats:
            if f in seen:
                continue
            seen.add(f)
            deduped.append(f)
        return deduped

    def get_detailed_analysis(self, session_file: Path) -> Dict[str, Any]:
        trigger = self.select_trigger(session_file)
        if not trigger:
            return {"groups": [], "total_turns": 0, "format": None}
        return trigger.get_detailed_analysis(session_file)
