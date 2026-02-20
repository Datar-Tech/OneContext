"""
Turn Triggers - 对话轮次触发器系统

提供可插拔的trigger机制用于检测和提取session中的完整对话轮次。
"""

from .base import TurnTrigger, TurnInfo
from .claude_trigger import ClaudeTrigger
from .codex_trigger import CodexTrigger
from .gemini_trigger import GeminiTrigger
from .next_turn_trigger import NextTurnTrigger
from .registry import TriggerRegistry, get_global_registry

__all__ = [
    "TurnTrigger",
    "TurnInfo",
    "NextTurnTrigger",
    "ClaudeTrigger",
    "CodexTrigger",
    "GeminiTrigger",
    "TriggerRegistry",
    "get_global_registry",
]
