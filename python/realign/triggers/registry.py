"""
Trigger注册表

负责管理、注册和选择合适的turn trigger。
"""

import logging
from pathlib import Path
from typing import Dict, Type, Optional
from .base import TurnTrigger
from .claude_trigger import ClaudeTrigger
from .codex_trigger import CodexTrigger
from .gemini_trigger import GeminiTrigger
from .next_turn_trigger import NextTurnTrigger

logger = logging.getLogger(__name__)


class TriggerRegistry:
    """Trigger注册表，负责管理和选择trigger"""

    def __init__(self):
        self._triggers: Dict[str, Type[TurnTrigger]] = {}

        # 按session类型映射trigger
        self._type_to_trigger: Dict[str, Type[TurnTrigger]] = {
            "claude": ClaudeTrigger,
            "codex": CodexTrigger,
            "gemini": GeminiTrigger,
        }

        # 注册所有triggers (用于list_triggers和get_trigger)
        self.register("next_turn", NextTurnTrigger)
        self.register("claude", ClaudeTrigger)
        self.register("codex", CodexTrigger)
        self.register("gemini", GeminiTrigger)

    def register(self, name: str, trigger_class: Type[TurnTrigger]):
        """
        注册一个trigger

        Args:
            name: trigger名称
            trigger_class: trigger类（不是实例）
        """
        self._triggers[name] = trigger_class

    def get_trigger(self, name: str, config: Optional[Dict] = None) -> Optional[TurnTrigger]:
        """
        根据名称获取trigger实例

        Args:
            name: trigger名称
            config: 可选的配置字典

        Returns:
            Trigger实例，如果不存在返回None
        """
        trigger_class = self._triggers.get(name)
        if trigger_class:
            return trigger_class(config)
        return None

    def get_trigger_for_type(
        self, session_type: str, config: Optional[Dict] = None
    ) -> Optional[TurnTrigger]:
        """
        根据session类型获取对应的trigger

        Args:
            session_type: session类型 ("claude", "codex", "gemini")
            config: 可选的配置字典

        Returns:
            Trigger实例，如果类型不支持返回None并记录错误
        """
        trigger_class = self._type_to_trigger.get(session_type)
        if trigger_class:
            return trigger_class(config)
        logger.error(f"No trigger registered for session type: {session_type}")
        return None

    def list_triggers(self) -> list:
        """
        列出所有已注册的triggers

        Returns:
            trigger名称列表
        """
        return list(self._triggers.keys())

    def list_supported_types(self) -> list:
        """
        列出所有支持的session类型

        Returns:
            session类型列表
        """
        return list(self._type_to_trigger.keys())

    def auto_select_trigger(
        self, session_file: Path, config: Optional[Dict] = None
    ) -> Optional[TurnTrigger]:
        """
        Auto-select an appropriate trigger for the given session file.

        Returns a concrete trigger instance (e.g., CodexTrigger), not NextTurnTrigger.
        """
        try:
            selected = NextTurnTrigger(config).select_trigger(session_file)
            return selected or NextTurnTrigger(config)
        except Exception:
            # Fallback to next_turn delegator (safe default)
            return NextTurnTrigger(config)


# 全局注册表实例
_global_registry = TriggerRegistry()


def get_global_registry() -> TriggerRegistry:
    """
    获取全局trigger注册表

    Returns:
        全局TriggerRegistry实例
    """
    return _global_registry
