"""
Turn trigger抽象基类

定义了对话轮次触发器的标准接口，用于检测和提取session中的完整轮次。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    pass


@dataclass
class TurnInfo:
    """单个对话轮次的信息"""

    turn_number: int  # 轮次编号（1-based）
    user_message: str  # 用户消息内容
    start_line: int  # 在session文件中的起始行
    end_line: int  # 在session文件中的结束行
    timestamp: Optional[str] = None  # 时间戳
    turn_hash: Optional[str] = None  # 轮次内容hash


class TurnTrigger(ABC):
    """
    对话轮次触发器的抽象基类

    每个trigger负责：
    1. 检测session中有多少完整的轮次
    2. 提取指定轮次的信息
    3. 判断轮次是否完整
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化trigger

        Args:
            config: 可选的配置字典，用于传递模型特定的参数
        """
        self.config = config or {}

    @abstractmethod
    def count_complete_turns(self, session_file: Path) -> int:
        """
        计算session中完整轮次的数量

        Args:
            session_file: session文件路径

        Returns:
            完整轮次的数量（1-based，表示有多少个完整的对话轮次）
        """
        pass

    @abstractmethod
    def extract_turn_info(self, session_file: Path, turn_number: int) -> Optional[TurnInfo]:
        """
        提取指定轮次的详细信息

        Args:
            session_file: session文件路径
            turn_number: 轮次编号（1-based）

        Returns:
            TurnInfo对象，如果轮次不存在返回None
        """
        pass

    @abstractmethod
    def is_turn_complete(self, session_file: Path, turn_number: int) -> bool:
        """
        判断指定轮次是否已完整

        Args:
            session_file: session文件路径
            turn_number: 轮次编号（1-based）

        Returns:
            True如果轮次完整，False otherwise
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        返回此trigger支持的session格式

        Returns:
            格式列表，例如 ["claude_code_2.0", "codex_1.x"]
        """
        pass

    @abstractmethod
    def get_detailed_analysis(self, session_file: Path) -> Dict[str, Any]:
        """
        获取session的详细分析信息，用于watcher集成

        Returns:
            包含以下字段的字典:
            - groups: List[Dict] - 每个turn的详细信息，每个group包含:
                - turn_number: int
                - user_message: str
                - summary_message: str (可选)
                - turn_status: str ('completed', 'interrupted', etc.)
                - start_line: int (可选)
                - end_line: int (可选)
            - total_turns: int - 总轮次数
            - format: str - session格式 (可选)
        """
        pass

    def detect_session_format(self, session_file: Path) -> Optional[str]:
        """
        检测session文件的格式（可选重写）

        Args:
            session_file: session文件路径

        Returns:
            格式字符串，如果无法检测返回None
        """
        import json

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if not first_line:
                    return None

                data = json.loads(first_line)

                # 检测Claude Code格式
                if "version" in data and data["version"].startswith("2.0"):
                    return f"claude_code_{data['version']}"

                # 检测Codex格式
                if data.get("type") == "event_msg" and "payload" in data:
                    return "codex"

                return None
        except Exception:
            return None
