"""
Claude Code Stop Hook 自动安装器

在 watcher 启动时自动将 Aline 的 Stop hook 安装到 Claude Code 的配置中。
使用追加模式，不会覆盖用户现有的 hooks 配置。
"""

import json
import sys
from pathlib import Path
from typing import Optional

from ..logging_config import setup_logger

logger = setup_logger("realign.hooks.installer", "hooks_installer.log")

# 用于标识 Aline hook 的标记，方便后续识别和卸载
ALINE_HOOK_MARKER = "aline-stop-hook"


def get_stop_hook_script_path() -> Path:
    """获取 stop_hook.py 脚本的路径"""
    # 脚本与此文件在同一目录
    return Path(__file__).parent / "stop_hook.py"


def get_stop_hook_command() -> str:
    """
    获取 Stop hook 的执行命令。

    使用直接脚本路径调用，避免模块导入问题。
    """
    script_path = get_stop_hook_script_path()
    return f"{sys.executable} {script_path}"


def get_settings_path(project_path: Optional[Path] = None) -> Path:
    """
    获取 Claude Code settings.json 的路径。

    Args:
        project_path: 如果提供，返回项目级配置路径；否则返回全局配置路径。

    Returns:
        settings.json 的路径
    """
    if project_path:
        return project_path / ".claude" / "settings.local.json"
    return Path.home() / ".claude" / "settings.json"


def is_hook_installed(settings_path: Path) -> bool:
    """
    检查 Aline Stop hook 是否已安装。

    Args:
        settings_path: settings.json 的路径

    Returns:
        True 如果已安装
    """
    if not settings_path.exists():
        return False

    try:
        settings = json.loads(settings_path.read_text())
        stop_hooks = settings.get("hooks", {}).get("Stop", [])

        for hook_config in stop_hooks:
            hooks_list = hook_config.get("hooks", [])
            for h in hooks_list:
                command = h.get("command", "")
                if ALINE_HOOK_MARKER in command or "realign.hooks.stop_hook" in command:
                    return True

        return False

    except Exception:
        return False


def install_stop_hook(
    settings_path: Optional[Path] = None,
    force: bool = False,
    quiet: bool = False,
) -> bool:
    """
    安装 Claude Code Stop hook。

    使用追加模式：不覆盖用户现有的 hooks 配置。

    Args:
        settings_path: settings.json 的路径（默认为全局配置）
        force: 如果为 True，即使已安装也重新安装
        quiet: 如果为 True，不输出任何消息

    Returns:
        True 如果安装成功或已安装
    """
    if settings_path is None:
        settings_path = get_settings_path()

    try:
        # 检查是否已安装
        if not force and is_hook_installed(settings_path):
            logger.debug("Aline Stop hook already installed")
            return True

        # 读取现有设置
        settings = {}
        if settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text())
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {settings_path}, creating new settings")
                settings = {}

        # 确保 hooks 结构存在
        if "hooks" not in settings:
            settings["hooks"] = {}
        if "Stop" not in settings["hooks"]:
            settings["hooks"]["Stop"] = []

        # 如果强制安装，先移除旧的 Aline hook
        if force:
            stop_hooks = settings["hooks"]["Stop"]
            new_hooks = []
            for hook_config in stop_hooks:
                hooks_list = hook_config.get("hooks", [])
                filtered = [
                    h
                    for h in hooks_list
                    if ALINE_HOOK_MARKER not in h.get("command", "")
                    and "realign.hooks.stop_hook" not in h.get("command", "")
                ]
                if filtered:
                    hook_config["hooks"] = filtered
                    new_hooks.append(hook_config)
            settings["hooks"]["Stop"] = new_hooks

        # 追加 Aline hook
        hook_command = get_stop_hook_command()
        aline_hook = {
            "hooks": [{"type": "command", "command": f"{hook_command}  # {ALINE_HOOK_MARKER}"}]
        }
        settings["hooks"]["Stop"].append(aline_hook)

        # 写回设置
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings, indent=2))

        logger.info(f"Aline Stop hook installed to {settings_path}")
        if not quiet:
            print(f"[Aline] Stop hook installed to {settings_path}", file=sys.stderr)

        return True

    except Exception as e:
        logger.error(f"Failed to install Stop hook: {e}")
        if not quiet:
            print(f"[Aline] Failed to install Stop hook: {e}", file=sys.stderr)
        return False


def uninstall_stop_hook(
    settings_path: Optional[Path] = None,
    quiet: bool = False,
) -> bool:
    """
    卸载 Aline Stop hook。

    Args:
        settings_path: settings.json 的路径（默认为全局配置）
        quiet: 如果为 True，不输出任何消息

    Returns:
        True 如果卸载成功
    """
    if settings_path is None:
        settings_path = get_settings_path()

    try:
        if not settings_path.exists():
            return True

        settings = json.loads(settings_path.read_text())
        stop_hooks = settings.get("hooks", {}).get("Stop", [])

        # 过滤掉 Aline hook
        new_hooks = []
        removed = False
        for hook_config in stop_hooks:
            hooks_list = hook_config.get("hooks", [])
            filtered = [
                h
                for h in hooks_list
                if ALINE_HOOK_MARKER not in h.get("command", "")
                and "realign.hooks.stop_hook" not in h.get("command", "")
            ]
            if len(filtered) < len(hooks_list):
                removed = True
            if filtered:
                hook_config["hooks"] = filtered
                new_hooks.append(hook_config)

        if removed:
            settings["hooks"]["Stop"] = new_hooks
            settings_path.write_text(json.dumps(settings, indent=2))
            logger.info("Aline Stop hook uninstalled")
            if not quiet:
                print("[Aline] Stop hook uninstalled", file=sys.stderr)

        return True

    except Exception as e:
        logger.error(f"Failed to uninstall Stop hook: {e}")
        if not quiet:
            print(f"[Aline] Failed to uninstall Stop hook: {e}", file=sys.stderr)
        return False


def ensure_stop_hook_installed(quiet: bool = True) -> bool:
    """
    确保 Stop hook 已安装（用于 watcher 启动时调用）。

    这是一个幂等操作，如果已安装则不会重复安装。

    Args:
        quiet: 如果为 True，只在首次安装时输出消息

    Returns:
        True 如果 hook 可用
    """
    settings_path = get_settings_path()

    if is_hook_installed(settings_path):
        logger.debug("Stop hook already installed, skipping")
        return True

    return install_stop_hook(settings_path, quiet=quiet)
