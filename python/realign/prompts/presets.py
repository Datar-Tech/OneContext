"""
Prompt preset definitions and management.

This module defines different prompt presets for chat agents,
allowing users to customize the agent's behavior for different scenarios.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
import yaml

from ..logging_config import setup_logger

logger = setup_logger("realign.prompts.presets", "presets.log")


@dataclass
class PromptPreset:
    """A prompt preset for chat agents."""

    id: str
    name: str
    description: str
    allow_custom_instructions: bool = True
    custom_instructions_placeholder: str = ""
    # Note: system_prompt_template is handled in TypeScript backend
    # We only store metadata here for Python side selection UI


# Built-in presets
BUILTIN_PRESETS = [
    PromptPreset(
        id="default",
        name="Default Assistant",
        description="Help users explore conversation history with a general-purpose assistant",
        allow_custom_instructions=False,
        custom_instructions_placeholder="",
    ),
    PromptPreset(
        id="work-report",
        name="Work Report Agent",
        description="Act as you to report work to colleagues/managers",
        allow_custom_instructions=True,
        custom_instructions_placeholder=(
            "例如：突出我的工作贡献，弱化失误，注意礼貌\n"
            "Example: Highlight my contributions, downplay mistakes, be professional"
        ),
    ),
    PromptPreset(
        id="knowledge-agent",
        name="Knowledge Agent",
        description="Share your deep thinking as founder/architect/author",
        allow_custom_instructions=True,
        custom_instructions_placeholder=(
            "例如：分享技术决策背景，强调思考过程\n"
            "Example: Share technical decision context, emphasize thought process"
        ),
    ),
    PromptPreset(
        id="personality-analyzer",
        name="Personality Analyzer",
        description="Analyze personality based on conversation",
        allow_custom_instructions=False,
        custom_instructions_placeholder="",
    ),
]


def get_all_presets(include_custom: bool = True) -> List[PromptPreset]:
    """
    Get all available presets (built-in + custom).

    Args:
        include_custom: Whether to include user custom presets

    Returns:
        List of PromptPreset objects
    """
    presets = BUILTIN_PRESETS.copy()

    if include_custom:
        try:
            custom_presets = load_custom_presets()
            presets.extend(custom_presets)
            logger.info(f"Loaded {len(custom_presets)} custom presets")
        except Exception as e:
            logger.warning(f"Failed to load custom presets: {e}")

    return presets


def get_preset_by_id(preset_id: str, include_custom: bool = True) -> Optional[PromptPreset]:
    """
    Get a preset by its ID.

    Args:
        preset_id: The preset ID to search for
        include_custom: Whether to search in custom presets

    Returns:
        PromptPreset object or None if not found
    """
    all_presets = get_all_presets(include_custom=include_custom)

    for preset in all_presets:
        if preset.id == preset_id:
            return preset

    return None


def get_preset_by_index(index: int, include_custom: bool = True) -> Optional[PromptPreset]:
    """
    Get a preset by its index (1-based).

    Args:
        index: The preset index (1-based, as shown to users)
        include_custom: Whether to include custom presets

    Returns:
        PromptPreset object or None if index is out of range
    """
    all_presets = get_all_presets(include_custom=include_custom)

    if index < 1 or index > len(all_presets):
        return None

    return all_presets[index - 1]


def load_custom_presets() -> List[PromptPreset]:
    """
    Load custom presets from user configuration file.

    Looks for ~/.aline/prompt_presets.yaml

    Returns:
        List of custom PromptPreset objects
    """
    config_path = Path.home() / ".aline" / "prompt_presets.yaml"

    if not config_path.exists():
        logger.debug(f"No custom presets file found at {config_path}")
        return []

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config or "custom_presets" not in config:
            logger.warning("Custom presets file exists but has no 'custom_presets' key")
            return []

        custom_presets = []
        for preset_data in config["custom_presets"]:
            try:
                preset = PromptPreset(
                    id=preset_data["id"],
                    name=preset_data["name"],
                    description=preset_data["description"],
                    allow_custom_instructions=preset_data.get("allow_custom_instructions", True),
                    custom_instructions_placeholder=preset_data.get(
                        "custom_instructions_placeholder", ""
                    ),
                )
                custom_presets.append(preset)
                logger.debug(f"Loaded custom preset: {preset.id}")
            except KeyError as e:
                logger.warning(f"Invalid custom preset (missing key {e}): {preset_data}")
                continue

        return custom_presets

    except yaml.YAMLError as e:
        logger.error(f"Failed to parse custom presets YAML: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading custom presets: {e}")
        return []


def display_preset_menu(presets: Optional[List[PromptPreset]] = None) -> str:
    """
    Display preset selection menu and return formatted string.

    Args:
        presets: List of presets to display (defaults to all presets)

    Returns:
        Formatted menu string
    """
    if presets is None:
        presets = get_all_presets()

    lines = ["\n📋 Select a prompt preset for the chat agent:\n"]

    for idx, preset in enumerate(presets, 1):
        lines.append(f"[{idx}] {preset.name} ({preset.id})")
        lines.append(f"    {preset.description}")

        # Add a visual indicator if custom instructions are allowed
        if preset.allow_custom_instructions:
            lines.append("    ✏️  Allows custom instructions")

        lines.append("")  # Empty line between presets

    return "\n".join(lines)


def prompt_for_custom_instructions(preset: PromptPreset) -> str:
    """
    Prompt user for custom instructions if allowed by preset.

    Args:
        preset: The selected preset

    Returns:
        Custom instructions string (may be empty)
    """
    if not preset.allow_custom_instructions:
        return ""

    print("\n🔧 Custom instructions (optional):")

    if preset.custom_instructions_placeholder:
        print(f"提示：{preset.custom_instructions_placeholder}")
    else:
        print("提示：你可以告诉 agent 应该如何代表你")

    print()
    instructions = input("Instructions: ").strip()

    if instructions:
        print(
            f"\n✓ Custom instructions added: {instructions[:50]}{'...' if len(instructions) > 50 else ''}"
        )
    else:
        print("\n✓ No custom instructions")

    return instructions
