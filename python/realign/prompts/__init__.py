"""Prompt preset system for chat agents."""

from .presets import (
    PromptPreset,
    get_all_presets,
    get_preset_by_id,
    get_preset_by_index,
    load_custom_presets,
    display_preset_menu,
    prompt_for_custom_instructions,
)

__all__ = [
    "PromptPreset",
    "get_all_presets",
    "get_preset_by_id",
    "get_preset_by_index",
    "load_custom_presets",
    "display_preset_menu",
    "prompt_for_custom_instructions",
]
