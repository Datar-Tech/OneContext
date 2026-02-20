"""
Session Adapters Module

Provides unified adapters for different AI coding CLI tools.
Each adapter encapsulates session discovery, project path extraction,
and turn detection logic for a specific CLI.
"""

from .base import SessionAdapter
from .registry import AdapterRegistry, get_adapter_registry
from .claude import ClaudeAdapter
from .codex import CodexAdapter
from .gemini import GeminiAdapter

# Register all built-in adapters
registry = get_adapter_registry()
registry.register(ClaudeAdapter)
registry.register(CodexAdapter)
registry.register(GeminiAdapter)

__all__ = [
    "SessionAdapter",
    "AdapterRegistry",
    "get_adapter_registry",
    "ClaudeAdapter",
    "CodexAdapter",
    "GeminiAdapter",
]
