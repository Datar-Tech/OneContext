"""
Adapter Registry

Handles registration and auto-detection of session adapters.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Tuple

from .base import SessionAdapter

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Registry for managing and selecting session adapters."""

    def __init__(self):
        self._adapters: Dict[str, Type[SessionAdapter]] = {}

    def register(self, adapter_class: Type[SessionAdapter]):
        """
        Register an adapter class.

        Args:
            adapter_class: The adapter class (not instance)
        """
        if not adapter_class.name:
            raise ValueError(f"Adapter class {adapter_class.__name__} must have a name")
        self._adapters[adapter_class.name] = adapter_class
        logger.debug(f"Registered adapter: {adapter_class.name}")

    def get_adapter(self, name: str, config: Optional[Dict] = None) -> Optional[SessionAdapter]:
        """
        Get an adapter instance by name.

        Args:
            name: Adapter name
            config: Optional configuration

        Returns:
            Adapter instance or None
        """
        adapter_class = self._adapters.get(name)
        if adapter_class:
            return adapter_class(config)
        return None

    def list_adapters(self) -> List[str]:
        """List all registered adapter names."""
        return list(self._adapters.keys())

    def discover_all_sessions(
        self, config: Optional[Dict] = None
    ) -> List[Tuple[Path, SessionAdapter]]:
        """
        Discover sessions from all registered adapters.

        Args:
            config: Optional configuration (passed to adapters)

        Returns:
            List of (session_path, adapter_instance) tuples
        """
        all_sessions = []
        for name, adapter_class in self._adapters.items():
            # Check if this adapter is enabled in config if needed
            # For now, we discover from all
            try:
                adapter = adapter_class(config)
                sessions = adapter.discover_sessions()
                for session_path in sessions:
                    all_sessions.append((session_path, adapter))
            except Exception as e:
                logger.warning(f"Error discovering sessions for adapter {name}: {e}")

        return all_sessions

    def auto_detect_adapter(
        self, session_file: Path, config: Optional[Dict] = None
    ) -> Optional[SessionAdapter]:
        """
        Detect which adapter should handle a given session file.

        Args:
            session_file: Path to the session file
            config: Optional configuration

        Returns:
            Matching Adapter instance or None
        """
        for name, adapter_class in self._adapters.items():
            try:
                adapter = adapter_class(config)
                if adapter.is_session_valid(session_file):
                    return adapter
            except Exception:
                continue
        return None


# Global registry instance
_global_registry = AdapterRegistry()


def get_adapter_registry() -> AdapterRegistry:
    """Get the global adapter registry."""
    return _global_registry
