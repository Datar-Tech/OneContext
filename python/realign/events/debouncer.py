"""Debounce utility for delayed execution."""

from threading import Timer
from typing import Dict, Callable
import logging

logger = logging.getLogger(__name__)


class Debouncer:
    """Memory-based debouncer using threading Timer."""

    def __init__(self):
        self._timers: Dict[str, Timer] = {}
        self._callbacks: Dict[str, Callable] = {}

    def schedule(self, key: str, delay_seconds: float, callback: Callable) -> None:
        """
        Schedule a debounced callback.

        If a callback with the same key is already scheduled, it will be
        cancelled and replaced with the new one.

        Args:
            key: Unique identifier for this callback
            delay_seconds: Delay before executing the callback
            callback: Function to execute after the delay
        """
        # Cancel existing timer for this key
        if key in self._timers:
            self._timers[key].cancel()

        # Create new timer
        timer = Timer(delay_seconds, lambda: self._execute(key))
        self._timers[key] = timer
        self._callbacks[key] = callback
        timer.start()
        logger.debug(f"Scheduled debounced callback for key={key}, delay={delay_seconds}s")

    def _execute(self, key: str) -> None:
        """Execute callback and cleanup."""
        if key in self._callbacks:
            try:
                self._callbacks[key]()
            except Exception as e:
                logger.error(f"Error executing debounced callback for key={key}: {e}")
            finally:
                self._cleanup(key)

    def _cleanup(self, key: str) -> None:
        """Remove timer and callback for the given key."""
        self._timers.pop(key, None)
        self._callbacks.pop(key, None)

    def flush(self, key: str) -> None:
        """Immediately execute the callback for the given key."""
        if key in self._timers:
            self._timers[key].cancel()
        if key in self._callbacks:
            try:
                self._callbacks[key]()
            except Exception as e:
                logger.error(f"Error flushing debounced callback for key={key}: {e}")
        self._cleanup(key)

    def flush_all(self) -> None:
        """Immediately execute all pending callbacks."""
        keys = list(self._timers.keys())
        for key in keys:
            self.flush(key)

    def cancel(self, key: str) -> None:
        """Cancel the callback for the given key without executing it."""
        if key in self._timers:
            self._timers[key].cancel()
        self._cleanup(key)

    def cancel_all(self) -> None:
        """Cancel all pending callbacks without executing them."""
        for timer in self._timers.values():
            timer.cancel()
        self._timers.clear()
        self._callbacks.clear()

    def pending_count(self) -> int:
        """Return the number of pending callbacks."""
        return len(self._timers)

    def is_pending(self, key: str) -> bool:
        """Check if a callback is pending for the given key."""
        return key in self._timers
