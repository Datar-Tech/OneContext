"""Terminal backends for native terminal support."""

from .iterm2 import ITermBackend
from .kitty import KittyBackend

__all__ = ["ITermBackend", "KittyBackend"]
