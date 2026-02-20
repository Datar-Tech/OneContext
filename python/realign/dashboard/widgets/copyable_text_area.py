"""TextArea subclass that copies selected text to the system clipboard on mouse-up."""

from __future__ import annotations

from textual import events
from textual.widgets import TextArea

from ..clipboard import copy_text


class CopyableTextArea(TextArea):
    """A TextArea that auto-copies selected text to the system clipboard.

    In terminal TUI apps, the terminal's native Cmd+C cannot access
    Textual's internal text selection.  This subclass works around that
    by writing the selected text to the system clipboard as soon as the
    mouse selection ends, so Cmd+C (or any paste shortcut) will have the
    correct content ready.
    """

    async def _on_mouse_up(self, event: events.MouseUp) -> None:
        await super()._on_mouse_up(event)
        selected = self.selected_text
        if selected:
            copy_text(self.app, selected)
