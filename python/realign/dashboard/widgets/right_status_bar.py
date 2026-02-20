"""Right-side status bar for the dashboard footer."""

from __future__ import annotations

from typing import Optional

from textual.content import Content
from textual.timer import Timer
from textual.widgets import Static


class RightStatusBar(Static):
    """A fixed-width, single-line status bar with an optional spinner."""

    _SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, *, width: int = 35, **kwargs) -> None:
        self._width = int(width)
        self._display: str = " " * self._width
        super().__init__("", **kwargs)
        self._text: str = ""
        self._spinning: bool = False
        self._frame_index: int = 0
        self._spinner_timer: Optional[Timer] = None
        self._clear_timer: Optional[Timer] = None

    def on_mount(self) -> None:
        self._spinner_timer = self.set_interval(0.12, self._tick_spinner)
        self._spinner_timer.pause()
        self._update_display()

    def clear_status(self) -> None:
        self.set_status("", spinning=False)

    def set_status(
        self,
        text: str,
        *,
        spinning: bool = True,
        variant: str = "active",
        clear_after_s: float | None = None,
    ) -> None:
        """Set status text. Optionally spin and auto-clear after a delay."""
        if self._clear_timer is not None:
            try:
                self._clear_timer.stop()
            except Exception:
                pass
            self._clear_timer = None

        self._text = text or ""
        self._spinning = bool(spinning and self._text.strip())
        self._frame_index = 0

        self.remove_class("active", "done", "error")
        if variant:
            self.add_class(variant)

        if self._spinner_timer is not None:
            if self._spinning:
                self._spinner_timer.resume()
            else:
                self._spinner_timer.pause()

        self._update_display()

        if clear_after_s is not None and clear_after_s > 0:
            self._clear_timer = self.set_timer(clear_after_s, self.clear_status)

    def _tick_spinner(self) -> None:
        if not self._spinning:
            if self._spinner_timer is not None:
                self._spinner_timer.pause()
            return
        self._frame_index = (self._frame_index + 1) % len(self._SPINNER_FRAMES)
        self._update_display()

    def _update_display(self) -> None:
        # Keep a small gap from the Footer, but avoid drawing a visible divider.
        separator = " " if self._width > 0 else ""
        avail = max(0, self._width - len(separator))

        if not self._text.strip():
            self._display = f"{separator}{' ' * avail}"
            self.refresh()
            return

        prefix = ""
        if self._spinning:
            prefix = f"{self._SPINNER_FRAMES[self._frame_index]} "

        content = f"{prefix}{self._text}".strip()
        if len(content) > avail:
            if avail <= 0:
                content = ""
            elif avail == 1:
                content = "…"
            else:
                content = content[: max(0, avail - 1)] + "…"
        self._display = f"{separator}{content.ljust(avail)}"
        self.refresh()

    def render(self) -> Content:
        return Content(self._display)
