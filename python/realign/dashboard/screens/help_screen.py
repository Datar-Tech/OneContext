"""Help screen modal for the dashboard."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class HelpScreen(ModalScreen):
    """Modal showing keyboard shortcuts and help information."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
        Binding("?", "close", "Close", show=False),
    ]

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }

    HelpScreen #help-root {
        width: 50;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $background;
        border: solid $accent;
    }

    HelpScreen #help-title {
        height: auto;
        margin-bottom: 1;
        text-style: bold;
        text-align: center;
    }

    HelpScreen .shortcut-section {
        height: auto;
        margin-bottom: 1;
    }

    HelpScreen .section-title {
        height: auto;
        color: $text-muted;
        margin-bottom: 0;
    }

    HelpScreen .shortcut-row {
        height: auto;
        padding: 0 1;
    }

    HelpScreen .shortcut-key {
        width: 14;
        height: auto;
        color: $accent;
    }

    HelpScreen .shortcut-desc {
        width: 1fr;
        height: auto;
    }

    HelpScreen #close-btn {
        margin-top: 1;
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(id="help-root"):
            yield Static("Keyboard Shortcuts", id="help-title")

            with Vertical(classes="shortcut-section"):
                yield Static("Navigation", classes="section-title")
                yield self._shortcut_row("Tab", "Next tab")
                yield self._shortcut_row("Shift+Tab", "Previous tab")
                yield self._shortcut_row("n", "Next page")
                yield self._shortcut_row("p", "Previous page")

            with Vertical(classes="shortcut-section"):
                yield Static("Actions", classes="section-title")
                yield self._shortcut_row("Enter", "Open selected item")
                yield self._shortcut_row("Space", "Toggle selection")
                yield self._shortcut_row("c", "Create event")
                yield self._shortcut_row("l", "Load context")
                yield self._shortcut_row("y", "Import share")
                yield self._shortcut_row("s", "Switch view")
                yield self._shortcut_row("r", "Refresh")

            with Vertical(classes="shortcut-section"):
                yield Static("General", classes="section-title")
                yield self._shortcut_row("?", "Show this help")
                yield self._shortcut_row("Ctrl+C x2", "Quit")

            yield Button("Close", id="close-btn", variant="primary")

    def _shortcut_row(self, key: str, description: str) -> Static:
        """Create a shortcut row with key and description."""
        return Static(f"[bold $accent]{key:<12}[/] {description}", classes="shortcut-row")

    def on_mount(self) -> None:
        self.query_one("#close-btn", Button).focus()

    def action_close(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-btn":
            self.app.pop_screen()
