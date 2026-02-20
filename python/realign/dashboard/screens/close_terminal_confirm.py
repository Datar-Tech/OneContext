"""Close terminal confirmation modal for the dashboard."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Static

from .share_warning import BetterCheckbox


class CloseTerminalConfirmScreen(ModalScreen):
    """Modal that asks for confirmation before closing a terminal."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
        Binding("enter", "confirm", "Confirm", show=False),
    ]

    DEFAULT_CSS = """
    CloseTerminalConfirmScreen {
        align: center middle;
    }

    CloseTerminalConfirmScreen #close-term-root {
        width: 100%;
        max-width: 56;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $background;
        border: solid $warning;
    }

    CloseTerminalConfirmScreen #close-term-title {
        height: auto;
        margin-bottom: 1;
    }

    CloseTerminalConfirmScreen #close-term-body {
        height: auto;
    }

    CloseTerminalConfirmScreen #close-term-body Static {
        height: auto;
        margin-bottom: 1;
    }

    CloseTerminalConfirmScreen #close-term-skip {
        width: 1fr;
        border: none;
        background: transparent;
        padding: 0;
    }

    CloseTerminalConfirmScreen #close-term-skip:focus {
        border: none;
    }

    CloseTerminalConfirmScreen #close-term-actions {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    CloseTerminalConfirmScreen #close-term-actions Button {
        width: auto;
        min-width: 0;
        margin-left: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(id="close-term-root"):
            yield Static("[bold]Close Terminal[/bold]", id="close-term-title")
            with Vertical(id="close-term-body"):
                yield Static("Are you sure you want to close this terminal?")
                yield BetterCheckbox("Don't show this again", id="close-term-skip")
            with Horizontal(id="close-term-actions"):
                yield Button("Cancel", id="cancel")
                yield Button("Close", id="confirm", variant="warning")

    def on_mount(self) -> None:
        self.query_one("#confirm", Button).focus()

    def action_close(self) -> None:
        self.dismiss({"confirmed": False, "dont_show_again": False})

    def action_confirm(self) -> None:
        self._confirm()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss({"confirmed": False, "dont_show_again": False})
            return
        if event.button.id == "confirm":
            self._confirm()
            return

    def _confirm(self) -> None:
        skip = self.query_one("#close-term-skip", Checkbox).value
        self.dismiss({"confirmed": True, "dont_show_again": bool(skip)})
