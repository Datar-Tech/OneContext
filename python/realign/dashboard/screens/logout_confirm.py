"""Exit / Logout confirmation modal for the dashboard."""

from __future__ import annotations

from typing import Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ExitConfirmScreen(ModalScreen):
    """Modal that confirms before exiting (or logging out of) Aline."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
        Binding("enter", "confirm", "Confirm", show=False),
    ]

    DEFAULT_CSS = """
    ExitConfirmScreen {
        align: center middle;
    }

    ExitConfirmScreen #exit-root {
        width: 100%;
        max-width: 60;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $background;
        border: solid $warning;
    }

    ExitConfirmScreen #exit-title {
        height: auto;
        margin-bottom: 1;
    }

    ExitConfirmScreen #exit-body {
        height: auto;
    }

    ExitConfirmScreen #exit-body Static {
        height: auto;
        margin-bottom: 1;
    }

    ExitConfirmScreen #exit-actions {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    ExitConfirmScreen #exit-actions Button {
        width: auto;
        min-width: 0;
        margin-left: 1;
    }
    """

    def __init__(self, mode: Literal["exit", "logout"] = "exit") -> None:
        super().__init__()
        self._mode = mode

    def compose(self) -> ComposeResult:
        if self._mode == "logout":
            title = "Logout"
            body = (
                "OneContext core services (session tracking, summaries, "
                "cloud sync) require login to function.\n\n"
                "Logging out will stop background services (watcher & worker). "
                "You can log back in at any time to resume."
            )
            confirm_label = "Logout"
        else:
            title = "Exit OneContext"
            body = (
                "Are you sure you want to exit OneContext?\n\n"
                "Your tmux sessions will [bold]not[/bold] be killed — "
                "all running processes inside tmux will continue as normal."
            )
            confirm_label = "Exit"

        with Container(id="exit-root"):
            yield Static(f"[bold]{title}[/bold]", id="exit-title")
            with Vertical(id="exit-body"):
                yield Static(body)
            with Horizontal(id="exit-actions"):
                yield Button("Cancel", id="cancel")
                yield Button(confirm_label, id="confirm", variant="warning")

    def on_mount(self) -> None:
        self.query_one("#confirm", Button).focus()

    def action_close(self) -> None:
        self.dismiss(False)

    def action_confirm(self) -> None:
        self.dismiss(True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(False)
            return
        if event.button.id == "confirm":
            self.dismiss(True)
            return
