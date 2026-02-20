"""Startup import-history suggestion modal."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Static

from .share_warning import BetterCheckbox


class ImportHistoryPromptScreen(ModalScreen):
    """Prompt users to import history when opening the dashboard."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
    ]

    DEFAULT_CSS = """
    ImportHistoryPromptScreen {
        align: center middle;
    }

    ImportHistoryPromptScreen #import-history-prompt-root {
        width: 100%;
        max-width: 76;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $background;
        border: solid $accent;
    }

    ImportHistoryPromptScreen #import-history-prompt-title {
        height: auto;
        margin-bottom: 1;
    }

    ImportHistoryPromptScreen #import-history-prompt-body {
        height: auto;
    }

    ImportHistoryPromptScreen #import-history-prompt-body Static {
        height: auto;
        margin-bottom: 1;
    }

    ImportHistoryPromptScreen #import-history-prompt-skip {
        width: 1fr;
        border: none;
        background: transparent;
        padding: 0;
    }

    ImportHistoryPromptScreen #import-history-prompt-skip:focus {
        border: none;
    }

    ImportHistoryPromptScreen #import-history-prompt-actions {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    ImportHistoryPromptScreen #import-history-prompt-actions Button {
        width: auto;
        min-width: 0;
        margin-left: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(id="import-history-prompt-root"):
            yield Static("[bold]Import History[/bold]", id="import-history-prompt-title")
            with Vertical(id="import-history-prompt-body"):
                yield Static(
                    "Would you like to import your workspace history now?\n\n"
                    "You can always open Import History later from the Config tab."
                )
                yield BetterCheckbox("Never Show Again", id="import-history-prompt-skip")
            with Horizontal(id="import-history-prompt-actions"):
                yield Button("NO", id="no")
                yield Button("GO", id="go", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#go", Button).focus()

    def action_close(self) -> None:
        self.dismiss({"go": False, "never_show_again": self._never_show_again()})

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "no":
            self.dismiss({"go": False, "never_show_again": self._never_show_again()})
            return
        if event.button.id == "go":
            self.dismiss({"go": True, "never_show_again": self._never_show_again()})
            return

    def _never_show_again(self) -> bool:
        return bool(self.query_one("#import-history-prompt-skip", Checkbox).value)
