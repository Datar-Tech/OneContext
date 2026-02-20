"""Archive confirmation modal for the dashboard."""

from __future__ import annotations

from typing import Optional

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Static

from .share_warning import BetterCheckbox


class ArchiveConfirmScreen(ModalScreen):
    """Modal that asks for confirmation before archiving an agent."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
        Binding("enter", "confirm", "Confirm", show=False),
    ]

    DEFAULT_CSS = """
    ArchiveConfirmScreen {
        align: center middle;
    }

    ArchiveConfirmScreen #archive-confirm-root {
        width: 100%;
        max-width: 64;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $background;
        border: solid $warning;
    }

    ArchiveConfirmScreen #archive-confirm-title {
        height: auto;
        margin-bottom: 1;
    }

    ArchiveConfirmScreen #archive-confirm-body {
        height: auto;
    }

    ArchiveConfirmScreen #archive-confirm-body Static {
        height: auto;
        margin-bottom: 1;
    }

    ArchiveConfirmScreen #archive-confirm-skip {
        width: 1fr;
        border: none;
        background: transparent;
        padding: 0;
    }

    ArchiveConfirmScreen #archive-confirm-skip:focus {
        border: none;
    }

    ArchiveConfirmScreen #archive-confirm-actions {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    ArchiveConfirmScreen #archive-confirm-actions Button {
        width: auto;
        min-width: 0;
        margin-left: 1;
    }
    """

    def __init__(
        self,
        *,
        agent_name: Optional[str] = None,
        terminal_count: int = 0,
    ) -> None:
        super().__init__()
        self.agent_name = agent_name or "Unknown"
        self.terminal_count = terminal_count

    def compose(self) -> ComposeResult:
        safe_name = escape(str(self.agent_name))
        title = f"[bold]Archive Agent[/bold] [dim]({safe_name})[/dim]"
        lines: list[str] = []
        if self.terminal_count > 0:
            lines.append(
                f"This will close {self.terminal_count} open terminal(s) "
                "and archive the agent."
            )
        else:
            lines.append("This will archive the agent.")
        lines.append("You can restore it later via Config > Show Archived.")
        lines.append("Are you sure you want to continue?")
        body = "\n".join(lines)

        with Container(id="archive-confirm-root"):
            yield Static(title, id="archive-confirm-title")
            with Vertical(id="archive-confirm-body"):
                yield Static(body)
                yield BetterCheckbox("Don't show this again", id="archive-confirm-skip")
            with Horizontal(id="archive-confirm-actions"):
                yield Button("Cancel", id="cancel")
                yield Button("Archive", id="confirm", variant="warning")

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
        skip = self.query_one("#archive-confirm-skip", Checkbox).value
        self.dismiss({"confirmed": True, "dont_show_again": bool(skip)})
