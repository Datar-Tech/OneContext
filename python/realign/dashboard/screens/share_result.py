"""Share result modal for the dashboard."""

from __future__ import annotations

from typing import Optional

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ..widgets.copyable_text_area import CopyableTextArea

from ..clipboard import copy_text


class ShareResultScreen(ModalScreen):
    """Modal that shows the generated Slack message and share link."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
    ]

    DEFAULT_CSS = """
    ShareResultScreen {
        align: center middle;
    }

    ShareResultScreen #share-result-root {
        width: 100%;
        max-width: 80;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $background;
        border: solid $accent;
    }

    ShareResultScreen #share-result-title {
        height: auto;
        margin-bottom: 1;
    }

    ShareResultScreen #share-result-body {
        height: auto;
    }

    ShareResultScreen .label {
        height: auto;
        margin-top: 1;
        margin-bottom: 0;
        color: $text-muted;
    }

    ShareResultScreen #slack-message {
        height: 15;
        width: 1fr;
        background: $surface;
        border: round $surface-lighten-2;
        padding: 0 1;
    }

    ShareResultScreen #share-link {
        height: 3;
        width: 1fr;
        background: $surface;
        border: round $surface-lighten-2;
        padding: 0 1;
    }

    ShareResultScreen #share-result-actions {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    ShareResultScreen #share-result-actions Button {
        width: auto;
        min-width: 0;
        margin-left: 1;
    }
    """

    def __init__(self, *, share_link: str, slack_message: Optional[str], agent_name: Optional[str] = None) -> None:
        super().__init__()
        self.share_link = share_link
        self.slack_message = slack_message
        self.agent_name = agent_name

    def _get_title(self) -> str:
        if self.agent_name:
            safe_name = escape(str(self.agent_name))
            return f"[bold]Context [italic]{safe_name}[/italic] is ready to share[/bold]"
        return "[bold]Context is ready to share[/bold]"

    def compose(self) -> ComposeResult:
        with Container(id="share-result-root"):
            yield Static(self._get_title(), id="share-result-title")
            with Vertical(id="share-result-body"):
                yield Static("Slack message", classes="label")
                yield CopyableTextArea(
                    self.slack_message or "",
                    id="slack-message",
                    read_only=True,
                    soft_wrap=True,
                    show_line_numbers=False,
                )
                yield Static("Share link", classes="label")
                yield CopyableTextArea(
                    self.share_link,
                    id="share-link",
                    read_only=True,
                    soft_wrap=True,
                    show_line_numbers=False,
                )
                with Horizontal(id="share-result-actions"):
                    yield Button("Cancel", id="cancel")
                    yield Button("Copy Link", id="copy-link")
                    yield Button("Copy Both", id="copy-both", variant="primary")

    def on_mount(self) -> None:
        self._copy_both()
        self.query_one("#copy-both", Button).focus()

    def action_close(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.app.pop_screen()
            return
        if event.button.id == "copy-link":
            self._copy_link()
            self.app.pop_screen()
            return
        if event.button.id == "copy-both":
            self._copy_both()
            self.app.pop_screen()
            return

    def _copy_link(self) -> None:
        ok = copy_text(self.app, self.share_link)
        if ok:
            self.app.notify("Share link copied to clipboard", title="Share", timeout=3)
        else:
            self.app.notify("Copy failed", title="Share", severity="warning", timeout=4)

    def _copy_both(self) -> None:
        if self.slack_message:
            text = f"{self.slack_message}\n\n{self.share_link}"
        else:
            text = self.share_link
        ok = copy_text(self.app, text)
        if ok:
            self.app.notify("Slack message + link copied", title="Share", timeout=3)
        else:
            self.app.notify("Copy failed", title="Share", severity="warning", timeout=4)
