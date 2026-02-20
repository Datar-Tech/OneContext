"""Share warning modal for the dashboard."""

from __future__ import annotations

from typing import Optional

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.content import Content
from textual.style import Style
from textual.widgets import Button, Checkbox, Static


class BetterCheckbox(Checkbox):
    """Checkbox with a clearer on/off glyph than the default 'X' toggle."""

    def _get_glyph(self) -> str:
        return "[✓]" if self.value else "[ ]"

    @property
    def _button(self) -> Content:  # type: ignore[override]
        button_style = self.get_visual_style("toggle--button")
        # Keep a subtle separation from the label without relying on color-only changes.
        return Content.assemble((self._get_glyph(), button_style), (" ", Style()))


class ShareWarningScreen(ModalScreen):
    """Modal that warns about share link privacy and consent before proceeding."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
        Binding("enter", "confirm", "Confirm", show=False),
    ]

    DEFAULT_CSS = """
    ShareWarningScreen {
        align: center middle;
    }

    ShareWarningScreen #share-warning-root {
        width: 100%;
        max-width: 84;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $background;
        border: solid $warning;
    }

    ShareWarningScreen #share-warning-title {
        height: auto;
        margin-bottom: 1;
    }

    ShareWarningScreen #share-warning-body {
        height: auto;
    }

    ShareWarningScreen #share-warning-body Static {
        height: auto;
        margin-bottom: 1;
    }

    ShareWarningScreen #share-warning-actions {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    ShareWarningScreen #share-warning-actions Button {
        width: auto;
        min-width: 0;
        margin-left: 1;
    }

    ShareWarningScreen #share-warning-skip {
        width: 1fr;
        border: none;
        background: transparent;
        padding: 0;
    }

    ShareWarningScreen #share-warning-skip:focus {
        border: none;
    }
    """

    def __init__(
        self,
        *,
        mode_label: str = "Share",
        agent_name: Optional[str] = None,
        expiry_days: int = 7,
        max_views: int = 100,
        requires_login: bool = True,
        processed_only: bool = True,
        recipients_can_contribute: bool = False,
    ) -> None:
        super().__init__()
        self.mode_label = mode_label
        self.agent_name = agent_name
        self.expiry_days = int(expiry_days)
        self.max_views = int(max_views)
        self.requires_login = bool(requires_login)
        self.processed_only = bool(processed_only)
        self.recipients_can_contribute = bool(recipients_can_contribute)

    def _title(self) -> str:
        action = escape(str(self.mode_label)) if self.mode_label else "Share"
        if self.agent_name:
            safe_name = escape(str(self.agent_name))
            return f"[bold]Share Link Notice[/bold] [dim]({action} • {safe_name})[/dim]"
        return f"[bold]Share Link Notice[/bold] [dim]({action})[/dim]"

    def _body_text(self) -> str:
        lines: list[str] = []
        lines.append(
            "• Please ensure you have the right to share this information and that it does not include "
            "sensitive or confidential data."
        )
        if self.requires_login:
            lines.append("• Anyone with the link (after signing in) may access and use this context.")
        else:
            lines.append("• Anyone with the link may access and use this context.")
        if self.processed_only:
            lines.append("• Uploaded data is processed context, not the full raw chat transcript.")
        lines.append("• Access may be logged for security and abuse prevention.")
        lines.append(f"• The share link will expire in {self.expiry_days} days.")
        return "\n".join(lines)

    def compose(self) -> ComposeResult:
        with Container(id="share-warning-root"):
            yield Static(self._title(), id="share-warning-title")
            with Vertical(id="share-warning-body"):
                yield Static(self._body_text())
                yield BetterCheckbox("Don't show this again", id="share-warning-skip")
            with Horizontal(id="share-warning-actions"):
                yield Button("Cancel", id="cancel")
                yield Button("Confirm", id="confirm", variant="primary")

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
        skip = self.query_one("#share-warning-skip", Checkbox).value
        self.dismiss({"confirmed": True, "dont_show_again": bool(skip)})
