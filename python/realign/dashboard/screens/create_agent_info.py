"""Create agent info modal for the dashboard."""

from __future__ import annotations

import re
import uuid
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static
from textual.worker import Worker, WorkerState

from ...logging_config import setup_logger

logger = setup_logger("realign.dashboard.screens.create_agent_info", "dashboard.log")


class CreateAgentInfoScreen(ModalScreen[Optional[dict]]):
    """Modal to create a new agent profile or import from a share link.

    Both options are shown together; the user picks one.
    Returns a dict with agent info on success, None on cancel.
    """

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
    ]

    DEFAULT_CSS = """
    CreateAgentInfoScreen {
        align: center middle;
    }

    CreateAgentInfoScreen #create-agent-info-root {
        width: 100%;
        max-width: 65;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $background;
        border: solid $accent;
    }

    CreateAgentInfoScreen #create-agent-info-title {
        height: auto;
        margin-bottom: 1;
        text-style: bold;
    }

    CreateAgentInfoScreen .section-label {
        height: auto;
        margin-top: 1;
        margin-bottom: 0;
        color: $text-muted;
    }

    CreateAgentInfoScreen Input {
        margin-top: 0;
    }

    CreateAgentInfoScreen #import-status {
        height: auto;
        margin-top: 1;
        color: $text-muted;
    }

    CreateAgentInfoScreen #add-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    CreateAgentInfoScreen #add-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._import_worker: Optional[Worker] = None
        from ...agent_names import generate_agent_name

        self._default_name: str = generate_agent_name()

    def compose(self) -> ComposeResult:
        with Container(id="create-agent-info-root"):
            yield Static("Add Context", id="create-agent-info-title")

            yield Label("Enter Name or Share Link", classes="section-label")
            yield Input(placeholder=self._default_name, id="agent-input")

            yield Static("", id="import-status")

            with Horizontal(id="add-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Add", id="add", variant="primary")

    def on_mount(self) -> None:
        agent_input = self.query_one("#agent-input", Input)
        agent_input.styles.border = ("round", "black")
        agent_input.focus()
        for btn in (self.query_one("#cancel", Button), self.query_one("#add", Button)):
            btn.styles.border = ("round", "black")
            btn.styles.text_align = "center"
            btn.styles.content_align = ("center", "middle")

    def action_close(self) -> None:
        self.dismiss(None)

    def _set_busy(self, busy: bool) -> None:
        self.query_one("#agent-input", Input).disabled = busy
        self.query_one("#add", Button).disabled = busy
        self.query_one("#cancel", Button).disabled = busy

    SHARE_LINK_PREFIX = "https://realign-server.vercel.app/share/"
    MAX_NAME_LENGTH = 40
    _NAME_PATTERN = re.compile(r"^[A-Za-z0-9 _\-]+$")

    def _is_share_link(self, value: str) -> bool:
        """Check if the input is a valid share link."""
        return value.startswith(self.SHARE_LINK_PREFIX)

    def _looks_like_url(self, value: str) -> bool:
        """Check if the input looks like a URL but is not a valid share link."""
        return value.startswith(("http://", "https://", "www."))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""

        if button_id == "cancel":
            self.dismiss(None)
            return

        if button_id == "add":
            await self._add_agent()
            return

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle enter key in input fields."""
        if event.input.id == "agent-input":
            await self._add_agent()

    async def _add_agent(self) -> None:
        """Create a new agent or import from link based on input content."""
        value = self.query_one("#agent-input", Input).value.strip()
        status = self.query_one("#import-status", Static)

        if self._is_share_link(value):
            status.update("")
            await self._import_agent(value)
        elif self._looks_like_url(value):
            status.update(f"Invalid link. Share links start with:\n{self.SHARE_LINK_PREFIX}")
        else:
            name = value or self._default_name
            if len(name) > self.MAX_NAME_LENGTH:
                status.update(f"Name too long (max {self.MAX_NAME_LENGTH} characters)")
            elif not self._NAME_PATTERN.match(name):
                status.update(
                    "Name can only contain letters, numbers, spaces, hyphens and underscores"
                )
            else:
                status.update("")
                await self._create_agent(name)

    async def _create_agent(self, name: str) -> None:
        """Create the agent profile."""
        try:
            from ...db import get_database

            agent_id = str(uuid.uuid4())

            db = get_database(read_only=False)
            record = db.get_or_create_agent_info(agent_id, name=name)

            self.dismiss(
                {
                    "id": record.id,
                    "name": record.name,
                    "title": record.title or "",
                    "description": record.description or "",
                }
            )
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            self.app.notify(f"Failed to create agent: {e}", severity="error")

    async def _import_agent(self, share_url: str) -> None:
        """Import an agent from a share link."""
        status = self.query_one("#import-status", Static)
        status.update("Importing...")
        self._set_busy(True)

        def do_import() -> dict:
            from ...commands.import_shares import import_agent_from_share

            return import_agent_from_share(share_url, password=None)

        self._import_worker = self.run_worker(do_import, thread=True, exit_on_error=False)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if self._import_worker is None or event.worker is not self._import_worker:
            return

        status = self.query_one("#import-status", Static)

        if event.state == WorkerState.ERROR:
            err = self._import_worker.error if self._import_worker else "Unknown error"
            status.update(f"Error: {err}")
            self._set_busy(False)
            return

        if event.state != WorkerState.SUCCESS:
            return

        result = self._import_worker.result if self._import_worker else {}
        if not result or not result.get("success"):
            error_msg = (result or {}).get("error", "Import failed")
            status.update(f"Error: {error_msg}")
            self._set_busy(False)
            return

        self.dismiss(
            {
                "id": result["agent_id"],
                "name": result["agent_name"],
                "title": result.get("agent_title", ""),
                "description": result.get("agent_description", ""),
                "imported": True,
                "sessions_imported": result.get("sessions_imported", 0),
                "limit_reached": result.get("limit_reached", False),
                "blocked_by_limit": result.get("blocked_by_limit", 0),
                "limit_value": result.get("limit_value", 1000),
            }
        )
