"""Create event modal for the dashboard."""

from __future__ import annotations

from typing import Iterable, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.worker import Worker, WorkerState
from textual.widgets import Button, Static

from ..widgets.copyable_text_area import CopyableTextArea


class CreateEventScreen(ModalScreen):
    """Modal to create a new event from selected sessions."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
    ]

    DEFAULT_CSS = """
    CreateEventScreen {
        align: center middle;
    }

    CreateEventScreen #create-event-root {
        width: 90%;
        height: 70%;
        padding: 1;
        background: $background;
        border: solid $accent;
    }

    CreateEventScreen #create-event-title {
        height: auto;
        margin-bottom: 1;
    }

    CreateEventScreen #event-title {
        width: 1fr;
        margin-bottom: 1;
    }

    CreateEventScreen #session-ids {
        height: 1fr;
        width: 1fr;
    }

    CreateEventScreen #buttons {
        height: 3;
        margin-top: 1;
        align: right middle;
    }
    """

    def __init__(self, session_ids: Iterable[str]) -> None:
        super().__init__()
        self.session_ids = list(session_ids)
        self._worker: Optional[Worker[Optional[str]]] = None

    def compose(self) -> ComposeResult:
        with Container(id="create-event-root"):
            yield Static(
                f"[bold]Create Event[/bold] (from {len(self.session_ids)} selected sessions)",
                id="create-event-title",
            )
            yield CopyableTextArea(
                "\n".join(self.session_ids),
                id="session-ids",
                read_only=True,
                soft_wrap=True,
                show_line_numbers=False,
            )
            with Horizontal(id="buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Generate", id="create", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#create", Button).focus()

    def action_close(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.app.pop_screen()
            return
        if event.button.id == "create":
            self._start_generate()

    def _start_generate(self) -> None:
        if self._worker is not None and self._worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        ):
            return

        if not self.session_ids:
            self.app.notify("No sessions selected", title="Create Event", severity="warning")
            return

        selector = ",".join(self.session_ids)

        def work() -> Optional[str]:
            from ...commands.watcher import watcher_event_generate_command
            from ...db import get_database

            db = get_database()
            before_ids = {e.id for e in db.list_events(limit=1000)}

            exit_code = watcher_event_generate_command(selector)
            if exit_code != 0:
                return None

            for e in db.list_events(limit=1000):
                if e.id not in before_ids:
                    return e.id
            newest = db.list_events(limit=1)
            return newest[0].id if newest else None

        self.query_one("#create", Button).disabled = True
        self.query_one("#cancel", Button).disabled = True
        self.app.notify(
            "Generating event (onecontext watcher event generate)...", title="Create Event", timeout=2
        )

        self._worker = self.run_worker(work, thread=True, exit_on_error=False)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if self._worker is None or event.worker is not self._worker:
            return

        if event.state == WorkerState.ERROR:
            self.query_one("#create", Button).disabled = False
            self.query_one("#cancel", Button).disabled = False
            err = self._worker.error
            self.app.notify(f"Create event failed: {err}", title="Create Event", severity="error")
            return

        if event.state != WorkerState.SUCCESS:
            return

        self.query_one("#create", Button).disabled = False
        self.query_one("#cancel", Button).disabled = False

        event_id = self._worker.result
        if not event_id:
            self.app.notify("Create event failed", title="Create Event", severity="error")
            return

        self.app.notify(
            f"Created event with {len(self.session_ids)} sessions",
            title="Event Created",
            timeout=3,
        )

        try:
            from ..widgets.sessions_table import SessionsTable

            self.app.query_one(SessionsTable).clear_selection()
        except Exception:
            pass

        def _after_close_open_event() -> None:
            try:
                from textual.widgets import TabbedContent

                self.app.query_one(TabbedContent).active = "events"
            except Exception:
                pass

            try:
                from .event_detail import EventDetailScreen

                self.app.push_screen(EventDetailScreen(event_id))
            except Exception:
                pass

        self.app.pop_screen()
        self.app.call_later(_after_close_open_event)
