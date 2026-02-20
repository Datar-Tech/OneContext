"""Share import modal for the dashboard."""

from __future__ import annotations

import contextlib
import io
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.worker import Worker, WorkerState
from textual.widgets import Button, Checkbox, Input, Static


class ShareImportScreen(ModalScreen):
    """Modal that imports a shared conversation into the local DB."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
    ]

    DEFAULT_CSS = """
    ShareImportScreen {
        align: center middle;
    }

    ShareImportScreen #share-import-root {
        width: 90%;
        height: 60%;
        padding: 1;
        background: $background;
        border: solid $accent;
    }

    ShareImportScreen #share-import-title {
        height: auto;
        margin-bottom: 1;
    }

    ShareImportScreen .row {
        height: auto;
        margin-bottom: 1;
    }

    ShareImportScreen Input {
        width: 1fr;
    }

    ShareImportScreen #share-import-status {
        height: 3;
        margin-top: 1;
        color: $text-muted;
    }

    ShareImportScreen #share-import-actions Button {
        width: 12;
        margin-left: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._worker: Optional[Worker[dict]] = None

    def compose(self) -> ComposeResult:
        with Container(id="share-import-root"):
            yield Static("[bold]Share Import[/bold]", id="share-import-title")
            with Vertical():
                with Horizontal(classes="row"):
                    yield Input(
                        id="share-url",
                        placeholder="Paste share URL (e.g. https://.../share/abc123)",
                    )
                with Horizontal(classes="row"):
                    yield Input(
                        id="share-password", placeholder="Password (optional)", password=True
                    )
                with Horizontal(classes="row"):
                    yield Checkbox("Force re-import (override duplicates)", id="share-force")
                with Horizontal(id="share-import-actions", classes="row"):
                    yield Button("Import", id="import", variant="primary")
                    yield Button("Cancel", id="cancel")
            yield Static("", id="share-import-status")

    def on_mount(self) -> None:
        self.query_one("#share-url", Input).focus()

    def action_close(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.app.pop_screen()
            return
        if event.button.id == "import":
            self._start_import()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "share-url":
            self._start_import()

    def _set_busy(self, busy: bool) -> None:
        self.query_one("#share-url", Input).disabled = busy
        self.query_one("#share-password", Input).disabled = busy
        self.query_one("#share-force", Checkbox).disabled = busy
        self.query_one("#import", Button).disabled = busy
        self.query_one("#cancel", Button).disabled = busy

    def _start_import(self) -> None:
        if self._worker is not None and self._worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        ):
            return

        share_url = self.query_one("#share-url", Input).value.strip()
        password = self.query_one("#share-password", Input).value.strip()
        force = self.query_one("#share-force", Checkbox).value

        if not share_url:
            self.app.notify("Please paste a share URL", title="Share Import", severity="warning")
            return

        def work() -> dict:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                from ...commands import import_shares

                exit_code = import_shares.import_share_command(
                    share_url=share_url,
                    password=password or None,
                    force=bool(force),
                    non_interactive=True,
                )

            return {
                "exit_code": exit_code,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
            }

        status = self.query_one("#share-import-status", Static)
        status.update("[dim]Importing…[/dim]")
        self._set_busy(True)
        self._worker = self.run_worker(work, thread=True, exit_on_error=False)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if self._worker is None or event.worker is not self._worker:
            return

        status = self.query_one("#share-import-status", Static)

        if event.state == WorkerState.ERROR:
            self._set_busy(False)
            err = self._worker.error
            status.update(f"[red]Import failed:[/red] {err}")
            self.app.notify(f"Share import failed: {err}", title="Share Import", severity="error")
            return

        if event.state != WorkerState.SUCCESS:
            return

        self._set_busy(False)
        result = self._worker.result or {}
        raw_exit_code = result.get("exit_code", None)
        exit_code = 1 if raw_exit_code is None else int(raw_exit_code)

        if exit_code == 0:
            status.update("[green]Imported successfully.[/green]")
            self.app.notify("Share import completed", title="Share Import", timeout=3)
            try:
                from ..widgets.events_table import EventsTable

                self.app.query_one(EventsTable).refresh_data()
            except Exception:
                pass
            self.app.pop_screen()
            return

        stderr_text = (result.get("stderr") or "").strip()
        message = "Share import failed"
        if stderr_text:
            message = f"{message}: {stderr_text}"
        status.update(f"[red]{message}[/red]")
        self.app.notify(message, title="Share Import", severity="error", timeout=6)
