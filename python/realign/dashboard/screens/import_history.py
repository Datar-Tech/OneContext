"""Import history modal for the dashboard config support section."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.worker import Worker, WorkerState
from textual.widgets import Button, DataTable, Input, Select, Static, TabbedContent
from textual.widgets._select import SelectCurrent, SelectOverlay


class ImportSessionsTable(DataTable):
    """Session results table with row multi-select behavior."""

    BINDINGS = [
        Binding("space", "toggle_mark", "Toggle", show=False),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.owner: Optional["ImportHistoryScreen"] = None

    def action_toggle_mark(self) -> None:
        if self.owner is not None:
            self.owner.toggle_selection_at_cursor()

    async def _on_click(self, event: events.Click) -> None:
        style = event.style
        meta = style.meta if style else {}
        row_index = meta.get("row")
        is_data_row = isinstance(row_index, int) and row_index >= 0

        await super()._on_click(event)

        if self.owner is None or not is_data_row:
            return

        self.owner.toggle_selection_at_row(row_index)


class ContextSelect(Select):
    """Select that dismisses reliably and syncs external triangle state."""

    def _ensure_overlay_round_style(self) -> None:
        with contextlib.suppress(Exception):
            overlay = self.query_one(SelectOverlay)
            overlay.styles.border = ("round", "#6e6e6e")

    class Overlay(SelectOverlay):
        """Select overlay that dismisses on any non-option click."""

        def on_mount(self) -> None:
            # Ensure the expanded dropdown panel uses rounded borders as well.
            self.styles.border = ("round", "#6e6e6e")

        async def _on_click(self, event: events.Click) -> None:
            style = event.style
            meta = style.meta if style else {}
            clicked_option = meta.get("option")

            if clicked_option is None:
                self.post_message(self.Dismiss())
                event.stop()
                return

            await super()._on_click(event)

    def compose(self) -> ComposeResult:
        """Compose with custom overlay that can dismiss on outside click."""
        yield SelectCurrent(self.prompt)
        yield self.Overlay(type_to_search=self._type_to_search).data_bind(compact=Select.compact)

    async def _on_click(self, event: events.Click) -> None:
        style = event.style
        meta = style.meta if style else {}
        clicked_option = meta.get("option")

        # When the overlay is open, clicking outside options should dismiss it.
        if bool(self.expanded) and clicked_option is None:
            self.expanded = False
            event.stop()
            return

        await super()._on_click(event)

    def _watch_expanded(self, expanded: bool) -> None:
        """Keep the external expand button label in sync with Select overlay state."""
        super_watch = getattr(super(), "_watch_expanded", None)
        if callable(super_watch):
            try:
                super_watch(expanded)
            except Exception:
                pass
        if expanded:
            self._ensure_overlay_round_style()
        try:
            screen = self.screen
            updater = getattr(screen, "_update_context_expand_button", None)
            if callable(updater):
                updater(bool(expanded))
        except Exception:
            pass


class ImportHistoryScreen(ModalScreen):
    """Simple import history window with browse + address input."""

    ACCESS_FREE = "free"
    ACCESS_PRO = "pro"
    ACCESS_PRO_PLUS = "pro_plus"

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
    ]
    SESSION_LIST_HEIGHT_RATIO = 0.70

    DEFAULT_CSS = """
    ImportHistoryScreen {
        align: center middle;
    }

    ImportHistoryScreen #import-history-root {
        width: 95%;
        height: 78%;
        padding: 1;
        background: $background;
        border: solid $accent;
    }

    ImportHistoryScreen #import-history-title {
        width: 1fr;
        height: auto;
    }

    ImportHistoryScreen #title-row {
        height: auto;
        align: left middle;
        margin-bottom: 1;
    }

    ImportHistoryScreen #import-history-main {
        height: 1fr;
    }

    ImportHistoryScreen #history-search-status {
        height: auto;
        margin-bottom: 1;
        color: $text-muted;
    }

    ImportHistoryScreen .row {
        height: auto;
        align: left middle;
    }

    ImportHistoryScreen Input {
        width: 1fr;
    }

    ImportHistoryScreen #history-path {
        border: round $surface-lighten-2;
    }

    ImportHistoryScreen #history-path:focus {
        border: round $accent;
    }

    ImportHistoryScreen #browse-btn {
        width: 10;
        margin-left: 1;
    }

    ImportHistoryScreen #back-btn {
        width: 12;
        margin-left: 1;
    }

    ImportHistoryScreen #as-label {
        width: auto;
        height: 3;
        margin-right: 1;
        content-align: left middle;
    }

    ImportHistoryScreen #context-select {
        width: 1fr;
        height: 3;
    }

    ImportHistoryScreen #context-select .down-arrow {
        display: none;
    }

    ImportHistoryScreen #context-select .up-arrow {
        display: none;
    }

    ImportHistoryScreen #context-select > SelectOverlay {
        border: round $surface-lighten-2;
    }

    ImportHistoryScreen #context-expand-btn {
        width: 3;
        min-width: 3;
        margin-left: 1;
    }

    ImportHistoryScreen #import-btn {
        width: 12;
    }

    ImportHistoryScreen #context-import-row {
        height: auto;
        align: left middle;
        margin-top: 1;
        margin-bottom: 1;
    }

    ImportHistoryScreen #bottom-actions-row {
        margin-bottom: 0;
    }

    ImportHistoryScreen #bottom-actions-spacer {
        width: 1fr;
    }

    ImportHistoryScreen #session-results {
        height: auto;
        min-height: 8;
        margin-bottom: 0;
        overflow-x: scroll;
        overflow-y: auto;
    }

    ImportHistoryScreen #import-progress-status {
        width: 1fr;
        height: auto;
        content-align: right middle;
        color: $text-muted;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._default_path = str(os.path.expanduser("~"))
        self._worker: Optional[Worker[dict[str, object]]] = None  # discovery worker
        self._import_worker: Optional[Worker[dict[str, object]]] = None
        self._result_row_keys: list[str] = []
        self._selected_row_keys: set[str] = set()
        self._result_items_by_key: dict[str, dict[str, object]] = {}
        self._context_label_by_value: dict[str, str] = {}
        self._new_context_name: str = ""
        self._access_level: str = self.ACCESS_FREE
        self._access_level_cached_at: float = 0.0
        self._importing_dots_timer: Optional[Timer] = None
        self._importing_dots_frame: int = 0
        self._context_last_dismissed_at: float = 0.0
        self._context_prev_expanded: bool = False
        self._auto_browse_started: bool = False
        self._picker_open: bool = False

    def compose(self) -> ComposeResult:
        with Container(id="import-history-root"):
            with Horizontal(classes="row", id="title-row"):
                yield Static(
                    "[bold]Import History from Workspace...[/bold]",
                    id="import-history-title",
                )
                yield Button("Go Back", id="back-btn", variant="primary")
            with Vertical(id="import-history-main"):
                with Horizontal(classes="row"):
                    yield Input(
                        "", id="history-path", placeholder="Use Browse to select a workspace"
                    )
                    yield Button("Browse", id="browse-btn")
                yield Static(
                    "",
                    id="history-search-status",
                )
                yield ImportSessionsTable(id="session-results")
            with Horizontal(classes="row", id="context-import-row"):
                yield Static("As", id="as-label")
                yield ContextSelect(
                    [("Loading contexts...", "__loading__")],
                    id="context-select",
                    prompt="Context",
                    allow_blank=False,
                    disabled=True,
                )
                yield Button("▼", id="context-expand-btn", variant="default", disabled=True)
            with Horizontal(classes="row", id="bottom-actions-row"):
                yield Button("Clear Selection", id="toggle-select-btn", disabled=True)
                yield Static("", id="bottom-actions-spacer")
                yield Button("Import", id="import-btn", variant="primary", disabled=True)
            yield Static("", id="import-progress-status")

    def on_mount(self) -> None:
        history_input = self.query_one("#history-path", Input)
        history_input.styles.border = ("round", "#6e6e6e")
        history_input.focus()

        context_select = self.query_one("#context-select", Select)
        with contextlib.suppress(Exception):
            select_current = context_select.query_one(SelectCurrent)
            select_current.styles.border = ("round", "#6e6e6e")
            select_current.styles.height = 3

        table = self.query_one("#session-results", ImportSessionsTable)
        table.owner = self
        table.add_column("✓", key="sel", width=2)
        table.add_column("Source", key="source", width=10)
        table.add_column("Session", key="session", width=40)
        table.add_column("Updated", key="updated", width=16)
        table.add_column("File", key="file", width=260)
        table.cursor_type = "row"
        table.styles.overflow_x = "scroll"
        table.styles.overflow_y = "auto"
        table.show_vertical_scrollbar = True
        table.show_horizontal_scrollbar = True
        self._sync_session_list_height()
        self._reload_context_options()
        self._update_context_expand_button()
        self._update_import_button()

        # Automatically open the browse dialog on mount in interactive runs.
        if self._is_auto_browse_enabled():
            self._auto_browse_on_mount()

    @staticmethod
    def _is_auto_browse_enabled() -> bool:
        """Whether auto-open browse should run on screen mount."""
        raw = str(os.environ.get("ALINE_IMPORT_HISTORY_AUTO_BROWSE", "")).strip().lower()
        if raw:
            return raw not in {"0", "false", "no", "off"}
        # pytest sets this env var per-test; avoid popping native picker dialogs while testing.
        return not bool(os.environ.get("PYTEST_CURRENT_TEST"))

    def _auto_browse_on_mount(self) -> None:
        """Trigger the browse dialog automatically when the screen opens."""
        if not self._is_auto_browse_enabled():
            return
        if self._auto_browse_started:
            return
        self._auto_browse_started = True

        async def _do_auto_browse() -> None:
            if not bool(getattr(self, "is_mounted", False)):
                return
            selected = await self._select_path()
            if (
                selected
                and os.path.isdir(selected)
                and bool(getattr(self, "is_mounted", False))
            ):
                self.query_one("#history-path", Input).value = selected
                self._start_discovery()

        self.call_later(_do_auto_browse)

    def on_resize(self) -> None:
        self._sync_session_list_height()

    def _sync_session_list_height(self) -> None:
        """Keep session list height fixed as a ratio of the main window."""
        try:
            table = self.query_one("#session-results", DataTable)
        except Exception:
            return

        percent = int(self.SESSION_LIST_HEIGHT_RATIO * 100)
        table.styles.height = f"{percent}%"

    def action_close(self) -> None:
        self.app.pop_screen()

    def _is_discovery_busy(self) -> bool:
        return self._worker is not None and self._worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        )

    def _set_busy(self, busy: bool) -> None:
        self.query_one("#history-path", Input).disabled = busy
        self.query_one("#browse-btn", Button).disabled = busy

    @staticmethod
    def _checkbox_cell(selected: bool) -> str:
        return "[bold green]✓[/bold green]" if selected else ""

    def _update_select_toggle_button(self) -> None:
        btn = self.query_one("#toggle-select-btn", Button)
        total = len(self._result_row_keys)
        selected = len(self._selected_row_keys)
        if total <= 0:
            btn.disabled = True
            btn.label = "Clear Selection"
            self._update_import_button()
            return
        btn.disabled = False
        btn.label = "Clear Selection" if selected > 0 else "Select All"
        self._update_import_button()

    def _update_import_button(self) -> None:
        import_btn = self.query_one("#import-btn", Button)
        select = self.query_one("#context-select", Select)

        has_rows = len(self._result_row_keys) > 0
        has_selection = len(self._selected_row_keys) > 0
        has_context = select.value is not Select.BLANK
        import_busy = self._import_worker is not None and self._import_worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        )
        import_btn.disabled = not (has_rows and has_selection and has_context) or import_busy

    def _update_context_expand_button(self, expanded: bool | None = None) -> None:
        btn = self.query_one("#context-expand-btn", Button)
        select = self.query_one("#context-select", Select)
        if select.disabled:
            btn.disabled = True
            btn.label = "▼"
            return
        btn.disabled = False
        is_expanded = bool(select.expanded if expanded is None else expanded)
        btn.label = "▲" if is_expanded else "▼"
        if self._context_prev_expanded and not is_expanded:
            self._context_last_dismissed_at = time.monotonic()
        self._context_prev_expanded = is_expanded

    def _reload_context_options(self, *, prefer_value: str | None = None) -> None:
        from ...agent_names import generate_agent_name
        from ...db import get_database

        select = self.query_one("#context-select", Select)
        current_value = prefer_value
        if current_value is None and select.value is not Select.BLANK:
            current_value = str(select.value)

        self._new_context_name = generate_agent_name()
        new_context_value = f"__new__:{self._new_context_name}"
        options: list[tuple[str, str]] = [
            (f"New Context ({self._new_context_name})", new_context_value),
        ]

        db = get_database(read_only=True, connect_timeout_seconds=2.0)
        try:
            infos = db.list_agent_info(include_invisible=True)
        finally:
            try:
                db.close()
            except Exception:
                pass

        infos.sort(key=lambda info: (0 if (info.visibility or "visible") == "visible" else 1))

        for info in infos:
            is_archived = (info.visibility or "visible") != "visible"
            label = info.name if not is_archived else f"{info.name} (Archived)"
            options.append((label, info.id))

        self._context_label_by_value = {str(value): str(label) for label, value in options}
        select.set_options(options)
        select.disabled = False

        valid_values = {str(value) for _, value in options}
        target_value = current_value if current_value in valid_values else new_context_value
        select.value = target_value
        self._update_context_expand_button()
        self._update_import_button()

    def _refresh_selection_cells(self) -> None:
        table = self.query_one("#session-results", DataTable)
        if table.row_count == 0:
            self._update_select_toggle_button()
            return
        for row_index in range(table.row_count):
            try:
                row_key = str(table.coordinate_to_cell_key((row_index, 0))[0].value)
            except Exception:
                continue
            if not row_key:
                continue
            try:
                table.update_cell(
                    row_key,
                    "sel",
                    self._checkbox_cell(row_key in self._selected_row_keys),
                )
            except Exception:
                continue
        self._update_select_toggle_button()

    def toggle_selection_at_cursor(self) -> None:
        table = self.query_one("#session-results", DataTable)
        if table.row_count == 0:
            return
        try:
            row_key = str(table.coordinate_to_cell_key(table.cursor_coordinate)[0].value)
        except Exception:
            return
        if not row_key:
            return
        if row_key in self._selected_row_keys:
            self._selected_row_keys.remove(row_key)
        else:
            self._selected_row_keys.add(row_key)
        self._refresh_selection_cells()

    def toggle_selection_at_row(self, row_index: int) -> None:
        table = self.query_one("#session-results", DataTable)
        if table.row_count == 0:
            return
        if row_index < 0 or row_index >= table.row_count:
            return
        try:
            row_key = str(table.coordinate_to_cell_key((row_index, 0))[0].value)
        except Exception:
            return
        if not row_key:
            return
        if row_key in self._selected_row_keys:
            self._selected_row_keys.remove(row_key)
        else:
            self._selected_row_keys.add(row_key)
        self._refresh_selection_cells()

    def _toggle_select_all_none(self) -> None:
        if not self._result_row_keys:
            self._selected_row_keys.clear()
            self._refresh_selection_cells()
            return

        # Button semantics:
        # - Any selection => "Clear Selection"
        # - No selection => "Select All"
        if self._selected_row_keys:
            self._selected_row_keys.clear()
        else:
            self._selected_row_keys = set(self._result_row_keys)
        self._refresh_selection_cells()

    def _selected_session_paths(self) -> list[Path]:
        selected_paths: list[Path] = []
        for key in self._result_row_keys:
            if key not in self._selected_row_keys:
                continue
            item = self._result_items_by_key.get(key)
            if not item:
                continue
            path_str = str(item.get("path") or "").strip()
            if not path_str:
                continue
            selected_paths.append(Path(path_str).expanduser())
        return selected_paths

    def _start_import(self) -> None:
        if self._import_worker is not None and self._import_worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        ):
            return

        select = self.query_one("#context-select", Select)
        if select.value is Select.BLANK:
            self.app.notify(
                "Please choose a context first.", title="Import history", severity="warning"
            )
            return

        selected_paths = self._selected_session_paths()
        if not selected_paths:
            self.app.notify(
                "No sessions selected. Please select at least one session.",
                title="Import history",
                severity="warning",
            )
            return

        context_value = str(select.value)
        context_label = self._context_label_by_value.get(context_value, context_value)

        self.query_one("#history-search-status", Static).update(
            f"[dim]Importing {len(selected_paths)} session(s) into {context_label} ...[/dim]"
        )
        self._start_importing_dots()
        self.query_one("#import-btn", Button).disabled = True

        def work() -> dict[str, object]:
            return self._import_sessions_to_context(
                session_paths=selected_paths,
                context_value=context_value,
            )

        self._import_worker = self.run_worker(work, thread=True, exit_on_error=False)

    @staticmethod
    def _importing_text(frame: int) -> str:
        dots = "." * ((max(0, frame) % 3) + 1)
        return f"importing{dots}"

    def _tick_importing_dots(self) -> None:
        label = self.query_one("#import-progress-status", Static)
        label.update(f"[dim]{self._importing_text(self._importing_dots_frame)}[/dim]")
        self._importing_dots_frame += 1

    def _start_importing_dots(self) -> None:
        self._importing_dots_frame = 0
        self._tick_importing_dots()
        if self._importing_dots_timer is None:
            self._importing_dots_timer = self.set_interval(0.45, self._tick_importing_dots)
            return
        self._importing_dots_timer.resume()

    def _stop_importing_dots(self) -> None:
        if self._importing_dots_timer is not None:
            self._importing_dots_timer.pause()
        self.query_one("#import-progress-status", Static).update("")

    @staticmethod
    def _import_sessions_to_context(
        *,
        session_paths: list[Path],
        context_value: str,
    ) -> dict[str, object]:
        from ...commands import watcher as watcher_cmd
        from ...config import ReAlignConfig
        from ...db import (
            AgentSessionLimitExceededError,
            MAX_SESSION_NUM,
            get_database,
        )
        from ...events.session_summarizer import force_update_session_summary

        db = get_database(read_only=False)
        config = ReAlignConfig.load()

        created_new = False
        context_not_found = False
        context_error_message = ""
        context_visibility = "visible"
        if context_value.startswith("__new__:"):
            context_name = context_value.split(":", 1)[1].strip() or "New Context"
            agent_id = str(uuid.uuid4())
            info = db.get_or_create_agent_info(agent_id, name=context_name)
            created_new = True
        else:
            agent_id = context_value
            info = db.get_agent_info(agent_id)
            if info is None:
                context_not_found = True
                context_error_message = "Selected context no longer exists. Please reopen Import History and choose again."
                return {
                    "created_new_context": False,
                    "context_name": agent_id or "Context",
                    "agent_id": agent_id,
                    "selected_count": len(session_paths),
                    "imported_count": 0,
                    "failed_count": len(session_paths),
                    "summary_failed_count": 0,
                    "missing_count": 0,
                    "limit_reached": False,
                    "blocked_by_limit": 0,
                    "limit_value": MAX_SESSION_NUM,
                    "context_not_found": context_not_found,
                    "error_message": context_error_message,
                    "context_visibility": context_visibility,
                }

        context_name = getattr(info, "name", "") or "Context"
        context_visibility = str(getattr(info, "visibility", "visible") or "visible")

        imported = 0
        failed = 0
        summary_failed = 0
        missing = 0
        blocked_by_limit = 0
        limit_reached = False

        for index, session_file in enumerate(session_paths):
            if not session_file.exists() or not session_file.is_file():
                missing += 1
                continue

            session_id = session_file.stem
            if (
                not db.is_session_linked_to_agent(agent_id, session_id)
                and db.get_agent_session_count(agent_id) >= MAX_SESSION_NUM
            ):
                limit_reached = True
                blocked_by_limit = len(session_paths) - index
                failed += blocked_by_limit
                break

            try:
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    ok = watcher_cmd._import_single_session(
                        session_file=session_file,
                        config=config,
                        force=False,
                        show_header=False,
                        regenerate=False,
                        queue=False,
                    )
            except Exception:
                ok = False

            if not ok:
                failed += 1
                continue

            try:
                linked = False
                try:
                    linked = bool(db.link_session_to_agent(agent_id, session_id))
                except AttributeError:
                    linked = False
                if not linked:
                    db.update_session_agent_id(session_id, agent_id)
            except AgentSessionLimitExceededError:
                limit_reached = True
                blocked_by_limit = len(session_paths) - index
                failed += blocked_by_limit
                break
            except Exception:
                failed += 1
                continue

            try:
                if not db.is_session_linked_to_agent(agent_id, session_id):
                    failed += 1
                    continue
            except Exception:
                failed += 1
                continue

            try:
                force_update_session_summary(db, session_id)
            except Exception:
                summary_failed += 1

            imported += 1

        return {
            "created_new_context": created_new,
            "context_name": context_name,
            "agent_id": agent_id,
            "selected_count": len(session_paths),
            "imported_count": imported,
            "failed_count": failed,
            "summary_failed_count": summary_failed,
            "missing_count": missing,
            "limit_reached": limit_reached,
            "blocked_by_limit": blocked_by_limit,
            "limit_value": MAX_SESSION_NUM,
            "context_not_found": context_not_found,
            "error_message": context_error_message,
            "context_visibility": context_visibility,
        }

    def _resolve_directory(self) -> Path | None:
        raw = self.query_one("#history-path", Input).value.strip()
        if not raw:
            self.app.notify(
                "Please enter a directory path.", title="Import history", severity="warning"
            )
            return None

        try:
            directory = Path(os.path.expanduser(raw)).resolve()
        except Exception:
            directory = Path(os.path.expanduser(raw))

        if not directory.exists() or not directory.is_dir():
            self.query_one("#history-search-status", Static).update(
                "[red]Directory not found.[/red]"
            )
            self.app.notify(
                "Directory not found. Please choose a valid directory.",
                title="Import history",
                severity="warning",
            )
            return None

        self.query_one("#history-path", Input).value = str(directory)
        return directory

    def _start_discovery(self) -> None:
        if self._worker is not None and self._worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        ):
            return

        directory = self._resolve_directory()
        if directory is None:
            return

        self._set_busy(True)
        self.query_one("#history-search-status", Static).update(
            f"[dim]Searching sessions for {directory} ...[/dim]"
        )

        def work() -> dict[str, object]:
            results = self._discover_sessions_for_directory(directory)
            access_level, access_error = self._resolve_access_level()
            return {
                "results": results,
                "access_level": access_level,
                "access_error": access_error or "",
            }

        self._worker = self.run_worker(work, thread=True, exit_on_error=False)

    @classmethod
    def _normalize_access_level(cls, value: object) -> str:
        raw = str(value or "").strip().lower()
        if raw == cls.ACCESS_PRO_PLUS:
            return cls.ACCESS_PRO_PLUS
        if raw == cls.ACCESS_PRO:
            return cls.ACCESS_PRO
        return cls.ACCESS_FREE

    def _resolve_access_level(self) -> tuple[str, str | None]:
        now = time.time()
        if self._access_level_cached_at > 0 and (now - self._access_level_cached_at) < 30:
            return self._access_level, None

        access_level = self.ACCESS_FREE
        error: str | None = None

        try:
            from ...auth import get_billing_status

            billing, billing_error = get_billing_status()
            if isinstance(billing, dict):
                is_pro = bool(billing.get("is_pro") is True)
                plan_tier = str(billing.get("plan_tier") or "").strip().lower()
                if is_pro and plan_tier == self.ACCESS_PRO_PLUS:
                    access_level = self.ACCESS_PRO_PLUS
                elif is_pro:
                    access_level = self.ACCESS_PRO
                else:
                    access_level = self.ACCESS_FREE
            elif billing_error:
                error = str(billing_error)
        except Exception as exc:
            error = str(exc)

        access_level = self._normalize_access_level(access_level)
        self._access_level = access_level
        self._access_level_cached_at = now
        return access_level, error

    def _discover_sessions_for_directory(self, directory: Path) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        seen: set[str] = set()

        for item in self._discover_cloud_sessions(directory):
            key = str(item.get("path") or "")
            if key and key not in seen:
                seen.add(key)
                results.append(item)

        for item in self._discover_codex_sessions(directory):
            key = str(item.get("path") or "")
            if key and key not in seen:
                seen.add(key)
                results.append(item)

        results.sort(key=lambda item: float(item.get("mtime") or 0.0), reverse=True)
        return results

    def _discover_cloud_sessions(self, directory: Path) -> list[dict[str, object]]:
        from ...claude_detector import get_claude_project_name

        results: list[dict[str, object]] = []
        project_key = get_claude_project_name(directory)
        cloud_roots = [
            ("cloud", Path.home() / ".cloud" / "projects"),
            ("claude", Path.home() / ".claude" / "projects"),
        ]

        for source, root in cloud_roots:
            project_dir = root / project_key
            if not project_dir.exists() or not project_dir.is_dir():
                continue
            for session_file in project_dir.glob("*.jsonl"):
                if session_file.name.startswith("agent-"):
                    continue
                row = self._to_session_row(source=source, session_file=session_file)
                if row is not None:
                    results.append(row)

        return results

    def _discover_codex_sessions(self, directory: Path) -> list[dict[str, object]]:
        from ...codex_detector import find_codex_sessions_for_project

        results: list[dict[str, object]] = []
        codex_root = (Path.home() / ".codex" / "sessions").expanduser()
        if not codex_root.exists():
            return results

        try:
            codex_root_resolved = codex_root.resolve()
        except Exception:
            codex_root_resolved = codex_root

        sessions = find_codex_sessions_for_project(directory, days_back=self._codex_days_back())
        for session_file in sessions:
            if not self._is_under_root(session_file, codex_root_resolved):
                continue
            row = self._to_session_row(source="codex", session_file=session_file)
            if row is not None:
                results.append(row)

        return results

    @staticmethod
    def _codex_days_back() -> int:
        raw = os.environ.get("ALINE_IMPORT_HISTORY_CODEX_DAYS", "365")
        try:
            return max(1, int(raw))
        except Exception:
            return 365

    @staticmethod
    def _is_under_root(path: Path, root: Path) -> bool:
        try:
            return path.resolve().is_relative_to(root)
        except Exception:
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            root_text = str(root)
            path_text = str(resolved)
            return path_text == root_text or path_text.startswith(root_text + os.sep)

    @staticmethod
    def _to_session_row(source: str, session_file: Path) -> dict[str, object] | None:
        try:
            mtime = float(session_file.stat().st_mtime)
        except Exception:
            mtime = 0.0

        created_ts = ImportHistoryScreen._file_created_ts(session_file)
        session_label = ImportHistoryScreen._session_label_from_file(session_file)
        if session_label == "No Query Found":
            return None

        if mtime > 0:
            updated = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        else:
            updated = "-"

        return {
            "source": source,
            "session_id": session_file.stem,
            "session_label": session_label,
            "updated": updated,
            "path": str(session_file),
            "mtime": mtime,
            "created_ts": created_ts,
        }

    @staticmethod
    def _session_label_from_file(session_file: Path) -> str:
        fallback_message: str | None = None
        try:
            with session_file.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    if line_no > 4000:
                        break
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    primary_message = ImportHistoryScreen._extract_primary_user_request(data)
                    if primary_message:
                        return ImportHistoryScreen._compact_preview(primary_message)

                    if fallback_message is None:
                        fallback_message = ImportHistoryScreen._extract_fallback_user_request(data)
        except Exception:
            return "No Query Found"

        if fallback_message:
            return ImportHistoryScreen._compact_preview(fallback_message)
        return "No Query Found"

    @staticmethod
    def _extract_primary_user_request(data: dict[str, object]) -> str | None:
        msg_type = str(data.get("type") or "").strip()
        if not msg_type:
            return None

        if msg_type == "user":
            message = data.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            text = ImportHistoryScreen._extract_claude_user_text(content)
            return ImportHistoryScreen._clean_user_text(text)

        if msg_type == "event_msg":
            payload = data.get("payload")
            if isinstance(payload, dict) and payload.get("type") == "user_message":
                text = ImportHistoryScreen._extract_codex_user_message_payload(
                    payload.get("message")
                )
                return ImportHistoryScreen._clean_codex_request(text)

        return None

    @staticmethod
    def _extract_fallback_user_request(data: dict[str, object]) -> str | None:
        msg_type = str(data.get("type") or "").strip()
        if not msg_type:
            return None

        if msg_type == "response_item":
            payload = data.get("payload")
            if (
                isinstance(payload, dict)
                and payload.get("type") == "message"
                and payload.get("role") == "user"
            ):
                text = ImportHistoryScreen._extract_codex_text(payload.get("content"))
                return ImportHistoryScreen._clean_codex_request(text)

        if msg_type == "message" and str(data.get("role") or "").strip() == "user":
            text = ImportHistoryScreen._extract_codex_text(data.get("content"))
            return ImportHistoryScreen._clean_codex_request(text)

        return None

    @staticmethod
    def _extract_codex_user_message_payload(message: object) -> str:
        if isinstance(message, str):
            text = message.strip()
            if not text:
                return ""
            try:
                nested = json.loads(text)
            except Exception:
                return text
            if isinstance(nested, dict):
                nested_text = ImportHistoryScreen._extract_primary_user_request(nested)
                if nested_text:
                    return nested_text
            return text

        if isinstance(message, dict):
            nested_primary = ImportHistoryScreen._extract_primary_user_request(message)
            if nested_primary:
                return nested_primary

            role = str(message.get("role") or "").strip()
            if role == "user":
                text = ImportHistoryScreen._extract_codex_text(message.get("content"))
                if text:
                    return text

            for key in ("message", "text"):
                if key in message:
                    nested = ImportHistoryScreen._extract_codex_user_message_payload(
                        message.get(key)
                    )
                    if nested:
                        return nested

            if "content" in message:
                text = ImportHistoryScreen._extract_codex_text(message.get("content"))
                if text:
                    return text
            return ""

        if isinstance(message, list):
            parts: list[str] = []
            for item in message:
                part = ImportHistoryScreen._extract_codex_user_message_payload(item)
                if part:
                    parts.append(part)
            return "\n".join(parts).strip()

        return ""

    @staticmethod
    def _extract_claude_user_text(content: object) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_result":
                return ""

        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                part = str(item.get("text") or "").strip()
                if part:
                    parts.append(part)
        return "\n".join(parts).strip()

    @staticmethod
    def _extract_codex_text(content: object) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, dict):
            return str(content.get("text") or "").strip()
        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type not in ("input_text", "output_text", "text"):
                continue
            part = str(item.get("text") or "").strip()
            if part:
                parts.append(part)
        return "\n".join(parts).strip()

    @staticmethod
    def _clean_user_text(text: str) -> str | None:
        cleaned = str(text or "").strip()
        if not cleaned:
            return None
        try:
            from ...hooks import clean_user_message

            cleaned = clean_user_message(cleaned).strip()
        except Exception:
            pass
        return cleaned or None

    @staticmethod
    def _clean_codex_request(text: str) -> str | None:
        cleaned = str(text or "").strip()
        if not cleaned:
            return None
        if cleaned.startswith("<environment_context>") or cleaned.startswith(
            "<environment_context"
        ):
            return None

        marker = "## My request for Codex:"
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[1].strip()
        return ImportHistoryScreen._clean_user_text(cleaned)

    @staticmethod
    def _compact_preview(text: str, max_chars: int = 90) -> str:
        single_line = " ".join(str(text or "").split())
        if len(single_line) <= max_chars:
            return single_line
        return single_line[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _file_created_ts(session_file: Path) -> float:
        """Best-effort file creation timestamp."""
        try:
            st = session_file.stat()
        except Exception:
            return 0.0

        # macOS / BSD
        birth = getattr(st, "st_birthtime", None)
        if isinstance(birth, (int, float)) and birth > 0:
            return float(birth)

        # Fallback for platforms without birthtime.
        mtime = getattr(st, "st_mtime", None)
        if isinstance(mtime, (int, float)) and mtime > 0:
            return float(mtime)

        return 0.0

    @staticmethod
    def _tier_counts(results: list[dict[str, object]]) -> dict[str, dict[str, int]]:
        now_ts = datetime.now().timestamp()
        counts = {
            "30d": {"codex": 0, "claude": 0},
            "1_6m": {"codex": 0, "claude": 0},
            "gt_6m": {"codex": 0, "claude": 0},
            "unknown": {"codex": 0, "claude": 0},
        }

        for item in results:
            source = str(item.get("source") or "").strip().lower()
            provider = "codex" if source == "codex" else "claude"

            created_ts_raw = item.get("created_ts")
            try:
                created_ts = float(created_ts_raw) if created_ts_raw is not None else 0.0
            except Exception:
                created_ts = 0.0
            if created_ts <= 0:
                counts["unknown"][provider] += 1
                continue

            age_days = max(0.0, (now_ts - created_ts) / 86400.0)
            if age_days <= 30:
                counts["30d"][provider] += 1
            elif age_days <= 180:
                counts["1_6m"][provider] += 1
            else:
                counts["gt_6m"][provider] += 1

        return counts

    @staticmethod
    def _is_result_visible_for_access(
        item: dict[str, object],
        access_level: str,
        *,
        now_ts: float | None = None,
    ) -> bool:
        current_ts = datetime.now().timestamp() if now_ts is None else now_ts
        try:
            created_ts = float(item.get("created_ts") or 0.0)
        except Exception:
            created_ts = 0.0

        if created_ts <= 0:
            # Unknown age: keep visible to avoid accidentally hiding valid sessions.
            return True

        age_days = max(0.0, (current_ts - created_ts) / 86400.0)
        if access_level == ImportHistoryScreen.ACCESS_FREE:
            return age_days <= 30
        if access_level == ImportHistoryScreen.ACCESS_PRO:
            return age_days <= 180
        return True

    @staticmethod
    def _filter_results_for_access(
        results: list[dict[str, object]],
        access_level: str,
    ) -> list[dict[str, object]]:
        normalized = ImportHistoryScreen._normalize_access_level(access_level)
        now_ts = datetime.now().timestamp()
        return [
            item
            for item in results
            if ImportHistoryScreen._is_result_visible_for_access(
                item,
                normalized,
                now_ts=now_ts,
            )
        ]

    @staticmethod
    def _provider_counts(results: list[dict[str, object]]) -> dict[str, int]:
        provider_counts = {"codex": 0, "claude": 0}
        for item in results:
            source = str(item.get("source") or "").strip().lower()
            provider = "codex" if source == "codex" else "claude"
            provider_counts[provider] += 1
        return provider_counts

    @staticmethod
    def _build_tier_summary(results: list[dict[str, object]], access_level: str) -> str:
        normalized = ImportHistoryScreen._normalize_access_level(access_level)
        counts = ImportHistoryScreen._tier_counts(results)
        visible_results = ImportHistoryScreen._filter_results_for_access(results, normalized)
        visible_counts = ImportHistoryScreen._provider_counts(visible_results)

        c30_codex = counts["30d"]["codex"]
        c30_claude = counts["30d"]["claude"]
        c30_total = c30_codex + c30_claude

        c16_codex = counts["1_6m"]["codex"]
        c16_claude = counts["1_6m"]["claude"]

        c6p_codex = counts["gt_6m"]["codex"]
        c6p_claude = counts["gt_6m"]["claude"]
        c6p_total = c6p_codex + c6p_claude

        cu_codex = counts["unknown"]["codex"]
        cu_claude = counts["unknown"]["claude"]
        cu_total = cu_codex + cu_claude

        visible_total = len(visible_results)
        visible_codex = visible_counts["codex"]
        visible_claude = visible_counts["claude"]

        if normalized == ImportHistoryScreen.ACCESS_FREE:
            c16_total = c16_codex + c16_claude
            lines = [
                f"Free users can view sessions from the past 30 days. (codex: {c30_codex}, claude: {c30_claude})",
                f"Upgrade to Pro to access {c16_total} additional sessions from the past 6 months.",
            ]
            if c6p_total > 0:
                lines.append(
                    f"Upgrade to Pro+ to access {c6p_total} more sessions older than 6 months."
                )
            if cu_total > 0:
                lines.append(f"{cu_total} session(s) have unknown age and are shown by default.")
            return "\n".join(lines)

        if normalized == ImportHistoryScreen.ACCESS_PRO:
            hidden_total = c6p_total
            c6m_codex = c30_codex + c16_codex
            c6m_claude = c30_claude + c16_claude
            lines = [
                f"Pro users can view sessions from the past 6 months. (codex: {c6m_codex}, claude: {c6m_claude})"
            ]
            if hidden_total > 0:
                lines.append(
                    f"Upgrade to Pro to access {hidden_total} sessions from your full history."
                )
            if cu_total > 0:
                lines.append(f"{cu_total} session(s) have unknown age and are shown by default.")
            return "\n".join(lines)

        return (
            f"Showing all {visible_total} matching sessions for Pro+ "
            f"(Codex: {visible_codex}; Claude Code: {visible_claude}).\n"
            f"Time buckets: <=30 days ({c30_total}), 1-6 months ({c16_codex + c16_claude}), >6 months ({c6p_total})."
        )

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if self._import_worker is not None and event.worker is self._import_worker:
            self._on_import_worker_state_changed(event)
            return

        if self._worker is None or event.worker is not self._worker:
            return

        status = self.query_one("#history-search-status", Static)

        if event.state == WorkerState.ERROR:
            self._set_busy(False)
            err = self._worker.error
            status.update(f"[red]Search failed:[/red] {err}")
            self.app.notify(f"Search failed: {err}", title="Import history", severity="error")
            return

        if event.state != WorkerState.SUCCESS:
            return

        self._set_busy(False)
        payload = self._worker.result or {}
        if isinstance(payload, dict):
            results = list(payload.get("results") or [])
            access_level = self._normalize_access_level(payload.get("access_level"))
            self._access_level = access_level
        else:
            results = list(payload or [])
            access_level = self._access_level
        visible_results = ImportHistoryScreen._filter_results_for_access(results, access_level)
        table = self.query_one("#session-results", DataTable)
        table.clear()
        self._result_row_keys = []
        self._selected_row_keys = set()
        self._result_items_by_key = {}

        for idx, item in enumerate(visible_results):
            row_key = str(item.get("path") or f"row-{idx}")
            self._result_row_keys.append(row_key)
            self._selected_row_keys.add(row_key)  # default: all selected
            self._result_items_by_key[row_key] = item
            table.add_row(
                self._checkbox_cell(True),
                str(item.get("source") or ""),
                str(item.get("session_label") or item.get("session_id") or ""),
                str(item.get("updated") or ""),
                str(item.get("path") or ""),
                key=row_key,
            )
        self._update_select_toggle_button()

        status.update(ImportHistoryScreen._build_tier_summary(results, access_level))
        self._update_import_button()

    def _on_import_worker_state_changed(self, event: Worker.StateChanged) -> None:
        status = self.query_one("#history-search-status", Static)
        import_btn = self.query_one("#import-btn", Button)

        if event.state == WorkerState.ERROR:
            err = self._import_worker.error if self._import_worker is not None else "Unknown error"
            self._import_worker = None
            self._stop_importing_dots()
            status.update(f"[red]Import failed:[/red] {err}")
            self.app.notify(f"Import failed: {err}", title="Import history", severity="error")
            self._update_import_button()
            return

        if event.state != WorkerState.SUCCESS:
            return

        result = self._import_worker.result if self._import_worker is not None else {}
        self._import_worker = None
        self._stop_importing_dots()
        import_btn.disabled = False

        created_new = bool((result or {}).get("created_new_context"))
        context_name = str((result or {}).get("context_name") or "Context")
        selected_count = int((result or {}).get("selected_count") or 0)
        imported_count = int((result or {}).get("imported_count") or 0)
        failed_count = int((result or {}).get("failed_count") or 0)
        summary_failed_count = int((result or {}).get("summary_failed_count") or 0)
        missing_count = int((result or {}).get("missing_count") or 0)
        agent_id = str((result or {}).get("agent_id") or "")
        limit_reached = bool((result or {}).get("limit_reached"))
        blocked_by_limit = int((result or {}).get("blocked_by_limit") or 0)
        limit_value = int((result or {}).get("limit_value") or 1000)
        context_not_found = bool((result or {}).get("context_not_found"))
        error_message = str((result or {}).get("error_message") or "").strip()
        context_visibility = str((result or {}).get("context_visibility") or "visible")

        if created_new and agent_id:
            self._reload_context_options(prefer_value=agent_id)

        if context_not_found:
            msg = error_message or "Selected context no longer exists."
            status.update(msg)
            self.app.notify(msg, title="Import history", severity="error", timeout=6)
            self._update_import_button()
            return

        msg = (
            f"Imported {imported_count}/{selected_count} session(s) into {context_name}."
            f" Failed: {failed_count}, Missing: {missing_count}, Summary failed: {summary_failed_count}."
        )
        imported_to_archived = context_visibility != "visible"
        if imported_to_archived:
            msg = (
                f"{msg} Note: target context is archived. Enable 'Show Archived' or restore it "
                "to view imported sessions in Agents."
            )
        has_partial_failure = failed_count > 0 or missing_count > 0
        if limit_reached:
            msg = (
                f"{msg} 当前“Context”关联的session已经达到上限（{limit_value}），"
                f"额外拦截 {blocked_by_limit} 个 session 关联。"
            )
        status.update(msg)
        if limit_reached:
            self.app.notify(msg, title="Import history", severity="warning", timeout=6)
        elif imported_to_archived:
            self.app.notify(msg, title="Import history", severity="warning", timeout=6)
        elif has_partial_failure:
            self.app.notify(msg, title="Import history", severity="warning", timeout=6)
        else:
            self.app.notify(msg, title="Import history", timeout=6)
        self._update_import_button()
        try:
            tabbed = self.app.query_one(TabbedContent)
            tabbed.active = "agents"
        except Exception:
            pass
        self.app.pop_screen()

    def on_unmount(self) -> None:
        if self._importing_dots_timer is not None:
            try:
                self._importing_dots_timer.stop()
            except Exception:
                pass
            self._importing_dots_timer = None

    async def _select_path(self) -> str | None:
        """Open a path picker and return the selected path."""
        if self._picker_open:
            return None

        self._picker_open = True
        with contextlib.suppress(Exception):
            self.query_one("#browse-btn", Button).disabled = True

        current = self.query_one("#history-path", Input).value.strip() or self._default_path

        try:
            if sys.platform == "darwin":
                current_escaped = current.replace('"', '\\"')
                script = f"""
                    try
                        set defaultFolder to POSIX file "{current_escaped}" as alias
                        set selectedItem to choose folder with prompt "Select history directory" default location defaultFolder
                        return POSIX path of selectedItem
                    on error
                        return ""
                    end try
                """
                return await self._run_picker_cmd(["osascript", "-e", script])

            if shutil.which("zenity"):
                return await self._run_picker_cmd(
                    ["zenity", "--file-selection", "--directory", f"--filename={current}/"]
                )
            if shutil.which("kdialog"):
                return await self._run_picker_cmd(["kdialog", "--getexistingdirectory", current])
            return None
        finally:
            self._picker_open = False
            with contextlib.suppress(Exception):
                if not self._is_discovery_busy():
                    self.query_one("#browse-btn", Button).disabled = False

    async def _run_picker_cmd(self, cmd: list[str]) -> str | None:
        try:
            proc = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                ),
            )
            result = (proc.stdout or "").strip()
            return result or None
        except Exception:
            return None

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if (event.input.id or "") == "history-path":
            self._start_discovery()

    async def on_select_changed(self, event: Select.Changed) -> None:
        if (event.select.id or "") != "context-select":
            return
        self._update_import_button()
        self._update_context_expand_button()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "back-btn":
            self.app.pop_screen()
            return

        if button_id == "context-expand-btn":
            select = self.query_one("#context-select", Select)
            if not select.disabled:
                now = time.monotonic()
                # Guard against blur/dismiss racing this click and reopening instantly.
                if (not bool(select.expanded)) and (now - self._context_last_dismissed_at) < 0.25:
                    self._update_context_expand_button(False)
                    return
                select.focus()
                target_expanded = not bool(select.expanded)
                select.expanded = target_expanded
                self._update_context_expand_button(target_expanded)
            return

        if button_id == "import-btn":
            self._start_import()
            return

        if button_id == "toggle-select-btn":
            self._toggle_select_all_none()
            return

        if button_id == "browse-btn":
            if self._picker_open:
                return
            selected = await self._select_path()
            if selected and os.path.isdir(selected):
                self.query_one("#history-path", Input).value = selected
                self._start_discovery()
                return
            self.app.notify(
                "No directory selected. You can type a directory path in the address bar.",
                title="Import history",
                severity="warning",
            )
