"""Agent detail modal for the dashboard."""

from __future__ import annotations

import shlex
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Static
from textual.worker import Worker, WorkerState

from ..widgets.openable_table import OpenableDataTable
from ..system_prompts import build_claude_system_prompt, build_codex_instructions


def _format_dt(dt: object) -> str:
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(dt, str):
        raw = dt.strip()
        if not raw:
            return "-"
        # Best-effort ISO parsing; fall back to raw string on failure.
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return parsed.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return raw
    if dt is None:
        return "-"
    return str(dt)


def _format_relative_time(dt: datetime) -> str:
    now = datetime.now(dt.tzinfo) if dt.tzinfo is not None else datetime.now()
    diff = now - dt
    seconds = diff.total_seconds()

    if seconds < 60:
        return "just now"
    if seconds < 3600:
        mins = int(seconds / 60)
        return f"{mins}m ago"
    if seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours}h ago"
    days = int(seconds / 86400)
    return f"{days}d ago"


def _format_iso_relative(dt: Optional[datetime]) -> str:
    if not dt:
        return "-"
    try:
        return _format_relative_time(dt)
    except Exception:
        return _format_dt(dt)


def _datetime_sort_timestamp(value: object) -> float:
    """Return a sortable timestamp for mixed datetime/string/None values."""
    dt: Optional[datetime] = None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        raw = value.strip()
        if raw:
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except Exception:
                dt = None
    if dt is None:
        return 0.0
    try:
        return float(dt.timestamp())
    except Exception:
        return 0.0


def _shorten_id(val: str | None) -> str:
    if not val:
        return "-"
    if len(val) <= 20:
        return val
    return f"{val[:8]}...{val[-8:]}"


class _SectionDivider(Static):
    DEFAULT_CSS = """
    _SectionDivider {
        width: 1fr;
        height: 1;
        color: $text-muted;
        background: transparent;
        content-align: center middle;
    }
    """

    def __init__(self, title: str) -> None:
        super().__init__()
        self._title = (title or "").strip()

    def render(self) -> Text:
        width = max(0, int(self.size.width))
        title = self._title
        if not width:
            return Text("")

        mid = f" {title} " if title else ""
        if len(mid) >= width:
            return Text(mid[:width])

        fill = width - len(mid)
        left = fill // 2
        right = fill - left
        line = ("─" * left) + mid + ("─" * right)
        return Text(line)


class AgentDetailScreen(ModalScreen):
    """Modal that shows agent details and its sessions."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
    ]

    DEFAULT_CSS = """
    AgentDetailScreen {
        align: center middle;
    }

    AgentDetailScreen #agent-detail-root {
        width: 95%;
        height: 95%;
        padding: 1;
        background: $background;
        border: none;
    }

    AgentDetailScreen #agent-meta {
        height: auto;
        margin-bottom: 1;
    }

    AgentDetailScreen #agent-name {
        height: auto;
        margin-bottom: 1;
    }

    AgentDetailScreen #agent-times {
        height: auto;
        margin-bottom: 0;
        color: $text-muted;
    }

    AgentDetailScreen #agent-detail-body {
        height: 1fr;
    }

    AgentDetailScreen #agent-title {
        height: auto;
        margin-bottom: 0;
        color: $text;
    }

    AgentDetailScreen #agent-description {
        height: auto;
        margin-bottom: 1;
    }

    AgentDetailScreen #agent-sessions-table {
        width: 1fr;
        height: 1fr;
        max-height: 26;
    }

    AgentDetailScreen #agent-session-preview {
        width: 1fr;
        height: auto;
        margin-top: 1;
    }

    AgentDetailScreen #agent-hint {
        height: 1;
        margin-top: 1;
        color: $text-muted;
        text-align: right;
    }

    AgentDetailScreen #resume-bar {
        height: auto;
        width: 100%;
        align: center middle;
    }

    AgentDetailScreen #resume-btn {
        min-width: 16;
        margin-left: 2;
    }

    AgentDetailScreen #back-btn {
        min-width: 16;
    }
    """

    def __init__(self, agent_id: str) -> None:
        super().__init__()
        self.agent_id = agent_id
        self._load_error: Optional[str] = None
        self._agent_info = None
        self._sessions: list[dict] = []
        self._session_record_cache: dict[str, object] = {}
        self._session_index_by_id: dict[str, int] = {}
        self._session_source_by_id: dict[str, str] = {}
        self._session_type_by_id: dict[str, str] = {}
        self._session_workspace_by_id: dict[str, str] = {}
        self._user_name_cache: dict[str, str] = {}
        self._active_session_ids: set[str] = set()
        self._active_session_to_terminal: dict[str, str] = {}
        self._initialized: bool = False
        self._current_user_uid: Optional[str] = None
        self._current_user_name: Optional[str] = None

        try:
            from ...config import ReAlignConfig

            config = ReAlignConfig.load()
            uid = str(getattr(config, "uid", "") or "").strip()
            user_name = str(getattr(config, "user_name", "") or "").strip()
            self._current_user_uid = uid or None
            self._current_user_name = user_name or None
        except Exception:
            self._current_user_uid = None
            self._current_user_name = None

    def compose(self) -> ComposeResult:
        with Container(id="agent-detail-root"):
            yield _SectionDivider("Context Details")

            with Vertical(id="agent-meta"):
                yield Static(id="agent-name")
                yield Static(id="agent-times")

            yield Static(id="agent-title")
            yield Static(id="agent-description")

            yield _SectionDivider("Sessions")

            with Vertical(id="agent-detail-body"):
                sessions_table = OpenableDataTable(id="agent-sessions-table")
                sessions_table.add_columns(
                    "#",
                    "Source",
                    "Turns",
                    "Title",
                    "Last Activity",
                )
                sessions_table.cursor_type = "row"
                yield sessions_table
                yield Static(
                    "Single click to preview summary, double click for session details",
                    id="agent-hint",
                )
                yield Static(id="agent-session-preview")
            with Horizontal(id="resume-bar"):
                yield Button("Open", id="resume-btn", variant="primary", disabled=True)
                yield Button("Cancel", id="back-btn", variant="default")

    def on_show(self) -> None:
        self.call_later(self._ensure_initialized)

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        sessions_table = self.query_one("#agent-sessions-table", DataTable)
        self._load_data()
        self._update_display()

        if sessions_table.row_count > 0:
            sessions_table.focus()

    def action_close(self) -> None:
        self.app.pop_screen()

    def on_openable_data_table_row_activated(self, event: OpenableDataTable.RowActivated) -> None:
        if event.data_table.id != "agent-sessions-table":
            return

        session_id = str(event.row_key.value)
        if not session_id:
            return

        from .session_detail import SessionDetailScreen

        self.app.push_screen(SessionDetailScreen(session_id))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "agent-sessions-table":
            return
        session_id = str(event.row_key.value)
        self._update_session_preview(session_id)
        self._update_resume_button(session_id)

    def _load_data(self) -> None:
        try:
            from ...db import MAX_SESSION_NUM, get_database

            db = get_database()
            self._agent_info = db.get_agent_info(self.agent_id)
            sessions = db.get_sessions_by_agent_id(self.agent_id, limit=MAX_SESSION_NUM)

            source_map = {
                "claude": "Claude",
                "claude_code": "Claude Code",
                "codex": "Codex",
                "gemini": "Gemini",
                "opencode": "Opencode",
                "shell": "Shell",
                "zsh": "Shell",  # backward compat for old DB records
            }

            self._sessions = []
            self._session_type_by_id = {}
            self._session_workspace_by_id = {}
            for s in sessions:
                session_id = str(s.id)
                session_type = getattr(s, "session_type", None) or "unknown"
                workspace = getattr(s, "workspace_path", None)
                title = getattr(s, "session_title", None) or "(no title)"
                last_activity = getattr(s, "last_activity_at", None)
                turns = int(getattr(s, "total_turns", 0) or 0)

                project = str(workspace).split("/")[-1] if workspace else "-"
                source = source_map.get(session_type, session_type)

                self._session_type_by_id[session_id] = session_type
                self._session_workspace_by_id[session_id] = str(workspace) if workspace else ""

                self._sessions.append(
                    {
                        "id": session_id,
                        "short_id": _shorten_id(session_id),
                        "source": source,
                        "project": project,
                        "turns": turns,
                        "title": title,
                        "last_activity": last_activity,
                    }
                )

            self._sessions.sort(
                key=lambda item: _datetime_sort_timestamp(item.get("last_activity")),
                reverse=True,
            )

            # Build set of session IDs currently open in active terminals
            # and map each active session to its terminal_id for window switching.
            self._active_session_ids = set()
            self._active_session_to_terminal = {}
            try:
                active_agents = db.list_agents(status="active", limit=1000)
                active_terminal_ids = set()
                for agent in active_agents:
                    if (agent.source or "") == f"agent:{self.agent_id}":
                        if agent.session_id:
                            self._active_session_ids.add(agent.session_id)
                            self._active_session_to_terminal[agent.session_id] = agent.id
                        active_terminal_ids.add(agent.id)
                if active_terminal_ids:
                    try:
                        links = db.list_latest_window_links(limit=1000)
                        for link in links:
                            if link.terminal_id in active_terminal_ids and link.session_id:
                                self._active_session_ids.add(link.session_id)
                                self._active_session_to_terminal.setdefault(
                                    link.session_id, link.terminal_id
                                )
                    except Exception:
                        pass
            except Exception:
                pass

            self._load_error = None
        except Exception as e:
            self._agent_info = None
            self._sessions = []
            self._session_record_cache = {}
            self._session_index_by_id = {}
            self._session_source_by_id = {}
            self._session_type_by_id = {}
            self._session_workspace_by_id = {}
            self._active_session_ids = set()
            self._active_session_to_terminal = {}
            self._load_error = str(e)

    def _get_share_link(self) -> str:
        if not self._agent_info:
            return ""

        share_url = getattr(self._agent_info, "share_url", None)
        share_link = str(share_url).strip() if share_url else ""
        return share_link

    def _update_display(self) -> None:
        name_widget = self.query_one("#agent-name", Static)
        times_widget = self.query_one("#agent-times", Static)
        title_widget = self.query_one("#agent-title", Static)
        description = self.query_one("#agent-description", Static)
        preview = self.query_one("#agent-session-preview", Static)

        if self._load_error:
            name_widget.update(
                f"[red]Failed to load agent details:[/red] {escape(self._load_error)}"
            )
            times_widget.update("")
            title_widget.update("")
            description.update("")
            preview.update("")
            return

        name = getattr(self._agent_info, "name", None) if self._agent_info else None
        title = getattr(self._agent_info, "title", None) if self._agent_info else None
        desc = getattr(self._agent_info, "description", None) if self._agent_info else None
        created_at = getattr(self._agent_info, "created_at", None) if self._agent_info else None
        updated_at = getattr(self._agent_info, "updated_at", None) if self._agent_info else None

        display_name = name or "(no name)"
        name_widget.update(f"[bold]{escape(display_name)}[/bold]")

        time_lines: list[str] = [
            f"[dim]Created:[/dim] {_format_dt(created_at)}    [dim]Updated:[/dim] {_format_dt(updated_at)}"
        ]
        time_lines.append(f"[dim]Sessions:[/dim] {len(self._sessions)}")
        times_widget.update("\n".join(time_lines))

        title_widget.update(f"[bold]{escape(title)}[/bold]" if title else "")
        description.update(desc or "(no description)")
        preview.update("")

        table = self.query_one("#agent-sessions-table", DataTable)
        selected_session_id: Optional[str] = None
        try:
            if table.row_count > 0:
                selected_session_id = str(
                    table.coordinate_to_cell_key(table.cursor_coordinate)[0].value
                )
        except Exception:
            selected_session_id = None
        table.clear()

        self._session_index_by_id = {}
        self._session_source_by_id = {}
        for idx, s in enumerate(self._sessions, 1):
            title_cell = s["title"]
            if len(title_cell) > 40:
                title_cell = title_cell[:40] + "..."

            last_activity_str = _format_iso_relative(s["last_activity"])

            session_id = str(s["id"])
            self._session_index_by_id[session_id] = idx
            self._session_source_by_id[session_id] = str(s.get("source") or "Unknown")
            table.add_row(
                str(idx),
                s["source"],
                str(s["turns"]),
                title_cell,
                last_activity_str,
                key=s["id"],
            )

        if table.row_count > 0:
            if selected_session_id:
                try:
                    table.cursor_coordinate = (table.get_row_index(selected_session_id), 0)
                except Exception:
                    table.cursor_coordinate = (0, 0)
            else:
                table.cursor_coordinate = (0, 0)

            row_key = table.coordinate_to_cell_key(table.cursor_coordinate)[0]
            self._update_session_preview(str(row_key.value))
            self._update_resume_button(str(row_key.value))

    def _get_session_record(self, session_id: str) -> Optional[object]:
        if session_id in self._session_record_cache:
            return self._session_record_cache[session_id]
        try:
            from ...db import get_database

            db = get_database()
            record = db.get_session_by_id(session_id)
            self._session_record_cache[session_id] = record
            return record
        except Exception:
            self._session_record_cache[session_id] = None
            return None

    @staticmethod
    def _short_uid(uid: str) -> str:
        uid = uid.strip()
        if not uid:
            return "-"
        if len(uid) <= 8:
            return uid
        return uid[:8] + "..."

    def _resolve_user_display(self, uid: Optional[str]) -> Optional[str]:
        if not uid:
            return None
        uid = uid.strip()
        if not uid:
            return None
        cached = self._user_name_cache.get(uid)
        if cached is not None:
            return cached

        display = self._short_uid(uid)
        try:
            from ...db import get_database

            db = get_database()
            user = db.get_user(uid)
            user_name = str(getattr(user, "user_name", "") or "").strip() if user else ""
            user_email = str(getattr(user, "user_email", "") or "").strip() if user else ""
            if user_name:
                display = user_name
            elif user_email:
                display = user_email
            else:
                try:
                    from ...user_directory import fetch_remote_user_profile

                    remote_user = fetch_remote_user_profile(uid)
                except Exception:
                    remote_user = None

                if remote_user:
                    try:
                        db.upsert_user(uid, remote_user.user_name, remote_user.user_email)
                    except Exception:
                        pass
                    if remote_user.user_name:
                        display = remote_user.user_name
                    elif remote_user.user_email:
                        display = remote_user.user_email
        except Exception:
            pass
        self._user_name_cache[uid] = display
        return display

    def _get_creator_display(self, record: object) -> str:
        created_by = str(getattr(record, "created_by", "") or "").strip()
        if not created_by:
            return "-"
        short_uid = self._short_uid(created_by)
        resolved = self._resolve_user_display(created_by) or short_uid
        if self._current_user_uid and created_by == self._current_user_uid:
            self_name = self._current_user_name or ""
            if self_name:
                return f"you ({self_name})"
            return f"you ({resolved})"
        return resolved

    def _is_shared_from_other_user(self, record: object) -> bool:
        created_by = str(getattr(record, "created_by", "") or "").strip()
        shared_by = str(getattr(record, "shared_by", "") or "").strip()
        if created_by and self._current_user_uid and created_by != self._current_user_uid:
            return True
        if created_by and shared_by and created_by != shared_by:
            return True
        return False

    @staticmethod
    def _has_local_session_file(record: object) -> bool:
        raw = str(getattr(record, "session_file_path", "") or "").strip()
        if not raw or raw in {".", ".."}:
            return False
        try:
            return Path(raw).expanduser().exists()
        except Exception:
            return False

    def _get_resume_unavailable_reason(self, session_id: str) -> Optional[str]:
        record = self._get_session_record(session_id)
        if not record:
            return None
        if self._has_local_session_file(record):
            return None
        if self._is_shared_from_other_user(record):
            return "Cannot resume sessions shared by other users."
        return "Cannot resume this session because the local transcript file is missing."

    def _update_session_preview(self, session_id: str) -> None:
        preview = self.query_one("#agent-session-preview", Static)
        if not session_id:
            preview.update("")
            return

        record = self._get_session_record(session_id)
        if not record:
            preview.update("[dim]No session details available.[/dim]")
            return

        title = getattr(record, "session_title", None) or "(no title)"
        summary = getattr(record, "session_summary", None) or "(no summary)"
        creator = self._get_creator_display(record)

        idx = self._session_index_by_id.get(session_id)
        source = self._session_source_by_id.get(session_id, "Unknown")
        header = f"#{idx} {source}" if idx is not None else f"#{source}"

        lines = [f"[bold]{escape(header)}[/bold]"]
        lines.extend(
            [
                f"[bold]Creator:[/bold] {escape(creator)}",
                f"[bold]Title:[/bold] {escape(title)}",
                f"[bold]Summary:[/bold] {escape(summary)}",
            ]
        )
        preview.update("\n".join(lines))

    def _update_resume_button(self, session_id: str) -> None:
        try:
            btn = self.query_one("#resume-btn", Button)
        except Exception:
            return
        if not session_id:
            btn.disabled = True
            return
        session_type = self._session_type_by_id.get(session_id, "")
        if session_type not in ("claude", "claude_code", "codex"):
            btn.disabled = True
            return
        btn.disabled = False
        btn.label = "Open" if session_id in self._active_session_ids else "Resume"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.action_close()
            return
        if event.button.id != "resume-btn":
            return
        table = self.query_one("#agent-sessions-table", DataTable)
        try:
            row_key = table.coordinate_to_cell_key(table.cursor_coordinate)[0]
            session_id = str(row_key.value)
        except Exception:
            return
        if not session_id:
            return
        session_type = self._session_type_by_id.get(session_id, "")
        workspace = self._session_workspace_by_id.get(session_id, "")
        if session_type not in ("claude", "claude_code", "codex"):
            self.app.notify("Open not supported for this session type", severity="warning")
            return

        # If the session is already active, jump to its terminal window.
        if session_id in self._active_session_ids:
            terminal_id = self._active_session_to_terminal.get(session_id)
            if terminal_id:
                self._focus_terminal(terminal_id)
                self.action_close()
            return

        unavailable_reason = self._get_resume_unavailable_reason(session_id)
        if unavailable_reason:
            self.app.notify(unavailable_reason, title="Resume", severity="warning")
            return

        # Otherwise, resume the session in a new terminal.
        btn = self.query_one("#resume-btn", Button)
        btn.disabled = True
        self.run_worker(
            self._do_resume(session_id, session_type, workspace),
            name="resume_session",
            exclusive=True,
        )

    def _focus_terminal(self, terminal_id: str) -> None:
        """Switch to the tmux window that owns *terminal_id*."""
        from .. import tmux_manager

        try:
            windows = tmux_manager.list_inner_windows()
            for w in windows:
                if w.terminal_id == terminal_id:
                    tmux_manager.select_inner_window(w.window_id)
                    tmux_manager.focus_right_pane()
                    return
        except Exception:
            pass

    async def _do_resume(self, session_id: str, session_type: str, workspace: str) -> None:
        from .. import tmux_manager
        from ...db import AgentSessionLimitExceededError, MAX_SESSION_NUM, get_database

        terminal_id = tmux_manager.new_terminal_id()
        db = get_database(read_only=False)

        limit_msg = f"当前“Context”关联的session已经达到上限（{MAX_SESSION_NUM}）。"
        try:
            if not db.can_link_session_to_agent(self.agent_id, session_id):
                raise RuntimeError(limit_msg)
            db.link_session_to_agent(self.agent_id, session_id)
        except AgentSessionLimitExceededError as e:
            raise RuntimeError(limit_msg) from e

        if session_type in ("claude", "claude_code"):
            provider = "claude"

            try:
                from ...codex_home import prepare_codex_home

                codex_home = prepare_codex_home(terminal_id)
            except Exception:
                codex_home = None

            env = {
                tmux_manager.ENV_TERMINAL_ID: terminal_id,
                tmux_manager.ENV_TERMINAL_PROVIDER: provider,
                tmux_manager.ENV_INNER_SOCKET: tmux_manager.INNER_SOCKET,
                tmux_manager.ENV_INNER_SESSION: tmux_manager.INNER_SESSION,
                "ALINE_AGENT_ID": self.agent_id,
            }
            if codex_home:
                env["CODEX_HOME"] = str(codex_home)

            if workspace:
                self._install_claude_hooks(workspace)

            system_prompt = build_claude_system_prompt(
                agent_title=str(getattr(self._agent_info, "title", "") or ""),
                agent_description=str(getattr(self._agent_info, "description", "") or ""),
            )
            claude_cmd = (
                f"claude --resume {shlex.quote(session_id)}"
                f" --append-system-prompt {shlex.quote(system_prompt)}"
            )
            cmd_dir = workspace or str(Path.home())
            command = (
                f"cd {shlex.quote(cmd_dir)} && {tmux_manager.shell_run_and_keep_open(claude_cmd)}"
            )

            created = tmux_manager.create_inner_window(
                "cc",
                tmux_manager.shell_command_with_env(command, env),
                terminal_id=terminal_id,
                provider=provider,
            )

        elif session_type == "codex":
            provider = "codex"

            from ...hooks import extract_codex_rollout_hash

            resume_id = extract_codex_rollout_hash(session_id)
            if not resume_id:
                resume_id = session_id

            try:
                from ...codex_home import prepare_codex_home

                codex_home = prepare_codex_home(terminal_id, agent_id=self.agent_id)
            except Exception:
                codex_home = None

            if codex_home:
                try:
                    instructions_file = Path(codex_home) / "instructions.md"
                    instructions = build_codex_instructions(
                        agent_title=str(getattr(self._agent_info, "title", "") or ""),
                        agent_description=str(getattr(self._agent_info, "description", "") or ""),
                    )
                    instructions_file.write_text(f"{instructions}\n")
                except Exception:
                    pass

            # Symlink original session file into the new CODEX_HOME so
            # `codex resume` can find it (it only searches $CODEX_HOME/sessions/).
            if codex_home:
                try:
                    record = self._get_session_record(session_id)
                    orig = Path(str(record.session_file_path)) if record else None
                    if orig and orig.exists():
                        # Replicate the date-based directory structure
                        dest_dir = (
                            Path(codex_home)
                            / "sessions"
                            / orig.parent.relative_to(
                                next(p for p in orig.parents if p.name == "sessions")
                            )
                        )
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest = dest_dir / orig.name
                        if not dest.exists():
                            dest.symlink_to(orig)
                except Exception:
                    pass

            env = {
                tmux_manager.ENV_TERMINAL_ID: terminal_id,
                tmux_manager.ENV_TERMINAL_PROVIDER: provider,
                tmux_manager.ENV_INNER_SOCKET: tmux_manager.INNER_SOCKET,
                tmux_manager.ENV_INNER_SESSION: tmux_manager.INNER_SESSION,
                "ALINE_AGENT_ID": self.agent_id,
            }
            if codex_home:
                env["CODEX_HOME"] = str(codex_home)

            codex_cmd = f"codex resume {shlex.quote(resume_id)}"
            cmd_dir = workspace or str(Path.home())
            command = (
                f"cd {shlex.quote(cmd_dir)} && {tmux_manager.shell_run_and_keep_open(codex_cmd)}"
            )

            created = tmux_manager.create_inner_window(
                "codex",
                tmux_manager.shell_command_with_env(command, env),
                terminal_id=terminal_id,
                provider=provider,
            )

        else:
            raise ValueError(f"Unsupported session type: {session_type}")

        if created:
            tmux_manager.focus_right_pane()
            try:
                db.get_or_create_agent(
                    terminal_id,
                    provider=provider,
                    session_type=session_type,
                    session_id=session_id,
                    cwd=workspace,
                    project_dir=workspace,
                    source=f"agent:{self.agent_id}",
                )
                # Insert windowlink so the agents panel can look up the title
                db.insert_window_link(
                    terminal_id=terminal_id,
                    agent_id=self.agent_id,
                    session_id=session_id,
                    provider=provider,
                    source=f"agent:{self.agent_id}",
                )
            except Exception:
                pass
            # Resuming a session implies the context is active again.
            try:
                info = db.get_agent_info(self.agent_id)
                visibility = str(getattr(info, "visibility", "visible") or "visible").strip()
                if visibility != "visible":
                    db.update_agent_info(self.agent_id, visibility="visible")
            except Exception:
                pass
            self._active_session_ids.add(session_id)
        else:
            raise RuntimeError("Failed to create terminal for resumed session")

    def _install_claude_hooks(self, workspace: str) -> None:
        try:
            from ...claude_hooks.stop_hook_installer import (
                ensure_stop_hook_installed,
                get_settings_path as get_stop_settings_path,
                install_stop_hook,
            )
            from ...claude_hooks.user_prompt_submit_hook_installer import (
                ensure_user_prompt_submit_hook_installed,
                get_settings_path as get_submit_settings_path,
                install_user_prompt_submit_hook,
            )
            from ...claude_hooks.permission_request_hook_installer import (
                ensure_permission_request_hook_installed,
                get_settings_path as get_permission_settings_path,
                install_permission_request_hook,
            )

            ensure_stop_hook_installed(quiet=True)
            ensure_user_prompt_submit_hook_installed(quiet=True)
            ensure_permission_request_hook_installed(quiet=True)

            project_root = Path(workspace)
            install_stop_hook(get_stop_settings_path(project_root), quiet=True)
            install_user_prompt_submit_hook(get_submit_settings_path(project_root), quiet=True)
            install_permission_request_hook(get_permission_settings_path(project_root), quiet=True)
        except Exception:
            pass

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "resume_session":
            return
        if event.state in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            if event.state == WorkerState.ERROR:
                self.app.notify(
                    str(event.worker.error) if event.worker.error else "Resume failed",
                    severity="error",
                )
            elif event.state == WorkerState.SUCCESS:
                self.app.notify("Session resumed", title="Resume")
                try:
                    from ..widgets.agents_panel import AgentsPanel

                    agents_panel = self.app.query_one(AgentsPanel)
                    agents_panel.refresh_data()
                except Exception:
                    pass
                self.action_close()
                return
            # Update button state for current selection
            try:
                table = self.query_one("#agent-sessions-table", DataTable)
                row_key = table.coordinate_to_cell_key(table.cursor_coordinate)[0]
                self._update_resume_button(str(row_key.value))
            except Exception:
                pass
