"""Agents Panel Widget - Lists agent profiles with their terminals."""

from __future__ import annotations

import asyncio
import json as _json
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from textual import events
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.timer import Timer
from textual.widgets import Button, Rule, Select, Static
from textual.widgets._select import SelectOverlay
from textual.message import Message
from textual.worker import Worker, WorkerState
from rich.text import Text

from .. import tmux_manager
from ...logging_config import setup_logger
from ..clipboard import copy_text
from ..branding import BRANDING
from ..system_prompts import build_claude_system_prompt, build_codex_instructions

logger = setup_logger("realign.dashboard.widgets.agents_panel", "dashboard.log")


@dataclass(frozen=True)
class _ProcessSnapshot:
    created_at_monotonic: float
    tty_key: str
    children: dict[int, list[int]]
    comms: dict[int, str]


class TruncButton(Button):
    """Button that truncates its label with '...' based on actual widget width."""

    class Hovered(Message, bubble=True):
        def __init__(self, button: "TruncButton", tooltip_text: str) -> None:
            super().__init__()
            self.button = button
            self.tooltip_text = tooltip_text

    class Unhovered(Message, bubble=True):
        def __init__(self, button: "TruncButton") -> None:
            super().__init__()
            self.button = button

    def __init__(
        self,
        full_text: str,
        *,
        tooltip_text: str | None = None,
        always_hover: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(full_text, **kwargs)
        self._full_text = full_text
        self._tooltip_text = tooltip_text
        self._always_hover = always_hover
        self._is_truncated = False

    def _is_inside_agents_panel(self) -> bool:
        parent = self.parent
        while parent is not None:
            if isinstance(parent, AgentsPanel):
                return True
            parent = parent.parent
        return False

    def render(self) -> Text:
        width = self.size.width
        gutter = self.styles.gutter
        avail = max(0, width - gutter.width)
        text = self._full_text
        self._is_truncated = len(text) > avail

        tooltip = self._tooltip_text if self._tooltip_text and self._is_truncated else None
        # In the real dashboard we render a custom tooltip via `AgentDescTooltip` to control
        # placement/width. Avoid also triggering Textual's built-in tooltip in that context.
        in_agents_panel = self._is_inside_agents_panel()
        mouse_over = False
        if in_agents_panel:
            try:
                mouse_over = bool(self.is_mouse_over)
            except Exception:
                # During teardown, widgets may no longer have a screen.
                mouse_over = False
        self.tooltip = None if (in_agents_panel and mouse_over) else tooltip

        if avail > 3 and self._is_truncated:
            return Text(text[: avail - 3] + "...")
        return Text(text)

    def on_enter(self, _event: events.Enter) -> None:
        if self._is_inside_agents_panel():
            self.tooltip = None
        if self._tooltip_text and (self._is_truncated or self._always_hover):
            self.post_message(self.Hovered(self, self._tooltip_text))

    def on_leave(self, _event: events.Leave) -> None:
        if self._is_inside_agents_panel():
            self.tooltip = self._tooltip_text if self._tooltip_text and self._is_truncated else None
        if self._tooltip_text:
            self.post_message(self.Unhovered(self))


class HoverButton(Button):
    """Button that shows a tooltip on hover."""

    class Hovered(Message, bubble=True):
        def __init__(self, button: "HoverButton", tooltip_text: str) -> None:
            super().__init__()
            self.button = button
            self.tooltip_text = tooltip_text

    class Unhovered(Message, bubble=True):
        def __init__(self, button: "HoverButton") -> None:
            super().__init__()
            self.button = button

    def __init__(self, *args, tooltip_text: str = "", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tooltip_text = tooltip_text

    def on_enter(self, _event: events.Enter) -> None:
        if self._tooltip_text:
            self.post_message(self.Hovered(self, self._tooltip_text))

    def on_leave(self, _event: events.Leave) -> None:
        if self._tooltip_text:
            self.post_message(self.Unhovered(self))


class ActionSelect(Select):
    """Select that hides the blank/prompt option from the dropdown overlay."""

    def _setup_options_renderables(self) -> None:
        from textual.widgets.option_list import Option

        options: list[Option] = []
        for prompt, value in self._options:
            if value is self.BLANK:
                options.append(Option("", disabled=True))
            else:
                options.append(Option(prompt))
        overlay = self.query_one(SelectOverlay)
        overlay.clear_options()
        overlay.add_options(options)


class AgentDescTooltip(Static):
    DEFAULT_CSS = """
    AgentDescTooltip {
        position: absolute;
        layer: tooltip;
        background: $surface;
        color: $text;
        border: round $surface-lighten-2;
        padding: 0 1;
        height: auto;
    }
    """

    def set_text(self, text: str) -> None:
        self.update(text)


class AgentNameButton(Button):
    """Button that emits a message on double-click."""

    class SingleClicked(Message, bubble=True):
        def __init__(self, button: "AgentNameButton", agent_id: str) -> None:
            super().__init__()
            self.button = button
            self.agent_id = agent_id

    class Hovered(Message, bubble=True):
        def __init__(self, button: "AgentNameButton", agent_id: str, description: str) -> None:
            super().__init__()
            self.button = button
            self.agent_id = agent_id
            self.description = description

    class Unhovered(Message, bubble=True):
        def __init__(self, button: "AgentNameButton") -> None:
            super().__init__()
            self.button = button

    def __init__(self, *args, description: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._description = description
        self._single_click_timer: Optional[Timer] = None

    class DoubleClicked(Message, bubble=True):
        def __init__(self, button: "AgentNameButton", agent_id: str) -> None:
            super().__init__()
            self.button = button
            self.agent_id = agent_id

    def _emit_single_click(self) -> None:
        self._single_click_timer = None
        self.post_message(self.SingleClicked(self, self.name or ""))

    async def _on_click(self, event: events.Click) -> None:
        await super()._on_click(event)
        if event.chain >= 2:
            if self._single_click_timer is not None:
                try:
                    self._single_click_timer.stop()
                except Exception:
                    pass
                self._single_click_timer = None
            self.post_message(self.DoubleClicked(self, self.name or ""))
            return

        # Delay single-click so a double-click can cancel it.
        if self._single_click_timer is not None:
            try:
                self._single_click_timer.stop()
            except Exception:
                pass
        self._single_click_timer = self.set_timer(0.25, self._emit_single_click)

    def on_enter(self, _event: events.Enter) -> None:
        if self._description:
            self.post_message(self.Hovered(self, self.name or "", self._description))

    def on_leave(self, _event: events.Leave) -> None:
        if self._description:
            self.post_message(self.Unhovered(self))


class _TerminalRowWidget(Horizontal):
    def __init__(
        self,
        *,
        terminal_id: str,
        term_safe_id: str,
        prefix: str,
        title: str,
        running_agent: str | None,
        pane_current_command: str | None,
        is_active: bool,
    ) -> None:
        super().__init__(classes="terminal-row")

        switch_classes = "terminal-switch active-terminal" if is_active else "terminal-switch"

        pane_cmd = (pane_current_command or "").strip().lower()
        title_is_relevant = bool(title) and (
            bool(running_agent) or (pane_cmd in {"claude", "codex", "opencode"}) or (not pane_cmd)
        )

        if title_is_relevant:
            label = f"{prefix}{title}"
            switch_btn: Button = TruncButton(
                label,
                id=f"switch-{term_safe_id}",
                name=terminal_id,
                classes=switch_classes,
                tooltip_text=title,
            )
        elif running_agent:
            provider = (running_agent or "").strip().lower()
            provider_label = (running_agent or "").strip().capitalize()
            switch_btn = TruncButton(
                f"{prefix}{provider_label}",
                id=f"switch-{term_safe_id}",
                name=terminal_id,
                classes=switch_classes,
            )
        else:
            display = (pane_current_command or "").strip() or "Terminal"
            switch_btn = TruncButton(
                f"{prefix}{display}",
                id=f"switch-{term_safe_id}",
                name=terminal_id,
                classes=switch_classes,
            )

        self._switch_btn = switch_btn
        self._close_btn = HoverButton(
            "✕",
            id=f"close-{term_safe_id}",
            name=terminal_id,
            variant="error",
            classes="terminal-close",
            tooltip_text="Close Terminal",
        )

    def compose(self) -> ComposeResult:
        yield self._switch_btn
        yield self._close_btn


class _TerminalConnectorWidget(Horizontal):
    def __init__(self) -> None:
        super().__init__(classes="terminal-row")

    def compose(self) -> ComposeResult:
        yield Button("│", classes="terminal-switch")


class _AgentBlockWidget(Container):
    def __init__(self, *, agent: dict, active_terminal_id: str | None, expanded: bool) -> None:
        safe_id = AgentsPanel._safe_id(agent["id"])
        super().__init__(id=f"agent-block-{safe_id}", classes="agent-block")
        self._agent = agent
        self._active_terminal_id = active_terminal_id
        self._expanded = bool(expanded)
        self._visibility = (agent.get("visibility") or "visible").strip() or "visible"
        self._safe_id = safe_id

    def _has_terminals(self) -> bool:
        return bool(self._agent.get("terminals") or [])

    def _terminal_widgets(self) -> list[Horizontal]:
        terminals = self._agent.get("terminals") or []
        widgets: list[Horizontal] = []
        for term_idx, term in enumerate(terminals):
            is_last_term = term_idx == len(terminals) - 1
            prefix = "└ " if is_last_term else "├ "

            terminal_id = term["terminal_id"]
            term_safe_id = AgentsPanel._safe_id(terminal_id)
            is_active = terminal_id == self._active_terminal_id

            widgets.append(
                _TerminalRowWidget(
                    terminal_id=terminal_id,
                    term_safe_id=term_safe_id,
                    prefix=prefix,
                    title=term.get("title", "") or "",
                    running_agent=term.get("running_agent"),
                    pane_current_command=term.get("pane_current_command"),
                    is_active=is_active,
                )
            )
            if not is_last_term:
                widgets.append(_TerminalConnectorWidget())
        return widgets

    def _update_agent_label(self) -> None:
        try:
            agent_btn = self.query_one(f"#agent-{self._safe_id}", AgentNameButton)
        except Exception:
            return

        if self._has_terminals():
            arrow = "▾" if self._expanded else "▸"
        else:
            arrow = " "
        style = "bold dim" if self._visibility != "visible" else "bold"
        agent_btn.label = Text(f"{arrow} {self._agent['name']}", style=style)

    async def set_expanded(self, expanded: bool) -> None:
        if not self._has_terminals():
            return
        expanded = bool(expanded)
        if expanded == self._expanded:
            return

        self._expanded = expanded
        self._update_agent_label()

        try:
            term_list = self.query_one(f"#agent-terminals-{self._safe_id}", Vertical)
        except Exception:
            return

        if self._expanded:
            term_list.display = True
            if not term_list.children:
                await term_list.mount_all(self._terminal_widgets())
        else:
            try:
                await term_list.remove_children()
            except Exception:
                pass
            term_list.display = False

        try:
            term_list.refresh(layout=True)
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        agent = self._agent
        safe_id = self._safe_id
        is_archived = self._visibility != "visible"

        with Horizontal(classes="agent-row"):
            arrow = (
                "▾"
                if (self._expanded and self._has_terminals())
                else ("▸" if self._has_terminals() else " ")
            )
            name_style = "bold dim" if is_archived else "bold"
            name_label = Text(f"{arrow} {agent['name']}", style=name_style)
            agent_btn = AgentNameButton(
                name_label,
                id=f"agent-{safe_id}",
                name=agent["id"],
                classes="agent-name",
                description=agent.get("description", ""),
            )
            yield agent_btn

            if not is_archived:
                # Phase 0 UX: one button. First click shares; subsequent clicks sync.
                yield HoverButton(
                    "Share",
                    id=f"share-{safe_id}",
                    name=agent["id"],
                    classes="agent-share",
                )

                # Direct button to create new session
                yield HoverButton(
                    "+",
                    id=f"create-term-{safe_id}",
                    name=agent["id"],
                    classes="agent-create",
                    tooltip_text="New Session",
                )

                # Button to resume session
                yield HoverButton(
                    "↩",
                    id=f"resume-term-{safe_id}",
                    name=agent["id"],
                    classes="agent-resume",
                    tooltip_text="Resume Session",
                )
            if is_archived:
                yield HoverButton(
                    "↩",
                    id=f"delete-{safe_id}",
                    name=agent["id"],
                    classes="agent-delete",
                    tooltip_text="Restore Agent",
                )
            else:
                yield HoverButton(
                    "x",
                    id=f"delete-{safe_id}",
                    name=agent["id"],
                    classes="agent-delete",
                    tooltip_text="Archive",
                )

        agent_title = agent.get("title") or ""
        if agent_title:
            # Build tooltip: bold title + blank line + description
            tooltip_parts = [f"[bold]{agent_title}[/bold]"]
            if agent.get("description"):
                tooltip_parts.append("")
                tooltip_parts.append(agent["description"])
            tooltip_text = "\n".join(tooltip_parts)
            yield TruncButton(
                f"  {agent_title}",
                id=f"agent-title-{safe_id}",
                name=agent["id"],
                classes="agent-title",
                tooltip_text=tooltip_text,
                always_hover=True,
            )

        terminals = agent.get("terminals") or []
        if not terminals:
            return

        with Vertical(
            classes="terminal-list",
            id=f"agent-terminals-{safe_id}",
        ) as term_list:
            term_list.display = self._expanded
            if self._expanded:
                for widget in self._terminal_widgets():
                    yield widget


class AgentsPanel(Container, can_focus=True):
    """Panel displaying agent profiles with their associated terminals."""

    REFRESH_INTERVAL_SECONDS = 2.0
    TOOLTIP_HOVER_DELAY_SECONDS = 1.0
    DEFAULT_CSS = """
    AgentsPanel {
        height: 100%;
        padding: 0 1;
        overflow: hidden;
    }

    AgentsPanel:focus {
        border: none;
    }

    AgentsPanel .summary {
        height: auto;
        margin: 0;
        padding: 0;
        background: transparent;
        border: none;
        align: right middle;
    }

    AgentsPanel Button {
        min-width: 0;
        padding: 0 1;
        background: transparent;
        border: none;
    }

    AgentsPanel Button:hover {
        background: $surface-lighten-1;
    }

    AgentsPanel .summary Button {
        width: auto;
        margin-right: 1;
    }

    AgentsPanel .list {
        height: 1fr;
        padding: 0;
        overflow-y: auto;
        border: none;
        background: transparent;
    }

    AgentsPanel Rule {
        margin: 0;
    }

    AgentsPanel .agent-row {
        height: 1;
        margin: 0 0 1 0;
    }

    AgentsPanel .agent-block {
        height: auto;
        margin-bottom: 1;
    }

    AgentsPanel Button.agent-title {
        width: 1fr;
        height: 1;
        margin: 0 0 1 0;
        padding: 0 1;
        color: $text-muted;
        text-style: italic;
        text-align: left;
        content-align: left middle;
    }

    AgentsPanel Button.agent-title:hover {
        color: $text-muted;
        text-style: italic;
    }

    AgentsPanel .agent-row Button.agent-name {
        width: 1fr;
        height: 1;
        margin: 0;
        padding: 0 1;
        text-align: left;
        content-align: left middle;
    }

    AgentsPanel .agent-row Select.agent-create {
        width: 5;
        min-width: 5;
        height: 1;
        margin: 0;
        padding: 0;
        border: none;
    }

    AgentsPanel .agent-row Select.agent-create > SelectCurrent {
        width: 5;
        min-width: 5;
        height: 1;
        margin: 0;
        padding: 0;
        border: none;
        content-align: center middle;
    }

    AgentsPanel .agent-row Select.agent-create > SelectCurrent Static#label {
        text-align: center;
        width: 1fr;
    }

    AgentsPanel .agent-row Select.agent-create > SelectCurrent .arrow {
        display: none;
    }

    AgentsPanel .agent-row Select.agent-create > SelectOverlay {
        width: 20;
        offset-x: -12;
    }

    AgentsPanel .agent-row Select.agent-create > SelectOverlay .option-list--option-disabled {
        display: none;
    }

    AgentsPanel .agent-row Button.agent-delete {
        width: 3;
        min-width: 3;
        height: 1;
        margin-left: 0;
        padding: 0;
        content-align: center middle;
    }

    AgentsPanel .agent-row Button.agent-share {
        width: auto;
        min-width: 6;
        height: 1;
        margin-left: 0;
        padding: 0 1;
        content-align: center middle;
    }

    AgentsPanel .terminal-list {
        margin: 0 0 1 2;
        padding: 0;
        height: auto;
        background: transparent;
        border: none;
    }

    AgentsPanel .terminal-row {
        height: 1;
        margin: 0;
    }

    AgentsPanel .terminal-row Button.terminal-switch {
        width: 1fr;
        height: 1;
        margin: 0;
        padding: 0 1;
        text-align: left;
        content-align: left middle;
        text-opacity: 60%;
    }

    AgentsPanel .terminal-row Button.terminal-switch:hover {
        text-opacity: 100%;
    }

    AgentsPanel .terminal-row Button.active-terminal {
        text-opacity: 100%;
        text-style: bold;
    }

    AgentsPanel .terminal-row Button.terminal-close {
        width: 3;
        min-width: 3;
        height: 1;
        margin: 0;
        padding: 0;
        content-align: center middle;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._refresh_lock = asyncio.Lock()
        self._agents: list[dict] = []
        self._active_terminal_id: str | None = None
        self._rendered_fingerprint: str = ""
        self._switch_worker: Optional[Worker] = None
        self._switch_seq: int = 0
        self._refresh_worker: Optional[Worker] = None
        self._render_worker: Optional[Worker] = None
        self._pending_render: bool = False
        self._pending_render_reason: str | None = None
        self._share_worker: Optional[Worker] = None
        self._sync_worker: Optional[Worker] = None
        self._share_agent_id: Optional[str] = None
        self._share_spinner_timer: Optional[Timer] = None
        self._share_spinner_frame: int = 0
        self._sync_agent_id: Optional[str] = None
        self._refresh_timer = None
        self._last_refresh_error_at: float | None = None
        self._last_render_started_at: float | None = None
        self._last_render_completed_at: float | None = None
        self._process_snapshot: _ProcessSnapshot | None = None
        self._collapsed_agent_ids: set[str] = set()
        self._tooltip_timer: Optional[Timer] = None
        self._pending_tooltip: (
            tuple[Button, str, Literal["agent_desc", "trunc", "hover"]] | None
        ) = None
        self._tooltip_owner: Button | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="summary"):
            yield Button("Add Context", id="create-agent", variant="primary")
        with Vertical(id="agents-list", classes="list"):
            yield Static("No contexts yet. Click 'Add Context' to add one.")
        tip = AgentDescTooltip("", id="agent-desc-tooltip")
        tip.display = False
        yield tip

    def on_mount(self) -> None:
        btn = self.query_one("#create-agent", Button)
        btn.styles.border = ("round", "black")
        btn.styles.text_align = "center"
        btn.styles.content_align = ("center", "middle")

    def _cancel_tooltip_timer(self) -> None:
        if self._tooltip_timer is None:
            return
        try:
            self._tooltip_timer.stop()
        except Exception:
            pass
        self._tooltip_timer = None

    def _hide_tooltip(self) -> None:
        try:
            tip = self.query_one("#agent-desc-tooltip", AgentDescTooltip)
            tip.display = False
        except Exception:
            pass
        self._tooltip_owner = None

    def _show_tooltip_now(
        self, *, btn: Button, text: str, kind: Literal["agent_desc", "trunc", "hover"]
    ) -> None:
        if not text.strip():
            return
        if not self._is_button_mouse_over(btn):
            return
        try:
            tip = self.query_one("#agent-desc-tooltip", AgentDescTooltip)
            tip.set_text(text)

            origin = self.content_region.offset
            rel_x = btn.region.offset.x - origin.x
            rel_y = btn.region.offset.y - origin.y + btn.region.height

            if kind in {"agent_desc", "trunc"}:
                tip.styles.width = btn.region.width
                tip.styles.offset = (rel_x, rel_y)
            else:
                # text + padding (1 each side) + border (1 each side)
                tip_width = len(text) + 4
                panel_width = self.content_region.width
                if rel_x + tip_width > panel_width:
                    rel_x = max(0, panel_width - tip_width)

                tip.styles.width = tip_width
                tip.styles.offset = (rel_x, rel_y)

            tip.display = True
            tip.refresh(layout=True)
            self._tooltip_owner = btn
        except Exception:
            return

    def _is_button_mouse_over(self, btn: Button) -> bool:
        try:
            return bool(getattr(btn, "is_mouse_over", False))
        except Exception:
            # Guard against Textual lifecycle race (e.g., NoScreen during unmount).
            return False

    def _show_pending_tooltip(self) -> None:
        self._tooltip_timer = None
        pending = self._pending_tooltip
        if pending is None:
            return
        btn, text, kind = pending
        # If the cursor moved off the button before the delay elapsed, don't show anything.
        if not self._is_button_mouse_over(btn):
            return
        self._show_tooltip_now(btn=btn, text=text, kind=kind)

    def _schedule_tooltip(
        self, *, btn: Button, text: str, kind: Literal["agent_desc", "trunc", "hover"]
    ) -> None:
        if not text.strip():
            return
        self._cancel_tooltip_timer()
        self._pending_tooltip = (btn, text, kind)

        if self._tooltip_owner is not None and self._tooltip_owner is not btn:
            self._hide_tooltip()

        self._tooltip_timer = self.set_timer(
            self.TOOLTIP_HOVER_DELAY_SECONDS, self._show_pending_tooltip
        )

    def _clear_tooltip_for_button(self, btn: Button) -> None:
        pending = self._pending_tooltip
        if pending is not None and pending[0] is btn:
            self._pending_tooltip = None
            self._cancel_tooltip_timer()
        if self._tooltip_owner is btn:
            self._hide_tooltip()

    def on_agent_name_button_hovered(self, message: AgentNameButton.Hovered) -> None:
        self._schedule_tooltip(
            btn=message.button,
            text=message.description,
            kind="agent_desc",
        )

    def on_agent_name_button_unhovered(self, message: AgentNameButton.Unhovered) -> None:
        self._clear_tooltip_for_button(message.button)

    def on_trunc_button_hovered(self, message: TruncButton.Hovered) -> None:
        self._schedule_tooltip(btn=message.button, text=message.tooltip_text, kind="trunc")

    def on_trunc_button_unhovered(self, message: TruncButton.Unhovered) -> None:
        self._clear_tooltip_for_button(message.button)

    def on_hover_button_hovered(self, message: HoverButton.Hovered) -> None:
        self._schedule_tooltip(btn=message.button, text=message.tooltip_text, kind="hover")

    def on_hover_button_unhovered(self, message: HoverButton.Unhovered) -> None:
        self._clear_tooltip_for_button(message.button)

    def on_show(self) -> None:
        if self._refresh_timer is None:
            # Refresh frequently, but avoid hammering SQLite/tmux (can cause transient empty UI).
            self._refresh_timer = self.set_interval(
                self.REFRESH_INTERVAL_SECONDS, self._on_refresh_timer
            )
        else:
            try:
                self._refresh_timer.resume()
            except Exception:
                pass

        self.refresh_data()

    def on_hide(self) -> None:
        if self._refresh_timer is not None:
            try:
                self._refresh_timer.pause()
            except Exception:
                pass
        self._pending_tooltip = None
        self._cancel_tooltip_timer()
        self._hide_tooltip()

    def on_unmount(self) -> None:
        self._pending_tooltip = None
        self._cancel_tooltip_timer()
        self._hide_tooltip()

    def _on_refresh_timer(self) -> None:
        self.refresh_data()

    def refresh_data(self) -> None:
        if not self.display:
            return
        if self._refresh_worker is not None and self._refresh_worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        ):
            return
        self._refresh_worker = self.run_worker(
            self._collect_agents, thread=True, exit_on_error=False
        )

    def diagnostics_state(self) -> dict[str, object]:
        """Small, safe snapshot for watchdog logging (never raises)."""
        state: dict[str, object] = {}
        try:
            state["agents_count"] = int(len(self._agents))
            state["rendered_fingerprint_len"] = int(len(self._rendered_fingerprint))
            state["refresh_worker_state"] = str(
                getattr(getattr(self, "_refresh_worker", None), "state", None)
            )
            state["render_worker_state"] = str(
                getattr(getattr(self, "_render_worker", None), "state", None)
            )
            state["pending_render"] = bool(self._pending_render)
            state["last_refresh_error_at"] = self._last_refresh_error_at
            state["last_render_started_at"] = self._last_render_started_at
            state["last_render_completed_at"] = self._last_render_completed_at
        except Exception:
            pass
        try:
            container = self.query_one("#agents-list", Vertical)
            state["agents_list_children"] = int(len(getattr(container, "children", []) or []))
        except Exception:
            pass
        return state

    def _ui_blank(self) -> bool:
        try:
            container = self.query_one("#agents-list", Vertical)
            return len(container.children) == 0
        except Exception:
            return False

    def _schedule_render(self, *, reason: str) -> None:
        if not self.display:
            return

        if self._render_worker is not None and self._render_worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        ):
            self._pending_render = True
            self._pending_render_reason = reason
            return

        self._pending_render = False
        self._pending_render_reason = None
        self._last_render_started_at = time.monotonic()
        self._render_worker = self.run_worker(
            self._render_agents(), group="agents-render", exit_on_error=False
        )

    def force_render(self, *, reason: str) -> None:
        """Force a re-render even if the collected data didn't change."""
        # Reset fingerprint so the next refresh doesn't treat the UI as "up to date".
        self._rendered_fingerprint = ""
        if self._agents:
            self._schedule_render(reason=reason)

    @staticmethod
    def _detect_running_agent(
        pane_pid: int | None, snapshot: _ProcessSnapshot | None
    ) -> str | None:
        """Detect whether a known agent process exists under a tmux pane PID.

        Uses a cached process snapshot built once per refresh to avoid spawning `ps`
        repeatedly for every terminal row.
        """
        if not pane_pid or snapshot is None:
            return None

        children = snapshot.children
        comms = snapshot.comms
        stack = [pane_pid]
        seen: set[int] = set()
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            comm = (comms.get(p) or "").lower()
            for kw in ("claude", "codex", "opencode"):
                if kw in comm:
                    return kw
            stack.extend(children.get(p, []))
        return None

    @staticmethod
    def _normalize_tty(tty: str) -> str:
        s = (tty or "").strip()
        if not s:
            return ""
        if s.startswith("/dev/"):
            s = s[len("/dev/") :]
        return s.strip()

    def _get_process_snapshot(self, *, pane_ttys: set[str]) -> _ProcessSnapshot | None:
        """Return a cached process snapshot scoped to the given TTY set.

        TTL is aligned with the panel refresh interval so the snapshot is at most one refresh stale.
        """
        ttys = sorted(
            {self._normalize_tty(t) for t in (pane_ttys or set()) if self._normalize_tty(t)}
        )
        if not ttys:
            return None
        tty_key = ",".join(ttys)

        now = time.monotonic()
        cached = self._process_snapshot
        if (
            cached
            and cached.tty_key == tty_key
            and (now - cached.created_at_monotonic) < float(self.REFRESH_INTERVAL_SECONDS)
        ):
            return cached

        try:
            import subprocess

            # Prefer TTY-limited `ps` to keep output small. BSD ps supports comma-separated TTY list.
            result = subprocess.run(
                ["ps", "-o", "pid=,ppid=,comm=", "-t", tty_key],
                capture_output=True,
                text=True,
                timeout=2,
            )
            stdout = result.stdout or ""
            if result.returncode != 0 or not stdout.strip():
                # Fallback: some environments may not accept `-t` list; fall back to a single
                # snapshot of all processes and filter by TTY.
                result = subprocess.run(
                    ["ps", "-axo", "pid=,ppid=,tty=,comm="],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                stdout = result.stdout or ""

                allowed = set(ttys)
                filtered_lines: list[str] = []
                for line in stdout.splitlines():
                    parts = line.split(None, 3)
                    if len(parts) < 4:
                        continue
                    tty = self._normalize_tty(parts[2])
                    if tty and tty in allowed:
                        filtered_lines.append(f"{parts[0]} {parts[1]} {parts[3]}")
                stdout = "\n".join(filtered_lines)

            children: dict[int, list[int]] = {}
            comms: dict[int, str] = {}
            for line in stdout.splitlines():
                parts = line.split(None, 2)
                if len(parts) < 3:
                    continue
                try:
                    pid, ppid = int(parts[0]), int(parts[1])
                except ValueError:
                    continue
                children.setdefault(ppid, []).append(pid)
                comms[pid] = parts[2]

            snap = _ProcessSnapshot(
                created_at_monotonic=now,
                tty_key=tty_key,
                children=children,
                comms=comms,
            )
            self._process_snapshot = snap
            return snap
        except Exception:
            return None

    def _collect_agents(self) -> dict:
        """Collect agent info with their terminals."""
        agents: list[dict] = []

        from ...db import get_database

        show_archived = False
        try:
            from ..state import get_dashboard_state_value

            show_archived = bool(get_dashboard_state_value("show_archived_agents", False))
        except Exception:
            show_archived = False

        # Dashboard should prefer correctness/stability over ultra-low lock timeouts.
        #
        # Important: `get_database(read_only=True)` returns a lightweight instance and does not
        # use the global singleton. Close it after use so we don't leak SQLite connections
        # (which can manifest as intermittent "unable to open database file" under WAL).
        db = get_database(read_only=True, connect_timeout_seconds=2.0)
        # Critical: if this fails, let it surface as a worker ERROR so we can keep
        # the last rendered UI instead of flashing an empty agent list.
        try:
            agent_infos = db.list_agent_info(include_invisible=show_archived)
        except TypeError:
            # Some tests use a minimal DB fake that doesn't accept keyword args.
            agent_infos = db.list_agent_info()

        # Best-effort: missing pieces should degrade gracefully (names still render).
        try:
            active_terminals = db.list_agents(status="active", limit=1000)
        except Exception as e:
            logger.debug(f"Failed to list active terminals: {e}")
            active_terminals = []

        try:
            latest_links = db.list_latest_window_links(limit=2000)
        except Exception:
            latest_links = []
        link_by_terminal = {
            l.terminal_id: l for l in latest_links if getattr(l, "terminal_id", None)
        }

        tmux_windows_ok = True
        try:
            tmux_windows = tmux_manager.list_inner_windows()
        except Exception as e:
            logger.debug(f"Failed to list tmux windows: {e}")
            tmux_windows_ok = False
            tmux_windows = []
        terminal_to_window = {
            w.terminal_id: w for w in tmux_windows if getattr(w, "terminal_id", None)
        }

        # If we are running inside the managed tmux dashboard environment, only show terminals
        # that still exist in the inner tmux server. This prevents stale buttons when a user
        # exits a shell directly (tmux window disappears but DB record may still be "active").
        try:
            if tmux_manager.managed_env_enabled() and tmux_windows_ok:
                present = {tid for tid in terminal_to_window.keys() if tid}
                if present:
                    active_terminals = [
                        t for t in active_terminals if getattr(t, "id", None) in present
                    ]
                else:
                    # Inner session exists but no tracked terminals are present.
                    active_terminals = []
        except Exception:
            pass

        pane_ttys: set[str] = set()
        for w in tmux_windows:
            if getattr(w, "pane_pid", None) and getattr(w, "pane_tty", None):
                pane_ttys.add(str(w.pane_tty))
        ps_snapshot = self._get_process_snapshot(pane_ttys=pane_ttys) if pane_ttys else None

        # Detect currently active terminal
        active_window = next((w for w in tmux_windows if w.active), None)
        active_terminal_id = active_window.terminal_id if active_window else None

        # Collect all session_ids for title lookup
        session_ids: list[str] = []
        for t in active_terminals:
            link = link_by_terminal.get(t.id)
            if link and getattr(link, "session_id", None):
                session_ids.append(link.session_id)
                continue
            window = terminal_to_window.get(t.id)
            if window and getattr(window, "session_id", None):
                session_ids.append(window.session_id)
                continue
            # Fallback: agent record (works even when WindowLink isn't available yet / at all).
            agent_session_id = getattr(t, "session_id", None)
            if agent_session_id:
                session_ids.append(agent_session_id)

        titles: dict[str, str] = {}
        session_agent_id: dict[str, str] = {}
        try:
            uniq_session_ids = sorted({sid for sid in session_ids if sid})
            if uniq_session_ids:
                sessions = db.get_sessions_by_ids(uniq_session_ids)
                for s in sessions:
                    sid = getattr(s, "id", None)
                    if not sid:
                        continue
                    title = (getattr(s, "session_title", "") or "").strip()
                    if title:
                        titles[sid] = title
                    agent_id = getattr(s, "agent_id", None)
                    if agent_id:
                        session_agent_id[sid] = agent_id
        except Exception:
            titles = {}
            session_agent_id = {}

        # Map agent_info.id -> list of terminals
        agent_to_terminals: dict[str, list[dict]] = {}
        for t in active_terminals:
            # Find which agent_info this terminal belongs to
            agent_info_id = None

            link = link_by_terminal.get(t.id)

            # Method 1: Check source field for "agent:{agent_info_id}" format
            source = t.source or ""
            if source.startswith("agent:"):
                agent_info_id = source[6:]

            # Method 2: WindowLink agent_id
            if not agent_info_id and link and getattr(link, "agent_id", None):
                agent_info_id = link.agent_id

            # Method 3: Fallback - check tmux window's session.agent_id
            if not agent_info_id:
                window = terminal_to_window.get(t.id)
                if window and getattr(window, "session_id", None):
                    agent_info_id = session_agent_id.get(window.session_id) or None

            if agent_info_id:
                agent_to_terminals.setdefault(agent_info_id, [])

                # Get session_id from windowlink (preferred) or tmux window
                window = terminal_to_window.get(t.id)
                session_id = (
                    link.session_id
                    if link and getattr(link, "session_id", None)
                    else (
                        window.session_id
                        if window and getattr(window, "session_id", None)
                        else getattr(t, "session_id", None)
                    )
                )
                title = titles.get(session_id, "") if session_id else ""

                pane_cmd = window.pane_current_command if window else None
                provider = (
                    link.provider
                    if link and getattr(link, "provider", None)
                    else (t.provider or "")
                )
                pane_pid = window.pane_pid if window else None
                running_agent = self._detect_running_agent(pane_pid, ps_snapshot)

                entry = {
                    "terminal_id": t.id,
                    "window_id": window.window_id if window else None,
                    "session_id": session_id,
                    "provider": provider,
                    "session_type": t.session_type or "",
                    "title": title,
                    "cwd": t.cwd or "",
                    "running_agent": running_agent,
                    "pane_current_command": pane_cmd,
                }
                agent_to_terminals[agent_info_id].append(entry)

        for info in agent_infos:
            terminals = agent_to_terminals.get(info.id, [])
            agents.append(
                {
                    "id": info.id,
                    "name": info.name,
                    "title": info.title or "",
                    "description": info.description or "",
                    "visibility": getattr(info, "visibility", "visible") or "visible",
                    "terminals": terminals,
                    "share_url": getattr(info, "share_url", None),
                    "last_synced_at": getattr(info, "last_synced_at", None),
                }
            )
        # Sort: active (visible) agents first, then archived (invisible)
        agents.sort(key=lambda a: (0 if a["visibility"] == "visible" else 1))
        try:
            db.close()
        except Exception:
            pass
        return {"agents": agents, "active_terminal_id": active_terminal_id}

    @staticmethod
    def _fingerprint(agents: list[dict]) -> str:
        """Fast serialisation used only for change detection."""
        try:
            return _json.dumps(agents, sort_keys=True, default=str)
        except Exception:
            return ""

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        # Handle switch worker
        if self._switch_worker is not None and event.worker is self._switch_worker:
            if event.state == WorkerState.SUCCESS:
                result = self._switch_worker.result or {}
                if isinstance(result, dict):
                    seq = int(result.get("seq", 0) or 0)
                    if seq != self._switch_seq:
                        return
                    terminal_id = str(result.get("terminal_id", "") or "").strip() or None
                    ok = bool(result.get("ok", False))
                    if ok and terminal_id:
                        self._active_terminal_id = terminal_id
                        self._update_active_terminal_ui(terminal_id)
                        # Prevent the next refresh from forcing a full re-render just due to
                        # active-terminal changes.
                        self._rendered_fingerprint = (
                            self._fingerprint(self._agents) + f"|active:{self._active_terminal_id}"
                        )
                    else:
                        msg = str(result.get("error", "") or "Failed to switch").strip()
                        self.app.notify(msg, title="Agent", severity="error")
                        self.refresh_data()
            elif event.state == WorkerState.ERROR:
                err = self._switch_worker.error
                msg = "Failed to switch"
                if isinstance(err, BaseException):
                    msg = f"Failed to switch: {err}"
                elif err:
                    msg = f"Failed to switch: {err}"
                self.app.notify(msg, title="Agent", severity="error")
                self.refresh_data()

            if event.state in {
                WorkerState.SUCCESS,
                WorkerState.ERROR,
                WorkerState.CANCELLED,
            }:
                self._switch_worker = None
            return

        # Handle refresh worker
        if self._refresh_worker is not None and event.worker is self._refresh_worker:
            if event.state == WorkerState.ERROR:
                # Keep the last successfully-rendered list on refresh errors to avoid
                # flashing an empty Agents tab during transient tmux/SQLite hiccups.
                self._last_refresh_error_at = time.monotonic()
                err = self._refresh_worker.error
                if isinstance(err, BaseException):
                    logger.warning(
                        "Agents refresh failed",
                        exc_info=(type(err), err, err.__traceback__),
                    )
                else:
                    logger.warning(f"Agents refresh failed: {err}")
                return
            elif event.state == WorkerState.SUCCESS:
                prev_agents = self._agents
                prev_active = self._active_terminal_id
                payload = self._refresh_worker.result or {}
                if isinstance(payload, dict) and "agents" in payload:
                    self._agents = payload.get("agents") or []
                    self._active_terminal_id = payload.get("active_terminal_id") or None
                else:
                    self._agents = payload or []
                self._last_refresh_error_at = None
                try:
                    current_ids = {
                        str(a.get("id") or "")
                        for a in (self._agents or [])
                        if isinstance(a, dict) and a.get("id")
                    }
                    self._collapsed_agent_ids.intersection_update(current_ids)
                except Exception:
                    # Never drop the agents list due to expand/collapse bookkeeping.
                    pass
            else:
                return

            # Fast path: only the active terminal changed; update button classes in-place
            # to avoid full re-render flicker / input lag.
            try:
                prev_agents_fp = self._fingerprint(prev_agents)
                new_agents_fp = self._fingerprint(self._agents)
                if (
                    prev_agents_fp == new_agents_fp
                    and prev_active != self._active_terminal_id
                    and not self._ui_blank()
                ):
                    self._update_active_terminal_ui(self._active_terminal_id)
                    self._rendered_fingerprint = (
                        new_agents_fp + f"|active:{self._active_terminal_id}"
                    )
                    return
            except Exception:
                new_agents_fp = self._fingerprint(self._agents)

            fp = new_agents_fp + f"|active:{self._active_terminal_id}"
            # Important: the UI can become blank due to cancellation or transient Textual issues.
            # If data didn't change but the widget tree is empty, force a re-render.
            should_render = (fp != self._rendered_fingerprint) or (
                self._agents and self._ui_blank()
            )
            if not should_render:
                return  # nothing changed – skip re-render to avoid flicker
            self._rendered_fingerprint = fp
            self._schedule_render(reason="refresh")
            return

        # Handle render worker
        if self._render_worker is not None and event.worker is self._render_worker:
            if event.state == WorkerState.ERROR:
                err = self._render_worker.error
                if isinstance(err, BaseException):
                    logger.warning(
                        "Agents render failed",
                        exc_info=(type(err), err, err.__traceback__),
                    )
                else:
                    logger.warning(f"Agents render failed: {err}")
                # Force next refresh to re-render even if data is unchanged.
                self._rendered_fingerprint = ""
            elif event.state == WorkerState.SUCCESS:
                self._last_render_completed_at = time.monotonic()
            elif event.state == WorkerState.CANCELLED:
                self._rendered_fingerprint = ""

            if event.state in {
                WorkerState.SUCCESS,
                WorkerState.ERROR,
                WorkerState.CANCELLED,
            }:
                self._last_render_completed_at = time.monotonic()
                self._render_worker = None
                if self._pending_render:
                    reason = self._pending_render_reason or "pending"
                    self._schedule_render(reason=reason)
            return

        # Handle share worker
        if self._share_worker is not None and event.worker is self._share_worker:
            self._handle_share_worker_state_changed(event)

        # Handle sync worker
        if self._sync_worker is not None and event.worker is self._sync_worker:
            self._handle_sync_worker_state_changed(event)

    async def _render_agents(self) -> None:
        async with self._refresh_lock:
            try:
                container = self.query_one("#agents-list", Vertical)
                with self.app.batch_update():
                    await container.remove_children()

                    if not self._agents:
                        await container.mount(
                            Static("No contexts yet. Click 'Add Context' to add one.")
                        )
                        return

                    blocks = [
                        _AgentBlockWidget(
                            agent=agent,
                            active_terminal_id=self._active_terminal_id,
                            expanded=(agent.get("id") not in self._collapsed_agent_ids),
                        )
                        for agent in self._agents
                    ]
                    await container.mount_all(blocks)
            except Exception:
                logger.exception("Failed to render agents list")
                try:
                    container = self.query_one("#agents-list", Vertical)
                    await container.remove_children()
                    await container.mount(
                        Static("Agents UI error (see ~/.aline/.logs/dashboard.log)")
                    )
                except Exception:
                    pass
                return
            except asyncio.CancelledError:
                # Best-effort: avoid leaving the UI in a permanently blank state.
                logger.warning("Agents render cancelled")
                self._rendered_fingerprint = ""
                return

    @staticmethod
    def _short_id(val: str | None) -> str:
        if not val:
            return ""
        if len(val) > 20:
            return val[:8] + "..." + val[-8:]
        return val

    def _fetch_session_titles(self, session_ids: list[str]) -> dict[str, str]:
        """Fetch session titles from database (same method as Terminal panel)."""
        if not session_ids:
            return {}
        try:
            from ...db import get_database

            db = get_database(read_only=True)
            try:
                sessions = db.get_sessions_by_ids(session_ids)
                titles: dict[str, str] = {}
                for s in sessions:
                    title = (s.session_title or "").strip()
                    if title:
                        titles[s.id] = title
                return titles
            finally:
                try:
                    db.close()
                except Exception:
                    pass
        except Exception:
            return {}

    @staticmethod
    def _safe_id(raw: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "-", raw).strip("-_")
        if not safe:
            return "a"
        if safe[0].isdigit():
            return f"a-{safe}"
        return safe

    def _find_window(self, terminal_id: str) -> str | None:
        if not terminal_id:
            return None
        try:
            for w in tmux_manager.list_inner_windows():
                if w.terminal_id == terminal_id:
                    return w.window_id
        except Exception:
            pass
        return None

    def _update_active_terminal_ui(self, terminal_id: str | None) -> None:
        """Update just the active-terminal button classes (no full re-render)."""
        try:
            for btn in self.query("Button.terminal-switch"):
                btn_id = btn.id or ""
                if not btn_id.startswith("switch-"):
                    continue
                is_active = bool(terminal_id and (btn.name == terminal_id))
                if is_active:
                    btn.add_class("active-terminal")
                else:
                    btn.remove_class("active-terminal")
        except Exception:
            return

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""

        if btn_id == "create-agent":
            await self._create_agent()
            return

        if btn_id.startswith("agent-"):
            # Agent name click is handled by AgentNameButton.SingleClicked to avoid
            # toggling when the user double-clicks to open details.
            return

        if btn_id.startswith("delete-"):
            agent_id = event.button.name or ""
            if self._agent_is_invisible(agent_id):
                await self._restore_agent(agent_id)
            else:
                await self._delete_agent(agent_id)
            return

        if btn_id.startswith("share-"):
            agent_id = event.button.name or ""
            await self._share_agent(agent_id)
            return

        if btn_id.startswith("create-term-"):
            agent_id = event.button.name or ""
            await self._create_terminal_for_agent(agent_id)
            return

        if btn_id.startswith("resume-term-"):
            agent_id = event.button.name or ""
            from ..screens import AgentDetailScreen

            self.app.push_screen(AgentDetailScreen(agent_id))
            return

        if btn_id.startswith("switch-"):
            terminal_id = event.button.name or ""
            self._request_switch_to_terminal(terminal_id)
            return

        if btn_id.startswith("close-"):
            terminal_id = event.button.name or ""
            await self._close_terminal(terminal_id)
            return

    async def on_select_changed(self, event: Select.Changed) -> None:
        select = event.select
        select_id = select.id or ""
        if not select_id.startswith("resume-term-"):
            return
        if event.value is Select.BLANK:
            return
        agent_id = select.name or ""
        if event.value == "resume":
            from ..screens import AgentDetailScreen

            self.app.push_screen(AgentDetailScreen(agent_id))
        select.clear()

    async def on_agent_name_button_double_clicked(
        self, event: AgentNameButton.DoubleClicked
    ) -> None:
        agent_id = event.agent_id
        if not agent_id:
            return

        from ..screens import AgentDetailScreen

        self.app.push_screen(AgentDetailScreen(agent_id))

    async def on_agent_name_button_single_clicked(
        self, event: AgentNameButton.SingleClicked
    ) -> None:
        agent_id = event.agent_id
        if not agent_id:
            return
        await self._toggle_agent_terminals(agent_id)

    async def _toggle_agent_terminals(self, agent_id: str) -> None:
        agent = next((a for a in self._agents if a.get("id") == agent_id), None)
        if not agent:
            return
        if not (agent.get("terminals") or []):
            return

        if agent_id in self._collapsed_agent_ids:
            self._collapsed_agent_ids.remove(agent_id)
            expanded = True
        else:
            self._collapsed_agent_ids.add(agent_id)
            expanded = False

        if self._render_worker is not None and self._render_worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        ):
            # Avoid fighting a full re-render; it will pick up the new collapse state.
            self._pending_render = True
            self._pending_render_reason = "toggle-expand"
            return

        safe_id = self._safe_id(agent_id)
        try:
            block = self.query_one(f"#agent-block-{safe_id}", _AgentBlockWidget)
            await block.set_expanded(expanded)
        except Exception:
            # If the widget tree changed mid-toggle (refresh/render), fall back to a full render.
            self.force_render(reason="toggle-expand")

    async def _create_agent(self) -> None:
        try:
            from ..screens import CreateAgentInfoScreen

            self.app.push_screen(CreateAgentInfoScreen(), self._on_create_result)
        except ImportError:
            try:
                from ...agent_names import generate_agent_name
                from ...db import get_database
                import uuid

                db = get_database(read_only=False)
                agent_id = str(uuid.uuid4())
                name = generate_agent_name()
                db.get_or_create_agent_info(agent_id, name=name)
                self.app.notify(f"Created: {name}", title="Agent")
                self.refresh_data()
            except Exception as e:
                self.app.notify(f"Failed: {e}", title="Agent", severity="error")

    def _on_create_result(self, result: dict | None) -> None:
        if result:
            if result.get("imported"):
                n = result.get("sessions_imported", 0)
                if result.get("limit_reached"):
                    limit_value = int(result.get("limit_value") or 1000)
                    blocked_count = int(result.get("blocked_by_limit") or 0)
                    self.app.notify(
                        f"Imported: {result.get('name')} ({n} sessions). "
                        f"当前“Context”关联的session已经达到上限（{limit_value}），"
                        f"拦截 {blocked_count} 个额外关联。",
                        title="Agent",
                        severity="warning",
                    )
                else:
                    self.app.notify(f"Imported: {result.get('name')} ({n} sessions)", title="Agent")
            else:
                self.app.notify(f"Created: {result.get('name')}", title="Agent")
        self.refresh_data()

    @staticmethod
    def _agent_session_limit_message(limit: int) -> str:
        return f"当前“Context”关联的session已经达到上限（{limit}）。"

    def _is_agent_session_limit_reached(self, agent_id: str) -> bool:
        try:
            from ...db import MAX_SESSION_NUM, get_database

            db = get_database(read_only=True)
            try:
                count = db.get_agent_session_count(agent_id)
            finally:
                try:
                    db.close()
                except Exception:
                    pass
            if count >= MAX_SESSION_NUM:
                self.app.notify(
                    self._agent_session_limit_message(MAX_SESSION_NUM),
                    title="Context",
                    severity="warning",
                )
                return True
        except Exception:
            return False
        return False

    async def _create_terminal_for_agent(self, agent_id: str) -> None:
        """Create a new terminal under the specified agent."""
        if not agent_id:
            return

        # Get agent info
        agent = next((a for a in self._agents if a["id"] == agent_id), None)
        if not agent:
            self.app.notify("Agent not found", title="Agent", severity="error")
            return

        if self._is_agent_session_limit_reached(agent_id):
            return

        # Show create terminal screen with agent context
        try:
            from ..screens import CreateAgentScreen

            self.app.push_screen(
                CreateAgentScreen(),
                lambda result: self._on_create_terminal_result(result, agent_id),
            )
        except ImportError as e:
            self.app.notify(f"Failed: {e}", title="Agent", severity="error")

    def _on_create_terminal_result(
        self, result: tuple[str, str, bool, bool] | None, agent_id: str
    ) -> None:
        """Handle result from CreateAgentScreen."""
        if result is None:
            return

        agent_type, workspace, skip_permissions, no_track = result

        # Create the terminal with agent association
        self.run_worker(
            self._do_create_terminal(agent_type, workspace, skip_permissions, no_track, agent_id),
            group="terminal-create",
            exclusive=True,
            exit_on_error=False,
        )

    async def _do_create_terminal(
        self,
        agent_type: str,
        workspace: str,
        skip_permissions: bool,
        no_track: bool,
        agent_id: str,
    ) -> None:
        """Actually create the terminal with agent association."""
        if self._is_agent_session_limit_reached(agent_id):
            return

        if agent_type == "claude":
            await self._create_claude_terminal(workspace, skip_permissions, no_track, agent_id)
        elif agent_type == "codex":
            await self._create_codex_terminal(workspace, skip_permissions, no_track, agent_id)
        elif agent_type == "opencode":
            await self._create_opencode_terminal(workspace, agent_id)
        elif agent_type == "shell":
            await self._create_shell_terminal(workspace, agent_id)

        self.refresh_data()

    async def _create_claude_terminal(
        self, workspace: str, skip_permissions: bool, no_track: bool, agent_id: str
    ) -> None:
        """Create a Claude terminal associated with an agent."""
        terminal_id = tmux_manager.new_terminal_id()
        logger.info(
            "Create terminal requested: provider=claude terminal_id=%s agent_id=%s workspace=%s "
            "no_track=%s skip_permissions=%s",
            terminal_id,
            agent_id,
            workspace,
            no_track,
            skip_permissions,
        )

        # Prepare CODEX_HOME so user can run codex in this terminal
        try:
            from ...codex_home import prepare_codex_home

            codex_home = prepare_codex_home(terminal_id)
        except Exception:
            codex_home = None

        env = {
            tmux_manager.ENV_TERMINAL_ID: terminal_id,
            tmux_manager.ENV_TERMINAL_PROVIDER: "claude",
            tmux_manager.ENV_INNER_SOCKET: tmux_manager.INNER_SOCKET,
            tmux_manager.ENV_INNER_SESSION: tmux_manager.INNER_SESSION,
            "ALINE_AGENT_ID": agent_id,  # Pass agent_id to hooks
        }
        if codex_home:
            env["CODEX_HOME"] = str(codex_home)
        if no_track:
            env["ALINE_NO_TRACK"] = "1"

        # Install hooks
        self._install_claude_hooks(workspace)

        claude_cmd = "claude"
        if skip_permissions:
            claude_cmd = "claude --dangerously-skip-permissions"
        agent = next((a for a in self._agents if a.get("id") == agent_id), None)
        agent_title = ""
        agent_description = ""
        if agent:
            agent_title = str(agent.get("title") or "")
            agent_description = str(agent.get("description") or "")
        system_prompt = build_claude_system_prompt(
            agent_title=agent_title,
            agent_description=agent_description,
        )
        claude_cmd += f" --append-system-prompt {shlex.quote(system_prompt)}"

        command = self._command_in_directory(
            tmux_manager.shell_run_and_keep_open(claude_cmd), workspace
        )

        created = tmux_manager.create_inner_window(
            "cc",
            tmux_manager.shell_command_with_env(command, env),
            terminal_id=terminal_id,
            provider="claude",
            no_track=no_track,
        )

        if created:
            logger.info(
                "Create terminal success: provider=claude terminal_id=%s window_id=%s",
                terminal_id,
                created.window_id,
            )
            if not tmux_manager.focus_right_pane():
                logger.warning("Create terminal: focus_right_pane failed (provider=claude)")
            # Store agent association in database with agent_info_id in source
            try:
                from ...db import get_database

                db = get_database(read_only=False)
                db.get_or_create_agent(
                    terminal_id,
                    provider="claude",
                    session_type="claude",
                    cwd=workspace,
                    project_dir=workspace,
                    source=f"agent:{agent_id}",  # Store agent_info_id in source
                )
            except Exception:
                pass
        else:
            logger.warning("Create terminal failed: provider=claude terminal_id=%s", terminal_id)
            self.app.notify("Failed to create terminal", title="Agent", severity="error")

    async def _create_codex_terminal(
        self, workspace: str, skip_permissions: bool, no_track: bool, agent_id: str
    ) -> None:
        """Create a Codex terminal associated with an agent."""
        try:
            from ...db import get_database
            from datetime import datetime, timedelta

            db = get_database(read_only=True)
            try:
                cutoff = datetime.now() - timedelta(seconds=10)
                for agent in db.list_agents(status="active", limit=1000):
                    if agent.provider != "codex":
                        continue
                    if (agent.source or "") != f"agent:{agent_id}":
                        continue
                    if agent.created_at >= cutoff and not agent.session_id:
                        self.app.notify(
                            "Please wait a few seconds before opening another Codex terminal for this agent.",
                            title="Agent",
                            severity="warning",
                        )
                        return
            finally:
                try:
                    db.close()
                except Exception:
                    pass
        except Exception:
            pass

        terminal_id = tmux_manager.new_terminal_id()
        logger.info(
            "Create terminal requested: provider=codex terminal_id=%s agent_id=%s workspace=%s "
            "no_track=%s skip_permissions=%s",
            terminal_id,
            agent_id,
            workspace,
            no_track,
            skip_permissions,
        )

        try:
            from ...codex_home import prepare_codex_home

            codex_home = prepare_codex_home(terminal_id, agent_id=agent_id)
        except Exception:
            codex_home = None

        # Write OneContext instructions
        if codex_home:
            try:
                instructions_file = Path(codex_home) / "instructions.md"
                agent = next((a for a in self._agents if a.get("id") == agent_id), None)
                agent_title = ""
                agent_description = ""
                if agent:
                    agent_title = str(agent.get("title") or "")
                    agent_description = str(agent.get("description") or "")
                instructions = build_codex_instructions(
                    agent_title=agent_title,
                    agent_description=agent_description,
                )
                instructions_file.write_text(f"{instructions}\n")
            except Exception:
                pass

        env = {
            tmux_manager.ENV_TERMINAL_ID: terminal_id,
            tmux_manager.ENV_TERMINAL_PROVIDER: "codex",
            tmux_manager.ENV_INNER_SOCKET: tmux_manager.INNER_SOCKET,
            tmux_manager.ENV_INNER_SESSION: tmux_manager.INNER_SESSION,
            "ALINE_AGENT_ID": agent_id,
        }
        if codex_home:
            env["CODEX_HOME"] = str(codex_home)
        if no_track:
            env["ALINE_NO_TRACK"] = "1"

        # Store agent in database with agent_info_id in source
        try:
            from ...db import get_database

            db = get_database(read_only=False)
            db.get_or_create_agent(
                terminal_id,
                provider="codex",
                session_type="codex",
                cwd=workspace,
                project_dir=workspace,
                source=f"agent:{agent_id}",  # Store agent_info_id in source
            )
        except Exception:
            pass

        codex_cmd = "codex"
        if skip_permissions:
            codex_cmd = "codex --full-auto"

        command = self._command_in_directory(
            tmux_manager.shell_run_and_keep_open(codex_cmd), workspace
        )

        created = tmux_manager.create_inner_window(
            "codex",
            tmux_manager.shell_command_with_env(command, env),
            terminal_id=terminal_id,
            provider="codex",
            no_track=no_track,
        )

        if created:
            logger.info(
                "Create terminal success: provider=codex terminal_id=%s window_id=%s",
                terminal_id,
                created.window_id,
            )
            if not tmux_manager.focus_right_pane():
                logger.warning("Create terminal: focus_right_pane failed (provider=codex)")
        else:
            logger.warning("Create terminal failed: provider=codex terminal_id=%s", terminal_id)
            self.app.notify("Failed to create terminal", title="Agent", severity="error")

    async def _create_opencode_terminal(self, workspace: str, agent_id: str) -> None:
        """Create an Opencode terminal associated with an agent."""
        terminal_id = tmux_manager.new_terminal_id()
        logger.info(
            "Create terminal requested: provider=opencode terminal_id=%s agent_id=%s workspace=%s",
            terminal_id,
            agent_id,
            workspace,
        )

        # Prepare CODEX_HOME so user can run codex in this terminal
        try:
            from ...codex_home import prepare_codex_home

            codex_home = prepare_codex_home(terminal_id, agent_id=agent_id)
        except Exception:
            codex_home = None

        env = {
            tmux_manager.ENV_TERMINAL_ID: terminal_id,
            tmux_manager.ENV_TERMINAL_PROVIDER: "opencode",
            tmux_manager.ENV_INNER_SOCKET: tmux_manager.INNER_SOCKET,
            tmux_manager.ENV_INNER_SESSION: tmux_manager.INNER_SESSION,
            "ALINE_AGENT_ID": agent_id,
        }
        if codex_home:
            env["CODEX_HOME"] = str(codex_home)

        # Install Claude hooks in case user runs claude manually
        self._install_claude_hooks(workspace)

        command = self._command_in_directory(
            tmux_manager.shell_run_and_keep_open("opencode"), workspace
        )

        created = tmux_manager.create_inner_window(
            "opencode",
            tmux_manager.shell_command_with_env(command, env),
            terminal_id=terminal_id,
            provider="opencode",
        )

        if created:
            logger.info(
                "Create terminal success: provider=opencode terminal_id=%s window_id=%s",
                terminal_id,
                created.window_id,
            )
            if not tmux_manager.focus_right_pane():
                logger.warning("Create terminal: focus_right_pane failed (provider=opencode)")
            # Store agent association in database
            try:
                from ...db import get_database

                db = get_database(read_only=False)
                db.get_or_create_agent(
                    terminal_id,
                    provider="opencode",
                    session_type="opencode",
                    cwd=workspace,
                    project_dir=workspace,
                    source=f"agent:{agent_id}",
                )
            except Exception:
                pass
        else:
            logger.warning("Create terminal failed: provider=opencode terminal_id=%s", terminal_id)
            self.app.notify("Failed to create terminal", title="Agent", severity="error")

    async def _create_shell_terminal(self, workspace: str, agent_id: str) -> None:
        """Create a shell terminal associated with an agent."""
        import os

        terminal_id = tmux_manager.new_terminal_id()
        shell_name = tmux_manager._user_shell_name()
        logger.info(
            "Create terminal requested: provider=shell terminal_id=%s agent_id=%s workspace=%s shell=%s",
            terminal_id,
            agent_id,
            workspace,
            shell_name,
        )

        # Prepare CODEX_HOME so user can run codex in this terminal
        try:
            from ...codex_home import prepare_codex_home

            codex_home = prepare_codex_home(terminal_id, agent_id=agent_id)
        except Exception:
            codex_home = None

        env = {
            tmux_manager.ENV_TERMINAL_ID: terminal_id,
            tmux_manager.ENV_TERMINAL_PROVIDER: "shell",
            tmux_manager.ENV_INNER_SOCKET: tmux_manager.INNER_SOCKET,
            tmux_manager.ENV_INNER_SESSION: tmux_manager.INNER_SESSION,
            "ALINE_AGENT_ID": agent_id,
        }
        if codex_home:
            env["CODEX_HOME"] = str(codex_home)

        # Install Claude hooks in case user runs claude manually
        self._install_claude_hooks(workspace)

        user_shell = os.environ.get("SHELL", "/bin/sh")
        command = self._command_in_directory(user_shell, workspace)

        created = tmux_manager.create_inner_window(
            shell_name,
            tmux_manager.shell_command_with_env(command, env),
            terminal_id=terminal_id,
            provider="shell",
        )

        if created:
            logger.info(
                "Create terminal success: provider=shell terminal_id=%s window_id=%s",
                terminal_id,
                created.window_id,
            )
            if not tmux_manager.focus_right_pane():
                logger.warning("Create terminal: focus_right_pane failed (provider=shell)")
            # Store agent association in database
            try:
                from ...db import get_database

                db = get_database(read_only=False)
                db.get_or_create_agent(
                    terminal_id,
                    provider="shell",
                    session_type="shell",
                    cwd=workspace,
                    project_dir=workspace,
                    source=f"agent:{agent_id}",
                )
            except Exception:
                pass
        else:
            logger.warning("Create terminal failed: provider=shell terminal_id=%s", terminal_id)
            self.app.notify("Failed to create terminal", title="Agent", severity="error")

    def _install_claude_hooks(self, workspace: str) -> None:
        """Install Claude hooks for a workspace."""
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

    @staticmethod
    def _command_in_directory(command: str, directory: str) -> str:
        return f"cd {shlex.quote(directory)} && {command}"

    async def _delete_agent(self, agent_id: str) -> None:
        if not agent_id:
            return

        # Gather info for the confirmation dialog.
        try:
            from ...db import get_database

            db = get_database(read_only=False)
            info = db.get_agent_info(agent_id)
            name = info.name if info else "Unknown"
        except Exception as e:
            self.app.notify(f"Failed: {e}", title="Agent", severity="error")
            return

        # Find open terminals for this agent.
        terminals = []
        for agent in self._agents:
            if agent.get("id") == agent_id:
                terminals = agent.get("terminals") or []
                break

        from ..state import get_dashboard_state_value, set_dashboard_state_value

        # Skip confirmation if user previously checked "Don't show this again".
        if bool(get_dashboard_state_value("skip_archive_confirm", False)):
            self._do_archive_agent(agent_id, name, terminals)
            return

        try:
            from ..screens import ArchiveConfirmScreen

            def _on_confirm(result: dict | None) -> None:
                if not result or not result.get("confirmed"):
                    return
                if result.get("dont_show_again"):
                    set_dashboard_state_value("skip_archive_confirm", True)
                self.app.call_later(lambda: self._do_archive_agent(agent_id, name, terminals))

            self.app.push_screen(
                ArchiveConfirmScreen(
                    agent_name=name,
                    terminal_count=len(terminals),
                ),
                _on_confirm,
            )
        except Exception:
            # Fallback: archive without confirmation if dialog fails.
            self._do_archive_agent(agent_id, name, terminals)

    def _agent_is_invisible(self, agent_id: str) -> bool:
        agent_id = (agent_id or "").strip()
        if not agent_id:
            return False
        try:
            agent = next((a for a in self._agents if a.get("id") == agent_id), None)
            visibility = (agent or {}).get("visibility") or "visible"
            return str(visibility).strip() != "visible"
        except Exception:
            return False

    async def _restore_agent(self, agent_id: str) -> None:
        agent_id = (agent_id or "").strip()
        if not agent_id:
            return
        try:
            from ...db import get_database

            db = get_database(read_only=False)
            info = db.get_agent_info(agent_id)
            name = info.name if info else "Unknown"
            record = db.update_agent_info(agent_id, visibility="visible")
            if record:
                self.app.notify(f"Restored: {name}", title="Agent")
            self.refresh_data()
        except Exception as e:
            self.app.notify(f"Failed: {e}", title="Agent", severity="error")

    def _do_archive_agent(self, agent_id: str, name: str, terminals: list[dict]) -> None:
        """Close tmux windows and archive the agent."""
        # Close all tmux windows belonging to this agent.
        for term in terminals:
            window_id = (term.get("window_id") or "").strip()
            if window_id:
                tmux_manager.kill_inner_window(window_id)
            # Mark terminal as inactive in the database.
            terminal_id = term.get("terminal_id", "")
            if terminal_id:
                try:
                    from ...db import get_database

                    db = get_database(read_only=False)
                    db.update_agent(terminal_id, status="inactive")
                except Exception:
                    pass

        # Archive the agent.
        try:
            from ...db import get_database

            db = get_database(read_only=False)
            record = db.update_agent_info(agent_id, visibility="invisible")
            if record:
                self.app.notify(f"Archived: {name}", title="Agent")
            self.refresh_data()
        except Exception as e:
            self.app.notify(f"Failed: {e}", title="Agent", severity="error")

    def _request_switch_to_terminal(self, terminal_id: str) -> None:
        """Switch terminals without blocking the UI thread."""
        terminal_id = (terminal_id or "").strip()
        if not terminal_id:
            return

        self._active_terminal_id = terminal_id
        self._update_active_terminal_ui(terminal_id)
        self._rendered_fingerprint = (
            self._fingerprint(self._agents) + f"|active:{self._active_terminal_id}"
        )

        self._switch_seq += 1
        seq = self._switch_seq

        if self._switch_worker is not None and self._switch_worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        ):
            try:
                self._switch_worker.cancel()
            except Exception:
                pass

        agents_snapshot = self._agents

        def work() -> dict:
            # Prefer cached window_id to avoid a blocking tmux scan on every click.
            window_id = None
            try:
                for agent in agents_snapshot:
                    for term in agent.get("terminals") or []:
                        if term.get("terminal_id") == terminal_id:
                            window_id = (term.get("window_id") or "").strip() or None
                            break
                    if window_id:
                        break
            except Exception:
                window_id = None

            if not window_id:
                # Fall back to querying tmux (best-effort).
                window_id = self._find_window(terminal_id)

            if not window_id:
                # If the terminal was closed from within the shell (e.g. `exit`), the DB record can
                # remain "active" briefly. Trigger a refresh so the stale button disappears.
                try:
                    from ...db import get_database

                    db = get_database(read_only=False)
                    db.update_agent(terminal_id, status="inactive")
                except Exception:
                    pass
                return {
                    "seq": seq,
                    "terminal_id": terminal_id,
                    "ok": False,
                    "error": "Window not found",
                }

            if not tmux_manager.select_inner_window(window_id):
                return {
                    "seq": seq,
                    "terminal_id": terminal_id,
                    "window_id": window_id,
                    "ok": False,
                    "error": "Failed to switch",
                }

            # Prefer showing the terminal after switching (best-effort).
            if not tmux_manager.focus_right_pane():
                logger.warning(
                    "Switch terminal: focus_right_pane failed (terminal_id=%s window_id=%s)",
                    terminal_id,
                    window_id,
                )
            tmux_manager.clear_attention(window_id)
            return {
                "seq": seq,
                "terminal_id": terminal_id,
                "window_id": window_id,
                "ok": True,
            }

        self._switch_worker = self.run_worker(work, thread=True, exit_on_error=False)

    async def _close_terminal(self, terminal_id: str) -> None:
        if not terminal_id:
            return

        from ..state import get_dashboard_state_value, set_dashboard_state_value

        # Skip confirmation if user previously checked "Don't show this again".
        if bool(get_dashboard_state_value("skip_close_terminal_confirm", False)):
            self._do_close_terminal(terminal_id)
            return

        try:
            from ..screens import CloseTerminalConfirmScreen

            def _on_confirm(result: dict | None) -> None:
                if not result or not result.get("confirmed"):
                    return
                if result.get("dont_show_again"):
                    set_dashboard_state_value("skip_close_terminal_confirm", True)
                self.app.call_later(lambda: self._do_close_terminal(terminal_id))

            self.app.push_screen(CloseTerminalConfirmScreen(), _on_confirm)
        except Exception:
            # Fallback: close without confirmation if dialog fails.
            self._do_close_terminal(terminal_id)

    def _do_close_terminal(self, terminal_id: str) -> None:
        """Actually close the tmux window and mark terminal inactive."""
        # Try to close the tmux window if it exists
        window_id = self._find_window(terminal_id)
        if window_id:
            tmux_manager.kill_inner_window(window_id)

            # If killing that window destroyed the inner session (it was the last
            # window), the outer right pane dies too.  Re-establish it so the
            # dashboard doesn't lose its tmux layout.
            try:
                tmux_manager.ensure_right_pane_ready()
            except Exception:
                pass

        # Also update the agent status in the database to mark it as inactive
        try:
            from ...db import get_database

            db = get_database(read_only=False)
            db.update_agent(terminal_id, status="inactive")
        except Exception as e:
            logger.debug(f"Failed to update agent status: {e}")

        self.refresh_data()

    def _get_sharing_agent_name(self) -> str:
        """Return the name of the agent currently being shared."""
        if not self._share_agent_id:
            return "agent"
        try:
            from ...db import get_database

            db = get_database(read_only=True)
            try:
                info = db.get_agent_info(self._share_agent_id)
                return getattr(info, "name", None) or "agent"
            finally:
                db.close()
        except Exception:
            return "agent"

    async def _share_agent(self, agent_id: str) -> None:
        """Share an agent; if already shared, this acts like sync."""
        if not agent_id:
            return

        from ..state import get_dashboard_state_value, set_dashboard_state_value

        # Check if share is already in progress
        if self._share_worker is not None and self._share_worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        ):
            busy_name = self._get_sharing_agent_name()
            self.app.notify(
                f"Another share is in progress ({busy_name}). Please try again later",
                title="Share",
                severity="warning",
            )
            return

        # Check if agent has sessions and whether it already has a share link.
        already_shared = False
        agent_name: str | None = None
        db = None
        try:
            from ...db import get_database

            db = get_database(read_only=True)
            agent_info = db.get_agent_info(agent_id)
            already_shared = bool(agent_info and agent_info.share_url)
            agent_name = getattr(agent_info, "name", None) if agent_info else None
            sessions = db.get_sessions_by_agent_id(agent_id)
            if not sessions:
                self.app.notify("Agent has no sessions to share", title="Share", severity="warning")
                return
        except Exception as e:
            self.app.notify(f"Failed to check sessions: {e}", title="Share", severity="error")
            return
        finally:
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass

        mode_label = "Sync" if already_shared else "Share"

        # Pre-share consent / warning (skippable)
        if not bool(get_dashboard_state_value("skip_share_warning", False)):
            try:
                from ..screens import ShareWarningScreen

                def _on_warning_result(result: dict | None) -> None:
                    if not result or not result.get("confirmed"):
                        return
                    if result.get("dont_show_again"):
                        set_dashboard_state_value("skip_share_warning", True)
                    self.app.call_later(
                        lambda: self._run_share_agent_worker(
                            agent_id=agent_id, already_shared=already_shared
                        )
                    )

                self.app.push_screen(
                    ShareWarningScreen(
                        mode_label=mode_label,
                        agent_name=agent_name,
                        expiry_days=7,
                        max_views=100,
                        requires_login=True,
                        processed_only=True,
                        recipients_can_contribute=False,
                    ),
                    _on_warning_result,
                )
                return
            except Exception:
                # If the dialog fails for any reason, fall back to the existing share flow.
                pass

        self._run_share_agent_worker(agent_id=agent_id, already_shared=already_shared)

    # -- Share button spinner helpers ------------------------------------------

    _SHARE_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def _start_share_spinner(self) -> None:
        """Start animating the Share button for the current _share_agent_id."""
        self._share_spinner_frame = 0
        self._update_share_btn_label()
        if self._share_spinner_timer is None:
            self._share_spinner_timer = self.set_interval(0.12, self._tick_share_spinner)
        else:
            self._share_spinner_timer.resume()

    def _stop_share_spinner(self) -> None:
        """Stop the spinner and restore the Share button label."""
        if self._share_spinner_timer is not None:
            self._share_spinner_timer.pause()
        if not self._share_agent_id:
            return
        safe_id = self._safe_id(self._share_agent_id)
        try:
            btn = self.query_one(f"#share-{safe_id}", HoverButton)
            btn.label = "Share"
        except Exception:
            pass

    def _tick_share_spinner(self) -> None:
        self._share_spinner_frame = (self._share_spinner_frame + 1) % len(
            self._SHARE_SPINNER_FRAMES
        )
        self._update_share_btn_label()

    def _update_share_btn_label(self) -> None:
        if not self._share_agent_id:
            return
        safe_id = self._safe_id(self._share_agent_id)
        try:
            btn = self.query_one(f"#share-{safe_id}", HoverButton)
            frame = self._SHARE_SPINNER_FRAMES[self._share_spinner_frame]
            btn.label = f"{frame} Share"
        except Exception:
            pass

    # --------------------------------------------------------------------------

    def _run_share_agent_worker(self, agent_id: str, *, already_shared: bool) -> None:
        """Start the agent share/sync worker (assumes pre-checks are already done)."""
        if not agent_id:
            return

        # Check if share is already in progress
        if self._share_worker is not None and self._share_worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        ):
            busy_name = self._get_sharing_agent_name()
            self.app.notify(
                f"Another share is in progress ({busy_name}). Please try again later",
                title="Share",
                severity="warning",
            )
            return

        mode_label = "Sync" if already_shared else "Share"

        # Store agent_id for the worker callback
        self._share_agent_id = agent_id
        self._start_share_spinner()

        # Create progress callback that posts notifications from worker thread
        app = self.app  # Capture reference for closure

        def compact_status(message: str) -> str:
            msg = (message or "").strip()
            if not msg:
                return f"{mode_label}: Working"
            msg = re.sub(r"[.]{2,}$", "", msg).strip()
            msg = msg.removesuffix("...").removesuffix(".").strip()
            lowered = msg.lower()
            if "chunked upload" in lowered:
                msg = "Upload: init"
            elif "uploading chunk" in lowered:
                match = re.search(r"chunk\s+(\d+)\s*/\s*(\d+)", msg, re.IGNORECASE)
                if match:
                    msg = f"Upload: chunk {match.group(1)}/{match.group(2)}"
                else:
                    msg = "Upload: chunk"
            elif "finalizing upload" in lowered:
                msg = "Upload: finalize"
            elif "upload complete" in lowered:
                msg = "Upload: complete"
            return f"{mode_label}: {msg}"

        def progress_callback(message: str) -> None:
            """Send progress status from worker thread."""
            try:
                app.call_from_thread(app.set_status_bar, compact_status(message))
            except Exception:
                pass  # Ignore errors if app is closing

        def work() -> dict:
            import contextlib
            import io
            import json as json_module
            import re

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                from ...commands import export_shares

                exit_code = export_shares.export_agent_shares_command(
                    agent_id=agent_id,
                    password=None,
                    json_output=True,
                    compact=True,
                    progress_callback=progress_callback,
                )

            output = stdout.getvalue().strip()
            error_text = stderr.getvalue().strip()
            result: dict = {
                "exit_code": exit_code,
                "output": output,
                "stderr": error_text,
            }

            if output:
                try:
                    result["json"] = json_module.loads(output)
                except Exception:
                    result["json"] = None
                    try:
                        from ...llm_client import extract_json

                        result["json"] = extract_json(output)
                    except Exception:
                        result["json"] = None
                        try:
                            match = re.search(r"\{.*\}", output, re.DOTALL)
                            if match:
                                result["json"] = json_module.loads(match.group(0), strict=False)
                        except Exception:
                            result["json"] = None

            if not result.get("json") and output:
                match = re.search(r"https?://[^\s\"']+/share/[^\s\"']+", output)
                if match:
                    result["share_link_guess"] = match.group(0)

            return result

        try:
            self.app.set_status_bar(f"{mode_label}: Starting")
        except Exception:
            pass
        self._share_worker = self.run_worker(work, thread=True, exit_on_error=False)

    def _handle_share_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle share worker state changes."""
        from ..clipboard import copy_text

        if event.state == WorkerState.ERROR:
            self._stop_share_spinner()
            err = self._share_worker.error if self._share_worker else "Unknown error"
            try:
                self.app.set_status_bar(
                    "Share: Error", spinning=False, variant="error", clear_after_s=5
                )
            except Exception:
                pass
            self.app.notify(f"Share failed: {err}", title="Share", severity="error")
            return

        if event.state != WorkerState.SUCCESS:
            return

        self._stop_share_spinner()

        result = self._share_worker.result if self._share_worker else {}
        raw_exit_code = result.get("exit_code", None)
        exit_code = 1 if raw_exit_code is None else int(raw_exit_code)
        payload = result.get("json") or {}
        share_link = payload.get("share_link") or payload.get("share_url")
        if not share_link:
            share_link = result.get("share_link_guess")
        agent_name = payload.get("agent_name") if isinstance(payload, dict) else None
        slack_message = payload.get("slack_message") if isinstance(payload, dict) else None
        synced = bool(payload.get("synced")) if isinstance(payload, dict) else False
        if not (isinstance(slack_message, str) and slack_message.strip()):
            verb = "Synced" if synced else "Shared"
            if agent_name:
                slack_message = (
                    f"{verb} {BRANDING.product_name} agent “{agent_name}”.\n"
                    "Take a look when you have a moment."
                )
            else:
                slack_message = (
                    f"{verb} a {BRANDING.product_name} agent.\nTake a look when you have a moment."
                )

        if exit_code == 0 and share_link:
            try:
                self.app.set_status_bar(
                    "Sync: Done" if synced else "Share: Done",
                    spinning=False,
                    variant="done",
                    clear_after_s=3,
                )
            except Exception:
                pass
            self.refresh_data()
            try:
                from ..screens.share_result import ShareResultScreen

                self.app.push_screen(
                    ShareResultScreen(
                        share_link=str(share_link),
                        slack_message=str(slack_message) if slack_message else None,
                        agent_name=str(agent_name) if agent_name else None,
                    )
                )
            except Exception:
                copied = copy_text(self.app, str(share_link))
                if copied:
                    verb = "Synced" if synced else "Shared"
                    self.app.notify(f"{verb} (link copied)", title="Share", timeout=4)
                else:
                    self.app.notify(
                        f"Done (copy failed). Link: {share_link}",
                        title="Share",
                        timeout=6,
                        severity="warning",
                    )
        elif exit_code == 0:
            try:
                self.app.set_status_bar(
                    "Share: Done", spinning=False, variant="done", clear_after_s=3
                )
            except Exception:
                pass
            self.app.notify("Share completed", title="Share", timeout=3)
        else:
            try:
                self.app.set_status_bar(
                    "Share: Failed", spinning=False, variant="error", clear_after_s=6
                )
            except Exception:
                pass
            extra = result.get("stderr") or ""
            suffix = f": {extra}" if extra else ""
            self.app.notify(f"Share failed (exit {exit_code}){suffix}", title="Share", timeout=6)

    async def _sync_agent(self, agent_id: str) -> None:
        """Sync all sessions for an agent with remote share."""
        if not agent_id:
            return

        # Check if sync is already in progress
        if self._sync_worker is not None and self._sync_worker.state in (
            WorkerState.PENDING,
            WorkerState.RUNNING,
        ):
            return

        self._sync_agent_id = agent_id

        app = self.app

        def progress_callback(message: str) -> None:
            try:
                msg = (message or "").strip()
                if msg:
                    msg = msg.removesuffix("...").removesuffix(".").strip()
                app.call_from_thread(app.set_status_bar, f"Sync: {msg or 'Working'}")
            except Exception:
                pass

        def work() -> dict:
            from ...commands.sync_agent import sync_agent_command

            return sync_agent_command(
                agent_id=agent_id,
                progress_callback=progress_callback,
            )

        try:
            self.app.set_status_bar("Sync: Starting")
        except Exception:
            pass
        self._sync_worker = self.run_worker(work, thread=True, exit_on_error=False)

    def _handle_sync_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle sync worker state changes."""
        if event.state == WorkerState.ERROR:
            err = self._sync_worker.error if self._sync_worker else "Unknown error"
            try:
                self.app.set_status_bar(
                    "Sync: Error", spinning=False, variant="error", clear_after_s=5
                )
            except Exception:
                pass
            self.app.notify(f"Sync failed: {err}", title="Sync", severity="error")
            return

        if event.state != WorkerState.SUCCESS:
            return

        result = self._sync_worker.result if self._sync_worker else {}

        if result.get("success"):
            try:
                self.app.set_status_bar(
                    "Sync: Done", spinning=False, variant="done", clear_after_s=3
                )
            except Exception:
                pass
            pulled = result.get("sessions_pulled", 0)
            pushed = result.get("sessions_pushed", 0)

            # Copy share URL to clipboard
            agent_id = self._sync_agent_id
            share_url = None
            if agent_id:
                agent = next((a for a in self._agents if a["id"] == agent_id), None)
                if agent:
                    share_url = agent.get("share_url")

            if share_url:
                copied = copy_text(self.app, share_url)
                suffix = " (link copied)" if copied else ""
            else:
                suffix = ""

            self.app.notify(
                f"Synced: pulled {pulled}, pushed {pushed} session(s){suffix}",
                title="Sync",
                timeout=6,
            )
            self.refresh_data()
        else:
            try:
                self.app.set_status_bar(
                    "Sync: Failed", spinning=False, variant="error", clear_after_s=6
                )
            except Exception:
                pass
            error = result.get("error", "Unknown error")
            self.app.notify(f"Sync failed: {error}", title="Sync", severity="error")

    async def _copy_share_link(self, agent_id: str) -> None:
        """Copy the share link for an agent to clipboard."""
        if not agent_id:
            return

        agent = next((a for a in self._agents if a["id"] == agent_id), None)
        if not agent:
            self.app.notify("Agent not found", title="Link", severity="error")
            return

        share_url = agent.get("share_url")
        if not share_url:
            self.app.notify("No share link available", title="Link", severity="warning")
            return

        copied = copy_text(self.app, share_url)
        if copied:
            self.app.notify("Share link copied to clipboard", title="Link", timeout=3)
        else:
            self.app.notify(f"Failed to copy. Link: {share_url}", title="Link", severity="warning")
