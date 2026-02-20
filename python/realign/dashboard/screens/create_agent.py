"""Create agent modal for the dashboard."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from ..tmux_manager import _user_shell_name

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Static

from ..state import load_dashboard_state, set_dashboard_state_value


def _load_last_workspace() -> str:
    """Load the last used workspace path from state file."""
    try:
        state = load_dashboard_state()
        path = state.get("last_workspace", "")
        if path and os.path.isdir(path):
            return path
    except Exception:
        pass
    # Default to current working directory or home
    try:
        return os.getcwd()
    except Exception:
        return str(Path.home())


def _save_last_workspace(path: str) -> None:
    """Save the last used workspace path to state file."""
    _save_state("last_workspace", path)


def _load_last_agent_type() -> str:
    """Load the last used agent type from state file."""
    try:
        state = load_dashboard_state()
        agent_type = state.get("last_agent_type", "claude")
        # Backward compat: saved "zsh" → "shell"
        if agent_type == "zsh":
            agent_type = "shell"
        if agent_type in {"claude", "codex", "shell"}:
            return agent_type
    except Exception:
        pass
    return "claude"


def _save_last_agent_type(agent_type: str) -> None:
    """Save the last used agent type to state file."""
    if agent_type in {"claude", "codex", "shell"}:
        _save_state("last_agent_type", agent_type)


def _load_permission_mode(provider: str = "claude") -> str:
    """Load the last used permission mode from state file."""
    key = f"{provider}_permission_mode"
    try:
        state = load_dashboard_state()
        return state.get(key, "normal")
    except Exception:
        pass
    return "normal"


def _save_permission_mode(mode: str, provider: str = "claude") -> None:
    """Save the permission mode to state file."""
    _save_state(f"{provider}_permission_mode", mode)


def _load_tracking_mode(provider: str = "claude") -> str:
    """Load the last used tracking mode from state file."""
    key = f"{provider}_tracking_mode"
    try:
        state = load_dashboard_state()
        return state.get(key, "track")
    except Exception:
        pass
    return "track"


def _save_tracking_mode(mode: str, provider: str = "claude") -> None:
    """Save the tracking mode to state file."""
    _save_state(f"{provider}_tracking_mode", mode)


def _save_state(key: str, value: str) -> None:
    """Save a key-value pair to the state file."""
    try:
        set_dashboard_state_value(key, value)
    except Exception:
        pass


class CreateAgentScreen(ModalScreen[Optional[tuple[str, str, bool, bool]]]):
    """Modal to create a new agent terminal.

    Returns a tuple of (agent_type, workspace_path, skip_permissions, no_track) on success, None on cancel.
    """

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
    ]

    DEFAULT_CSS = """
    CreateAgentScreen {
        align: center middle;
    }

    CreateAgentScreen #create-agent-root {
        width: 100%;
        max-width: 60;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $background;
        border: solid $accent;
    }

    CreateAgentScreen #create-agent-scroll {
        height: auto;
        max-height: 100%;
    }

    CreateAgentScreen #create-agent-title {
        height: auto;
        margin-bottom: 1;
        text-style: bold;
    }

    CreateAgentScreen .section-label {
        height: auto;
        margin-top: 1;
        margin-bottom: 0;
        color: $text-muted;
    }

    CreateAgentScreen RadioSet {
        height: auto;
        margin: 0;
        padding: 0;
        border: none;
        background: transparent;
    }

    CreateAgentScreen RadioButton {
        height: auto;
        padding: 0;
        margin: 0;
        background: transparent;
    }

    CreateAgentScreen #workspace-section {
        height: auto;
        margin-top: 1;
    }

    CreateAgentScreen #claude-options {
        height: auto;
        margin-top: 1;
    }

    CreateAgentScreen #claude-options.hidden {
        display: none;
    }

    CreateAgentScreen #codex-options {
        height: auto;
        margin-top: 1;
    }

    CreateAgentScreen #codex-options.hidden {
        display: none;
    }

    CreateAgentScreen #workspace-row {
        height: auto;
        margin-top: 0;
    }

    CreateAgentScreen #workspace-path {
        width: 1fr;
        height: auto;
        border: solid $primary-lighten-2;
    }

    CreateAgentScreen #browse-btn {
        width: auto;
        min-width: 10;
        margin-left: 1;
    }

    CreateAgentScreen #buttons {
        height: auto;
        margin-top: 2;
        align: right middle;
    }

    CreateAgentScreen #buttons Button {
        margin-left: 1;
    }

    """

    # Providers that support permission/tracking options
    _OPTION_PROVIDERS = {"claude", "codex"}

    def __init__(self) -> None:
        super().__init__()
        self._workspace_path = _load_last_workspace()
        self._agent_type = _load_last_agent_type()
        self._claude_permission_mode = _load_permission_mode("claude")
        self._claude_tracking_mode = _load_tracking_mode("claude")
        self._codex_permission_mode = _load_permission_mode("codex")
        self._codex_tracking_mode = _load_tracking_mode("codex")

    def compose(self) -> ComposeResult:
        with Container(id="create-agent-root"):
            with VerticalScroll(id="create-agent-scroll"):
                yield Static("Create New Agent", id="create-agent-title")

                yield Label("Agent Type", classes="section-label")
                with RadioSet(id="agent-type"):
                    yield RadioButton("Claude", id="type-claude", value=self._agent_type == "claude")
                    yield RadioButton("Codex", id="type-codex", value=self._agent_type == "codex")
                    yield RadioButton(f"Shell ({_user_shell_name()})", id="type-shell", value=self._agent_type == "shell")

                with Vertical(id="workspace-section"):
                    yield Label("Workspace", classes="section-label")
                    with Horizontal(id="workspace-row"):
                        yield Input(self._workspace_path, id="workspace-path")
                        yield Button("Browse", id="browse-btn", variant="default")

                with Vertical(id="claude-options"):
                    yield Label("Permission Mode", classes="section-label")
                    with RadioSet(id="claude-permission-mode"):
                        yield RadioButton("Normal", id="claude-perm-normal", value=True)
                        yield RadioButton(
                            "Skip (--dangerously-skip-permissions)",
                            id="claude-perm-skip",
                        )

                    yield Label("Tracking", classes="section-label")
                    with RadioSet(id="claude-tracking-mode"):
                        yield RadioButton("Track", id="claude-track-track", value=True)
                        yield RadioButton("No Track (skip LLM summaries)", id="claude-track-notrack")

                with Vertical(id="codex-options", classes="hidden"):
                    yield Label("Permission Mode", classes="section-label")
                    with RadioSet(id="codex-permission-mode"):
                        yield RadioButton("Normal", id="codex-perm-normal", value=True)
                        yield RadioButton(
                            "Full Auto (--full-auto)",
                            id="codex-perm-skip",
                        )

                    yield Label("Tracking", classes="section-label")
                    with RadioSet(id="codex-tracking-mode"):
                        yield RadioButton("Track", id="codex-track-track", value=True)
                        yield RadioButton("No Track (skip LLM summaries)", id="codex-track-notrack")

            with Horizontal(id="buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Create", id="create", variant="primary")

    def _sync_provider_options(self, provider: str) -> None:
        """Show/hide provider options based on selected agent type."""
        claude_options = self.query_one("#claude-options", Vertical)
        codex_options = self.query_one("#codex-options", Vertical)
        if provider == "claude":
            claude_options.remove_class("hidden")
            codex_options.add_class("hidden")
        elif provider == "codex":
            claude_options.add_class("hidden")
            codex_options.remove_class("hidden")
        else:
            claude_options.add_class("hidden")
            codex_options.add_class("hidden")

    def on_mount(self) -> None:
        # Restore saved Claude modes
        if self._claude_permission_mode == "skip":
            self.query_one("#claude-perm-skip", RadioButton).value = True
        # Restore saved Claude tracking mode
        if self._claude_tracking_mode == "notrack":
            self.query_one("#claude-track-notrack", RadioButton).value = True
        # Restore saved Codex modes
        if self._codex_permission_mode == "skip":
            self.query_one("#codex-perm-skip", RadioButton).value = True
        # Restore saved Codex tracking mode
        if self._codex_tracking_mode == "notrack":
            self.query_one("#codex-track-notrack", RadioButton).value = True
        self._sync_provider_options(self._agent_type)
        # Sync highlight (blue background) to match pressed button in all RadioSets
        self._sync_radio_set_highlights()
        self.query_one("#create", Button).focus()

    def _sync_radio_set_highlights(self) -> None:
        """Ensure the highlight (blue background) follows the selected (pressed) button."""
        for radio_set in self.query(RadioSet):
            pressed_index = radio_set.pressed_index
            if pressed_index >= 0:
                radio_set._selected = pressed_index

    def action_close(self) -> None:
        self.dismiss(None)

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Show/hide provider options based on selected agent type."""
        if event.radio_set.id != "agent-type":
            return
        provider = "claude"
        if event.pressed:
            provider_map = {
                "type-claude": "claude",
                "type-codex": "codex",
                "type-shell": "shell",
            }
            provider = provider_map.get(event.pressed.id or "", "claude")
        self._agent_type = provider
        _save_last_agent_type(provider)
        self._sync_provider_options(provider)

    def _update_workspace_display(self) -> None:
        """Update the workspace path display."""
        self.query_one("#workspace-path", Input).value = self._workspace_path

    async def _select_workspace(self) -> str | None:
        """Open a folder picker dialog and return selected path, or None if cancelled."""
        import shutil
        import sys

        default_path = self._workspace_path

        if sys.platform == "darwin":
            return await self._select_workspace_macos(default_path)

        # Linux: try zenity, then kdialog
        if shutil.which("zenity"):
            return await self._select_workspace_cmd(
                ["zenity", "--file-selection", "--directory", f"--filename={default_path}/"],
            )
        if shutil.which("kdialog"):
            return await self._select_workspace_cmd(
                ["kdialog", "--getexistingdirectory", default_path],
            )

        self.app.notify(
            "No file picker available. Edit the path directly.",
            title="Browse",
            severity="warning",
        )
        self.query_one("#workspace-path", Input).focus()
        return None

    async def _select_workspace_macos(self, default_path: str) -> str | None:
        """Open macOS folder picker via AppleScript."""
        default_path_escaped = default_path.replace('"', '\\"')
        prompt_escaped = "Select workspace folder"
        script = f"""
            set defaultFolder to POSIX file "{default_path_escaped}" as alias
            try
                set selectedFolder to choose folder with prompt "{prompt_escaped}" default location defaultFolder
                return POSIX path of selectedFolder
            on error
                return ""
            end try
        """
        return await self._select_workspace_cmd(["osascript", "-e", script])

    async def _select_workspace_cmd(self, cmd: list[str]) -> str | None:
        """Run a folder picker command and return the selected path."""
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
            if result and os.path.isdir(result):
                return result
            return None
        except Exception:
            return None

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""

        if button_id == "cancel":
            self.dismiss(None)
            return

        if button_id == "browse-btn":
            new_path = await self._select_workspace()
            if new_path:
                self._workspace_path = new_path
                self._update_workspace_display()
            return

        if button_id == "create":
            # Get selected agent type
            radio_set = self.query_one("#agent-type", RadioSet)
            pressed_button = radio_set.pressed_button
            if pressed_button is None:
                self.app.notify("Please select an agent type", severity="warning")
                return

            agent_type_map = {
                "type-claude": "claude",
                "type-codex": "codex",
                "type-shell": "shell",
            }
            agent_type = agent_type_map.get(pressed_button.id or "", "claude")
            _save_last_agent_type(agent_type)

            # Get permission mode and tracking mode (for Claude and Codex)
            skip_permissions = False
            no_track = False
            if agent_type in self._OPTION_PROVIDERS:
                perm_radio_set = self.query_one(f"#{agent_type}-permission-mode", RadioSet)
                perm_pressed = perm_radio_set.pressed_button
                skip_permissions = (
                    perm_pressed is not None
                    and perm_pressed.id == f"{agent_type}-perm-skip"
                )
                _save_permission_mode("skip" if skip_permissions else "normal", agent_type)

                track_radio_set = self.query_one(f"#{agent_type}-tracking-mode", RadioSet)
                track_pressed = track_radio_set.pressed_button
                no_track = (
                    track_pressed is not None
                    and track_pressed.id == f"{agent_type}-track-notrack"
                )
                _save_tracking_mode("notrack" if no_track else "track", agent_type)

            # Read workspace path from Input (user may have edited it manually)
            workspace = self.query_one("#workspace-path", Input).value.strip()
            if workspace:
                self._workspace_path = workspace

            # Save the workspace path for next time
            _save_last_workspace(self._workspace_path)

            # Return the result
            self.dismiss((agent_type, self._workspace_path, skip_permissions, no_track))
