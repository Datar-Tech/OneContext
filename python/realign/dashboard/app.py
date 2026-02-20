"""Aline Dashboard - Main Application."""

import asyncio
import os
import threading
import time
import traceback
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.messages import ExitApp
from textual.worker import Worker, WorkerState
from textual.widgets import Footer, TabbedContent, TabPane

from ..logging_config import setup_logger
from .branding import BRANDING
from .diagnostics import DashboardDiagnostics
from .widgets import (
    AlineHeader,
    ConfigPanel,
    AgentsPanel,
    RightStatusBar,
    WatcherPanel,
    WorkerPanel,
)
from .state import get_dashboard_state_value

# Environment variable to control terminal mode
ENV_TERMINAL_MODE = "ALINE_TERMINAL_MODE"

# Set up dashboard logger - logs to ~/.aline/.logs/dashboard.log
logger = setup_logger("realign.dashboard", "dashboard.log")


class AlineDashboard(App):
    """Aline Interactive Dashboard - TUI for monitoring and managing Aline."""

    CSS_PATH = "styles/dashboard.tcss"
    TITLE = BRANDING.dashboard_title
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("r", "refresh", "Refresh", show=False),
        Binding("?", "help", "Help"),
        Binding("tab", "next_tab", "Next Tab", priority=True, show=False),
        Binding("shift+tab", "prev_tab", "Prev Tab", priority=True, show=False),
        Binding("ctrl+d", "dump_diagnostics", "Dump Diagnostics", show=False),
        Binding("ctrl+c", "quit_confirm", "Quit", priority=True),
    ]

    def __init__(self, use_native_terminal: bool | None = None, *, dev: bool = False):
        """Initialize the dashboard.

        Args:
            use_native_terminal: If True, use native terminal backend (iTerm2/Kitty).
                                 If False, use tmux.
                                 If None (default), auto-detect from ALINE_TERMINAL_MODE env var.
            dev: If True, enable developer mode (shows Watcher and Worker tabs).
        """
        super().__init__()
        self.use_native_terminal = use_native_terminal
        self.dev = bool(dev)
        self._native_terminal_mode = self._detect_native_mode()
        self._local_api_server = None
        self._diagnostics = DashboardDiagnostics.start()
        self._tmux_width_timer = None
        self._diagnostics.install_global_exception_hooks()
        self._watchdog_timer = None
        self._policy_timer = None
        self._policy_fetching = False
        self._policy_last_checked_at = 0.0
        self._upgrade_gate_triggered = False
        self._exit_reason = "not_set"
        self._exit_context: dict[str, str] = {}
        self._exit_call_count = 0
        self._explicit_exit_requested = False
        self._apply_saved_theme()
        self._diagnostics.event(
            "dashboard_init",
            native_terminal=bool(self._native_terminal_mode),
        )
        logger.info(f"AlineDashboard initialized (native_terminal={self._native_terminal_mode})")

    def _detect_native_mode(self) -> bool:
        """Detect if native terminal mode should be used."""
        if self.use_native_terminal is not None:
            return self.use_native_terminal

        mode = os.environ.get(ENV_TERMINAL_MODE, "").strip().lower()
        return mode in {"native", "iterm2", "iterm", "kitty"}

    def _normalize_exit_context(self, context: dict[str, Any]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for key, value in context.items():
            if value is None:
                continue
            text = str(value)
            if len(text) > 240:
                text = text[:240] + "..."
            normalized[str(key)] = text
        return normalized

    def _set_exit_reason(self, reason: str, **context: Any) -> None:
        if reason and self._exit_reason == "not_set":
            self._exit_reason = str(reason)
        normalized = self._normalize_exit_context(context)
        if normalized:
            self._exit_context.update(normalized)

    def _request_exit(self, reason: str, **context: Any) -> None:
        self._explicit_exit_requested = True
        self._set_exit_reason(reason, **context)
        self.exit()

    def _record_exit_hook(self, hook: str, **context: Any) -> None:
        if self._exit_reason == "not_set":
            self._set_exit_reason(f"textual_{hook}")
        normalized = self._normalize_exit_context(context)
        logger.info(
            "Exit hook %s (reason=%s explicit=%s exit_calls=%s context=%s hook_context=%s)",
            hook,
            self._exit_reason,
            bool(self._explicit_exit_requested),
            self._exit_call_count,
            self._exit_context,
            normalized,
        )
        try:
            self._diagnostics.event(
                f"dashboard_exit_hook_{hook}",
                exit_reason=self._exit_reason,
                exit_calls=self._exit_call_count,
                explicit_exit=bool(self._explicit_exit_requested),
                context=str(self._exit_context),
                hook_context=str(normalized),
            )
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        logger.debug("compose() started")
        try:
            yield AlineHeader()
            with TabbedContent(initial="agents"):
                with TabPane("Contexts", id="agents"):
                    yield AgentsPanel()
                with TabPane("Config", id="config"):
                    yield ConfigPanel()
                if self.dev:
                    with TabPane("Watcher", id="watcher"):
                        yield WatcherPanel()
                    with TabPane("Worker", id="worker"):
                        yield WorkerPanel()
            with Horizontal(id="dashboard-footer-row"):
                yield Footer()
                yield RightStatusBar(id="status-bar", width=35)
            logger.debug("compose() completed successfully")
        except Exception as e:
            logger.error(f"compose() failed: {e}\n{traceback.format_exc()}")
            raise

    def set_status_bar(
        self,
        text: str,
        *,
        spinning: bool = True,
        variant: str = "active",
        clear_after_s: float | None = None,
    ) -> None:
        try:
            bar = self.query_one("#status-bar", RightStatusBar)
        except Exception:
            return
        bar.set_status(
            text,
            spinning=spinning,
            variant=variant,
            clear_after_s=clear_after_s,
        )

    def clear_status_bar(self) -> None:
        self.set_status_bar("", spinning=False)

    def _tab_ids(self) -> list[str]:
        tabs = ["agents", "config"]
        if self.dev:
            tabs.extend(["watcher", "worker"])
        return tabs

    def _auto_copy_selection(self, screen) -> None:
        """Auto-copy selected text to the system clipboard.

        macOS Cmd+C is handled by the terminal emulator and never reaches
        the Textual app, so we proactively write selected text to the
        clipboard the moment the user finishes dragging.
        """
        try:
            selected = screen.get_selected_text()
            if selected:
                from .clipboard import copy_text

                copy_text(self, selected)
        except Exception:
            pass

    def _install_auto_copy(self, screen) -> None:
        """Install a watcher on screen._selecting so we auto-copy on mouse-up."""
        try:
            self.watch(
                screen,
                "_selecting",
                lambda selecting: (None if selecting else self._auto_copy_selection(screen)),
            )
        except Exception:
            pass

    def push_screen(self, screen, *args, **kwargs):
        result = super().push_screen(screen, *args, **kwargs)
        self._install_auto_copy(self.screen)
        return result

    def on_mount(self) -> None:
        """Apply dashboard theme based on saved preference."""
        logger.info("on_mount() started")
        self._install_auto_copy(self.screen)
        try:
            try:
                loop = asyncio.get_running_loop()
                self._diagnostics.install_asyncio_exception_handler(loop)
            except Exception:
                pass

            # Start local API server for one-click browser import
            self._start_local_api_server()

            # Set up side-by-side layout for native terminal mode
            if self._native_terminal_mode:
                self._setup_native_terminal_layout()

            # Watchdog: capture snapshots for intermittent blank/stuck UI issues.
            if self._watchdog_timer is None:
                self._watchdog_timer = self.set_interval(5.0, self._watchdog_check)
            if self._policy_timer is None:
                self._policy_timer = self.set_interval(5.0, self._poll_server_upgrade_policy)
            self._poll_server_upgrade_policy(force=True)
            self.call_later(self._maybe_prompt_import_history)

            logger.info("on_mount() completed successfully")
        except Exception as e:
            logger.error(f"on_mount() failed: {e}\n{traceback.format_exc()}")
            raise

    def _maybe_prompt_import_history(self) -> None:
        try:
            if bool(get_dashboard_state_value("skip_import_history_prompt", False)):
                return
            tabbed = self.query_one(TabbedContent)
            if str(getattr(tabbed, "active", "") or "") != "agents":
                return
        except Exception:
            return

        try:
            from .screens.import_history import ImportHistoryScreen
            from .screens.import_history_prompt import ImportHistoryPromptScreen
            from .state import set_dashboard_state_value
        except Exception:
            return

        def _on_decision(result: dict[str, object] | None) -> None:
            decision = result or {}
            if bool(decision.get("never_show_again")):
                set_dashboard_state_value("skip_import_history_prompt", True)
            if bool(decision.get("go")):
                def _open_import_history() -> None:
                    for screen in getattr(self, "screen_stack", []):
                        if isinstance(screen, ImportHistoryScreen):
                            return
                    self.push_screen(ImportHistoryScreen())

                self.call_later(_open_import_history)

        self.push_screen(ImportHistoryPromptScreen(), _on_decision)

    def on_resize(self) -> None:
        """Keep the tmux outer layout stable when the terminal is resized."""
        if self._native_terminal_mode:
            return

        # Debounce: terminal resize can generate many events; keep the UI responsive.
        try:
            if self._tmux_width_timer is not None:
                try:
                    self._tmux_width_timer.stop()
                except Exception:
                    pass
            self._tmux_width_timer = self.set_timer(0.05, self._enforce_tmux_dashboard_width)
        except Exception:
            return

    def _enforce_tmux_dashboard_width(self) -> None:
        try:
            from . import tmux_manager

            tmux_manager.enforce_outer_dashboard_pane_width()
        except Exception:
            return

    def on_unmount(self) -> None:
        reason = self._exit_reason
        exit_calls = self._exit_call_count
        explicit = bool(self._explicit_exit_requested)
        context = dict(self._exit_context)

        logger.info(
            "on_unmount() (reason=%s explicit=%s exit_calls=%s context=%s)",
            reason,
            explicit,
            exit_calls,
            context,
        )
        if reason == "not_set":
            logger.warning("Dashboard unmounted without explicit recorded exit reason")

        try:
            self._diagnostics.event(
                "dashboard_unmount",
                exit_reason=reason,
                exit_calls=exit_calls,
                explicit_exit=explicit,
                context=str(context),
            )
        except Exception:
            pass

        if self._native_terminal_mode:
            return
        if self._policy_timer is not None:
            try:
                self._policy_timer.stop()
            except Exception:
                pass
        try:
            from . import tmux_manager

            panes = tmux_manager.list_outer_panes(timeout_s=0.2)
            logger.info(
                "on_unmount outer pane snapshot (count=%s panes=%s)",
                len(panes),
                panes,
            )
            msgs = tmux_manager.list_outer_messages_tail(160, timeout_s=1.5)
            if msgs:
                lowered_keywords = (
                    "kill",
                    "respawn",
                    "detach",
                    "attach",
                    "split-window",
                    "destroy",
                    "has-session",
                )
                interesting = [ln for ln in msgs if any(k in ln.lower() for k in lowered_keywords)]
                logger.info(
                    "on_unmount outer tmux messages (tail=%s interesting=%s)",
                    len(msgs),
                    interesting[-40:],
                )
                try:
                    self._diagnostics.event(
                        "dashboard_unmount_outer_tmux_messages",
                        tail_count=len(msgs),
                        interesting_count=len(interesting),
                        interesting_tail=str(interesting[-40:]),
                    )
                except Exception:
                    pass
            else:
                logger.info("on_unmount outer tmux messages unavailable/empty")
                try:
                    self._diagnostics.event(
                        "dashboard_unmount_outer_tmux_messages",
                        tail_count=0,
                        interesting_count=0,
                        interesting_tail="[]",
                    )
                except Exception:
                    pass
        except Exception:
            return

    def exit(
        self,
        result: object | None = None,
        return_code: int = 0,
        message: object | None = None,
    ) -> None:
        self._exit_call_count += 1
        if self._exit_reason == "not_set":
            self._set_exit_reason("direct_exit_call")

        stack = "".join(traceback.format_stack(limit=20))
        logger.info(
            "AlineDashboard.exit called (count=%s reason=%s explicit=%s context=%s)\n%s",
            self._exit_call_count,
            self._exit_reason,
            bool(self._explicit_exit_requested),
            self._exit_context,
            stack,
        )
        try:
            self._diagnostics.event(
                "dashboard_exit_called",
                exit_calls=self._exit_call_count,
                exit_reason=self._exit_reason,
                explicit_exit=bool(self._explicit_exit_requested),
                return_code=return_code,
                context=str(self._exit_context),
            )
        except Exception:
            pass
        return super().exit(result=result, return_code=return_code, message=message)

    async def action_quit(self) -> None:
        self._record_exit_hook("action_quit")
        await super().action_quit()

    async def on_exit_app(self, message: ExitApp) -> None:
        sender = getattr(message, "sender", None)
        self._record_exit_hook(
            "on_exit_app_message",
            sender_type=type(sender).__name__ if sender is not None else "None",
            sender_repr=repr(sender),
        )

    async def _on_exit_app(self) -> None:
        self._record_exit_hook("_on_exit_app")
        await super()._on_exit_app()

    async def _shutdown(self) -> None:
        self._record_exit_hook(
            "_shutdown",
            running=str(getattr(self, "_running", "")),
            mounted=str(bool(self.is_mounted)),
        )
        await super()._shutdown()

    async def _message_loop_exit(self) -> None:
        self._record_exit_hook("_message_loop_exit")
        await super()._message_loop_exit()

    def _handle_exception(self, error: Exception) -> None:
        self._set_exit_reason(
            "textual_handle_exception",
            error_type=type(error).__name__,
            error=str(error),
        )
        tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        logger.error(
            "Textual unhandled exception (type=%s reason=%s explicit=%s exit_calls=%s)\n%s",
            type(error).__name__,
            self._exit_reason,
            bool(self._explicit_exit_requested),
            self._exit_call_count,
            tb,
        )
        try:
            self._diagnostics.exception(
                "textual_unhandled_exception",
                error,
                exit_reason=self._exit_reason,
                exit_calls=self._exit_call_count,
                explicit_exit=bool(self._explicit_exit_requested),
                context=str(self._exit_context),
            )
        except Exception:
            pass
        super()._handle_exception(error)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state != WorkerState.ERROR:
            return
        try:
            worker = event.worker
            err = getattr(worker, "error", None)
            if isinstance(err, BaseException):
                self._diagnostics.exception(
                    "worker_error",
                    err,
                    worker_name=str(getattr(worker, "name", "") or ""),
                    worker_group=str(getattr(worker, "group", "") or ""),
                    worker_state=str(event.state),
                )
            else:
                self._diagnostics.event(
                    "worker_error",
                    error=str(err),
                    worker_name=str(getattr(worker, "name", "") or ""),
                    worker_group=str(getattr(worker, "group", "") or ""),
                    worker_state=str(event.state),
                )
        except Exception:
            return

    def _watchdog_check(self) -> None:
        """Lightweight watchdog to detect UI blanking and missing panes."""
        try:
            panel = self.query_one(AgentsPanel)
            state = panel.diagnostics_state()
            agents_count = int(state.get("agents_count", 0) or 0)
            children = int(state.get("agents_list_children", 1) or 0)
            if agents_count > 0 and children == 0:
                self._diagnostics.snapshot(
                    reason="agents_panel_blank", app=self, agents_panel=state
                )
                panel.force_render(reason="watchdog_blank")
        except Exception:
            pass

    def _poll_server_upgrade_policy(self, *, force: bool = False) -> None:
        """Check server policy periodically and force logout/exit when required."""
        if self._upgrade_gate_triggered or self._policy_fetching:
            return

        now = time.time()
        if not force and (now - self._policy_last_checked_at) < 8.0:
            return

        self._policy_last_checked_at = now
        self._policy_fetching = True

        def do_fetch() -> None:
            try:
                from importlib.metadata import version
                from ..auth import get_remote_client_policy

                try:
                    current_version = version("aline-ai")
                except Exception:
                    current_version = "0.0.0"

                policy, _error = get_remote_client_policy(
                    current_version=current_version,
                    timeout_s=3.0,
                )
                if isinstance(policy, dict) and bool(policy.get("upgrade_required")):
                    self.call_from_thread(self._enforce_upgrade_gate_runtime, policy)
            except Exception as e:
                logger.debug(f"Client policy check skipped: {e}")
            finally:
                self._policy_fetching = False

        threading.Thread(target=do_fetch, daemon=True).start()

    def _enforce_upgrade_gate_runtime(self, policy: dict[str, Any]) -> None:
        """Apply force-upgrade policy while dashboard is running."""
        if self._upgrade_gate_triggered:
            return
        self._upgrade_gate_triggered = True

        required_version = str(policy.get("required_version") or "").strip() or "latest"
        force_logout = bool(policy.get("force_logout"))
        message = str(policy.get("message") or "").strip()
        if not message:
            message = (
                f"Aline {required_version} or newer is required. " "Please run `onecontext update`."
            )

        if force_logout:
            try:
                from ..auth import clear_credentials, load_credentials, open_logout_page

                had_credentials = load_credentials() is not None
                clear_credentials()
                threading.Thread(target=ConfigPanel._stop_daemons_quiet, daemon=True).start()
                if had_credentials:
                    try:
                        open_logout_page()
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Forced logout failed: {e}")

        self.notify(message, title="Upgrade Required", severity="error", timeout=8)
        self.set_timer(
            0.8,
            lambda: self._request_exit(
                "server_upgrade_required",
                required_version=required_version,
                force_logout=str(force_logout),
            ),
        )

        # Only relevant for tmux mode.
        if self._native_terminal_mode:
            return
        try:
            from . import tmux_manager

            panes = tmux_manager.list_outer_panes(timeout_s=0.2)
            if panes and len(panes) < 2:
                try:
                    tmux_state = tmux_manager.collect_tmux_debug_state()
                except Exception:
                    tmux_state = {}
                self._diagnostics.snapshot(
                    reason="tmux_outer_missing_right_pane",
                    app=self,
                    tmux_state=tmux_state,
                )
                try:
                    tmux_manager.ensure_right_pane_ready()
                except Exception:
                    pass
            else:
                pane1_cmd = ""
                for ln in panes:
                    parts = ln.split("\t")
                    if parts and parts[0] == "1":
                        pane1_cmd = parts[2] if len(parts) > 2 else ""
                        break
                if pane1_cmd and pane1_cmd != "tmux":
                    try:
                        tmux_state = tmux_manager.collect_tmux_debug_state()
                    except Exception:
                        tmux_state = {}
                    self._diagnostics.snapshot(
                        reason="tmux_right_pane_not_attached",
                        app=self,
                        pane1_current_command=pane1_cmd,
                        tmux_state=tmux_state,
                    )
                    try:
                        tmux_manager.ensure_right_pane_ready()
                    except Exception:
                        pass
        except Exception:
            pass

    def _start_local_api_server(self) -> None:
        """Start the local HTTP API server for browser-based agent import."""
        try:
            from ..config import ReAlignConfig
            from .local_api import LocalAPIServer

            config = ReAlignConfig.load()
            self._local_api_server = LocalAPIServer(port=config.local_api_port)
            self._local_api_server.start()
        except Exception as e:
            logger.warning(f"Could not start local API server: {e}")

    def _apply_saved_theme(self) -> None:
        theme_choice = str(get_dashboard_state_value("theme", "dark")).strip().lower()
        if theme_choice == "light":
            self.theme = "textual-light"
        else:
            self.theme = "textual-dark"

    def _setup_native_terminal_layout(self) -> None:
        """Set up side-by-side layout for Dashboard and native terminal."""
        # Skip if using iTerm2 split pane mode (already set up by CLI)
        if os.environ.get("ALINE_ITERM2_RIGHT_PANE"):
            logger.info("Using iTerm2 split pane mode, skipping window layout")
            return

        try:
            from .layout import setup_side_by_side_layout

            # Determine the target terminal app
            mode = os.environ.get(ENV_TERMINAL_MODE, "").strip().lower()
            if mode == "kitty":
                terminal_app = "Kitty"
            else:
                terminal_app = "iTerm2"

            # Set up side-by-side layout (Dashboard on left, terminal on right)
            success = setup_side_by_side_layout(
                terminal_app=terminal_app,
                dashboard_on_left=True,
                dashboard_width_percent=40,  # Dashboard takes 40%, terminal takes 60%
            )

            if success:
                logger.info(f"Set up side-by-side layout with {terminal_app}")
            else:
                logger.warning("Failed to set up side-by-side layout")
        except Exception as e:
            logger.warning(f"Could not set up native terminal layout: {e}")

    def action_next_tab(self) -> None:
        """Switch to next tab."""
        tabbed_content = self.query_one(TabbedContent)
        tabs = self._tab_ids()
        current = tabbed_content.active
        if current in tabs:
            idx = tabs.index(current)
            next_idx = (idx + 1) % len(tabs)
            tabbed_content.active = tabs[next_idx]

    def action_prev_tab(self) -> None:
        """Switch to previous tab."""
        tabbed_content = self.query_one(TabbedContent)
        tabs = self._tab_ids()
        current = tabbed_content.active
        if current in tabs:
            idx = tabs.index(current)
            prev_idx = (idx - 1) % len(tabs)
            tabbed_content.active = tabs[prev_idx]

    async def action_refresh(self) -> None:
        """Refresh the current tab."""
        try:
            self._diagnostics.event("action_refresh")
        except Exception:
            pass
        tabbed_content = self.query_one(TabbedContent)
        active_tab_id = tabbed_content.active

        if active_tab_id == "agents":
            self.query_one(AgentsPanel).refresh_data()
        elif active_tab_id == "config":
            self.query_one(ConfigPanel).refresh_data()
        elif active_tab_id == "watcher" and self.dev:
            self.query_one(WatcherPanel).refresh_data()
        elif active_tab_id == "worker" and self.dev:
            self.query_one(WorkerPanel).refresh_data()

    def action_help(self) -> None:
        """Show help information."""
        from .screens import HelpScreen

        self.push_screen(HelpScreen())

    def action_dump_diagnostics(self) -> None:
        try:
            self._diagnostics.snapshot(reason="manual_dump", app=self)
            path = self._diagnostics.path
            if path:
                self.notify(f"Diagnostics written: {path}", title="Diagnostics", timeout=4)
            else:
                self.notify(
                    "Diagnostics written to stderr (no log dir)", title="Diagnostics", timeout=4
                )
        except Exception as e:
            self.notify(
                f"Diagnostics failed: {e}", title="Diagnostics", severity="error", timeout=4
            )

    def action_quit_confirm(self) -> None:
        """Copy selected text if any, otherwise show exit confirmation dialog."""
        # If there is selected text, copy it instead of quitting.
        selected = self.screen.get_selected_text()
        if selected:
            from .clipboard import copy_text

            copy_text(self, selected)
            self.screen.clear_selection()
            return

        # Prevent stacking multiple exit dialogs.
        from .screens.logout_confirm import ExitConfirmScreen

        if isinstance(self.screen, ExitConfirmScreen):
            return

        def _on_confirm(confirmed: bool) -> None:
            if confirmed:
                self._request_exit("ctrl_c_confirmed")

        self.push_screen(ExitConfirmScreen(mode="exit"), _on_confirm)


def run_dashboard(use_native_terminal: bool | None = None) -> None:
    """Run the Aline Dashboard.

    Args:
        use_native_terminal: If True, use native terminal backend (iTerm2/Kitty).
                             If False, use tmux.
                             If None (default), auto-detect from ALINE_TERMINAL_MODE env var.
    """
    logger.info("Starting Aline Dashboard")
    try:
        app = AlineDashboard(use_native_terminal=use_native_terminal)
        app.run()
        logger.info(
            "Aline Dashboard exited normally (reason=%s exit_calls=%s explicit=%s return_code=%s return_value=%s context=%s)",
            getattr(app, "_exit_reason", "unknown"),
            getattr(app, "_exit_call_count", 0),
            bool(getattr(app, "_explicit_exit_requested", False)),
            getattr(app, "return_code", "unknown"),
            getattr(app, "return_value", "unknown"),
            getattr(app, "_exit_context", {}),
        )
    except Exception as e:
        logger.error(f"Dashboard crashed: {e}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    run_dashboard()
