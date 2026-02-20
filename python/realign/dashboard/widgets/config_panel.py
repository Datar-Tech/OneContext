"""Config Panel Widget for viewing and editing configuration."""

import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Center, Horizontal, VerticalScroll
from textual.widgets import Button, Static, Switch

from ..tmux_manager import _run_outer_tmux
from ..state import get_dashboard_state_value, set_dashboard_state_value
from ...auth import (
    load_credentials,
    save_credentials,
    clear_credentials,
    open_login_page,
    open_logout_page,
    get_current_user,
    get_billing_status,
    get_today_llm_calls,
    find_free_port,
    start_callback_server,
    validate_cli_token,
)
from ...config import ReAlignConfig
from ..branding import BRANDING
from ...logging_config import setup_logger

logger = setup_logger("realign.dashboard.widgets.config_panel", "dashboard.log")


class ConfigPanel(Static):
    """Panel for viewing and editing Aline configuration."""

    DEFAULT_CSS = """
    ConfigPanel {
        height: 100%;
        padding: 0;
    }

    ConfigPanel VerticalScroll {
        height: 1fr;
        padding: 1;
    }

    ConfigPanel .section-title {
        text-style: bold;
        margin-bottom: 1;
    }

    ConfigPanel .account-section {
        height: 3;
        align: left middle;
    }

    ConfigPanel .account-section .account-email {
        width: 1fr;
        margin-right: 1;
        content-align: left middle;
        height: 3;
    }

    ConfigPanel .account-section Button {
        width: auto;
        height: 3;
    }

    ConfigPanel .account-limit-row {
        height: auto;
        align: left middle;
    }

    ConfigPanel .account-usage-row {
        height: 3;
        align: left middle;
        margin-top: 1;
    }

    ConfigPanel #usage-label {
        width: auto;
        height: 3;
        margin-right: 1;
        content-align: left middle;
    }

    ConfigPanel #usage-progress-bar {
        width: 32;
        height: 3;
        padding: 0;
        margin-right: 1;
        content-align: left middle;
    }

    ConfigPanel #usage-percentage {
        width: auto;
        height: 3;
        margin-right: 1;
        color: $text-muted;
        content-align: left middle;
    }

    ConfigPanel .account-usage-row Button {
        width: auto;
        height: 3;
    }

    ConfigPanel #account-limit-note {
        width: 1fr;
        margin-right: 1;
    }

    ConfigPanel #upgrade-pro-btn {
        width: auto;
        margin-right: 2;
    }

    ConfigPanel .settings-section {
        height: auto;
        margin-top: 2;
    }

    ConfigPanel #version-label {
        content-align: center middle;
        width: 100%;
    }

    ConfigPanel .settings-section .setting-row {
        height: auto;
        align: left middle;
        margin-bottom: 1;
    }

    ConfigPanel .settings-section .setting-label {
        width: 1fr;
    }

    ConfigPanel .settings-section .setting-control {
        width: auto;
        height: auto;
        align: right middle;
    }

    ConfigPanel .settings-section .toggle-label {
        width: auto;
        height: auto;
        color: $text-muted;
    }

    ConfigPanel .settings-section Switch.compact-toggle {
        width: auto;
        margin: 0 1;
        border: none;
        background: transparent;
    }

    ConfigPanel .settings-section Switch.compact-toggle:focus {
        border: none;
        background-tint: transparent;
    }

    ConfigPanel .settings-section .button-row {
        height: 3;
        align: right middle;
    }

    ConfigPanel .settings-section .button-row Button {
        margin-right: 0;
    }

    ConfigPanel .settings-section Center {
        height: auto;
    }

    ConfigPanel .settings-section Center Button {
        width: 30;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._border_resize_enabled: bool = True  # Track tmux border resize state
        self._syncing_controls: bool = False  # Flag to prevent recursive UI updates
        self._login_in_progress: bool = False  # Track login state
        self._refresh_timer = None  # Timer for auto-refresh
        self._billing_cache: Optional[dict] = None
        self._billing_cache_uid: Optional[str] = None
        self._billing_cache_at: float = 0.0
        self._billing_fetching: bool = False
        self._usage_cache_calls_today: Optional[int] = None
        self._usage_cache_uid: Optional[str] = None
        self._usage_cache_at: float = 0.0
        self._usage_fetching: bool = False

    @staticmethod
    def _get_version() -> str:
        """Get the Aline version."""
        try:
            from importlib.metadata import version

            return version("aline-ai")
        except Exception:
            return "0.0.0"

    def compose(self) -> ComposeResult:
        """Compose the config panel layout."""
        with VerticalScroll():
            with Static(classes="settings-section"):
                yield Static("[bold]Account[/bold]", classes="section-title")
                with Horizontal(classes="account-section"):
                    yield Static(id="account-email", classes="account-email")
                    yield Button("Login", id="auth-btn", variant="primary")
                    yield Button("Exit", id="exit-btn", variant="error")
                with Horizontal(classes="account-usage-row"):
                    yield Static("Daily Usage: ", id="usage-label")
                    yield Static("", id="usage-progress-bar")
                    yield Static("", id="usage-percentage")
                    yield Button("Choose Plan", id="upgrade-pro-btn", variant="success")
                with Horizontal(classes="account-limit-row"):
                    yield Static("", id="account-limit-note")

            with Static(classes="settings-section"):
                yield Static("[bold]Settings[/bold]", classes="section-title")
                with Horizontal(classes="setting-row"):
                    yield Static("Dark mode", classes="setting-label")
                    with Horizontal(classes="setting-control"):
                        yield Static("Off", classes="toggle-label")
                        yield Switch(
                            bool(
                                str(get_dashboard_state_value("theme", "dark")).strip().lower()
                                != "light"
                            ),
                            id="theme-toggle",
                            classes="compact-toggle",
                        )
                        yield Static("On", classes="toggle-label")

                with Horizontal(classes="setting-row"):
                    yield Static("Border resize", classes="setting-label")
                    with Horizontal(classes="setting-control"):
                        yield Static("Off", classes="toggle-label")
                        yield Switch(id="border-resize-toggle", classes="compact-toggle")
                        yield Static("On", classes="toggle-label")

                with Horizontal(classes="setting-row"):
                    yield Static("Show archived agents", classes="setting-label")
                    with Horizontal(classes="setting-control"):
                        yield Static("Off", classes="toggle-label")
                        yield Switch(
                            bool(get_dashboard_state_value("show_archived_agents", False)),
                            id="show-archived-toggle",
                            classes="compact-toggle",
                        )
                        yield Static("On", classes="toggle-label")

            with Static(classes="settings-section"):
                yield Static("[bold]Version[/bold]", classes="section-title")
                yield Static(f"OneContext v{self._get_version()}", id="version-label")
                yield Center(Button("Check for Updates", id="check-updates-btn", variant="default"))

            with Static(classes="settings-section"):
                yield Static("[bold]Tools[/bold]", classes="section-title")
                yield Center(Button("Import history", id="import-history-btn", variant="default"))
                yield Center(Button(BRANDING.doctor_label, id="doctor-btn", variant="default"))

            with Static(classes="settings-section"):
                yield Static("[bold]Support[/bold]", classes="section-title")
                yield Center(Button("Report Issues", id="report-issues-btn", variant="default"))
                yield Center(Button("Homepage", id="home-page-btn", variant="default"))

    def on_mount(self) -> None:
        """Set up the panel on mount."""
        # Update account status display
        self._update_account_status()

        # Sync theme from persisted preference
        self._sync_theme_toggle()

        # Query and set the actual tmux border resize state
        self._sync_border_resize_toggle()

        # Sync archived-agent toggle from persisted preference
        self._sync_show_archived_toggle()

    def on_show(self) -> None:
        """Start periodic refresh when visible."""
        if self._refresh_timer is None:
            self._refresh_timer = self.set_interval(5.0, self._update_account_status)
        else:
            try:
                self._refresh_timer.resume()
            except Exception:
                pass
        self._update_account_status()

    def on_hide(self) -> None:
        """Pause periodic refresh when hidden."""
        if self._refresh_timer is None:
            return
        try:
            self._refresh_timer.pause()
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "auth-btn":
            credentials = get_current_user()
            if credentials:
                self._handle_logout()
            else:
                self._handle_login()
            return

        if event.button.id == "exit-btn":
            self._handle_exit()
            return

        if event.button.id == "doctor-btn":
            self._handle_doctor()
            return

        if event.button.id == "import-history-btn":
            from ..screens.import_history import ImportHistoryScreen

            for screen in getattr(self.app, "screen_stack", []):
                if isinstance(screen, ImportHistoryScreen):
                    return
            self.app.push_screen(ImportHistoryScreen())
            return

        if event.button.id == "check-updates-btn":
            self._handle_check_updates()
            return

        if event.button.id == "report-issues-btn":
            webbrowser.open("https://github.com/TheAgentContextLab/OneContext/issues")
            return

        if event.button.id == "home-page-btn":
            webbrowser.open("https://one-context.com/")
            return

        if event.button.id == "upgrade-pro-btn":
            self._handle_upgrade_pro()
            return

        if self._syncing_controls:
            return

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if self._syncing_controls:
            return

        switch_id = (event.switch.id or "").strip()

        if switch_id == "theme-toggle":
            self._set_theme("dark" if event.value else "light")
            return

        if switch_id == "border-resize-toggle":
            self._toggle_border_resize(bool(event.value))
            return

        if switch_id == "show-archived-toggle":
            set_dashboard_state_value("show_archived_agents", bool(event.value))
            try:
                from .agents_panel import AgentsPanel

                self.app.query_one(AgentsPanel).refresh_data()
            except Exception:
                pass
            return

    def _update_account_status(self) -> None:
        """Update the account status display."""
        try:
            email_widget = self.query_one("#account-email", Static)
            auth_btn = self.query_one("#auth-btn", Button)
            limit_note = self.query_one("#account-limit-note", Static)
            upgrade_btn = self.query_one("#upgrade-pro-btn", Button)
            usage_bar = self.query_one("#usage-progress-bar", Static)
            usage_pct = self.query_one("#usage-percentage", Static)
        except Exception:
            # Widget not ready yet
            return

        # Don't update if login is in progress
        if self._login_in_progress:
            return

        credentials = get_current_user()
        if credentials:
            self._maybe_refresh_billing_status(credentials.user_id)
            self._maybe_refresh_usage_stats(credentials.user_id)
            billing = (
                self._billing_cache if self._billing_cache_uid == credentials.user_id else None
            )
            is_pro = bool(billing and billing.get("is_pro") is True)
            plan_tier = (
                str((billing or {}).get("plan_tier") or "").strip().lower()
                if isinstance(billing, dict)
                else ""
            )
            is_pro_plus = is_pro and plan_tier == "pro_plus"

            if is_pro:
                if is_pro_plus:
                    email_widget.update(f"[bold]{credentials.email}[/bold] [cyan]PRO+[/cyan]")
                else:
                    email_widget.update(f"[bold]{credentials.email}[/bold] [green]PRO[/green]")
            else:
                email_widget.update(f"[bold]{credentials.email}[/bold]")

            auth_btn.label = "Logout"
            auth_btn.variant = "warning"
            upgrade_btn.display = True

            # Show quota notice only when the latest LLM result is a quota/rate-limit failure.
            limit_code = self._get_latest_llm_limit_code(
                billing if isinstance(billing, dict) else None
            )
            if limit_code == "DAILY_QUOTA_EXCEEDED":
                if is_pro:
                    limit_note.update(
                        "[yellow]Today's plan quota has reached the limit. It will reset automatically tomorrow.[/yellow]"
                    )
                else:
                    limit_note.update(
                        "[yellow]Today's free quota has reached the limit. Upgrade your plan for higher quota.[/yellow]"
                    )
            elif limit_code in (
                "QPS_LIMIT_EXCEEDED",
                "MONTHLY_BUDGET_EXCEEDED",
                "RATE_LIMITED",
                "USER_BLOCKED",
            ):
                if is_pro:
                    limit_note.update(
                        "[yellow]Your plan usage limit has been reached. Please try again later.[/yellow]"
                    )
                else:
                    limit_note.update(
                        "[yellow]LLM usage limit reached. Upgrade your plan for higher quota.[/yellow]"
                    )
            elif is_pro:
                limit_note.update("")
            else:
                limit_note.update("")

            if is_pro_plus:
                usage_bar.update("")
                usage_bar.display = False
                usage_pct.update("[cyan]Usage details hidden for Pro+[/cyan]")
            elif billing and isinstance(billing, dict):
                daily_quota_raw = billing.get("daily_quota")
                try:
                    daily_quota = int(daily_quota_raw) if daily_quota_raw is not None else 0
                except Exception:
                    daily_quota = 0
                calls_today = (
                    self._usage_cache_calls_today
                    if self._usage_cache_uid == credentials.user_id
                    and self._usage_cache_calls_today is not None
                    else 0
                )
                if daily_quota > 0:
                    usage_bar.display = True
                    self._update_usage_bar(usage_bar, usage_pct, calls_today, daily_quota)
                else:
                    usage_bar.display = False
                    usage_bar.update("")
                    usage_pct.update("N/A")
            else:
                usage_bar.display = False
                usage_bar.update("")
                usage_pct.update("[dim]Checking...[/dim]")
        else:
            email_widget.update("[dim]Not logged in[/dim]")
            auth_btn.label = "Login"
            auth_btn.variant = "primary"
            limit_note.update("")
            upgrade_btn.display = False
            usage_bar.display = False
            usage_bar.update("")
            usage_pct.update("")
            self._billing_cache = None
            self._billing_cache_uid = None
            self._billing_cache_at = 0.0
            self._usage_cache_calls_today = None
            self._usage_cache_uid = None
            self._usage_cache_at = 0.0
        auth_btn.disabled = False

    def _maybe_refresh_billing_status(self, user_id: str) -> None:
        """Refresh billing status in background with short cache TTL to avoid UI blocking."""
        now = time.time()
        if self._billing_fetching:
            return
        if self._billing_cache_uid == user_id and (now - self._billing_cache_at) < 30:
            return

        self._billing_fetching = True

        def do_fetch() -> None:
            try:
                billing, error = get_billing_status()
                if billing and isinstance(billing, dict):
                    self._billing_cache = billing
                    self._billing_cache_uid = user_id
                    self._billing_cache_at = time.time()
                elif error:
                    logger.debug(f"Billing status refresh skipped: {error}")
            finally:
                self._billing_fetching = False
                try:
                    self.app.call_from_thread(self._update_account_status)
                except Exception:
                    pass

        threading.Thread(target=do_fetch, daemon=True).start()

    def _maybe_refresh_usage_stats(self, user_id: str) -> None:
        """Refresh today's usage in background with short cache TTL."""
        now = time.time()
        if self._usage_fetching:
            return
        if self._usage_cache_uid == user_id and (now - self._usage_cache_at) < 30:
            return

        self._usage_fetching = True

        def do_fetch() -> None:
            try:
                calls_today, error = get_today_llm_calls()
                if calls_today is not None:
                    self._usage_cache_calls_today = max(0, int(calls_today))
                    self._usage_cache_uid = user_id
                    self._usage_cache_at = time.time()
                elif error:
                    logger.debug(f"Usage refresh skipped: {error}")
            finally:
                self._usage_fetching = False
                try:
                    self.app.call_from_thread(self._update_account_status)
                except Exception:
                    pass

        threading.Thread(target=do_fetch, daemon=True).start()

    def _update_usage_bar(
        self, bar_widget: Static, pct_widget: Static, calls_today: int, daily_quota: int
    ) -> None:
        """Update the usage progress bar display."""
        if daily_quota <= 0:
            daily_quota = 900  # Fallback to free tier default

        # Calculate percentage (cap at 100%)
        percentage = min(100, int((calls_today / daily_quota) * 100))

        # Update percentage text
        pct_widget.update(f"{percentage}%")

        # Create progress bar (80% of previous 40-char width to preserve button space)
        bar_width = 32
        filled_width = int((percentage / 100) * bar_width)

        # Get current theme to determine colors
        is_dark_mode = str(get_dashboard_state_value("theme", "dark")).strip().lower() != "light"

        # In dark mode: empty=dim, filled=bright
        # In light mode: empty=bright, filled=dim (inverted)
        if is_dark_mode:
            filled_char = "█"
            empty_char = "░"
            filled_style = ""  # Default bright color
            empty_style = "dim"
        else:
            filled_char = "█"
            empty_char = "░"
            filled_style = "dim"  # Dimmed in light mode
            empty_style = ""  # Bright in light mode

        # Build the bar
        if filled_width > 0:
            filled_text = filled_char * filled_width
            filled_part = (
                f"[{filled_style}]{filled_text}[/{filled_style}]" if filled_style else filled_text
            )
        else:
            filled_part = ""

        if (bar_width - filled_width) > 0:
            empty_text = empty_char * (bar_width - filled_width)
            empty_part = (
                f"[{empty_style}]{empty_text}[/{empty_style}]" if empty_style else empty_text
            )
        else:
            empty_part = ""

        bar = filled_part + empty_part

        bar_widget.update(bar)

    def _get_latest_llm_limit_code(self, billing: Optional[dict]) -> Optional[str]:
        """
        Read latest limit code from billing API payload.

        Returns:
            - DAILY_QUOTA_EXCEEDED
            - QPS_LIMIT_EXCEEDED
            - MONTHLY_BUDGET_EXCEEDED
            - RATE_LIMITED
            - USER_BLOCKED
            - None
        """
        if not isinstance(billing, dict):
            return None
        raw_code = billing.get("latest_limit_code")
        if not isinstance(raw_code, str):
            return None
        code = raw_code.strip().upper()
        if code in {
            "DAILY_QUOTA_EXCEEDED",
            "QPS_LIMIT_EXCEEDED",
            "MONTHLY_BUDGET_EXCEEDED",
            "RATE_LIMITED",
            "USER_BLOCKED",
        }:
            return code
        return None

    def _handle_upgrade_pro(self) -> None:
        """Open hosted billing plan selector page."""
        try:
            upgrade_btn = self.query_one("#upgrade-pro-btn", Button)
        except Exception:
            self.app.notify(
                "Upgrade entry is unavailable right now.",
                title="Billing",
                severity="error",
            )
            return

        config = ReAlignConfig.load()
        backend_url = (config.share_backend_url or "https://realign-server.vercel.app").rstrip("/")
        billing_url = f"{backend_url}/billing/upgrade"

        try:
            webbrowser.open(billing_url)
            self.app.notify("Opened billing page", title="Billing")
        except Exception as e:
            self.app.notify(f"Failed to open billing page: {e}", title="Billing", severity="error")

    def _handle_login(self) -> None:
        """Handle login button click - start login flow in background."""
        if self._login_in_progress:
            self.app.notify("Login already in progress...", title="Login")
            return

        self._login_in_progress = True

        # Update UI to show login in progress
        auth_btn = self.query_one("#auth-btn", Button)
        auth_btn.disabled = True
        email_widget = self.query_one("#account-email", Static)
        email_widget.update("[cyan]Opening browser...[/cyan]")

        # Start login flow in background thread
        def do_login():
            try:
                port = find_free_port()
                open_login_page(callback_port=port)

                # Wait for callback (up to 5 minutes)
                cli_token, error = start_callback_server(port, timeout=300)

                if error:
                    self.app.call_from_thread(
                        self.app.notify, f"Login failed: {error}", title="Login", severity="error"
                    )
                    self.app.call_from_thread(self._update_account_status)
                    return

                if not cli_token:
                    self.app.call_from_thread(
                        self.app.notify, "No token received", title="Login", severity="error"
                    )
                    self.app.call_from_thread(self._update_account_status)
                    return

                # Validate token
                credentials = validate_cli_token(cli_token)
                if not credentials:
                    self.app.call_from_thread(
                        self.app.notify, "Invalid token", title="Login", severity="error"
                    )
                    self.app.call_from_thread(self._update_account_status)
                    return

                # Save credentials
                if save_credentials(credentials):
                    # Sync Supabase uid to local config
                    try:
                        config = ReAlignConfig.load()
                        old_uid = config.uid
                        config.uid = credentials.user_id
                        if not config.user_name:
                            config.user_name = credentials.email.split("@")[0]
                        config.save()
                        logger.info(f"Synced Supabase uid to config: {credentials.user_id[:8]}...")

                        # V18: Upsert user info to users table
                        try:
                            from ...db import get_database

                            db = get_database()
                            db.upsert_user(config.uid, config.user_name, credentials.email)
                        except Exception as e:
                            logger.debug(f"Failed to upsert user to users table: {e}")
                    except Exception as e:
                        logger.warning(f"Failed to sync uid to config: {e}")

                    # Start watcher and worker daemons after login
                    try:
                        from ...commands import watcher as watcher_cmd

                        watcher_cmd.watcher_start_command()
                        logger.info("Daemons started after dashboard login")
                    except Exception as e:
                        logger.debug(f"Failed to start daemons after dashboard login: {e}")

                    self.app.call_from_thread(
                        self.app.notify, f"Logged in as {credentials.email}", title="Login"
                    )
                else:
                    self.app.call_from_thread(
                        self.app.notify,
                        "Failed to save credentials",
                        title="Login",
                        severity="error",
                    )

                self.app.call_from_thread(self._update_account_status)

            finally:
                self._login_in_progress = False

        thread = threading.Thread(target=do_login, daemon=True)
        thread.start()

        self.app.notify("Complete login in browser...", title="Login")

    def _handle_logout(self) -> None:
        """Handle logout button click - confirm, then clear credentials and stop daemons."""
        from ..screens.logout_confirm import ExitConfirmScreen

        def _on_confirm(confirmed: bool) -> None:
            if not confirmed:
                return
            credentials = load_credentials()
            email = credentials.email if credentials else "user"

            if clear_credentials():
                open_logout_page()
                self.app.notify(f"Logged out: {email}", title="Account")
                self._update_account_status()
                # Stop watcher/worker daemons in background so they don't run
                # with invalid credentials.
                threading.Thread(target=self._stop_daemons_quiet, daemon=True).start()
            else:
                self.app.notify("Failed to logout", title="Account", severity="error")

        self.app.push_screen(ExitConfirmScreen(mode="logout"), _on_confirm)

    @staticmethod
    def _stop_daemons_quiet() -> None:
        """Stop watcher and worker daemons (best-effort, no console output)."""
        import os
        import signal
        import time
        from pathlib import Path

        for name in ("watcher", "worker"):
            pid_file = Path.home() / ".aline" / ".logs" / f"{name}.pid"
            if not pid_file.exists():
                continue
            try:
                pid = int(pid_file.read_text().strip())
            except Exception:
                continue
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pid_file.unlink(missing_ok=True)
                continue
            except PermissionError:
                pass
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Sent SIGTERM to {name} daemon (PID {pid})")
            except ProcessLookupError:
                pid_file.unlink(missing_ok=True)
                continue
            except Exception:
                continue
            for _ in range(50):
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    logger.info(f"{name} daemon (PID {pid}) stopped")
                    pid_file.unlink(missing_ok=True)
                    break
                except PermissionError:
                    break
            else:
                try:
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(0.5)
                    logger.info(f"{name} daemon (PID {pid}) force killed")
                except Exception:
                    pass
                pid_file.unlink(missing_ok=True)

    def _handle_exit(self) -> None:
        """Handle exit button click - show confirmation then exit."""
        from ..screens.logout_confirm import ExitConfirmScreen

        def _on_confirm(confirmed: bool) -> None:
            if confirmed:
                self.app.exit()

        self.app.push_screen(ExitConfirmScreen(mode="exit"), _on_confirm)

    def _sync_theme_toggle(self) -> None:
        """Sync theme toggle with persisted dashboard preference."""
        try:
            theme = str(get_dashboard_state_value("theme", "dark")).strip().lower()
            self._syncing_controls = True
            try:
                switch = self.query_one("#theme-toggle", Switch)
                switch.value = theme != "light"
            finally:
                self._syncing_controls = False
        except Exception:
            pass

    def _set_theme(self, theme: str) -> None:
        theme_value = "light" if theme.strip().lower() == "light" else "dark"
        set_dashboard_state_value("theme", theme_value)
        self.app.theme = "textual-light" if theme_value == "light" else "textual-dark"
        self.app.notify(f"Theme set to {theme_value}", title="Appearance")

    def _sync_border_resize_toggle(self) -> None:
        """Query tmux state and sync the toggle to match."""
        try:
            # Check if MouseDrag1Border is bound by listing keys
            result = _run_outer_tmux(["list-keys", "-T", "root"], capture=True, timeout_s=0.5)
            output = result.stdout or ""

            # If MouseDrag1Border is in the output, resize is enabled
            is_enabled = "MouseDrag1Border" in output
            self._border_resize_enabled = is_enabled
            try:
                set_dashboard_state_value("tmux_border_resize_enabled", bool(is_enabled))
            except Exception:
                pass

            self._syncing_controls = True
            try:
                switch = self.query_one("#border-resize-toggle", Switch)
                switch.value = bool(is_enabled)
            finally:
                self._syncing_controls = False
        except Exception:
            # If we can't query, assume enabled (default tmux behavior)
            pass

    def _sync_show_archived_toggle(self) -> None:
        try:
            show = bool(get_dashboard_state_value("show_archived_agents", False))
            self._syncing_controls = True
            try:
                switch = self.query_one("#show-archived-toggle", Switch)
                switch.value = bool(show)
            finally:
                self._syncing_controls = False
        except Exception:
            return

    def _toggle_border_resize(self, enabled: bool) -> None:
        """Enable or disable tmux border resize functionality."""
        try:
            if enabled:
                # Re-enable border resize by binding MouseDrag1Border to default resize behavior
                _run_outer_tmux(
                    ["bind", "-n", "MouseDrag1Border", "resize-pane", "-M"], timeout_s=0.5
                )
                self._border_resize_enabled = True
                try:
                    set_dashboard_state_value("tmux_border_resize_enabled", True)
                except Exception:
                    pass
                self.app.notify("Border resize enabled", title="Tmux")
            else:
                # Disable border resize by unbinding MouseDrag1Border
                _run_outer_tmux(["unbind", "-n", "MouseDrag1Border"], timeout_s=0.5)
                self._border_resize_enabled = False
                try:
                    set_dashboard_state_value("tmux_border_resize_enabled", False)
                except Exception:
                    pass
                try:
                    from .. import tmux_manager

                    tmux_manager.remember_outer_dashboard_pane_width_cols()
                    tmux_manager.enforce_outer_dashboard_pane_width()
                except Exception:
                    pass
                self.app.notify("Border resize disabled", title="Tmux")
        except Exception as e:
            self.app.notify(f"Error toggling border resize: {e}", title="Tmux", severity="error")

    def _handle_doctor(self) -> None:
        """Run aline doctor directly in background thread."""
        self.app.notify(f"Running {BRANDING.doctor_label}...", title="Doctor")

        def do_doctor():
            try:
                import contextlib
                import io
                from ...commands.doctor import run_doctor

                # Suppress Rich console output (would corrupt TUI)
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    exit_code = run_doctor(
                        restart_daemons=True,
                        start_if_not_running=True,
                        verbose=False,
                        clear_cache=True,
                        auto_fix=True,
                    )

                if exit_code == 0:
                    self.app.call_from_thread(
                        self.app.notify, "Doctor completed successfully", title="Doctor"
                    )
                else:
                    self.app.call_from_thread(
                        self.app.notify,
                        "Doctor completed with errors",
                        title="Doctor",
                        severity="error",
                    )
            except Exception as e:
                self.app.call_from_thread(
                    self.app.notify, f"Doctor error: {e}", title="Doctor", severity="error"
                )

        thread = threading.Thread(target=do_doctor, daemon=True)
        thread.start()

    def _handle_check_updates(self) -> None:
        """Check for updates in background thread."""
        self.app.notify("Checking for updates...", title="Update")

        def do_check():
            try:
                from importlib.metadata import version
                from ...commands.upgrade import get_latest_pypi_version, compare_versions

                current = version("aline-ai")
                latest = get_latest_pypi_version()
                if latest is None:
                    self.app.call_from_thread(
                        self.app.notify,
                        "Could not check for updates. Please check your network.",
                        title="Update",
                        severity="error",
                    )
                    return

                if compare_versions(current, latest) >= 0:
                    self.app.call_from_thread(
                        self.app.notify,
                        f"Already up to date (v{current})",
                        title="Update",
                    )
                else:
                    self.app.call_from_thread(
                        self.app.notify,
                        f"Update available: v{current} → v{latest}. Run `onecontext update` in terminal.",
                        title="Update",
                    )
            except Exception as e:
                self.app.call_from_thread(
                    self.app.notify,
                    f"Update check failed: {e}",
                    title="Update",
                    severity="error",
                )

        thread = threading.Thread(target=do_check, daemon=True)
        thread.start()

    def refresh_data(self) -> None:
        """Refresh account status (called by app refresh action)."""
        self._update_account_status()
