#!/usr/bin/env python3
"""
Authentication commands for Aline CLI.

Commands:
- aline login   - Login via web browser
- aline logout  - Clear local credentials
- aline whoami  - Show current login status
"""

import sys
from datetime import datetime, timezone
from typing import Optional

try:
    from rich.console import Console
    from rich.prompt import Prompt

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..auth import (
    is_logged_in,
    get_current_user,
    load_credentials,
    save_credentials,
    clear_credentials,
    open_login_page,
    open_logout_page,
    validate_cli_token,
    find_free_port,
    start_callback_server,
    get_billing_status,
    get_today_llm_calls,
    HTTPX_AVAILABLE,
)
from ..config import ReAlignConfig
from ..logging_config import setup_logger

logger = setup_logger("realign.commands.auth", "auth.log")

if RICH_AVAILABLE:
    console = Console()
else:
    console = None


def login_command() -> int:
    """
    Login to Aline via web browser.

    Opens the web login page with automatic callback - no manual token copy needed.

    Returns:
        0 on success, 1 on error
    """
    # Check dependencies
    if not HTTPX_AVAILABLE:
        print("Error: httpx package not installed", file=sys.stderr)
        print("Install it with: pip install httpx", file=sys.stderr)
        return 1

    # Check if already logged in
    credentials = load_credentials()
    if credentials and is_logged_in():
        if console:
            console.print(f"[yellow]Already logged in as {credentials.email}[/yellow]")
            console.print("Run 'aline logout' first if you want to switch accounts.")
        else:
            print(f"Already logged in as {credentials.email}")
            print("Run 'aline logout' first if you want to switch accounts.")
        return 0

    # Start local callback server
    port = find_free_port()

    if console:
        console.print("[cyan]Opening browser for login...[/cyan]")
        console.print("[dim]Waiting for authentication...[/dim]\n")
    else:
        print("Opening browser for login...")
        print("Waiting for authentication...\n")

    # Open browser with callback URL
    login_url = open_login_page(callback_port=port)

    if console:
        console.print(f"[dim]If browser doesn't open, visit:[/dim]")
        console.print(f"[link={login_url}]{login_url}[/link]\n")
    else:
        print(f"If browser doesn't open, visit:")
        print(f"{login_url}\n")

    # Wait for callback with token
    cli_token, error = start_callback_server(port, timeout=300)

    if error:
        if console:
            console.print(f"[red]Error: {error}[/red]")
        else:
            print(f"Error: {error}", file=sys.stderr)
        return 1

    if not cli_token:
        if console:
            console.print("[red]Error: No token received[/red]")
            console.print("Please try again with 'aline login'")
        else:
            print("Error: No token received", file=sys.stderr)
            print("Please try again with 'aline login'")
        return 1

    # Validate token
    if console:
        console.print("[cyan]Validating token...[/cyan]")
    else:
        print("Validating token...")

    credentials = validate_cli_token(cli_token)

    if not credentials:
        if console:
            console.print("[red]Error: Invalid or expired token[/red]")
            console.print("Please try again with 'aline login'")
        else:
            print("Error: Invalid or expired token", file=sys.stderr)
            print("Please try again with 'aline login'")
        return 1

    # Save credentials
    if not save_credentials(credentials):
        if console:
            console.print("[red]Error: Failed to save credentials[/red]")
        else:
            print("Error: Failed to save credentials", file=sys.stderr)
        return 1

    # Sync Supabase uid to local config
    # This ensures all new Events/Sessions/Turns use the same uid as shares
    try:
        config = ReAlignConfig.load()
        old_uid = config.uid
        config.uid = credentials.user_id
        # Use email as user_name if not already set
        if not config.user_name:
            config.user_name = credentials.email.split("@")[0]  # Use email prefix as username
        config.save()
        logger.info(
            f"Synced Supabase uid to config: {credentials.user_id[:8]}... (was: {old_uid[:8] if old_uid else 'not set'}...)"
        )

        # V18: Upsert user info to users table
        try:
            from ..db import get_database

            db = get_database()
            db.upsert_user(config.uid, config.user_name, credentials.email)
        except Exception as e:
            logger.debug(f"Failed to upsert user to users table: {e}")
    except Exception as e:
        # Non-fatal: continue even if config sync fails
        logger.warning(f"Failed to sync uid to config: {e}")

    # Success
    if console:
        console.print(f"\n[green]Login successful![/green]")
        console.print(f"Logged in as: [bold]{credentials.email}[/bold]")
        if credentials.provider and credentials.provider != "email":
            console.print(f"Provider: {credentials.provider}")
        console.print(f"[dim]User ID synced to local config[/dim]")
    else:
        print(f"\nLogin successful!")
        print(f"Logged in as: {credentials.email}")
        if credentials.provider and credentials.provider != "email":
            print(f"Provider: {credentials.provider}")
        print("User ID synced to local config")

    logger.info(f"Login successful for {credentials.email}")

    # Start watcher and worker daemons after login
    _start_daemons()

    return 0


def logout_command() -> int:
    """
    Logout from Aline and clear local credentials.

    Also stops watcher and worker daemons since they require authentication.

    Returns:
        0 on success, 1 on error
    """
    credentials = load_credentials()

    if not credentials:
        if console:
            console.print("[yellow]Not currently logged in.[/yellow]")
        else:
            print("Not currently logged in.")
        return 0

    email = credentials.email

    # Stop watcher and worker daemons before logout
    if console:
        console.print("[dim]Stopping daemons...[/dim]")
    else:
        print("Stopping daemons...")

    _stop_daemons()

    if not clear_credentials():
        if console:
            console.print("[red]Error: Failed to clear credentials[/red]")
        else:
            print("Error: Failed to clear credentials", file=sys.stderr)
        return 1

    # Open browser to sign out from web session
    if console:
        console.print("[dim]Signing out from web session...[/dim]")
    else:
        print("Signing out from web session...")

    open_logout_page()

    if console:
        console.print(f"[green]Logged out successfully.[/green]")
        console.print(f"Cleared credentials for: {email}")
    else:
        print("Logged out successfully.")
        print(f"Cleared credentials for: {email}")

    logger.info(f"Logout successful for {email}")
    return 0


def _start_daemons() -> None:
    """Start watcher and worker daemons if not already running."""
    try:
        from . import watcher as watcher_cmd

        exit_code = watcher_cmd.watcher_start_command()
        if exit_code == 0:
            if console:
                console.print("[dim]Watcher and worker daemons started.[/dim]")
            logger.info("Daemons started after login")
        else:
            logger.debug(f"watcher_start_command returned {exit_code}")
    except Exception as e:
        logger.debug(f"Failed to start daemons after login: {e}")


def _stop_daemons() -> None:
    """Stop watcher and worker daemons."""
    try:
        from . import watcher as watcher_cmd

        watcher_cmd.watcher_stop_command()
    except Exception:
        pass
    try:
        from . import worker as worker_cmd

        worker_cmd.worker_stop_command()
    except Exception:
        pass


def whoami_command() -> int:
    """
    Display current login status.

    Returns:
        0 if logged in, 1 if not logged in
    """
    credentials = get_current_user()

    if not credentials:
        if console:
            console.print("[yellow]Not logged in.[/yellow]")
            console.print("Run 'aline login' to authenticate.")
        else:
            print("Not logged in.")
            print("Run 'aline login' to authenticate.")
        return 1

    if console:
        console.print(f"[green]Logged in as:[/green] [bold]{credentials.email}[/bold]")
        console.print(f"[dim]User ID:[/dim] {credentials.user_id}")
        if credentials.provider:
            console.print(f"[dim]Provider:[/dim] {credentials.provider}")
        console.print(
            f"[dim]Token expires:[/dim] {credentials.expires_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
    else:
        print(f"Logged in as: {credentials.email}")
        print(f"User ID: {credentials.user_id}")
        if credentials.provider:
            print(f"Provider: {credentials.provider}")
        print(f"Token expires: {credentials.expires_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    billing, billing_error = get_billing_status()
    calls_today, usage_error = get_today_llm_calls()

    plan_label = "Unknown"
    usage_label = "Unknown"
    renewal_label = "Unknown"

    if billing and isinstance(billing, dict):
        is_pro = bool(billing.get("is_pro"))
        tier = str(billing.get("plan_tier") or "").strip().lower()
        mode = str(billing.get("billing_mode") or "").strip().lower()
        status = str(billing.get("status") or "").strip().lower()
        cancel_at_period_end = bool(billing.get("cancel_at_period_end"))
        pro_until_raw = billing.get("pro_until")
        daily_quota_raw = billing.get("daily_quota")

        if is_pro:
            if tier == "pro_plus":
                tier_label = "Pro+"
            else:
                tier_label = "Pro"
            plan_parts = [mode or "subscription", status or "active"]
            if mode == "subscription" and cancel_at_period_end:
                plan_parts.append("cancel_at_period_end")
            plan_label = f"{tier_label} ({', '.join(plan_parts)})"
        else:
            plan_label = "Free"

        daily_quota: Optional[int] = None
        if daily_quota_raw is not None:
            try:
                daily_quota = int(daily_quota_raw)
            except Exception:
                daily_quota = None

        if calls_today is not None:
            if tier == "pro_plus":
                usage_label = "N/A - no daily quota limit"
            elif daily_quota and daily_quota > 0:
                percentage = (calls_today / daily_quota) * 100
                usage_label = f"{percentage:.1f}%"
            else:
                usage_label = "N/A - no daily quota limit"
        elif usage_error:
            usage_label = f"Unavailable - {usage_error}"

        if isinstance(pro_until_raw, str) and pro_until_raw:
            try:
                dt = datetime.fromisoformat(pro_until_raw.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt_utc = dt.astimezone(timezone.utc)
                when = dt_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
                if mode == "subscription" and cancel_at_period_end:
                    renewal_label = f"Cancels at {when}"
                elif mode == "subscription" and status in ("active", "past_due", "trialing"):
                    renewal_label = f"Renews at {when}"
                else:
                    renewal_label = f"Expires at {when}"
            except Exception:
                renewal_label = f"Unknown ({pro_until_raw})"
        else:
            renewal_label = "Not applicable"
    elif billing_error:
        plan_label = f"Unavailable ({billing_error})"
        if usage_error:
            usage_label = f"Unavailable ({usage_error})"
        renewal_label = "Unavailable"

    if console:
        console.print(f"[dim]Plan:[/dim] {plan_label}")
        console.print(f"[dim]Today's usage:[/dim] {usage_label}")
        console.print(f"[dim]Renewal/expiry:[/dim] {renewal_label}")
    else:
        print(f"Plan: {plan_label}")
        print(f"Today's usage: {usage_label}")
        print(f"Renewal/expiry: {renewal_label}")

    return 0
