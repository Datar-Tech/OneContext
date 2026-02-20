#!/usr/bin/env python3
"""
Import shares command - Import shared conversations from URL into local database.

This allows users to download shared conversations and import them with full
Event/Session/Turn structure preserved.
"""

import sys
import os
import hashlib
import json
import base64
import uuid as uuid_lib
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from rich.console import Console
    from rich.progress import Progress
    from rich.prompt import Prompt

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..logging_config import setup_logger
from ..db.base import (
    DatabaseInterface,
    EventRecord,
    SessionRecord,
    TurnRecord,
    MAX_SESSION_NUM,
    AgentSessionLimitExceededError,
)
from ..config import ReAlignConfig
from ..auth import get_auth_headers, is_logged_in

logger = setup_logger("realign.commands.import_shares", "import_shares.log")

if RICH_AVAILABLE:
    console = Console()
else:
    console = None


def _extract_share_owner_identity(
    data: Dict[str, Any],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Best-effort extract of share owner identity from imported payload."""
    owner_name_raw = data.get("username")
    owner_name = str(owner_name_raw).strip() if owner_name_raw is not None else ""
    if not owner_name:
        owner_name = ""
    owner_email_raw = data.get("user_email")
    owner_email = str(owner_email_raw).strip() if owner_email_raw is not None else ""
    if not owner_email:
        owner_email = ""

    event_data = data.get("event")
    owner_uid = ""
    if isinstance(event_data, dict):
        owner_uid = str(
            event_data.get("created_by")
            or event_data.get("uid")
            or event_data.get("creator_id")
            or ""
        ).strip()

    if not owner_uid:
        sessions = data.get("sessions")
        if isinstance(sessions, list):
            creator_uids: set[str] = set()
            for session_data in sessions:
                if not isinstance(session_data, dict):
                    continue
                uid = str(
                    session_data.get("created_by")
                    or session_data.get("uid")
                    or session_data.get("creator_id")
                    or ""
                ).strip()
                if uid:
                    creator_uids.add(uid)
            if len(creator_uids) == 1:
                owner_uid = next(iter(creator_uids))

    return (owner_uid or None, owner_name or None, owner_email or None)


def _upsert_share_owner_user(
    db: DatabaseInterface,
    owner_uid: Optional[str],
    owner_name: Optional[str],
    owner_email: Optional[str] = None,
) -> None:
    """Persist remote share owner identity into local users table when possible."""
    if not owner_uid:
        return
    if not owner_name and not owner_email:
        return
    try:
        db.upsert_user(owner_uid, owner_name, owner_email)
    except Exception as e:
        logger.debug(f"Failed to upsert share owner user info: {e}")


def download_share_data(
    share_url: str,
    password: Optional[str] = None,
    cache_buster: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Download share data from a share URL.

    Extracts download logic (URL parse, auth, fetch) into a reusable function.

    Args:
        share_url: Full share URL (e.g., https://realign-server.vercel.app/share/abc123)
        password: Password for encrypted shares

    Returns:
        {"success": True, "data": conversation_data} on success
        {"success": False, "error": str} on failure
    """
    if not HTTPX_AVAILABLE:
        return {
            "success": False,
            "error": "httpx package not installed. Install with: pip install httpx",
        }

    share_id = extract_share_id(share_url)
    if not share_id:
        return {"success": False, "error": f"Invalid share URL format: {share_url}"}

    logger.info(f"download_share_data: share_id={share_id}")

    config = ReAlignConfig.load()
    backend_url = config.share_backend_url or "https://realign-server.vercel.app"

    # Get share info
    try:
        info_response = httpx.get(f"{backend_url}/api/share/{share_id}/info", timeout=10.0)
        info_response.raise_for_status()
        info = info_response.json()
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch share info: {e}"}

    # Authenticate
    if info.get("requires_password"):
        if not password:
            return {"success": False, "error": "This share requires a password"}
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        headers = {"X-Password-Hash": password_hash}
    else:
        # Phase 0: import requires login for unencrypted shares.
        if not is_logged_in():
            return {
                "success": False,
                "error": "Not logged in. Please run 'aline login' first.",
            }
        headers = get_auth_headers()
        if not headers:
            return {
                "success": False,
                "error": "Login expired. Please run 'aline login' again.",
            }

    # Download export data
    try:
        export_url = f"{backend_url}/api/share/{share_id}/export"
        if cache_buster:
            export_url = f"{export_url}?cache_bust={cache_buster}"

        export_response = httpx.get(export_url, headers=headers, timeout=30.0)
        export_data = export_response.json()

        if export_response.status_code == 413 or export_data.get("needs_chunked_download"):
            total_chunks = export_data.get("total_chunks", 1)
            raw_data = _download_chunks(backend_url, share_id, headers, total_chunks, cache_buster)
            conversation_data = json.loads(raw_data)
            export_data = {
                "success": True,
                "data": conversation_data,
                "metadata": export_data.get("metadata", {}),
            }
        else:
            export_response.raise_for_status()
    except Exception as e:
        return {"success": False, "error": f"Failed to download data: {e}"}

    if not export_data.get("success"):
        return {"success": False, "error": export_data.get("error", "Unknown error")}

    return {"success": True, "data": export_data["data"]}


def import_agent_from_share(
    share_url: str,
    password: Optional[str] = None,
    db: Optional[DatabaseInterface] = None,
) -> Dict[str, Any]:
    """
    Import an agent from a share link.

    Downloads share data, creates agent_info record, imports sessions with
    created_by/shared_by tracking, and links them to the agent.

    Args:
        share_url: Full share URL
        password: Password for encrypted shares
        db: Database instance (auto-created if None)

    Returns:
        {"success": True, "agent_id", "agent_name", "agent_description",
         "sessions_imported", "turns_imported"} on success
        {"success": False, "error": str} on failure
    """
    os.environ["REALIGN_DISABLE_AUTO_SUMMARIES"] = "1"

    result = download_share_data(share_url, password)
    if not result["success"]:
        return result

    conversation_data = result["data"]
    # Extract agent identity from share data
    event_data = conversation_data.get("event", {})
    event_id = event_data.get("event_id", "")

    # Agent ID: strip "agent-" prefix if present, otherwise generate new UUID
    if event_id.startswith("agent-"):
        agent_id = event_id[6:]
    else:
        agent_id = str(uuid_lib.uuid4())

    agent_name = event_data.get("title") or "Imported Agent"
    ui_meta = conversation_data.get("ui_metadata") or {}
    agent_title = (
        (event_data.get("context_title") or "").strip()
        or (ui_meta.get("context_title") or "").strip()
        or ""
    )
    agent_description = event_data.get("description") or ""

    # Set up database
    if db is None:
        from ..db.sqlite_db import SQLiteDatabase

        config = ReAlignConfig.load()
        db_path = Path(config.sqlite_db_path).expanduser()
        db = SQLiteDatabase(db_path=db_path)
    owner_uid, owner_name, owner_email = _extract_share_owner_identity(conversation_data)
    _upsert_share_owner_user(db, owner_uid, owner_name, owner_email)

    # Create agent_info record
    try:
        record = db.get_or_create_agent_info(agent_id, name=agent_name)
        updates = {}
        if getattr(record, "visibility", "visible") != "visible":
            # If the agent was previously archived (invisible), importing implies the user
            # expects to see it again.
            updates["visibility"] = "visible"
        if agent_title:
            updates["title"] = agent_title
        if agent_description:
            updates["description"] = agent_description
        if updates:
            db.update_agent_info(agent_id, **updates)
    except Exception as e:
        return {"success": False, "error": f"Failed to create agent info: {e}"}

    # Import sessions and link to agent
    sessions_data = conversation_data.get("sessions", [])
    total_sessions = 0
    total_turns = 0
    limit_reached = False
    blocked_by_limit = 0

    for index, session_data in enumerate(sessions_data):
        session_id = session_data.get("session_id") or generate_uuid()

        # Use existing import logic (handles created_by/shared_by)
        try:
            # Use a dummy event_id — we don't need event linkage for agent imports
            import_result = import_session_with_turns(
                session_data, event_id or agent_id, share_url, db, force=False
            )
            total_sessions += import_result["sessions"]
            total_turns += import_result["turns"]
        except Exception as e:
            logger.error(f"Failed to import session {session_id}: {e}")
            # Don't continue - still try to link the session if it exists

        # Link session to agent (always attempt, even if import was skipped/failed)
        try:
            db.update_session_agent_id(session_id, agent_id)
        except AgentSessionLimitExceededError:
            limit_reached = True
            blocked_by_limit = len(sessions_data) - index
            logger.warning(
                "Agent import reached per-context session cap: agent_id=%s max=%s blocked=%s",
                agent_id,
                MAX_SESSION_NUM,
                blocked_by_limit,
            )
            break
        except Exception as e:
            logger.error(f"Failed to link session {session_id} to agent: {e}")

    # Extract and store sync metadata (for unencrypted shares)
    sync_meta = conversation_data.get("sync_metadata", {})
    contributor_token = sync_meta.get("contributor_token")
    sync_enabled = False

    if contributor_token and not password:
        sync_enabled = True
        share_id = extract_share_id(share_url)
        try:
            db.update_agent_sync_metadata(
                agent_id,
                share_id=share_id,
                share_url=share_url,
                share_contributor_token=contributor_token,
                # No admin_token for importers
                last_synced_at=datetime.now().isoformat(),
                sync_version=sync_meta.get("sync_version", 0),
            )
        except Exception as e:
            logger.warning(f"Failed to store sync metadata: {e}")
            sync_enabled = False

    return {
        "success": True,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "agent_title": agent_title,
        "agent_description": agent_description,
        "sessions_imported": total_sessions,
        "turns_imported": total_turns,
        "sync_enabled": sync_enabled,
        "share_url": share_url if sync_enabled else None,
        "limit_reached": limit_reached,
        "blocked_by_limit": blocked_by_limit,
        "limit_value": MAX_SESSION_NUM,
    }


def import_share_command(
    share_url: str,
    password: Optional[str] = None,
    force: bool = False,
    db: Optional[DatabaseInterface] = None,
    non_interactive: bool = False,
) -> int:
    """
    Import shared conversation from URL.

    Args:
        share_url: Full share URL (e.g., https://realign-server.vercel.app/share/abc123)
        password: Password for encrypted shares
        force: Re-import existing sessions (skip deduplication)
        db: Database instance (auto-created if None)

    Returns:
        0 on success, 1 on error
    """
    logger.info(f"======== Import share command started: {share_url} ========")

    # Imported shares already contain summaries/metadata; avoid enqueuing background jobs
    # during bulk turn insertion.
    os.environ["REALIGN_DISABLE_AUTO_SUMMARIES"] = "1"

    # Check dependencies
    if not HTTPX_AVAILABLE:
        print("❌ Error: httpx package not installed", file=sys.stderr)
        print("Install it with: pip install httpx", file=sys.stderr)
        return 1

    # 1. Parse share URL to extract share_id
    share_id = extract_share_id(share_url)
    if not share_id:
        print(f"[ERROR] Invalid share URL format: {share_url}", file=sys.stderr)
        logger.error(f"Invalid share URL format: {share_url}")
        return 1

    logger.info(f"Extracted share_id: {share_id}")

    # Use download_share_data helper for non-interactive password case
    # For interactive mode, we still need the prompt flow
    config = ReAlignConfig.load()
    backend_url = config.share_backend_url or "https://realign-server.vercel.app"
    logger.info(f"Backend URL: {backend_url}")

    # 2. Get share info and handle password prompt interactively
    try:
        if console:
            console.print(f"[cyan]Fetching share info from {backend_url}...[/cyan]")
        else:
            print(f"Fetching share info from {backend_url}...")

        info_response = httpx.get(f"{backend_url}/api/share/{share_id}/info", timeout=10.0)
        info_response.raise_for_status()
        info = info_response.json()
        logger.info(f"Share info retrieved: requires_password={info.get('requires_password')}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch share info: {e}", file=sys.stderr)
        logger.error(f"Failed to fetch share info: {e}", exc_info=True)
        return 1

    # 3. Authenticate if needed
    if info.get("requires_password"):
        if not password:
            if non_interactive:
                print(
                    "[ERROR] This share requires a password but none was provided", file=sys.stderr
                )
                return 1
            if console and RICH_AVAILABLE:
                password = Prompt.ask("Enter password", password=True)
            else:
                import getpass

                password = getpass.getpass("Enter password: ")

        password_hash = hashlib.sha256(password.encode()).hexdigest()
        headers = {"X-Password-Hash": password_hash}
        logger.info("Using password authentication")
    else:
        # Phase 0: import requires login for unencrypted shares.
        if not is_logged_in():
            print("❌ Error: Not logged in. Please run 'aline login' first.", file=sys.stderr)
            return 1
        headers = get_auth_headers()
        if not headers:
            print("❌ Error: Login expired. Please run 'aline login' again.", file=sys.stderr)
            return 1

    # 4. Download data from /api/share/[id]/export (with chunked download support)
    try:
        if console:
            console.print(f"[cyan]Downloading data from {backend_url}...[/cyan]")
        else:
            print(f"Downloading data from {backend_url}...")

        # First, try standard download
        export_response = httpx.get(
            f"{backend_url}/api/share/{share_id}/export", headers=headers, timeout=30.0
        )

        export_data = export_response.json()

        # Check if chunked download is needed
        if export_response.status_code == 413 or export_data.get("needs_chunked_download"):
            logger.info("Data too large, switching to chunked download")
            total_chunks = export_data.get("total_chunks", 1)
            data_size = export_data.get("data_size", 0)

            if console:
                console.print(
                    f"[yellow]Large file detected ({data_size / 1024 / 1024:.2f}MB), using chunked download...[/yellow]"
                )
            else:
                print(
                    f"Large file detected ({data_size / 1024 / 1024:.2f}MB), using chunked download..."
                )

            # Download chunks
            raw_data = _download_chunks(backend_url, share_id, headers, total_chunks)

            # Parse the combined data
            conversation_data = json.loads(raw_data)
            export_data = {
                "success": True,
                "data": conversation_data,
                "metadata": export_data.get("metadata", {}),
            }
        else:
            export_response.raise_for_status()

        logger.info(f"Export data downloaded: {len(str(export_data))} bytes")
    except Exception as e:
        print(f"[ERROR] Failed to download data: {e}", file=sys.stderr)
        logger.error(f"Failed to download data: {e}", exc_info=True)
        return 1

    if not export_data.get("success"):
        error_msg = export_data.get("error", "Unknown error")
        print(f"[ERROR] Export failed: {error_msg}", file=sys.stderr)
        logger.error(f"Export failed: {error_msg}")
        return 1

    conversation_data = export_data["data"]
    version = conversation_data.get("version", "1.0")
    logger.info(f"Conversation data version: {version}")

    # 5. Import to local database
    if db is None:
        from ..db.sqlite_db import SQLiteDatabase
        from pathlib import Path

        db_path = Path(config.sqlite_db_path).expanduser()
        db = SQLiteDatabase(db_path=db_path)

    try:
        if version == "2.0":
            return import_v2_data(conversation_data, share_url, db, force, non_interactive)
        else:
            return import_v1_data(conversation_data, share_url, db, force)
    except Exception as e:
        print(f"[ERROR] Import failed: {e}", file=sys.stderr)
        logger.error(f"Import failed: {e}", exc_info=True)
        import traceback

        traceback.print_exc()
        return 1


def import_v2_data(
    data: Dict[str, Any],
    share_url: str,
    db: DatabaseInterface,
    force: bool,
    non_interactive: bool = False,
) -> int:
    """Import v2.0 format data (with Event/Session/Turn structure)."""

    if console:
        console.print("[cyan]Importing v2.0 format data...[/cyan]")
    else:
        print("Importing v2.0 format data...")

    logger.info("Starting v2.0 data import")

    # Load config to get current user's UID for shared_by
    config = ReAlignConfig.load()
    owner_uid, owner_name, owner_email = _extract_share_owner_identity(data)
    _upsert_share_owner_user(db, owner_uid, owner_name, owner_email)

    # 1. Create Event
    event_data = data.get("event", {})
    event_id = event_data.get("event_id")

    # Check if event already exists
    existing_event = db.get_event_by_id(event_id) if event_id else None

    if existing_event and not force:
        if console:
            console.print(f"[yellow]Event '{existing_event.title}' already exists.[/yellow]")
            console.print(
                "[yellow]Use --force to re-import or press Enter to use existing event.[/yellow]"
            )
        else:
            print(f"Event '{existing_event.title}' already exists.")
            print("Use --force to re-import or press Enter to use existing event.")

        if non_interactive:
            use_existing = True
        else:
            try:
                response = input("Use existing event? (Y/n): ").strip().lower()
                use_existing = response in ["", "y", "yes"]
            except (EOFError, KeyboardInterrupt):
                use_existing = True

        if use_existing:
            event_id = existing_event.id
            logger.info(f"Using existing event: {event_id}")
        else:
            logger.info("User declined to use existing event, aborting")
            return 1
    else:
        # Create new event or update existing
        event = EventRecord(
            id=event_id or generate_uuid(),
            title=event_data.get("title", "Imported Event"),
            description=event_data.get("description"),
            event_type=event_data.get("event_type", "imported"),
            status=event_data.get("status", "archived"),
            start_timestamp=parse_datetime(event_data.get("created_at")),
            end_timestamp=parse_datetime(event_data.get("updated_at")),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={"source": "share_import", "share_url": share_url},
            preset_questions=None,
            slack_message=None,
            share_url=share_url,
            commit_hashes=[],
            # V18: user identity (with backward compatibility for old format)
            created_by=event_data.get("created_by")
            or event_data.get("uid")
            or event_data.get("creator_id"),
            shared_by=config.uid,  # Current user is the importer
        )

        # Use sync_events for both create and update (upsert behavior)
        db.sync_events([event])
        if existing_event and force:
            if console:
                console.print(f"[green]✓ Updated event: {event.title}[/green]")
            else:
                print(f"✓ Updated event: {event.title}")
            logger.info(f"Updated event: {event_id}")
        else:
            if console:
                console.print(f"[green]✓ Created event: {event.title}[/green]")
            else:
                print(f"✓ Created event: {event.title}")
            logger.info(f"Created event: {event_id}")

        event_id = event.id

    # 2. Import Sessions and Turns
    sessions_data = data.get("sessions", [])
    imported_sessions = 0
    imported_turns = 0
    skipped_turns = 0

    if console and RICH_AVAILABLE:
        with Progress() as progress:
            task = progress.add_task("[cyan]Importing sessions...", total=len(sessions_data))

            for session_data in sessions_data:
                result = import_session_with_turns(session_data, event_id, share_url, db, force)
                imported_sessions += result["sessions"]
                imported_turns += result["turns"]
                skipped_turns += result["skipped"]
                progress.update(task, advance=1)
    else:
        for idx, session_data in enumerate(sessions_data, 1):
            print(f"Importing session {idx}/{len(sessions_data)}...")
            result = import_session_with_turns(session_data, event_id, share_url, db, force)
            imported_sessions += result["sessions"]
            imported_turns += result["turns"]
            skipped_turns += result["skipped"]

    # 3. Display summary
    if console:
        console.print(f"\n[green]✅ Import completed successfully![/green]")
        console.print(f"[cyan]Event:[/cyan] {event_data.get('title', 'Untitled')}")
        console.print(f"[cyan]Sessions imported:[/cyan] {imported_sessions}")
        console.print(f"[cyan]Turns imported:[/cyan] {imported_turns}")
        if skipped_turns > 0:
            console.print(f"[yellow]Turns skipped (duplicates):[/yellow] {skipped_turns}")
    else:
        print(f"\n✅ Import completed successfully!")
        print(f"Event: {event_data.get('title', 'Untitled')}")
        print(f"Sessions imported: {imported_sessions}")
        print(f"Turns imported: {imported_turns}")
        if skipped_turns > 0:
            print(f"Turns skipped (duplicates): {skipped_turns}")

    logger.info(
        f"Import completed: {imported_sessions} sessions, {imported_turns} turns, {skipped_turns} skipped"
    )

    return 0


def import_session_with_turns(
    session_data: Dict[str, Any], event_id: str, share_url: str, db: DatabaseInterface, force: bool
) -> Dict[str, int]:
    """
    Import a single session with all its turns.

    Returns:
        Dict with counts: {'sessions': int, 'turns': int, 'skipped': int}
    """
    # Load config to get current user's UID for shared_by
    config = ReAlignConfig.load()

    session_id = session_data.get("session_id")
    imported_sessions = 0
    imported_turns = 0
    skipped_turns = 0

    # Check if session exists
    existing_session = db.get_session_by_id(session_id) if session_id else None

    if existing_session and not force:
        logger.info(f"Session {session_id} already exists, checking for new turns...")
    else:
        # Create new session
        session = SessionRecord(
            id=session_id or generate_uuid(),
            session_file_path=Path(""),  # Not applicable for imported sessions
            session_type=session_data.get("session_type", "imported"),
            workspace_path=session_data.get("workspace_path"),
            started_at=parse_datetime(session_data.get("started_at")) or datetime.now(),
            last_activity_at=parse_datetime(session_data.get("last_activity_at")) or datetime.now(),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={"source": "share_import", "share_url": share_url},
            session_title=session_data.get("session_title"),
            session_summary=session_data.get("session_summary"),
            summary_updated_at=None,
            summary_status="completed",
            summary_locked_until=None,
            summary_error=None,
            # V18: user identity (with backward compatibility for old format)
            created_by=session_data.get("created_by")
            or session_data.get("uid")
            or session_data.get("creator_id"),
            shared_by=config.uid,  # Current user is the importer
        )

        if existing_session and force:
            # For force update, we need to update the session
            # SQLiteDatabase doesn't have update_session, so we'll use get_or_create which does upsert
            logger.info(f"Force flag set, will re-create session: {session_id}")
        else:
            db.get_or_create_session(
                session_id=session.id,
                session_file_path=session.session_file_path,
                session_type=session.session_type,
                started_at=session.started_at,
                workspace_path=session.workspace_path,
                metadata=session.metadata,
            )
            logger.info(f"Created session: {session_id}")

        # Update session with title, summary, and creator info (not supported by get_or_create_session)
        with db._get_connection() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET session_title = ?,
                    session_summary = ?,
                    summary_updated_at = ?,
                    summary_status = ?,
                    summary_locked_until = ?,
                    summary_error = ?,
                    created_by = ?,
                    shared_by = ?
                WHERE id = ?
            """,
                (
                    session.session_title,
                    session.session_summary,
                    session.summary_updated_at,
                    session.summary_status,
                    session.summary_locked_until,
                    session.summary_error,
                    session.created_by,
                    session.shared_by,
                    session.id,
                ),
            )
            conn.commit()

        imported_sessions += 1

    # Import turns
    turns_data = session_data.get("turns", [])
    for turn_data in turns_data:
        content_hash = turn_data.get("content_hash")

        # Check for duplicates using content_hash
        if content_hash and not force:
            existing_turn = db.get_turn_by_hash(session_id, content_hash)
            if existing_turn:
                skipped_turns += 1
                logger.debug(f"Skipped duplicate turn: {turn_data.get('turn_id')}")
                continue

        # Create turn
        turn = TurnRecord(
            id=turn_data.get("turn_id") or generate_uuid(),
            session_id=session_id,
            turn_number=turn_data.get("turn_number", 0),
            user_message=turn_data.get("user_message"),
            assistant_summary=turn_data.get("assistant_summary"),
            turn_status="completed",
            llm_title=turn_data.get("llm_title", ""),
            llm_description=turn_data.get("llm_description"),
            model_name=turn_data.get("model_name"),
            if_last_task="no",
            satisfaction="unknown",
            content_hash=content_hash or generate_content_hash(turn_data.get("messages", [])),
            timestamp=parse_datetime(turn_data.get("timestamp")) or datetime.now(),
            created_at=datetime.now(),
            git_commit_hash=turn_data.get("git_commit_hash"),
        )

        # Store turn content (JSONL)
        messages = turn_data.get("messages", [])
        jsonl_content = "\n".join([json.dumps(msg) for msg in messages])

        db.create_turn(turn, jsonl_content)
        logger.debug(f"Created turn: {turn.id}")

        imported_turns += 1

    # Link session to event
    db.link_session_to_event(event_id, session_id)

    return {"sessions": imported_sessions, "turns": imported_turns, "skipped": skipped_turns}


def import_v1_data(data: Dict[str, Any], share_url: str, db: DatabaseInterface, force: bool) -> int:
    """Import v1.0 format data (flat messages without Event/Turn structure)."""

    if console:
        console.print("[yellow]⚠ Importing v1.0 format data (limited metadata)[/yellow]")
    else:
        print("⚠ Importing v1.0 format data (limited metadata)")

    logger.info("Starting v1.0 data import (legacy format)")

    # Create a generic imported event
    event = EventRecord(
        id=generate_uuid(),
        title=f"Imported from {share_url[:50]}...",
        description="Imported from share (v1.0 format)",
        event_type="imported",
        status="archived",
        start_timestamp=datetime.now(),
        end_timestamp=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={"source": "share_import", "share_url": share_url, "version": "1.0"},
        preset_questions=None,
        slack_message=None,
        share_url=share_url,
        commit_hashes=[],
    )
    db.sync_events([event])
    logger.info(f"Created legacy event: {event.id}")

    # Import sessions (without turn structure)
    sessions_data = data.get("sessions", [])
    for session_data in sessions_data:
        session_id = session_data.get("session_id", generate_uuid())
        messages = session_data.get("messages", [])

        # Create session
        session = SessionRecord(
            id=session_id,
            session_file_path=Path(""),
            session_type="imported",
            workspace_path=None,
            started_at=datetime.now(),
            last_activity_at=datetime.now(),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={"source": "share_import", "version": "1.0"},
            session_title=None,
            session_summary=None,
            summary_updated_at=None,
            summary_status="idle",
            summary_locked_until=None,
            summary_error=None,
        )
        db.get_or_create_session(
            session_id=session.id,
            session_file_path=session.session_file_path,
            session_type=session.session_type,
            started_at=session.started_at,
            workspace_path=session.workspace_path,
            metadata=session.metadata,
        )

        # Create a single turn with all messages
        turn = TurnRecord(
            id=generate_uuid(),
            session_id=session_id,
            turn_number=1,
            user_message=None,
            assistant_summary=None,
            turn_status="completed",
            llm_title="Imported Messages",
            llm_description=None,
            model_name=None,
            if_last_task="no",
            satisfaction="unknown",
            content_hash=generate_content_hash(messages),
            timestamp=datetime.now(),
            created_at=datetime.now(),
            git_commit_hash=None,
        )

        # Store content
        jsonl_content = "\n".join([json.dumps(msg) for msg in messages])
        db.create_turn(turn, jsonl_content)

        # Link to event
        db.link_session_to_event(event.id, session_id)

    if console:
        console.print(f"[green]✓ Imported {len(sessions_data)} sessions in legacy format[/green]")
    else:
        print(f"✓ Imported {len(sessions_data)} sessions in legacy format")

    logger.info(f"Legacy import completed: {len(sessions_data)} sessions")

    return 0


# Helper functions


def extract_share_id(share_url: str) -> Optional[str]:
    """Extract share ID from URL."""
    import re

    match = re.search(r"/share/([a-zA-Z0-9_-]+)", share_url)
    return match.group(1) if match else None


def parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO format datetime string."""
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception as e:
        logger.warning(f"Failed to parse datetime: {dt_str}, error: {e}")
        return None


def generate_uuid() -> str:
    """Generate UUID."""
    return str(uuid_lib.uuid4())


def generate_content_hash(messages: List[Dict]) -> str:
    """Generate content hash from messages."""
    content = json.dumps(messages, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _download_chunks(
    backend_url: str,
    share_id: str,
    headers: Dict[str, str],
    total_chunks: int,
    cache_buster: Optional[str] = None,
) -> str:
    """
    Download data in chunks and combine them.

    Args:
        backend_url: Backend URL
        share_id: Share ID
        headers: Authentication headers
        total_chunks: Total number of chunks to download

    Returns:
        Combined raw data string
    """
    chunks = []

    if console and RICH_AVAILABLE:
        with Progress() as progress:
            task = progress.add_task("[cyan]Downloading chunks...", total=total_chunks)

            for i in range(total_chunks):
                url = f"{backend_url}/api/share/{share_id}/export?chunk={i}"
                if cache_buster:
                    url = f"{url}&cache_bust={cache_buster}"
                chunk_response = httpx.get(
                    url,
                    headers=headers,
                    timeout=60.0,
                )
                chunk_response.raise_for_status()
                chunk_data = chunk_response.json()

                if not chunk_data.get("success"):
                    raise RuntimeError(f"Failed to download chunk {i}: {chunk_data.get('error')}")

                # Decode base64 chunk
                encoded_chunk = chunk_data.get("chunk_data", "")
                decoded_chunk = base64.b64decode(encoded_chunk).decode("utf-8")
                chunks.append(decoded_chunk)

                progress.update(task, advance=1)
                logger.debug(f"Downloaded chunk {i + 1}/{total_chunks}")
    else:
        for i in range(total_chunks):
            print(f"Downloading chunk {i + 1}/{total_chunks}...")

            url = f"{backend_url}/api/share/{share_id}/export?chunk={i}"
            if cache_buster:
                url = f"{url}&cache_bust={cache_buster}"
            chunk_response = httpx.get(
                url,
                headers=headers,
                timeout=60.0,
            )
            chunk_response.raise_for_status()
            chunk_data = chunk_response.json()

            if not chunk_data.get("success"):
                raise RuntimeError(f"Failed to download chunk {i}: {chunk_data.get('error')}")

            # Decode base64 chunk
            encoded_chunk = chunk_data.get("chunk_data", "")
            decoded_chunk = base64.b64decode(encoded_chunk).decode("utf-8")
            chunks.append(decoded_chunk)

            logger.debug(f"Downloaded chunk {i + 1}/{total_chunks}")

    # Combine all chunks
    combined_data = "".join(chunks)
    logger.info(f"Combined {total_chunks} chunks into {len(combined_data)} bytes")

    return combined_data
