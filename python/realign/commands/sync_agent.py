#!/usr/bin/env python3
"""
Sync agent command - Bidirectional sync for shared agents.

Pull remote sessions, merge locally (union of sessions, dedup by content_hash),
push merged result back. Uses optimistic locking via sync_version.

Sync works with unencrypted shares only.
"""

import json
import os
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Callable

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from ..logging_config import setup_logger

logger = setup_logger("realign.commands.sync_agent", "sync_agent.log")

MAX_SYNC_RETRIES = 3


def _ensure_aware_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_iso_datetime_to_utc(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return _ensure_aware_utc(dt)
    except Exception:
        return None


def _extract_httpx_conflict_current_version(err: Exception) -> Optional[int]:
    if not HTTPX_AVAILABLE:
        return None
    if not isinstance(err, httpx.HTTPStatusError):
        return None
    if err.response is None or err.response.status_code != 409:
        return None
    try:
        payload = err.response.json()
    except Exception:
        return None
    current = payload.get("current_version")
    try:
        return int(current)
    except Exception:
        return None


def sync_agent_command(
    agent_id: str,
    backend_url: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Sync an agent's sessions with the remote share.

    Algorithm:
    1. Load local state (agent_info, sessions, content hashes)
    2. Pull remote state (full download via export endpoint)
    3. Merge: union of sessions deduped by content_hash, last-write-wins for name/desc
    4. Push merged state via PUT with optimistic locking
    5. Update local sync metadata

    Args:
        agent_id: The agent_info ID to sync
        backend_url: Backend server URL (uses config default if None)
        progress_callback: Optional callback for progress updates

    Returns:
        {"success": True, "sessions_pulled": N, "sessions_pushed": N, ...} on success
        {"success": False, "error": str} on failure
    """

    def _progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    if not HTTPX_AVAILABLE:
        return {"success": False, "error": "httpx package not installed"}

    # Phase 0: remote pull/push requires login.
    from ..auth import get_auth_headers, is_logged_in

    if not is_logged_in():
        return {"success": False, "error": "Not logged in. Please run 'aline login' first."}
    if not get_auth_headers():
        return {"success": False, "error": "Login expired. Please run 'aline login' again."}

    # Get backend URL
    if backend_url is None:
        from ..config import ReAlignConfig

        config = ReAlignConfig.load()
        backend_url = config.share_backend_url or "https://realign-server.vercel.app"

    # Get database
    from ..db import AgentSessionLimitExceededError, MAX_SESSION_NUM, get_database

    db = get_database()

    # 1. Load local state
    _progress("Loading local agent data...")

    # Support prefix matching for agent_id
    agent_info = db.get_agent_info(agent_id)
    if not agent_info:
        # Try prefix match
        all_agents = db.list_agent_info()
        matches = [a for a in all_agents if a.id.startswith(agent_id)]
        if len(matches) == 1:
            agent_info = matches[0]
            agent_id = agent_info.id
        elif len(matches) > 1:
            return {
                "success": False,
                "error": f"Ambiguous agent_id prefix '{agent_id}' matches {len(matches)} agents",
            }
        else:
            return {"success": False, "error": f"Agent not found: {agent_id}"}

    if not agent_info.share_id or not agent_info.share_url:
        return {"success": False, "error": "Agent has no share metadata (not shared yet)"}

    local_saved_slack_message = getattr(agent_info, "slack_message", None)

    token = agent_info.share_admin_token or agent_info.share_contributor_token
    if not token:
        return {
            "success": False,
            "error": "No token available for sync (need admin or contributor token)",
        }

    share_id = agent_info.share_id
    local_sync_version = agent_info.sync_version or 0
    linked_session_ids = set(db.get_session_ids_by_agent_id(agent_id))

    # Repair: backfill agent-session links using windowlink hints.
    # This handles cases where session rows exist but the mapping link is missing.
    try:
        conn = db._get_connection()
        candidates = conn.execute(
            """SELECT DISTINCT w.session_id
               FROM windowlink w
               JOIN sessions s ON s.id = w.session_id
               WHERE (w.agent_id = ? OR w.source = ?)
                 AND w.session_id IS NOT NULL""",
            (agent_id, f"agent:{agent_id}"),
        ).fetchall()
        repaired_count = 0
        for row in candidates:
            sid = row[0]
            if not sid or sid in linked_session_ids:
                continue
            if db.is_session_linked_to_agent(agent_id, sid):
                linked_session_ids.add(sid)
                continue
            try:
                if db.link_session_to_agent(agent_id, sid):
                    linked_session_ids.add(sid)
                    repaired_count += 1
                    logger.info(f"Sync repair: linked session {sid} to agent {agent_id}")
            except AgentSessionLimitExceededError:
                limit_msg = (
                    f"当前“Context”关联的session已经达到上限（{MAX_SESSION_NUM}），"
                    "无法继续同步更多session。"
                )
                return {"success": False, "error": limit_msg}
        if repaired_count:
            _progress(f"Repaired {repaired_count} unlinked session(s)")
    except Exception as e:
        logger.warning(f"Session repair step failed (non-fatal): {e}")

    local_sessions = db.get_sessions_by_agent_id(agent_id)
    local_content_hashes = db.get_agent_content_hashes(agent_id)

    logger.info(
        f"Sync: agent={agent_id}, share={share_id}, "
        f"local_sessions={len(local_sessions)}, local_hashes={len(local_content_hashes)}"
    )

    # 2. Pull remote state
    _progress("Pulling remote data...")

    remote_data = _pull_remote(backend_url, share_id)
    if not remote_data.get("success"):
        return {"success": False, "error": f"Failed to pull remote: {remote_data.get('error')}"}

    conversation_data = remote_data["data"]
    remote_sync_meta = conversation_data.get("sync_metadata", {})
    remote_sync_version = remote_sync_meta.get("sync_version", 0)

    remote_sessions_data = conversation_data.get("sessions", [])
    remote_event = conversation_data.get("event", {})
    remote_ui_metadata = conversation_data.get("ui_metadata", {}) or {}
    remote_agent_sections = (
        remote_ui_metadata.get("agent_sections") if isinstance(remote_ui_metadata, dict) else None
    )
    if not isinstance(remote_agent_sections, list):
        remote_agent_sections = None

    # If local doesn't have saved UI metadata yet (e.g., imported agent), backfill from remote.
    try:
        remote_slack_message = remote_ui_metadata.get("slack_message")

        needs_backfill = False
        backfill_kwargs = {}

        if not (isinstance(local_saved_slack_message, str) and local_saved_slack_message.strip()):
            if isinstance(remote_slack_message, str) and remote_slack_message.strip():
                backfill_kwargs["slack_message"] = remote_slack_message
                local_saved_slack_message = remote_slack_message
                needs_backfill = True

        if needs_backfill:
            db.update_agent_sync_metadata(agent_id, **backfill_kwargs)
            agent_info = db.get_agent_info(agent_id) or agent_info
    except Exception:
        pass

    # 3. Merge
    _progress("Merging sessions...")

    # Collect remote content hashes
    remote_content_hashes = set()
    for session_data in remote_sessions_data:
        for turn_data in session_data.get("turns", []):
            h = turn_data.get("content_hash")
            if h:
                remote_content_hashes.add(h)

    # Import new remote sessions/turns locally
    sessions_pulled = 0
    from .import_shares import import_session_with_turns

    for session_data in remote_sessions_data:
        session_id = session_data.get("session_id", "")
        session_turns = session_data.get("turns", [])

        # Check if any turns in this session are new to THIS AGENT (not globally)
        new_turns = [
            t
            for t in session_turns
            if t.get("content_hash") and t["content_hash"] not in local_content_hashes
        ]

        # Check if session exists and whether it's linked to this agent
        existing_session = db.get_session_by_id(session_id)
        session_is_new = existing_session is None
        session_needs_linking = bool(existing_session and session_id not in linked_session_ids)

        # Import if: new turns, or session is new, or session needs linking
        if not new_turns and not session_is_new and not session_needs_linking:
            continue

        # Import the session (import_session_with_turns handles dedup by content_hash)
        should_count = session_is_new or session_needs_linking
        try:
            # Suppress auto-summaries during sync
            os.environ["REALIGN_DISABLE_AUTO_SUMMARIES"] = "1"
            import_result = import_session_with_turns(
                session_data, f"agent-{agent_id}", agent_info.share_url, db, force=False
            )
            # Count as pulled if: created new session/turns, or session was new/needed linking
            if (
                import_result.get("sessions", 0) > 0
                or import_result.get("turns", 0) > 0
                or should_count
            ):
                sessions_pulled += 1
        except Exception as e:
            logger.error(f"Failed to import remote session {session_id}: {e}")
            # Still count if we intended to import this session
            if should_count:
                sessions_pulled += 1

        # Always link session to agent (even if import was skipped)
        try:
            if db.link_session_to_agent(agent_id, session_id):
                linked_session_ids.add(session_id)
        except AgentSessionLimitExceededError:
            limit_msg = (
                f"当前“Context”关联的session已经达到上限（{MAX_SESSION_NUM}），"
                "无法继续同步更多session。"
            )
            return {"success": False, "error": limit_msg}
        except Exception as e:
            logger.error(f"Failed to link session {session_id} to agent: {e}")

    # Merge name/description: last-write-wins by updated_at
    description_updated = False
    remote_updated_at = remote_event.get("updated_at")
    if remote_updated_at:
        try:
            remote_dt = _parse_iso_datetime_to_utc(remote_updated_at)
            local_dt = (
                _ensure_aware_utc(agent_info.updated_at)
                if isinstance(agent_info.updated_at, datetime)
                else None
            )
            if remote_dt and local_dt and remote_dt > local_dt:
                remote_name = remote_event.get("title")
                remote_desc = remote_event.get("description")
                remote_context_title = remote_event.get("context_title")
                if isinstance(remote_context_title, str) and not remote_context_title.strip():
                    remote_context_title = None
                updates = {}
                if remote_name and remote_name != agent_info.name:
                    updates["name"] = remote_name
                if remote_context_title is not None and remote_context_title != agent_info.title:
                    updates["title"] = remote_context_title
                if remote_desc is not None and remote_desc != agent_info.description:
                    updates["description"] = remote_desc
                if updates:
                    db.update_agent_info(agent_id, **updates)
                    description_updated = True
                    agent_info = db.get_agent_info(agent_id)
        except Exception as e:
            logger.warning(f"Failed to compare timestamps for name/desc merge: {e}")

    # 4. Build merged data and push
    _progress("Pushing merged data...")

    # Reload local state after merge
    local_sessions = db.get_sessions_by_agent_id(agent_id)
    local_content_hashes = db.get_agent_content_hashes(agent_id)

    # Count sessions pushed (local sessions with turns not in remote)
    sessions_pushed = 0
    for session in local_sessions:
        turns = db.get_turns_for_session(session.id)
        new_local_turns = [t for t in turns if t.content_hash not in remote_content_hashes]
        if new_local_turns:
            sessions_pushed += 1

    # Skip push if there's nothing new to send.
    # This avoids re-uploading large, unchanged payloads (which can hit serverless limits and show up as 403/413).
    needs_push_metadata = False
    try:
        remote_title = remote_event.get("title")
        remote_context_title = remote_event.get("context_title")
        if not isinstance(remote_context_title, str):
            remote_context_title = ""
        elif not remote_context_title.strip():
            remote_context_title = ""
        remote_desc = remote_event.get("description")

        local_title = agent_info.name
        local_context_title = agent_info.title or ""
        local_desc = agent_info.description

        has_metadata_diff = (
            (remote_title != local_title)
            or (remote_context_title != local_context_title)
            or (remote_desc != local_desc)
        )

        remote_sections = (
            remote_ui_metadata.get("agent_sections")
            if isinstance(remote_ui_metadata, dict)
            else None
        )
        remote_has_sections = isinstance(remote_sections, list) and any(
            isinstance(section, dict)
            and isinstance(section.get("heading"), str)
            and section.get("heading").strip()
            for section in remote_sections
        )

        # Backfill older shares that were created before agent_sections was persisted.
        if not remote_has_sections:
            needs_push_metadata = True

        if has_metadata_diff and not description_updated:
            remote_updated_at = remote_event.get("updated_at")
            remote_dt = _parse_iso_datetime_to_utc(remote_updated_at)

            local_dt = getattr(agent_info, "updated_at", None)
            if isinstance(local_dt, datetime):
                local_dt = _ensure_aware_utc(local_dt)

            # If remote has no timestamp, assume local should win. Otherwise, push only if local is newer.
            if remote_dt is None or (local_dt and remote_dt and local_dt > remote_dt):
                needs_push_metadata = True
    except Exception as e:
        logger.warning(f"Failed to compute metadata push necessity (non-fatal): {e}")

    if sessions_pushed == 0 and not needs_push_metadata:
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            db.update_agent_sync_metadata(
                agent_id,
                last_synced_at=now_iso,
                sync_version=remote_sync_version,
            )
        except Exception as e:
            logger.warning(f"Failed to update local sync metadata after no-op sync: {e}")

        _progress("No changes to push.")
        _progress("Sync complete!")

        return {
            "success": True,
            "sessions_pulled": sessions_pulled,
            "sessions_pushed": 0,
            "description_updated": description_updated,
            "new_sync_version": remote_sync_version,
            "slack_message": local_saved_slack_message,
        }

    # Build full conversation data for push
    merged_conversation = _build_merged_conversation_data(
        agent_info=agent_info,
        agent_id=agent_id,
        sessions=local_sessions,
        db=db,
        contributor_token=agent_info.share_contributor_token,
    )

    # Do NOT preserve remote ui_metadata: we want OG/UI metadata to reflect the latest local agent profile.
    try:
        merged_conversation["ui_metadata"] = (
            merged_conversation.get("ui_metadata")
            if isinstance(merged_conversation.get("ui_metadata"), dict)
            else {}
        )
    except Exception:
        merged_conversation["ui_metadata"] = {}

    new_slack_message = None
    new_agent_sections = None

    def _apply_sync_ui_metadata(payload: Dict[str, Any]) -> None:
        """Apply latest sync UI metadata (title/desc/sections/message) to payload in-place."""
        payload.setdefault("ui_metadata", {})
        if not isinstance(payload.get("ui_metadata"), dict):
            payload["ui_metadata"] = {}

        ui = payload["ui_metadata"]
        ui["title"] = agent_info.title or agent_info.name or "Agent Sessions"
        ui["description"] = agent_info.description or ""
        ui["share_kind"] = "agent"
        if agent_info.title:
            ui["context_title"] = agent_info.title
        ui["agent_name"] = agent_info.name

        if isinstance(new_slack_message, str) and new_slack_message.strip():
            ui["slack_message"] = new_slack_message

        if new_agent_sections is not None:
            ui["agent_sections"] = new_agent_sections
        elif isinstance(remote_agent_sections, list) and remote_agent_sections:
            ui["agent_sections"] = remote_agent_sections

    try:
        _progress("Generating share message...")
        from ..config import ReAlignConfig
        from .export_shares import generate_ui_metadata_with_llm

        config = ReAlignConfig.load()
        ui_metadata, _ = generate_ui_metadata_with_llm(
            merged_conversation,
            [],  # no commits, use turn summaries as input
            event_title=agent_info.title or agent_info.name or "Agent Sessions",
            event_description=agent_info.description or "",
            provider=config.llm_provider,
            preset_id="default",
            silent=True,
        )
        maybe_sections = ui_metadata.get("agent_sections") if ui_metadata else None
        if isinstance(maybe_sections, list):
            new_agent_sections = maybe_sections

        maybe_msg = ui_metadata.get("slack_message") if ui_metadata else None
        if isinstance(maybe_msg, str) and maybe_msg.strip():
            new_slack_message = maybe_msg.strip()
    except Exception as e:
        logger.warning(f"Failed to regenerate Slack message for sync push (non-fatal): {e}")

    _apply_sync_ui_metadata(merged_conversation)

    # Push with optimistic locking + retry
    from .export_shares import _update_share_content

    new_version = remote_sync_version
    for attempt in range(MAX_SYNC_RETRIES):
        try:
            push_result = _update_share_content(
                backend_url=backend_url,
                share_id=share_id,
                token=token,
                conversation_data=merged_conversation,
                expected_version=new_version,
            )
            new_version = push_result.get("version", new_version + 1)
            break
        except Exception as e:
            conflict_current_version = _extract_httpx_conflict_current_version(e)
            is_conflict = (
                HTTPX_AVAILABLE
                and isinstance(e, httpx.HTTPStatusError)
                and e.response is not None
                and e.response.status_code == 409
            )
            if is_conflict and attempt < MAX_SYNC_RETRIES - 1:
                if conflict_current_version is not None:
                    _progress(
                        "Version conflict "
                        f"(remote={conflict_current_version}, local_expected={new_version}), "
                        f"retrying ({attempt + 2}/{MAX_SYNC_RETRIES})..."
                    )
                else:
                    _progress(f"Version conflict, retrying ({attempt + 2}/{MAX_SYNC_RETRIES})...")

                # Re-pull to re-merge any remote changes; also bypass potential CDN caching.
                remote_data = _pull_remote(backend_url, share_id)
                if remote_data.get("success"):
                    conv = remote_data["data"]
                    remote_version = conv.get("sync_metadata", {}).get("sync_version", 0)
                    try:
                        remote_ui_metadata = conv.get("ui_metadata", {}) or remote_ui_metadata
                        remote_sections = (
                            remote_ui_metadata.get("agent_sections")
                            if isinstance(remote_ui_metadata, dict)
                            else None
                        )
                        remote_agent_sections = (
                            remote_sections if isinstance(remote_sections, list) else None
                        )
                    except Exception:
                        pass
                    try:
                        remote_version_int = int(remote_version)
                    except Exception:
                        remote_version_int = 0

                    if conflict_current_version is not None:
                        remote_version_int = max(remote_version_int, conflict_current_version)

                    new_version = remote_version_int

                    # Rebuild merge inputs from refreshed remote snapshot.
                    remote_sessions_data = conv.get("sessions", [])
                    remote_event = conv.get("event", {})

                    remote_content_hashes = set()
                    for session_data in remote_sessions_data:
                        for turn_data in session_data.get("turns", []):
                            h = turn_data.get("content_hash")
                            if h:
                                remote_content_hashes.add(h)

                    # Re-import remote sessions (idempotent via content_hash dedup) and re-merge metadata.
                    try:
                        from .import_shares import import_session_with_turns

                        local_content_hashes = db.get_agent_content_hashes(agent_id)
                        linked_session_ids = set(db.get_session_ids_by_agent_id(agent_id))
                        for session_data in remote_sessions_data:
                            session_id = session_data.get("session_id", "")
                            session_turns = session_data.get("turns", [])

                            new_turns = [
                                t
                                for t in session_turns
                                if t.get("content_hash")
                                and t["content_hash"] not in local_content_hashes
                            ]

                            existing_session = db.get_session_by_id(session_id)
                            session_is_new = existing_session is None
                            session_needs_linking = bool(
                                existing_session and session_id not in linked_session_ids
                            )

                            if not new_turns and not session_is_new and not session_needs_linking:
                                continue

                            should_count = session_is_new or session_needs_linking
                            try:
                                os.environ["REALIGN_DISABLE_AUTO_SUMMARIES"] = "1"
                                import_result = import_session_with_turns(
                                    session_data,
                                    f"agent-{agent_id}",
                                    agent_info.share_url,
                                    db,
                                    force=False,
                                )
                                if (
                                    import_result.get("sessions", 0) > 0
                                    or import_result.get("turns", 0) > 0
                                    or should_count
                                ):
                                    sessions_pulled += 1
                            except Exception as ie:
                                logger.error(f"Failed to import remote session {session_id}: {ie}")
                                if should_count:
                                    sessions_pulled += 1

                            try:
                                if db.link_session_to_agent(agent_id, session_id):
                                    linked_session_ids.add(session_id)
                            except AgentSessionLimitExceededError:
                                limit_msg = (
                                    f"当前“Context”关联的session已经达到上限（{MAX_SESSION_NUM}），"
                                    "无法继续同步更多session。"
                                )
                                return {"success": False, "error": limit_msg}
                            except Exception as le:
                                logger.error(f"Failed to link session {session_id} to agent: {le}")

                        # Re-merge name/description: last-write-wins by updated_at.
                        refreshed_remote_updated_at = remote_event.get("updated_at")
                        remote_dt = _parse_iso_datetime_to_utc(refreshed_remote_updated_at)
                        local_dt = (
                            _ensure_aware_utc(agent_info.updated_at)
                            if isinstance(agent_info.updated_at, datetime)
                            else None
                        )
                        if remote_dt and local_dt and remote_dt > local_dt:
                            remote_name = remote_event.get("title")
                            remote_desc = remote_event.get("description")
                            remote_context_title = remote_event.get("context_title")
                            if (
                                isinstance(remote_context_title, str)
                                and not remote_context_title.strip()
                            ):
                                remote_context_title = None
                            updates = {}
                            if remote_name and remote_name != agent_info.name:
                                updates["name"] = remote_name
                            if (
                                remote_context_title is not None
                                and remote_context_title != agent_info.title
                            ):
                                updates["title"] = remote_context_title
                            if remote_desc is not None and remote_desc != agent_info.description:
                                updates["description"] = remote_desc
                            if updates:
                                db.update_agent_info(agent_id, **updates)
                                description_updated = True
                                agent_info = db.get_agent_info(agent_id)
                    except Exception as merge_e:
                        logger.warning(
                            f"Failed to refresh merge after conflict (non-fatal): {merge_e}"
                        )

                    # Rebuild payload from refreshed local state before retrying.
                    local_sessions = db.get_sessions_by_agent_id(agent_id)
                    merged_conversation = _build_merged_conversation_data(
                        agent_info=agent_info,
                        agent_id=agent_id,
                        sessions=local_sessions,
                        db=db,
                        contributor_token=agent_info.share_contributor_token,
                    )
                    # Re-apply regenerated content (including agent_sections) after rebuild.
                    _apply_sync_ui_metadata(merged_conversation)

                elif conflict_current_version is not None:
                    # If pull fails, fall back to server-provided current_version.
                    new_version = max(new_version, conflict_current_version)
                continue
            else:
                logger.error(f"Push failed after {attempt + 1} attempts: {e}")
                return {"success": False, "error": f"Push failed: {e}"}

    # 5. Update local sync metadata
    now_iso = datetime.now(timezone.utc).isoformat()
    db.update_agent_sync_metadata(
        agent_id,
        last_synced_at=now_iso,
        sync_version=new_version,
        slack_message=new_slack_message if new_slack_message else None,
    )

    _progress("Sync complete!")

    return {
        "success": True,
        "sessions_pulled": sessions_pulled,
        "sessions_pushed": sessions_pushed,
        "description_updated": description_updated,
        "new_sync_version": new_version,
        "slack_message": new_slack_message or local_saved_slack_message,
    }


def _pull_remote(backend_url: str, share_id: str) -> dict:
    """Pull remote share data via the download_share_data helper."""
    try:
        from .import_shares import download_share_data

        share_url = f"{backend_url}/share/{share_id}"
        cache_buster = str(int(time.time() * 1000))
        return download_share_data(share_url, password=None, cache_buster=cache_buster)
    except Exception as e:
        return {"success": False, "error": str(e)}


def _build_merged_conversation_data(
    agent_info,
    agent_id: str,
    sessions,
    db,
    contributor_token: Optional[str] = None,
) -> dict:
    """
    Build a full conversation data dict from local agent state.

    Mirrors the structure of build_enhanced_conversation_data but works
    directly from DB records without ExportableSession wrappers.
    """
    import json as json_module

    event_data = {
        "event_id": f"agent-{agent_id}",
        "title": agent_info.name or "Agent Sessions",
        "context_title": agent_info.title or "",
        "description": agent_info.description or "",
        "event_type": "agent",
        "status": "active",
        "created_at": (
            _ensure_aware_utc(agent_info.created_at).isoformat() if agent_info.created_at else None
        ),
        "updated_at": (
            _ensure_aware_utc(agent_info.updated_at).isoformat() if agent_info.updated_at else None
        ),
    }

    sessions_data = []
    for session in sessions:
        turns = db.get_turns_for_session(session.id)
        turns_data = []
        for turn in turns:
            turns_data.append(
                {
                    "turn_id": turn.id,
                    "turn_number": turn.turn_number,
                    "content_hash": turn.content_hash,
                    "timestamp": turn.timestamp.isoformat() if turn.timestamp else None,
                    "llm_title": turn.llm_title or "",
                    "llm_description": turn.llm_description,
                    "assistant_summary": turn.assistant_summary,
                    "messages": [],  # metadata-only sync (no content)
                    "content_available": False,
                }
            )

        sessions_data.append(
            {
                "session_id": session.id,
                "session_type": session.session_type or "unknown",
                "session_title": session.session_title,
                "session_summary": session.session_summary,
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "last_activity_at": (
                    session.last_activity_at.isoformat() if session.last_activity_at else None
                ),
                "created_by": session.created_by,
                "turns": turns_data,
            }
        )

    username = os.environ.get("USER") or os.environ.get("USERNAME") or "anonymous"

    result = {
        "version": "2.1",
        "username": username,
        "time": datetime.now(timezone.utc).isoformat(),
        "event": event_data,
        "sessions": sessions_data,
        "ui_metadata": {
            "share_kind": "agent",
            "title": agent_info.title or agent_info.name or "Agent Sessions",
            "description": agent_info.description or "",
            "context_title": agent_info.title or "",
            "agent_name": agent_info.name,
        },
    }

    # Phase 0: do not include contributor_token in payload blobs (avoid accidental leakage).
    # Server can inject contributor_token for owner on export when needed.
    result["sync_metadata"] = {
        "sync_version": agent_info.sync_version or 0,
    }

    return result
