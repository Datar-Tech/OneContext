"""Session summary generation from turns using LLM."""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

from ..db.sqlite_db import SQLiteDatabase
from ..db.base import TurnRecord
from ..db.locks import lease_lock, lock_key_for_session_summary, make_lock_owner
from ..llm_client import extract_json, call_llm_cloud

logger = logging.getLogger(__name__)

SUMMARY_LEASE_SECONDS = 10 * 60  # 10 minutes TTL to avoid stuck processing

# Prompt cache
_SESSION_SUMMARY_PROMPT_CACHE: Optional[str] = None


def schedule_session_summary_update(db: SQLiteDatabase, session_id: str) -> None:
    """
    Enqueue a session summary update job (durable).

    Call this after a completed turn is inserted into the database.

    Args:
        db: Database instance
        session_id: ID of the session to update
    """
    try:
        db.enqueue_session_summary_job(session_id=session_id)
        logger.debug(f"Enqueued session summary job for session_id={session_id}")
    except Exception as e:
        # Best-effort: if jobs table isn't available, fall back to direct update.
        logger.debug(f"Failed to enqueue session summary job, falling back: {e}")
        update_session_summary_now(db, session_id)


def force_update_session_summary(db: SQLiteDatabase, session_id: str) -> None:
    """
    Immediately update session summary, bypassing debounce.

    Args:
        db: Database instance
        session_id: ID of the session to update
    """
    update_session_summary_now(db, session_id)


def update_session_summary_now(db: SQLiteDatabase, session_id: str) -> bool:
    """
    Actually perform the session summary update.

    Args:
        db: Database instance
        session_id: ID of the session to update
    """
    logger.info(f"Updating session summary for session_id={session_id}")

    now = datetime.now()
    owner = make_lock_owner("session_summary")
    lock_key = lock_key_for_session_summary(session_id)

    with lease_lock(
        db,
        lock_key,
        owner=owner,
        ttl_seconds=SUMMARY_LEASE_SECONDS,
        wait_timeout_seconds=0.0,
    ) as acquired:
        if not acquired:
            logger.debug(f"Session summary lock held by another process: session_id={session_id}")
            return False

        # Update runtime status (best-effort, no hard dependency on V7 schema).
        try:
            db.update_session_summary_runtime(
                session_id=session_id,
                summary_status="processing",
                summary_locked_until=now + timedelta(seconds=SUMMARY_LEASE_SECONDS),
                summary_error=None,
            )
        except Exception:
            pass

        turns = db.get_turns_for_session(session_id)
        # Exclude processing/failed placeholder turns from aggregation.
        turns = [t for t in turns if (t.turn_status in (None, "completed"))]
        if not turns:
            logger.warning(f"No turns found for session_id={session_id}")
            try:
                db.update_session_summary_runtime(session_id=session_id, summary_status="idle")
            except Exception:
                pass
            return True

        # Check session metadata for no_track mode
        is_no_track = False
        try:
            session = db.get_session_by_id(session_id)
            if session:
                session_meta = getattr(session, "metadata", None) or {}
                is_no_track = bool(session_meta.get("no_track", False))
        except Exception:
            pass

        try:
            # Skip LLM call for no-track mode
            if is_no_track:
                title, summary = "No Track", "No Track"
                logger.info(f"No-track mode: skipping LLM for session summary {session_id}")
            else:
                # Generate title and summary using LLM
                title, summary = _generate_session_summary_llm(turns)

            # Update database
            db.update_session_summary(session_id, title, summary)
            try:
                db.update_session_summary_runtime(
                    session_id=session_id,
                    summary_status="completed",
                    summary_locked_until=None,
                    summary_error=None,
                )
            except Exception:
                pass
            logger.info(
                f"Session summary updated: title='{title[:50]}...' for session_id={session_id}"
            )
        except Exception as e:
            logger.warning(f"Session summary update failed: {e}")
            try:
                db.update_session_summary_runtime(
                    session_id=session_id,
                    summary_status="failed",
                    summary_locked_until=None,
                    summary_error=str(e)[:2000],
                )
            except Exception:
                pass
            return False

        # Trigger event summary updates for all parent events (if any)
        _trigger_parent_event_updates(db, session_id)

        # Trigger agent description update if session is linked to an agent
        _trigger_agent_description_update(db, session_id)
        return True


def _trigger_parent_event_updates(db: SQLiteDatabase, session_id: str) -> None:
    """
    Trigger summary updates for all events containing this session.

    Args:
        db: Database instance
        session_id: ID of the session whose parents should be updated
    """
    try:
        from .event_summarizer import schedule_event_summary_update

        events = db.get_events_for_session(session_id)
        for event in events:
            schedule_event_summary_update(db, event.id)
            logger.debug(f"Triggered event summary update for event_id={event.id}")
    except Exception as e:
        # Don't fail session summary update if event trigger fails
        logger.warning(f"Failed to trigger event updates: {e}")


def _trigger_agent_description_update(db: SQLiteDatabase, session_id: str) -> None:
    """
    Trigger agent description update if the session is linked to an agent.

    Args:
        db: Database instance
        session_id: ID of the session whose agent should be updated
    """
    try:
        agent_ids = db.get_agent_ids_by_session_id(session_id)
        if not agent_ids:
            return
        from .agent_summarizer import schedule_agent_description_update

        for agent_id in agent_ids:
            schedule_agent_description_update(db, agent_id)
            logger.debug(f"Triggered agent description update for agent_id={agent_id}")
    except Exception as e:
        # Don't fail session summary update if agent trigger fails
        logger.warning(f"Failed to trigger agent description update: {e}")


def _get_session_summary_prompt() -> str:
    """
    Load session summary prompt with user customization support.

    Priority:
    1. User custom prompt (~/.aline/prompts/session_summary.md)
    2. Built-in prompt (tools/commit_message_prompts/session_summary.md)
    3. Hardcoded default

    Returns:
        System prompt for session summary generation
    """
    global _SESSION_SUMMARY_PROMPT_CACHE
    if _SESSION_SUMMARY_PROMPT_CACHE is not None:
        return _SESSION_SUMMARY_PROMPT_CACHE

    # Try user-customized prompt first (~/.aline/prompts/session_summary.md)
    user_prompt_path = Path.home() / ".aline" / "prompts" / "session_summary.md"
    try:
        if user_prompt_path.exists():
            text = user_prompt_path.read_text(encoding="utf-8").strip()
            if text:
                _SESSION_SUMMARY_PROMPT_CACHE = text
                logger.debug(
                    f"Loaded user-customized session summary prompt from {user_prompt_path}"
                )
                return text
    except Exception:
        logger.debug(
            "Failed to load user-customized session summary prompt, falling back", exc_info=True
        )

    # Fall back to built-in prompt (tools/commit_message_prompts/session_summary.md)
    candidate = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "commit_message_prompts"
        / "session_summary.md"
    )
    try:
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                _SESSION_SUMMARY_PROMPT_CACHE = text
                logger.debug(f"Loaded session summary prompt from {candidate}")
                return text
    except Exception:
        logger.debug("Failed to load built-in session summary prompt, using default", exc_info=True)

    # Hardcoded default
    default_prompt = """You are summarizing a coding session that contains multiple conversation turns.

Your task:
1. Generate a concise SESSION TITLE (max 80 characters) that captures the main goal or theme of the entire session.
2. Generate a SESSION SUMMARY (2-5 sentences) that describes what was accomplished across all turns.

Output in English.

Output STRICT JSON only with this schema (no extra text, no markdown):
{
  "session_title": "string (max 80 chars)",
  "session_summary": "string (2-5 sentences)"
}

Rules:
- The title should be concise and descriptive, capturing the overall goal.
- The summary should highlight key accomplishments, not just list each turn.
- If the session involves multiple unrelated tasks, focus on the most significant ones.
- Prefer action-oriented language (e.g., "Implement X", "Fix Y", "Refactor Z")."""

    _SESSION_SUMMARY_PROMPT_CACHE = default_prompt
    return _SESSION_SUMMARY_PROMPT_CACHE


def _generate_session_summary_llm(turns: List[TurnRecord]) -> Tuple[str, str]:
    """
    Use LLM to generate session title and summary from all turns.

    Args:
        turns: List of turns in the session

    Returns:
        Tuple of (title, summary)
    """
    if not turns:
        return "Untitled Session", ""

    # Build turns payload for prompt
    turns_data = []
    for i, turn in enumerate(turns):
        turns_data.append(
            {
                "turn_number": i + 1,
                "title": turn.llm_title or "(no title)",
                "summary": turn.assistant_summary or "(no summary)",
                "user_request": (turn.user_message or "")[:200],  # Truncate long messages
            }
        )

    # Try cloud provider first if user is logged in
    try:
        from ..auth import is_logged_in

        if is_logged_in():
            logger.debug("Attempting cloud LLM for session summary")
            # Load user custom prompt if available
            custom_prompt = None
            user_prompt_path = Path.home() / ".aline" / "prompts" / "session_summary.md"
            try:
                if user_prompt_path.exists():
                    custom_prompt = user_prompt_path.read_text(encoding="utf-8").strip()
            except Exception:
                pass

            _, result = call_llm_cloud(
                task="session_summary",
                payload={"turns": turns_data},
                custom_prompt=custom_prompt,
                silent=True,
            )

            if result:
                title = result.get("session_title", "Untitled Session")[:80]
                summary = result.get("session_summary", "")
                logger.info(f"Cloud LLM session summary success: title={title[:50]}...")
                return title, summary
            else:
                # Cloud LLM failed, use fallback (local fallback disabled)
                logger.warning("Cloud LLM session summary failed, using fallback")
                return _fallback_summary(turns)
    except ImportError:
        logger.debug("Auth module not available, skipping cloud LLM")

    # User not logged in, use fallback (local fallback disabled)
    logger.warning("Not logged in, cannot use cloud LLM for session summary")
    return _fallback_summary(turns)

    # =========================================================================
    # LOCAL LLM FALLBACK DISABLED - Code kept for reference
    # =========================================================================
    # system_prompt = _get_session_summary_prompt()
    #
    # user_prompt = json.dumps(
    #     {
    #         "total_turns": len(turns),
    #         "turns": turns_data,
    #     },
    #     ensure_ascii=False,
    #     indent=2,
    # )
    #
    # try:
    #     # Use unified LLM client
    #     _, response = call_llm(
    #         system_prompt=system_prompt,
    #         user_prompt=user_prompt,
    #         provider="auto",  # Try Claude first, fallback to OpenAI
    #         max_tokens=500,
    #         purpose="session_summary",
    #     )
    #
    #     if not response:
    #         logger.warning("LLM returned empty response, using fallback")
    #         return _fallback_summary(turns)
    #
    #     result = extract_json(response)
    #
    #     title = result.get("session_title", "Untitled Session")[:80]
    #     summary = result.get("session_summary", "")
    #
    #     return title, summary
    #
    # except Exception as e:
    #     logger.warning(f"LLM session summary failed, using fallback: {e}")
    #     return _fallback_summary(turns)


def _fallback_summary(turns: List[TurnRecord]) -> Tuple[str, str]:
    """Fallback when LLM fails: use simple concatenation."""
    titles = [t.llm_title for t in turns if t.llm_title]
    summaries = [t.assistant_summary for t in turns if t.assistant_summary]

    # Title: use last title with count
    if titles:
        title = f"{titles[-1]} (+{len(titles)-1})" if len(titles) > 1 else titles[0]
    else:
        title = "Untitled Session"

    # Summary: concatenate recent summaries
    recent = summaries[-5:]
    summary = "\n".join(f"- {s}" for s in recent) if recent else ""

    return title, summary
