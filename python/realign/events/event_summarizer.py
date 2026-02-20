"""Event summary generation from sessions using LLM."""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional

from ..db.sqlite_db import SQLiteDatabase
from ..db.base import SessionRecord
from ..db.locks import lease_lock, lock_key_for_event_summary, make_lock_owner
from .debouncer import Debouncer
from ..llm_client import extract_json, call_llm_cloud

logger = logging.getLogger(__name__)

# Prompt cache
_EVENT_SUMMARY_PROMPT_CACHE: Optional[str] = None

# Global debouncer instance (separate from session debouncer)
_event_debouncer = Debouncer()
EVENT_DEBOUNCE_SECONDS = 10.0  # 10 seconds debounce (longer than session)


def schedule_event_summary_update(db: SQLiteDatabase, event_id: str) -> None:
    """
    Schedule an event summary update with debounce.

    Call this after a session summary is updated.

    Args:
        db: Database instance
        event_id: ID of the event to update
    """

    def do_update():
        _update_event_summary(db, event_id)

    _event_debouncer.schedule(
        key=f"event:{event_id}",
        delay_seconds=EVENT_DEBOUNCE_SECONDS,
        callback=do_update,
    )
    logger.debug(f"Scheduled event summary update for event_id={event_id}")


def force_update_event_summary(db: SQLiteDatabase, event_id: str) -> None:
    """
    Immediately update event summary, bypassing debounce.

    Args:
        db: Database instance
        event_id: ID of the event to update
    """
    _event_debouncer.cancel(f"event:{event_id}")
    _update_event_summary(db, event_id)


def flush_all_event_updates() -> None:
    """Immediately execute all pending event updates."""
    _event_debouncer.flush_all()


def _update_event_summary(db: SQLiteDatabase, event_id: str) -> None:
    """
    Actually perform the event summary update.

    Args:
        db: Database instance
        event_id: ID of the event to update
    """
    logger.info(f"Updating event summary for event_id={event_id}")

    owner = make_lock_owner("event_summary")
    lock_key = lock_key_for_event_summary(event_id)

    with lease_lock(
        db,
        lock_key,
        owner=owner,
        ttl_seconds=10 * 60,  # 10 minutes
        wait_timeout_seconds=0.0,
    ) as acquired:
        if not acquired:
            logger.debug(f"Event summary lock held by another process: event_id={event_id}")
            return

        sessions = db.get_sessions_for_event(event_id)
        if not sessions:
            logger.warning(f"No sessions found for event_id={event_id}")
            return

        # Generate title and description using LLM
        title, description = _generate_event_summary_llm(sessions)

        # Update database
        db.update_event_summary(event_id, title, description)
        logger.info(f"Event summary updated: title='{title[:50]}...' for event_id={event_id}")


def _get_event_summary_prompt() -> str:
    """
    Load event summary prompt with user customization support.

    Priority:
    1. User custom prompt (~/.aline/prompts/event_summary.md)
    2. Built-in prompt (tools/commit_message_prompts/event_summary.md)
    3. Hardcoded default

    Returns:
        System prompt for event summary generation
    """
    global _EVENT_SUMMARY_PROMPT_CACHE
    if _EVENT_SUMMARY_PROMPT_CACHE is not None:
        return _EVENT_SUMMARY_PROMPT_CACHE

    # Try user-customized prompt first (~/.aline/prompts/event_summary.md)
    user_prompt_path = Path.home() / ".aline" / "prompts" / "event_summary.md"
    try:
        if user_prompt_path.exists():
            text = user_prompt_path.read_text(encoding="utf-8").strip()
            if text:
                _EVENT_SUMMARY_PROMPT_CACHE = text
                logger.debug(f"Loaded user-customized event summary prompt from {user_prompt_path}")
                return text
    except Exception:
        logger.debug(
            "Failed to load user-customized event summary prompt, falling back", exc_info=True
        )

    # Fall back to built-in prompt (tools/commit_message_prompts/event_summary.md)
    candidate = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "commit_message_prompts"
        / "event_summary.md"
    )
    try:
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                _EVENT_SUMMARY_PROMPT_CACHE = text
                logger.debug(f"Loaded event summary prompt from {candidate}")
                return text
    except Exception:
        logger.debug("Failed to load built-in event summary prompt, using default", exc_info=True)

    # Hardcoded default
    default_prompt = """You are summarizing a development event that spans multiple coding sessions.

Your task:
1. Generate a concise EVENT TITLE (max 100 characters) that captures the main goal or theme across all sessions.
2. Generate an EVENT DESCRIPTION (3-6 sentences) that synthesizes what was accomplished across all sessions.

Output in English.

Output STRICT JSON only with this schema (no extra text, no markdown):
{
  "event_title": "string (max 100 chars)",
  "event_description": "string (3-6 sentences)"
}

Rules:
- The title should be high-level and describe the overall objective or feature.
- The description should highlight key accomplishments, not just list each session.
- Focus on the outcome and impact, not the process.
- Use action-oriented language (e.g., "Implemented X", "Refactored Y", "Fixed Z")."""

    _EVENT_SUMMARY_PROMPT_CACHE = default_prompt
    return _EVENT_SUMMARY_PROMPT_CACHE


def _generate_event_summary_llm(sessions: List[SessionRecord]) -> Tuple[str, str]:
    """
    Use LLM to generate event title and description from all sessions.

    Args:
        sessions: List of sessions in the event

    Returns:
        Tuple of (title, description)
    """
    if not sessions:
        return "Untitled Event", ""

    # If only one session, use it directly
    if len(sessions) == 1:
        s = sessions[0]
        title = s.session_title or f"Session {s.id[:8]}"
        description = s.session_summary or ""
        return title, description

    # Build sessions payload for prompt
    sessions_data = []
    for i, session in enumerate(sessions):
        sessions_data.append(
            {
                "session_number": i + 1,
                "title": session.session_title or f"Session {session.id[:8]}",
                "summary": session.session_summary or "(no summary)",
                "session_type": session.session_type,
            }
        )

    # Try cloud provider first if user is logged in
    try:
        from ..auth import is_logged_in

        if is_logged_in():
            logger.debug("Attempting cloud LLM for event summary")
            # Load user custom prompt if available
            custom_prompt = None
            user_prompt_path = Path.home() / ".aline" / "prompts" / "event_summary.md"
            try:
                if user_prompt_path.exists():
                    custom_prompt = user_prompt_path.read_text(encoding="utf-8").strip()
            except Exception:
                pass

            _, result = call_llm_cloud(
                task="event_summary",
                payload={"sessions": sessions_data},
                custom_prompt=custom_prompt,
                silent=True,
            )

            if result:
                title = result.get("event_title", "Untitled Event")[:100]
                description = result.get("event_description", "")
                logger.info(f"Cloud LLM event summary success: title={title[:50]}...")
                return title, description
            else:
                # Cloud LLM failed, use fallback (local fallback disabled)
                logger.warning("Cloud LLM event summary failed, using fallback")
                return _fallback_event_summary(sessions)
    except ImportError:
        logger.debug("Auth module not available, skipping cloud LLM")

    # User not logged in, use fallback (local fallback disabled)
    logger.warning("Not logged in, cannot use cloud LLM for event summary")
    return _fallback_event_summary(sessions)

    # =========================================================================
    # LOCAL LLM FALLBACK DISABLED - Code kept for reference
    # =========================================================================
    # system_prompt = _get_event_summary_prompt()
    #
    # user_prompt = json.dumps(
    #     {
    #         "total_sessions": len(sessions),
    #         "sessions": sessions_data,
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
    #         purpose="event_summary",
    #     )
    #
    #     if not response:
    #         logger.warning("LLM returned empty response, using fallback")
    #         return _fallback_event_summary(sessions)
    #
    #     result = extract_json(response)
    #
    #     title = result.get("event_title", "Untitled Event")[:100]
    #     description = result.get("event_description", "")
    #
    #     return title, description
    #
    # except Exception as e:
    #     logger.warning(f"LLM event summary failed, using fallback: {e}")
    #     return _fallback_event_summary(sessions)


def _fallback_event_summary(sessions: List[SessionRecord]) -> Tuple[str, str]:
    """Fallback when LLM fails: use simple concatenation."""
    titles = [s.session_title or f"Session {s.id[:8]}" for s in sessions]

    if len(titles) == 1:
        title = titles[0]
    else:
        title = f"{titles[-1]} (+{len(titles)-1} sessions)"

    # Build description from session summaries
    summaries = [s.session_summary for s in sessions if s.session_summary]
    if summaries:
        description = f"Event containing {len(sessions)} sessions. " + " ".join(summaries[-3:])
    else:
        description = f"Event containing {len(sessions)} sessions."

    return title[:100], description
