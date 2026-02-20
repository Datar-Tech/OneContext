"""Agent description generation from session summaries using LLM."""

import logging
from pathlib import Path
from typing import List, Optional

from ..db.sqlite_db import SQLiteDatabase
from ..db.base import SessionRecord
from ..db.locks import lease_lock, lock_key_for_agent_description, make_lock_owner
from ..llm_client import call_llm_cloud

logger = logging.getLogger(__name__)


def schedule_agent_description_update(db: SQLiteDatabase, agent_id: str) -> None:
    """
    Enqueue an agent description update job (durable).

    Call this after a session summary is updated for an agent-linked session.

    Args:
        db: Database instance
        agent_id: ID of the agent to update
    """
    try:
        db.enqueue_agent_description_job(agent_id=agent_id)
        logger.debug(f"Enqueued agent description job for agent_id={agent_id}")
    except Exception as e:
        logger.debug(f"Failed to enqueue agent description job, falling back: {e}")
        force_update_agent_description(db, agent_id)


def force_update_agent_description(db: SQLiteDatabase, agent_id: str) -> None:
    """
    Immediately update agent description, bypassing the job queue.

    Args:
        db: Database instance
        agent_id: ID of the agent to update
    """
    _update_agent_description(db, agent_id)


def _update_agent_description(db: SQLiteDatabase, agent_id: str) -> None:
    """
    Actually perform the agent description update.

    Args:
        db: Database instance
        agent_id: ID of the agent to update
    """
    logger.info(f"Updating agent description for agent_id={agent_id}")

    owner = make_lock_owner("agent_description")
    lock_key = lock_key_for_agent_description(agent_id)

    with lease_lock(
        db,
        lock_key,
        owner=owner,
        ttl_seconds=10 * 60,  # 10 minutes
        wait_timeout_seconds=0.0,
    ) as acquired:
        if not acquired:
            logger.debug(f"Agent description lock held by another process: agent_id={agent_id}")
            return

        sessions = db.get_sessions_by_agent_id(agent_id)
        if not sessions:
            logger.warning(f"No sessions found for agent_id={agent_id}")
            return

        # Filter to sessions with non-empty title or summary
        sessions_with_content = [
            s for s in sessions if s.session_title or s.session_summary
        ]
        if not sessions_with_content:
            logger.warning(f"No sessions with summaries for agent_id={agent_id}")
            return

        title, description = _generate_agent_description_llm(sessions_with_content)
        db.update_agent_info(agent_id, title=title, description=description)
        logger.info(
            f"Agent description updated: title='{title}', desc='{description[:60]}...' for agent_id={agent_id}"
        )


def _generate_agent_description_llm(sessions: List[SessionRecord]) -> tuple:
    """
    Use LLM to generate an agent title and description from all its sessions.

    Args:
        sessions: List of sessions linked to the agent

    Returns:
        Tuple of (title, description)
    """
    if not sessions:
        return ("", "")

    # Build sessions payload
    sessions_data = []
    for i, session in enumerate(sessions):
        sessions_data.append(
            {
                "session_number": i + 1,
                "title": session.session_title or f"Session {session.id[:8]}",
                "summary": session.session_summary or "(no summary)",
            }
        )

    # Try cloud provider
    try:
        from ..auth import is_logged_in

        if is_logged_in():
            logger.debug("Attempting cloud LLM for agent description")
            custom_prompt = None
            user_prompt_path = Path.home() / ".aline" / "prompts" / "agent_description.md"
            try:
                if user_prompt_path.exists():
                    custom_prompt = user_prompt_path.read_text(encoding="utf-8").strip()
            except Exception:
                pass

            _, result = call_llm_cloud(
                task="agent_description",
                payload={"sessions": sessions_data},
                custom_prompt=custom_prompt,
                silent=True,
            )

            if result:
                title = result.get("title", "")
                description = result.get("description", "")
                logger.info(f"Cloud LLM agent description success: title='{title}', desc='{description[:60]}...'")
                return (title, description)
            else:
                logger.warning("Cloud LLM agent description failed, using fallback")
                return _fallback_agent_description(sessions)
    except ImportError:
        logger.debug("Auth module not available, skipping cloud LLM")

    logger.warning("Not logged in, cannot use cloud LLM for agent description")
    return _fallback_agent_description(sessions)



def _fallback_agent_description(sessions: List[SessionRecord]) -> tuple:
    """Fallback when LLM fails: simple concatenation of recent session summaries."""
    summaries = [s.session_summary for s in sessions if s.session_summary]
    if not summaries:
        titles = [s.session_title for s in sessions if s.session_title]
        if titles:
            return ("", f"Context with {len(sessions)} sessions. Recent work: {'; '.join(titles[-3:])}")
        return ("", f"Context with {len(sessions)} sessions.")

    recent = summaries[:3]  # sessions already ordered by last_activity_at DESC
    return ("", f"Context with {len(sessions)} sessions. " + " ".join(recent))
