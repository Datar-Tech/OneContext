from __future__ import annotations

import os
import socket
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Iterator

from .base import DatabaseInterface


def make_lock_owner(prefix: str = "proc") -> str:
    return f"{prefix}:{socket.gethostname()}:{os.getpid()}:{uuid.uuid4()}"


def lock_key_for_project_commit(project_path: Path) -> str:
    return f"commit_pipeline:{str(project_path.resolve())}"


def lock_key_for_turn_processing(session_id: str, turn_number: int) -> str:
    return f"turn_process:{session_id}:{int(turn_number)}"


def lock_key_for_session_summary(session_id: str) -> str:
    return f"session_summary:{session_id}"


def lock_key_for_event_summary(event_id: str) -> str:
    return f"event_summary:{event_id}"


def lock_key_for_agent_description(agent_id: str) -> str:
    return f"agent_description:{agent_id}"


@contextmanager
def lease_lock(
    db: DatabaseInterface,
    lock_key: str,
    *,
    owner: Optional[str] = None,
    ttl_seconds: float,
    wait_timeout_seconds: float = 0.0,
    poll_interval_seconds: float = 0.1,
) -> Iterator[bool]:
    """
    DB-backed lease lock context manager.

    Yields:
        acquired (bool)
    """
    lock_owner = owner or make_lock_owner("lock")
    deadline = time.time() + float(wait_timeout_seconds)
    acquired = False

    while True:
        acquired = bool(db.try_acquire_lock(lock_key, owner=lock_owner, ttl_seconds=ttl_seconds))
        if acquired or time.time() >= deadline:
            break
        time.sleep(float(poll_interval_seconds))

    try:
        yield acquired
    finally:
        if acquired:
            try:
                db.release_lock(lock_key, owner=lock_owner)
            except Exception:
                pass
