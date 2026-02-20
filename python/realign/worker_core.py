"""
Background worker for durable jobs queue.

This process consumes jobs from the SQLite `jobs` table:
- turn_summary: generate/store a turn (LLM + content snapshot)
- session_summary: aggregate session title/summary from turns
- agent_description: regenerate agent description from session summaries
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ReAlignConfig
from .dashboard_tracking import (
    is_dashboard_only_enabled,
    is_session_trackable_for_dashboard_only,
)
from .db.sqlite_db import SQLiteDatabase
from .db.locks import make_lock_owner
from .logging_config import setup_logger

logger = setup_logger("realign.worker_core", "worker_core.log")


class PermanentJobError(RuntimeError):
    """Non-retryable job failure."""


def _backoff_seconds(attempts: int, *, base: float = 2.0, cap: float = 300.0) -> float:
    n = max(0, int(attempts))
    return float(min(cap, base * math.pow(2.0, min(n, 8))))


def _max_attempts() -> int:
    """
    Maximum retry attempts before marking a job as failed.

    Set via REALIGN_JOB_MAX_ATTEMPTS (default: 10).
    """
    raw = os.getenv("REALIGN_JOB_MAX_ATTEMPTS", "10")
    try:
        v = int(raw)
        return max(1, v)
    except Exception:
        return 10


class AlineWorker:
    def __init__(self, db: SQLiteDatabase, *, poll_interval_seconds: float = 0.5):
        self.db = db
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.worker_id = make_lock_owner("worker")
        self.running = False
        try:
            self.config = ReAlignConfig.load()
        except Exception:
            self.config = ReAlignConfig()

        # Commit/turn processing pipeline (no watcher loop / polling).
        from .commit_pipeline import CommitPipeline

        self._pipeline = CommitPipeline(lock_owner_prefix="worker")

    async def start(self) -> None:
        self.running = True
        logger.info(f"Worker started: id={self.worker_id}")

        while self.running:
            try:
                job = self.db.claim_next_job(
                    worker_id=self.worker_id,
                    kinds=[
                        "session_process",
                        "turn_summary",
                        "session_summary",
                        "agent_description",
                    ],
                )
                if not job:
                    await asyncio.sleep(self.poll_interval_seconds)
                    continue

                await self._process_job(job)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}", exc_info=True)
                await asyncio.sleep(1.0)

        logger.info("Worker stopped")

    async def stop(self) -> None:
        self.running = False

    async def _process_job(self, job: Dict[str, Any]) -> None:
        job_id = str(job.get("id"))
        kind = str(job.get("kind"))
        payload = job.get("payload") or {}

        try:
            if kind == "session_process":
                should_enqueue_summary = await self._process_session_process_job(payload)
                try:
                    session_id = str(payload.get("session_id") or "")
                    if should_enqueue_summary and session_id:
                        self.db.enqueue_session_summary_job(session_id=session_id)
                except Exception as e:
                    logger.warning(f"Failed to enqueue session summary after session_process: {e}")
                self.db.finish_job(job_id=job_id, worker_id=self.worker_id, success=True)
                return

            if kind == "turn_summary":
                await self._process_turn_summary_job(payload)
                # Always enqueue a session_summary job after a successful turn job.
                # This ensures session summaries update even if the turn already existed in DB
                # (i.e. the turn job was effectively a validation/no-op).
                try:
                    session_id = str(payload.get("session_id") or "")
                    if session_id:
                        self.db.enqueue_session_summary_job(session_id=session_id)
                except Exception as e:
                    logger.warning(f"Failed to enqueue session summary after turn job: {e}")
                    print(
                        f"[Worker] ⚠ Failed to enqueue session summary job for session_id={session_id}: {e}",
                        file=sys.stderr,
                    )
                self.db.finish_job(job_id=job_id, worker_id=self.worker_id, success=True)
                return

            if kind == "session_summary":
                ok = await self._process_session_summary_job(payload)
                if ok:
                    self.db.finish_job(job_id=job_id, worker_id=self.worker_id, success=True)
                else:
                    attempts = int(job.get("attempts") or 0)
                    next_attempt = attempts + 1
                    if next_attempt >= _max_attempts():
                        self.db.finish_job(
                            job_id=job_id,
                            worker_id=self.worker_id,
                            success=False,
                            error="session summary failed (max attempts reached)",
                            permanent_fail=True,
                        )
                    else:
                        delay = _backoff_seconds(attempts)
                        self.db.finish_job(
                            job_id=job_id,
                            worker_id=self.worker_id,
                            success=False,
                            error="session summary failed",
                            retry_after_seconds=delay,
                        )
                return

            if kind == "agent_description":
                await self._process_agent_description_job(payload)
                self.db.finish_job(job_id=job_id, worker_id=self.worker_id, success=True)
                return

            # Unknown job kind: mark as permanently failed to avoid infinite loops.
            self.db.finish_job(
                job_id=job_id,
                worker_id=self.worker_id,
                success=False,
                error=f"Unknown job kind: {kind}",
                permanent_fail=True,
            )
        except PermanentJobError as e:
            logger.warning(f"Permanent job failure: {kind} id={job_id} err={e}")
            self.db.finish_job(
                job_id=job_id,
                worker_id=self.worker_id,
                success=False,
                error=str(e),
                permanent_fail=True,
            )
        except Exception as e:
            attempts = int(job.get("attempts") or 0)
            next_attempt = attempts + 1
            if next_attempt >= _max_attempts():
                logger.warning(
                    f"Job failed (max attempts reached): {kind} id={job_id} err={e}",
                    exc_info=True,
                )
                self.db.finish_job(
                    job_id=job_id,
                    worker_id=self.worker_id,
                    success=False,
                    error=f"{e} (max attempts reached)",
                    permanent_fail=True,
                )
                return
            delay = _backoff_seconds(attempts)
            logger.warning(f"Job failed: {kind} id={job_id} err={e}", exc_info=True)
            self.db.finish_job(
                job_id=job_id,
                worker_id=self.worker_id,
                success=False,
                error=str(e),
                retry_after_seconds=delay,
            )

    async def _process_session_process_job(self, payload: Dict[str, Any]) -> bool:
        t0 = time.monotonic()
        session_id = str(payload.get("session_id") or "").strip()
        session_file_path = Path(str(payload.get("session_file_path") or ""))
        session_type_raw = str(payload.get("session_type") or "").strip().lower()
        source_event = str(payload.get("source_event") or "").strip().lower()
        workspace_path_raw = payload.get("workspace_path")
        no_track = bool(payload.get("no_track") or False)
        agent_id = str(payload.get("agent_id") or "").strip()
        terminal_id = str(payload.get("terminal_id") or "").strip()

        if not session_file_path:
            raise ValueError(f"Invalid session_process payload: {payload}")
        if not session_file_path.exists():
            raise FileNotFoundError(f"Session file not found: {session_file_path}")
        if not session_id:
            session_id = session_file_path.stem

        logger.info(
            "session_process start: session_id=%s source_event=%s session_type=%s file=%s",
            session_id,
            source_event or "",
            session_type_raw or "",
            str(session_file_path),
        )

        session_type = session_type_raw or str(
            self._pipeline._detect_session_type(session_file_path)
        )

        if session_type == "codex" and (not agent_id or not terminal_id):
            # Best-effort: infer linkage from Aline-managed CODEX_HOME layouts even if the notify
            # runner didn't propagate env vars into the job payload.
            try:
                from .codex_home import (
                    agent_id_from_codex_session_file,
                    terminal_id_from_codex_session_file,
                )

                if not agent_id:
                    agent_id = str(
                        agent_id_from_codex_session_file(session_file_path) or ""
                    ).strip()
                if not terminal_id:
                    terminal_id = str(
                        terminal_id_from_codex_session_file(session_file_path) or ""
                    ).strip()
            except Exception:
                pass

        if is_dashboard_only_enabled(self.config):
            allowed = is_session_trackable_for_dashboard_only(
                session_type=session_type,
                session_file=session_file_path,
                session_id=session_id,
                terminal_id=terminal_id,
                agent_id=agent_id,
                db=self.db,
            )
            if not allowed:
                logger.info(
                    "session_process skipped by dashboard_only: session_id=%s session_type=%s file=%s",
                    session_id,
                    session_type,
                    str(session_file_path),
                )
                return False

        # We intentionally do NOT validate that workspace_path exists; it is used mostly
        # for stable grouping/metadata, and the commit pipeline can operate without a real repo.
        project_path: Path
        if isinstance(workspace_path_raw, str) and workspace_path_raw.strip():
            project_path = Path(workspace_path_raw.strip())
        else:
            # Best-effort: use session metadata for Codex; otherwise fall back to existing path.
            project_path = session_file_path.parent
            if session_type == "codex":
                try:
                    from .codex_terminal_linker import read_codex_session_meta

                    meta = read_codex_session_meta(session_file_path)
                    if meta and (meta.cwd or "").strip():
                        project_path = Path(str(meta.cwd).strip())
                except Exception:
                    pass

        # Ensure the session record exists so agent associations are not lost on "no-op" jobs
        # (e.g. when all turns are already committed and session_process returns early).
        try:
            started_at = datetime.fromtimestamp(session_file_path.stat().st_mtime)
            if session_type == "codex":
                try:
                    from .codex_terminal_linker import read_codex_session_meta

                    meta = read_codex_session_meta(session_file_path)
                    if meta and meta.started_at is not None:
                        dt = meta.started_at
                        if dt.tzinfo is not None:
                            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                        started_at = dt
                except Exception:
                    pass

            self.db.get_or_create_session(
                session_id=session_id,
                session_file_path=session_file_path,
                session_type=session_type,
                started_at=started_at,
                workspace_path=str(project_path) if str(project_path).strip() else None,
                metadata={"source_event": source_event or "", "source": "worker"},
                agent_id=agent_id or None,
            )
        except Exception:
            pass

        # Best-effort: link session to agent/terminal metadata even when no turns need processing.
        if agent_id and session_id:
            try:
                self.db.update_session_agent_id(session_id, agent_id)
            except Exception:
                pass
        if terminal_id and session_id:
            try:
                self.db.insert_window_link(
                    terminal_id=terminal_id,
                    agent_id=agent_id or None,
                    session_id=session_id,
                    provider=session_type or session_type_raw or "",
                    source=f"{session_type or session_type_raw or 'unknown'}:worker",
                    ts=time.time(),
                )
            except Exception:
                pass

        # Determine the safe "max turn" boundary.
        if source_event == "stop":
            max_turn = int(self._pipeline._get_total_turn_count(session_file_path))
        else:
            # Polling/idle paths: respect trigger semantics (Claude excludes last turn).
            max_turn = int(self._pipeline._count_complete_turns(session_file_path))

        if max_turn <= 0:
            logger.info(
                "session_process noop: session_id=%s max_turn=%s duration_s=%.3f",
                session_id,
                max_turn,
                time.monotonic() - t0,
            )
            return True

        committed = set()
        try:
            committed = self.db.get_committed_turn_numbers(session_id)
        except Exception:
            committed = set()

        processed = 0
        skipped = 0

        missing_turns = [t for t in range(1, max_turn + 1) if t not in committed]
        if not missing_turns:
            logger.info(
                "session_process noop: session_id=%s max_turn=%s duration_s=%.3f",
                session_id,
                max_turn,
                time.monotonic() - t0,
            )
            return True

        # Batch turns under one project lease lock to reduce overhead.
        if len(missing_turns) > 1 and hasattr(self._pipeline, "_run_realign_commit_batch"):
            results = self._pipeline._run_realign_commit_batch(
                project_path,
                session_file=session_file_path,
                target_turns=missing_turns,
                quiet=True,
                skip_session_summary=True,
                no_track=no_track,
            )
            for t in missing_turns:
                if results.get(int(t)):
                    committed.add(int(t))
                processed += 1
                await asyncio.sleep(0)
        else:
            # Fallback: per-turn commit (single missing turn or old pipeline).
            for turn_number in missing_turns:
                created = self._pipeline._run_realign_commit(
                    project_path,
                    session_file=session_file_path,
                    target_turn=int(turn_number),
                    quiet=True,
                    skip_session_summary=True,
                    no_track=no_track,
                )
                if created:
                    committed.add(int(turn_number))
                processed += 1
                await asyncio.sleep(0)

        logger.info(
            "session_process done: session_id=%s max_turn=%s processed=%s skipped=%s duration_s=%.3f",
            session_id,
            max_turn,
            processed,
            skipped,
            time.monotonic() - t0,
        )
        return True

    async def _process_turn_summary_job(self, payload: Dict[str, Any]) -> None:
        session_id = str(payload.get("session_id") or "")
        turn_number = int(payload.get("turn_number") or 0)
        session_file_path = Path(str(payload.get("session_file_path") or ""))
        workspace_path_raw = payload.get("workspace_path")
        skip_session_summary = bool(payload.get("skip_session_summary") or False)
        expected_turns_raw = payload.get("expected_turns")
        expected_turns = int(expected_turns_raw) if expected_turns_raw is not None else None
        skip_dedup = bool(payload.get("skip_dedup") or False)
        no_track = bool(payload.get("no_track") or False)
        agent_id = str(payload.get("agent_id") or "")

        if not session_id or turn_number <= 0 or not session_file_path:
            raise ValueError(f"Invalid turn_summary payload: {payload}")

        if not session_file_path.exists():
            raise FileNotFoundError(f"Session file not found: {session_file_path}")

        # Intentionally avoid expensive/fragile project path extraction. If no explicit path is
        # provided, fall back to the session file's parent directory.
        if isinstance(workspace_path_raw, str) and workspace_path_raw.strip():
            project_path = Path(workspace_path_raw.strip())
        else:
            project_path = session_file_path.parent

        # Run the existing commit pipeline (writes turn record + content).
        created = self._pipeline._run_realign_commit(
            project_path,
            session_file=session_file_path,
            target_turn=turn_number,
            quiet=True,
            skip_session_summary=skip_session_summary,
            skip_dedup=skip_dedup,
            no_track=no_track,
        )

        # Link session to agent after commit ensures session exists in DB
        if session_id:
            try:
                if agent_id:
                    self.db.update_session_agent_id(session_id, agent_id)
                else:
                    self._maybe_link_session_from_terminal(session_id)
            except Exception:
                pass

        if created:
            if expected_turns:
                self._enqueue_session_summary_if_complete(session_id, expected_turns)
            return

        # If no new turn was created, decide if this job is actually complete.
        existing = self.db.get_turn_by_number(session_id, turn_number)
        if existing is None:
            raise RuntimeError(
                f"Turn not created and not present in DB: {session_id} #{turn_number}"
            )

        status = getattr(existing, "turn_status", None)
        if status in (None, "completed"):
            if expected_turns:
                self._enqueue_session_summary_if_complete(session_id, expected_turns)
            return
        if status == "processing":
            # Another worker may be processing; retry shortly.
            raise RuntimeError(f"Turn is processing: {session_id} #{turn_number}")
        if status == "failed":
            raise PermanentJobError(f"Turn failed previously: {session_id} #{turn_number}")

    def _enqueue_session_summary_if_complete(self, session_id: str, expected_turns: int) -> None:
        try:
            completed = self.db.get_completed_turn_count(session_id, up_to=int(expected_turns))
            if completed >= int(expected_turns):
                self.db.enqueue_session_summary_job(session_id=session_id)
        except Exception as e:
            logger.warning(f"Failed to enqueue session summary after import for {session_id}: {e}")

    def _maybe_link_session_from_terminal(self, session_id: str) -> None:
        """Best-effort: backfill sessions.agent_id using terminal mapping."""
        def _agent_id_from_source(raw_source: object) -> str:
            source = str(raw_source or "").strip()
            if not source.startswith("agent:"):
                return ""
            return source[6:].strip()

        def _is_valid_agent_info_id(candidate: object) -> bool:
            agent_id = str(candidate or "").strip()
            if not agent_id:
                return False
            try:
                conn = self.db._get_connection()
                row = conn.execute(
                    "SELECT 1 FROM agent_info WHERE id = ? LIMIT 1",
                    (agent_id,),
                ).fetchone()
                return bool(row)
            except Exception:
                return False

        try:
            session = self.db.get_session_by_id(session_id)
            if session and getattr(session, "agent_id", None):
                return
        except Exception:
            return

        try:
            agents = self.db.list_agents(status=None, limit=1000)
        except Exception:
            return

        agent_info_id = None

        # Method 1: Match via agents table session_id (works for Codex)
        for agent in agents:
            try:
                if (agent.session_id or "").strip() != session_id:
                    continue
                agent_info_id = _agent_id_from_source(agent.source)
                if _is_valid_agent_info_id(agent_info_id):
                    break
                agent_info_id = None
            except Exception:
                continue

        # Method 2: Match via windowlink table (works for Claude).
        # Guard against polluted rows where windowlink.agent_id was incorrectly
        # written as terminal_id by validating against agent_info.
        if not agent_info_id:
            try:
                conn = self.db._get_connection()
                row = conn.execute(
                    """SELECT
                           CASE
                               WHEN w.agent_id IS NOT NULL
                                    AND w.agent_id != ''
                                    AND EXISTS (
                                        SELECT 1 FROM agent_info ai WHERE ai.id = w.agent_id
                                    ) THEN w.agent_id
                               WHEN w.source LIKE 'agent:%'
                                    AND length(substr(w.source, 7)) > 0
                                    AND EXISTS (
                                        SELECT 1 FROM agent_info ai
                                        WHERE ai.id = substr(w.source, 7)
                                    ) THEN substr(w.source, 7)
                               ELSE NULL
                           END AS resolved_agent_id
                       FROM windowlink w
                       WHERE w.session_id = ?
                       ORDER BY w.ts DESC, w.id DESC
                       LIMIT 1""",
                    (session_id,),
                ).fetchone()
                if row and row[0]:
                    agent_info_id = str(row[0]).strip()
            except Exception:
                pass

        if agent_info_id:
            try:
                self.db.update_session_agent_id(session_id, agent_info_id)
            except Exception:
                pass

    async def _process_session_summary_job(self, payload: Dict[str, Any]) -> bool:
        session_id = str(payload.get("session_id") or "")
        if not session_id:
            raise ValueError(f"Invalid session_summary payload: {payload}")

        from .events.session_summarizer import update_session_summary_now

        return bool(update_session_summary_now(self.db, session_id))

    async def _process_agent_description_job(self, payload: Dict[str, Any]) -> None:
        agent_id = str(payload.get("agent_id") or "")
        if not agent_id:
            raise ValueError(f"Invalid agent_description payload: {payload}")

        from .events.agent_summarizer import force_update_agent_description

        force_update_agent_description(self.db, agent_id)
