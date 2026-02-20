"""
Commit/turn processing pipeline shared by watcher + worker.

This module intentionally contains the "heavy" logic:
- Parse session files via triggers
- Extract turn content
- Call LLM (best-effort) to generate title/description
- Write turns + turn_content into SQLite
- Use DB-backed lease locks to prevent cross-process races

The watcher should only enqueue work; the worker should execute this pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional

from .config import ReAlignConfig
from .hooks import find_all_active_sessions
from .logging_config import setup_logger

logger = setup_logger("realign.commit_pipeline", "commit_pipeline.log")

SessionType = Literal["claude", "codex", "gemini", "unknown"]


class CommitPipeline:
    def __init__(
        self,
        *,
        config: ReAlignConfig | None = None,
        lock_owner_prefix: str = "worker",
        processing_turn_ttl_seconds: float = 20 * 60,
    ) -> None:
        self.config = config or ReAlignConfig.load()

        # Trigger support for pluggable turn detection
        from .triggers.registry import get_global_registry

        self.trigger_registry = get_global_registry()
        self.trigger_name = "next_turn"
        self._session_triggers: Dict[str, Any] = {}

        # Owner id for DB-backed lease locks (cross-process).
        try:
            from .db.locks import make_lock_owner

            self.lock_owner = make_lock_owner(str(lock_owner_prefix))
        except Exception:
            self.lock_owner = f"{lock_owner_prefix}:{os.getpid()}"

        # Per-turn "processing" TTL: if a processing placeholder exists longer than this,
        # a new run may take over and re-process it to avoid permanent stuck states.
        self.processing_turn_ttl_seconds = float(processing_turn_ttl_seconds)

    def _detect_session_type(self, session_file: Path) -> SessionType:
        """Detect the type of session file."""
        try:
            from .adapters import get_adapter_registry

            registry = get_adapter_registry()
            adapter = registry.auto_detect_adapter(session_file)
            if adapter:
                name = adapter.name
                if name in ["claude", "codex", "gemini"]:
                    return name
            return "unknown"
        except Exception as e:
            print(
                f"[Commit] Error detecting session type for {session_file.name}: {e}",
                file=sys.stderr,
            )
            return "unknown"

    def _get_trigger_for_session(self, session_file: Path):
        """Get or create the session trigger."""
        session_path = str(session_file)
        if session_path not in self._session_triggers:
            from .adapters import get_adapter_registry

            registry = get_adapter_registry()
            adapter = registry.auto_detect_adapter(session_file)
            if not adapter:
                logger.error(f"Unknown session type for {session_file.name}, cannot select trigger")
                return None
            self._session_triggers[session_path] = adapter.trigger
        return self._session_triggers[session_path]

    def _count_complete_turns(self, session_file: Path) -> int:
        """Unified interface to count complete dialogue turns for any session type."""
        trigger = self._get_trigger_for_session(session_file)
        if not trigger:
            return 0
        try:
            return int(trigger.count_complete_turns(session_file))
        except Exception as e:
            logger.error(f"Trigger error for {session_file.name}: {e}")
            return 0

    def _get_total_turn_count(self, session_file: Path) -> int:
        """Get total turns for a session file (including the last turn)."""
        try:
            trigger = self._get_trigger_for_session(session_file)
            if not trigger:
                return 0
            if hasattr(trigger, "get_detailed_analysis"):
                analysis = trigger.get_detailed_analysis(session_file)
                return int(analysis.get("total_turns", 0))
            return int(trigger.count_complete_turns(session_file)) + 1
        except Exception as e:
            logger.debug(f"Error getting total turn count for {session_file.name}: {e}")
            return 0

    def _find_latest_session(self, project_path: Path) -> Optional[Path]:
        """Find the most recently modified session file for this project."""
        try:
            session_files = find_all_active_sessions(self.config, project_path)
            if not session_files:
                return None
            return max(session_files, key=lambda f: f.stat().st_mtime)
        except Exception as e:
            logger.error(f"Failed to find latest session: {e}")
            return None

    def _handle_session_redaction(
        self, session_file: Path, project_path: Path, quiet: bool = False
    ) -> Path:
        """Check and redact sensitive information from session file (best-effort)."""
        if not self.config.redact_on_match:
            return session_file
        try:
            from .redactor import check_and_redact_session, save_original_session

            content = session_file.read_text(encoding="utf-8")
            redacted_content, has_secrets, secrets = check_and_redact_session(
                content, redact_mode="auto", quiet=quiet
            )
            if has_secrets:
                logger.warning(f"Secrets detected: {len(secrets)} secret(s)")
                backup_path = save_original_session(session_file, project_path)
                session_file.write_text(redacted_content, encoding="utf-8")
                logger.info(f"Session redacted, original saved to {backup_path}")
            return session_file
        except Exception as e:
            logger.error(f"Failed to redact session: {e}")
            return session_file

    def _get_current_turn_number(self, session_file: Path) -> int:
        return self._count_complete_turns(session_file)

    def _extract_user_message_for_turn(self, session_file: Path, turn_number: int) -> str:
        """Extract user message for a specific turn using the active trigger."""
        try:
            trigger = self._get_trigger_for_session(session_file)
            info = trigger.extract_turn_info(session_file, turn_number)
            if info and info.user_message:
                return info.user_message
        except Exception as e:
            logger.error(f"Failed to extract user message for turn {turn_number}: {e}")
        return "No user message found"

    def _extract_turn_content_by_number(self, session_file: Path, turn_number: int) -> str:
        """Extract content for a specific turn (supports JSONL and JSON formats)."""
        try:
            trigger = self._get_trigger_for_session(session_file)
            analysis = trigger.get_detailed_analysis(session_file)
            group = None
            for g in analysis.get("groups", []):
                if g.get("turn_number") == turn_number:
                    group = g
                    break
            if not group:
                return ""

            session_format = analysis.get("format", "")
            if session_format in ("gemini_json", "gemini"):
                turn_info = trigger.extract_turn_info(session_file, turn_number)
                if turn_info and turn_info.get("turn_content"):
                    return turn_info["turn_content"]
                return json.dumps(
                    {
                        "turn_number": turn_number,
                        "user_message": group.get("user_message", ""),
                        "assistant_response": group.get("summary_message", ""),
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            start_line = group.get("start_line") or (group.get("lines") or [None])[0]
            end_line = group.get("end_line") or (group.get("lines") or [None])[-1]
            if not start_line or not end_line:
                return ""

            lines: list[str] = []
            with open(session_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, 1):
                    if start_line <= idx <= end_line:
                        lines.append(line)
                    if idx > end_line:
                        break
            return "".join(lines)
        except Exception as e:
            logger.error(f"Failed to extract turn content for turn {turn_number}: {e}")
            print(f"[Commit] Failed to extract turn content: {e}", file=sys.stderr)
            return ""

    def _find_latest_structured_summary(self, session_file: Path) -> Optional[str]:
        """Find the latest agent-authored summary record in the session (Claude only)."""
        try:
            if session_file.is_dir():
                return None
            with open(session_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "summary":
                    summary = data.get("summary") or ""
                    if summary and summary.strip():
                        return summary.strip()
            return None
        except Exception as e:
            logger.error(f"Failed to find structured summary: {e}")
            return None

    def _extract_assistant_summary(self, session_file: Path) -> str:
        """Extract a summary of the assistant's response from session file."""
        try:
            if session_file.is_dir():
                # For directory sessions (Antigravity), we don't have a simple way to extract assistant summary
                # from a single file scan. Return generic message or use trigger if possible.
                return "Antigravity Session State"

            summary = self._find_latest_structured_summary(session_file)
            if summary:
                summary = summary.strip()
                return summary[:300] + ("..." if len(summary) > 300 else "")
        except Exception as e:
            logger.debug(f"Structured summary extraction failed: {e}")

        try:
            # Extract last assistant response text
            assistant_text = ""

            if session_file.is_dir():
                return "Antigravity Session"

            with open(session_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())

                        if data.get("type") == "assistant":
                            message = data.get("message", {})
                            content = message.get("content", [])

                            if isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        assistant_text = item.get("text", "")

                    except json.JSONDecodeError:
                        continue

            # Truncate to reasonable length
            if assistant_text:
                # Take first 300 characters as summary
                summary = assistant_text[:300]
                if len(assistant_text) > 300:
                    summary += "..."
                return summary
            else:
                return "Assistant response"

        except Exception as e:
            logger.error(f"Failed to extract assistant summary: {e}")
            return "Error extracting summary"

    def _extract_current_turn_content(self, session_file: Path) -> str:
        """Extract only the content for the current turn being committed (best-effort)."""
        try:
            lines: list[str] = []
            user_message_indices: list[int] = []

            if session_file.is_dir():
                trigger = self._get_trigger_for_session(session_file)
                if trigger:
                    turn = self._get_current_turn_number(session_file)
                    info = trigger.extract_turn_info(session_file, turn)
                    if info:
                        return info.user_message
                return ""

            with open(session_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    lines.append(line)
                    try:
                        data = json.loads(line.strip())
                        if data.get("type") == "user":
                            message = data.get("message", {})
                            content = message.get("content", "")
                            is_real_message = False
                            if isinstance(content, str):
                                if not content.startswith(
                                    "This session is being continued"
                                ) and not content.startswith("<ide_opened_file>"):
                                    is_real_message = True
                            elif isinstance(content, list):
                                text_parts = [
                                    item.get("text", "")
                                    for item in content
                                    if isinstance(item, dict) and item.get("type") == "text"
                                ]
                                if text_parts:
                                    combined_text = "\n".join(text_parts)
                                    if not combined_text.startswith(
                                        "This session is being continued"
                                    ) and not combined_text.startswith("<ide_opened_file>"):
                                        is_real_message = True
                            if is_real_message:
                                user_message_indices.append(idx)
                    except json.JSONDecodeError:
                        continue

            if len(user_message_indices) >= 2:
                start_idx = user_message_indices[-2]
                end_idx = user_message_indices[-1]
                turn_lines = lines[start_idx:end_idx]
            elif len(user_message_indices) == 1:
                start_idx = user_message_indices[0]
                turn_lines = lines[start_idx:]
            else:
                return ""

            return "".join(turn_lines)
        except Exception as e:
            logger.error(f"Failed to extract current turn content: {e}", exc_info=True)
            return ""

    def _generate_llm_summary(
        self,
        session_file: Optional[Path],
        turn_number: Optional[int] = None,
        turn_content: Optional[str] = None,
        user_message: Optional[str] = None,
        debug_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        session_id: Optional[str] = None,
    ) -> Optional[tuple[str, str, str, str, str]]:
        """
        Generate LLM-powered summary for the CURRENT TURN only.

        Priority:
        1. MCP Sampling API (if enabled and available)
        2. Direct Claude/OpenAI API calls (existing fallback)

        Returns:
            Tuple of (title, model_name, description, if_last_task, satisfaction), or None if LLM is disabled or fails
        """
        try:
            if not self.config.use_LLM:
                logger.debug("LLM summary disabled in config")
                return None

            if turn_number is None and session_file is not None:
                turn_number = self._get_current_turn_number(session_file)

            # Resolve session_id from file or parameter
            resolved_session_id = session_id
            if resolved_session_id is None and session_file is not None:
                resolved_session_id = session_file.stem

            recent_ctx = ""
            previous_records = []
            previous_commit_title = None
            try:
                # Get recent turns from database for context
                from .db import get_database

                db = get_database()
                session_id = resolved_session_id
                recent_turns = db.get_turns_for_session(session_id)
                if recent_turns:
                    # Get last 5 turn titles
                    for turn in recent_turns[-5:]:
                        if turn.llm_title:
                            previous_records.append(turn.llm_title)
                    # Get the most recent title
                    if previous_records:
                        previous_commit_title = previous_records[-1]
                        recent_ctx = "Recent turns:\n" + "\n".join(
                            f"- {t}" for t in previous_records
                        )
            except Exception:
                recent_ctx = ""
                previous_records = []

            # Extract full turn content first (includes all messages, thinking, etc.)
            if turn_content is None and session_file is not None:
                turn_content = self._extract_turn_content_by_number(session_file, turn_number)

            # Prefer trigger-derived fields: user_message + assistant summary + turn_status
            group = None
            if session_file is not None:
                try:
                    trigger = self._get_trigger_for_session(session_file)
                    analysis = trigger.get_detailed_analysis(session_file)
                    group = next(
                        (
                            g
                            for g in analysis.get("groups", [])
                            if g.get("turn_number") == turn_number
                        ),
                        None,
                    )
                except Exception:
                    group = None

            assistant_summary = None
            turn_status = "unknown"

            if group:
                if not user_message:
                    user_message = group.get("user_message") or user_message
                assistant_summary = group.get("summary_message") or assistant_summary
                turn_status = group.get("turn_status") or turn_status

            # Robust fallback for directory sessions (Antigravity) if group lookup failed
            if (
                session_file is not None
                and session_file.is_dir()
                and (not user_message or not assistant_summary)
            ):
                logger.info("Using fallback extraction for Antigravity directory session")
                print(
                    f"[Debug] Antigravity fallback: user_message={bool(user_message)}, assistant_summary={bool(assistant_summary)}",
                    file=sys.stderr,
                )
                if not user_message:
                    # For Antigravity, turn_content is essentially the user message (full state)
                    user_message = turn_content
                    print(
                        f"[Debug] Set user_message from turn_content: {len(user_message) if user_message else 0} chars",
                        file=sys.stderr,
                    )
                if not assistant_summary:
                    assistant_summary = "Antigravity Session State"
                turn_status = "completed"

            print(
                f"[Debug] Before LLM call: user_message={len(user_message) if user_message else 0} chars, assistant_summary={bool(assistant_summary)}",
                file=sys.stderr,
            )
            if user_message and assistant_summary:
                from .hooks import generate_summary_with_llm_from_turn_context

                # Pass full turn content to include all messages (user, assistant text, thinking)
                # but exclude tool use and code changes (handled by filter_session_content)
                title, model_name, description, if_last_task, satisfaction = (
                    generate_summary_with_llm_from_turn_context(
                        user_message=user_message,
                        assistant_summary=assistant_summary,
                        turn_status=turn_status,
                        recent_commit_context=recent_ctx,
                        provider=self.config.llm_provider,
                        previous_commit_title=previous_commit_title,
                        full_turn_content=turn_content,  # Pass full turn content
                        previous_records=previous_records,  # Pass extracted records from git history
                        debug_callback=debug_callback,  # Pass debug callback
                    )
                )

                if title:
                    logger.info(f"Generated LLM summary from turn context using {model_name}")
                    print(
                        f"[Watcher] ✓ Generated summary from turn context ({model_name})",
                        file=sys.stderr,
                    )
                    return (
                        title,
                        model_name or "unknown",
                        description or "",
                        if_last_task,
                        satisfaction,
                    )

                if session_file is not None and session_file.is_dir():
                    # Fallback if LLM fails for Antigravity
                    print(
                        f"[Watcher] ⚠ LLM summary failed/empty, using generic fallback for Antigravity",
                        file=sys.stderr,
                    )
                    return (
                        "Update Antigravity Brain",
                        "fallback",
                        "Automatic update of brain artifacts",
                        "yes",
                        "fine",
                    )

            # Fallback: Extract turn content and use the legacy pipeline
            if turn_content is None and session_file is not None:
                turn_content = self._extract_turn_content_by_number(session_file, turn_number)
            if not turn_content:
                logger.warning("No content found for current turn")
                return None

            if recent_ctx:
                try:
                    recent_line = json.dumps(
                        {
                            "type": "assistant",
                            "message": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Recent commit context:\n{recent_ctx}",
                                    }
                                ]
                            },
                        },
                        ensure_ascii=False,
                    )
                    if not turn_content.endswith("\n"):
                        turn_content += "\n"
                    turn_content += recent_line + "\n"
                except Exception:
                    pass

            # Use direct API calls for LLM summary
            from .hooks import generate_summary_with_llm

            title, model_name, description, if_last_task, satisfaction = generate_summary_with_llm(
                turn_content,
                max_chars=500,
                provider=self.config.llm_provider,
                previous_commit_title=previous_commit_title,
                debug_callback=debug_callback,
            )

            if title:
                if model_name:
                    logger.info(f"Generated LLM summary using {model_name}")
                    print(f"[Watcher] ✓ Generated LLM summary using {model_name}", file=sys.stderr)
                return (
                    title,
                    model_name or "unknown",
                    description or "",
                    if_last_task,
                    satisfaction,
                )
            else:
                logger.warning("LLM summary generation returned empty result")

                if session_file is not None and session_file.is_dir():
                    # Fallback if LLM fails for Antigravity (generic path)
                    print(
                        f"[Watcher] ⚠ LLM summary returned empty, using fallback for Antigravity",
                        file=sys.stderr,
                    )
                    return (
                        "Update Antigravity Brain",
                        "fallback",
                        "Automatic update of brain artifacts",
                        "yes",
                        "fine",
                    )

                return None

        except Exception as e:
            logger.error(f"Failed to generate LLM summary: {e}", exc_info=True)
            print(f"[Watcher] Failed to generate LLM summary: {e}", file=sys.stderr)

            # Record the error for later use in fallback logic
            from .hooks import set_last_llm_error

            set_last_llm_error(str(e))

            # Robust fallback for Antigravity directory sessions if anything fails
            if session_file is not None and session_file.is_dir():
                print(
                    f"[Watcher] ⚠ Using generic fallback after exception for Antigravity",
                    file=sys.stderr,
                )
                return (
                    "Update Antigravity Brain",
                    "fallback",
                    "Automatic update of brain artifacts",
                    "yes",
                    "fine",
                )

            return None

    def _run_realign_commit(
        self,
        project_path: Path,
        session_file: Optional[Path] = None,
        target_turn: Optional[int] = None,
        turn_content: Optional[str] = None,
        user_message_override: Optional[str] = None,
        quiet: bool = False,
        debug_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        skip_dedup: bool = False,
        skip_session_summary: bool = False,
        no_track: bool = False,
    ) -> bool:
        """
        Execute commit with DB-backed lease locking to prevent cross-process races.

        Args:
            project_path: Path to the project directory
            quiet: If True, suppress console output

        Returns:
            True if commit was created, False otherwise

        The method will:
        - Acquire a DB lease lock to prevent concurrent commits across processes
        - Generate LLM-powered semantic commit message
        - Create DB record
        """
        try:
            from .db import get_database
            from .db.locks import lease_lock, lock_key_for_project_commit

            db = get_database()
            lock_key = lock_key_for_project_commit(project_path)

            with lease_lock(
                db,
                lock_key,
                owner=self.lock_owner,
                ttl_seconds=30 * 60,  # 30 minutes
                wait_timeout_seconds=5.0,
            ) as acquired:
                if not acquired:
                    print(
                        f"[Watcher] Another process is committing to {project_path.name}, skipping",
                        file=sys.stderr,
                    )
                    return False

                return self._do_commit_locked(
                    project_path,
                    session_file=session_file,
                    target_turn=target_turn,
                    turn_content=turn_content,
                    user_message_override=user_message_override,
                    quiet=quiet,
                    debug_callback=debug_callback,
                    skip_dedup=skip_dedup,
                    skip_session_summary=skip_session_summary,
                    no_track=no_track,
                )
        except Exception as e:
            print(f"[Watcher] Commit error: {e}", file=sys.stderr)
            return False

    def _run_realign_commit_batch(
        self,
        project_path: Path,
        *,
        session_file: Path,
        target_turns: list[int],
        quiet: bool = False,
        skip_dedup: bool = False,
        skip_session_summary: bool = False,
        no_track: bool = False,
    ) -> dict[int, bool]:
        """
        Batch commit multiple target turns under a single project lease lock.

        This reduces overhead for session_process jobs that need to backfill many turns.
        """
        results: dict[int, bool] = {}
        turns = [int(t) for t in (target_turns or []) if int(t) > 0]
        if not turns:
            return results

        try:
            from .db import get_database
            from .db.locks import lease_lock, lock_key_for_project_commit

            db = get_database()
            lock_key = lock_key_for_project_commit(project_path)

            with lease_lock(
                db,
                lock_key,
                owner=self.lock_owner,
                ttl_seconds=30 * 60,
                wait_timeout_seconds=5.0,
            ) as acquired:
                if not acquired:
                    print(
                        f"[Watcher] Another process is committing to {project_path.name}, skipping",
                        file=sys.stderr,
                    )
                    return results

                for t in turns:
                    try:
                        ok = self._do_commit_locked(
                            project_path,
                            session_file=session_file,
                            target_turn=int(t),
                            quiet=quiet,
                            skip_dedup=skip_dedup,
                            skip_session_summary=skip_session_summary,
                            no_track=no_track,
                        )
                        results[int(t)] = bool(ok)
                    except Exception:
                        results[int(t)] = False
                return results
        except Exception:
            return results

    def _do_commit_locked(
        self,
        project_path: Path,
        session_file: Optional[Path] = None,
        target_turn: Optional[int] = None,
        turn_content: Optional[str] = None,
        user_message_override: Optional[str] = None,
        quiet: bool = False,
        debug_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        skip_dedup: bool = False,
        skip_session_summary: bool = False,
        no_track: bool = False,
    ) -> bool:
        """
        Perform the actual commit operation to SQLite database.

        This method:
        1. Finds the latest session file for the project
        2. Redacts sensitive information from the session
        3. Generates LLM-powered semantic commit message
        4. Creates DB record

        Args:
            project_path: Path to the project directory
            session_file: Target session file (if None, will locate latest)
            target_turn: If provided, commit this specific turn
            turn_content: Optional precomputed turn content
            user_message_override: Optional precomputed user message
            quiet: If True, suppress console output

        Returns:
            True if commit was created, False otherwise
        """
        try:
            # Find the latest session file for this project if not provided
            if not session_file:
                session_file = self._find_latest_session(project_path)

            if not session_file:
                logger.warning("No session file found for commit")
                return False

            # Redact sensitive information from session file before committing
            session_file = self._handle_session_redaction(session_file, project_path, quiet=quiet)

            # Extract session information
            session_id = session_file.stem  # e.g., "minhao_claude_abc123"
            turn_number = target_turn or self._get_current_turn_number(session_file)
            user_message = user_message_override or self._extract_user_message_for_turn(
                session_file, turn_number
            )

            # V9: Get user identity for creator tracking
            from .config import ReAlignConfig

            config = ReAlignConfig.load()

            # Compute hash of current turn content (not the whole session file)
            if not turn_content:
                turn_content = self._extract_turn_content_by_number(session_file, turn_number)

            turn_hash = hashlib.md5((turn_content or "").encode("utf-8")).hexdigest()

            # SQLite Storage (authoritative): dedupe by (session_id, turn_number)
            from .db import get_database
            from .db.base import TurnRecord
            import uuid

            db = get_database()

            file_stat = session_file.stat()
            file_created = datetime.fromtimestamp(
                getattr(file_stat, "st_birthtime", file_stat.st_ctime)
            )
            session = db.get_or_create_session(
                session_id=session_id,
                session_file_path=session_file,
                session_type=self._detect_session_type(session_file),
                started_at=file_created,
                workspace_path=str(project_path) if project_path else None,
            )

            # Check no_track from parameter or existing session metadata (polling path)
            is_no_track = no_track
            if not is_no_track and session:
                session_meta = getattr(session, "metadata", None) or {}
                is_no_track = bool(session_meta.get("no_track", False))

            # Store no_track flag in session metadata if applicable
            if is_no_track:
                try:
                    db.update_session_metadata_flag(session_id, "no_track", True)
                except Exception:
                    pass

            takeover_attempt = False
            existing_turn = db.get_turn_by_number(session_id, turn_number)
            if existing_turn and not skip_dedup:
                existing_status = getattr(existing_turn, "turn_status", None)
                if existing_status in (None, "completed"):
                    logger.info(f"Turn already exists in DB: {session_id} #{turn_number}, skipping")
                    return False

                if existing_status == "processing":
                    # If a processing placeholder exists, avoid duplicate LLM calls unless it's stale.
                    try:
                        age_seconds = max(
                            0.0,
                            (
                                datetime.now()
                                - getattr(existing_turn, "created_at", datetime.now())
                            ).total_seconds(),
                        )
                    except Exception:
                        age_seconds = 0.0

                    if age_seconds < float(self.processing_turn_ttl_seconds):
                        logger.info(
                            f"Turn is already processing in DB: {session_id} #{turn_number} ({age_seconds:.0f}s), skipping"
                        )
                        return False

                    logger.warning(
                        f"Processing turn appears stale: {session_id} #{turn_number} ({age_seconds:.0f}s), taking over"
                    )
                    takeover_attempt = True

                if existing_status == "failed":
                    logger.warning(f"Turn previously failed: {session_id} #{turn_number}, skipping")
                    return False

            # Insert a processing placeholder BEFORE calling LLM so we can reflect runtime status
            # and avoid duplicate work in crash/restart scenarios.
            placeholder_hash = hashlib.md5(
                f"processing:{session_id}:{turn_number}:{time.time()}".encode("utf-8")
            ).hexdigest()
            processing_created_at = datetime.now()
            processing_turn = TurnRecord(
                id=str(uuid.uuid4()),
                session_id=session_id,
                turn_number=turn_number,
                user_message=user_message,
                assistant_summary=None,
                turn_status="processing",
                llm_title="running...",
                llm_description=None,
                model_name=None,
                if_last_task="unknown",
                satisfaction="unknown",
                content_hash=placeholder_hash,
                timestamp=processing_created_at,
                created_at=processing_created_at,
                git_commit_hash=None,
            )
            try:
                db.create_turn(processing_turn, content="")
            except Exception as e:
                # If we fail to store processing state, continue anyway (best-effort).
                logger.debug(f"Failed to write processing placeholder: {e}")

            try:
                # Skip LLM call for no-track mode
                if is_no_track:
                    llm_result = ("No Track", None, "No Track", "no", "fine")
                    logger.info(f"No-track mode: skipping LLM for {session_id} turn {turn_number}")
                else:
                    # Generate LLM summary with fallback for errors
                    llm_result = self._generate_llm_summary(
                        session_file,
                        turn_number=turn_number,
                        turn_content=turn_content,
                        user_message=user_message,
                        debug_callback=debug_callback,
                    )

                if not llm_result:
                    # LLM summary failed, use error marker to continue commit
                    logger.warning(
                        f"LLM summary generation failed for {session_file.name} turn {turn_number} - using error marker"
                    )
                    print(
                        f"[Watcher] ⚠ LLM API unavailable - using error marker for commit",
                        file=sys.stderr,
                    )

                    # Check if it's an API key problem
                    from .hooks import get_last_llm_error

                    last_error = get_last_llm_error()
                    if last_error:
                        if "API_KEY not set" in last_error or "api_key" in last_error.lower():
                            print(
                                f"[Watcher] ⓘ Configure API keys in Acme Settings to enable LLM summaries",
                                file=sys.stderr,
                            )
                        else:
                            print(f"[Watcher] ⓘ LLM Error: {last_error[:100]}", file=sys.stderr)

                    # Use explicit error marker
                    title = "⚠ LLM API Error - Summary unavailable"
                    model_name = "error-fallback"
                    description = f"LLM API failed. Error: {last_error[:200] if last_error else 'Unknown error'}"
                    if_last_task = "unknown"
                    satisfaction = "unknown"

                    llm_result = (title, model_name, description, if_last_task, satisfaction)

                title, model_name, description, if_last_task, satisfaction = llm_result

                # Validate title - reject if it's empty, too short, or looks like truncated JSON
                if not title or len(title.strip()) < 2:
                    logger.error(f"Invalid LLM title generated: '{title}' - skipping commit")
                    print(f"[Watcher] ✗ Invalid commit message title: '{title}'", file=sys.stderr)
                    raise RuntimeError(f"Invalid LLM title: {title!r}")

                if (
                    title.strip() in ["{", "}", "[", "]"]
                    or title.startswith("{")
                    and not title.endswith("}")
                ):
                    logger.error(f"Title appears to be truncated JSON: '{title}' - skipping commit")
                    print(f"[Watcher] ✗ Truncated JSON in title: '{title}'", file=sys.stderr)
                    raise RuntimeError(f"Truncated JSON title: {title!r}")

                logger.info(f"Committing turn {turn_number} for session {session_id}")
                new_turn = TurnRecord(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    turn_number=turn_number,
                    user_message=user_message,
                    assistant_summary=description,
                    turn_status="completed",
                    llm_title=title,
                    llm_description=description,
                    model_name=model_name,
                    if_last_task=if_last_task,
                    satisfaction=satisfaction,
                    content_hash=turn_hash,
                    timestamp=datetime.now(),
                    created_at=datetime.now(),
                    git_commit_hash=None,
                )
                db.create_turn(
                    new_turn,
                    content=turn_content or "",
                    skip_session_summary=skip_session_summary,
                )
                logger.info(f"✓ Saved turn {turn_number} to SQLite DB")
                print(f"[Watcher] ✓ Saved turn {turn_number} to SQLite DB", file=sys.stderr)
                return True
            except Exception as e:
                # If we were taking over a stale processing turn, a failure here should stop further retries.
                if takeover_attempt:
                    logger.error(
                        f"Takeover attempt failed for {session_id} #{turn_number}: {e}",
                        exc_info=True,
                    )
                    failed_turn = TurnRecord(
                        id=str(uuid.uuid4()),
                        session_id=session_id,
                        turn_number=turn_number,
                        user_message=user_message,
                        assistant_summary=None,
                        turn_status="failed",
                        llm_title="failed",
                        llm_description=str(e)[:2000],
                        model_name=None,
                        if_last_task="unknown",
                        satisfaction="unknown",
                        content_hash=placeholder_hash,
                        timestamp=datetime.now(),
                        created_at=processing_created_at,
                        git_commit_hash=None,
                    )
                    try:
                        db.create_turn(
                            failed_turn,
                            content="",
                            skip_session_summary=skip_session_summary,
                        )
                    except Exception:
                        pass

                logger.error(f"Failed to write to SQLite DB: {e}", exc_info=True)
                print(f"[Watcher] ⚠ Failed to write to SQLite DB: {e}", file=sys.stderr)
                return False

        except Exception as e:
            logger.error(f"Commit error for {project_path.name}: {e}", exc_info=True)
            print(f"[Watcher] Commit error for {project_path.name}: {e}", file=sys.stderr)
            return False
