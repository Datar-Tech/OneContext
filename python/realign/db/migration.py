"""
Migration utilities for converting git-based storage to SQLite.
"""

import re
import subprocess
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from .base import TurnRecord
from .sqlite_db import SQLiteDatabase

logger = logging.getLogger(__name__)


def parse_commit_message(message: str) -> Dict[str, Any]:
    """
    Parse a commit message to extract structured data.

    Expected format:
    {llm_title}

    {llm_description}

    ---
    Session: {session_id} | Turn: #{turn_number}
    Request: {user_message}

    Returns:
        Dictionary with extracted fields
    """
    result = {
        "llm_title": "",
        "llm_description": "",
        "session_id": "",
        "turn_number": 0,
        "user_message": "",
    }

    if not message:
        return result

    lines = message.strip().split("\n")
    if not lines:
        return result

    # First line is the title
    result["llm_title"] = lines[0].strip()

    # Find the --- separator
    separator_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == "---":
            separator_idx = i
            break

    if separator_idx > 1:
        # Description is between title and separator
        description_lines = lines[1:separator_idx]
        result["llm_description"] = "\n".join(description_lines).strip()

    # Parse metadata after separator
    if separator_idx >= 0:
        metadata_lines = lines[separator_idx + 1 :]
        for line in metadata_lines:
            line = line.strip()

            # Parse "Session: xxx | Turn: #N"
            session_match = re.match(r"Session:\s*(.+?)\s*\|\s*Turn:\s*#?(\d+)", line)
            if session_match:
                result["session_id"] = session_match.group(1).strip()
                result["turn_number"] = int(session_match.group(2))
                continue

            # Parse "Request: xxx"
            if line.startswith("Request:"):
                result["user_message"] = line[8:].strip()

    # Fallback: try legacy format "Session xxx, Turn N: ..."
    if not result["session_id"] and result["llm_title"]:
        legacy_match = re.match(r"Session\s+(\S+),\s*Turn\s+(\d+):", result["llm_title"])
        if legacy_match:
            result["session_id"] = legacy_match.group(1)
            result["turn_number"] = int(legacy_match.group(2))

    return result


def get_all_commits(realign_dir: Path) -> List[Dict[str, Any]]:
    """
    Get all commits from a .aline git repository.

    Args:
        realign_dir: Path to .aline directory

    Returns:
        List of commit dictionaries with hash, timestamp, and message
    """
    if not (realign_dir / ".git").exists():
        logger.warning(f"No git repo found at {realign_dir}")
        return []

    try:
        # Use record separator for robust parsing
        rs = "\x1e"
        us = "\x1f"
        fmt = f"%H{us}%at{us}%B{rs}"

        result = subprocess.run(
            ["git", "log", "--reverse", f"--pretty=format:{fmt}"],
            cwd=realign_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"git log failed: {result.stderr}")
            return []

        commits = []
        chunks = [c for c in result.stdout.split(rs) if c.strip()]

        for chunk in chunks:
            parts = chunk.split(us)
            if len(parts) < 3:
                continue

            commit_hash = parts[0].strip()
            timestamp = parts[1].strip()
            message = parts[2].strip()

            if not commit_hash:
                continue

            # Skip initial commit
            if message.lower().startswith("initial commit"):
                continue

            commits.append(
                {
                    "hash": commit_hash,
                    "timestamp": datetime.fromtimestamp(int(timestamp)),
                    "message": message,
                }
            )

        return commits

    except Exception as e:
        logger.error(f"Failed to get commits: {e}", exc_info=True)
        return []


def migrate_project(
    project_path: Path,
    db: SQLiteDatabase,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Migrate a project's git history to SQLite database.

    Args:
        project_path: Path to the project directory
        db: SQLite database instance
        dry_run: If True, only report what would be migrated

    Returns:
        Tuple of (migrated_count, skipped_count)
    """
    from realign import get_realign_dir

    realign_dir = get_realign_dir(project_path)

    if not realign_dir.exists():
        logger.warning(f"No .aline directory found for {project_path}")
        return (0, 0)

    commits = get_all_commits(realign_dir)
    if not commits:
        logger.info(f"No commits found for {project_path}")
        return (0, 0)

    logger.info(f"Found {len(commits)} commits for {project_path.name}")

    if dry_run:
        for commit in commits:
            parsed = parse_commit_message(commit["message"])
            logger.info(f"Would migrate: {commit['hash'][:8]} - {parsed['llm_title'][:50]}")
        return (len(commits), 0)

    # Get or create project
    project_rec = db.get_or_create_project(project_path)

    migrated = 0
    skipped = 0

    # Group commits by session
    session_commits: Dict[str, List[Dict]] = {}
    for commit in commits:
        parsed = parse_commit_message(commit["message"])
        session_id = parsed.get("session_id", "")

        if not session_id:
            logger.debug(f"Skipping commit without session: {commit['hash'][:8]}")
            skipped += 1
            continue

        if session_id not in session_commits:
            session_commits[session_id] = []
        session_commits[session_id].append({**commit, "parsed": parsed})

    # Process each session
    for session_id, session_data in session_commits.items():
        if not session_data:
            continue

        # Create session record
        first_commit = session_data[0]
        session_rec = db.get_or_create_session(
            session_id=session_id,
            session_file_path=Path(f"~/.claude/projects/{project_path.name}/{session_id}.jsonl"),
            session_type="claude",  # Default assumption
            started_at=first_commit["timestamp"],
            workspace_path=str(project_path),
        )

        # Create turn records
        for commit_data in session_data:
            parsed = commit_data["parsed"]
            turn_number = parsed.get("turn_number", 0)

            if turn_number == 0:
                skipped += 1
                continue

            # Generate content hash from commit message (since we don't have original content)
            content_hash = hashlib.md5(commit_data["message"].encode()).hexdigest()

            # Check if already migrated
            existing = db.get_turn_by_hash(session_id, content_hash)
            if existing:
                logger.debug(f"Turn already exists: {session_id} #{turn_number}")
                skipped += 1
                continue

            try:
                import uuid

                turn = TurnRecord(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    turn_number=turn_number,
                    user_message=parsed.get("user_message", ""),
                    assistant_summary=parsed.get("llm_description", ""),
                    turn_status="completed",
                    llm_title=parsed.get("llm_title", "Migrated commit"),
                    llm_description=parsed.get("llm_description", ""),
                    model_name="migrated",
                    if_last_task="unknown",
                    satisfaction="unknown",
                    content_hash=content_hash,
                    timestamp=commit_data["timestamp"],
                    created_at=datetime.now(),
                    git_commit_hash=commit_data["hash"],
                )

                # We don't have the original turn content, so store the commit message
                db.create_turn(turn, content=commit_data["message"])
                migrated += 1
                logger.debug(f"Migrated: {session_id} #{turn_number}")

            except Exception as e:
                logger.error(f"Failed to migrate turn: {e}")
                skipped += 1

    logger.info(
        f"Migration complete for {project_path.name}: {migrated} migrated, {skipped} skipped"
    )
    return (migrated, skipped)


def migrate_all_projects(
    db: Optional[SQLiteDatabase] = None,
    dry_run: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """
    Migrate all projects in ~/.aline/ to SQLite.

    Args:
        db: SQLite database instance (creates one if not provided)
        dry_run: If True, only report what would be migrated

    Returns:
        Dictionary mapping project name to (migrated, skipped) counts
    """
    from . import get_database

    if db is None:
        db = get_database()

    aline_base = Path.home() / ".aline"
    if not aline_base.exists():
        logger.info("No .aline directory found")
        return {}

    results = {}

    # Find all project directories (those with .git subdirectory)
    for item in aline_base.iterdir():
        if not item.is_dir():
            continue
        if item.name in ("db", "logs", "cache"):
            continue
        if not (item / ".git").exists():
            continue

        # This is a project's .aline directory
        # The actual project path needs to be reconstructed
        # For now, assume project name matches directory name
        project_name = item.name

        # Try to find the actual project path from stored metadata or guess
        # In practice, we might need to store the original project path
        logger.info(f"Migrating project: {project_name}")

        # Create a placeholder project path
        # The real path should be stored in project metadata
        project_path = Path.home() / "Projects" / project_name

        # Use the .aline directory directly for migration
        migrated, skipped = migrate_project_from_realign_dir(item, db, dry_run)
        results[project_name] = (migrated, skipped)

    return results


def migrate_project_from_realign_dir(
    realign_dir: Path,
    db: SQLiteDatabase,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Migrate from a .aline directory directly.

    Args:
        realign_dir: Path to .aline directory
        db: SQLite database instance
        dry_run: If True, only report what would be migrated

    Returns:
        Tuple of (migrated_count, skipped_count)
    """
    if not (realign_dir / ".git").exists():
        logger.warning(f"No git repo at {realign_dir}")
        return (0, 0)

    commits = get_all_commits(realign_dir)
    if not commits:
        return (0, 0)

    logger.info(f"Found {len(commits)} commits in {realign_dir.name}")

    if dry_run:
        for commit in commits:
            parsed = parse_commit_message(commit["message"])
            title = parsed.get("llm_title", "No title")[:50]
            logger.info(f"  Would migrate: {commit['hash'][:8]} - {title}")
        return (len(commits), 0)

    # Get or create project (use realign_dir name as project name)
    project_name = realign_dir.name
    project_path = Path.home() / "Projects" / project_name  # Placeholder
    project_rec = db.get_or_create_project(project_path, name=project_name)

    migrated = 0
    skipped = 0

    for commit in commits:
        parsed = parse_commit_message(commit["message"])
        session_id = parsed.get("session_id", "")
        turn_number = parsed.get("turn_number", 0)

        if not session_id or turn_number == 0:
            skipped += 1
            continue

        # Ensure session exists
        session_rec = db.get_or_create_session(
            session_id=session_id,
            session_file_path=Path(f"migrated/{session_id}.jsonl"),
            session_type="claude",
            started_at=commit["timestamp"],
            workspace_path=str(project_path),
        )

        content_hash = hashlib.md5(commit["message"].encode()).hexdigest()

        if db.get_turn_by_hash(session_id, content_hash):
            skipped += 1
            continue

        try:
            import uuid

            turn = TurnRecord(
                id=str(uuid.uuid4()),
                session_id=session_id,
                turn_number=turn_number,
                user_message=parsed.get("user_message", ""),
                assistant_summary=parsed.get("llm_description", ""),
                turn_status="completed",
                llm_title=parsed.get("llm_title", "Migrated"),
                llm_description=parsed.get("llm_description", ""),
                model_name="migrated",
                if_last_task="unknown",
                satisfaction="unknown",
                content_hash=content_hash,
                timestamp=commit["timestamp"],
                created_at=datetime.now(),
                git_commit_hash=commit["hash"],
            )

            db.create_turn(turn, content=commit["message"])
            migrated += 1

        except Exception as e:
            logger.error(f"Migration error: {e}")
            skipped += 1

    return (migrated, skipped)
