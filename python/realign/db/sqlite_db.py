"""
SQLite implementation of the ReAlign database interface.
"""

import re
import sqlite3
import json
import logging
import os
import uuid
import time
import threading
import stat
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from .base import (
    DatabaseInterface,
    ProjectRecord,
    SessionRecord,
    TurnRecord,
    EventRecord,
    LockRecord,
    AgentRecord,
    AgentInfoRecord,
    AgentContextRecord,
    WindowLinkRecord,
    UserRecord,
    MAX_SESSION_NUM,
    AgentSessionLimitExceededError,
)
from .schema import (
    INIT_SCRIPTS,
    SCHEMA_VERSION,
    FTS_EVENTS_SCRIPTS,
    get_migration_scripts,
)

from ..logging_config import setup_logger

logger = setup_logger(__name__, "dashboard.log")

_LAST_SQLITE_CANTOPEN_LOG_AT: float = 0.0
_SQLITE_CANTOPEN_LOG_THROTTLE_SECONDS = 30.0


def _log_sqlite_cantopen(db_path: Path, *, read_only: bool, err: BaseException) -> None:
    """Best-effort logging for intermittent SQLite open failures (throttled)."""
    global _LAST_SQLITE_CANTOPEN_LOG_AT
    now = time.time()
    if (now - _LAST_SQLITE_CANTOPEN_LOG_AT) < _SQLITE_CANTOPEN_LOG_THROTTLE_SECONDS:
        return
    _LAST_SQLITE_CANTOPEN_LOG_AT = now

    def _path_info(path: Path) -> dict[str, object]:
        out: dict[str, object] = {
            "path": str(path),
            "exists": path.exists(),
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "is_symlink": path.is_symlink(),
            "readable": os.access(path, os.R_OK),
            "writable": os.access(path, os.W_OK),
            "executable": os.access(path, os.X_OK),
        }
        try:
            st = path.stat()
            out.update(
                {
                    "mode": oct(stat.S_IMODE(st.st_mode)),
                    "uid": int(st.st_uid),
                    "gid": int(st.st_gid),
                    "inode": int(getattr(st, "st_ino", 0) or 0),
                    "size_bytes": int(getattr(st, "st_size", 0) or 0),
                    "mtime": float(getattr(st, "st_mtime", 0.0) or 0.0),
                }
            )
        except Exception as stat_err:
            out["stat_error"] = str(stat_err)
        return out

    def _probe_create_unlink(dir_path: Path) -> dict[str, object]:
        """Probe whether we can create and remove a file in dir_path (best-effort)."""
        probe = dir_path / ".aline_sqlite_cantopen_probe"
        out: dict[str, object] = {"path": str(probe), "ok": False}
        try:
            fd = os.open(str(probe), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            try:
                probe.unlink()
            except Exception:
                out["unlink_failed"] = True
            out["ok"] = True
            return out
        except OSError as oe:
            out.update({"errno": int(getattr(oe, "errno", 0) or 0), "error": str(oe)})
            return out
        except Exception as e:
            out["error"] = str(e)
            return out

    wal_path = Path(str(db_path) + "-wal")
    shm_path = Path(str(db_path) + "-shm")

    info: dict[str, object] = {
        "read_only": bool(read_only),
        "pid": int(os.getpid()),
        "uid": int(os.getuid()),
        "gid": int(os.getgid()),
        "euid": int(os.geteuid()),
        "egid": int(os.getegid()),
        "sqlite_version": getattr(sqlite3, "sqlite_version", ""),
        "db": _path_info(db_path),
        "db_parent": _path_info(db_path.parent),
        "db_parent_probe_create": _probe_create_unlink(db_path.parent),
        "wal": _path_info(wal_path),
        "shm": _path_info(shm_path),
    }

    try:
        logger.warning("SQLite connect failed: %s (info=%s)", err, info)
    except Exception:
        return


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat()


def _truncate_error(text: Optional[str], limit: int = 2000) -> Optional[str]:
    if text is None:
        return None
    s = str(text)
    return s if len(s) <= limit else (s[: limit - 3] + "...")


def _sqlite_dt(dt: datetime) -> str:
    """SQLite-friendly UTC timestamp (matches datetime('now') lexical ordering)."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _env_truthy(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _log_lock_operation(
    operation: str,
    lock_key: str,
    owner: str,
    success: bool,
    ttl_seconds: Optional[float] = None,
):
    # Lock operation file logging is intentionally disabled to keep only dashboard.log in ~/.aline/.logs.
    return


class SQLiteDatabase(DatabaseInterface):
    def __init__(
        self,
        db_path: str,
        *,
        read_only: bool = False,
        connect_timeout_seconds: float = 5.0,
    ):
        self.db_path = Path(db_path).expanduser()
        # Important: this DB object is a process-wide singleton in some contexts and is accessed
        # from Textual worker threads. `sqlite3.Connection` objects are not safe for concurrent
        # use across threads, even when `check_same_thread=False`. Keep one connection per thread.
        self._connections: dict[int, sqlite3.Connection] = {}
        self._connections_lock = threading.Lock()
        self.read_only = bool(read_only)
        self.connect_timeout_seconds = float(connect_timeout_seconds)
        self.ensure_db_dir()

    def ensure_db_dir(self):
        """Ensure database directory exists."""
        if self.read_only:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection for the current thread."""
        thread_id = threading.get_ident()
        conn = self._connections.get(thread_id)
        if conn is not None:
            return conn

        with self._connections_lock:
            conn = self._connections.get(thread_id)
            if conn is not None:
                return conn

            timeout = max(0.0, float(self.connect_timeout_seconds))
            ro_fallback = False
            if self.read_only:
                # Read-only connections should not block the CLI under worker/watcher write load.
                # Also avoid write PRAGMAs (e.g., journal_mode) that can require exclusive locks.
                uri = f"file:{self.db_path}?mode=ro"
                try:
                    conn = sqlite3.connect(
                        uri,
                        uri=True,
                        timeout=timeout,
                        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                        check_same_thread=False,
                    )
                except sqlite3.OperationalError as e:
                    # Some environments intermittently fail to open a WAL-mode DB with `mode=ro`.
                    # Capture high-signal diagnostics and fall back to a normal connection guarded
                    # by `query_only` to preserve read-only semantics.
                    _log_sqlite_cantopen(self.db_path, read_only=True, err=e)
                    if not self.db_path.exists():
                        raise
                    ro_fallback = True
                    conn = sqlite3.connect(
                        str(self.db_path),
                        timeout=timeout,
                        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                        check_same_thread=False,
                    )
            else:
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=timeout,
                    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                    check_same_thread=False,
                )
            conn.row_factory = sqlite3.Row

            if self.read_only and ro_fallback:
                try:
                    conn.execute("PRAGMA query_only=ON;")
                except Exception:
                    pass
                try:
                    conn.execute("PRAGMA foreign_keys=ON;")
                except Exception:
                    pass

            if not self.read_only:
                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA foreign_keys=ON;")
                # Wait briefly on writer contention instead of failing immediately.
                # Prevents transient "database is locked" during concurrent watcher/worker writes.
                conn.execute("PRAGMA busy_timeout=5000;")

            # Register REGEXP function for regex search support
            self._register_regexp_function(conn)

            self._connections[thread_id] = conn
            return conn

    def _register_regexp_function(self, conn: sqlite3.Connection) -> None:
        """Register REGEXP function for regex searches in SQLite."""

        def regexp(pattern: str, value: str) -> bool:
            if value is None:
                return False
            try:
                return re.search(pattern, value) is not None
            except re.error:
                return False

        conn.create_function("REGEXP", 2, regexp)

    def initialize(self) -> bool:
        """Initialize database schema with migration support."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Check current version first (if schema_version table exists)
            current_version = 0
            try:
                cursor.execute("SELECT MAX(version) FROM schema_version")
                result = cursor.fetchone()
                current_version = result[0] if result and result[0] is not None else 0
            except sqlite3.OperationalError:
                # Table doesn't exist yet, will be created by INIT_SCRIPTS
                pass

            # For fresh databases, execute init scripts
            if current_version == 0:
                for script in INIT_SCRIPTS:
                    cursor.execute(script)
                cursor.execute(
                    "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                    (SCHEMA_VERSION, "Initial schema V2"),
                )
            # For existing databases, run migrations
            elif current_version < SCHEMA_VERSION:
                migration_scripts = get_migration_scripts(current_version, SCHEMA_VERSION)
                for script in migration_scripts:
                    try:
                        cursor.execute(script)
                    except sqlite3.OperationalError as e:
                        # Some migrations may fail if column already exists, etc.
                        logger.debug(f"Migration script skipped (may already be applied): {e}")

                cursor.execute(
                    "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                    (
                        SCHEMA_VERSION,
                        f"Migrated from V{current_version} to V{SCHEMA_VERSION}",
                    ),
                )

            # Self-heal: some historical DBs reported schema V28 but missed agent_sessions.
            # Ensure the V28 mapping table and indexes exist, and backfill from sessions.agent_id.
            try:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_sessions (
                        agent_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        added_at TEXT DEFAULT (datetime('now')),
                        PRIMARY KEY (agent_id, session_id)
                    )
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_sessions_agent ON agent_sessions(agent_id);"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_sessions_session ON agent_sessions(session_id);"
                )
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO agent_sessions (agent_id, session_id)
                    SELECT agent_id, id
                    FROM sessions
                    WHERE agent_id IS NOT NULL
                      AND TRIM(agent_id) <> ''
                    """
                )
            except sqlite3.OperationalError as e:
                logger.debug(f"agent_sessions self-heal skipped: {e}")

            # Ensure FTS exists even for migrated DBs, and rebuild the index so it's in sync.
            try:
                for script in FTS_EVENTS_SCRIPTS:
                    cursor.execute(script)
                cursor.execute("INSERT INTO fts_events(fts_events) VALUES('rebuild');")
            except sqlite3.OperationalError as e:
                logger.debug(f"FTS events setup skipped: {e}")

            conn.commit()
            logger.info(f"Database initialized at version {SCHEMA_VERSION}")
            return True
        except Exception as e:
            logger.error(f"Database initialization failed: {e}", exc_info=True)
            return False

    def _get_project_by_path(self, path: Path) -> Optional[ProjectRecord]:
        """Get project by path, or None if not found."""
        conn = self._get_connection()
        cursor = conn.cursor()
        path_str = str(path.absolute())
        cursor.execute("SELECT * FROM projects WHERE path = ?", (path_str,))
        row = cursor.fetchone()
        if row:
            return self._row_to_project(row)
        return None

    def get_project_by_path(self, path: Path) -> Optional[ProjectRecord]:
        """Public wrapper for fetching a project by path."""
        return self._get_project_by_path(path)

    def get_or_create_project(self, path: Path, name: Optional[str] = None) -> ProjectRecord:
        """Get existing project or create new one."""
        conn = self._get_connection()
        cursor = conn.cursor()

        path_str = str(path.absolute())

        # Try to find existing
        existing = self._get_project_by_path(path)
        if existing:
            return existing

        # Create new
        new_id = str(uuid.uuid4())
        name = name or path.name
        now = datetime.now()

        cursor.execute(
            """
            INSERT INTO projects (id, name, path, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (new_id, name, path_str, now, now, "{}"),
        )
        conn.commit()

        return ProjectRecord(
            id=new_id, name=name, path=path, created_at=now, updated_at=now, metadata={}
        )

    def get_or_create_session(
        self,
        session_id: str,
        session_file_path: Path,
        session_type: str,
        started_at: datetime,
        workspace_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> SessionRecord:
        """Get existing session or create new one."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()

        if row:
            return self._row_to_session(row)

        # Get user identity from config
        from ..config import ReAlignConfig

        config = ReAlignConfig.load()

        now = datetime.now()
        metadata_json = json.dumps(metadata or {})
        normalized_agent_id = (agent_id or "").strip() or None
        if normalized_agent_id and not self._agent_info_exists(conn, normalized_agent_id):
            normalized_agent_id = None

        cursor.execute(
            """
            INSERT INTO sessions (
                id, session_file_path, session_type, workspace_path,
                started_at, last_activity_at, created_at, updated_at, metadata,
                created_by, agent_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                str(session_file_path),
                session_type,
                workspace_path,
                started_at,
                now,
                now,
                now,
                metadata_json,
                config.uid,
                normalized_agent_id,
            ),
        )
        if normalized_agent_id:
            self._insert_agent_session_link(conn, normalized_agent_id, session_id)
        conn.commit()

        # Upsert current user to users table
        if config.uid:
            try:
                self.upsert_user(config.uid, config.user_name)
            except Exception:
                pass

        return SessionRecord(
            id=session_id,
            session_file_path=session_file_path,
            session_type=session_type,
            started_at=started_at,
            last_activity_at=now,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            workspace_path=workspace_path,
            created_by=config.uid,
            agent_id=normalized_agent_id,
        )

    def _insert_agent_session_link(
        self, conn: sqlite3.Connection, agent_id: str, session_id: str
    ) -> None:
        """Best-effort insert into agent_sessions (V28)."""
        if not agent_id or not session_id:
            return
        if not self._session_exists(conn, session_id):
            return
        if not self._agent_info_exists(conn, agent_id):
            return
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO agent_sessions (agent_id, session_id)
                VALUES (?, ?)
                """,
                (agent_id, session_id),
            )
        except sqlite3.OperationalError:
            # Older schema without agent_sessions table.
            return

    def _session_exists(self, conn: sqlite3.Connection, session_id: str) -> bool:
        if not session_id:
            return False
        try:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
            return row is not None
        except sqlite3.OperationalError:
            return False

    def _agent_info_exists(self, conn: sqlite3.Connection, agent_id: str) -> bool:
        if not agent_id:
            return False
        try:
            row = conn.execute(
                "SELECT 1 FROM agent_info WHERE id = ? LIMIT 1",
                (agent_id,),
            ).fetchone()
            return row is not None
        except sqlite3.OperationalError:
            # Legacy schema without agent_info table.
            return True

    def _count_agent_session_links(self, conn: sqlite3.Connection, agent_id: str) -> int:
        """Count distinct sessions linked to an agent across mapping + legacy owner field."""
        try:
            row = conn.execute(
                """
                SELECT COUNT(DISTINCT session_id) AS cnt
                FROM (
                    SELECT links.session_id AS session_id
                    FROM agent_sessions links
                    JOIN sessions s ON s.id = links.session_id
                    WHERE links.agent_id = ?
                    UNION
                    SELECT s.id AS session_id
                    FROM sessions s
                    WHERE s.agent_id = ?
                ) ids
                """,
                (agent_id, agent_id),
            ).fetchone()
        except sqlite3.OperationalError:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM sessions WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
        if not row:
            return 0
        value = row[0]
        return int(value) if value is not None else 0

    def _ensure_agent_session_capacity(
        self,
        conn: sqlite3.Connection,
        *,
        agent_id: str,
        session_id: str,
    ) -> None:
        """Raise when linking a new session would exceed the per-agent cap."""
        if self.is_session_linked_to_agent(agent_id, session_id):
            return
        if self._count_agent_session_links(conn, agent_id) >= MAX_SESSION_NUM:
            raise AgentSessionLimitExceededError(agent_id=agent_id, limit=MAX_SESSION_NUM)

    def update_session_agent_id(self, session_id: str, agent_id: Optional[str]) -> None:
        """Set or update the session owner agent_id (legacy) with dual-write mapping."""
        normalized_session_id = (session_id or "").strip()
        normalized_agent_id = (agent_id or "").strip() or None
        if not normalized_session_id:
            return
        conn = self._get_connection()
        if not self._session_exists(conn, normalized_session_id):
            return
        if normalized_agent_id and not self._agent_info_exists(conn, normalized_agent_id):
            return
        if normalized_agent_id:
            self._ensure_agent_session_capacity(
                conn, agent_id=normalized_agent_id, session_id=normalized_session_id
            )
        cursor = conn.execute(
            "UPDATE sessions SET agent_id = ?, updated_at = ? WHERE id = ?",
            (normalized_agent_id, datetime.now(), normalized_session_id),
        )
        if normalized_agent_id and cursor.rowcount > 0:
            self._insert_agent_session_link(conn, normalized_agent_id, normalized_session_id)
        conn.commit()

    def update_session_activity(self, session_id: str, last_activity_at: datetime) -> None:
        """Update last activity timestamp."""
        conn = self._get_connection()
        conn.execute(
            "UPDATE sessions SET last_activity_at = ?, updated_at = ? WHERE id = ?",
            (last_activity_at, datetime.now(), session_id),
        )
        conn.commit()

    def update_session_summary(
        self,
        session_id: str,
        title: str,
        summary: str,
    ) -> None:
        """Update session's aggregated title and summary."""
        conn = self._get_connection()
        conn.execute(
            """UPDATE sessions
               SET session_title = ?,
                   session_summary = ?,
                   summary_updated_at = datetime('now'),
                   updated_at = datetime('now')
               WHERE id = ?""",
            (title, summary, session_id),
        )
        conn.commit()

    def update_session_summary_runtime(
        self,
        session_id: str,
        summary_status: str,
        summary_locked_until: Optional[datetime] = None,
        summary_error: Optional[str] = None,
    ) -> None:
        """Update session summary runtime status (V7)."""
        conn = self._get_connection()
        try:
            conn.execute(
                """UPDATE sessions
                   SET summary_status = ?,
                       summary_locked_until = ?,
                       summary_error = ?,
                       updated_at = datetime('now')
                   WHERE id = ?""",
                (
                    summary_status,
                    summary_locked_until.isoformat() if summary_locked_until else None,
                    summary_error,
                    session_id,
                ),
            )
            conn.commit()
        except sqlite3.OperationalError:
            # Older schema: ignore.
            conn.rollback()
        except Exception:
            conn.rollback()
            raise

    def update_session_total_turns(self, session_id: str, total_turns: int) -> None:
        """Update session's cached total turn count (V10)."""
        conn = self._get_connection()
        conn.execute(
            """UPDATE sessions
               SET total_turns = ?,
                   updated_at = datetime('now')
               WHERE id = ?""",
            (total_turns, session_id),
        )
        conn.commit()

    def update_session_total_turns_with_mtime(
        self, session_id: str, total_turns: int, mtime: float
    ) -> None:
        """Update session's cached total turn count with file mtime for validation (V12)."""
        conn = self._get_connection()
        conn.execute(
            """UPDATE sessions
               SET total_turns = ?,
                   total_turns_mtime = ?,
                   updated_at = datetime('now')
               WHERE id = ?""",
            (total_turns, mtime, session_id),
        )
        conn.commit()

    def update_session_metadata_flag(self, session_id: str, key: str, value: Any) -> None:
        """Update a single key in session metadata JSON."""
        conn = self._get_connection()
        row = conn.execute("SELECT metadata FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row:
            meta = json.loads(row[0] or "{}")
            meta[key] = value
            conn.execute(
                "UPDATE sessions SET metadata = ?, updated_at = datetime('now') WHERE id = ?",
                (json.dumps(meta), session_id),
            )
            conn.commit()

    def backfill_session_total_turns(self) -> int:
        """Backfill total_turns for all sessions from turns table (V10 migration).

        Returns:
            Number of sessions updated
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Update sessions.total_turns from MAX(turn_number) in turns table
        cursor.execute(
            """UPDATE sessions
               SET total_turns = (
                   SELECT COALESCE(MAX(turn_number), 0)
                   FROM turns
                   WHERE turns.session_id = sessions.id
               )
               WHERE EXISTS (
                   SELECT 1 FROM turns WHERE turns.session_id = sessions.id
               )
               AND (total_turns IS NULL OR total_turns = 0)
            """
        )
        conn.commit()
        return cursor.rowcount

    def list_sessions(
        self, limit: int = 100, workspace_path: Optional[str] = None
    ) -> List[SessionRecord]:
        """
        List sessions ordered by last activity (most recent first).

        Args:
            limit: Maximum number of sessions to return
            workspace_path: Optional filter by workspace path

        Returns:
            List of SessionRecord objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if workspace_path:
            cursor.execute(
                """SELECT * FROM sessions
                   WHERE workspace_path = ?
                   ORDER BY last_activity_at DESC
                   LIMIT ?""",
                (workspace_path, limit),
            )
        else:
            cursor.execute(
                """SELECT * FROM sessions
                   ORDER BY last_activity_at DESC
                   LIMIT ?""",
                (limit,),
            )

        return [self._row_to_session(row) for row in cursor.fetchall()]

    def get_session_by_id(self, session_id: str) -> Optional[SessionRecord]:
        """Get a session by its ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        if row:
            return self._row_to_session(row)
        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its associated data.

        Due to ON DELETE CASCADE foreign keys, deleting the session will
        automatically delete associated turns, turn_content, and event_sessions.
        """
        conn = self._get_connection()
        try:
            # agent_sessions has no FK by design; clean links explicitly.
            conn.execute("DELETE FROM agent_sessions WHERE session_id = ?", (session_id,))
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}", exc_info=True)
            conn.rollback()
            return False

    def get_sessions_by_ids(self, session_ids: List[str]) -> List[SessionRecord]:
        """Get multiple sessions by their IDs in a single query."""
        if not session_ids:
            return []

        conn = self._get_connection()
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(session_ids))
        cursor.execute(
            f"SELECT * FROM sessions WHERE id IN ({placeholders})",
            session_ids,
        )
        return [self._row_to_session(row) for row in cursor.fetchall()]

    def get_sessions_by_agent_id(
        self, agent_id: str, limit: int = MAX_SESSION_NUM
    ) -> List[SessionRecord]:
        """Get all sessions linked to an agent (V28 mapping + legacy fallback)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """SELECT s.*
                   FROM sessions s
                   WHERE s.agent_id = ?
                      OR EXISTS (
                          SELECT 1
                          FROM agent_sessions links
                          WHERE links.agent_id = ?
                            AND links.session_id = s.id
                      )
                   ORDER BY last_activity_at DESC
                   LIMIT ?""",
                (agent_id, agent_id, limit),
            )
            return [self._row_to_session(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            # Legacy fallback for DBs without agent_sessions table.
            try:
                cursor.execute(
                    """SELECT * FROM sessions
                       WHERE agent_id = ?
                       ORDER BY last_activity_at DESC
                       LIMIT ?""",
                    (agent_id, limit),
                )
                return [self._row_to_session(row) for row in cursor.fetchall()]
            except sqlite3.OperationalError:
                return []

    def get_session_ids_by_agent_id(self, agent_id: str, limit: int = MAX_SESSION_NUM) -> List[str]:
        """Get session IDs linked to an agent (lightweight lookup)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """SELECT s.id
                   FROM sessions s
                   WHERE s.agent_id = ?
                      OR EXISTS (
                          SELECT 1
                          FROM agent_sessions links
                          WHERE links.agent_id = ?
                            AND links.session_id = s.id
                      )
                   ORDER BY last_activity_at DESC
                   LIMIT ?""",
                (agent_id, agent_id, limit),
            )
            return [str(row[0]) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            try:
                cursor.execute(
                    """SELECT id
                       FROM sessions
                       WHERE agent_id = ?
                       ORDER BY last_activity_at DESC
                       LIMIT ?""",
                    (agent_id, limit),
                )
                return [str(row[0]) for row in cursor.fetchall()]
            except sqlite3.OperationalError:
                return []

    def get_agent_session_count(self, agent_id: str) -> int:
        """Get number of distinct sessions linked to an agent."""
        normalized_agent_id = (agent_id or "").strip()
        if not normalized_agent_id:
            return 0
        conn = self._get_connection()
        return self._count_agent_session_links(conn, normalized_agent_id)

    def can_link_session_to_agent(self, agent_id: str, session_id: str) -> bool:
        """Return True when this link is already present or still within link cap."""
        normalized_agent_id = (agent_id or "").strip()
        normalized_session_id = (session_id or "").strip()
        if not normalized_agent_id or not normalized_session_id:
            return False
        conn = self._get_connection()
        if not self._session_exists(conn, normalized_session_id):
            return False
        if not self._agent_info_exists(conn, normalized_agent_id):
            return False
        if self.is_session_linked_to_agent(normalized_agent_id, normalized_session_id):
            return True
        return self._count_agent_session_links(conn, normalized_agent_id) < MAX_SESSION_NUM

    def is_session_linked_to_agent(self, agent_id: str, session_id: str) -> bool:
        """Return True when a session is linked to an agent."""
        if not agent_id or not session_id:
            return False
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT 1
                FROM (
                    SELECT 1 AS linked
                    FROM agent_sessions links
                    JOIN sessions s ON s.id = links.session_id
                    WHERE links.agent_id = ?
                      AND links.session_id = ?
                    UNION ALL
                    SELECT 1 AS linked
                    FROM sessions s
                    WHERE s.id = ?
                      AND s.agent_id = ?
                ) checks
                LIMIT 1
                """,
                (agent_id, session_id, session_id, agent_id),
            )
            return cursor.fetchone() is not None
        except sqlite3.OperationalError:
            try:
                cursor.execute(
                    "SELECT 1 FROM sessions WHERE id = ? AND agent_id = ? LIMIT 1",
                    (session_id, agent_id),
                )
                return cursor.fetchone() is not None
            except sqlite3.OperationalError:
                return False

    def link_session_to_agent(self, agent_id: str, session_id: str) -> bool:
        """Create an idempotent link between agent and session (V28)."""
        normalized_agent_id = (agent_id or "").strip()
        normalized_session_id = (session_id or "").strip()
        if not normalized_agent_id or not normalized_session_id:
            return False
        conn = self._get_connection()
        if not self._session_exists(conn, normalized_session_id):
            return False
        if not self._agent_info_exists(conn, normalized_agent_id):
            return False
        try:
            self._ensure_agent_session_capacity(
                conn, agent_id=normalized_agent_id, session_id=normalized_session_id
            )
            self._insert_agent_session_link(conn, normalized_agent_id, normalized_session_id)
            # Keep legacy single-owner field populated if currently empty.
            conn.execute(
                """
                UPDATE sessions
                SET agent_id = CASE
                                   WHEN agent_id IS NULL OR TRIM(agent_id) = '' THEN ?
                                   ELSE agent_id
                               END,
                    updated_at = ?
                WHERE id = ?
                """,
                (normalized_agent_id, datetime.now(), normalized_session_id),
            )
            conn.commit()
            return True
        except sqlite3.OperationalError:
            try:
                self._ensure_agent_session_capacity(
                    conn, agent_id=normalized_agent_id, session_id=normalized_session_id
                )
                conn.execute(
                    "UPDATE sessions SET agent_id = ?, updated_at = ? WHERE id = ?",
                    (normalized_agent_id, datetime.now(), normalized_session_id),
                )
                conn.commit()
                return True
            except sqlite3.OperationalError:
                conn.rollback()
                return False
        except AgentSessionLimitExceededError:
            conn.rollback()
            raise
        except Exception:
            conn.rollback()
            return False

    def get_agent_ids_by_session_id(self, session_id: str) -> List[str]:
        """List all agent IDs linked to a session (V28 mapping + legacy fallback)."""
        if not session_id:
            return []
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT DISTINCT agent_id
                FROM (
                    SELECT agent_id
                    FROM agent_sessions
                    WHERE session_id = ?
                      AND agent_id IS NOT NULL
                      AND TRIM(agent_id) <> ''
                    UNION
                    SELECT agent_id
                    FROM sessions
                    WHERE id = ?
                      AND agent_id IS NOT NULL
                      AND TRIM(agent_id) <> ''
                ) ids
                """,
                (session_id, session_id),
            )
            return sorted({str(row[0]) for row in cursor.fetchall() if row[0]})
        except sqlite3.OperationalError:
            try:
                cursor.execute(
                    """
                    SELECT agent_id
                    FROM sessions
                    WHERE id = ?
                      AND agent_id IS NOT NULL
                      AND TRIM(agent_id) <> ''
                    """,
                    (session_id,),
                )
                return sorted({str(row[0]) for row in cursor.fetchall() if row[0]})
            except sqlite3.OperationalError:
                return []

    def get_turn_content(self, turn_id: str) -> Optional[str]:
        """Get the JSONL content for a turn."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM turn_content WHERE turn_id = ?", (turn_id,))
        row = cursor.fetchone()
        if row:
            return row[0]
        return None

    def get_turn_by_hash(self, session_id: str, content_hash: str) -> Optional[TurnRecord]:
        """Check for existing turn by content hash."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM turns WHERE session_id = ? AND content_hash = ?",
            (session_id, content_hash),
        )
        row = cursor.fetchone()

        if row:
            return self._row_to_turn(row)
        return None

    def get_turn_by_number(self, session_id: str, turn_number: int) -> Optional[TurnRecord]:
        """Get a turn by session_id and turn_number."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM turns WHERE session_id = ? AND turn_number = ?",
            (session_id, int(turn_number)),
        )
        row = cursor.fetchone()
        if row:
            return self._row_to_turn(row)
        return None

    def get_max_turn_number(self, session_id: str) -> int:
        """Get the maximum completed turn_number stored for a session, or 0 if none."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT MAX(turn_number)
            FROM turns
            WHERE session_id = ?
              AND (turn_status IS NULL OR turn_status = 'completed')
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        if not row:
            return 0
        value = row[0]
        return int(value) if value is not None else 0

    def get_max_turn_numbers_batch(self, session_ids: List[str]) -> Dict[str, int]:
        """Get max turn numbers for multiple sessions in a single query."""
        if not session_ids:
            return {}

        conn = self._get_connection()
        cursor = conn.cursor()

        # Use a single query with CASE WHEN to get max turn numbers for all sessions
        # This replaces N individual queries with 1
        placeholders = ",".join("?" * len(session_ids))
        cursor.execute(
            f"""
            SELECT session_id, MAX(turn_number) as max_turn
            FROM turns
            WHERE session_id IN ({placeholders})
              AND (turn_status IS NULL OR turn_status = 'completed')
            GROUP BY session_id
            """,
            session_ids,
        )

        return {row[0]: int(row[1]) if row[1] is not None else 0 for row in cursor.fetchall()}

    def get_committed_turn_numbers(self, session_id: str) -> set[int]:
        """Get the set of turn numbers that have been committed for a session."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT turn_number FROM turns
               WHERE session_id = ? AND turn_status = 'completed'""",
            (session_id,),
        )
        return {int(row[0]) for row in cursor.fetchall()}

    def get_turns_for_session(self, session_id: str) -> List[TurnRecord]:
        """Get all turns for a session, ordered by turn_number."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT * FROM turns
               WHERE session_id = ?
               ORDER BY turn_number""",
            (session_id,),
        )
        return [self._row_to_turn(row) for row in cursor.fetchall()]

    def create_turn(
        self, turn: TurnRecord, content: str, *, skip_session_summary: bool = False
    ) -> TurnRecord:
        """Create turn and store content."""
        conn = self._get_connection()

        try:
            if turn.temp_title is None:
                try:
                    row = conn.execute(
                        "SELECT temp_title FROM turns WHERE session_id = ? AND turn_number = ?",
                        (turn.session_id, int(turn.turn_number)),
                    ).fetchone()
                    if row and row[0]:
                        turn.temp_title = row[0]
                except sqlite3.OperationalError:
                    # Older schema without temp_title column.
                    pass

            # Insert turn record (V18: no user identity fields)
            conn.execute(
                """
                INSERT OR REPLACE INTO turns (
                    id, session_id, turn_number, user_message, assistant_summary,
                    turn_status, llm_title, temp_title, llm_description, model_name,
                    if_last_task, satisfaction, content_hash, timestamp,
                    created_at, git_commit_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.id,
                    turn.session_id,
                    turn.turn_number,
                    turn.user_message,
                    turn.assistant_summary,
                    turn.turn_status,
                    turn.llm_title,
                    turn.temp_title,
                    turn.llm_description,
                    turn.model_name,
                    turn.if_last_task,
                    turn.satisfaction,
                    turn.content_hash,
                    turn.timestamp,
                    turn.created_at,
                    turn.git_commit_hash,
                ),
            )

            # Insert content
            conn.execute(
                "INSERT INTO turn_content (turn_id, content, content_size) VALUES (?, ?, ?)",
                (turn.id, content, len(content)),
            )

            # Update session last activity
            self.update_session_activity(turn.session_id, turn.timestamp)

            conn.execute(
                """UPDATE sessions
                   SET total_turns = MAX(COALESCE(total_turns, 0), ?)
                   WHERE id = ?""",
                (turn.turn_number, turn.session_id),
            )

            conn.commit()

            # Trigger session summary job only for completed turns.
            # This avoids generating session summaries while a turn is still "processing".
            try:
                # Avoid background jobs during pytest and bulk import/migration.
                if (
                    not skip_session_summary
                    and not os.getenv("PYTEST_CURRENT_TEST")
                    and not _env_truthy("REALIGN_DISABLE_AUTO_SUMMARIES")
                    and turn.turn_status in (None, "completed")
                ):
                    self.enqueue_session_summary_job(session_id=turn.session_id)
            except Exception as e:
                # Don't fail turn creation if summary scheduling fails
                logger.warning(f"Failed to enqueue session summary job: {e}")

            return turn

        except sqlite3.IntegrityError as e:
            logger.error(f"Failed to create turn: {e}")
            conn.rollback()
            raise

    def get_completed_turn_count(self, session_id: str, *, up_to: Optional[int] = None) -> int:
        """Get count of distinct completed turns (optionally up to a max turn number)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        if up_to is not None:
            cursor.execute(
                """
                SELECT COUNT(DISTINCT turn_number)
                FROM turns
                WHERE session_id = ?
                  AND (turn_status IS NULL OR turn_status = 'completed')
                  AND turn_number <= ?
                """,
                (session_id, int(up_to)),
            )
        else:
            cursor.execute(
                """
                SELECT COUNT(DISTINCT turn_number)
                FROM turns
                WHERE session_id = ?
                  AND (turn_status IS NULL OR turn_status = 'completed')
                """,
                (session_id,),
            )
        row = cursor.fetchone()
        if not row:
            return 0
        value = row[0]
        return int(value) if value is not None else 0

    def try_acquire_lock(self, lock_key: str, *, owner: str, ttl_seconds: float) -> bool:
        """
        Try to acquire a cross-process lease lock.

        Stored in the DB (locks table) and expires automatically after TTL.
        If the existing lock is expired, it can be taken over. The same owner can renew.
        """
        conn = self._get_connection()
        now = datetime.now()
        locked_until = now + timedelta(seconds=float(ttl_seconds))

        try:
            cur = conn.execute(
                """
                INSERT INTO locks (lock_key, owner, locked_until, created_at, updated_at)
                VALUES (?, ?, ?, datetime('now'), datetime('now'))
                ON CONFLICT(lock_key) DO UPDATE SET
                    owner = excluded.owner,
                    locked_until = excluded.locked_until,
                    updated_at = datetime('now')
                WHERE
                    locks.locked_until < ?
                    OR locks.owner = excluded.owner
                """,
                (lock_key, owner, locked_until.isoformat(), now.isoformat()),
            )
            conn.commit()
            success = cur.rowcount == 1
            _log_lock_operation("ACQUIRE", lock_key, owner, success, ttl_seconds)
            return success
        except sqlite3.OperationalError:
            # Older schema without locks table: act as if lock acquired.
            try:
                conn.rollback()
            except Exception:
                pass
            _log_lock_operation("ACQUIRE", lock_key, owner, True, ttl_seconds)
            return True
        except Exception:
            conn.rollback()
            _log_lock_operation("ACQUIRE", lock_key, owner, False, ttl_seconds)
            raise

    def release_lock(self, lock_key: str, *, owner: str) -> None:
        """Release a lease lock (best-effort). Only the owner can release."""
        conn = self._get_connection()
        try:
            cur = conn.execute(
                "DELETE FROM locks WHERE lock_key = ? AND owner = ?",
                (lock_key, owner),
            )
            conn.commit()
            success = cur.rowcount > 0
            _log_lock_operation("RELEASE", lock_key, owner, success)
        except sqlite3.OperationalError:
            try:
                conn.rollback()
            except Exception:
                pass
            _log_lock_operation("RELEASE", lock_key, owner, True)
        except Exception:
            conn.rollback()
            _log_lock_operation("RELEASE", lock_key, owner, False)
            raise

    def get_all_locks(self, include_expired: bool = False) -> List[LockRecord]:
        """Get all locks, optionally including expired ones."""
        conn = self._get_connection()
        try:
            if include_expired:
                query = "SELECT lock_key, owner, locked_until, created_at, updated_at, metadata FROM locks ORDER BY created_at DESC"
                rows = conn.execute(query).fetchall()
            else:
                query = "SELECT lock_key, owner, locked_until, created_at, updated_at, metadata FROM locks WHERE locked_until >= datetime('now') ORDER BY created_at DESC"
                rows = conn.execute(query).fetchall()

            locks = []
            for row in rows:
                locks.append(
                    LockRecord(
                        lock_key=row[0],
                        owner=row[1],
                        locked_until=datetime.fromisoformat(row[2]),
                        created_at=datetime.fromisoformat(row[3]),
                        updated_at=datetime.fromisoformat(row[4]),
                        metadata=row[5],
                    )
                )
            return locks
        except sqlite3.OperationalError:
            # Older schema without locks table
            return []
        except Exception as e:
            logger.error(f"Failed to get locks: {e}")
            return []

    # -------------------------------------------------------------------------
    # Durable jobs queue (Schema V11)
    # -------------------------------------------------------------------------

    def enqueue_job(
        self,
        *,
        kind: str,
        dedupe_key: str,
        payload: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        next_run_at: Optional[datetime] = None,
        requeue_done: bool = True,
    ) -> str:
        """
        Enqueue a durable job (idempotent by dedupe_key).

        If a job with the same dedupe_key already exists:
        - If it's processing, set reschedule=1 and keep it processing.
        - Otherwise, set status back to queued (unless requeue_done=False and the job is done).
          Optionally advances next_run_at earlier.
        """
        conn = self._get_connection()
        job_id = str(uuid.uuid4())
        payload_json = json.dumps(payload or {}, ensure_ascii=False)
        # Use UTC timestamps to match SQLite's `datetime('now')` (UTC by default).
        run_at = _sqlite_dt(next_run_at or datetime.utcnow())
        requeue_done_int = 1 if requeue_done else 0

        try:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, kind, dedupe_key, payload, status, priority, attempts, next_run_at,
                    locked_until, locked_by, reschedule, last_error, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, 'queued', ?, 0, ?,
                    NULL, NULL, 0, NULL, datetime('now'), datetime('now')
                )
                ON CONFLICT(dedupe_key) DO UPDATE SET
                    kind=excluded.kind,
                    payload=excluded.payload,
                    priority=MAX(COALESCE(jobs.priority, 0), COALESCE(excluded.priority, 0)),
                    attempts=CASE
                        WHEN jobs.status='retry' THEN 0
                        ELSE COALESCE(jobs.attempts, 0)
                    END,
                    updated_at=datetime('now'),
                    reschedule=CASE
                        WHEN jobs.status='processing' THEN 1
                        ELSE COALESCE(jobs.reschedule, 0)
                    END,
                    last_error=CASE
                        WHEN jobs.status='retry' THEN NULL
                        ELSE jobs.last_error
                    END,
                    status=CASE
                        WHEN jobs.status='processing' THEN jobs.status
                        WHEN jobs.status='queued' THEN jobs.status
                        WHEN jobs.status='retry' THEN 'queued'
                        WHEN jobs.status='done' AND ? = 0 THEN jobs.status
                        ELSE 'queued'
                    END,
                    next_run_at=CASE
                        WHEN jobs.status='processing' THEN jobs.next_run_at
                        WHEN jobs.status='done' AND ? = 0 THEN jobs.next_run_at
                        WHEN jobs.next_run_at IS NULL THEN excluded.next_run_at
                        WHEN excluded.next_run_at < jobs.next_run_at THEN excluded.next_run_at
                        ELSE jobs.next_run_at
                    END
                """,
                (
                    job_id,
                    kind,
                    dedupe_key,
                    payload_json,
                    int(priority),
                    run_at,
                    requeue_done_int,
                    requeue_done_int,
                ),
            )
            conn.commit()

            row = conn.execute(
                "SELECT id FROM jobs WHERE dedupe_key = ?",
                (dedupe_key,),
            ).fetchone()
            return str(row[0]) if row else job_id
        except Exception:
            conn.rollback()
            raise

    def enqueue_turn_summary_job(
        self,
        *,
        session_file_path: Path,
        workspace_path: Optional[Path],
        turn_number: int,
        session_type: Optional[str] = None,
        skip_session_summary: bool = False,
        expected_turns: Optional[int] = None,
        skip_dedup: bool = False,
        no_track: bool = False,
        agent_id: Optional[str] = None,
    ) -> str:
        session_id = session_file_path.stem
        dedupe_key = f"turn:{session_id}:{int(turn_number)}"
        payload: Dict[str, Any] = {
            "session_id": session_id,
            "turn_number": int(turn_number),
            "session_file_path": str(session_file_path),
            "workspace_path": str(workspace_path) if workspace_path else None,
            "session_type": session_type,
        }
        if skip_session_summary:
            payload["skip_session_summary"] = True
        if expected_turns is not None:
            payload["expected_turns"] = int(expected_turns)
        if skip_dedup:
            payload["skip_dedup"] = True
        if no_track:
            payload["no_track"] = True
        if agent_id:
            payload["agent_id"] = agent_id

        # For append-only session formats (Claude/Codex/Gemini), a turn is immutable once completed.
        # Avoid re-running already-done turn jobs on repeated enqueue attempts.
        #
        # Exceptions:
        # - If the caller requested regeneration (`skip_dedup=True`), we must allow requeue.
        # - If the DB is missing the corresponding turn row (historical bug / manual DB edits),
        #   requeue so the system can self-heal instead of getting stuck in a "done but missing" state.
        requeue_done = bool(skip_dedup)
        try:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT 1 FROM turns WHERE session_id = ? AND turn_number = ? LIMIT 1",
                (session_id, int(turn_number)),
            ).fetchone()
            if row is None:
                requeue_done = True
        except Exception:
            # Best-effort: don't block enqueue on existence checks.
            pass
        return self.enqueue_job(
            kind="turn_summary",
            dedupe_key=dedupe_key,
            payload=payload,
            priority=10,
            requeue_done=requeue_done,
        )

    def enqueue_session_summary_job(self, *, session_id: str) -> str:
        dedupe_key = f"session:{session_id}"
        payload: Dict[str, Any] = {"session_id": session_id}
        # Slightly higher than turn_summary (10) so session summaries run promptly after each turn.
        # Session summaries are expected to re-run after each completed turn, even if the last
        # session_summary job is already done.
        return self.enqueue_job(
            kind="session_summary",
            dedupe_key=dedupe_key,
            payload=payload,
            priority=11,
            requeue_done=True,
        )

    def enqueue_session_process_job(
        self,
        *,
        session_file_path: Path,
        session_id: str | None = None,
        workspace_path: str | Path | None = None,
        session_type: str | None = None,
        source_event: str | None = None,
        no_track: bool = False,
        agent_id: str | None = None,
        terminal_id: str | None = None,
        priority: int = 15,
    ) -> str:
        """Enqueue a per-session processing job.

        This is the preferred way to react to Stop hooks / file changes:
        - Dedupe by session_id so repeated enqueues don't pile up.
        - Worker will process all missing turns up to the safe boundary.
        """
        sid = (session_id or session_file_path.stem or "").strip()
        if not sid:
            raise ValueError("session_id is required for session_process job")
        dedupe_key = f"session_process:{sid}"
        payload: Dict[str, Any] = {
            "session_id": sid,
            "session_file_path": str(session_file_path),
        }
        if workspace_path is not None:
            payload["workspace_path"] = str(workspace_path)
        if session_type:
            payload["session_type"] = str(session_type)
        if source_event:
            payload["source_event"] = str(source_event)
        if no_track:
            payload["no_track"] = True
        if agent_id:
            payload["agent_id"] = agent_id
        if terminal_id:
            payload["terminal_id"] = terminal_id

        # Always requeue: session_process is an "edge triggered" event that may arrive
        # after new turns were appended. Even if the last job is done, re-run to catch up.
        return self.enqueue_job(
            kind="session_process",
            dedupe_key=dedupe_key,
            payload=payload,
            priority=int(priority),
            requeue_done=True,
        )

    def enqueue_agent_description_job(self, *, agent_id: str) -> str:
        dedupe_key = f"agent_desc:{agent_id}"
        payload: Dict[str, Any] = {"agent_id": agent_id}
        return self.enqueue_job(
            kind="agent_description",
            dedupe_key=dedupe_key,
            payload=payload,
            priority=12,
            requeue_done=True,
        )

    def claim_next_job(
        self,
        *,
        worker_id: str,
        kinds: Optional[List[str]] = None,
        lease_seconds: float = 10 * 60,
    ) -> Optional[Dict[str, Any]]:
        """
        Claim the next runnable job (best-effort, cross-process safe via lease columns).

        Returns a dict with job fields, or None if nothing is runnable.
        """
        conn = self._get_connection()
        # Use UTC timestamps to match SQLite's `datetime('now')` (UTC by default).
        now = datetime.utcnow()
        lease_until = _sqlite_dt(now + timedelta(seconds=float(lease_seconds)))
        now_str = _sqlite_dt(now)

        # Self-heal: if a worker crashes mid-job, jobs can be left in `processing` forever.
        # Requeue those whose lease has expired so they can be reclaimed.
        try:
            cur = conn.execute(
                """
                UPDATE jobs
                   SET status='retry',
                       attempts=COALESCE(attempts, 0) + 1,
                       locked_until=NULL,
                       locked_by=NULL,
                       last_error=COALESCE(last_error, 'lease expired'),
                       next_run_at=?,
                       updated_at=datetime('now')
                 WHERE status='processing'
                   AND locked_until IS NOT NULL
                   AND locked_until < ?
                """,
                (now_str, now_str),
            )
            if cur.rowcount:
                conn.commit()
        except sqlite3.OperationalError:
            # Older schema without jobs table.
            try:
                conn.rollback()
            except Exception:
                pass
        except Exception:
            conn.rollback()
            raise

        kinds_clause = ""
        params: list[Any] = []
        if kinds:
            placeholders = ",".join(["?"] * len(kinds))
            kinds_clause = f" AND kind IN ({placeholders})"
            params.extend(kinds)

        try:
            # Recover orphaned processing jobs whose lease has expired.
            #
            # Without this, a worker crash or SIGTERM can leave jobs permanently stuck
            # in 'processing' because claim_next_job() only scans queued/retry rows.
            try:
                conn.execute(
                    """
                    UPDATE jobs
                       SET status='queued',
                           locked_until=NULL,
                           locked_by=NULL,
                           updated_at=datetime('now')
                     WHERE status='processing'
                       AND (locked_until IS NULL OR locked_until < datetime('now'))
                    """
                )
                conn.commit()
            except Exception:
                # Best-effort: never block claiming new work on recovery failure.
                try:
                    conn.rollback()
                except Exception:
                    pass

            rows = conn.execute(
                f"""
                SELECT id, kind, dedupe_key, payload, status, priority, attempts, next_run_at, reschedule
                FROM jobs
                WHERE status IN ('queued', 'retry')
                  AND (locked_until IS NULL OR locked_until < ?)
                  AND (next_run_at IS NULL OR next_run_at <= ?)
                  {kinds_clause}
                ORDER BY priority DESC, created_at ASC
                LIMIT 20
                """,
                tuple([now_str, now_str] + params),
            ).fetchall()

            for row in rows:
                job_id = str(row["id"])
                cur = conn.execute(
                    """
                    UPDATE jobs
                       SET status='processing',
                           locked_until=?,
                           locked_by=?,
                           updated_at=datetime('now')
                     WHERE id=?
                       AND status IN ('queued', 'retry')
                       AND (locked_until IS NULL OR locked_until < datetime('now'))
                    """,
                    (lease_until, worker_id, job_id),
                )
                if cur.rowcount == 1:
                    conn.commit()
                    payload_raw = row["payload"] or "{}"
                    try:
                        payload_obj = json.loads(payload_raw)
                    except Exception:
                        payload_obj = {}
                    return {
                        "id": job_id,
                        "kind": row["kind"],
                        "dedupe_key": row["dedupe_key"],
                        "payload": payload_obj,
                        "status": "processing",
                        "priority": row["priority"],
                        "attempts": row["attempts"],
                        "next_run_at": row["next_run_at"],
                        "reschedule": row["reschedule"],
                    }

            return None
        except sqlite3.OperationalError:
            # Older schema without jobs table: act as if no jobs.
            try:
                conn.rollback()
            except Exception:
                pass
            return None
        except Exception:
            conn.rollback()
            raise

    def requeue_stale_processing_jobs(self, *, force: bool = False) -> int:
        """
        Requeue 'processing' jobs that appear orphaned.

        By default, only requeues jobs whose lease has expired (locked_until < now) or is missing.
        With force=True, requeues all processing jobs (use with care if a worker is actively running).
        """
        conn = self._get_connection()
        try:
            where = "status='processing'"
            if not force:
                where += " AND (locked_until IS NULL OR locked_until < datetime('now'))"
            cur = conn.execute(
                f"""
                UPDATE jobs
                   SET status='queued',
                       locked_until=NULL,
                       locked_by=NULL,
                       updated_at=datetime('now')
                 WHERE {where}
                """
            )
            conn.commit()
            return int(cur.rowcount or 0)
        except sqlite3.OperationalError:
            try:
                conn.rollback()
            except Exception:
                pass
            return 0
        except Exception:
            conn.rollback()
            raise

    def requeue_processing_jobs_locked_by(self, *, worker_id: str) -> int:
        """Requeue jobs currently locked by a specific worker id (best-effort)."""
        conn = self._get_connection()
        try:
            cur = conn.execute(
                """
                UPDATE jobs
                   SET status='queued',
                       locked_until=NULL,
                       locked_by=NULL,
                       updated_at=datetime('now')
                 WHERE status='processing' AND locked_by = ?
                """,
                (str(worker_id),),
            )
            conn.commit()
            return int(cur.rowcount or 0)
        except sqlite3.OperationalError:
            try:
                conn.rollback()
            except Exception:
                pass
            return 0
        except Exception:
            conn.rollback()
            raise

    def finish_job(
        self,
        *,
        job_id: str,
        worker_id: str,
        success: bool,
        error: Optional[str] = None,
        retry_after_seconds: Optional[float] = None,
        permanent_fail: bool = False,
    ) -> None:
        conn = self._get_connection()
        try:
            if success:
                row = conn.execute(
                    "SELECT reschedule FROM jobs WHERE id = ?",
                    (job_id,),
                ).fetchone()
                reschedule = int(row[0]) if row and row[0] is not None else 0
                if reschedule:
                    conn.execute(
                        """
                        UPDATE jobs
                           SET status='queued',
                               locked_until=NULL,
                               locked_by=NULL,
                               reschedule=0,
                               next_run_at=datetime('now'),
                               last_error=NULL,
                               updated_at=datetime('now')
                         WHERE id = ? AND locked_by = ?
                        """,
                        (job_id, worker_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE jobs
                           SET status='done',
                               locked_until=NULL,
                               locked_by=NULL,
                               last_error=NULL,
                               updated_at=datetime('now')
                         WHERE id = ? AND locked_by = ?
                        """,
                        (job_id, worker_id),
                    )
                conn.commit()
                return

            # Failure
            err = _truncate_error(error)
            if permanent_fail:
                conn.execute(
                    """
                    UPDATE jobs
                       SET status='failed',
                           locked_until=NULL,
                           locked_by=NULL,
                           last_error=?,
                           updated_at=datetime('now')
                     WHERE id = ? AND locked_by = ?
                    """,
                    (err, job_id, worker_id),
                )
                conn.commit()
                return

            delay = float(retry_after_seconds) if retry_after_seconds is not None else 5.0
            # Use UTC timestamps to match SQLite's `datetime('now')` (UTC by default).
            next_run = datetime.utcnow() + timedelta(seconds=delay)
            conn.execute(
                """
                UPDATE jobs
                   SET status='retry',
                       attempts=COALESCE(attempts, 0) + 1,
                       locked_until=NULL,
                       locked_by=NULL,
                       last_error=?,
                       next_run_at=?,
                       updated_at=datetime('now')
                 WHERE id = ? AND locked_by = ?
                """,
                (err, _sqlite_dt(next_run), job_id, worker_id),
            )
            conn.commit()
        except sqlite3.OperationalError:
            try:
                conn.rollback()
            except Exception:
                pass
        except Exception:
            conn.rollback()
            raise

    def get_job_counts(self) -> Dict[str, int]:
        conn = self._get_connection()
        counts: Dict[str, int] = {}
        try:
            rows = conn.execute("SELECT status, COUNT(*) AS c FROM jobs GROUP BY status").fetchall()
            for row in rows:
                counts[str(row[0])] = int(row[1])
            return counts
        except sqlite3.OperationalError:
            return {}
        except Exception:
            return {}

    def list_jobs(
        self,
        *,
        limit: int = 30,
        offset: int = 0,
        kinds: Optional[List[str]] = None,
        statuses: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        List recent jobs for status display.

        Ordered so active jobs appear first:
        processing -> queued/retry -> done -> failed.
        """
        conn = self._get_connection()
        try:
            where_clauses: list[str] = []
            params: list[Any] = []
            if kinds:
                placeholders = ",".join(["?"] * len(kinds))
                where_clauses.append(f"kind IN ({placeholders})")
                params.extend(kinds)
            if statuses:
                placeholders = ",".join(["?"] * len(statuses))
                where_clauses.append(f"status IN ({placeholders})")
                params.extend(statuses)

            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)

            rows = conn.execute(
                f"""
                SELECT id, kind, dedupe_key, status, priority, attempts,
                       next_run_at, locked_until, locked_by, reschedule,
                       last_error, created_at, updated_at, payload
                  FROM jobs
                  {where_sql}
                 ORDER BY
                   CASE status
                     WHEN 'processing' THEN 0
                     WHEN 'queued' THEN 1
                     WHEN 'retry' THEN 1
                     WHEN 'done' THEN 2
                     WHEN 'failed' THEN 3
                     ELSE 9
                   END,
                   datetime(updated_at) DESC
                 LIMIT ? OFFSET ?
                """,
                tuple(params + [int(limit), int(offset)]),
            ).fetchall()

            out: List[Dict[str, Any]] = []
            for row in rows:
                payload_raw = row["payload"] or "{}"
                try:
                    payload_obj = json.loads(payload_raw)
                except Exception:
                    payload_obj = {}
                out.append(
                    {
                        "id": str(row["id"]),
                        "kind": row["kind"],
                        "dedupe_key": row["dedupe_key"],
                        "status": row["status"],
                        "priority": row["priority"],
                        "attempts": row["attempts"],
                        "next_run_at": row["next_run_at"],
                        "locked_until": row["locked_until"],
                        "locked_by": row["locked_by"],
                        "reschedule": row["reschedule"],
                        "last_error": row["last_error"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "payload": payload_obj,
                    }
                )
            return out
        except sqlite3.OperationalError:
            return []
        except Exception:
            return []

    def count_jobs(
        self,
        *,
        kinds: Optional[List[str]] = None,
        statuses: Optional[List[str]] = None,
    ) -> int:
        """Count jobs matching optional filters."""
        conn = self._get_connection()
        try:
            where_clauses: list[str] = []
            params: list[Any] = []
            if kinds:
                placeholders = ",".join(["?"] * len(kinds))
                where_clauses.append(f"kind IN ({placeholders})")
                params.extend(kinds)
            if statuses:
                placeholders = ",".join(["?"] * len(statuses))
                where_clauses.append(f"status IN ({placeholders})")
                params.extend(statuses)

            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)

            row = conn.execute(
                f"SELECT COUNT(*) AS c FROM jobs {where_sql}",
                tuple(params),
            ).fetchone()
            if not row:
                return 0
            return int(row[0])
        except sqlite3.OperationalError:
            return 0
        except Exception:
            return 0

    def requeue_failed_jobs(
        self,
        *,
        kinds: Optional[List[str]] = None,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Requeue all failed jobs, optionally filtering by kind.

        Returns:
            (count, jobs) - number of jobs requeued and their details
        """
        conn = self._get_connection()
        try:
            # First, get the failed jobs
            where_clauses: list[str] = ["status = 'failed'"]
            params: list[Any] = []
            if kinds:
                placeholders = ",".join(["?"] * len(kinds))
                where_clauses.append(f"kind IN ({placeholders})")
                params.extend(kinds)

            where_sql = "WHERE " + " AND ".join(where_clauses)

            rows = conn.execute(
                f"""
                SELECT id, kind, dedupe_key, payload, last_error, attempts
                FROM jobs
                {where_sql}
                ORDER BY updated_at DESC
                """,
                tuple(params),
            ).fetchall()

            if not rows:
                return 0, []

            jobs_info: List[Dict[str, Any]] = []
            for row in rows:
                payload_raw = row["payload"] or "{}"
                try:
                    payload_obj = json.loads(payload_raw)
                except Exception:
                    payload_obj = {}
                jobs_info.append(
                    {
                        "id": str(row["id"]),
                        "kind": row["kind"],
                        "dedupe_key": row["dedupe_key"],
                        "payload": payload_obj,
                        "last_error": row["last_error"],
                        "attempts": row["attempts"],
                    }
                )

            # Requeue them
            conn.execute(
                f"""
                UPDATE jobs
                SET status = 'queued',
                    attempts = 0,
                    next_run_at = datetime('now'),
                    last_error = NULL,
                    locked_until = NULL,
                    locked_by = NULL,
                    updated_at = datetime('now')
                {where_sql}
                """,
                tuple(params),
            )
            conn.commit()

            return len(jobs_info), jobs_info
        except sqlite3.OperationalError:
            return 0, []
        except Exception:
            conn.rollback()
            return 0, []

    def update_turn_summary(
        self,
        turn_id: str,
        llm_title: str,
        llm_description: Optional[str],
        assistant_summary: Optional[str],
        model_name: Optional[str],
        if_last_task: str,
        satisfaction: str,
    ) -> bool:
        """
        Update an existing turn's LLM-generated summary fields.

        Used by the refresh command to regenerate summaries.

        Returns:
            True if update succeeded, False otherwise
        """
        conn = self._get_connection()
        try:
            conn.execute(
                """UPDATE turns SET
                    llm_title = ?,
                    llm_description = ?,
                    assistant_summary = ?,
                    model_name = ?,
                    if_last_task = ?,
                    satisfaction = ?
                WHERE id = ?""",
                (
                    llm_title,
                    llm_description,
                    assistant_summary,
                    model_name,
                    if_last_task,
                    satisfaction,
                    turn_id,
                ),
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update turn summary: {e}")
            conn.rollback()
            return False

    def sync_events(self, events: List[EventRecord]) -> None:
        """Sync events to database."""
        if not events:
            return

        conn = self._get_connection()
        try:
            for event in events:
                # Upsert Event (V18: created_by/shared_by)
                conn.execute(
                    """
                    INSERT INTO events (
                        id, title, description, event_type, status,
                        start_timestamp, end_timestamp, created_at, updated_at, metadata,
                        created_by, shared_by
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        title=excluded.title,
                        description=excluded.description,
                        event_type=excluded.event_type,
                        status=excluded.status,
                        start_timestamp=excluded.start_timestamp,
                        end_timestamp=excluded.end_timestamp,
                        updated_at=excluded.updated_at,
                        metadata=excluded.metadata,
                        created_by=excluded.created_by,
                        shared_by=excluded.shared_by
                    """,
                    (
                        event.id,
                        event.title,
                        event.description,
                        event.event_type,
                        event.status,
                        event.start_timestamp,
                        event.end_timestamp,
                        event.created_at,
                        event.updated_at,
                        json.dumps(event.metadata),
                        event.created_by,
                        event.shared_by,
                    ),
                )

                # Sync Commits (Delete all existing links and re-insert)
                conn.execute("DELETE FROM event_commits WHERE event_id = ?", (event.id,))

                if event.commit_hashes:
                    conn.executemany(
                        "INSERT INTO event_commits (event_id, commit_hash) VALUES (?, ?)",
                        [(event.id, h) for h in event.commit_hashes],
                    )

            conn.commit()

        except Exception as e:
            logger.error(f"Failed to sync events: {e}", exc_info=True)
            conn.rollback()
            raise

    def search_events(
        self,
        query: str,
        limit: int = 20,
        use_regex: bool = False,
        ignore_case: bool = True,
    ) -> List[EventRecord]:
        """Search events by full-text query or regex pattern.

        Args:
            query: Search query (keywords or regex pattern)
            limit: Maximum number of results
            use_regex: If True, use REGEXP instead of FTS/LIKE
            ignore_case: If True, ignore case in regex matching
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if use_regex:
            # Use REGEXP for regex mode
            pattern = f"(?i){query}" if ignore_case else query
            cursor.execute(
                """
                SELECT *
                FROM events
                WHERE title REGEXP ? OR description REGEXP ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (pattern, pattern, limit),
            )
        else:
            # Prefer FTS5 (fts_events) when available; fall back to LIKE for older/migrated DBs.
            try:
                cursor.execute(
                    """
                    SELECT e.*
                    FROM fts_events
                    JOIN events e ON e.rowid = fts_events.rowid
                    WHERE fts_events MATCH ?
                    ORDER BY bm25(fts_events)
                    LIMIT ?
                    """,
                    (query, limit),
                )
            except sqlite3.OperationalError as e:
                # Common cases:
                # - older schema missing the FTS virtual table
                # - SQLite build without FTS5 enabled
                message = str(e).lower()
                if (
                    "no such table: fts_events" in message
                    or "no such module: fts5" in message
                    or "no such function: bm25" in message
                    or message.startswith("fts5:")
                    or "syntax error" in message
                ):
                    search_term = f"%{query}%"
                    cursor.execute(
                        """
                        SELECT *
                        FROM events
                        WHERE title LIKE ? OR description LIKE ?
                        ORDER BY updated_at DESC
                        LIMIT ?
                        """,
                        (search_term, search_term, limit),
                    )
                else:
                    raise

        events = []
        for row in cursor.fetchall():
            events.append(self._row_to_event(row))
        return events

    def get_event_by_id(self, event_id: str) -> Optional[EventRecord]:
        """Get event by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        row = cursor.fetchone()

        if row:
            return self._row_to_event(row)
        return None

    def list_events(self, limit: int = 50, offset: int = 0) -> List[EventRecord]:
        """List all events, ordered by updated_at descending."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """SELECT * FROM events
               ORDER BY updated_at DESC
               LIMIT ? OFFSET ?""",
            (limit, offset),
        )

        events = []
        for row in cursor.fetchall():
            events.append(self._row_to_event(row))
        return events

    def delete_event(self, event_id: str) -> bool:
        """Delete an event and its associations."""
        conn = self._get_connection()
        try:
            # Delete from event_sessions first (foreign key)
            conn.execute("DELETE FROM event_sessions WHERE event_id = ?", (event_id,))
            # Delete from event_commits
            conn.execute("DELETE FROM event_commits WHERE event_id = ?", (event_id,))
            # Delete from events
            cursor = conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete event {event_id}: {e}", exc_info=True)
            conn.rollback()
            return False

    def get_sessions_for_event(self, event_id: str) -> List[SessionRecord]:
        """Get all sessions linked to an event."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT s.* FROM sessions s
               JOIN event_sessions es ON s.id = es.session_id
               WHERE es.event_id = ?
               ORDER BY s.last_activity_at DESC""",
            (event_id,),
        )
        return [self._row_to_session(row) for row in cursor.fetchall()]

    def get_events_for_session(self, session_id: str) -> List[EventRecord]:
        """Get all events that contain this session."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """SELECT e.* FROM events e
               JOIN event_sessions es ON e.id = es.event_id
               WHERE es.session_id = ?
               ORDER BY e.updated_at DESC""",
            (session_id,),
        )
        return [self._row_to_event(row) for row in cursor.fetchall()]

    def link_session_to_event(self, event_id: str, session_id: str) -> None:
        """Link a session to an event."""
        conn = self._get_connection()
        conn.execute(
            """INSERT OR IGNORE INTO event_sessions (event_id, session_id)
               VALUES (?, ?)""",
            (event_id, session_id),
        )
        conn.commit()

    def unlink_session_from_event(self, event_id: str, session_id: str) -> None:
        """Unlink a session from an event."""
        conn = self._get_connection()
        conn.execute(
            "DELETE FROM event_sessions WHERE event_id = ? AND session_id = ?",
            (event_id, session_id),
        )
        conn.commit()

    def update_event_summary(
        self,
        event_id: str,
        title: str,
        description: str,
    ) -> None:
        """Update event's aggregated title and description."""
        conn = self._get_connection()
        conn.execute(
            """UPDATE events
               SET title = ?,
                   description = ?,
                   updated_at = datetime('now')
               WHERE id = ?""",
            (title, description, event_id),
        )
        conn.commit()

    def update_event_share_metadata(
        self,
        event_id: str,
        preset_questions: Optional[List[str]] = None,
        slack_message: Optional[str] = None,
        share_url: Optional[str] = None,
        share_id: Optional[str] = None,
        share_admin_token: Optional[str] = None,
        share_expiry_at: Optional[datetime] = None,
    ) -> None:
        """Update event's share metadata (preset questions, slack message, and share URL)."""
        import json

        conn = self._get_connection()

        # Convert preset_questions list to JSON string
        preset_questions_json = json.dumps(preset_questions) if preset_questions else None

        share_expiry_str = (
            share_expiry_at.isoformat() if isinstance(share_expiry_at, datetime) else None
        )

        conn.execute(
            """UPDATE events
               SET preset_questions = COALESCE(?, preset_questions),
                   slack_message = COALESCE(?, slack_message),
                   share_url = COALESCE(?, share_url),
                   share_id = COALESCE(?, share_id),
                   share_admin_token = COALESCE(?, share_admin_token),
                   share_expiry_at = COALESCE(?, share_expiry_at),
                   updated_at = datetime('now')
               WHERE id = ?""",
            (
                preset_questions_json,
                slack_message,
                share_url,
                share_id,
                share_admin_token,
                share_expiry_str,
                event_id,
            ),
        )
        conn.commit()

    def search_conversations(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
        use_regex: bool = False,
        ignore_case: bool = True,
        session_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search conversation turns by title and summary only.

        Args:
            query: Search query (keywords or regex pattern)
            limit: Maximum number of results
            offset: Number of matching rows to skip
            use_regex: If True, use REGEXP instead of LIKE
            ignore_case: If True, ignore case in matching
            session_ids: Optional list of session IDs to limit search scope
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Build session filter clause
        session_filter = ""
        if session_ids:
            placeholders = ",".join("?" * len(session_ids))
            session_filter = f" AND t.session_id IN ({placeholders})"

        if use_regex:
            # Use REGEXP for regex mode
            pattern = f"(?i){query}" if ignore_case else query
            params: List[Any] = [pattern, pattern] + (session_ids or []) + [limit, offset]
            cursor.execute(
                f"""
                SELECT
                    t.id as turn_id,
                    t.session_id,
                    t.turn_number,
                    t.llm_title,
                    t.assistant_summary,
                    t.timestamp
                FROM turns t
                WHERE
                    (t.llm_title REGEXP ? OR
                    t.assistant_summary REGEXP ?){session_filter}
                ORDER BY t.timestamp DESC
                LIMIT ?
                OFFSET ?
                """,
                params,
            )
        else:
            # Use LIKE for normal search
            search_term = f"%{query}%"
            params = [search_term, search_term] + (session_ids or []) + [limit, offset]
            cursor.execute(
                f"""
                SELECT
                    t.id as turn_id,
                    t.session_id,
                    t.turn_number,
                    t.llm_title,
                    t.assistant_summary,
                    t.timestamp
                FROM turns t
                WHERE
                    (t.llm_title LIKE ? OR
                    t.assistant_summary LIKE ?){session_filter}
                ORDER BY t.timestamp DESC
                LIMIT ?
                OFFSET ?
                """,
                params,
            )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "turn_id": row["turn_id"],
                    "session_id": row["session_id"],
                    "turn_number": row["turn_number"],
                    "title": row["llm_title"],
                    "summary": row["assistant_summary"],
                    "timestamp": row["timestamp"],
                }
            )
        return results

    def count_search_conversations(
        self,
        query: str,
        use_regex: bool = False,
        ignore_case: bool = True,
        session_ids: Optional[List[str]] = None,
    ) -> int:
        """Count conversation turns matching title/summary query."""
        conn = self._get_connection()
        cursor = conn.cursor()

        session_filter = ""
        if session_ids:
            placeholders = ",".join("?" * len(session_ids))
            session_filter = f" AND t.session_id IN ({placeholders})"

        if use_regex:
            pattern = f"(?i){query}" if ignore_case else query
            params: List[Any] = [pattern, pattern] + (session_ids or [])
            cursor.execute(
                f"""
                SELECT COUNT(*) as count
                FROM turns t
                WHERE
                    (t.llm_title REGEXP ? OR
                    t.assistant_summary REGEXP ?){session_filter}
                """,
                params,
            )
        else:
            search_term = f"%{query}%"
            params = [search_term, search_term] + (session_ids or [])
            cursor.execute(
                f"""
                SELECT COUNT(*) as count
                FROM turns t
                WHERE
                    (t.llm_title LIKE ? OR
                    t.assistant_summary LIKE ?){session_filter}
                """,
                params,
            )

        row = cursor.fetchone()
        return int(row["count"] if row and row["count"] is not None else 0)

    def search_turn_content(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
        use_regex: bool = False,
        ignore_case: bool = True,
        session_ids: Optional[List[str]] = None,
        turn_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search turn content (full JSONL content).

        Args:
            query: Search query (keywords or regex pattern)
            limit: Maximum number of results
            offset: Number of matching rows to skip
            use_regex: If True, use REGEXP instead of LIKE
            ignore_case: If True, ignore case in matching
            session_ids: Optional list of session IDs to limit search scope
            turn_ids: Optional list of turn IDs to limit search scope
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Build filter clauses
        filters = []
        filter_params: List[Any] = []

        if session_ids:
            placeholders = ",".join("?" * len(session_ids))
            filters.append(f"t.session_id IN ({placeholders})")
            filter_params.extend(session_ids)

        if turn_ids:
            placeholders = ",".join("?" * len(turn_ids))
            filters.append(f"t.id IN ({placeholders})")
            filter_params.extend(turn_ids)

        filter_clause = ""
        if filters:
            filter_clause = " AND " + " AND ".join(filters)

        if use_regex:
            # Use REGEXP for regex mode
            pattern = f"(?i){query}" if ignore_case else query
            params: List[Any] = [pattern] + filter_params + [limit, offset]
            cursor.execute(
                f"""
                SELECT
                    t.id as turn_id,
                    t.session_id,
                    t.turn_number,
                    t.llm_title,
                    t.assistant_summary,
                    tc.content
                FROM turns t
                JOIN turn_content tc ON t.id = tc.turn_id
                WHERE tc.content REGEXP ?{filter_clause}
                ORDER BY t.timestamp DESC
                LIMIT ?
                OFFSET ?
                """,
                params,
            )
        else:
            # Use LIKE for normal search
            search_term = f"%{query}%"
            params = [search_term] + filter_params + [limit, offset]
            cursor.execute(
                f"""
                SELECT
                    t.id as turn_id,
                    t.session_id,
                    t.turn_number,
                    t.llm_title,
                    t.assistant_summary,
                    tc.content
                FROM turns t
                JOIN turn_content tc ON t.id = tc.turn_id
                WHERE tc.content LIKE ?{filter_clause}
                ORDER BY t.timestamp DESC
                LIMIT ?
                OFFSET ?
                """,
                params,
            )

        results = []
        for row in cursor.fetchall():
            content = row["content"]
            results.append(
                {
                    "turn_id": row["turn_id"],
                    "session_id": row["session_id"],
                    "turn_number": row["turn_number"],
                    "title": row["llm_title"],
                    "summary": row["assistant_summary"],
                    "content": content,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content,
                }
            )
        return results

    def count_search_turn_content(
        self,
        query: str,
        use_regex: bool = False,
        ignore_case: bool = True,
        session_ids: Optional[List[str]] = None,
        turn_ids: Optional[List[str]] = None,
    ) -> int:
        """Count turn content rows matching query."""
        conn = self._get_connection()
        cursor = conn.cursor()

        filters = []
        filter_params: List[Any] = []

        if session_ids:
            placeholders = ",".join("?" * len(session_ids))
            filters.append(f"t.session_id IN ({placeholders})")
            filter_params.extend(session_ids)

        if turn_ids:
            placeholders = ",".join("?" * len(turn_ids))
            filters.append(f"t.id IN ({placeholders})")
            filter_params.extend(turn_ids)

        filter_clause = ""
        if filters:
            filter_clause = " AND " + " AND ".join(filters)

        if use_regex:
            pattern = f"(?i){query}" if ignore_case else query
            params: List[Any] = [pattern] + filter_params
            cursor.execute(
                f"""
                SELECT COUNT(*) as count
                FROM turns t
                JOIN turn_content tc ON t.id = tc.turn_id
                WHERE tc.content REGEXP ?{filter_clause}
                """,
                params,
            )
        else:
            search_term = f"%{query}%"
            params = [search_term] + filter_params
            cursor.execute(
                f"""
                SELECT COUNT(*) as count
                FROM turns t
                JOIN turn_content tc ON t.id = tc.turn_id
                WHERE tc.content LIKE ?{filter_clause}
                """,
                params,
            )

        row = cursor.fetchone()
        return int(row["count"] if row and row["count"] is not None else 0)

    def search_sessions(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
        use_regex: bool = False,
        ignore_case: bool = True,
        session_ids: Optional[List[str]] = None,
    ) -> List[SessionRecord]:
        """Search sessions by title and summary.

        Args:
            query: Search query (keywords or regex pattern)
            limit: Maximum number of results
            offset: Number of matching rows to skip
            use_regex: If True, use REGEXP instead of LIKE
            ignore_case: If True, ignore case in matching
            session_ids: Optional list of session IDs to limit search scope
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Build session filter clause
        session_filter = ""
        if session_ids:
            placeholders = ",".join("?" * len(session_ids))
            session_filter = f" AND id IN ({placeholders})"

        if use_regex:
            # Use REGEXP for regex mode
            pattern = f"(?i){query}" if ignore_case else query
            params: List[Any] = [pattern, pattern] + (session_ids or []) + [limit, offset]
            cursor.execute(
                f"""
                SELECT *
                FROM sessions
                WHERE
                    (session_title REGEXP ? OR
                    session_summary REGEXP ?){session_filter}
                ORDER BY updated_at DESC
                LIMIT ?
                OFFSET ?
                """,
                params,
            )
        else:
            # Use LIKE for normal search
            search_term = f"%{query}%"
            params = [search_term, search_term] + (session_ids or []) + [limit, offset]
            cursor.execute(
                f"""
                SELECT *
                FROM sessions
                WHERE
                    (session_title LIKE ? OR
                    session_summary LIKE ?){session_filter}
                ORDER BY updated_at DESC
                LIMIT ?
                OFFSET ?
                """,
                params,
            )

        sessions = []
        for row in cursor.fetchall():
            sessions.append(self._row_to_session(row))
        return sessions

    def count_search_sessions(
        self,
        query: str,
        use_regex: bool = False,
        ignore_case: bool = True,
        session_ids: Optional[List[str]] = None,
    ) -> int:
        """Count sessions matching title/summary query."""
        conn = self._get_connection()
        cursor = conn.cursor()

        session_filter = ""
        if session_ids:
            placeholders = ",".join("?" * len(session_ids))
            session_filter = f" AND id IN ({placeholders})"

        if use_regex:
            pattern = f"(?i){query}" if ignore_case else query
            params: List[Any] = [pattern, pattern] + (session_ids or [])
            cursor.execute(
                f"""
                SELECT COUNT(*) as count
                FROM sessions
                WHERE
                    (session_title REGEXP ? OR
                    session_summary REGEXP ?){session_filter}
                """,
                params,
            )
        else:
            search_term = f"%{query}%"
            params = [search_term, search_term] + (session_ids or [])
            cursor.execute(
                f"""
                SELECT COUNT(*) as count
                FROM sessions
                WHERE
                    (session_title LIKE ? OR
                    session_summary LIKE ?){session_filter}
                """,
                params,
            )

        row = cursor.fetchone()
        return int(row["count"] if row and row["count"] is not None else 0)

    def close(self):
        thread_id = threading.get_ident()
        conn: sqlite3.Connection | None = None
        with self._connections_lock:
            conn = self._connections.pop(thread_id, None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Agent methods (Schema V15 - replaces terminal.json)
    # -------------------------------------------------------------------------

    def get_or_create_agent(
        self,
        agent_id: str,
        provider: str,
        session_type: str,
        *,
        session_id: Optional[str] = None,
        context_id: Optional[str] = None,
        transcript_path: Optional[str] = None,
        cwd: Optional[str] = None,
        project_dir: Optional[str] = None,
        source: Optional[str] = None,
        status: str = "active",
        attention: Optional[str] = None,
    ) -> AgentRecord:
        """Get existing agent or create new one."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM agents WHERE id = ?", (agent_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_agent(row)
        except sqlite3.OperationalError:
            # Table may not exist in older schema
            raise

        # Get user identity from config
        try:
            from ..config import ReAlignConfig

            config = ReAlignConfig.load()
            created_by = config.uid
            user_name_for_upsert = config.user_name
        except Exception:
            created_by = None
            user_name_for_upsert = None

        now = datetime.now()
        cursor.execute(
            """
            INSERT INTO agents (
                id, provider, session_type, session_id, context_id,
                transcript_path, cwd, project_dir, status, attention, source,
                created_at, updated_at, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                provider,
                session_type,
                session_id,
                context_id,
                transcript_path,
                cwd,
                project_dir,
                status,
                attention,
                source,
                now,
                now,
                created_by,
            ),
        )
        conn.commit()

        # Upsert current user to users table
        if created_by:
            try:
                self.upsert_user(created_by, user_name_for_upsert)
            except Exception:
                pass

        return AgentRecord(
            id=agent_id,
            provider=provider,
            session_type=session_type,
            session_id=session_id,
            context_id=context_id,
            transcript_path=transcript_path,
            cwd=cwd,
            project_dir=project_dir,
            status=status,
            attention=attention,
            source=source,
            created_at=now,
            updated_at=now,
            created_by=created_by,
        )

    def update_agent(
        self,
        agent_id: str,
        *,
        provider: Optional[str] = None,
        session_type: Optional[str] = None,
        session_id: Optional[str] = None,
        context_id: Optional[str] = None,
        transcript_path: Optional[str] = None,
        cwd: Optional[str] = None,
        project_dir: Optional[str] = None,
        status: Optional[str] = None,
        attention: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Optional[AgentRecord]:
        """Update an existing agent. Returns the updated record or None if not found."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Build dynamic UPDATE statement
        updates: List[str] = []
        params: List[Any] = []

        if provider is not None:
            updates.append("provider = ?")
            params.append(provider)
        if session_type is not None:
            updates.append("session_type = ?")
            params.append(session_type)
        if session_id is not None:
            updates.append("session_id = ?")
            params.append(session_id)
        if context_id is not None:
            updates.append("context_id = ?")
            params.append(context_id)
        if transcript_path is not None:
            updates.append("transcript_path = ?")
            params.append(transcript_path)
        if cwd is not None:
            updates.append("cwd = ?")
            params.append(cwd)
        if project_dir is not None:
            updates.append("project_dir = ?")
            params.append(project_dir)
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if attention is not None:
            updates.append("attention = ?")
            params.append(attention if attention != "" else None)
        if source is not None:
            updates.append("source = ?")
            params.append(source)

        if not updates:
            # Nothing to update, just return existing
            return self.get_agent_by_id(agent_id)

        updates.append("updated_at = datetime('now')")
        params.append(agent_id)

        try:
            cursor.execute(
                f"UPDATE agents SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()
            return self.get_agent_by_id(agent_id)
        except sqlite3.OperationalError:
            conn.rollback()
            return None

    def get_agent_by_id(self, agent_id: str) -> Optional[AgentRecord]:
        """Get an agent by its ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM agents WHERE id = ?", (agent_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_agent(row)
        except sqlite3.OperationalError:
            pass
        return None

    def list_agents(
        self,
        *,
        status: Optional[str] = None,
        context_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AgentRecord]:
        """List agents, optionally filtered by status or context."""
        conn = self._get_connection()
        cursor = conn.cursor()

        where_clauses: List[str] = []
        params: List[Any] = []

        if status:
            where_clauses.append("status = ?")
            params.append(status)
        if context_id:
            where_clauses.append("context_id = ?")
            params.append(context_id)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        params.append(limit)

        try:
            cursor.execute(
                f"""
                SELECT * FROM agents
                {where_sql}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                params,
            )
            return [self._row_to_agent(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            return []

    def insert_window_link(
        self,
        *,
        terminal_id: str,
        agent_id: Optional[str],
        session_id: Optional[str],
        provider: Optional[str],
        source: Optional[str],
        ts: Optional[float] = None,
    ) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        ts_value = ts if ts is not None else time.time()
        try:
            cursor.execute(
                """
                INSERT INTO windowlink (
                    terminal_id, agent_id, session_id, provider, source, ts
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    terminal_id,
                    agent_id,
                    session_id,
                    provider,
                    source,
                    float(ts_value),
                ),
            )
            conn.commit()
        except sqlite3.OperationalError:
            conn.rollback()

    def list_latest_window_links(
        self, *, agent_id: Optional[str] = None, limit: int = 1000
    ) -> List[WindowLinkRecord]:
        conn = self._get_connection()
        cursor = conn.cursor()
        params: List[Any] = []
        where_clause = ""
        if agent_id:
            where_clause = "WHERE agent_id = ?"
            params.append(agent_id)
        params.append(limit)
        try:
            cursor.execute(
                f"""
                SELECT terminal_id, agent_id, session_id, provider, source, ts, created_at
                FROM (
                    SELECT terminal_id,
                           agent_id,
                           session_id,
                           provider,
                           source,
                           ts,
                           created_at,
                           ROW_NUMBER() OVER (
                               PARTITION BY terminal_id
                               ORDER BY ts DESC, id DESC
                           ) AS rn
                    FROM windowlink
                    {where_clause}
                ) AS ranked
                WHERE rn = 1
                LIMIT ?
                """,
                params,
            )
            rows = cursor.fetchall()
            out: List[WindowLinkRecord] = []
            for row in rows:
                out.append(
                    WindowLinkRecord(
                        terminal_id=row[0],
                        agent_id=row[1],
                        session_id=row[2],
                        provider=row[3],
                        source=row[4],
                        ts=row[5],
                        created_at=self._parse_datetime(row[6]) if row[6] else None,
                    )
                )
            return out
        except sqlite3.OperationalError:
            return []

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.OperationalError:
            conn.rollback()
            return False

    # -------------------------------------------------------------------------
    # Agent Context methods (Schema V15 - replaces load.json)
    # -------------------------------------------------------------------------

    def get_or_create_agent_context(
        self,
        context_id: str,
        *,
        workspace: Optional[str] = None,
        loaded_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentContextRecord:
        """Get existing agent context or create new one."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM agent_contexts WHERE id = ?", (context_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_agent_context(row)
        except sqlite3.OperationalError:
            raise

        now = datetime.now()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

        cursor.execute(
            """
            INSERT INTO agent_contexts (id, workspace, loaded_at, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (context_id, workspace, loaded_at, now, now, metadata_json),
        )
        conn.commit()

        return AgentContextRecord(
            id=context_id,
            workspace=workspace,
            loaded_at=loaded_at,
            created_at=now,
            updated_at=now,
            metadata=metadata,
            session_ids=[],
            event_ids=[],
        )

    def update_agent_context(
        self,
        context_id: str,
        *,
        workspace: Optional[str] = None,
        loaded_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AgentContextRecord]:
        """Update an existing agent context."""
        conn = self._get_connection()
        cursor = conn.cursor()

        updates: List[str] = []
        params: List[Any] = []

        if workspace is not None:
            updates.append("workspace = ?")
            params.append(workspace)
        if loaded_at is not None:
            updates.append("loaded_at = ?")
            params.append(loaded_at)
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata, ensure_ascii=False))

        if not updates:
            return self.get_agent_context_by_id(context_id)

        updates.append("updated_at = datetime('now')")
        params.append(context_id)

        try:
            cursor.execute(
                f"UPDATE agent_contexts SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()
            return self.get_agent_context_by_id(context_id)
        except sqlite3.OperationalError:
            conn.rollback()
            return None

    def get_agent_context_by_id(self, context_id: str) -> Optional[AgentContextRecord]:
        """Get an agent context by its ID, including linked sessions and events."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM agent_contexts WHERE id = ?", (context_id,))
            row = cursor.fetchone()
            if not row:
                return None

            record = self._row_to_agent_context(row)

            # Fetch linked session IDs
            cursor.execute(
                "SELECT session_id FROM agent_context_sessions WHERE context_id = ?",
                (context_id,),
            )
            record.session_ids = [r[0] for r in cursor.fetchall()]

            # Fetch linked event IDs
            cursor.execute(
                "SELECT event_id FROM agent_context_events WHERE context_id = ?",
                (context_id,),
            )
            record.event_ids = [r[0] for r in cursor.fetchall()]

            return record
        except sqlite3.OperationalError:
            return None

    def get_agent_context_by_workspace(self, workspace: str) -> Optional[AgentContextRecord]:
        """Get an agent context by workspace path."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM agent_contexts WHERE workspace = ?", (workspace,))
            row = cursor.fetchone()
            if not row:
                return None

            record = self._row_to_agent_context(row)
            context_id = record.id

            # Fetch linked session IDs
            cursor.execute(
                "SELECT session_id FROM agent_context_sessions WHERE context_id = ?",
                (context_id,),
            )
            record.session_ids = [r[0] for r in cursor.fetchall()]

            # Fetch linked event IDs
            cursor.execute(
                "SELECT event_id FROM agent_context_events WHERE context_id = ?",
                (context_id,),
            )
            record.event_ids = [r[0] for r in cursor.fetchall()]

            return record
        except sqlite3.OperationalError:
            return None

    def list_agent_contexts(self, limit: int = 100) -> List[AgentContextRecord]:
        """List all agent contexts."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT * FROM agent_contexts
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            contexts = []
            for row in cursor.fetchall():
                record = self._row_to_agent_context(row)
                context_id = record.id

                # Fetch linked session IDs
                cursor.execute(
                    "SELECT session_id FROM agent_context_sessions WHERE context_id = ?",
                    (context_id,),
                )
                record.session_ids = [r[0] for r in cursor.fetchall()]

                # Fetch linked event IDs
                cursor.execute(
                    "SELECT event_id FROM agent_context_events WHERE context_id = ?",
                    (context_id,),
                )
                record.event_ids = [r[0] for r in cursor.fetchall()]

                contexts.append(record)
            return contexts
        except sqlite3.OperationalError:
            return []

    def delete_agent_context(self, context_id: str) -> bool:
        """Delete an agent context and all its links."""
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM agent_context_sessions WHERE context_id = ?", (context_id,))
            conn.execute("DELETE FROM agent_context_events WHERE context_id = ?", (context_id,))
            cursor = conn.execute("DELETE FROM agent_contexts WHERE id = ?", (context_id,))
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.OperationalError:
            conn.rollback()
            return False

    def link_session_to_agent_context(self, context_id: str, session_id: str) -> bool:
        """Link a session to an agent context.

        Silently skips if session doesn't exist (FK constraint).
        """
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO agent_context_sessions (context_id, session_id)
                VALUES (?, ?)
                """,
                (context_id, session_id),
            )
            conn.commit()
            return True
        except (sqlite3.OperationalError, sqlite3.IntegrityError):
            conn.rollback()
            return False

    def unlink_session_from_agent_context(self, context_id: str, session_id: str) -> bool:
        """Unlink a session from an agent context."""
        conn = self._get_connection()
        try:
            conn.execute(
                "DELETE FROM agent_context_sessions WHERE context_id = ? AND session_id = ?",
                (context_id, session_id),
            )
            conn.commit()
            return True
        except (sqlite3.OperationalError, sqlite3.IntegrityError):
            conn.rollback()
            return False

    def link_event_to_agent_context(self, context_id: str, event_id: str) -> bool:
        """Link an event to an agent context.

        Silently skips if event doesn't exist (FK constraint).
        """
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO agent_context_events (context_id, event_id)
                VALUES (?, ?)
                """,
                (context_id, event_id),
            )
            conn.commit()
            return True
        except (sqlite3.OperationalError, sqlite3.IntegrityError):
            conn.rollback()
            return False

    def unlink_event_from_agent_context(self, context_id: str, event_id: str) -> bool:
        """Unlink an event from an agent context."""
        conn = self._get_connection()
        try:
            conn.execute(
                "DELETE FROM agent_context_events WHERE context_id = ? AND event_id = ?",
                (context_id, event_id),
            )
            conn.commit()
            return True
        except (sqlite3.OperationalError, sqlite3.IntegrityError):
            conn.rollback()
            return False

    def set_agent_context_sessions(self, context_id: str, session_ids: List[str]) -> bool:
        """Replace all sessions linked to a context.

        Silently skips sessions that don't exist in DB (FK constraint).
        """
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM agent_context_sessions WHERE context_id = ?", (context_id,))
            if session_ids:
                # Insert one at a time to skip FK failures
                for sid in session_ids:
                    try:
                        conn.execute(
                            "INSERT OR IGNORE INTO agent_context_sessions (context_id, session_id) VALUES (?, ?)",
                            (context_id, sid),
                        )
                    except sqlite3.IntegrityError:
                        # Session doesn't exist in DB, skip
                        pass
            conn.commit()
            return True
        except (sqlite3.OperationalError, sqlite3.IntegrityError):
            conn.rollback()
            return False

    def set_agent_context_events(self, context_id: str, event_ids: List[str]) -> bool:
        """Replace all events linked to a context.

        Silently skips events that don't exist in DB (FK constraint).
        """
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM agent_context_events WHERE context_id = ?", (context_id,))
            if event_ids:
                # Insert one at a time to skip FK failures
                for eid in event_ids:
                    try:
                        conn.execute(
                            "INSERT OR IGNORE INTO agent_context_events (context_id, event_id) VALUES (?, ?)",
                            (context_id, eid),
                        )
                    except sqlite3.IntegrityError:
                        # Event doesn't exist in DB, skip
                        pass
            conn.commit()
            return True
        except (sqlite3.OperationalError, sqlite3.IntegrityError):
            conn.rollback()
            return False

    # Helper methods for row mapping
    def _row_to_project(self, row: sqlite3.Row) -> ProjectRecord:
        return ProjectRecord(
            id=row["id"],
            name=row["name"],
            path=Path(row["path"]),
            created_at=self._parse_datetime(row["created_at"]),
            updated_at=self._parse_datetime(row["updated_at"]),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def _row_to_session(self, row: sqlite3.Row) -> SessionRecord:
        # Handle both V1 (project_id) and V2 (workspace_path) schemas
        workspace_path = None
        try:
            workspace_path = row["workspace_path"]
        except (IndexError, KeyError):
            # V1 schema - try to get from project_id lookup if needed
            pass

        # V3 fields: session summary
        session_title = None
        session_summary = None
        summary_updated_at = None
        try:
            session_title = row["session_title"]
            session_summary = row["session_summary"]
            summary_updated_at = (
                self._parse_datetime(row["summary_updated_at"])
                if row["summary_updated_at"]
                else None
            )
        except (IndexError, KeyError):
            # V2 schema - these fields don't exist yet
            pass

        summary_status = None
        summary_locked_until = None
        summary_error = None
        try:
            summary_status = row["summary_status"]
            summary_locked_until = (
                self._parse_datetime(row["summary_locked_until"])
                if row["summary_locked_until"]
                else None
            )
            summary_error = row["summary_error"]
        except (IndexError, KeyError):
            pass

        # V18: user identity fields
        created_by = None
        shared_by = None
        try:
            created_by = row["created_by"]
        except (IndexError, KeyError):
            pass
        try:
            shared_by = row["shared_by"]
        except (IndexError, KeyError):
            pass

        # V10: total_turns cache
        total_turns = None
        try:
            total_turns = row["total_turns"]
        except (IndexError, KeyError):
            pass

        # V12: total_turns_mtime for cache validation
        total_turns_mtime = None
        try:
            total_turns_mtime = row["total_turns_mtime"]
        except (IndexError, KeyError):
            pass

        # V19: agent association
        agent_id = None
        try:
            agent_id = row["agent_id"]
        except (IndexError, KeyError):
            pass

        return SessionRecord(
            id=row["id"],
            session_file_path=Path(row["session_file_path"]),
            session_type=row["session_type"],
            started_at=self._parse_datetime(row["started_at"]),
            last_activity_at=self._parse_datetime(row["last_activity_at"]),
            created_at=self._parse_datetime(row["created_at"]),
            updated_at=self._parse_datetime(row["updated_at"]),
            metadata=json.loads(row["metadata"] or "{}"),
            workspace_path=workspace_path,
            session_title=session_title,
            session_summary=session_summary,
            summary_updated_at=summary_updated_at,
            summary_status=summary_status,
            summary_locked_until=summary_locked_until,
            summary_error=summary_error,
            created_by=created_by,
            shared_by=shared_by,
            total_turns=total_turns,
            total_turns_mtime=total_turns_mtime,
            agent_id=agent_id,
        )

    def _row_to_turn(self, row: sqlite3.Row) -> TurnRecord:
        temp_title = None
        try:
            temp_title = row["temp_title"]
        except (IndexError, KeyError):
            temp_title = None

        return TurnRecord(
            id=row["id"],
            session_id=row["session_id"],
            turn_number=row["turn_number"],
            user_message=row["user_message"],
            assistant_summary=row["assistant_summary"],
            turn_status=row["turn_status"],
            llm_title=row["llm_title"],
            llm_description=row["llm_description"],
            model_name=row["model_name"],
            if_last_task=row["if_last_task"],
            satisfaction=row["satisfaction"],
            content_hash=row["content_hash"],
            timestamp=self._parse_datetime(row["timestamp"]),
            created_at=self._parse_datetime(row["created_at"]),
            git_commit_hash=row["git_commit_hash"],
            temp_title=temp_title,
        )

    def _parse_datetime(self, val: Any) -> datetime:
        if isinstance(val, datetime):
            return val
        if not val:
            return datetime.now()
        try:
            return datetime.fromisoformat(val)
        except ValueError:
            # Handle potential format variations if needed
            return datetime.now()

    def _row_to_event(self, row: sqlite3.Row) -> EventRecord:
        """Convert row to EventRecord, fetching commits lazily."""
        # For simplicity in this iteration, we don't fetch commits here to avoid N+1.
        # Use get_commits_for_event if needed specifically.
        # But wait, our usage pattern in search likely wants commits to verify.
        # Let's fetch them for now as typically we fetch few events.

        commits = []
        try:
            # We need a new cursor or connection safe way if nested,
            # but here we are in same thread.
            # Ideally we should do a JOIN or separate query.
            # For now, let's just do a separate query.
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("SELECT commit_hash FROM event_commits WHERE event_id = ?", (row["id"],))
            commits = [r[0] for r in cur.fetchall()]
        except Exception:
            pass  # Indicate no commits or error

        event_type = row["event_type"]
        if event_type == "task":
            event_type = "user"

        # Parse preset_questions from JSON (handle missing column for pre-V5 DBs)
        preset_questions = None
        slack_message = None
        share_url = None

        try:
            if row["preset_questions"]:
                preset_questions = json.loads(row["preset_questions"])
        except (KeyError, json.JSONDecodeError, TypeError):
            preset_questions = None

        try:
            slack_message = row["slack_message"]
        except KeyError:
            slack_message = None

        try:
            share_url = row["share_url"]
        except KeyError:
            share_url = None

        share_id = None
        share_admin_token = None
        share_expiry_at = None
        try:
            share_id = row["share_id"]
            share_admin_token = row["share_admin_token"]
            expiry_raw = row["share_expiry_at"]
            share_expiry_at = self._parse_datetime(expiry_raw) if expiry_raw else None
        except KeyError:
            # Pre-V14 DBs won't have these columns.
            share_id = None
            share_admin_token = None
            share_expiry_at = None

        # V18: user identity fields
        created_by = None
        shared_by = None
        try:
            created_by = row["created_by"]
        except KeyError:
            pass
        try:
            shared_by = row["shared_by"]
        except KeyError:
            pass

        return EventRecord(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            event_type=event_type,
            status=row["status"],
            start_timestamp=self._parse_datetime(row["start_timestamp"]),
            end_timestamp=self._parse_datetime(row["end_timestamp"]),
            created_at=self._parse_datetime(row["created_at"]),
            updated_at=self._parse_datetime(row["updated_at"]),
            metadata=json.loads(row["metadata"] or "{}"),
            commit_hashes=commits,
            preset_questions=preset_questions,
            slack_message=slack_message,
            share_url=share_url,
            share_id=share_id,
            share_admin_token=share_admin_token,
            share_expiry_at=share_expiry_at,
            created_by=created_by,
            shared_by=shared_by,
        )

    def _row_to_agent(self, row: sqlite3.Row) -> AgentRecord:
        """Convert a database row to an AgentRecord."""
        # V18: user identity field
        created_by = None
        try:
            created_by = row["created_by"]
        except (IndexError, KeyError):
            pass

        return AgentRecord(
            id=row["id"],
            provider=row["provider"],
            session_type=row["session_type"],
            session_id=row["session_id"],
            context_id=row["context_id"],
            transcript_path=row["transcript_path"],
            cwd=row["cwd"],
            project_dir=row["project_dir"],
            status=row["status"] or "active",
            attention=row["attention"],
            source=row["source"],
            created_at=self._parse_datetime(row["created_at"]),
            updated_at=self._parse_datetime(row["updated_at"]),
            created_by=created_by,
        )

    def _row_to_agent_context(self, row: sqlite3.Row) -> AgentContextRecord:
        """Convert a database row to an AgentContextRecord."""
        metadata = None
        try:
            metadata_raw = row["metadata"]
            if metadata_raw:
                metadata = json.loads(metadata_raw)
        except (IndexError, KeyError, json.JSONDecodeError):
            pass

        return AgentContextRecord(
            id=row["id"],
            workspace=row["workspace"],
            loaded_at=row["loaded_at"],
            created_at=self._parse_datetime(row["created_at"]),
            updated_at=self._parse_datetime(row["updated_at"]),
            metadata=metadata,
            session_ids=None,  # Populated separately
            event_ids=None,  # Populated separately
        )

    # -------------------------------------------------------------------------
    # Users table methods (Schema V18)
    # -------------------------------------------------------------------------

    def upsert_user(
        self,
        uid: str,
        user_name: Optional[str] = None,
        user_email: Optional[str] = None,
    ) -> None:
        """Insert or update a user in the users table."""
        if not uid:
            return
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO users (uid, user_name, user_email, created_at, updated_at)
                VALUES (?, ?, ?, datetime('now'), datetime('now'))
                ON CONFLICT(uid) DO UPDATE SET
                    user_name = COALESCE(excluded.user_name, users.user_name),
                    user_email = COALESCE(excluded.user_email, users.user_email),
                    updated_at = datetime('now')
                """,
                (uid, user_name, user_email),
            )
            conn.commit()
        except sqlite3.OperationalError:
            # Older schema without users.user_email column.
            try:
                conn.rollback()
            except Exception:
                pass
            try:
                conn.execute(
                    """
                    INSERT INTO users (uid, user_name, created_at, updated_at)
                    VALUES (?, ?, datetime('now'), datetime('now'))
                    ON CONFLICT(uid) DO UPDATE SET
                        user_name = COALESCE(excluded.user_name, users.user_name),
                        updated_at = datetime('now')
                    """,
                    (uid, user_name),
                )
                conn.commit()
            except sqlite3.OperationalError:
                # Older schema without users table
                try:
                    conn.rollback()
                except Exception:
                    pass

    def get_user(self, uid: str) -> Optional[UserRecord]:
        """Get a user by UID from the users table."""
        if not uid:
            return None
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT * FROM users WHERE uid = ?", (uid,))
            row = cursor.fetchone()
            if row:
                keys = row.keys() if hasattr(row, "keys") else []
                return UserRecord(
                    uid=row["uid"],
                    user_name=row["user_name"],
                    user_email=row["user_email"] if "user_email" in keys else None,
                    created_at=self._parse_datetime(row["created_at"]),
                    updated_at=self._parse_datetime(row["updated_at"]),
                )
        except sqlite3.OperationalError:
            pass
        return None

    # -------------------------------------------------------------------------
    # Agent info table methods (Schema V20)
    # -------------------------------------------------------------------------

    def _row_to_agent_info(self, row: sqlite3.Row) -> AgentInfoRecord:
        """Convert a database row to an AgentInfoRecord."""
        import json

        keys = row.keys()
        visibility = "visible"
        try:
            if "visibility" in keys:
                visibility = row["visibility"] or "visible"
        except Exception:
            visibility = "visible"

        def _safe_get(key: str, default=None):
            try:
                return row[key] if key in keys else default
            except Exception:
                return default

        preset_questions = None
        try:
            raw = _safe_get("preset_questions")
            if raw:
                preset_questions = json.loads(raw)
        except Exception:
            preset_questions = None

        return AgentInfoRecord(
            id=row["id"],
            name=row["name"],
            title=_safe_get("title", ""),
            description=row["description"] or "",
            created_at=self._parse_datetime(row["created_at"]),
            updated_at=self._parse_datetime(row["updated_at"]),
            visibility=visibility,
            slack_message=_safe_get("slack_message"),
            preset_questions=preset_questions,
            share_id=_safe_get("share_id"),
            share_url=_safe_get("share_url"),
            share_admin_token=_safe_get("share_admin_token"),
            share_contributor_token=_safe_get("share_contributor_token"),
            share_expiry_at=_safe_get("share_expiry_at"),
            last_synced_at=_safe_get("last_synced_at"),
            sync_version=_safe_get("sync_version", 0) or 0,
        )

    def get_or_create_agent_info(
        self, agent_id: str, name: Optional[str] = None
    ) -> AgentInfoRecord:
        """Get existing agent info or create a new one.

        Args:
            agent_id: UUID for the agent.
            name: Display name (if None, caller should provide a generated name).
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM agent_info WHERE id = ?", (agent_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_agent_info(row)
        except sqlite3.OperationalError:
            pass

        display_name = name or agent_id[:8]
        cursor.execute(
            """
            INSERT INTO agent_info (id, name, description, visibility, created_at, updated_at)
            VALUES (?, ?, '', 'visible', datetime('now'), datetime('now'))
            """,
            (agent_id, display_name),
        )
        conn.commit()

        cursor.execute("SELECT * FROM agent_info WHERE id = ?", (agent_id,))
        row = cursor.fetchone()
        return self._row_to_agent_info(row)

    def get_agent_info(self, agent_id: str) -> Optional[AgentInfoRecord]:
        """Get agent info by ID, or None if not found."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT * FROM agent_info WHERE id = ?", (agent_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_agent_info(row)
        except sqlite3.OperationalError:
            pass
        return None

    def update_agent_info(
        self,
        agent_id: str,
        *,
        name: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        visibility: Optional[str] = None,
    ) -> Optional[AgentInfoRecord]:
        """Update agent info fields. Returns updated record or None if not found."""
        conn = self._get_connection()
        sets: list[str] = []
        params: list[Any] = []

        if name is not None:
            sets.append("name = ?")
            params.append(name)
        if title is not None:
            sets.append("title = ?")
            params.append(title)
        if description is not None:
            sets.append("description = ?")
            params.append(description)
        if visibility is not None:
            sets.append("visibility = ?")
            params.append(visibility)

        if not sets:
            return self.get_agent_info(agent_id)

        sets.append("updated_at = datetime('now')")
        params.append(agent_id)

        try:
            conn.execute(
                f"UPDATE agent_info SET {', '.join(sets)} WHERE id = ?",
                params,
            )
            conn.commit()
        except sqlite3.OperationalError:
            return None

        return self.get_agent_info(agent_id)

    def list_agent_info(self, *, include_invisible: bool = False) -> list[AgentInfoRecord]:
        """List agent info records, ordered by updated_at descending."""
        conn = self._get_connection()
        try:
            if include_invisible:
                try:
                    cursor = conn.execute(
                        "SELECT * FROM agent_info ORDER BY updated_at DESC, created_at DESC"
                    )
                except sqlite3.OperationalError:
                    cursor = conn.execute("SELECT * FROM agent_info ORDER BY created_at DESC")
            else:
                try:
                    try:
                        cursor = conn.execute(
                            "SELECT * FROM agent_info WHERE visibility = 'visible' "
                            "ORDER BY updated_at DESC, created_at DESC"
                        )
                    except sqlite3.OperationalError:
                        cursor = conn.execute(
                            "SELECT * FROM agent_info WHERE visibility = 'visible' "
                            "ORDER BY created_at DESC"
                        )
                except sqlite3.OperationalError:
                    try:
                        cursor = conn.execute(
                            "SELECT * FROM agent_info ORDER BY updated_at DESC, created_at DESC"
                        )
                    except sqlite3.OperationalError:
                        cursor = conn.execute("SELECT * FROM agent_info ORDER BY created_at DESC")
            return [self._row_to_agent_info(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            return []

    def update_agent_sync_metadata(
        self,
        agent_id: str,
        *,
        share_id: Optional[str] = None,
        share_url: Optional[str] = None,
        share_admin_token: Optional[str] = None,
        share_contributor_token: Optional[str] = None,
        share_expiry_at: Optional[str] = None,
        slack_message: Optional[str] = None,
        preset_questions: Optional[List[str]] = None,
        last_synced_at: Optional[str] = None,
        sync_version: Optional[int] = None,
    ) -> Optional[AgentInfoRecord]:
        """Update sync-related metadata on an agent_info record."""
        import json

        conn = self._get_connection()
        sets: list[str] = []
        params: list[Any] = []

        if share_id is not None:
            sets.append("share_id = ?")
            params.append(share_id)
        if share_url is not None:
            sets.append("share_url = ?")
            params.append(share_url)
        if share_admin_token is not None:
            sets.append("share_admin_token = ?")
            params.append(share_admin_token)
        if share_contributor_token is not None:
            sets.append("share_contributor_token = ?")
            params.append(share_contributor_token)
        if share_expiry_at is not None:
            sets.append("share_expiry_at = ?")
            params.append(share_expiry_at)
        if slack_message is not None:
            sets.append("slack_message = ?")
            params.append(slack_message)
        if preset_questions is not None:
            sets.append("preset_questions = ?")
            params.append(json.dumps(preset_questions))
        if last_synced_at is not None:
            sets.append("last_synced_at = ?")
            params.append(last_synced_at)
        if sync_version is not None:
            sets.append("sync_version = ?")
            params.append(sync_version)

        if not sets:
            return self.get_agent_info(agent_id)

        sets.append("updated_at = datetime('now')")
        params.append(agent_id)

        try:
            conn.execute(
                f"UPDATE agent_info SET {', '.join(sets)} WHERE id = ?",
                params,
            )
            conn.commit()
        except sqlite3.OperationalError:
            return None

        return self.get_agent_info(agent_id)

    def get_agent_content_hashes(self, agent_id: str) -> set[str]:
        """Return content hashes for turns linked to this agent (V28 mapping + fallback)."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT DISTINCT t.content_hash
                FROM turns t
                JOIN sessions s ON t.session_id = s.id
                WHERE s.agent_id = ?
                   OR EXISTS (
                       SELECT 1
                       FROM agent_sessions links
                       WHERE links.agent_id = ?
                         AND links.session_id = s.id
                   )
                """,
                (agent_id, agent_id),
            )
            return {row["content_hash"] for row in cursor.fetchall()}
        except sqlite3.OperationalError:
            try:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT t.content_hash
                    FROM turns t
                    JOIN sessions s ON t.session_id = s.id
                    WHERE s.agent_id = ?
                    """,
                    (agent_id,),
                )
                return {row["content_hash"] for row in cursor.fetchall()}
            except sqlite3.OperationalError:
                return set()
