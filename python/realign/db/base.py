"""
Abstract base classes and data models for ReAlign database interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any, Dict
import uuid

MAX_SESSION_NUM = 1000


class AgentSessionLimitExceededError(RuntimeError):
    """Raised when attempting to link a new session beyond the per-agent cap."""

    def __init__(self, agent_id: str, limit: int = MAX_SESSION_NUM):
        self.agent_id = agent_id
        self.limit = int(limit)
        super().__init__(
            f"Agent '{agent_id}' already has {self.limit} linked sessions (max={self.limit})"
        )


@dataclass
class LockRecord:
    lock_key: str
    owner: str
    locked_until: datetime
    created_at: datetime
    updated_at: datetime
    metadata: Optional[str] = None


@dataclass
class ProjectRecord:
    id: str
    name: str
    path: Path
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


@dataclass
class SessionRecord:
    id: str
    session_file_path: Path
    session_type: str
    started_at: datetime
    last_activity_at: datetime
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    workspace_path: Optional[str] = None  # V2: optional workspace context
    # V3 fields: session summary
    session_title: Optional[str] = None
    session_summary: Optional[str] = None
    summary_updated_at: Optional[datetime] = None
    # V7: session summary runtime status
    summary_status: Optional[str] = None
    summary_locked_until: Optional[datetime] = None
    summary_error: Optional[str] = None
    # V18: user identity
    created_by: Optional[str] = None  # Creator UID
    shared_by: Optional[str] = None  # Sharer UID (who imported this)
    # V10: cached total turn count for session list performance
    total_turns: Optional[int] = None
    # V12: file mtime when total_turns was cached (for validation)
    total_turns_mtime: Optional[float] = None
    # V19: agent association
    agent_id: Optional[str] = None


@dataclass
class TurnRecord:
    id: str
    session_id: str
    turn_number: int
    user_message: Optional[str]
    assistant_summary: Optional[str]
    turn_status: str
    llm_title: str
    llm_description: Optional[str]
    model_name: Optional[str]
    if_last_task: str
    satisfaction: str
    content_hash: str
    timestamp: datetime
    created_at: datetime
    git_commit_hash: Optional[str] = None
    # V12: temporary title stored in DB turns.temp_title
    temp_title: Optional[str] = None


@dataclass
class EventRecord:
    id: str
    title: str
    description: Optional[str]
    event_type: str  # 'task', 'temporal', etc.
    status: str  # 'active', 'frozen', 'archived'
    start_timestamp: Optional[datetime]
    end_timestamp: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    commit_hashes: Optional[List[str]] = None  # Populated when reading
    preset_questions: Optional[List[str]] = None  # LLM-generated preset questions
    slack_message: Optional[str] = None  # LLM-generated Slack share message
    share_url: Optional[str] = None  # Public share URL
    share_id: Optional[str] = None  # V14: Share ID on server (for reuse)
    share_admin_token: Optional[str] = None  # V14: Admin token for extending expiry
    share_expiry_at: Optional[datetime] = None  # V14: Last known expiry timestamp
    # V18: user identity
    created_by: Optional[str] = None  # Creator UID
    shared_by: Optional[str] = None  # Sharer UID (who imported this)


@dataclass
class AgentRecord:
    """Represents a terminal/agent mapping (V15: replaces terminal.json)."""

    id: str  # terminal_id (UUID)
    provider: str  # 'claude', 'codex', 'opencode', 'shell'
    session_type: str
    created_at: datetime
    updated_at: datetime
    session_id: Optional[str] = None  # FK to sessions.id
    context_id: Optional[str] = None
    transcript_path: Optional[str] = None
    cwd: Optional[str] = None
    project_dir: Optional[str] = None
    status: str = "active"  # 'active', 'stopped'
    attention: Optional[str] = None  # 'permission_request', 'stop', or None
    source: Optional[str] = None
    # V18: user identity
    created_by: Optional[str] = None  # Creator UID


@dataclass
class AgentInfoRecord:
    """Agent profile/identity data (V20, V22: sync fields)."""

    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    title: Optional[str] = ""
    description: Optional[str] = ""
    visibility: str = "visible"
    slack_message: Optional[str] = None
    preset_questions: Optional[List[str]] = None
    # V22: sync metadata
    share_id: Optional[str] = None
    share_url: Optional[str] = None
    share_admin_token: Optional[str] = None
    share_contributor_token: Optional[str] = None
    share_expiry_at: Optional[str] = None
    last_synced_at: Optional[str] = None
    sync_version: int = 0


@dataclass
class WindowLinkRecord:
    """Represents a terminal/session association (V23)."""

    terminal_id: str
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    provider: Optional[str] = None
    source: Optional[str] = None
    ts: Optional[float] = None
    created_at: Optional[datetime] = None


@dataclass
class AgentContextRecord:
    """Represents a context entry (V15: replaces load.json)."""

    id: str  # context_id
    created_at: datetime
    updated_at: datetime
    workspace: Optional[str] = None
    loaded_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    # Populated when reading (from M2M tables)
    session_ids: Optional[List[str]] = None
    event_ids: Optional[List[str]] = None


@dataclass
class UserRecord:
    """Represents a user in the users table (V18)."""

    uid: str
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DatabaseInterface(ABC):
    """Abstract interface for ReAlign storage backend."""

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the database (create tables, migrations)."""
        pass

    @abstractmethod
    def get_project_by_path(self, path: Path) -> Optional[ProjectRecord]:
        """Get a project by its absolute path, or None if not found."""
        pass

    @abstractmethod
    def get_or_create_project(self, path: Path, name: Optional[str] = None) -> ProjectRecord:
        """Get existing project or create new one."""
        pass

    @abstractmethod
    def get_or_create_session(
        self,
        session_id: str,
        session_file_path: Path,
        session_type: str,
        started_at: datetime,
        workspace_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionRecord:
        """Get existing session or create new one."""
        pass

    @abstractmethod
    def update_session_activity(self, session_id: str, last_activity_at: datetime) -> None:
        """Update last activity timestamp for a session."""
        pass

    @abstractmethod
    def update_session_summary_runtime(
        self,
        session_id: str,
        summary_status: str,
        summary_locked_until: Optional[datetime] = None,
        summary_error: Optional[str] = None,
    ) -> None:
        """Update session summary runtime status (V7)."""
        pass

    @abstractmethod
    def update_session_total_turns(self, session_id: str, total_turns: int) -> None:
        """Update session's cached total turn count (V10)."""
        pass

    @abstractmethod
    def update_session_total_turns_with_mtime(
        self, session_id: str, total_turns: int, mtime: float
    ) -> None:
        """Update session's cached total turn count with file mtime for validation (V12)."""
        pass

    @abstractmethod
    def backfill_session_total_turns(self) -> int:
        """Backfill total_turns for all sessions from turns table (V10 migration)."""
        pass

    @abstractmethod
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
        """Insert a window link record (V23)."""
        pass

    @abstractmethod
    def list_latest_window_links(
        self, *, agent_id: Optional[str] = None, limit: int = 1000
    ) -> List[WindowLinkRecord]:
        """List latest window link per terminal (V23)."""
        pass

    @abstractmethod
    def list_sessions(
        self, limit: int = 100, workspace_path: Optional[str] = None
    ) -> List[SessionRecord]:
        """List sessions ordered by last activity (most recent first)."""
        pass

    @abstractmethod
    def get_session_by_id(self, session_id: str) -> Optional[SessionRecord]:
        """Get a session by its ID."""
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its associated data (turns, turn_content, event_sessions).

        Args:
            session_id: The session ID to delete

        Returns:
            True if session was deleted, False if not found or error
        """
        pass

    @abstractmethod
    def get_sessions_by_ids(self, session_ids: List[str]) -> List[SessionRecord]:
        """Get multiple sessions by their IDs in a single query.

        Args:
            session_ids: List of session IDs to fetch

        Returns:
            List of SessionRecord objects (order not guaranteed)
        """
        pass

    @abstractmethod
    def get_sessions_by_agent_id(
        self, agent_id: str, limit: int = MAX_SESSION_NUM
    ) -> List[SessionRecord]:
        """Get all sessions linked to an agent.

        Args:
            agent_id: The agent_info ID
            limit: Maximum number of sessions to return

        Returns:
            List of SessionRecord objects for this agent
        """
        pass

    @abstractmethod
    def get_session_ids_by_agent_id(self, agent_id: str, limit: int = MAX_SESSION_NUM) -> List[str]:
        """Get session IDs linked to an agent (lightweight lookup)."""
        pass

    @abstractmethod
    def get_agent_session_count(self, agent_id: str) -> int:
        """Get the number of distinct sessions linked to an agent."""
        pass

    @abstractmethod
    def can_link_session_to_agent(self, agent_id: str, session_id: str) -> bool:
        """Check if linking this session would respect the per-agent session cap."""
        pass

    @abstractmethod
    def is_session_linked_to_agent(self, agent_id: str, session_id: str) -> bool:
        """Return True when a session is linked to an agent."""
        pass

    @abstractmethod
    def link_session_to_agent(self, agent_id: str, session_id: str) -> bool:
        """Create a durable agent-session link (idempotent)."""
        pass

    @abstractmethod
    def get_agent_ids_by_session_id(self, session_id: str) -> List[str]:
        """List all agents linked to a session."""
        pass

    @abstractmethod
    def get_turn_by_hash(self, session_id: str, content_hash: str) -> Optional[TurnRecord]:
        """Check if a turn with this content hash already exists in the session."""
        pass

    @abstractmethod
    def get_turn_by_number(self, session_id: str, turn_number: int) -> Optional[TurnRecord]:
        """Get a turn by session_id and turn_number."""
        pass

    @abstractmethod
    def get_max_turn_number(self, session_id: str) -> int:
        """Get the maximum completed turn_number stored for a session, or 0 if none."""
        pass

    @abstractmethod
    def get_max_turn_numbers_batch(self, session_ids: List[str]) -> Dict[str, int]:
        """Get max turn numbers for multiple sessions in a single query.

        Args:
            session_ids: List of session IDs to query

        Returns:
            Dict mapping session_id -> max turn number (0 if none)
        """
        pass

    @abstractmethod
    def get_committed_turn_numbers(self, session_id: str) -> set[int]:
        """Get the set of turn numbers that have been committed for a session.

        This is used to detect gaps in turn numbers during catch-up.

        Args:
            session_id: The session ID to query

        Returns:
            Set of turn numbers that exist in the database for this session
        """
        pass

    @abstractmethod
    def create_turn(
        self, turn: TurnRecord, content: str, *, skip_session_summary: bool = False
    ) -> TurnRecord:
        """Create a new turn record and store its content."""
        pass

    @abstractmethod
    def get_completed_turn_count(self, session_id: str, *, up_to: Optional[int] = None) -> int:
        """Get the count of distinct completed turns for a session."""
        pass

    @abstractmethod
    def get_turn_content(self, turn_id: str) -> Optional[str]:
        """Get the JSONL content for a turn."""
        pass

    @abstractmethod
    def close(self):
        """Close database connections."""
        pass

    @abstractmethod
    def sync_events(self, events: List[EventRecord]) -> None:
        """Sync events to database."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_event_by_id(self, event_id: str) -> Optional[EventRecord]:
        """Get event by ID."""
        pass

    @abstractmethod
    def list_events(self, limit: int = 50, offset: int = 0) -> List[EventRecord]:
        """List all events, ordered by updated_at descending."""
        pass

    @abstractmethod
    def delete_event(self, event_id: str) -> bool:
        """Delete an event and its associations."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def count_search_conversations(
        self,
        query: str,
        use_regex: bool = False,
        ignore_case: bool = True,
        session_ids: Optional[List[str]] = None,
    ) -> int:
        """Count matching conversation turns by title and summary only."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def count_search_turn_content(
        self,
        query: str,
        use_regex: bool = False,
        ignore_case: bool = True,
        session_ids: Optional[List[str]] = None,
        turn_ids: Optional[List[str]] = None,
    ) -> int:
        """Count matching turn content rows."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def count_search_sessions(
        self,
        query: str,
        use_regex: bool = False,
        ignore_case: bool = True,
        session_ids: Optional[List[str]] = None,
    ) -> int:
        """Count matching sessions by title and summary."""
        pass

    @abstractmethod
    def try_acquire_lock(self, lock_key: str, *, owner: str, ttl_seconds: float) -> bool:
        """Try to acquire a cross-process lease lock."""
        pass

    @abstractmethod
    def release_lock(self, lock_key: str, *, owner: str) -> None:
        """Release a lease lock (best-effort)."""
        pass

    @abstractmethod
    def get_all_locks(self, include_expired: bool = False) -> List[LockRecord]:
        """Get all locks, optionally including expired ones."""
        pass

    @abstractmethod
    def upsert_user(
        self,
        uid: str,
        user_name: Optional[str] = None,
        user_email: Optional[str] = None,
    ) -> None:
        """Insert or update a user in the users table (V18)."""
        pass

    @abstractmethod
    def get_user(self, uid: str) -> Optional[UserRecord]:
        """Get a user by UID from the users table (V18)."""
        pass
