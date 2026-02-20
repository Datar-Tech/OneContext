"""Context management for limiting search scope.

This module manages context entries that can be used to limit search scope.
Data is stored in:
1. SQLite database (primary storage, V15+)
2. ~/.aline/load.json (backward compatibility fallback)
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Default path for load.json
LOAD_JSON_PATH = Path.home() / ".aline" / "load.json"

# Environment variable for context ID
CONTEXT_ID_ENV_VAR = "ALINE_CONTEXT_ID"


def _get_db():
    """Get database connection (lazy import to avoid circular deps)."""
    try:
        from .db import get_database

        return get_database(read_only=False)
    except Exception:
        return None


def _get_db_readonly():
    """Get read-only database connection."""
    try:
        from .db import get_database

        return get_database(read_only=True)
    except Exception:
        return None


@dataclass
class ContextEntry:
    """A single context entry in load.json."""

    context_sessions: List[str] = field(default_factory=list)
    context_events: List[str] = field(default_factory=list)
    context_id: Optional[str] = None
    workspace: Optional[str] = None
    loaded_at: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "context_sessions": self.context_sessions,
            "context_events": self.context_events,
        }
        if self.context_id:
            result["context_id"] = self.context_id
        if self.workspace:
            result["workspace"] = self.workspace
        if self.loaded_at:
            result["loaded_at"] = self.loaded_at
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ContextEntry":
        """Create from dictionary."""
        return cls(
            context_sessions=data.get("context_sessions", []),
            context_events=data.get("context_events", []),
            context_id=data.get("context_id"),
            workspace=data.get("workspace"),
            loaded_at=data.get("loaded_at"),
        )


@dataclass
class ContextConfig:
    """Complete context configuration from load.json."""

    contexts: List[ContextEntry] = field(default_factory=list)
    default: Optional[ContextEntry] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {"contexts": [c.to_dict() for c in self.contexts]}
        if self.default:
            result["default"] = self.default.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ContextConfig":
        """Create from dictionary."""
        contexts = [ContextEntry.from_dict(c) for c in data.get("contexts", [])]
        default_data = data.get("default")
        default = ContextEntry.from_dict(default_data) if default_data else None
        return cls(contexts=contexts, default=default)


def _load_context_config_from_db() -> Optional[ContextConfig]:
    """Load context configuration from database (best-effort)."""
    try:
        db = _get_db_readonly()
        if not db:
            return None

        db_contexts = db.list_agent_contexts(limit=100)
        if not db_contexts:
            return None

        contexts = []
        for ctx in db_contexts:
            entry = ContextEntry(
                context_sessions=ctx.session_ids or [],
                context_events=ctx.event_ids or [],
                context_id=ctx.id,
                workspace=ctx.workspace,
                loaded_at=ctx.loaded_at,
            )
            contexts.append(entry)

        return ContextConfig(contexts=contexts)
    except Exception:
        return None


def load_context_config(path: Optional[Path] = None) -> Optional[ContextConfig]:
    """Load context configuration.

    Priority:
    1. SQLite database (primary storage, V15+)
    2. ~/.aline/load.json (fallback for backward compatibility)

    Args:
        path: Optional path to load.json. Defaults to ~/.aline/load.json

    Returns:
        ContextConfig if data exists and is valid, None otherwise.
    """
    # Phase 1: Try to load from database
    db_config = _load_context_config_from_db()
    if db_config and db_config.contexts:
        return db_config

    # Phase 2: Fall back to JSON file
    config_path = path or LOAD_JSON_PATH

    if not config_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ContextConfig.from_dict(data)
    except (json.JSONDecodeError, IOError):
        return None


def _sync_context_to_db(entry: ContextEntry) -> bool:
    """Sync a context entry to the database (best-effort)."""
    if not entry.context_id:
        return False

    try:
        db = _get_db()
        if not db:
            return False

        # Get or create the context
        existing = db.get_agent_context_by_id(entry.context_id)
        if existing:
            db.update_agent_context(
                entry.context_id,
                workspace=entry.workspace,
                loaded_at=entry.loaded_at,
            )
        else:
            db.get_or_create_agent_context(
                entry.context_id,
                workspace=entry.workspace,
                loaded_at=entry.loaded_at,
            )

        # Update session links
        if entry.context_sessions:
            db.set_agent_context_sessions(entry.context_id, entry.context_sessions)
        else:
            db.set_agent_context_sessions(entry.context_id, [])

        # Update event links
        if entry.context_events:
            db.set_agent_context_events(entry.context_id, entry.context_events)
        else:
            db.set_agent_context_events(entry.context_id, [])

        # Note: Don't close - get_database() returns a singleton
        return True
    except Exception:
        return False


def save_context_config(config: ContextConfig, path: Optional[Path] = None) -> bool:
    """Save context configuration.

    Writes to both:
    1. SQLite database (primary storage, V15+)
    2. ~/.aline/load.json (backward compatibility fallback)

    Args:
        config: ContextConfig to save
        path: Optional path to load.json. Defaults to ~/.aline/load.json

    Returns:
        True if successful, False otherwise.
    """
    # Phase 1: Sync all contexts to database (best-effort)
    for entry in config.contexts:
        _sync_context_to_db(entry)
    if config.default and config.default.context_id:
        _sync_context_to_db(config.default)

    # Phase 2: Write to JSON (backward compatibility)
    config_path = path or LOAD_JSON_PATH

    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)
        return True
    except IOError:
        return False


def resolve_current_context(
    config: Optional[ContextConfig] = None,
    context_id: Optional[str] = None,
    workspace: Optional[str] = None,
) -> Optional[ContextEntry]:
    """Resolve the current context based on priority.

    Priority order:
    1. ALINE_CONTEXT_ID environment variable -> match context_id
    2. Current working directory -> match workspace
    3. Use default
    4. Return None if all empty

    Args:
        config: ContextConfig to search. If None, loads from default path.
        context_id: Override context_id (for testing). Uses env var if None.
        workspace: Override workspace (for testing). Uses cwd if None.

    Returns:
        Matching ContextEntry or None.
    """
    if config is None:
        config = load_context_config()

    if config is None:
        return None

    # 1. Check ALINE_CONTEXT_ID environment variable
    env_context_id = context_id or os.environ.get(CONTEXT_ID_ENV_VAR)
    if env_context_id:
        for entry in config.contexts:
            if entry.context_id == env_context_id:
                return entry

    # 2. Check current working directory
    cwd = workspace or os.getcwd()
    for entry in config.contexts:
        if entry.workspace and entry.workspace == cwd:
            return entry

    # 3. Use default
    if config.default:
        return config.default

    # 4. No context found
    return None


def get_context_session_ids(
    config: Optional[ContextConfig] = None,
) -> Optional[List[str]]:
    """Get session IDs from the current context.

    Returns:
        List of session IDs if context exists and has sessions, None otherwise.
    """
    context = resolve_current_context(config)
    if context and context.context_sessions:
        return context.context_sessions
    return None


def get_context_event_ids(
    config: Optional[ContextConfig] = None,
) -> Optional[List[str]]:
    """Get event IDs from the current context.

    Returns:
        List of event IDs if context exists and has events, None otherwise.
    """
    context = resolve_current_context(config)
    if context and context.context_events:
        return context.context_events
    return None


def add_context(
    sessions: Optional[List[str]] = None,
    events: Optional[List[str]] = None,
    context_id: Optional[str] = None,
    workspace: Optional[str] = None,
    config: Optional[ContextConfig] = None,
    path: Optional[Path] = None,
) -> ContextEntry:
    """Add or update a context in the configuration.

    If context_id or workspace is provided, updates existing or creates new.
    Otherwise, uses current workspace.

    Args:
        sessions: Session IDs to add
        events: Event IDs to add
        context_id: Context ID for this entry
        workspace: Workspace path for this entry
        config: Existing config to update. If None, loads from file.
        path: Path to load.json

    Returns:
        The created or updated ContextEntry.
    """
    if config is None:
        config = load_context_config(path) or ContextConfig()

    # Determine the key for this context
    use_context_id = context_id
    use_workspace = workspace or (os.getcwd() if not context_id else None)

    # Find existing entry
    existing_idx = None
    for i, entry in enumerate(config.contexts):
        if use_context_id and entry.context_id == use_context_id:
            existing_idx = i
            break
        if use_workspace and entry.workspace == use_workspace:
            existing_idx = i
            break

    # Create or update entry
    if existing_idx is not None:
        entry = config.contexts[existing_idx]
        # Add new sessions/events (merge, deduplicate)
        if sessions:
            entry.context_sessions = list(set(entry.context_sessions) | set(sessions))
        if events:
            entry.context_events = list(set(entry.context_events) | set(events))
        entry.loaded_at = datetime.utcnow().isoformat() + "Z"
    else:
        entry = ContextEntry(
            context_sessions=sessions or [],
            context_events=events or [],
            context_id=use_context_id,
            workspace=use_workspace,
            loaded_at=datetime.utcnow().isoformat() + "Z",
        )
        config.contexts.append(entry)

    save_context_config(config, path)
    return entry


def clear_context(
    context_id: Optional[str] = None,
    workspace: Optional[str] = None,
    config: Optional[ContextConfig] = None,
    path: Optional[Path] = None,
) -> bool:
    """Clear a context from the configuration.

    If no context_id or workspace provided, clears based on current env/cwd.

    Args:
        context_id: Context ID to clear
        workspace: Workspace to clear
        config: Existing config to update
        path: Path to load.json

    Returns:
        True if a context was cleared, False if not found.
    """
    if config is None:
        config = load_context_config(path)

    if config is None:
        return False

    # Determine what to clear
    env_context_id = context_id or os.environ.get(CONTEXT_ID_ENV_VAR)
    cwd = workspace or os.getcwd()

    # Find and remove
    original_len = len(config.contexts)
    config.contexts = [
        c
        for c in config.contexts
        if not (
            (env_context_id and c.context_id == env_context_id)
            or (not env_context_id and c.workspace == cwd)
        )
    ]

    if len(config.contexts) < original_len:
        save_context_config(config, path)
        return True

    return False


def get_context_by_id(
    context_id: str,
    config: Optional[ContextConfig] = None,
    path: Optional[Path] = None,
) -> Optional[ContextEntry]:
    """Get a specific context by its ID.

    Priority:
    1. SQLite database (primary storage, V15+)
    2. In-memory config if provided
    3. ~/.aline/load.json (fallback)

    Args:
        context_id: The context ID to look up
        config: Existing config to search
        path: Path to load.json

    Returns:
        The ContextEntry if found, None otherwise.
    """
    # Phase 1: Try to load from database
    try:
        db = _get_db_readonly()
        if db:
            ctx = db.get_agent_context_by_id(context_id)
            if ctx:
                return ContextEntry(
                    context_sessions=ctx.session_ids or [],
                    context_events=ctx.event_ids or [],
                    context_id=ctx.id,
                    workspace=ctx.workspace,
                    loaded_at=ctx.loaded_at,
                )
    except Exception:
        pass

    # Phase 2: Check in-memory config or load from JSON
    if config is None:
        config = load_context_config(path)

    if config is None:
        return None

    for entry in config.contexts:
        if entry.context_id == context_id:
            return entry

    return None
