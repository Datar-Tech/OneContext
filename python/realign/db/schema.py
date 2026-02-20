"""
SQL schema definitions for ReAlign SQLite database.

Schema V2: Sessions are decoupled from projects/workspaces.
- projects table is now optional (for backward compatibility)
- sessions.project_id replaced with optional workspace_path
- git_commit_hash is optional (for backward compatibility with migrated data)

Schema V3: Session summary fields for hierarchical aggregation.
- sessions.session_title: Aggregated title from turns
- sessions.session_summary: Aggregated summary from turns
- sessions.summary_updated_at: Timestamp of last summary update

Schema V4: Event-Session relationship (many-to-many).

Schema V5: Event share metadata.
- events.preset_questions: JSON array of LLM-generated preset questions
- events.slack_message: LLM-generated Slack share message

Schema V6: Event share URL.
- events.share_url: Public share URL for the event

Schema V7: Session summary runtime status.
- sessions.summary_status: Status of summary generation
- sessions.summary_locked_until: Lease/TTL for processing
- sessions.summary_error: Error message if failed

Schema V8: Cross-process lease locks.
- locks table for synchronization

Schema V9: User identity tracking.
- sessions.creator_name: Username who created the session
- sessions.creator_id: User UUID (based on MAC address)
- events.creator_name: Username who created the event
- events.creator_id: User UUID
- turns.creator_name: Username who created the turn
- turns.creator_id: User UUID

Schema V10: Cache total_turns for session list performance.
- sessions.total_turns: Cached count of total turns (avoids reading files)

Schema V11: Durable background job queue.
- jobs table for turn/session summary workers

Schema V12: Lazy cache validation for total_turns.
- sessions.total_turns_mtime: File mtime when total_turns was cached

Schema V13: Temporary turn titles.
- turns.temp_title: LLM-generated temporary title before final summary

Schema V14: Share reuse per event.
- events.share_id: Share ID on server
- events.share_admin_token: Admin token for extending expiry
- events.share_expiry_at: Last known expiry timestamp

Schema V15: Agents and contexts tables (replaces terminal.json and load.json).
- agents table: terminal_id -> session mapping (replaces terminal.json)
- agent_contexts table: context definitions (replaces load.json)
- agent_context_sessions table: M2M context-session links
- agent_context_events table: M2M context-event links

Schema V16: Remove FK constraints from agent_context_sessions/events.
- Context may reference sessions/events not yet imported to DB
- Recreate M2M tables without FK constraints on session_id/event_id

Schema V17: Rename creator_id/creator_name to uid/user_name.
- sessions.creator_id -> uid, sessions.creator_name -> user_name
- turns.creator_id -> uid, turns.creator_name -> user_name
- events.creator_id -> uid, events.creator_name -> user_name
- agents.creator_id -> uid, agents.creator_name -> user_name
- Update indexes accordingly

Schema V18: UID refactor - created_by/shared_by with users table.
- New users table: uid -> user_name mapping
- sessions/events: uid -> created_by, drop user_name, add shared_by
- turns: drop uid and user_name (inherit from session)
- agents: uid -> created_by, drop user_name
- Update indexes accordingly

Schema V19: Agent association for sessions.
- sessions.agent_id: Logical agent entity association

Schema V20: Agent identity/profile table.
- agent_info table: name, description for agent profiles

Schema V23: WindowLink mapping for terminal/session association.
- windowlink table: terminal_id/agent_id/session_id with timestamp

Schema V24: Persist per-agent Slack share message.
- agent_info.slack_message: LLM-generated Slack message for sharing the agent

Schema V25: Persist per-agent preset questions.
- agent_info.preset_questions: JSON array of LLM-generated preset questions

Schema V27: Users table email cache.
- users.user_email: Optional email address for UID mapping

Schema V28: Agent-session mapping table for multi-agent membership.
- agent_sessions table: M2M agent-session links for query scoping/sync
- Backfill from legacy sessions.agent_id
"""

SCHEMA_VERSION = 28

FTS_EVENTS_SCRIPTS = [
    # Full Text Search for Events
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_events USING fts5(
        title,
        description,
        content='events',
        content_rowid='rowid'
    );
    """,
    # Triggers to keep FTS updated
    """
    CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
        INSERT INTO fts_events(rowid, title, description) VALUES (new.rowid, new.title, new.description);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS events_ad AFTER DELETE ON events BEGIN
        INSERT INTO fts_events(fts_events, rowid, title, description) VALUES('delete', old.rowid, old.title, old.description);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS events_au AFTER UPDATE ON events BEGIN
        INSERT INTO fts_events(fts_events, rowid, title, description) VALUES('delete', old.rowid, old.title, old.description);
        INSERT INTO fts_events(rowid, title, description) VALUES (new.rowid, new.title, new.description);
    END;
    """,
]

INIT_SCRIPTS = [
    # Version tracking
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT DEFAULT (datetime('now')),
        description TEXT
    );
    """,
    # Projects table (kept for backward compatibility, now optional)
    """
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,              -- UUID
        name TEXT NOT NULL,               -- Project name
        path TEXT NOT NULL UNIQUE,        -- Absolute path
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        metadata TEXT                     -- JSON metadata
    );
    """,
    # Sessions table (V2: decoupled from projects, V3: summary fields, V18: created_by/shared_by, V10: total_turns cache)
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,              -- session ID (filename stem)
        session_file_path TEXT NOT NULL,  -- Original session file path
        session_type TEXT NOT NULL,       -- 'claude', 'codex', 'gemini', 'antigravity'
        workspace_path TEXT,              -- Optional: workspace/project path for context
        started_at TEXT NOT NULL,
        last_activity_at TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        metadata TEXT,
        session_title TEXT,               -- V3: Aggregated title from turns
        session_summary TEXT,             -- V3: Aggregated summary from turns
        summary_updated_at TEXT,          -- V3: Timestamp of last summary update
        summary_status TEXT DEFAULT 'idle',        -- V7: 'idle' | 'processing' | 'completed' | 'failed'
        summary_locked_until TEXT,                -- V7: lease/TTL to avoid stuck processing
        summary_error TEXT,                       -- V7: last error message if failed
        created_by TEXT,                          -- V18: Creator UID (FK to users.uid)
        shared_by TEXT,                           -- V18: Sharer UID (who imported this)
        total_turns INTEGER DEFAULT 0,            -- V10: Cached total turn count (avoids reading files)
        total_turns_mtime REAL,                   -- V12: File mtime when total_turns was cached (for validation)
        agent_id TEXT                             -- V19: Logical agent association
    );
    """,
    # Turns table (corresponds to git commits, V18: uid/user_name removed)
    """
    CREATE TABLE IF NOT EXISTS turns (
        id TEXT PRIMARY KEY,              -- UUID
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        turn_number INTEGER NOT NULL,
        user_message TEXT,
        assistant_summary TEXT,
        turn_status TEXT,                 -- 'completed', 'interrupted', etc.
        llm_title TEXT NOT NULL,          -- Commit title
        temp_title TEXT,                  -- Temporary title before final summary
        llm_description TEXT,
        model_name TEXT,
        if_last_task TEXT DEFAULT 'no',
        satisfaction TEXT DEFAULT 'fine',
        content_hash TEXT NOT NULL,       -- MD5 hash for deduplication
        timestamp TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        git_commit_hash TEXT,             -- Linked git commit hash
        UNIQUE(session_id, turn_number)
    );
    """,
    # Turn content table (separated for performance)
    """
    CREATE TABLE IF NOT EXISTS turn_content (
        turn_id TEXT PRIMARY KEY REFERENCES turns(id) ON DELETE CASCADE,
        content TEXT NOT NULL,            -- JSONL content
        content_size INTEGER NOT NULL
    );
    """,
    # Lease locks (cross-process synchronization)
    """
    CREATE TABLE IF NOT EXISTS locks (
        lock_key TEXT PRIMARY KEY,
        owner TEXT NOT NULL,
        locked_until TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        metadata TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_locks_locked_until ON locks(locked_until);",
    # Durable jobs queue (V11)
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,                           -- UUID
        kind TEXT NOT NULL,                            -- 'turn_summary' | 'session_summary'
        dedupe_key TEXT NOT NULL UNIQUE,               -- e.g. 'turn:<session>:<n>' or 'session:<session>'
        payload TEXT,                                  -- JSON payload
        status TEXT NOT NULL DEFAULT 'queued',         -- 'queued' | 'processing' | 'retry' | 'done' | 'failed'
        priority INTEGER DEFAULT 0,
        attempts INTEGER DEFAULT 0,
        next_run_at TEXT DEFAULT (datetime('now')),    -- when eligible to run
        locked_until TEXT,                             -- lease/TTL for processing
        locked_by TEXT,                                -- worker id
        reschedule INTEGER DEFAULT 0,                  -- enqueue while processing => rerun after completion
        last_error TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now'))
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_status_next_run ON jobs(status, next_run_at);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_locked_until ON jobs(locked_until);",
    # Indexes
    "CREATE INDEX IF NOT EXISTS idx_projects_path ON projects(path);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_workspace ON sessions(workspace_path);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_activity ON sessions(last_activity_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_type ON sessions(session_type);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_created_by ON sessions(created_by);",  # V18
    "CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON sessions(agent_id);",  # V19
    "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_turns_timestamp ON turns(timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_turns_hash ON turns(content_hash);",
    # Events table (V18: created_by/shared_by)
    """
    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,              -- UUID
        title TEXT NOT NULL,
        description TEXT,
        event_type TEXT NOT NULL,         -- 'task', 'temporal', etc.
        status TEXT NOT NULL,             -- 'active', 'frozen', 'archived'
        start_timestamp TEXT,
        end_timestamp TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        metadata TEXT,                    -- JSON metadata (tags, confidence, etc.)
        preset_questions TEXT,            -- JSON array of LLM-generated preset questions
        slack_message TEXT,               -- LLM-generated Slack share message
        share_url TEXT,                   -- Public share URL
        share_id TEXT,                    -- V14: Server share ID (for reuse)
        share_admin_token TEXT,           -- V14: Server admin token (extend expiry)
        share_expiry_at TEXT,             -- V14: Last known expiry timestamp
        created_by TEXT,                  -- V18: Creator UID (FK to users.uid)
        shared_by TEXT                    -- V18: Sharer UID (who imported this)
    );
    """,
    # Event-Commit relationship (Many-to-Many)
    """
    CREATE TABLE IF NOT EXISTS event_commits (
        event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
        commit_hash TEXT NOT NULL,
        PRIMARY KEY (event_id, commit_hash)
    );
    """,
    # Event-Session relationship (Many-to-Many) - V4
    """
    CREATE TABLE IF NOT EXISTS event_sessions (
        event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        added_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (event_id, session_id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_event_sessions_event ON event_sessions(event_id);",
    "CREATE INDEX IF NOT EXISTS idx_event_sessions_session ON event_sessions(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_events_created_by ON events(created_by);",  # V18
    # Agents table (V15: replaces terminal.json, V18: created_by)
    """
    CREATE TABLE IF NOT EXISTS agents (
        id TEXT PRIMARY KEY,              -- terminal_id (UUID)
        provider TEXT NOT NULL,           -- 'claude', 'codex', 'opencode', 'shell'
        session_type TEXT NOT NULL,
        session_id TEXT,                  -- FK to sessions.id (nullable, may not exist yet)
        context_id TEXT,
        transcript_path TEXT,
        cwd TEXT,
        project_dir TEXT,
        status TEXT DEFAULT 'active',     -- 'active', 'stopped'
        attention TEXT,                   -- 'permission_request', 'stop', NULL
        source TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        created_by TEXT                   -- V18: Creator UID (FK to users.uid)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_agents_session ON agents(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_agents_context ON agents(context_id);",
    "CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);",
    # Agent contexts table (V15: replaces load.json)
    """
    CREATE TABLE IF NOT EXISTS agent_contexts (
        id TEXT PRIMARY KEY,              -- context_id
        workspace TEXT,
        loaded_at TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        metadata TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_agent_contexts_workspace ON agent_contexts(workspace);",
    # Agent context sessions (M2M)
    # Note: No FK on session_id - context may reference sessions not yet imported
    """
    CREATE TABLE IF NOT EXISTS agent_context_sessions (
        context_id TEXT NOT NULL REFERENCES agent_contexts(id) ON DELETE CASCADE,
        session_id TEXT NOT NULL,
        added_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (context_id, session_id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_agent_context_sessions_context ON agent_context_sessions(context_id);",
    "CREATE INDEX IF NOT EXISTS idx_agent_context_sessions_session ON agent_context_sessions(session_id);",
    # Agent context events (M2M)
    # Note: No FK on event_id - context may reference events not yet created
    """
    CREATE TABLE IF NOT EXISTS agent_context_events (
        context_id TEXT NOT NULL REFERENCES agent_contexts(id) ON DELETE CASCADE,
        event_id TEXT NOT NULL,
        added_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (context_id, event_id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_agent_context_events_context ON agent_context_events(context_id);",
    "CREATE INDEX IF NOT EXISTS idx_agent_context_events_event ON agent_context_events(event_id);",
    # Agent sessions (M2M) - V28
    # Note: No FK on session_id to tolerate pre-linked or not-yet-imported sessions.
    """
    CREATE TABLE IF NOT EXISTS agent_sessions (
        agent_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        added_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (agent_id, session_id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_agent_sessions_agent ON agent_sessions(agent_id);",
    "CREATE INDEX IF NOT EXISTS idx_agent_sessions_session ON agent_sessions(session_id);",
    # Users table (V18: UID-to-user-info mapping)
    """
    CREATE TABLE IF NOT EXISTS users (
        uid TEXT PRIMARY KEY,
        user_name TEXT,
        user_email TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now'))
    );
    """,
    # Agent identity/profile table (V20, V22: sync columns, V26: title)
    """
    CREATE TABLE IF NOT EXISTS agent_info (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        title TEXT DEFAULT '',
        description TEXT DEFAULT '',
        visibility TEXT NOT NULL DEFAULT 'visible',
        slack_message TEXT,
        preset_questions TEXT,
        share_id TEXT,
        share_url TEXT,
        share_admin_token TEXT,
        share_contributor_token TEXT,
        share_expiry_at TEXT,
        last_synced_at TEXT,
        sync_version INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now'))
    );
    """,
    *FTS_EVENTS_SCRIPTS,
]

# Migration scripts from V1 to V2
# SQLite doesn't support ALTER COLUMN, so we need to recreate the table
MIGRATION_V1_TO_V2 = [
    # Step 1: Rename old table
    """
    ALTER TABLE sessions RENAME TO sessions_old;
    """,
    # Step 2: Create new table without project_id constraint
    """
    CREATE TABLE sessions (
        id TEXT PRIMARY KEY,
        session_file_path TEXT NOT NULL,
        session_type TEXT NOT NULL,
        workspace_path TEXT,
        started_at TEXT NOT NULL,
        last_activity_at TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        metadata TEXT
    );
    """,
    # Step 3: Migrate data from old table, converting project_id to workspace_path
    """
    INSERT INTO sessions (id, session_file_path, session_type, workspace_path, started_at, last_activity_at, created_at, updated_at, metadata)
    SELECT
        s.id,
        s.session_file_path,
        s.session_type,
        p.path,
        s.started_at,
        s.last_activity_at,
        s.created_at,
        s.updated_at,
        s.metadata
    FROM sessions_old s
    LEFT JOIN projects p ON s.project_id = p.id;
    """,
    # Step 4: Drop old table
    """
    DROP TABLE sessions_old;
    """,
    # Step 5: Create indexes
    """
    CREATE INDEX IF NOT EXISTS idx_sessions_workspace ON sessions(workspace_path);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sessions_type ON sessions(session_type);
    """,
    # WindowLink table (V23)
    """
    CREATE TABLE IF NOT EXISTS windowlink (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        terminal_id TEXT NOT NULL,
        agent_id TEXT,
        session_id TEXT,
        provider TEXT,
        source TEXT,
        ts REAL,
        created_at TEXT DEFAULT (datetime('now'))
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_windowlink_terminal_ts ON windowlink(terminal_id, ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_windowlink_agent_ts ON windowlink(agent_id, ts DESC);",
]

# Migration scripts from V2 to V3
# Add session summary fields
MIGRATION_V2_TO_V3 = [
    """
    ALTER TABLE sessions ADD COLUMN session_title TEXT;
    """,
    """
    ALTER TABLE sessions ADD COLUMN session_summary TEXT;
    """,
    """
    ALTER TABLE sessions ADD COLUMN summary_updated_at TEXT;
    """,
]

# Migration scripts from V3 to V4
# Add event_sessions table for event-session many-to-many relationship
MIGRATION_V3_TO_V4 = [
    """
    CREATE TABLE IF NOT EXISTS event_sessions (
        event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        added_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (event_id, session_id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_event_sessions_event ON event_sessions(event_id);",
    "CREATE INDEX IF NOT EXISTS idx_event_sessions_session ON event_sessions(session_id);",
]

MIGRATION_V4_TO_V5 = [
    "ALTER TABLE events ADD COLUMN preset_questions TEXT;",
    "ALTER TABLE events ADD COLUMN slack_message TEXT;",
]

MIGRATION_V5_TO_V6 = [
    "ALTER TABLE events ADD COLUMN share_url TEXT;",
]

MIGRATION_V6_TO_V7 = [
    "ALTER TABLE sessions ADD COLUMN summary_status TEXT;",
    "ALTER TABLE sessions ADD COLUMN summary_locked_until TEXT;",
    "ALTER TABLE sessions ADD COLUMN summary_error TEXT;",
]

MIGRATION_V7_TO_V8 = [
    """
    CREATE TABLE IF NOT EXISTS locks (
        lock_key TEXT PRIMARY KEY,
        owner TEXT NOT NULL,
        locked_until TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        metadata TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_locks_locked_until ON locks(locked_until);",
]

MIGRATION_V8_TO_V9 = [
    # Sessions table: add creator fields
    "ALTER TABLE sessions ADD COLUMN creator_name TEXT;",
    "ALTER TABLE sessions ADD COLUMN creator_id TEXT;",
    # Events table: add creator fields
    "ALTER TABLE events ADD COLUMN creator_name TEXT;",
    "ALTER TABLE events ADD COLUMN creator_id TEXT;",
    # Turns table: add creator fields
    "ALTER TABLE turns ADD COLUMN creator_name TEXT;",
    "ALTER TABLE turns ADD COLUMN creator_id TEXT;",
    # Create indexes for performance
    "CREATE INDEX IF NOT EXISTS idx_sessions_creator ON sessions(creator_id);",
    "CREATE INDEX IF NOT EXISTS idx_events_creator ON events(creator_id);",
    "CREATE INDEX IF NOT EXISTS idx_turns_creator ON turns(creator_id);",
]

MIGRATION_V10_TO_V11 = [
    # V11: Durable jobs queue
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        dedupe_key TEXT NOT NULL UNIQUE,
        payload TEXT,
        status TEXT NOT NULL DEFAULT 'queued',
        priority INTEGER DEFAULT 0,
        attempts INTEGER DEFAULT 0,
        next_run_at TEXT DEFAULT (datetime('now')),
        locked_until TEXT,
        locked_by TEXT,
        reschedule INTEGER DEFAULT 0,
        last_error TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now'))
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_status_next_run ON jobs(status, next_run_at);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_locked_until ON jobs(locked_until);",
]

MIGRATION_V13_TO_V14 = [
    "ALTER TABLE events ADD COLUMN share_id TEXT;",
    "ALTER TABLE events ADD COLUMN share_admin_token TEXT;",
    "ALTER TABLE events ADD COLUMN share_expiry_at TEXT;",
]

MIGRATION_V14_TO_V15 = [
    # Agents table (replaces terminal.json)
    """
    CREATE TABLE IF NOT EXISTS agents (
        id TEXT PRIMARY KEY,
        provider TEXT NOT NULL,
        session_type TEXT NOT NULL,
        session_id TEXT,
        context_id TEXT,
        transcript_path TEXT,
        cwd TEXT,
        project_dir TEXT,
        status TEXT DEFAULT 'active',
        attention TEXT,
        source TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        creator_name TEXT,
        creator_id TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_agents_session ON agents(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_agents_context ON agents(context_id);",
    "CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);",
    # Agent contexts table (replaces load.json)
    """
    CREATE TABLE IF NOT EXISTS agent_contexts (
        id TEXT PRIMARY KEY,
        workspace TEXT,
        loaded_at TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now')),
        metadata TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_agent_contexts_workspace ON agent_contexts(workspace);",
    # Agent context sessions (M2M)
    # Note: No FK on session_id - context may reference sessions not yet imported
    """
    CREATE TABLE IF NOT EXISTS agent_context_sessions (
        context_id TEXT NOT NULL REFERENCES agent_contexts(id) ON DELETE CASCADE,
        session_id TEXT NOT NULL,
        added_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (context_id, session_id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_agent_context_sessions_context ON agent_context_sessions(context_id);",
    "CREATE INDEX IF NOT EXISTS idx_agent_context_sessions_session ON agent_context_sessions(session_id);",
    # Agent context events (M2M)
    # Note: No FK on event_id - context may reference events not yet created
    """
    CREATE TABLE IF NOT EXISTS agent_context_events (
        context_id TEXT NOT NULL REFERENCES agent_contexts(id) ON DELETE CASCADE,
        event_id TEXT NOT NULL,
        added_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (context_id, event_id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_agent_context_events_context ON agent_context_events(context_id);",
    "CREATE INDEX IF NOT EXISTS idx_agent_context_events_event ON agent_context_events(event_id);",
]

# V15 to V16: Remove FK constraints from agent_context_sessions/events
# Context may reference sessions/events not yet in the database
MIGRATION_V15_TO_V16 = [
    # Step 1: Rename old tables
    "ALTER TABLE agent_context_sessions RENAME TO agent_context_sessions_old;",
    "ALTER TABLE agent_context_events RENAME TO agent_context_events_old;",
    # Step 2: Create new tables without FK constraints on session_id/event_id
    """
    CREATE TABLE agent_context_sessions (
        context_id TEXT NOT NULL REFERENCES agent_contexts(id) ON DELETE CASCADE,
        session_id TEXT NOT NULL,
        added_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (context_id, session_id)
    );
    """,
    """
    CREATE TABLE agent_context_events (
        context_id TEXT NOT NULL REFERENCES agent_contexts(id) ON DELETE CASCADE,
        event_id TEXT NOT NULL,
        added_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (context_id, event_id)
    );
    """,
    # Step 3: Copy data from old tables
    "INSERT INTO agent_context_sessions SELECT * FROM agent_context_sessions_old;",
    "INSERT INTO agent_context_events SELECT * FROM agent_context_events_old;",
    # Step 4: Drop old tables
    "DROP TABLE agent_context_sessions_old;",
    "DROP TABLE agent_context_events_old;",
    # Step 5: Recreate indexes
    "CREATE INDEX IF NOT EXISTS idx_agent_context_sessions_context ON agent_context_sessions(context_id);",
    "CREATE INDEX IF NOT EXISTS idx_agent_context_sessions_session ON agent_context_sessions(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_agent_context_events_context ON agent_context_events(context_id);",
    "CREATE INDEX IF NOT EXISTS idx_agent_context_events_event ON agent_context_events(event_id);",
]

# V16 to V17: Rename creator_id/creator_name to uid/user_name
MIGRATION_V16_TO_V17 = [
    # Sessions table: rename columns
    "ALTER TABLE sessions RENAME COLUMN creator_id TO uid;",
    "ALTER TABLE sessions RENAME COLUMN creator_name TO user_name;",
    # Turns table: rename columns
    "ALTER TABLE turns RENAME COLUMN creator_id TO uid;",
    "ALTER TABLE turns RENAME COLUMN creator_name TO user_name;",
    # Events table: rename columns
    "ALTER TABLE events RENAME COLUMN creator_id TO uid;",
    "ALTER TABLE events RENAME COLUMN creator_name TO user_name;",
    # Agents table: rename columns
    "ALTER TABLE agents RENAME COLUMN creator_id TO uid;",
    "ALTER TABLE agents RENAME COLUMN creator_name TO user_name;",
    # Update indexes: drop old, create new
    "DROP INDEX IF EXISTS idx_sessions_creator;",
    "DROP INDEX IF EXISTS idx_turns_creator;",
    "DROP INDEX IF EXISTS idx_events_creator;",
    "CREATE INDEX IF NOT EXISTS idx_sessions_uid ON sessions(uid);",
    "CREATE INDEX IF NOT EXISTS idx_turns_uid ON turns(uid);",
    "CREATE INDEX IF NOT EXISTS idx_events_uid ON events(uid);",
]


# V17 to V18: uid/user_name → created_by/shared_by, users table, remove turns uid
MIGRATION_V17_TO_V18 = [
    # 1. Create users table
    """
    CREATE TABLE IF NOT EXISTS users (
        uid TEXT PRIMARY KEY,
        user_name TEXT,
        user_email TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now'))
    );
    """,
    # 2. Extract user info from existing data into users table
    "INSERT OR IGNORE INTO users (uid, user_name) SELECT DISTINCT uid, user_name FROM sessions WHERE uid IS NOT NULL AND uid != '';",
    "INSERT OR IGNORE INTO users (uid, user_name) SELECT DISTINCT uid, user_name FROM events WHERE uid IS NOT NULL AND uid != '' AND uid NOT IN (SELECT uid FROM users);",
    "INSERT OR IGNORE INTO users (uid, user_name) SELECT DISTINCT uid, user_name FROM turns WHERE uid IS NOT NULL AND uid != '' AND uid NOT IN (SELECT uid FROM users);",
    "INSERT OR IGNORE INTO users (uid, user_name) SELECT DISTINCT uid, user_name FROM agents WHERE uid IS NOT NULL AND uid != '' AND uid NOT IN (SELECT uid FROM users);",
    # 3. Sessions: uid → created_by, drop user_name, add shared_by
    "DROP INDEX IF EXISTS idx_sessions_uid;",
    "ALTER TABLE sessions RENAME COLUMN uid TO created_by;",
    "ALTER TABLE sessions DROP COLUMN user_name;",
    "ALTER TABLE sessions ADD COLUMN shared_by TEXT;",
    "CREATE INDEX IF NOT EXISTS idx_sessions_created_by ON sessions(created_by);",
    # 4. Events: uid → created_by, drop user_name, add shared_by
    "DROP INDEX IF EXISTS idx_events_uid;",
    "ALTER TABLE events RENAME COLUMN uid TO created_by;",
    "ALTER TABLE events DROP COLUMN user_name;",
    "ALTER TABLE events ADD COLUMN shared_by TEXT;",
    "CREATE INDEX IF NOT EXISTS idx_events_created_by ON events(created_by);",
    # 5. Turns: drop uid and user_name
    "DROP INDEX IF EXISTS idx_turns_uid;",
    "ALTER TABLE turns DROP COLUMN uid;",
    "ALTER TABLE turns DROP COLUMN user_name;",
    # 6. Agents: uid → created_by, drop user_name (no shared_by needed)
    "ALTER TABLE agents RENAME COLUMN uid TO created_by;",
    "ALTER TABLE agents DROP COLUMN user_name;",
]


MIGRATION_V18_TO_V19 = [
    "ALTER TABLE sessions ADD COLUMN agent_id TEXT;",
    "CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON sessions(agent_id);",
]

MIGRATION_V19_TO_V20 = [
    """
    CREATE TABLE IF NOT EXISTS agent_info (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT DEFAULT '',
        visibility TEXT NOT NULL DEFAULT 'visible',
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now'))
    );
    """,
]

MIGRATION_V20_TO_V21 = [
    "ALTER TABLE agent_info ADD COLUMN visibility TEXT NOT NULL DEFAULT 'visible';",
]

MIGRATION_V21_TO_V22 = [
    "ALTER TABLE agent_info ADD COLUMN share_id TEXT;",
    "ALTER TABLE agent_info ADD COLUMN share_url TEXT;",
    "ALTER TABLE agent_info ADD COLUMN share_admin_token TEXT;",
    "ALTER TABLE agent_info ADD COLUMN share_contributor_token TEXT;",
    "ALTER TABLE agent_info ADD COLUMN share_expiry_at TEXT;",
    "ALTER TABLE agent_info ADD COLUMN last_synced_at TEXT;",
    "ALTER TABLE agent_info ADD COLUMN sync_version INTEGER DEFAULT 0;",
]

MIGRATION_V22_TO_V23 = [
    """
    CREATE TABLE IF NOT EXISTS windowlink (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        terminal_id TEXT NOT NULL,
        agent_id TEXT,
        session_id TEXT,
        provider TEXT,
        source TEXT,
        ts REAL,
        created_at TEXT DEFAULT (datetime('now'))
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_windowlink_terminal_ts ON windowlink(terminal_id, ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_windowlink_agent_ts ON windowlink(agent_id, ts DESC);",
]

MIGRATION_V23_TO_V24 = [
    "ALTER TABLE agent_info ADD COLUMN slack_message TEXT;",
]

MIGRATION_V24_TO_V25 = [
    "ALTER TABLE agent_info ADD COLUMN preset_questions TEXT;",
]

MIGRATION_V25_TO_V26 = [
    "ALTER TABLE agent_info ADD COLUMN title TEXT DEFAULT '';",
]

MIGRATION_V26_TO_V27 = [
    "ALTER TABLE users ADD COLUMN user_email TEXT;",
]

MIGRATION_V27_TO_V28 = [
    """
    CREATE TABLE IF NOT EXISTS agent_sessions (
        agent_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        added_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (agent_id, session_id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_agent_sessions_agent ON agent_sessions(agent_id);",
    "CREATE INDEX IF NOT EXISTS idx_agent_sessions_session ON agent_sessions(session_id);",
    """
    INSERT OR IGNORE INTO agent_sessions (agent_id, session_id)
    SELECT agent_id, id
    FROM sessions
    WHERE agent_id IS NOT NULL
      AND TRIM(agent_id) <> ''
    """,
]


def get_migration_scripts(from_version: int, to_version: int) -> list:
    """Get migration scripts for upgrading between versions."""
    scripts = []

    if from_version < 2 and to_version >= 2:
        scripts.extend(MIGRATION_V1_TO_V2)

    if from_version < 3 and to_version >= 3:
        scripts.extend(MIGRATION_V2_TO_V3)

    if from_version < 4 and to_version >= 4:
        scripts.extend(MIGRATION_V3_TO_V4)

    if from_version < 5 and to_version >= 5:
        scripts.extend(MIGRATION_V4_TO_V5)

    if from_version < 6 and to_version >= 6:
        scripts.extend(MIGRATION_V5_TO_V6)

    if from_version < 7 and to_version >= 7:
        scripts.extend(MIGRATION_V6_TO_V7)

    if from_version < 8 and to_version >= 8:
        scripts.extend(MIGRATION_V7_TO_V8)

    if from_version < 9 and to_version >= 9:
        scripts.extend(MIGRATION_V8_TO_V9)

    if from_version < 10 and to_version >= 10:
        # V10: Add total_turns column for session list performance
        scripts.append("ALTER TABLE sessions ADD COLUMN total_turns INTEGER DEFAULT 0;")
        scripts.append(
            "CREATE INDEX IF NOT EXISTS idx_sessions_total_turns ON sessions(total_turns);"
        )

    if from_version < 11 and to_version >= 11:
        scripts.extend(MIGRATION_V10_TO_V11)

    if from_version < 12 and to_version >= 12:
        # V12: Add total_turns_mtime for lazy cache validation
        scripts.append("ALTER TABLE sessions ADD COLUMN total_turns_mtime REAL;")

    if from_version < 13 and to_version >= 13:
        # V13: Temporary turn title
        scripts.append("ALTER TABLE turns ADD COLUMN temp_title TEXT;")

    if from_version < 14 and to_version >= 14:
        scripts.extend(MIGRATION_V13_TO_V14)

    if from_version < 15 and to_version >= 15:
        scripts.extend(MIGRATION_V14_TO_V15)

    if from_version < 16 and to_version >= 16:
        # Only run V15->V16 if coming from exactly V15 (tables exist with FK)
        # For V14 or earlier, V14_TO_V15 now creates tables without FK
        if from_version == 15:
            scripts.extend(MIGRATION_V15_TO_V16)

    if from_version < 17 and to_version >= 17:
        scripts.extend(MIGRATION_V16_TO_V17)

    if from_version < 18 and to_version >= 18:
        scripts.extend(MIGRATION_V17_TO_V18)

    if from_version < 19 and to_version >= 19:
        scripts.extend(MIGRATION_V18_TO_V19)

    if from_version < 20 and to_version >= 20:
        scripts.extend(MIGRATION_V19_TO_V20)

    if from_version < 21 and to_version >= 21:
        scripts.extend(MIGRATION_V20_TO_V21)

    if from_version < 22 and to_version >= 22:
        scripts.extend(MIGRATION_V21_TO_V22)

    if from_version < 23 and to_version >= 23:
        scripts.extend(MIGRATION_V22_TO_V23)

    if from_version < 24 and to_version >= 24:
        scripts.extend(MIGRATION_V23_TO_V24)

    if from_version < 25 and to_version >= 25:
        scripts.extend(MIGRATION_V24_TO_V25)

    if from_version < 26 and to_version >= 26:
        scripts.extend(MIGRATION_V25_TO_V26)

    if from_version < 27 and to_version >= 27:
        scripts.extend(MIGRATION_V26_TO_V27)

    if from_version < 28 and to_version >= 28:
        scripts.extend(MIGRATION_V27_TO_V28)

    return scripts
