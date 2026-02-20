"""
Codex CLI notify hook.

Codex (Rust CLI) can execute a `notify` command when the agent finishes a turn.
We use this hook to enqueue a durable `session_process` job in Aline's SQLite DB,
so the worker can process all missing turns without watcher polling.

This script must be dependency-light and never raise (it's executed from Codex).
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import uuid
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional


def _parse_config_sqlite_db_path(config_path: Path) -> Optional[str]:
    """Parse ~/.aline/config.yaml for sqlite_db_path (best-effort)."""
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None  # type: ignore[assignment]

    if yaml is None:
        return None
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("sqlite_db_path")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _resolve_sqlite_db_path() -> Path:
    env_db_path = (
        os.getenv("REALIGN_SQLITE_DB_PATH") or os.getenv("REALIGN_DB_PATH") or os.getenv("ALINE_DB_PATH")
    )
    if env_db_path:
        return Path(env_db_path).expanduser()

    config_path = Path.home() / ".aline" / "config.yaml"
    cfg = _parse_config_sqlite_db_path(config_path) if config_path.exists() else None
    if cfg:
        return Path(cfg).expanduser()

    return Path.home() / ".aline" / "db" / "aline.db"


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"


def _read_event_payload() -> dict[str, Any]:
    """Codex passes event JSON as argv[1] (docs); also accept stdin for safety."""
    raw = ""
    if len(sys.argv) >= 2:
        raw = sys.argv[1] or ""
    if not raw.strip():
        try:
            raw = sys.stdin.read() or ""
        except Exception:
            raw = ""
    try:
        data = json.loads(raw) if raw.strip() else {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _infer_agent_terminal_from_codex_home(codex_home: Path) -> tuple[str | None, str | None]:
    """Infer (agent_id, terminal_id) from an Aline-managed CODEX_HOME path.

    Supported layouts:
      - ~/.aline/codex_homes/<terminal_id>/
      - ~/.aline/codex_homes/agent-<agent_id>/<terminal_id>/
    """
    try:
        p = codex_home.expanduser().resolve()
    except Exception:
        p = Path(codex_home)

    parts = p.parts
    try:
        idx = parts.index("codex_homes")
    except ValueError:
        return None, None

    if idx + 1 >= len(parts):
        return None, None

    owner = (parts[idx + 1] or "").strip()
    if not owner:
        return None, None

    if owner.startswith("agent-"):
        agent_id = owner[len("agent-") :].strip() or None
        terminal_id = (parts[idx + 2] or "").strip() if idx + 2 < len(parts) else ""
        return agent_id, (terminal_id or None)

    # Terminal layout.
    return None, (owner or None)


def _extract_thread_id(evt: dict[str, Any]) -> str:
    for k in ("thread-id", "threadId", "thread_id", "thread"):
        v = evt.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_cwd(evt: dict[str, Any]) -> str:
    v = evt.get("cwd")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return ""


def _read_session_identifier(session_file: Path) -> str:
    """Best-effort: extract Codex 'thread id' from the session file header/metadata."""
    try:
        with session_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 25:
                    break
                raw = (line or "").strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except Exception:
                    continue

                # Newer header: {id, timestamp, git} without "type"
                if "id" in data and "type" not in data:
                    v = data.get("id")
                    if isinstance(v, str) and v.strip():
                        return v.strip()

                # session_meta payload may contain id/thread_id
                if data.get("type") == "session_meta":
                    payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
                    for k in ("thread_id", "threadId", "thread-id", "id"):
                        v = payload.get(k) if isinstance(payload, dict) else None
                        if isinstance(v, str) and v.strip():
                            return v.strip()
    except Exception:
        return ""
    return ""


def _read_session_cwd(session_file: Path) -> str:
    """Best-effort: extract cwd from session metadata/header."""
    try:
        with session_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 25:
                    break
                raw = (line or "").strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except Exception:
                    continue

                if data.get("type") == "session_meta":
                    payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
                    cwd = payload.get("cwd") if isinstance(payload, dict) else None
                    if isinstance(cwd, str) and cwd.strip():
                        return cwd.strip()

                if "type" not in data and isinstance(data.get("git"), dict):
                    cwd = data["git"].get("cwd")
                    if isinstance(cwd, str) and cwd.strip():
                        return cwd.strip()
                cwd2 = data.get("cwd")
                if isinstance(cwd2, str) and cwd2.strip():
                    return cwd2.strip()
    except Exception:
        return ""
    return ""


def _thread_map_path(codex_home: Path) -> Path:
    return codex_home / ".aline_thread_map.json"


def _load_thread_map(codex_home: Path) -> dict[str, str]:
    path = _thread_map_path(codex_home)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out


def _save_thread_map(codex_home: Path, mapping: dict[str, str]) -> None:
    path = _thread_map_path(codex_home)
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        return


def _iter_recent_codex_session_files(sessions_root: Path, *, days_back: int = 2) -> list[Path]:
    """List recent rollout session files under CODEX_HOME/sessions (YYYY/MM/DD layout)."""
    out: list[Path] = []
    if not sessions_root.exists():
        return out

    # Preferred: YYYY/MM/DD partitioned directories.
    now = datetime.now()
    for days_ago in range(max(0, int(days_back)) + 1):
        dt = now - timedelta(days=days_ago)
        p = sessions_root / str(dt.year) / f"{dt.month:02d}" / f"{dt.day:02d}"
        if not p.exists():
            continue
        try:
            out.extend(p.glob("rollout-*.jsonl"))
        except Exception:
            continue

    # Fallback: flat structure
    if not out:
        try:
            out.extend(list(sessions_root.glob("rollout-*.jsonl")))
        except Exception:
            pass

    # De-dup and sort by mtime desc.
    uniq: dict[str, Path] = {}
    for f in out:
        uniq[str(f)] = f
    files = list(uniq.values())
    try:
        files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    except Exception:
        pass
    return files


def _find_session_file_for_event(
    *, codex_home: Path, thread_id: str, cwd: str
) -> Optional[Path]:
    sessions_root = codex_home / "sessions"
    cache = _load_thread_map(codex_home)
    cached = cache.get(thread_id) if thread_id else None
    if cached:
        p = Path(cached)
        if p.exists():
            # Validate cached mapping (cheap header read) to avoid stale misroutes.
            try:
                if not thread_id or _read_session_identifier(p) == thread_id:
                    return p
            except Exception:
                return p

    candidates = _iter_recent_codex_session_files(sessions_root, days_back=2)
    cwd_norm = (cwd or "").strip()

    # First pass: match by thread id.
    if thread_id:
        for f in candidates[:50]:
            try:
                if _read_session_identifier(f) == thread_id:
                    cache[thread_id] = str(f)
                    _save_thread_map(codex_home, cache)
                    return f
            except Exception:
                continue

    # Second pass: match by cwd.
    if cwd_norm:
        for f in candidates[:50]:
            try:
                if _read_session_cwd(f) == cwd_norm:
                    if thread_id:
                        cache[thread_id] = str(f)
                        _save_thread_map(codex_home, cache)
                    return f
            except Exception:
                continue

    chosen = candidates[0] if candidates else None
    if chosen is not None and thread_id:
        cache[thread_id] = str(chosen)
        _save_thread_map(codex_home, cache)
    return chosen


def _try_enqueue_session_process_job(
    *,
    session_id: str,
    session_file_path: str,
    workspace_path: str | None,
    session_type: str | None,
    source_event: str | None,
    no_track: bool,
    agent_id: str | None,
    terminal_id: str | None,
    connect_timeout_seconds: float,
) -> bool:
    """Best-effort enqueue into sqlite jobs table. Never raises."""
    try:
        db_path = _resolve_sqlite_db_path()
        if not db_path.exists():
            return False

        payload: dict[str, Any] = {"session_id": session_id, "session_file_path": session_file_path}
        if workspace_path is not None:
            payload["workspace_path"] = workspace_path
        if session_type:
            payload["session_type"] = session_type
        if source_event:
            payload["source_event"] = source_event
        if no_track:
            payload["no_track"] = True
        if agent_id:
            payload["agent_id"] = agent_id
        if terminal_id:
            payload["terminal_id"] = terminal_id

        job_id = str(uuid.uuid4())
        payload_json = json.dumps(payload, ensure_ascii=False)
        dedupe_key = f"session_process:{session_id}"

        conn = sqlite3.connect(str(db_path), timeout=float(connect_timeout_seconds))
        try:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, kind, dedupe_key, payload, status, priority, attempts, next_run_at,
                    locked_until, locked_by, reschedule, last_error, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, 'queued', ?, 0, datetime('now'),
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
                        WHEN jobs.status='done' THEN 'queued'
                        ELSE 'queued'
                    END,
                    next_run_at=CASE
                        WHEN jobs.status='processing' THEN jobs.next_run_at
                        WHEN jobs.next_run_at IS NULL THEN excluded.next_run_at
                        WHEN excluded.next_run_at < jobs.next_run_at THEN excluded.next_run_at
                        ELSE jobs.next_run_at
                    END
                """,
                (
                    job_id,
                    "session_process",
                    dedupe_key,
                    payload_json,
                    15,
                ),
            )
            conn.commit()
            return True
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception:
        return False


def _write_fallback_signal(*, session_id: str, session_file: str, cwd: str, agent_id: str, terminal_id: str, no_track: bool) -> None:
    try:
        from . import codex_notify_signal_dir

        signal_dir = codex_notify_signal_dir()
        signal_dir.mkdir(parents=True, exist_ok=True)
        stamp_ms = int(time.time() * 1000)
        signal_file_path = signal_dir / f"{session_id}_{stamp_ms}.signal"
        tmp = signal_dir / f"{session_id}_{stamp_ms}.signal.tmp"
        data: dict[str, Any] = {
            "session_id": session_id,
            "transcript_path": session_file,
            "project_dir": cwd,
            "cwd": cwd,
            "timestamp": time.time(),
            "hook_event": "CodexNotify",
            "source_event": "notify",
            "session_type": "codex",
        }
        if agent_id:
            data["agent_id"] = agent_id
        if terminal_id:
            data["terminal_id"] = terminal_id
        if no_track:
            data["no_track"] = True
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(signal_file_path)
    except Exception:
        return


def main() -> None:
    evt = _read_event_payload()
    thread_id = _extract_thread_id(evt)
    cwd = _extract_cwd(evt)
    no_track = os.environ.get("ALINE_NO_TRACK", "") == "1"
    agent_id = os.environ.get("ALINE_AGENT_ID", "").strip()
    terminal_id = os.environ.get("ALINE_TERMINAL_ID", "").strip()

    codex_home = Path(os.environ.get("CODEX_HOME", "") or (Path.home() / ".codex")).expanduser()

    session_file = None
    try:
        session_file = _find_session_file_for_event(codex_home=codex_home, thread_id=thread_id, cwd=cwd)
    except Exception:
        session_file = None
    if session_file is None or not session_file.exists():
        # Nothing else to do; avoid crashing Codex.
        return

    # If the event payload doesn't include cwd, fall back to the session meta header.
    if not (cwd or "").strip():
        try:
            cwd = _read_session_cwd(session_file) or ""
        except Exception:
            cwd = ""

    # Some Codex notify runners don't propagate arbitrary env vars. Infer from CODEX_HOME when possible.
    if not agent_id or not terminal_id:
        inferred_agent_id, inferred_terminal_id = _infer_agent_terminal_from_codex_home(codex_home)
        if not agent_id and inferred_agent_id:
            agent_id = inferred_agent_id
        if not terminal_id and inferred_terminal_id:
            terminal_id = inferred_terminal_id

    session_id = session_file.stem
    connect_timeout = float(os.environ.get("ALINE_CODEX_NOTIFY_DB_TIMEOUT", "0.2"))

    ok = _try_enqueue_session_process_job(
        session_id=session_id,
        session_file_path=str(session_file),
        workspace_path=cwd or None,
        session_type="codex",
        source_event="notify",
        no_track=no_track,
        agent_id=agent_id or None,
        terminal_id=terminal_id or None,
        connect_timeout_seconds=connect_timeout,
    )

    if not ok:
        _write_fallback_signal(
            session_id=session_id,
            session_file=str(session_file),
            cwd=cwd or "",
            agent_id=agent_id,
            terminal_id=terminal_id,
            no_track=no_track,
        )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        # Hook must never crash Codex.
        sys.exit(0)
