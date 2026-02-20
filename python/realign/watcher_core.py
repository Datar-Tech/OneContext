"""Session file watcher for auto-commit per user request completion.

Supports both Claude Code and Codex session formats with unified interface.
"""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Dict, Literal
from datetime import datetime

from .config import ReAlignConfig
from .dashboard_tracking import (
    is_dashboard_only_enabled,
    is_session_trackable_for_dashboard_only,
)
from .hooks import find_all_active_sessions
from .logging_config import setup_logger

# Initialize logger for watcher
logger = setup_logger("realign.watcher_core", "watcher_core.log")


# Session type detection
SessionType = Literal["claude", "codex", "gemini", "unknown"]


@dataclass(frozen=True)
class StartupScanReport:
    prev_paths: int
    prev_missing: int
    prev_changed: int
    scan_paths: int
    scan_new: int
    scan_changed: int
    candidates: int


def is_path_blacklisted(project_path: Path) -> bool:
    """
    Check if a project path is blacklisted for auto-init.

    Blacklisted paths:
    - Anything inside ~/.aline/ (where aline data is stored)
    - Anything inside ~/.realign/ (legacy location)
    - Any path containing .aline or .realign directory components
    - User home directory itself (~)
    - ~/Desktop, ~/Documents, ~/Downloads (top-level only, subdirs allowed)

    Args:
        project_path: Absolute path to check

    Returns:
        True if blacklisted, False if allowed
    """
    try:
        # Normalize path (resolve symlinks, make absolute)
        normalized = project_path.resolve()
        home = Path.home().resolve()
        aline_global_dir = (home / ".aline").resolve()
        realign_global_dir = (home / ".realign").resolve()

        # Check if inside ~/.aline/ directory (where all project data is stored)
        try:
            normalized.relative_to(aline_global_dir)
            logger.debug(f"Blacklisted (inside ~/.aline): {normalized}")
            return True
        except ValueError:
            pass  # Not inside ~/.aline

        # Check if inside ~/.realign/ directory (legacy)
        try:
            normalized.relative_to(realign_global_dir)
            logger.debug(f"Blacklisted (inside ~/.realign): {normalized}")
            return True
        except ValueError:
            pass  # Not inside ~/.realign

        # Check if path contains .aline or .realign components anywhere
        # This prevents initializing within project's local .aline/.realign directories
        path_parts = normalized.parts
        for part in path_parts:
            if part in [".aline", ".realign"]:
                logger.debug(f"Blacklisted (contains {part} component): {normalized}")
                return True

        # Check if it IS the home directory itself
        if normalized == home:
            logger.debug(f"Blacklisted (home directory): {normalized}")
            return True

        # Check forbidden top-level home subdirectories
        # But allow their subdirectories (e.g., ~/Desktop/project is OK)
        forbidden_dirs = ["Desktop", "Documents", "Downloads"]
        for forbidden in forbidden_dirs:
            forbidden_path = (home / forbidden).resolve()
            if normalized == forbidden_path:
                logger.debug(f"Blacklisted (forbidden dir): {normalized}")
                return True

        return False

    except Exception as e:
        logger.error(f"Error checking blacklist for {project_path}: {e}")
        # If we can't determine, err on the side of caution
        return True


def decode_claude_project_path(project_dir_name: str) -> Optional[Path]:
    """
    Decode Claude Code project directory name to actual project path.

    Claude naming: -Users-huminhao-Projects-ReAlign
    Decoded: /Users/huminhao/Projects/ReAlign

    If naive decoding fails (e.g., paths with underscores/hyphens in directory names),
    falls back to reading the 'cwd' field from JSONL session files.

    Args:
        project_dir_name: Claude project directory name (or full path to Claude project dir)

    Returns:
        Decoded Path if valid, None otherwise
    """
    # Handle both directory name and full path
    if isinstance(project_dir_name, Path):
        project_dir = project_dir_name
        dir_name = project_dir.name
    elif "/" in project_dir_name:
        project_dir = Path(project_dir_name)
        dir_name = project_dir.name
    else:
        dir_name = project_dir_name
        project_dir = Path.home() / ".claude" / "projects" / dir_name

    if not dir_name.startswith("-"):
        return None

    # Try naive decoding first
    path_str = "/" + dir_name[1:].replace("-", "/")
    project_path = Path(path_str)

    if project_path.exists():
        return project_path

    # Naive decoding failed - try reading from JSONL files
    logger.debug(f"Naive decoding failed for {dir_name}, trying JSONL fallback")

    if not project_dir.exists() or not project_dir.is_dir():
        logger.debug(f"Claude project directory not found: {project_dir}")
        return None

    # Find any JSONL file (excluding agent files)
    try:
        jsonl_files = [
            f
            for f in project_dir.iterdir()
            if f.suffix == ".jsonl" and not f.name.startswith("agent-")
        ]

        if not jsonl_files:
            logger.debug(f"No JSONL session files found in {project_dir}")
            return None

        # Read lines from first JSONL file to find cwd field
        jsonl_file = jsonl_files[0]
        with jsonl_file.open("r", encoding="utf-8") as f:
            # Check up to first 20 lines for cwd field
            for i, line in enumerate(f):
                if i >= 20:
                    break

                line = line.strip()
                if not line:
                    continue

                session_data = json.loads(line)
                cwd = session_data.get("cwd")

                if cwd:
                    project_path = Path(cwd)
                    if project_path.exists():
                        logger.debug(f"Decoded path from JSONL: {dir_name} -> {project_path}")
                        return project_path
                    else:
                        logger.debug(f"Path from JSONL doesn't exist: {project_path}")
                        return None

            logger.debug(f"No 'cwd' field found in first 20 lines of {jsonl_file.name}")
            return None

    except Exception as e:
        logger.debug(f"Error reading JSONL files from {project_dir}: {e}")
        return None

    return None


def is_aline_initialized() -> bool:
    """
    Check if Aline has been initialized globally (config + database present).
    """
    try:
        from .config import ReAlignConfig

        config = ReAlignConfig.load()
        db_path = Path(config.sqlite_db_path).expanduser()
        config_path = Path.home() / ".aline" / "config.yaml"
        return config_path.exists() and db_path.exists()
    except Exception as e:
        logger.debug(f"Error checking global init status: {e}")
        return False


class DialogueWatcher:
    """Watch session files and auto-commit immediately after each user request completes."""

    def __init__(self):
        """Initialize watcher for multi-project monitoring - extracts project paths dynamically from sessions."""
        self.config = ReAlignConfig.load()
        self.last_commit_times: Dict[str, float] = {}  # Track last commit time per project
        self.last_session_sizes: Dict[str, int] = {}  # Track file sizes
        self.last_stop_reason_counts: Dict[str, int] = {}  # Track stop_reason counts per session
        self.last_session_mtimes: Dict[str, float] = {}  # Track last mtime of session files
        self.min_commit_interval = 5.0  # Minimum 5 seconds between commits (cooldown)
        self.debounce_delay = 10.0  # Wait 10 seconds after file change to ensure turn is complete (increased from 2.0 to handle streaming responses)
        self.running = False
        self.pending_commit_task: Optional[asyncio.Task] = None
        self._pending_changed_files: set[str] = (
            set()
        )  # Accumulate changed files instead of cancelling

        # Stop-hook signals: retry/backoff map so transient enqueue failures don't drop turns.
        # Key: absolute signal file path.
        self._stop_signal_retry_after: Dict[str, float] = {}
        self._stop_signal_failures: Dict[str, int] = {}

        # Trigger support for pluggable turn detection
        from .triggers.registry import get_global_registry

        self.trigger_registry = get_global_registry()
        self.trigger_name = "next_turn"  # Default trigger (可配置)
        self._session_triggers: Dict[str, "TurnTrigger"] = {}  # Cache triggers per session

        # Owner id for DB-backed lease locks (cross-process).
        try:
            from .db.locks import make_lock_owner

            self.lock_owner = make_lock_owner("watcher")
        except Exception:
            self.lock_owner = f"watcher:{os.getpid()}"

        # Per-turn "processing" TTL: if a processing placeholder turn exists longer than this,
        # a new run may take over and re-process it to avoid permanent stuck states.
        self.processing_turn_ttl_seconds = 20 * 60  # 20 minutes

        # Layer 1: Session list cache (avoid re-scanning directories every 0.5s)
        self._cached_session_list: list[Path] = []
        self._session_list_last_scan: float = 0.0
        self._session_list_scan_interval: float = 30.0

        # Layer 2: Per-cycle sizes stash (shared between check_for_changes and turn-count cache)
        self._last_cycle_sizes: Dict[str, int] = {}

        # Layer 3: Turn count cache (avoid re-parsing JSONL for unchanged files)
        self._cached_turn_counts: dict[str, int] = {}
        self._cached_total_turn_counts: dict[str, int] = {}
        self._turn_count_file_stats: dict[str, tuple[float, int]] = {}

        # Signal directory for Stop hook integration
        self.signal_dir = Path.home() / ".aline" / ".signals"
        self.signal_dir.mkdir(parents=True, exist_ok=True)
        self.user_prompt_signal_dir = self.signal_dir / "user_prompt_submit"
        self.user_prompt_signal_dir.mkdir(parents=True, exist_ok=True)

        # Polling control:
        # - When the Stop hook is available, periodic session polling/scanning is unnecessary.
        # - The watcher remains responsible for processing fallback .signal files.
        self._polling_enabled: bool = True

        # Startup scan enqueue priority (lower than realtime stop-hook/signal work).
        self._startup_scan_priority: int = 5

        # Codex notify-hook fallback:
        # Some Codex CLIs don't support (or don't run) the Rust notify hook. When the watcher is
        # in signal-driven mode (polling disabled), those Codex sessions would otherwise never be
        # discovered/processed. We keep a lightweight Codex-only polling loop for that case.
        self._codex_notify_supported: bool | None = None
        self._codex_fallback_poll_enabled: bool = False
        self._codex_fallback_poll_last_scan: float = 0.0
        self._codex_fallback_poll_interval_seconds: float = 2.0
        self._codex_fallback_last_sizes: Dict[str, int] = {}
        self._codex_fallback_last_mtimes: Dict[str, float] = {}
        self._codex_notify_hook_cache: dict[str, tuple[float, bool]] = {}
        self._codex_notify_hook_cache_ttl_seconds: float = 30.0
        try:
            raw = os.environ.get("ALINE_CODEX_POLL_INTERVAL_SECONDS", "").strip()
            if raw:
                self._codex_fallback_poll_interval_seconds = max(0.5, float(raw))
        except Exception:
            self._codex_fallback_poll_interval_seconds = 2.0

    def _codex_notify_hook_installed_for_session_file(self, session_file: Path) -> bool | None:
        """Best-effort detect whether Aline's Codex notify hook is installed for this session."""
        try:
            from .codex_hooks.notify_hook_installer import ALINE_HOOK_MARKER
        except Exception:
            ALINE_HOOK_MARKER = "aline-codex-notify-hook"

        try:
            from .codex_home import codex_home_from_session_file
        except Exception:
            codex_home_from_session_file = None  # type: ignore[assignment]

        codex_home: Path | None = None
        if codex_home_from_session_file is not None:
            try:
                codex_home = codex_home_from_session_file(session_file)
            except Exception:
                codex_home = None
        if codex_home is None:
            try:
                p = session_file.resolve()
                for parent in p.parents:
                    if parent.name == "sessions":
                        codex_home = parent.parent
                        break
            except Exception:
                codex_home = None

        if codex_home is None:
            return None

        now = time.time()
        key = str(codex_home)
        cached = self._codex_notify_hook_cache.get(key)
        if cached and (now - float(cached[0] or 0.0)) < float(
            self._codex_notify_hook_cache_ttl_seconds
        ):
            return bool(cached[1])

        installed = False
        try:
            toml_path = codex_home / "config.toml"
            if not toml_path.exists():
                installed = False
            else:
                raw = toml_path.read_text(encoding="utf-8", errors="ignore")
                if ALINE_HOOK_MARKER in raw:
                    installed = True
                elif "notify" in raw and "notify_hook.py" in raw:
                    installed = True
                else:
                    installed = False
        except Exception:
            installed = False

        self._codex_notify_hook_cache[key] = (now, installed)
        return installed

    def _codex_fallback_discover_sessions(self) -> list[Path]:
        """Discover recent Codex sessions (bounded scan), best-effort."""
        try:
            from .adapters import get_adapter_registry

            adapter = get_adapter_registry().get_adapter("codex")
            if not adapter:
                return []
            sessions = [Path(p) for p in (adapter.discover_sessions() or [])]
            if not is_dashboard_only_enabled(self.config):
                return sessions
            return [
                p
                for p in sessions
                if self._is_session_trackable(p, session_type="codex", session_id=p.stem)
            ]
        except Exception:
            return []

    async def _codex_fallback_poll_sessions(self) -> None:
        """Fallback: enqueue Codex session_process jobs by scanning recent session files.

        Only runs when:
        - Codex auto-detection is enabled, and
        - watcher polling is disabled (signal-driven), and
        - Codex notify hook isn't available/reliable.
        """
        if not getattr(self.config, "auto_detect_codex", False):
            return
        if not self._codex_fallback_poll_enabled:
            return
        if self._polling_enabled:
            # Main polling already covers Codex via adapter discovery.
            return

        now = time.time()
        interval = float(self._codex_fallback_poll_interval_seconds)
        if (now - float(self._codex_fallback_poll_last_scan or 0.0)) < interval:
            return
        self._codex_fallback_poll_last_scan = now

        session_files = self._codex_fallback_discover_sessions()
        if not session_files:
            return

        current_sizes: Dict[str, int] = {}
        current_mtimes: Dict[str, float] = {}
        for p in session_files:
            if not self._is_session_trackable(
                p,
                session_type="codex",
                session_id=p.stem,
            ):
                continue
            try:
                st = p.stat()
            except Exception:
                continue
            k = str(p)
            try:
                current_sizes[k] = int(st.st_size)
                current_mtimes[k] = float(st.st_mtime)
            except Exception:
                continue

        notify_supported = self._codex_notify_supported is True
        changed: list[Path] = []
        for path_key, size in current_sizes.items():
            old_size = self._codex_fallback_last_sizes.get(path_key)
            old_mtime = self._codex_fallback_last_mtimes.get(path_key)
            mtime = current_mtimes.get(path_key)
            if old_size is None or old_mtime is None:
                changed.append(Path(path_key))
                try:
                    self._maybe_link_codex_terminal(Path(path_key))
                except Exception:
                    pass
                continue
            # If notify hook works *and* is installed for this CODEX_HOME, we only need
            # "new session discovery" here. Let notify drive subsequent updates to avoid
            # duplicate enqueues. If the hook isn't installed (common when Codex sessions are
            # started outside dashboard-managed CODEX_HOME prep), keep polling updates so
            # turns still get imported and titles can be generated.
            if notify_supported:
                try:
                    hook_installed = self._codex_notify_hook_installed_for_session_file(
                        Path(path_key)
                    )
                except Exception:
                    hook_installed = None
                if hook_installed is True:
                    continue
            if size != old_size or (mtime is not None and mtime != old_mtime):
                changed.append(Path(path_key))

        self._codex_fallback_last_sizes = current_sizes
        self._codex_fallback_last_mtimes = current_mtimes

        if not changed:
            return

        try:
            changed.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        except Exception:
            pass

        from .db import get_database

        db = get_database()
        # Throttle burst enqueues to keep UI responsive if many Codex sessions exist.
        for session_file in changed[:50]:
            if not session_file.exists():
                continue
            try:
                db.enqueue_session_process_job(  # type: ignore[attr-defined]
                    session_file_path=session_file,
                    session_id=session_file.stem,
                    workspace_path=None,
                    session_type="codex",
                    source_event="codex_poll",
                    priority=20,
                )
            except Exception:
                continue

    def _get_cached_session_list(self, force_rescan: bool = False) -> list[Path]:
        """Return cached list of active session files, re-scanning only every 30s.

        Args:
            force_rescan: If True, bypass the time gate and re-scan immediately.
        """
        now = time.time()
        if (
            not force_rescan
            and self._cached_session_list
            and (now - self._session_list_last_scan) < self._session_list_scan_interval
        ):
            return self._cached_session_list

        try:
            self._cached_session_list = find_all_active_sessions(self.config, project_path=None)
            self._session_list_last_scan = now
        except PermissionError:
            if not hasattr(self, "_permission_error_logged"):
                self._permission_error_logged = True
                logger.error("PERMISSION DENIED: Cannot access Claude Code sessions directory")
        except Exception as e:
            logger.error(f"Error scanning session list: {e}", exc_info=True)

        return self._cached_session_list

    def _get_cycle_session_stats(
        self, *, force_rescan: bool = False
    ) -> tuple[list[Path], Dict[str, int], Dict[str, float]]:
        """Single stat() pass per cycle over the cached session list.

        Returns:
            (session_files, sizes_dict, mtimes_dict)
        """
        session_files = self._get_cached_session_list(force_rescan=force_rescan)
        sizes: Dict[str, int] = {}
        mtimes: Dict[str, float] = {}
        force_rescan = False

        for session_file in list(session_files):  # copy so we can mutate
            path_key = str(session_file)
            try:
                if session_file.is_dir():
                    # Handle directory-based sessions (e.g., Antigravity brain directories)
                    artifacts = ["task.md", "walkthrough.md", "implementation_plan.md"]
                    total_size = 0
                    max_mtime = 0.0
                    for artifact_name in artifacts:
                        artifact_path = session_file / artifact_name
                        if artifact_path.exists():
                            artifact_stat = artifact_path.stat()
                            total_size += artifact_stat.st_size
                            max_mtime = max(max_mtime, artifact_stat.st_mtime)
                    if max_mtime > 0:
                        sizes[path_key] = total_size
                        mtimes[path_key] = max_mtime
                else:
                    stat = session_file.stat()
                    sizes[path_key] = stat.st_size
                    mtimes[path_key] = stat.st_mtime
            except FileNotFoundError:
                # File disappeared — prune from cache, force re-scan next cycle
                if session_file in self._cached_session_list:
                    self._cached_session_list.remove(session_file)
                force_rescan = True
            except Exception as e:
                logger.debug(f"Error stat-ing {path_key}: {e}")

        if force_rescan:
            self._session_list_last_scan = 0.0  # Force re-scan next cycle

        live_files = [f for f in session_files if str(f) in sizes]
        return live_files, sizes, mtimes

    def _watcher_session_stats_path(self) -> Path:
        root = Path.home() / ".aline"
        try:
            root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return root / "watcher_session_stats.json"

    def _load_persisted_session_stats(self) -> dict[str, dict[str, float]]:
        """Load last-run session stats (best-effort)."""
        path = self._watcher_session_stats_path()
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict[str, float]] = {}
        for k, v in raw.items():
            if not isinstance(k, str) or not isinstance(v, dict):
                continue
            try:
                size = float(v.get("size") or 0.0)
                mtime = float(v.get("mtime") or 0.0)
            except Exception:
                continue
            out[k] = {"size": size, "mtime": mtime}
        return out

    def _save_persisted_session_stats(
        self, sizes: Dict[str, int], mtimes: Dict[str, float]
    ) -> None:
        """Persist current session stats for next startup scan (best-effort, atomic)."""
        path = self._watcher_session_stats_path()
        tmp = path.with_suffix(".json.tmp")
        payload: dict[str, dict[str, float]] = {}
        for p, size in (sizes or {}).items():
            try:
                payload[str(p)] = {
                    "size": float(size or 0),
                    "mtime": float(mtimes.get(p, 0.0) or 0.0),
                }
            except Exception:
                continue
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            return

    def _get_stats_for_paths(self, paths: list[Path]) -> tuple[Dict[str, int], Dict[str, float]]:
        """Stat session files by explicit path (fast path for startup scan)."""
        sizes: Dict[str, int] = {}
        mtimes: Dict[str, float] = {}
        for p in paths:
            try:
                st = p.stat()
            except Exception:
                continue
            key = str(p)
            try:
                sizes[key] = int(st.st_size)
                mtimes[key] = float(st.st_mtime)
            except Exception:
                continue
        return sizes, mtimes

    def _startup_scan_collect_candidates(
        self,
    ) -> tuple[list[Path], Dict[str, int], Dict[str, float], StartupScanReport]:
        """
        Collect backlog session files to enqueue on startup.

        Two phases:
        1) Fast path: re-stat previously persisted session paths (no directory scan).
        2) Full scan: scan active session directories to find new/unknown sessions.
        """
        prev = self._load_persisted_session_stats()

        prev_paths: list[Path] = []
        for k in prev.keys():
            if not isinstance(k, str):
                continue
            if not k or k in (".", ".."):
                continue
            try:
                prev_paths.append(Path(k))
            except Exception:
                continue

        # Phase 1: fast path stats for known session file paths.
        known_sizes, known_mtimes = self._get_stats_for_paths(prev_paths)
        prev_missing = max(0, len(prev_paths) - len(known_sizes))

        candidates: list[Path] = []
        candidate_keys: set[str] = set()
        prev_changed = 0

        for p in prev_paths:
            key = str(p)
            if key not in known_sizes:
                continue
            size = int(known_sizes.get(key, 0) or 0)
            mtime = float(known_mtimes.get(key, 0.0) or 0.0)
            prev_stats = prev.get(key) or {}
            if float(prev_stats.get("size") or 0.0) == float(size) and float(
                prev_stats.get("mtime") or 0.0
            ) == float(mtime):
                continue
            prev_changed += 1
            if key not in candidate_keys:
                candidates.append(p)
                candidate_keys.add(key)

        # Phase 2: full scan of watch paths to find unknown/new sessions.
        session_files, scan_sizes, scan_mtimes = self._get_cycle_session_stats(force_rescan=True)
        scan_new = 0
        scan_changed = 0

        for session_file in session_files:
            key = str(session_file)
            size = scan_sizes.get(key)
            mtime = scan_mtimes.get(key)
            if size is None or mtime is None:
                continue
            prev_stats = prev.get(key)
            if prev_stats is None:
                scan_new += 1
                if key not in candidate_keys:
                    candidates.append(session_file)
                    candidate_keys.add(key)
                continue
            if float(prev_stats.get("size") or 0.0) == float(size) and float(
                prev_stats.get("mtime") or 0.0
            ) == float(mtime):
                continue
            scan_changed += 1
            if key not in candidate_keys:
                candidates.append(session_file)
                candidate_keys.add(key)

        # Merge baselines for next cycle; include known paths even if no longer in scan.
        merged_sizes: Dict[str, int] = dict(known_sizes)
        merged_sizes.update(scan_sizes)
        merged_mtimes: Dict[str, float] = dict(known_mtimes)
        merged_mtimes.update(scan_mtimes)

        report = StartupScanReport(
            prev_paths=len(prev_paths),
            prev_missing=int(prev_missing),
            prev_changed=int(prev_changed),
            scan_paths=len(session_files),
            scan_new=int(scan_new),
            scan_changed=int(scan_changed),
            candidates=len(candidates),
        )
        return candidates, merged_sizes, merged_mtimes, report

    async def _startup_scan_enqueue_changed_sessions(self) -> None:
        """Scan sessions once on startup and enqueue backlog work (low priority)."""
        try:
            candidates, sizes, mtimes, report = self._startup_scan_collect_candidates()

            # Update in-memory baselines for potential later use (even if polling is disabled).
            self.last_session_sizes = sizes
            self.last_session_mtimes = mtimes

            from .db import get_database

            db = get_database()

            enqueued = 0
            for session_file in candidates:
                try:
                    session_type = self._detect_session_type(session_file)
                    if not self._is_session_trackable(
                        session_file,
                        session_type=session_type,
                        session_id=session_file.stem,
                    ):
                        continue
                    db.enqueue_session_process_job(  # type: ignore[attr-defined]
                        session_file_path=session_file,
                        session_id=session_file.stem,
                        workspace_path=None,
                        session_type=session_type,
                        source_event="startup_scan",
                        priority=self._startup_scan_priority,
                    )
                    enqueued += 1
                except Exception:
                    continue

            if enqueued:
                logger.info(
                    "Startup scan enqueued %s session_process job(s) (prev=%s missing=%s prev_changed=%s scan=%s new=%s scan_changed=%s)",
                    enqueued,
                    report.prev_paths,
                    report.prev_missing,
                    report.prev_changed,
                    report.scan_paths,
                    report.scan_new,
                    report.scan_changed,
                )
        except Exception as e:
            logger.warning(f"Startup scan failed: {e}")

    def _get_cached_turn_counts(
        self, session_file: Path, current_mtime: float, current_size: int
    ) -> tuple[int, int]:
        """Return (completed_turns, total_turns) for a session, using cache when file unchanged.

        Args:
            session_file: Path to the session file.
            current_mtime: The mtime from the current cycle's stat pass.
            current_size: The size from the current cycle's stat pass.

        Returns:
            (completed_turn_count, total_turn_count)
        """
        path_key = str(session_file)
        cached_stats = self._turn_count_file_stats.get(path_key)

        if cached_stats and cached_stats == (current_mtime, current_size):
            return (
                self._cached_turn_counts.get(path_key, 0),
                self._cached_total_turn_counts.get(path_key, 0),
            )

        # File changed — re-parse
        completed = self._count_complete_turns(session_file)
        total = self._get_total_turn_count(session_file)

        self._cached_turn_counts[path_key] = completed
        self._cached_total_turn_counts[path_key] = total
        self._turn_count_file_stats[path_key] = (current_mtime, current_size)

        return completed, total

    def _maybe_link_codex_terminal(self, session_file: Path) -> None:
        """Best-effort: bind a Codex session file to the most likely active Codex terminal."""
        try:
            if self._detect_session_type(session_file) != "codex":
                return
        except Exception:
            return

        try:
            from datetime import datetime, timezone

            from .codex_home import (
                agent_id_from_codex_session_file,
                codex_home_owner_from_session_file,
            )
            from .codex_terminal_linker import (
                read_codex_session_meta,
                select_agent_for_codex_session,
            )
            from .db import get_database

            meta = read_codex_session_meta(session_file)
            if meta is None:
                return

            db = get_database(read_only=False)
            agents = db.list_agents(status="active", limit=1000)
            # Deterministic mapping: session file stored under ~/.aline/codex_homes/<terminal_id>/...
            owner = codex_home_owner_from_session_file(session_file)
            agent_id = None
            agent_info_id = None
            path_agent_info_id = agent_id_from_codex_session_file(session_file)
            owner_agent_info_id = path_agent_info_id
            if owner:
                if owner[0] == "terminal":
                    agent_id = owner[1]
                elif owner[0] == "agent":
                    agent_info_id = owner[1]
                    owner_agent_info_id = agent_info_id
                    scoped_agents = [
                        a
                        for a in agents
                        if getattr(a, "provider", "") == "codex"
                        and getattr(a, "status", "") == "active"
                        and (getattr(a, "source", "") or "") == f"agent:{agent_info_id}"
                    ]
                    agent_id = select_agent_for_codex_session(
                        scoped_agents, session=meta, max_time_delta_seconds=None
                    )
            if not agent_id:
                # Fallback heuristic mapping (legacy default ~/.codex/sessions).
                agent_id = select_agent_for_codex_session(agents, session=meta)
            if not agent_id:
                return

            # Get existing agent to preserve agent_info_id in source field
            existing_agent = db.get_agent_by_id(agent_id)
            agent_info_id = None
            existing_source = None
            if existing_agent:
                existing_source = existing_agent.source or ""
                if existing_source.startswith("agent:"):
                    agent_info_id = existing_source[6:]

            if not agent_info_id and owner_agent_info_id:
                agent_info_id = owner_agent_info_id

            if existing_source:
                source = existing_source
            elif agent_info_id:
                source = f"agent:{agent_info_id}"
            else:
                source = "codex:auto-link"

            db.update_agent(
                agent_id,
                provider="codex",
                session_type="codex",
                session_id=session_file.stem,
                transcript_path=str(session_file),
                cwd=meta.cwd,
                project_dir=meta.cwd,
                source=source,
            )

            try:
                db.insert_window_link(
                    terminal_id=agent_id,
                    agent_id=agent_info_id,
                    session_id=session_file.stem,
                    provider="codex",
                    source="codex:watcher",
                    ts=time.time(),
                )
            except Exception:
                pass

            # Link session to agent_info if available (bidirectional linking)
            if agent_info_id:
                try:
                    # Ensure the session row exists so the agent association doesn't get lost
                    # when this is called before the session is processed into the DB.
                    started_at = meta.started_at
                    if started_at is None:
                        started_at = datetime.fromtimestamp(session_file.stat().st_mtime)
                    elif started_at.tzinfo is not None:
                        started_at = started_at.astimezone(timezone.utc).replace(tzinfo=None)
                    db.get_or_create_session(
                        session_id=session_file.stem,
                        session_file_path=session_file,
                        session_type="codex",
                        started_at=started_at,
                        workspace_path=meta.cwd,
                        metadata={"source": "codex:watcher"},
                        agent_id=agent_info_id,
                    )
                    db.update_session_agent_id(session_file.stem, agent_info_id)
                except Exception:
                    pass
        except Exception:
            return

    def _agent_info_id_for_codex_session(self, session_file: Path, *, db=None) -> Optional[str]:
        """Best-effort: resolve agent_info_id from a codex session file."""
        try:
            from .codex_home import (
                agent_id_from_codex_session_file,
                codex_home_owner_from_session_file,
            )
        except Exception:
            return None

        try:
            agent_id = agent_id_from_codex_session_file(session_file)
            if agent_id:
                return agent_id
        except Exception:
            pass

        try:
            owner = codex_home_owner_from_session_file(session_file)
        except Exception:
            owner = None
        if not owner or owner[0] != "terminal":
            return None
        terminal_id = owner[1]
        if not terminal_id:
            return None

        try:
            if db is None:
                from .db import get_database

                db = get_database(read_only=True)
            agent = db.get_agent_by_id(terminal_id)
        except Exception:
            return None

        source = (agent.source or "").strip() if agent else ""
        if source.startswith("agent:"):
            return source[6:]
        return None

    def _terminal_id_for_codex_session(self, session_file: Path) -> Optional[str]:
        """Best-effort: resolve terminal_id from codex session file path."""
        try:
            from .codex_home import codex_home_owner_from_session_file
        except Exception:
            return None

        try:
            owner = codex_home_owner_from_session_file(session_file)
        except Exception:
            owner = None
        if not owner or owner[0] != "terminal":
            return None
        return owner[1]

    def _is_session_trackable(
        self,
        session_file: Path,
        *,
        session_type: Optional[str] = None,
        session_id: str = "",
        terminal_id: str = "",
        agent_id: str = "",
    ) -> bool:
        """Apply dashboard-only tracking filter for Claude/Codex sessions."""
        if not is_dashboard_only_enabled(self.config):
            return True

        stype = str(session_type or "").strip().lower()
        if not stype:
            stype = str(self._detect_session_type(session_file) or "").strip().lower()

        try:
            from .db import get_database

            db = get_database(read_only=True)
        except Exception:
            db = None

        try:
            return is_session_trackable_for_dashboard_only(
                session_type=stype,
                session_file=session_file,
                session_id=session_id or session_file.stem,
                terminal_id=terminal_id,
                agent_id=agent_id,
                db=db,
            )
        finally:
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass

    async def start(self):
        """Start watching session files."""
        if not self.config.mcp_auto_commit:
            logger.info("Auto-commit disabled in config")
            print("[Watcher] Auto-commit disabled in config", file=sys.stderr)
            return

        self.running = True
        logger.info("Started watching for dialogue completion")
        logger.info(f"Mode: Multi-project monitoring (all Claude Code projects)")
        logger.info(f"Trigger: Per-request (at end of each AI response)")
        logger.info(f"Supports: Claude Code & Codex (auto-detected)")
        logger.info(f"Debounce: {self.debounce_delay}s, Cooldown: {self.min_commit_interval}s")
        print("[Watcher] Started watching for dialogue completion", file=sys.stderr)
        print(
            f"[Watcher] Mode: Multi-project monitoring (all Claude Code projects)", file=sys.stderr
        )
        print(f"[Watcher] Trigger: Per-request (at end of each AI response)", file=sys.stderr)
        print(f"[Watcher] Supports: Claude Code & Codex (auto-detected)", file=sys.stderr)
        print(
            f"[Watcher] Debounce: {self.debounce_delay}s, Cooldown: {self.min_commit_interval}s",
            file=sys.stderr,
        )

        # Codex: require Rust CLI notify hook for reliable integration.
        if getattr(self.config, "auto_detect_codex", False):
            try:
                from .codex_hooks.notify_hook_installer import codex_cli_supports_notify_hook

                supported = codex_cli_supports_notify_hook()
            except Exception:
                supported = None

            self._codex_notify_supported = supported

            if supported is False:
                # Notify hook unsupported: warn. We'll still run a Codex-only fallback poll in
                # signal-driven mode, so sessions are discovered even without notify.
                if not getattr(self, "_codex_legacy_warning_logged", False):
                    self._codex_legacy_warning_logged = True
                    msg = (
                        "[Watcher] Codex detected, but your Codex CLI does not support the Rust "
                        "notify hook. Please update Codex CLI to a recent Rust version. "
                        "Falling back to lightweight Codex polling."
                    )
                    print(msg, file=sys.stderr)
                    logger.warning(msg)
            elif supported is None:
                # If we can't detect support (e.g. codex not on PATH), don't spam logs. The
                # fallback poll will still provide best-effort discovery.
                self._codex_notify_supported = None

        # Auto-install Claude Code Stop hook for reliable turn completion detection
        stop_hook_ready = False
        try:
            from .claude_hooks.stop_hook_installer import ensure_stop_hook_installed

            if ensure_stop_hook_installed(quiet=True):
                logger.info("Claude Code Stop hook is ready")
                stop_hook_ready = True
            else:
                logger.warning("Failed to install Stop hook, falling back to polling-only mode")
        except Exception as e:
            logger.debug(f"Stop hook installation skipped: {e}")

        disable_polling = os.environ.get("ALINE_WATCHER_DISABLE_POLLING", "") == "1"
        force_polling = os.environ.get("ALINE_WATCHER_ENABLE_POLLING", "") == "1"
        if force_polling:
            self._polling_enabled = True
        elif disable_polling:
            self._polling_enabled = False
        else:
            # Default: no polling (event-driven hooks preferred). Use --enable env var to opt in.
            self._polling_enabled = False

        if self._polling_enabled:
            logger.info("Watcher polling enabled (legacy fallback mode)")
        else:
            logger.info("Watcher polling disabled (stop-hook/signal-driven mode)")

        # In signal-driven mode, also do a lightweight Codex-only fallback poll so new Codex
        # sessions show up even before the first notify event.
        try:
            disable_codex_fallback = os.environ.get("ALINE_CODEX_DISABLE_FALLBACK_POLL", "") == "1"
        except Exception:
            disable_codex_fallback = False
        if (
            not disable_codex_fallback
            and not self._polling_enabled
            and getattr(self.config, "auto_detect_codex", False)
        ):
            self._codex_fallback_poll_enabled = True

        if self.config.enable_early_session_title:
            # Auto-install Claude Code UserPromptSubmit hook for early session title
            try:
                from .claude_hooks.user_prompt_submit_hook_installer import (
                    ensure_user_prompt_submit_hook_installed,
                )

                if ensure_user_prompt_submit_hook_installed(quiet=True):
                    logger.info("Claude Code UserPromptSubmit hook is ready")
                else:
                    logger.warning("Failed to install UserPromptSubmit hook")
            except Exception as e:
                logger.debug(f"UserPromptSubmit hook installation skipped: {e}")

        # Note: Idle timeout checking is now integrated into main loop instead of separate task

        # Ensure global config/database exists (no per-project init)
        logger.info("Ensuring global Aline initialization")
        print("[Watcher] Ensuring global Aline initialization", file=sys.stderr)
        await self.auto_init_projects()

        # Startup scan: enqueue backlog sessions changed since last run (no debounce, low priority).
        await self._startup_scan_enqueue_changed_sessions()

        # Poll for file changes more frequently
        while self.running:
            try:
                # Priority 1: Check Stop hook signals (immediate trigger, no debounce)
                await self._check_stop_signals()

                # Priority 1.5: Codex notify fallback signals (rare).
                await self._check_codex_notify_signals()

                # Priority 1.6: Codex fallback polling when notify is unavailable.
                await self._codex_fallback_poll_sessions()

                # Priority 2: Check UserPromptSubmit signals (early session title)
                if self.config.enable_early_session_title:
                    await self._check_user_prompt_submit_signals()

                if self._polling_enabled:
                    # Single stat pass per cycle (Layer 2)
                    cycle_sessions, cycle_sizes, cycle_mtimes = self._get_cycle_session_stats()
                    self._last_cycle_sizes = cycle_sizes

                    # Legacy: fallback polling mechanism (debounced)
                    await self.check_for_changes(cycle_sizes, cycle_mtimes)

                await asyncio.sleep(0.5)  # Check every 0.5 seconds for responsiveness
            except Exception as e:
                logger.error(f"Error in check loop: {e}", exc_info=True)
                print(f"[Watcher] Error: {e}", file=sys.stderr)
                await asyncio.sleep(1.0)

    async def stop(self):
        """Stop watching."""
        self.running = False
        if self.pending_commit_task:
            self.pending_commit_task.cancel()
        try:
            self._save_persisted_session_stats(self.last_session_sizes, self.last_session_mtimes)
        except Exception:
            pass
        logger.info("Watcher stopped")
        print("[Watcher] Stopped", file=sys.stderr)

    async def _check_stop_signals(self):
        """
        Check for Stop hook signal files.

        When Claude Code's Stop hook fires, it writes a signal file to ~/.aline/.signals/.
        This method processes those signals for immediate turn completion detection,
        bypassing the 10-second debounce delay.

        The Stop hook is the authoritative signal that a turn has completed. We pass
        target_turn to _do_commit to ensure the turn count baseline is correctly updated,
        since count_complete_turns() intentionally excludes the last turn (to prevent
        false positives from the polling mechanism).
        """
        try:
            if not self.signal_dir.exists():
                return

            for signal_file in self.signal_dir.glob("*.signal"):
                try:
                    signal_key = str(signal_file)
                    now = time.time()
                    retry_after = float(self._stop_signal_retry_after.get(signal_key, 0.0) or 0.0)
                    if retry_after and now < retry_after:
                        continue

                    # Read signal data
                    signal_data = json.loads(signal_file.read_text())
                    session_id = signal_data.get("session_id", "")
                    project_dir = signal_data.get("project_dir", "")
                    transcript_path = signal_data.get("transcript_path", "")
                    no_track = bool(signal_data.get("no_track", False))
                    agent_id = signal_data.get("agent_id", "")
                    terminal_id = str(signal_data.get("terminal_id", "") or "").strip()

                    logger.info(f"Stop signal received for session {session_id}")
                    print(f"[Watcher] Stop signal received for {session_id}", file=sys.stderr)

                    # Find the session file
                    session_file = None
                    if transcript_path:
                        candidate = Path(transcript_path)
                        if candidate.exists():
                            session_file = candidate
                    elif session_id:
                        # Lightweight fallback: avoid global scanning; only consider sessions we already
                        # know about from startup scan / previous signals.
                        for known_path in list(self.last_session_sizes.keys()):
                            try:
                                p = Path(known_path)
                            except Exception:
                                continue
                            if p.stem != session_id:
                                continue
                            if p.exists():
                                session_file = p
                                break

                    if session_file and session_file.exists():
                        if not self._is_session_trackable(
                            session_file,
                            session_type=self._detect_session_type(session_file),
                            session_id=session_id or session_file.stem,
                            terminal_id=terminal_id,
                            agent_id=agent_id,
                        ):
                            signal_file.unlink(missing_ok=True)
                            continue
                        # Enqueue a per-session job; worker will determine which turns to process.
                        from .db import get_database

                        db = get_database()
                        try:
                            db.enqueue_session_process_job(  # type: ignore[attr-defined]
                                session_file_path=session_file,
                                session_id=session_id or session_file.stem,
                                workspace_path=(project_dir or None),
                                session_type=self._detect_session_type(session_file),
                                source_event="stop",
                                no_track=no_track,
                                agent_id=agent_id if agent_id else None,
                                terminal_id=terminal_id if terminal_id else None,
                            )

                            if agent_id and session_id:
                                try:
                                    db.update_session_agent_id(session_id, agent_id)
                                except Exception:
                                    pass

                            self._stop_signal_retry_after.pop(signal_key, None)
                            self._stop_signal_failures.pop(signal_key, None)
                            try:
                                st = session_file.stat()
                                self.last_session_sizes[str(session_file)] = int(st.st_size)
                                self.last_session_mtimes[str(session_file)] = float(st.st_mtime)
                            except Exception:
                                pass
                            signal_file.unlink(missing_ok=True)
                        except Exception as e:
                            failures = int(self._stop_signal_failures.get(signal_key, 0) or 0) + 1
                            self._stop_signal_failures[signal_key] = failures
                            delay = float(min(30.0, 1.0 * (2 ** min(failures, 5))))
                            self._stop_signal_retry_after[signal_key] = now + delay
                            logger.warning(
                                f"Failed to enqueue stop-hook session_process for {session_id} (retry in {delay:.0f}s): {e}"
                            )
                    else:
                        logger.warning(f"Session file not found for {session_id}")
                        # Keep signal for a short time; file discovery can lag behind hook.
                        failures = int(self._stop_signal_failures.get(signal_key, 0) or 0) + 1
                        self._stop_signal_failures[signal_key] = failures
                        self._stop_signal_retry_after[signal_key] = now + 5.0

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid signal file {signal_file.name}: {e}")
                    signal_file.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Error processing signal {signal_file.name}: {e}")
                    # Delete corrupted signal files to prevent infinite loops
                    signal_file.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Error checking stop signals: {e}", exc_info=True)

    async def _check_codex_notify_signals(self) -> None:
        """Process Codex notify-hook fallback signals (rare path)."""
        try:
            from .codex_hooks import codex_notify_signal_dir

            signal_dir = codex_notify_signal_dir()
            if not signal_dir.exists():
                return

            now = time.time()
            for signal_file in signal_dir.glob("*.signal"):
                try:
                    data = json.loads(signal_file.read_text(encoding="utf-8"))
                except Exception:
                    signal_file.unlink(missing_ok=True)
                    continue

                ts = float(data.get("timestamp") or 0.0)
                if ts and now - ts < 0.2:
                    continue

                transcript_path = str(data.get("transcript_path") or "") or str(
                    data.get("session_file_path") or ""
                )
                if not transcript_path:
                    signal_file.unlink(missing_ok=True)
                    continue

                session_file = Path(transcript_path)
                if not session_file.exists():
                    signal_file.unlink(missing_ok=True)
                    continue

                session_id = str(data.get("session_id") or session_file.stem).strip()
                project_dir = str(data.get("project_dir") or data.get("cwd") or "")
                no_track = bool(data.get("no_track") or False)
                agent_id = str(data.get("agent_id") or "").strip()
                terminal_id = str(data.get("terminal_id") or "").strip()

                if not self._is_session_trackable(
                    session_file,
                    session_type="codex",
                    session_id=session_id,
                    terminal_id=terminal_id,
                    agent_id=agent_id,
                ):
                    signal_file.unlink(missing_ok=True)
                    continue

                from .db import get_database

                db = get_database()
                try:
                    db.enqueue_session_process_job(  # type: ignore[attr-defined]
                        session_file_path=session_file,
                        session_id=session_id,
                        workspace_path=(project_dir or None),
                        session_type="codex",
                        source_event="notify",
                        no_track=no_track,
                        agent_id=agent_id if agent_id else None,
                        terminal_id=terminal_id if terminal_id else None,  # type: ignore[arg-type]
                    )
                except TypeError:
                    try:
                        db.enqueue_session_process_job(  # type: ignore[attr-defined]
                            session_file_path=session_file,
                            session_id=session_id,
                            workspace_path=(project_dir or None),
                            session_type="codex",
                            source_event="notify",
                            no_track=no_track,
                            agent_id=agent_id if agent_id else None,
                        )
                    except Exception:
                        pass
                except Exception:
                    pass

                signal_file.unlink(missing_ok=True)
        except Exception:
            return

    async def _check_user_prompt_submit_signals(self):
        """Process UserPromptSubmit hook signals for early session title generation."""
        try:
            if not self.user_prompt_signal_dir.exists():
                return

            for signal_file in self.user_prompt_signal_dir.glob("*.signal"):
                try:
                    signal_data = json.loads(signal_file.read_text())
                except (json.JSONDecodeError, Exception):
                    signal_file.unlink(missing_ok=True)
                    continue

                session_id = str(signal_data.get("session_id") or "")
                prompt = str(signal_data.get("prompt") or "")
                transcript_path = str(signal_data.get("transcript_path") or "")
                project_dir = str(signal_data.get("project_dir") or "")
                no_track = bool(signal_data.get("no_track", False))
                agent_id = str(signal_data.get("agent_id") or "")
                terminal_id = str(signal_data.get("terminal_id") or "")

                session_file = None
                if transcript_path and Path(transcript_path).exists():
                    session_file = Path(transcript_path)
                elif session_id:
                    session_file = self._find_session_by_id(session_id)

                if not session_file or not session_file.exists():
                    signal_file.unlink(missing_ok=True)
                    continue
                if not session_id:
                    session_id = session_file.stem

                # Only apply to Claude sessions for now.
                session_type = self._detect_session_type(session_file)
                if session_type != "claude":
                    signal_file.unlink(missing_ok=True)
                    continue
                if not self._is_session_trackable(
                    session_file,
                    session_type=session_type,
                    session_id=session_id,
                    terminal_id=terminal_id,
                    agent_id=agent_id,
                ):
                    signal_file.unlink(missing_ok=True)
                    continue

                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._set_early_session_title,
                    session_file,
                    session_id,
                    prompt,
                    project_dir,
                    no_track,
                )

                # Link session to agent/terminal/window_link
                # The hook's update_terminal_mapping may have failed (import issues)
                # or its update_session_agent_id may have been a no-op (session didn't
                # exist yet).  Re-do all linking now that the session record is guaranteed
                # to exist after _set_early_session_title.
                try:
                    from .db import get_database

                    db = get_database()
                    if agent_id and session_id:
                        db.update_session_agent_id(session_id, agent_id)
                    if terminal_id and session_id:
                        db.update_agent(
                            terminal_id,
                            session_id=session_id,
                            provider="claude",
                            session_type="claude",
                        )
                        db.insert_window_link(
                            terminal_id=terminal_id,
                            agent_id=agent_id if agent_id else None,
                            session_id=session_id,
                            provider="claude",
                            source="early_session_title",
                            ts=time.time(),
                        )
                except Exception:
                    pass

                signal_file.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Error checking user prompt signals: {e}", exc_info=True)

    def _get_total_turn_count(self, session_file: Path) -> int:
        """
        Get the total number of turns in a session file (including the last turn).

        Unlike _count_complete_turns() which excludes the last turn for safety,
        this method counts ALL turns. Used by Stop Hook processing where we have
        authoritative confirmation that the last turn is complete.

        Returns:
            Total number of turns in the session
        """
        try:
            trigger = self._get_trigger_for_session(session_file)
            if not trigger:
                return 0

            # For Claude trigger, use get_detailed_analysis to get total turns
            if hasattr(trigger, "get_detailed_analysis"):
                analysis = trigger.get_detailed_analysis(session_file)
                return analysis.get("total_turns", 0)

            # Fallback: use count_complete_turns + 1 (assuming last turn just completed)
            return trigger.count_complete_turns(session_file) + 1

        except Exception as e:
            logger.debug(f"Error getting total turn count for {session_file.name}: {e}")
            return 0

    def _get_new_completed_turn_numbers(self, session_file: Path) -> list[int]:
        """
        Determine which completed turn numbers are newly observed and should be enqueued.

        Note: for Claude Code, this only covers non-last turns, because
        ClaudeTrigger.count_complete_turns() intentionally excludes the last turn
        to avoid false positives. The last turn is handled by the Stop hook path.
        """
        session_path = str(session_file)
        session_type = self._detect_session_type(session_file)

        current_count = self._count_complete_turns(session_file)
        last_count = self.last_stop_reason_counts.get(session_path, 0)

        if current_count <= last_count:
            return []

        return list(range(int(last_count) + 1, int(current_count) + 1))

    def _find_session_by_id(self, session_id: str) -> Optional[Path]:
        """
        Find a session file by its ID.

        Args:
            session_id: The session ID (typically UUID or filename stem)

        Returns:
            Path to the session file, or None if not found
        """
        try:
            # Search in Claude Code sessions directory
            claude_base = Path.home() / ".claude" / "projects"
            if claude_base.exists():
                for project_dir in claude_base.iterdir():
                    if project_dir.is_dir():
                        session_file = project_dir / f"{session_id}.jsonl"
                        if session_file.exists():
                            return session_file

            # Also check currently tracked sessions
            for session_path in self.last_session_sizes.keys():
                path = Path(session_path)
                if path.stem == session_id and path.exists():
                    return path

            return None

        except Exception as e:
            logger.debug(f"Error finding session by ID {session_id}: {e}")
            return None

    def _get_session_stats(self) -> tuple[Dict[str, int], Dict[str, float]]:
        """Get (sizes, mtimes) for all active session files.

        Delegates to _get_cycle_session_stats() which uses the cached session list
        and performs a single stat() pass.
        """
        _, sizes, mtimes = self._get_cycle_session_stats()
        logger.debug(f"Tracked {len(sizes)} session file(s) across all projects")
        return sizes, mtimes

    def _get_session_sizes(self) -> Dict[str, int]:
        """Get current sizes of all active session files across all projects."""
        sizes, _ = self._get_session_stats()
        return sizes

    def _get_stop_reason_counts(self) -> Dict[str, int]:
        """Get current count of turn completion markers in all active session files across all projects."""
        counts = {}
        try:
            session_files = find_all_active_sessions(self.config, project_path=None)
            for session_file in session_files:
                if session_file.exists():
                    counts[str(session_file)] = self._count_complete_turns(session_file)
        except Exception as e:
            print(f"[Watcher] Error getting turn counts: {e}", file=sys.stderr)
        return counts

    def _detect_project_path(self) -> Optional[Path]:
        """
        Try to detect the current git repository root.

        Returns None if not inside a git repo (for multi-project mode).
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=True,
            )
            repo_path = Path(result.stdout.strip())
            if repo_path.exists():
                return repo_path
        except subprocess.CalledProcessError:
            current_dir = Path.cwd()
            if (current_dir / ".git").exists():
                return current_dir
        except Exception as e:
            logger.debug(f"Could not detect project path: {e}")
        return None

    async def _catch_up_uncommitted_turns(self):
        """
        On startup, attempt to commit any turns that were missed while watcher was offline.

        Uses SQLite (turns table) to know last committed turn.
        Limits catch-up to max_catchup_sessions (default: 3) most recent sessions.
        """
        try:
            session_files = find_all_active_sessions(self.config, project_path=None)
            from .db import get_database

            db = get_database()

            # Sort by mtime descending (newest first)
            session_files.sort(key=lambda f: f.stat().st_mtime if f.exists() else 0, reverse=True)

            # Limit to max_catchup_sessions
            max_sessions = getattr(self.config, "max_catchup_sessions", 3)
            if len(session_files) > max_sessions:
                sessions_to_process = session_files[:max_sessions]
                skipped_count = len(session_files) - max_sessions
                logger.info(
                    f"Limiting catch-up to {max_sessions} most recent sessions, skipping {skipped_count} older sessions"
                )
                print(
                    f"[Watcher] Limiting catch-up to {max_sessions} most recent sessions ({skipped_count} skipped)",
                    file=sys.stderr,
                )
                print(
                    f"[Watcher] Use 'aline watcher session list' to see all sessions",
                    file=sys.stderr,
                )
            else:
                sessions_to_process = session_files

            for session_file in sessions_to_process:
                if not session_file.exists():
                    continue

                project_path = self._extract_project_path(session_file)
                if not project_path:
                    logger.debug(f"Skip catch-up (no project) for {session_file.name}")
                    continue

                session_id = session_file.stem
                session_type = self._detect_session_type(session_file)
                # For catch-up, include last turn for Claude (Stop hook may have been missed).
                if session_type == "claude":
                    current_count = self._get_total_turn_count(session_file)
                else:
                    current_count = self._count_complete_turns(session_file)

                # Get the set of turn numbers that have been committed
                # This detects gaps in turn numbers, not just trailing uncommitted turns
                committed_turns = db.get_committed_turn_numbers(session_id)
                expected_turns = set(range(1, current_count + 1))
                missing_turns = sorted(expected_turns - committed_turns)

                if not missing_turns:
                    # All turns are committed, align in-memory baseline
                    self.last_stop_reason_counts[str(session_file)] = current_count
                    continue

                logger.info(
                    f"Catch-up: {session_file.name} missing {len(missing_turns)} turn(s): {missing_turns}"
                )
                print(
                    f"[Watcher] Catch-up {session_file.name}: {len(missing_turns)} missing turn(s)",
                    file=sys.stderr,
                )

                agent_id = None
                if session_type == "codex":
                    agent_id = self._agent_info_id_for_codex_session(session_file, db=db)
                    terminal_id = self._terminal_id_for_codex_session(session_file)
                    if terminal_id:
                        try:
                            db.insert_window_link(
                                terminal_id=terminal_id,
                                agent_id=agent_id,
                                session_id=session_id,
                                provider="codex",
                                source="codex:watcher",
                                ts=time.time(),
                            )
                        except Exception:
                            pass

                enqueued = 0
                for turn in missing_turns:
                    try:
                        db.enqueue_turn_summary_job(  # type: ignore[attr-defined]
                            session_file_path=session_file,
                            workspace_path=project_path,
                            turn_number=turn,
                            session_type=session_type,
                            agent_id=agent_id if agent_id else None,
                        )
                        enqueued += 1
                    except Exception as e:
                        logger.warning(
                            f"Error enqueuing catch-up for {session_file.name} turn {turn}: {e}"
                        )

                if enqueued:
                    # Align baseline for future polling (Claude baseline excludes last turn).
                    self.last_stop_reason_counts[str(session_file)] = self._count_complete_turns(
                        session_file
                    )

        except Exception as e:
            logger.error(f"Catch-up error: {e}", exc_info=True)

    def _get_file_hash(self, session_file: Path) -> Optional[str]:
        """Compute MD5 hash of session file for duplicate detection."""
        try:
            with open(session_file, "rb") as f:
                md5_hash = hashlib.md5()
                while chunk := f.read(8192):
                    md5_hash.update(chunk)
                return md5_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to compute hash for {session_file.name}: {e}")
            return None

    def _extract_project_path(self, session_file: Path) -> Optional[Path]:
        """
        Extract project path (cwd) from session file.
        Delegates to the appropriate adapter.
        """
        try:
            # Use AdapterRegistry to find the right adapter
            from .adapters import get_adapter_registry

            registry = get_adapter_registry()

            adapter = registry.auto_detect_adapter(session_file)
            if adapter:
                project_path = adapter.extract_project_path(session_file)
                if project_path:
                    return project_path

            # Fallback for legacy logic if adapter returns None or no adapter found
            # (Keep existing logic as backup if needed, or rely on adapters)

            # Method 3: For Gemini CLI / Antigravity - return a pseudo path if adapter failed
            if ".gemini/" in str(session_file):
                logger.debug(
                    f"Gemini/Antigravity session detected, using home as pseudo project fallback: {session_file.name}"
                )
                return Path.home()

            logger.debug(f"Could not extract project path from {session_file.name}")
            return None

        except Exception as e:
            logger.debug(f"Error extracting project path from {session_file}: {e}")
            return None

    def _detect_session_type(self, session_file: Path) -> SessionType:
        """
        Detect the type of session file.
        """
        try:
            # Delegate to registry logic to ensure consistency
            from .adapters import get_adapter_registry

            registry = get_adapter_registry()
            adapter = registry.auto_detect_adapter(session_file)
            if adapter:
                # Map adapter name to SessionType
                # Adapter names: "claude", "codex", "gemini"
                name = adapter.name
                if name in ["claude", "codex", "gemini"]:
                    return name

            return "unknown"

        except Exception as e:
            print(
                f"[Watcher] Error detecting session type for {session_file.name}: {e}",
                file=sys.stderr,
            )
            return "unknown"

    def _get_trigger_for_session(self, session_file: Path):
        """
        获取或创建session的trigger

        Args:
            session_file: session文件路径

        Returns:
            TurnTrigger实例，如果session类型不支持则返回None
        """
        session_path = str(session_file)

        if session_path not in self._session_triggers:
            # Use registry to get adapter and trigger
            from .adapters import get_adapter_registry

            registry = get_adapter_registry()
            adapter = registry.auto_detect_adapter(session_file)

            if not adapter:
                logger.error(f"Unknown session type for {session_file.name}, cannot select trigger")
                return None

            self._session_triggers[session_path] = adapter.trigger

        return self._session_triggers[session_path]

    def _count_complete_turns(self, session_file: Path) -> int:
        """
        Unified interface to count complete dialogue turns for any session type.

        Returns:
            Number of complete dialogue turns (user request + assistant response)
        """
        trigger = self._get_trigger_for_session(session_file)
        if not trigger:
            return 0

        try:
            return trigger.count_complete_turns(session_file)
        except Exception as e:
            logger.error(f"Trigger error for {session_file.name}: {e}")
            return 0

    async def check_for_changes(
        self,
        current_sizes: Optional[Dict[str, int]] = None,
        current_mtimes: Optional[Dict[str, float]] = None,
    ):
        """Check if any session file has been modified."""
        try:
            if current_sizes is None or current_mtimes is None:
                current_sizes, current_mtimes = self._get_session_stats()

            # Detect changed files
            changed_files = []
            for path, size in current_sizes.items():
                old_size = self.last_session_sizes.get(path)
                old_mtime = self.last_session_mtimes.get(path)
                mtime = current_mtimes.get(path)

                # Consider any file modification as "changed":
                # - Claude Code can compact/rewrite sessions (size can shrink)
                # - Some writes replace content without growing the file
                if old_size is None or old_mtime is None:
                    changed_files.append(Path(path))
                    logger.debug(f"Session file first seen: {Path(path).name} ({size} bytes)")
                    # Best-effort: link newly discovered Codex sessions to an active Codex terminal.
                    try:
                        self._maybe_link_codex_terminal(Path(path))
                    except Exception:
                        pass
                    continue

                if size != old_size or (mtime is not None and mtime != old_mtime):
                    changed_files.append(Path(path))
                    logger.debug(
                        f"Session file changed: {Path(path).name} (size {old_size} -> {size} bytes)"
                    )

            if changed_files:
                # Accumulate changed files instead of cancelling pending task
                # This fixes the bug where continuous activity prevents commits
                for f in changed_files:
                    self._pending_changed_files.add(str(f))

                # Only create new task if no pending task or previous one completed
                if not self.pending_commit_task or self.pending_commit_task.done():
                    logger.info(
                        f"Scheduling commit check for {len(self._pending_changed_files)} session file(s)"
                    )
                    self.pending_commit_task = asyncio.create_task(
                        self._debounced_commit_accumulated()
                    )
                else:
                    logger.debug(
                        f"Accumulated {len(changed_files)} file(s), total pending: {len(self._pending_changed_files)}"
                    )

            # Update tracked sizes
            self.last_session_sizes = current_sizes
            # Update tracked mtimes for change detection
            self.last_session_mtimes = current_mtimes

        except Exception as e:
            logger.error(f"Error checking for changes: {e}", exc_info=True)
            print(f"[Watcher] Error checking for changes: {e}", file=sys.stderr)

    async def _debounced_commit_accumulated(self):
        """Wait for debounce period, then enqueue per-session processing jobs."""
        try:
            # Wait for debounce period
            await asyncio.sleep(self.debounce_delay)

            # Grab and clear the accumulated files atomically
            changed_files = [Path(p) for p in self._pending_changed_files]
            self._pending_changed_files.clear()

            if not changed_files:
                return

            logger.info(f"Enqueueing session_process for {len(changed_files)} session(s)")

            # Prefer processing the most recently modified sessions first
            try:
                changed_files.sort(
                    key=lambda it: it.stat().st_mtime if it.exists() else 0,
                    reverse=True,
                )
            except Exception:
                pass

            from .db import get_database

            db = get_database()

            yield_often = len(changed_files) > 1
            for idx, session_file in enumerate(changed_files, start=1):
                if not session_file.exists():
                    continue
                session_type = self._detect_session_type(session_file)
                if not self._is_session_trackable(
                    session_file,
                    session_type=session_type,
                    session_id=session_file.stem,
                ):
                    continue
                try:
                    db.enqueue_session_process_job(  # type: ignore[attr-defined]
                        session_file_path=session_file,
                        session_id=session_file.stem,
                        workspace_path=None,
                        session_type=session_type,
                        source_event="poll",
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to enqueue session_process for {session_file.name}: {e}"
                    )
                if yield_often:
                    await asyncio.sleep(0)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in debounced enqueue: {e}", exc_info=True)
            print(f"[Watcher] Error in debounced enqueue: {e}", file=sys.stderr)

    async def _debounced_commit(self, changed_files: list):
        """Wait for debounce period, then enqueue per-session processing jobs.

        DEPRECATED: Use _debounced_commit_accumulated instead.
        Kept for backwards compatibility.
        """
        try:
            # Wait for debounce period
            await asyncio.sleep(self.debounce_delay)

            from .db import get_database

            db = get_database()

            for session_file in changed_files:
                if not isinstance(session_file, Path):
                    session_file = Path(str(session_file))
                if not session_file.exists():
                    continue
                session_type = self._detect_session_type(session_file)
                if not self._is_session_trackable(
                    session_file,
                    session_type=session_type,
                    session_id=session_file.stem,
                ):
                    continue

                try:
                    db.enqueue_session_process_job(  # type: ignore[attr-defined]
                        session_file_path=session_file,
                        session_id=session_file.stem,
                        workspace_path=None,
                        session_type=session_type,
                        source_event="poll",
                    )
                except Exception:
                    continue

        except asyncio.CancelledError:
            # Task was cancelled because a newer change was detected
            pass
        except Exception as e:
            print(f"[Watcher] Error in debounced enqueue: {e}", file=sys.stderr)

    async def _check_if_turn_complete(self, session_file: Path) -> bool:
        """
        Check if the session file has at least 1 new complete dialogue turn since last check.

        Supports both Claude Code and Codex formats:
        - Claude Code: Count user messages by timestamp
        - Codex: Uses token_count events (no deduplication needed)

        Each complete dialogue round consists of:
        1. User message/request
        2. Assistant response
        3. Turn completion marker (format-specific)

        Note: This method does NOT update last_stop_reason_counts.
        The count will be updated in _do_commit() after successful commit.
        """
        try:
            return bool(self._get_new_completed_turn_numbers(session_file))

        except Exception as e:
            logger.error(f"Error checking turn completion: {e}", exc_info=True)
            print(f"[Watcher] Error checking turn completion: {e}", file=sys.stderr)
            return False

    async def _do_commit(
        self,
        project_path: Path,
        session_file: Path,
        target_turn: Optional[int] = None,
        turn_content: Optional[str] = None,
        user_message_override: Optional[str] = None,
        from_catchup: bool = False,
        quiet: bool = False,
        debug_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        skip_dedup: bool = False,
    ) -> bool:
        """
        Async wrapper for committing a turn to the shadow git repository.

        Args:
            project_path: Path to the project directory
            session_file: Session file that triggered the commit
            target_turn: If provided, commit this specific turn number (catch-up)
            turn_content: Optional pre-extracted turn content
            user_message_override: Optional pre-extracted user message
            from_catchup: If True, indicates catch-up mode
            quiet: If True, suppress console output
        """
        try:
            # Delegate to synchronous commit method (runs in executor to avoid blocking)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._run_realign_commit,
                project_path,
                session_file,
                target_turn,
                turn_content,
                user_message_override,
                quiet,  # Pass quiet parameter
                debug_callback,  # Pass debug_callback
                skip_dedup,  # Pass skip_dedup
            )

            if result:
                logger.info(f"✓ Committed to {project_path.name}")
                if not quiet:
                    print(f"[Watcher] ✓ Auto-committed to {project_path.name}", file=sys.stderr)
                # Update last commit time for this project
                self.last_commit_times[str(project_path)] = time.time()

                # Update turn count baseline ONLY after successful commit
                # This prevents double-counting if commit fails
                session_path = str(session_file)
                current_count = self._count_complete_turns(session_file)
                if target_turn:
                    current_count = max(current_count, target_turn)
                self.last_stop_reason_counts[session_path] = current_count
                logger.debug(
                    f"Updated turn count baseline for {session_file.name}: {current_count}"
                )
            else:
                logger.warning(f"Commit failed for {project_path.name}")

            return bool(result)

        except Exception as e:
            logger.error(f"Error during commit for {project_path}: {e}", exc_info=True)
            print(f"[Watcher] Error during commit for {project_path}: {e}", file=sys.stderr)
            return False

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

    def _find_latest_session(self, project_path: Path) -> Optional[Path]:
        """Find the most recently modified session file for this project."""
        try:
            session_files = find_all_active_sessions(self.config, project_path)
            if not session_files:
                return None

            # Return most recently modified session
            return max(session_files, key=lambda f: f.stat().st_mtime)
        except Exception as e:
            logger.error(f"Failed to find latest session: {e}")
            return None

    def _handle_session_redaction(
        self, session_file: Path, project_path: Path, quiet: bool = False
    ) -> Path:
        """Check and redact sensitive information from session file.

        Args:
            session_file: Path to the session file
            project_path: Path to the project directory
            quiet: If True, suppress console output

        Returns:
            Path to the (possibly modified) session file
        """
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
            # Return original session file on error
            return session_file

    def _get_current_turn_number(self, session_file: Path) -> int:
        """Get the current turn number from a session file."""
        # Count the number of complete turns in the session
        return self._count_complete_turns(session_file)

    def _extract_last_user_message(self, session_file: Path) -> str:
        """
        Extract the user message for the current turn being committed.

        This is called AFTER a new user message arrives (which triggers the commit),
        so we need to extract the SECOND-TO-LAST valid user message, not the last one.
        The last user message belongs to the next turn that hasn't been processed yet.
        """
        from .hooks import clean_user_message

        try:
            user_messages = []

            with open(session_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())

                        # Check for user message
                        if data.get("type") == "user":
                            message = data.get("message", {})
                            content = message.get("content", "")

                            extracted_text = None

                            if isinstance(content, str):
                                extracted_text = content
                            elif isinstance(content, list):
                                # Extract text from content blocks
                                text_parts = []
                                for item in content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        text_parts.append(item.get("text", ""))

                                # Only add if we found actual text content
                                # Skip entries that only contain tool_result items
                                if text_parts:
                                    extracted_text = "\n".join(text_parts)

                            if extracted_text:
                                # Clean the message (remove IDE tags, etc.)
                                cleaned_text = clean_user_message(extracted_text)

                                # Skip empty messages after cleaning
                                if not cleaned_text.strip():
                                    continue

                                # Skip continuation messages
                                if cleaned_text.startswith("This session is being continued"):
                                    continue

                                user_messages.append(cleaned_text)

                    except json.JSONDecodeError:
                        continue

            # Return second-to-last message if available, otherwise last message
            # This is because the commit is triggered by a new user message,
            # so the last message is for the NEXT turn, not the current one being committed
            if len(user_messages) >= 2:
                return user_messages[-2]
            elif len(user_messages) == 1:
                return user_messages[0]
            else:
                return "No user message found"

        except Exception as e:
            logger.error(f"Failed to extract user message: {e}")
            return "Error extracting message"

    def _set_early_session_title(
        self,
        session_file: Path,
        session_id: str,
        prompt: str,
        project_dir: str,
        no_track: bool = False,
    ) -> None:
        """Generate and store an early session title from the first user prompt."""
        try:
            if not self.config.enable_early_session_title:
                return
            if not session_file.exists():
                return
            if no_track:
                return

            from .db import get_database

            db = get_database()
            sid = session_id or session_file.stem

            # Guard: only set title once per session
            existing = db.get_session_by_id(sid)
            if existing and existing.session_title:
                return

            # Ensure session record exists
            file_stat = session_file.stat()
            file_created = datetime.fromtimestamp(
                getattr(file_stat, "st_birthtime", file_stat.st_ctime)
            )
            project_path = None
            if project_dir:
                try:
                    candidate = Path(project_dir)
                    if candidate.exists():
                        project_path = candidate
                except Exception:
                    project_path = None
            if project_path is None:
                project_path = self._extract_project_path(session_file)
            db.get_or_create_session(
                session_id=sid,
                session_file_path=session_file,
                session_type=self._detect_session_type(session_file),
                started_at=file_created,
                workspace_path=str(project_path) if project_path else None,
            )

            # Clean prompt
            from .hooks import clean_user_message

            cleaned = clean_user_message(prompt) if prompt else ""
            if not cleaned:
                return

            # Try LLM-generated title via cloud proxy
            title = None
            try:
                from .llm_client import call_llm_cloud

                model_name, result = call_llm_cloud(
                    task="early_session_title",
                    payload={"user_prompt": cleaned[:500]},
                    timeout=15.0,
                    silent=True,
                )
                if result and isinstance(result, dict):
                    title = result.get("title") or result.get("session_title")
            except Exception as e:
                logger.debug(f"LLM early session title failed, using fallback: {e}")

            # Fallback: first line of cleaned prompt, truncated to 80 chars
            if not title:
                first_line = cleaned.split("\n", 1)[0].strip()
                title = first_line[:80] if first_line else cleaned[:80]

            if title:
                db.update_session_summary(sid, title=title, summary="")
                logger.info(f"Set early session title for {sid}: {title[:50]}")
        except Exception as e:
            logger.debug(f"Failed to set early session title: {e}")

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

            # For non-JSONL formats (e.g., Gemini JSON), use extract_turn_info
            session_format = analysis.get("format", "")
            if session_format in ("gemini_json", "gemini"):
                turn_info = trigger.extract_turn_info(session_file, turn_number)
                if turn_info and turn_info.get("turn_content"):
                    return turn_info["turn_content"]
                # Fallback: construct content from group data
                return json.dumps(
                    {
                        "turn_number": turn_number,
                        "user_message": group.get("user_message", ""),
                        "assistant_response": group.get("summary_message", ""),
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            # For JSONL formats, extract by line numbers
            start_line = group.get("start_line") or (group.get("lines") or [None])[0]
            end_line = group.get("end_line") or (group.get("lines") or [None])[-1]
            if not start_line or not end_line:
                return ""

            lines = []
            with open(session_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, 1):
                    if start_line <= idx <= end_line:
                        lines.append(line)
                    if idx > end_line:
                        break
            return "".join(lines)
        except Exception as e:
            logger.error(f"Failed to extract turn content for turn {turn_number}: {e}")
            print(f"[Debug] Failed to extract turn content: {e}", file=sys.stderr)
            return ""

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

    def _find_latest_structured_summary(self, session_file: Path) -> Optional[str]:
        """
        Find the latest agent-authored summary block in the session.

        Claude Code emits dedicated summary records (`{\"type\":\"summary\",\"summary\":\"...\"}`)
        after each turn. We scan from the end to pick the most recent one, which keeps the
        summary aligned with the turn that just finished.
        """
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

    def _extract_current_turn_content(self, session_file: Path) -> str:
        """
        Extract only the content for the current turn being committed.

        Since commit is triggered by a new user message (Turn N+1), we need to extract
        the content from the PREVIOUS turn (Turn N), which includes:
        - The second-to-last user message
        - All assistant responses after that user message
        - But BEFORE the last user message (which belongs to Turn N+1)

        Returns:
            JSONL content for the current turn only
        """
        try:
            lines = []
            user_message_indices = []

            lines = []
            user_message_indices = []

            if session_file.is_dir():
                # For directory sessions, delegate to extract_turn_content_by_number (via trigger)
                # We don't support partial "current turn" extraction for Antigravity yet
                # as it treats the whole state as one turn.
                # Just return an empty string or the full content if needed.
                # But _extract_current_turn_content is usually used for diffing?
                # or extracting just the User Message to identify intent.
                trigger = self._get_trigger_for_session(session_file)
                if trigger:
                    # Get current turn number
                    turn = self._get_current_turn_number(session_file)
                    info = trigger.extract_turn_info(session_file, turn)
                    if info:
                        return info.user_message
                return ""

            # Read all lines and track user message positions
            with open(session_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    lines.append(line)
                    try:
                        data = json.loads(line.strip())
                        if data.get("type") == "user":
                            message = data.get("message", {})
                            content = message.get("content", "")

                            # Check if this is a real user message (not tool result, IDE notification, etc.)
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

            # Determine the range for current turn
            if len(user_message_indices) >= 2:
                # Extract from second-to-last user message up to (but not including) last user message
                start_idx = user_message_indices[-2]
                end_idx = user_message_indices[-1]
                turn_lines = lines[start_idx:end_idx]
            elif len(user_message_indices) == 1:
                # First turn: from first user message to end
                start_idx = user_message_indices[0]
                turn_lines = lines[start_idx:]
            else:
                # No valid user messages
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

    @staticmethod
    def _extract_latest_commit_title(context: str) -> Optional[str]:
        """
        Parse the most recent commit title from a textual recent-commit context block.
        """
        if not context:
            return None

        for line in context.splitlines():
            stripped = line.strip()
            if not stripped or stripped.lower().startswith("recent commits"):
                continue
            if stripped.startswith("-"):
                payload = stripped[1:].strip()
                if not payload:
                    continue
                parts = payload.split(" ", 1)
                if len(parts) == 2:
                    return parts[1].strip() or None
                return parts[0].strip() or None
        return None

    def _get_session_start_time(self, session_file: Path) -> Optional[float]:
        """
        Get the session start time from the first message timestamp.

        Returns:
            Unix timestamp (float) or None if not found
        """
        try:
            if session_file.is_dir():
                # For directories, just use creation time
                try:
                    stat = session_file.stat()
                    return getattr(stat, "st_birthtime", stat.st_ctime)
                except:
                    return session_file.stat().st_ctime

            with open(session_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())

                        # Look for timestamp field in various formats
                        timestamp_str = data.get("timestamp")
                        if timestamp_str:
                            # Parse ISO 8601 timestamp
                            from datetime import datetime

                            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                            return dt.timestamp()

                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue

            # Fallback: use session file's creation time
            return session_file.stat().st_ctime

        except Exception as e:
            logger.error(f"Failed to get session start time: {e}")
            return None

    async def auto_init_projects(self):
        """
        Ensure global Aline config and database exist.
        """
        try:
            if is_aline_initialized():
                return

            from .commands.init import init_global

            result = await asyncio.get_event_loop().run_in_executor(None, init_global, False)
            if result.get("success"):
                logger.info("✓ Global Aline initialization ready")
            else:
                logger.error(f"✗ Global init failed: {result.get('message')}")

        except Exception as e:
            logger.error(f"Error in auto_init_projects: {e}", exc_info=True)
