"""Dashboard diagnostics and high-signal logging.

This module is intentionally best-effort: diagnostics should never crash the dashboard.
It provides a JSONL log (one JSON object per line) with correlation IDs and state
snapshots for debugging intermittent UI issues (blank panes, stuck refreshes, etc.).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":"))


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _selected_env() -> dict[str, str]:
    allow_prefixes = ("ALINE_", "REALIGN_")
    allow_keys = {
        "TERM",
        "TERM_PROGRAM",
        "TERM_PROGRAM_VERSION",
        "TMUX",
        "SHELL",
        "LANG",
        "LC_ALL",
    }
    out: dict[str, str] = {}
    for k, v in os.environ.items():
        if any(k.startswith(p) for p in allow_prefixes) or k in allow_keys:
            out[k] = str(v)
    return out


_DIAG_SINGLETON: "DashboardDiagnostics | None" = None


def reset_dashboard_diagnostics_for_tests() -> None:
    """Reset the diagnostics singleton/handlers (intended for pytest)."""
    global _DIAG_SINGLETON
    _DIAG_SINGLETON = None
    logger = logging.getLogger("realign.dashboard.diagnostics")
    for h in list(logger.handlers):
        try:
            logger.removeHandler(h)
        except Exception:
            pass
        try:
            h.close()
        except Exception:
            pass


@dataclass(frozen=True)
class DashboardDiagnostics:
    session_id: str
    path: Path | None
    logger: logging.Logger

    @classmethod
    def start(cls) -> "DashboardDiagnostics":
        """Create a per-run diagnostics logger (never raises)."""
        global _DIAG_SINGLETON
        if _DIAG_SINGLETON is not None:
            return _DIAG_SINGLETON

        session_id = uuid.uuid4().hex[:12]

        logger = logging.getLogger("realign.dashboard.diagnostics")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # Avoid duplicating handlers across hot reloads / repeated imports.
        if logger.handlers:
            diag = cls(session_id=session_id, path=None, logger=logger)
            _DIAG_SINGLETON = diag
            return diag

        path: Path | None = None
        handler: logging.Handler | None = None

        try:
            diag_dir = Path(tempfile.gettempdir()) / "aline-dashboard-diagnostics"
            diag_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = diag_dir / f"dashboard_{stamp}_pid{os.getpid()}_{session_id}.jsonl"
            handler = logging.FileHandler(path, encoding="utf-8")
        except Exception:
            # Fall back to stderr; still better than losing all diagnostics.
            handler = logging.StreamHandler()
            path = None

        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(fmt="%(message)s"))
        logger.addHandler(handler)

        diag = cls(session_id=session_id, path=path, logger=logger)
        _DIAG_SINGLETON = diag
        diag.event(
            "dashboard_diagnostics_started",
            pid=os.getpid(),
            python=sys.version.split()[0],
            platform=platform.platform(),
            env=_selected_env(),
        )
        return diag

    def event(self, name: str, **fields: Any) -> None:
        payload = {
            "ts": _now_iso(),
            "t_monotonic": round(time.monotonic(), 6),
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
            "session_id": self.session_id,
            "event": name,
            **fields,
        }
        try:
            self.logger.info(_safe_json(payload))
        except Exception:
            # Diagnostics must never crash the app.
            return

    def exception(self, name: str, exc: BaseException, **fields: Any) -> None:
        try:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        except Exception:
            tb = ""
        self.event(
            name,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=tb,
            **fields,
        )

    def snapshot(self, *, reason: str, app: Any | None = None, **fields: Any) -> None:  # noqa: ANN401
        """Capture a lightweight state snapshot (best-effort)."""
        snap: dict[str, Any] = {
            "reason": reason,
            "fields": fields,
        }

        if app is not None:
            snap["app"] = _snapshot_textual_app(app)

        try:
            from . import tmux_manager

            snap["tmux"] = tmux_manager.collect_tmux_debug_state()
        except Exception:
            pass

        self.event("dashboard_snapshot", **snap)

    def install_global_exception_hooks(self) -> None:
        """Install sys/threading exception hooks (best-effort)."""
        try:
            orig_sys_hook = sys.excepthook

            def sys_hook(exc_type, exc, tb):  # type: ignore[no-untyped-def]
                try:
                    self.exception(
                        "unhandled_exception",
                        exc if isinstance(exc, BaseException) else Exception(str(exc)),
                        where="sys.excepthook",
                    )
                except Exception:
                    pass
                try:
                    orig_sys_hook(exc_type, exc, tb)
                except Exception:
                    pass

            sys.excepthook = sys_hook
        except Exception:
            pass

        try:
            orig_thread_hook = threading.excepthook

            def thread_hook(args):  # type: ignore[no-untyped-def]
                try:
                    self.exception(
                        "unhandled_exception",
                        args.exc_value,
                        where="threading.excepthook",
                        thread=str(getattr(args, "thread", None)),
                    )
                except Exception:
                    pass
                try:
                    orig_thread_hook(args)
                except Exception:
                    pass

            threading.excepthook = thread_hook
        except Exception:
            pass

    def install_asyncio_exception_handler(self, loop: asyncio.AbstractEventLoop) -> None:
        """Install an asyncio loop exception handler (best-effort)."""

        def handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
            exc = context.get("exception")
            if isinstance(exc, BaseException):
                self.exception(
                    "asyncio_exception",
                    exc,
                    message=str(context.get("message") or ""),
                    context={k: v for k, v in context.items() if k not in {"exception"}},
                )
                return
            self.event(
                "asyncio_exception",
                message=str(context.get("message") or ""),
                context={k: v for k, v in context.items()},
            )

        try:
            loop.set_exception_handler(handler)
        except Exception:
            return


def _snapshot_textual_app(app: Any) -> dict[str, Any]:  # noqa: ANN401
    snap: dict[str, Any] = {}

    try:
        snap["title"] = str(getattr(app, "TITLE", "") or "")
        snap["screen"] = type(getattr(app, "screen", None)).__name__
        snap["dark"] = bool(getattr(app, "dark", False))
        snap["theme"] = str(getattr(app, "theme", "") or "")
    except Exception:
        pass

    # Best-effort widget checks for known failure modes.
    try:
        from textual.widgets import TabbedContent  # imported lazily

        tabbed = app.query_one(TabbedContent)
        snap["active_tab"] = str(getattr(tabbed, "active", "") or "")
    except Exception:
        pass

    try:
        from textual.containers import Vertical

        agents_list = app.query_one("#agents-list", Vertical)
        snap["agents_list_children"] = int(len(getattr(agents_list, "children", []) or []))
    except Exception:
        pass

    return snap
