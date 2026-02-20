#!/usr/bin/env python3
"""Standalone worker daemon for ReAlign.

This daemon consumes durable jobs from the SQLite jobs queue:
- turn_summary jobs (LLM-powered turn commits)
- session_summary jobs (aggregated session title/summary)
"""

import asyncio
import sys
from pathlib import Path
import signal
import atexit
import os

try:
    from .db.sqlite_db import SQLiteDatabase
    from .logging_config import setup_logger
    from .config import ReAlignConfig
    from .worker_core import AlineWorker
except ImportError:
    from realign.db.sqlite_db import SQLiteDatabase
    from realign.logging_config import setup_logger
    from realign.config import ReAlignConfig
    from realign.worker_core import AlineWorker


logger = setup_logger("realign.worker.daemon", "worker_daemon.log")


def get_pid_file() -> Path:
    return Path.home() / ".aline/.logs/worker.pid"


def write_pid() -> None:
    pid_file = get_pid_file()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))


def remove_pid() -> None:
    try:
        get_pid_file().unlink(missing_ok=True)
    except Exception as e:
        logger.error(f"Failed to remove PID file: {e}")


async def run_daemon() -> None:
    worker = None
    db = None

    # Check login status before starting
    try:
        from .auth import is_logged_in
    except ImportError:
        from realign.auth import is_logged_in

    if not is_logged_in():
        logger.error("Not logged in. Worker daemon requires authentication.")
        print("[Worker Daemon] Error: Not logged in. Run 'aline login' first.", file=sys.stderr)
        sys.exit(1)

    def handle_shutdown(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        try:
            if worker is not None:
                try:
                    worker.running = False
                except Exception:
                    pass
                try:
                    if db is not None:
                        n = db.requeue_processing_jobs_locked_by(worker_id=worker.worker_id)
                        if n:
                            logger.info(
                                f"Requeued {n} in-flight job(s) for worker_id={worker.worker_id}"
                            )
                except Exception as e:
                    logger.warning(f"Failed to requeue in-flight jobs on shutdown: {e}")
        except Exception:
            pass
        remove_pid()
        sys.exit(0)

    try:
        write_pid()
        atexit.register(remove_pid)
        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)

        logger.info("Starting standalone worker daemon")
        print("[Worker Daemon] Starting standalone worker", file=sys.stderr)

        config = ReAlignConfig.load()
        db = SQLiteDatabase(config.sqlite_db_path)
        db.initialize()

        worker = AlineWorker(db)
        await worker.start()
    except Exception as e:
        logger.error(f"Daemon error: {e}", exc_info=True)
        print(f"[Worker Daemon] Error: {e}", file=sys.stderr)
        remove_pid()
        sys.exit(1)
    finally:
        if worker:
            try:
                await worker.stop()
            except Exception:
                pass
        remove_pid()


def main() -> None:
    try:
        asyncio.run(run_daemon())
    except KeyboardInterrupt:
        logger.info("Daemon interrupted by user")
        remove_pid()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        remove_pid()
        sys.exit(1)


if __name__ == "__main__":
    main()
