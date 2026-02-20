#!/usr/bin/env python3
"""Standalone watcher daemon for ReAlign.

This daemon runs the DialogueWatcher in the background.
It monitors session files and auto-commits changes to .realign/.git.
"""

import asyncio
import sys
from pathlib import Path
import signal
import atexit
import os

# Handle both relative and absolute imports for script and module execution
try:
    from .watcher_core import DialogueWatcher
    from .logging_config import setup_logger
except ImportError:
    from realign.watcher_core import DialogueWatcher
    from realign.logging_config import setup_logger

# Initialize logger
logger = setup_logger("realign.daemon", "watcher_daemon.log")


def get_pid_file() -> Path:
    """Get the path to the PID file."""
    return Path.home() / ".aline/.logs/watcher.pid"


def write_pid():
    """Write current process PID to file."""
    pid_file = get_pid_file()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))


def remove_pid():
    """Remove PID file."""
    try:
        get_pid_file().unlink(missing_ok=True)
    except Exception as e:
        logger.error(f"Failed to remove PID file: {e}")


async def run_daemon():
    """Run the watcher daemon."""
    watcher = None

    # Check login status before starting
    try:
        from .auth import is_logged_in
    except ImportError:
        from realign.auth import is_logged_in

    if not is_logged_in():
        logger.error("Not logged in. Watcher daemon requires authentication.")
        print("[Watcher Daemon] Error: Not logged in. Run 'aline login' first.", file=sys.stderr)
        sys.exit(1)

    # Shutdown handler
    def handle_shutdown(signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        remove_pid()
        sys.exit(0)

    try:
        # Write PID file
        write_pid()

        # Register cleanup
        atexit.register(remove_pid)

        # Setup signal handlers
        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)

        logger.info("Starting standalone watcher daemon")
        print("[Watcher Daemon] Starting standalone watcher", file=sys.stderr)

        # Create and start watcher
        watcher = DialogueWatcher()
        await watcher.start()

    except Exception as e:
        logger.error(f"Daemon error: {e}", exc_info=True)
        print(f"[Watcher Daemon] Error: {e}", file=sys.stderr)
        remove_pid()
        sys.exit(1)
    finally:
        # Clean shutdown
        if watcher and watcher.running:
            await watcher.stop()
        remove_pid()


def main():
    """Main entry point for the daemon."""
    try:
        # Run the async daemon
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
