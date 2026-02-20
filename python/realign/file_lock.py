"""File-based locking mechanism for cross-process synchronization."""

import fcntl
import os
import time
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


class FileLock:
    """Simple file-based lock using fcntl (Unix/macOS only)."""

    def __init__(self, lock_file: Path, timeout: float = 10.0, *, inheritable: bool = False):
        """
        Initialize a file lock.

        Args:
            lock_file: Path to the lock file
            timeout: Maximum time to wait for lock acquisition (seconds)
            inheritable: If True, keep FD inheritable across exec() (needed for os.exec* handoffs)
        """
        self.lock_file = lock_file
        self.timeout = timeout
        self.inheritable = bool(inheritable)
        self.fd: Optional[int] = None

    def acquire(self, blocking: bool = True) -> bool:
        """
        Acquire the lock.

        Args:
            blocking: If True, wait for lock; if False, return immediately

        Returns:
            True if lock was acquired, False otherwise
        """
        # Create lock file directory if needed
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        # Open lock file
        self.fd = os.open(str(self.lock_file), os.O_CREAT | os.O_RDWR)
        try:
            os.set_inheritable(self.fd, self.inheritable)
        except Exception:
            # Best-effort: on some platforms/filesystems this may fail; lock still works.
            pass

        if blocking:
            # Try to acquire with timeout
            start_time = time.time()
            while True:
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return True
                except BlockingIOError:
                    if time.time() - start_time > self.timeout:
                        os.close(self.fd)
                        self.fd = None
                        return False
                    time.sleep(0.1)
        else:
            # Non-blocking attempt
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                os.close(self.fd)
                self.fd = None
                return False

    def release(self):
        """Release the lock."""
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                os.close(self.fd)
            except Exception:
                pass
            finally:
                self.fd = None

    def __enter__(self):
        """Context manager entry."""
        if not self.acquire():
            raise TimeoutError(f"Could not acquire lock on {self.lock_file} within {self.timeout}s")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False

    def __del__(self):
        """Cleanup on deletion."""
        self.release()


@contextmanager
def commit_lock(repo_path: Path, timeout: float = 10.0):
    """
    Context manager for acquiring a commit lock.

    Prevents multiple watchers from committing simultaneously.

    Usage:
        with commit_lock(repo_path):
            # Perform git commit
            subprocess.run(["git", "commit", ...])

    Args:
        repo_path: Path to the repository
        timeout: Maximum time to wait for lock (seconds)

    Yields:
        True if lock was acquired
    """
    from realign import get_realign_dir

    realign_dir = get_realign_dir(repo_path)
    lock_file = realign_dir / ".commit.lock"
    lock = FileLock(lock_file, timeout=timeout)

    try:
        if lock.acquire():
            yield True
        else:
            yield False
    finally:
        lock.release()
