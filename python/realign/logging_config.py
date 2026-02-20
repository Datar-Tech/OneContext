"""
ReAlign Logging Configuration

Provides centralized logging setup for the ReAlign project.
Supports file rotation, environment variable configuration, and structured logging.
"""

import logging
import os
import tempfile
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


_LOG_DIR_CACHE: tuple[str | None, Path] | None = None
PRIMARY_LOG_FILENAME = "dashboard.log"


def get_log_level() -> int:
    """
    Get log level from environment variable or default to INFO.

    Environment variable: REALIGN_LOG_LEVEL
    Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL

    Returns:
        int: Logging level constant from logging module
    """
    level_name = os.getenv("REALIGN_LOG_LEVEL", "INFO").upper()

    # Map string to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return level_map.get(level_name, logging.INFO)


def get_log_directory() -> Path:
    """
    Get the log directory path.

    Default: ~/.aline/.logs/
    Can be overridden with REALIGN_LOG_DIR environment variable.

    Returns:
        Path: Log directory path
    """
    global _LOG_DIR_CACHE

    log_dir_str = os.getenv("REALIGN_LOG_DIR")
    if _LOG_DIR_CACHE is not None and _LOG_DIR_CACHE[0] == log_dir_str:
        return _LOG_DIR_CACHE[1]

    candidates: list[Path] = []
    if log_dir_str:
        candidates.append(Path(log_dir_str).expanduser())
    else:
        candidates.append(Path.home() / ".aline" / ".logs")

    # Fallback for restricted environments (e.g., sandboxed runners).
    candidates.append(Path(tempfile.gettempdir()) / "aline-logs")

    def can_write_files(directory: Path) -> bool:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / f".write_probe_{os.getpid()}_{os.urandom(4).hex()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            if can_write_files(candidate):
                _LOG_DIR_CACHE = (log_dir_str, candidate)
                return candidate
        except Exception as e:
            last_error = e
            continue

    # Last resort: current working directory (best-effort, avoids crashing).
    try:
        cwd = Path.cwd() / ".aline-logs"
        if can_write_files(cwd):
            _LOG_DIR_CACHE = (log_dir_str, cwd)
            return cwd
        return cwd
    except Exception:
        # If even this fails, return the first candidate to keep callers deterministic.
        # The logger will fall back to stderr-only.
        if last_error:
            chosen = candidates[0]
        else:
            chosen = Path.cwd()
        _LOG_DIR_CACHE = (log_dir_str, chosen)
        return chosen


def get_primary_log_path() -> Path:
    """Get canonical log file path (~/.aline/.logs/dashboard.log by default)."""
    return get_log_directory() / PRIMARY_LOG_FILENAME


def is_primary_log_file(path: Path) -> bool:
    """Whether a path is the canonical dashboard log file."""
    return path.name == PRIMARY_LOG_FILENAME


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = False,
) -> logging.Logger:
    """
    Set up a logger with file rotation and optional console output.

    Args:
        name: Logger name (e.g., 'realign.hooks', 'realign.redactor')
        log_file: Log filename hint. Only `dashboard.log` is persisted to file.
        max_bytes: Maximum size of each log file before rotation (default: 10MB)
        backup_count: Legacy parameter. Backups are disabled to keep a single dashboard.log.
        console_output: Whether to also output to console/stderr (default: False)

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> logger = setup_logger('realign.hooks', 'hooks.log')
        >>> logger.info("Hook started")
        >>> logger.debug("Processing file: %s", file_path)
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Keep logger level in sync with env var, but avoid duplicating handlers.
        logger.setLevel(get_log_level())
        return logger

    logger.setLevel(get_log_level())
    logger.propagate = False  # Don't propagate to root logger

    # Standard formatter with timestamp, level, name, and message
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler is allowed only for the canonical dashboard log.
    if log_file and log_file == PRIMARY_LOG_FILENAME:
        try:
            log_path = get_primary_log_path()

            file_handler = RotatingFileHandler(
                log_path, maxBytes=max_bytes, backupCount=0, encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)  # Capture all levels to file
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        except Exception as e:
            # If file logging fails, fall back to stderr only
            # Don't let logging setup break the application
            import sys

            print(f"Warning: Failed to set up file logging: {e}", file=sys.stderr)

    # Optional console handler (stderr)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(get_log_level())
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.

    This is a convenience function for getting a logger that was already
    set up with setup_logger().

    Args:
        name: Logger name

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


# Context manager for temporary log level changes
class temporary_log_level:
    """
    Context manager to temporarily change log level.

    Example:
        >>> with temporary_log_level('realign.hooks', logging.DEBUG):
        ...     # Code that needs debug logging
        ...     process_session()
    """

    def __init__(self, logger_name: str, level: int):
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
        self.new_level = level

    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)
        return False
