"""Configuration management for ReAlign."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ReAlignConfig:
    """ReAlign configuration."""

    summary_max_chars: int = 500
    redact_on_match: bool = False  # Default: disable redaction (can be enabled in config)
    hooks_installation: str = "repo"
    sqlite_db_path: str = "~/.aline/db/aline.db"  # Path to SQLite database
    use_LLM: bool = True
    llm_provider: str = "auto"  # LLM provider: "auto", "claude", or "openai"
    auto_detect_claude: bool = True  # Enable Claude Code session auto-detection
    auto_detect_codex: bool = True  # Enable Codex session auto-detection
    auto_detect_gemini: bool = (
        False  # Enable Gemini CLI session auto-detection (disabled by default)
    )
    dashboard_only: bool = True  # Track Claude/Codex only when launched from Aline dashboard
    mcp_auto_commit: bool = True  # Enable watcher auto-commit after each user request completes
    enable_early_session_title: bool = True  # Generate early session title on first user prompt
    share_backend_url: str = (
        "https://realign-server.vercel.app"  # Backend URL for interactive share export
    )

    # User identity (V9, renamed in V17: user_id -> uid)
    user_name: str = ""  # User's display name (set during init or login)
    uid: str = ""  # User's UUID (from Supabase login)

    # Session catch-up settings
    max_catchup_sessions: int = 3  # Max sessions to auto-import on watcher startup

    # Local API server port (for one-click browser import)
    local_api_port: int = 17280

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "ReAlignConfig":
        """Load configuration from file with environment variable overrides."""
        if config_path is None:
            # Default to new location: ~/.aline/config.yaml
            config_path = Path.home() / ".aline" / "config.yaml"

            # Check for legacy config and migrate if needed
            legacy_path = Path.home() / ".config" / "realign" / "config.yaml"
            if not config_path.exists() and legacy_path.exists():
                try:
                    import shutil

                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(legacy_path, config_path)
                    # Try to remove empty legacy directory
                    try:
                        legacy_path.parent.rmdir()
                    except OSError:
                        pass
                except Exception:
                    # If migration fails, fall back to reading legacy file
                    config_path = legacy_path

        config_dict = {}

        # Load from file if exists
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}

        # Apply environment variable overrides
        env_overrides = {
            "summary_max_chars": os.getenv("REALIGN_SUMMARY_MAX_CHARS"),
            "redact_on_match": os.getenv("REALIGN_REDACT_ON_MATCH"),
            "hooks_installation": os.getenv("REALIGN_HOOKS_INSTALLATION"),
            "sqlite_db_path": os.getenv("REALIGN_SQLITE_DB_PATH"),
            "use_LLM": os.getenv("REALIGN_USE_LLM"),
            "llm_provider": os.getenv("REALIGN_LLM_PROVIDER"),
            "auto_detect_claude": os.getenv("REALIGN_AUTO_DETECT_CLAUDE"),
            "auto_detect_codex": os.getenv("REALIGN_AUTO_DETECT_CODEX"),
            "auto_detect_gemini": os.getenv("REALIGN_AUTO_DETECT_GEMINI"),
            "dashboard_only": os.getenv("REALIGN_DASHBOARD_ONLY"),
            "mcp_auto_commit": os.getenv("REALIGN_MCP_AUTO_COMMIT"),
            "enable_early_session_title": os.getenv("REALIGN_ENABLE_EARLY_SESSION_TITLE")
            or os.getenv("REALIGN_ENABLE_TEMP_TURN_TITLES"),
            "share_backend_url": os.getenv("REALIGN_SHARE_BACKEND_URL"),
            "user_name": os.getenv("REALIGN_USER_NAME"),
            "uid": os.getenv("REALIGN_UID"),
            "max_catchup_sessions": os.getenv("REALIGN_MAX_CATCHUP_SESSIONS"),
            "local_api_port": os.getenv("ALINE_LOCAL_API_PORT"),
        }

        for key, value in env_overrides.items():
            if value is not None:
                if key in ["summary_max_chars", "max_catchup_sessions", "local_api_port"]:
                    config_dict[key] = int(value)
                elif key in [
                    "redact_on_match",
                    "use_LLM",
                    "auto_detect_claude",
                    "auto_detect_codex",
                    "auto_detect_gemini",
                    "dashboard_only",
                    "mcp_auto_commit",
                    "enable_early_session_title",
                ]:
                    config_dict[key] = value.lower() in ("true", "1", "yes")
                else:
                    config_dict[key] = value

        # Migration: user_id -> uid (V17)
        if "user_id" in config_dict and "uid" not in config_dict:
            config_dict["uid"] = config_dict.pop("user_id")
        elif "user_id" in config_dict:
            # Both exist, prefer uid, discard user_id
            config_dict.pop("user_id")

        # Migration: enable_temp_turn_titles -> enable_early_session_title
        if (
            "enable_temp_turn_titles" in config_dict
            and "enable_early_session_title" not in config_dict
        ):
            config_dict["enable_early_session_title"] = config_dict.pop("enable_temp_turn_titles")
        elif "enable_temp_turn_titles" in config_dict:
            config_dict.pop("enable_temp_turn_titles")

        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    def save(self, config_path: Optional[Path] = None):
        """Save configuration to file."""
        if config_path is None:
            config_path = Path.home() / ".aline" / "config.yaml"

        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "summary_max_chars": self.summary_max_chars,
            "redact_on_match": self.redact_on_match,
            "hooks_installation": self.hooks_installation,
            "sqlite_db_path": self.sqlite_db_path,
            "use_LLM": self.use_LLM,
            "llm_provider": self.llm_provider,
            "auto_detect_claude": self.auto_detect_claude,
            "auto_detect_codex": self.auto_detect_codex,
            "auto_detect_gemini": self.auto_detect_gemini,
            "dashboard_only": self.dashboard_only,
            "mcp_auto_commit": self.mcp_auto_commit,
            "enable_early_session_title": self.enable_early_session_title,
            "share_backend_url": self.share_backend_url,
            "user_name": self.user_name,
            "uid": self.uid,
            "max_catchup_sessions": self.max_catchup_sessions,
            "local_api_port": self.local_api_port,
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


def generate_random_username() -> str:
    """
    Generate a random username with format: 3 lowercase letters + 3 digits.

    Example: abc123, xyz789

    Returns:
        str: Random username
    """
    import random
    import string

    letters = "".join(random.choices(string.ascii_lowercase, k=3))
    digits = "".join(random.choices(string.digits, k=3))
    return letters + digits


def get_default_config_content() -> str:
    """Get default configuration file content."""
    return """# ReAlign Global Configuration (User Home Directory)
summary_max_chars: 500           # Maximum length of commit message summaries
redact_on_match: false           # Automatically redact sensitive information (disabled by default)
                                 # Original sessions are backed up to .realign/sessions-original/
                                 # Set to true to enable if you plan to share sessions publicly
hooks_installation: "repo"       # Repo mode: sets core.hooksPath=.realign/hooks
sqlite_db_path: "~/.aline/db/aline.db" # Path to SQLite database file
use_LLM: true                    # Whether to use a cloud LLM to generate summaries
llm_provider: "auto"             # LLM provider: "auto" (try Claude then OpenAI), "claude", or "openai"
auto_detect_claude: true         # Automatically detect Claude Code session directory (~/.claude/projects/)
auto_detect_codex: true          # Automatically detect Codex session files (~/.codex/sessions/)
auto_detect_gemini: false        # Gemini auto-detect is disabled by default
dashboard_only: true             # Track Claude/Codex sessions only when launched from Aline dashboard
mcp_auto_commit: true            # Enable watcher to auto-commit after each user request completes
enable_early_session_title: true  # Generate early session title on first user prompt
share_backend_url: "https://realign-server.vercel.app"  # Backend URL for interactive share export
                                 # For local development, use: "http://localhost:3000"

# Session Catch-up Settings:
max_catchup_sessions: 3                # Max sessions to auto-import on watcher startup
                                       # Use 'aline watcher session list' to see all sessions
                                       # Import specific sessions from OneContext Dashboard -> Config -> Import history

# Secret Detection & Redaction:
# ReAlign can use detect-secrets to automatically scan for and redact:
# - API keys, tokens, passwords
# - Private keys, certificates
# - AWS credentials, database URLs
# Note: High-entropy strings (like Base64) are filtered out to reduce false positives
# To enable redaction: realign config set redact_on_match true
"""
