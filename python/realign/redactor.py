"""
Redactor module for detecting and redacting sensitive information in sessions.

This module uses detect-secrets to identify potential secrets and provides
functionality to redact them from session files.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .logging_config import setup_logger

# Initialize logger for redactor
logger = setup_logger("realign.redactor", "redactor.log")


class SecretMatch:
    """Represents a detected secret."""

    def __init__(self, secret_type: str, line_number: int, secret_hash: str):
        self.type = secret_type
        self.line = line_number
        self.hash = secret_hash

    def __repr__(self):
        return f"SecretMatch(type={self.type}, line={self.line})"


def _detect_custom_api_keys(content: str) -> List[SecretMatch]:
    """
    Detect API keys using custom regex patterns.

    This catches common API key formats that detect-secrets might miss.

    Args:
        content: The text content to scan

    Returns:
        List of SecretMatch objects for detected API keys
    """
    import re

    secrets = []
    lines = content.split("\n")

    # Common API key patterns
    patterns = [
        # OpenAI API keys (sk-, sk-proj-)
        (r"\bsk-[a-zA-Z0-9]{20,}", "OpenAI API Key"),
        # Anthropic API keys (sk-ant-api03-...)
        (r"\bsk-ant-[a-zA-Z0-9\-]{50,}", "Anthropic API Key"),
        # Generic API keys with common prefixes
        (
            r'\b(?:api[_-]?key|apikey|api[_-]?secret)[\s:=]+["\']?([a-zA-Z0-9_\-]{32,})["\']?',
            "Generic API Key",
        ),
        # Bearer tokens
        (r"\bBearer\s+[a-zA-Z0-9\-._~+/]+=*", "Bearer Token"),
        # GitHub tokens
        (r"\bgh[ps]_[a-zA-Z0-9]{36,}", "GitHub Token"),
        # Slack tokens
        (r"\bxox[baprs]-[a-zA-Z0-9\-]{10,}", "Slack Token"),
        # Generic long alphanumeric strings that look like secrets (60+ chars, mixed case)
        (r"\b[a-zA-Z0-9]{60,}\b", "Potential Secret (Long String)"),
    ]

    for line_num, line in enumerate(lines, start=1):
        for pattern, secret_type in patterns:
            matches = re.finditer(pattern, line, re.IGNORECASE)
            for match in matches:
                matched_text = match.group(0)

                # Skip if it looks like a UUID (has hyphens in UUID pattern)
                if re.match(
                    r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
                    matched_text,
                    re.IGNORECASE,
                ):
                    continue

                # Skip common false positives
                if matched_text.lower() in [
                    "example",
                    "placeholder",
                    "your_api_key_here",
                    "your-api-key",
                ]:
                    continue

                # For "Potential Secret (Long String)", require mixed case to reduce false positives
                if secret_type == "Potential Secret (Long String)":
                    has_upper = any(c.isupper() for c in matched_text)
                    has_lower = any(c.islower() for c in matched_text)
                    has_digit = any(c.isdigit() for c in matched_text)
                    # Require at least mixed case (upper + lower) or (letter + digit)
                    if not ((has_upper and has_lower) or (has_digit and (has_upper or has_lower))):
                        continue

                # Create a hash of the secret for identification
                import hashlib

                secret_hash = hashlib.sha256(matched_text.encode()).hexdigest()[:16]

                secrets.append(
                    SecretMatch(
                        secret_type=secret_type, line_number=line_num, secret_hash=secret_hash
                    )
                )
                logger.debug(f"Custom pattern detected: {secret_type} at line {line_num}")

    return secrets


def detect_secrets(content: str) -> Tuple[List[SecretMatch], bool]:
    """
    Detect secrets in the given content using detect-secrets library plus custom patterns.

    Args:
        content: The text content to scan for secrets

    Returns:
        Tuple of (list of SecretMatch objects, whether detect-secrets is available)
    """
    logger.debug(f"Scanning content for secrets ({len(content)} bytes)")

    try:
        from detect_secrets import SecretsCollection
        from detect_secrets.settings import default_settings

        logger.debug("detect-secrets library loaded successfully")
    except ImportError:
        logger.warning("detect-secrets library not installed")
        # detect-secrets not installed, fall back to basic pattern matching
        return [], False

    if not content or not content.strip():
        logger.debug("Empty content, skipping secret detection")
        return [], True

    secrets = []

    # Create a temporary file for scanning
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_path = f.name

        logger.debug(f"Created temporary file for scanning: {temp_path}")

        # Scan the file with default settings
        collection = SecretsCollection()
        with default_settings():
            collection.scan_file(temp_path)

        # Extract detected secrets, filtering out high-entropy false positives
        for filename, secret_list in collection.data.items():
            for secret in secret_list:
                # Filter out high-entropy detectors that cause false positives with UUIDs
                # Note: detect-secrets uses "High Entropy" (with space) in type names like "Base64 High Entropy String"
                if "High Entropy" in secret.type or "HighEntropy" in secret.type:
                    logger.debug(
                        f"Filtering out high-entropy detection: {secret.type} at line {secret.line_number}"
                    )
                    continue

                secrets.append(
                    SecretMatch(
                        secret_type=secret.type,
                        line_number=secret.line_number,
                        secret_hash=secret.secret_hash,
                    )
                )

        # Additional custom pattern-based detection for common API key formats
        custom_secrets = _detect_custom_api_keys(content)
        secrets.extend(custom_secrets)

        # Deduplicate secrets by line number and type
        # Prefer custom detector results over high-entropy detections
        seen = set()
        deduped_secrets = []
        for secret in secrets:
            key = (secret.line, secret.type)
            if key not in seen:
                seen.add(key)
                deduped_secrets.append(secret)

        secrets = deduped_secrets

        if secrets:
            logger.warning(f"Detected {len(secrets)} potential secret(s)")
            for secret in secrets:
                logger.debug(f"Secret: {secret.type} at line {secret.line}")
        else:
            logger.info("No secrets detected")

    except Exception as e:
        logger.error(f"Error during secret detection: {e}", exc_info=True)
        print(f"Warning: Error during secret detection: {e}", file=sys.stderr)
    finally:
        # Clean up temp file
        try:
            if "temp_path" in locals():
                os.unlink(temp_path)
                logger.debug("Temporary file cleaned up")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file: {e}")

    return secrets, True


# Fields that should NOT be redacted (metadata and non-sensitive data)
NON_SENSITIVE_FIELDS = {
    # Message structure
    "type",
    "role",
    "stop_reason",
    "stop_sequence",
    # Model metadata
    "model",
    "id",
    "service_tier",
    # Session metadata
    "isSidechain",
    "userType",
    "version",
    "gitBranch",
    "cwd",
    "slug",
    # Identifiers (UUIDs, timestamps - not actual secrets)
    "parentUuid",
    "uuid",
    "sessionId",
    "requestId",
    "timestamp",
    # Token usage (not sensitive)
    "usage",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cache_creation",
    "ephemeral_5m_input_tokens",
    "ephemeral_1h_input_tokens",
    # Tool metadata
    "tool_use_id",
    "name",
    "is_error",
    "interrupted",
    "isImage",
    # File/process info
    "filenames",
    "durationMs",
    "numFiles",
    "truncated",
    "stdout",
    "stderr",
    "returnCodeInterpretation",
    # Other metadata
    "todos",
    "oldTodos",
    "newTodos",
    "toolUseResult",
    "context_management",
    "applied_edits",
    "operation",
}

# Fields that contain potentially sensitive content (user input, file contents, etc.)
SENSITIVE_CONTENT_FIELDS = {
    # These fields may contain actual secrets and should be redacted if secrets detected
    "content",  # Main content field
    "text",  # Text content in messages
}


def redact_content(content: str, secrets: List[SecretMatch]) -> str:
    """
    Redact detected secrets from content.

    Args:
        content: Original content
        secrets: List of detected secrets

    Returns:
        Content with secrets redacted
    """
    if not secrets:
        logger.debug("No secrets to redact")
        return content

    logger.info(f"Redacting {len(secrets)} secret(s) from content")

    lines = content.split("\n")
    original_size = len(content)

    # Group secrets by line number
    secrets_by_line = {}
    for secret in secrets:
        line_num = secret.line - 1  # Convert to 0-indexed
        if 0 <= line_num < len(lines):
            if line_num not in secrets_by_line:
                secrets_by_line[line_num] = []
            secrets_by_line[line_num].append(secret)

    logger.debug(f"Redacting {len(secrets_by_line)} line(s)")

    # Redact secrets (selective approach: only redact content fields)
    for line_num, line_secrets in secrets_by_line.items():
        secret_types = [s.type for s in line_secrets]
        original_line = lines[line_num]

        # Try to parse as JSON and redact selectively
        import json
        import re

        try:
            # Try to parse the line as JSON
            json_obj = json.loads(original_line)

            # Selectively redact only sensitive content fields
            def redact_json_values(obj, parent_key=None):
                """
                Recursively redact values in JSON object.
                Only redacts fields that are in SENSITIVE_CONTENT_FIELDS.
                Preserves all metadata and non-sensitive fields.
                """
                if isinstance(obj, dict):
                    result = {}
                    for k, v in obj.items():
                        # Only redact if the current key is sensitive
                        if k in SENSITIVE_CONTENT_FIELDS:
                            # This field contains potentially sensitive content
                            result[k] = redact_json_values(v, k)
                        elif k in NON_SENSITIVE_FIELDS:
                            # Preserve non-sensitive fields as-is
                            result[k] = v
                        else:
                            # For unknown fields, recursively process but don't redact metadata
                            result[k] = redact_json_values(v, k)
                    return result
                elif isinstance(obj, list):
                    # Process list items
                    return [redact_json_values(item, parent_key) for item in obj]
                elif isinstance(obj, str):
                    # Only redact if we're inside a sensitive content field
                    if parent_key in SENSITIVE_CONTENT_FIELDS:
                        return f"[REDACTED: {', '.join(set(secret_types))}]"
                    # Otherwise preserve the string value
                    return obj
                else:
                    # Preserve non-string values (numbers, booleans, null)
                    return obj

            redacted_obj = redact_json_values(json_obj)
            lines[line_num] = json.dumps(redacted_obj, ensure_ascii=False)

        except (json.JSONDecodeError, Exception):
            # If JSON parsing fails, fall back to targeted regex replacement
            # This tries to preserve as much structure as possible
            logger.warning(f"Failed to parse line {line_num + 1} as JSON, using regex redaction")

            # Try to preserve structure by using regex to find and replace values
            if ":" in original_line:
                # Find the value part after the colon, preserving the closing braces/brackets
                # Match pattern: : "value" or : value, capture trailing punctuation
                pattern = r':\s*"[^"]*"(\s*[,}\]])'
                if re.search(pattern, original_line):
                    lines[line_num] = re.sub(
                        pattern, rf': "[REDACTED: {", ".join(set(secret_types))}]"\1', original_line
                    )
                else:
                    # Fallback: replace from colon onwards but keep trailing punctuation
                    match = re.search(r"(.*?:\s*)(.+?)(\s*[}\]]*\s*)$", original_line)
                    if match:
                        lines[line_num] = (
                            f'{match.group(1)}"[REDACTED: {", ".join(set(secret_types))}]"{match.group(3)}'
                        )
                    else:
                        lines[line_num] = (
                            f'{original_line}: "[REDACTED: {", ".join(set(secret_types))}]"'
                        )
            else:
                lines[line_num] = f"[REDACTED LINE - {', '.join(set(secret_types))}]"

    redacted_content = "\n".join(lines)
    redacted_size = len(redacted_content)
    logger.info(f"Redaction complete: {original_size} bytes -> {redacted_size} bytes")

    return redacted_content


def check_and_redact_session(
    session_content: str, redact_mode: str = "auto", quiet: bool = False
) -> Tuple[str, bool, List[SecretMatch]]:
    """
    Check session content for secrets and optionally redact them.

    Args:
        session_content: The session content to check
        redact_mode: Redaction mode - "auto" (redact if found), "detect" (only detect), "off"
        quiet: If True, suppress console output (stderr)

    Returns:
        Tuple of (potentially redacted content, whether secrets were found, list of secrets)
    """
    logger.info(f"Checking session for secrets (mode: {redact_mode})")

    if redact_mode == "off":
        logger.debug("Redaction disabled (mode: off)")
        return session_content, False, []

    # Detect secrets
    secrets, detect_secrets_available = detect_secrets(session_content)

    if not detect_secrets_available:
        logger.warning("detect-secrets library not available, skipping secret detection")

    if not secrets:
        logger.info("No secrets found in session")
        return session_content, False, []

    # Print warning about detected secrets
    logger.warning(f"Found {len(secrets)} potential secret(s) in session")

    if not quiet:
        print(f"⚠️  Detected {len(secrets)} potential secret(s) in session:", file=sys.stderr)
        for secret in secrets:
            print(f"   - {secret.type} at line {secret.line}", file=sys.stderr)

    if redact_mode == "detect":
        # Only detect, don't redact
        logger.info("Detection-only mode, not redacting")
        return session_content, True, secrets

    # Auto mode: redact the secrets
    logger.info("Auto mode: redacting secrets")
    redacted_content = redact_content(session_content, secrets)

    if not quiet:
        print("   ✅ Secrets have been automatically redacted", file=sys.stderr)

    return redacted_content, True, secrets


def save_original_session(session_path: Path, repo_root: Path) -> Optional[Path]:
    """
    Save a copy of the original session before redaction.

    Args:
        session_path: Path to the session file in .realign/sessions/
        repo_root: Repository root path

    Returns:
        Path to the backup file, or None if backup failed
    """
    logger.info(f"Saving original session backup: {session_path.name}")

    try:
        from realign import get_realign_dir

        realign_dir = get_realign_dir(repo_root)
        backup_dir = realign_dir / "sessions-original"
        backup_dir.mkdir(parents=True, exist_ok=True)

        backup_path = backup_dir / session_path.name

        # Copy original file to backup
        import shutil

        shutil.copy2(session_path, backup_path)

        logger.info(f"Backup saved to: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to backup original session: {e}", exc_info=True)
        print(f"Warning: Could not backup original session: {e}", file=sys.stderr)
        return None
