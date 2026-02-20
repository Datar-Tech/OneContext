#!/usr/bin/env python3
"""
Unified LLM Client for ReAlign.

This module provides a centralized interface for calling LLM providers (Claude, OpenAI)
with configurable models and parameters.
"""

import sys
import time
import json
import logging
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

from .logging_config import setup_logger
from . import __version__ as ALINE_VERSION

# Setup dedicated LLM logger
logger = logging.getLogger(__name__)

# Setup detailed LLM call logger (file output disabled by default for llm.log)
_llm_call_logger = None


def _setup_llm_call_logger():
    """Setup dedicated logger for LLM calls with detailed logging."""
    global _llm_call_logger
    if _llm_call_logger is not None:
        return _llm_call_logger

    _llm_call_logger = setup_logger("realign.llm_calls", "llm.log")
    # Keep LLM call logger verbose even when global log level is INFO.
    _llm_call_logger.setLevel(logging.DEBUG)
    return _llm_call_logger


def extract_json(response_text: str) -> Dict[str, Any]:
    """
    Extract JSON object from a raw LLM response, handling Markdown fences.
    Uses strict=False to tolerate control characters in JSON strings.

    Args:
        response_text: Raw LLM response

    Returns:
        Parsed JSON dict

    Raises:
        json.JSONDecodeError: If JSON parsing fails
    """
    if not response_text:
        raise json.JSONDecodeError("Empty response", "", 0)

    json_str = response_text.strip()

    # Remove markdown code fences if present
    if "```json" in response_text:
        json_start = response_text.find("```json") + 7
        json_end = response_text.find("```", json_start)
        if json_end != -1:
            json_str = response_text[json_start:json_end].strip()
    elif "```" in response_text:
        json_start = response_text.find("```") + 3
        json_end = response_text.find("```", json_start)
        if json_end != -1:
            json_str = response_text[json_start:json_end].strip()

    if not json_str:
        raise json.JSONDecodeError("No JSON content found", response_text, 0)

    # Use strict=False to allow control characters in JSON strings
    return json.loads(json_str, strict=False)


def call_llm_cloud(
    task: str,
    payload: Dict[str, Any],
    custom_prompt: Optional[str] = None,
    preset_id: Optional[str] = None,
    timeout: float = 60.0,
    silent: bool = False,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Call LLM via Aline server (cloud proxy).

    This function sends structured task/payload to the server which handles
    the LLM call, protecting API keys and prompts from being exposed to clients.

    Args:
        task: Task type ("summary" | "metadata" | "session_summary" | "event_summary" | "ui_metadata")
        payload: Task-specific data dict
        custom_prompt: Optional user custom prompt override (from ~/.aline/prompts/)
        preset_id: Optional preset ID for ui_metadata task
        timeout: Request timeout in seconds
        silent: If True, suppress progress messages to stderr

    Returns:
        (model_name, result_dict) or (None, None) on failure
    """
    # Setup logging
    call_logger = _setup_llm_call_logger()
    call_start_time = time.time()
    call_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log call initiation
    call_logger.info("=" * 80)
    call_logger.info("LLM CLOUD CALL INITIATED")
    call_logger.info(f"Timestamp: {call_timestamp}")
    call_logger.info(f"Task: {task}")
    call_logger.info(f"Payload keys: {list(payload.keys())}")
    call_logger.info(f"Custom prompt: {'yes' if custom_prompt else 'no'}")
    call_logger.info("-" * 80)

    # Check if httpx is available
    try:
        import httpx
    except ImportError:
        logger.error("httpx not available for cloud LLM calls")
        if not silent:
            print("   ❌ httpx package not installed", file=sys.stderr)
        call_logger.error("LLM CLOUD CALL FAILED: httpx not installed")
        call_logger.error("=" * 80 + "\n")
        return None, None

    # Get auth token
    try:
        from .auth import get_access_token, is_logged_in, load_credentials
    except ImportError:
        logger.error("Auth module not available")
        if not silent:
            print("   ❌ Auth module not available", file=sys.stderr)
        call_logger.error("LLM CLOUD CALL FAILED: auth module not available")
        call_logger.error("=" * 80 + "\n")
        return None, None

    if not is_logged_in():
        logger.debug("Not logged in, cannot use cloud LLM")
        if not silent:
            print("   ❌ Not logged in to Aline cloud", file=sys.stderr)
        call_logger.error("LLM CLOUD CALL FAILED: not logged in")
        call_logger.error("=" * 80 + "\n")
        return None, None

    access_token = get_access_token()
    if not access_token:
        logger.warning("Failed to get access token")
        if not silent:
            print("   ❌ Failed to get access token", file=sys.stderr)
        call_logger.error("LLM CLOUD CALL FAILED: no access token")
        call_logger.error("=" * 80 + "\n")
        return None, None

    client_uid: Optional[str] = None
    try:
        current_user = load_credentials()
        if current_user and getattr(current_user, "user_id", None):
            client_uid = str(current_user.user_id)
    except Exception:
        client_uid = None

    # Get backend URL from config
    from .config import ReAlignConfig

    config = ReAlignConfig.load()
    backend_url = config.share_backend_url or "https://realign-server.vercel.app"
    endpoint = f"{backend_url}/api/llm/invoke"

    # Build request body
    request_body: Dict[str, Any] = {
        "task": task,
        "payload": payload,
        "aline_version": ALINE_VERSION,
    }
    if client_uid:
        request_body["uid"] = client_uid
    if custom_prompt:
        request_body["custom_prompt"] = custom_prompt
    if preset_id:
        request_body["preset_id"] = preset_id

    if not silent:
        print(f"   → Calling Aline cloud LLM ({task})...", file=sys.stderr)

    call_logger.info(f"Endpoint: {endpoint}")
    call_logger.info(f"Request body: {json.dumps(request_body, ensure_ascii=False)[:2000]}")
    call_logger.info("-" * 80)

    try:
        start_time = time.time()
        response = httpx.post(
            endpoint,
            json=request_body,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

        elapsed = time.time() - start_time
        call_logger.info(f"Response status: {response.status_code}")
        call_logger.info(f"Response time: {elapsed:.2f}s")

        response_data: Dict[str, Any] | None = None
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                response_data = parsed
                call_logger.info(
                    f"Response body: {json.dumps(response_data, ensure_ascii=False)[:2000]}"
                )
            else:
                call_logger.info(f"Response body (non-dict JSON): {str(parsed)[:2000]}")
        except Exception:
            call_logger.info("Response body: <non-JSON>")

        # Handle HTTP errors
        if response.status_code == 401:
            logger.warning("Cloud LLM authentication failed")
            if not silent:
                print("   ❌ Cloud LLM authentication failed", file=sys.stderr)
            call_logger.error("LLM CLOUD CALL FAILED: authentication error (401)")
            call_logger.error("=" * 80 + "\n")
            return None, None

        if response.status_code == 429:
            limit_code = response_data.get("code") if isinstance(response_data, dict) else None
            limit_error = response_data.get("error") if isinstance(response_data, dict) else None
            limit_details = (
                response_data.get("details") if isinstance(response_data, dict) else None
            )
            code_suffix = (
                f", code={limit_code}" if isinstance(limit_code, str) and limit_code else ""
            )
            error_suffix = (
                f", error={limit_error}" if isinstance(limit_error, str) and limit_error else ""
            )
            logger.warning("Cloud LLM rate limited")
            if not silent:
                print("   ❌ Cloud LLM rate limited", file=sys.stderr)
            call_logger.error(
                f"LLM CLOUD CALL FAILED: rate limited (429){code_suffix}{error_suffix}"
            )
            if isinstance(limit_details, dict):
                call_logger.error(
                    f"LLM CLOUD CALL FAILED DETAILS: {json.dumps(limit_details, ensure_ascii=False)[:2000]}"
                )
            call_logger.error("=" * 80 + "\n")
            return None, None

        if response.status_code >= 500:
            logger.warning(f"Cloud LLM server error: {response.status_code}")
            if not silent:
                print(f"   ❌ Cloud LLM server error ({response.status_code})", file=sys.stderr)
            call_logger.error(f"LLM CLOUD CALL FAILED: server error ({response.status_code})")
            call_logger.error("=" * 80 + "\n")
            return None, None

        # Parse response
        if response_data is None:
            logger.warning("Cloud LLM call failed: non-JSON response")
            if not silent:
                print("   ❌ Cloud LLM error: invalid response format", file=sys.stderr)
            call_logger.error("LLM CLOUD CALL FAILED: non-JSON response")
            call_logger.error("=" * 80 + "\n")
            return None, None

        data = response_data

        if not data.get("success"):
            error_msg = data.get("error", "Unknown error")
            error_code = data.get("code")
            logger.warning(f"Cloud LLM call failed: {error_msg}")
            if not silent:
                print(f"   ❌ Cloud LLM error: {error_msg}", file=sys.stderr)
            if isinstance(error_code, str) and error_code:
                call_logger.error(f"LLM CLOUD CALL FAILED: {error_msg}, code={error_code}")
            else:
                call_logger.error(f"LLM CLOUD CALL FAILED: {error_msg}")
            call_logger.error("=" * 80 + "\n")
            return None, None

        model_name = data.get("model", "cloud")
        result = data.get("result", {})

        # Log success
        total_elapsed = time.time() - call_start_time
        call_logger.info("LLM CLOUD CALL SUCCEEDED")
        call_logger.info(f"Provider: Cloud ({model_name})")
        call_logger.info(f"Task: {task}")
        call_logger.info(f"Elapsed Time: {elapsed:.2f}s")
        call_logger.info(f"Total Time: {total_elapsed:.2f}s")
        call_logger.info("-" * 80)
        call_logger.info(f"RESULT: {json.dumps(result, ensure_ascii=False)}")
        call_logger.info("=" * 80 + "\n")

        if not silent:
            print(f"   ✅ Cloud LLM success ({model_name})", file=sys.stderr)

        return model_name, result

    except httpx.TimeoutException:
        logger.warning(f"Cloud LLM request timed out after {timeout}s")
        if not silent:
            print(f"   ❌ Cloud LLM request timed out", file=sys.stderr)
        total_elapsed = time.time() - call_start_time
        call_logger.error(f"LLM CLOUD CALL FAILED: timeout after {timeout}s")
        call_logger.error(f"Total Time: {total_elapsed:.2f}s")
        call_logger.error("=" * 80 + "\n")
        return None, None

    except httpx.RequestError as e:
        logger.warning(f"Cloud LLM request error: {e}")
        if not silent:
            print(f"   ❌ Cloud LLM connection error", file=sys.stderr)
        total_elapsed = time.time() - call_start_time
        call_logger.error(f"LLM CLOUD CALL FAILED: request error - {e}")
        call_logger.error(f"Total Time: {total_elapsed:.2f}s")
        call_logger.error("=" * 80 + "\n")
        return None, None

    except Exception as e:
        logger.error(f"Cloud LLM unexpected error: {e}", exc_info=True)
        if not silent:
            print(f"   ❌ Cloud LLM error: {e}", file=sys.stderr)
        total_elapsed = time.time() - call_start_time
        call_logger.error(f"LLM CLOUD CALL FAILED: unexpected error - {e}")
        call_logger.error(f"Total Time: {total_elapsed:.2f}s")
        call_logger.error("=" * 80 + "\n")
        return None, None
