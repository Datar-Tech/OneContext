"""Helpers for composing dashboard-injected system prompts."""

from __future__ import annotations

DEFAULT_CLAUDE_SYSTEM_PROMPT = "use onecontext skill for the session"
CONTEXT_HEADER = "current context:"


def build_claude_system_prompt(
    *, agent_title: str | None = None, agent_description: str | None = None
) -> str:
    """Build the Claude append-system-prompt payload for agent sessions."""
    title = (agent_title or "").strip() or "(empty)"
    description = (agent_description or "").strip() or "(empty)"
    return (
        f"{DEFAULT_CLAUDE_SYSTEM_PROMPT}\n\n"
        f"{CONTEXT_HEADER}\n"
        f"title: {title}\n"
        f"description: {description}"
    )


def build_codex_instructions(
    *, agent_title: str | None = None, agent_description: str | None = None
) -> str:
    """Build CODEX_HOME instructions.md content for agent sessions."""
    title = (agent_title or "").strip() or "(empty)"
    description = (agent_description or "").strip() or "(empty)"
    return (
        f"{DEFAULT_CLAUDE_SYSTEM_PROMPT}\n\n"
        f"{CONTEXT_HEADER}\n"
        f"title: {title}\n"
        f"description: {description}"
    )
