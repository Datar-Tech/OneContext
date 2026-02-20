"""Search command for exploring project history via SQLite."""

import os
import re
import json
from datetime import datetime
import typer
from rich.console import Console
from rich.text import Text
from typing import Any, List, Tuple, Optional

from ..db import get_database

console = Console()


def _resolve_option_default(value):
    """Return Typer option default when search_command is called directly in tests."""
    if hasattr(value, "default"):
        return value.default
    return value


def _extract_text_from_jsonl(content: str) -> List[Tuple[int, str]]:
    """Extract searchable text lines from JSONL content.

    Returns:
        List of (line_number, text) tuples
    """
    results = []
    for line_no, line in enumerate(content.split("\n"), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            text = _extract_text_from_json(data)
            if text:
                # Split multi-line text and preserve line numbers
                for sub_line in text.split("\n"):
                    if sub_line.strip():
                        results.append((line_no, sub_line))
        except json.JSONDecodeError:
            continue
    return results


def _extract_text_from_json(data: dict) -> Optional[str]:
    """Extract human-readable text from a JSONL record."""
    if not isinstance(data, dict):
        return None

    # Claude Code format
    msg_type = data.get("type")
    if msg_type == "assistant":
        parts = []
        for item in data.get("message", {}).get("content", []):
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts) if parts else None

    if msg_type == "user":
        content = data.get("message", {}).get("content", [])
        if isinstance(content, str):
            return content
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts) if parts else None

    return None


def _find_matches(text: str, pattern: str, ignore_case: bool) -> List[Tuple[int, int]]:
    """Find all match spans in text.

    Returns:
        List of (start, end) tuples for each match
    """
    if not text:
        return []

    flags = re.IGNORECASE if ignore_case else 0

    try:
        regex = re.compile(pattern, flags)
        return [(m.start(), m.end()) for m in regex.finditer(text)]
    except re.error:
        return []


def _print_grep_line(
    source_id: str,
    line_no: int,
    text: str,
    matches: List[Tuple[int, int]],
    show_line_numbers: bool,
    source_category: str,  # 'event', 'turn', 'session', 'content'
    field_type: str,  # 'title', 'desc', 'summary', 'content'
    session_id: Optional[str] = None,
    turn_number: Optional[int] = None,
) -> None:
    """Print a single match line with new format.

    Format:
      - Turns: [session]xxx | [turn]xxx | [title/summary] | [line]n: matched_text
      - Events: [event]xxx | [title/desc] | [line]n: matched_text
      - Sessions: [session]xxx | [title/summary] | [line]n: matched_text
      - Content: [session]xxx | [turn]xxx | [line]n: matched_text
    """
    from rich.text import Text

    # Build prefix with pipe separators and labels
    prefix = Text()

    # First field: source category and ID
    if source_category == "event":
        prefix.append("[event]", style="dim")
        prefix.append(source_id, style="dim")
    elif source_category == "turn":
        if session_id:
            prefix.append("[session]", style="dim")
            prefix.append(session_id, style="dim")
        else:
            prefix.append("[session]", style="dim")
            prefix.append(" ", style="dim")
        prefix.append(" | ", style="dim")
        prefix.append("[turn]", style="dim")
        prefix.append(source_id, style="dim")
    elif source_category == "session":
        prefix.append("[session]", style="dim")
        prefix.append(source_id, style="dim")
    elif source_category == "content":
        if session_id:
            prefix.append("[session]", style="dim")
            prefix.append(session_id, style="dim")
        else:
            prefix.append("[session]", style="dim")
            prefix.append(" ", style="dim")
        prefix.append(" | ", style="dim")
        prefix.append("[turn]", style="dim")
        prefix.append(source_id, style="dim")

    # Second field: field type (not for content)
    if source_category == "content":
        # Content doesn't have field type in the middle
        pass
    else:
        prefix.append(" | ", style="dim")
        prefix.append(f"[{field_type}]", style="dim")

    # Third field: line number
    if show_line_numbers:
        prefix.append(" | ", style="dim")
        prefix.append(f"[line {line_no}]", style="dim")

    prefix.append(": ", style="")

    console.print(prefix, end="")
    console.print(_build_highlighted_text(text, matches))


def _build_highlighted_text(text: str, matches: List[Tuple[int, int]]) -> Text:
    """Return highlighted text with matched spans emphasized."""
    highlighted = Text()
    last_end = 0
    for start, end in sorted(matches):
        if start > last_end:
            highlighted.append(text[last_end:start])
        highlighted.append(text[start:end], style="bold red")
        last_end = end
    if last_end < len(text):
        highlighted.append(text[last_end:])
    return highlighted


def _build_content_snippet(
    text: str,
    matches: List[Tuple[int, int]],
    focus_start: int,
    focus_end: int,
    context_chars: int,
) -> Tuple[str, List[Tuple[int, int]]]:
    """Build a compact snippet around a focused content match."""
    left = max(0, focus_start - context_chars)
    right = min(len(text), focus_end + context_chars)

    snippet = text[left:right]
    snippet_matches: List[Tuple[int, int]] = []

    for match_start, match_end in matches:
        if match_end <= left or match_start >= right:
            continue
        clipped_start = max(match_start, left) - left
        clipped_end = min(match_end, right) - left
        if clipped_start < clipped_end:
            snippet_matches.append((clipped_start, clipped_end))

    if left > 0:
        snippet = "..." + snippet
        snippet_matches = [(start + 3, end + 3) for start, end in snippet_matches]

    if right < len(text):
        snippet += "..."

    if not snippet_matches:
        fallback_start = max(focus_start, left) - left
        fallback_end = min(focus_end, right) - left
        if left > 0:
            fallback_start += 3
            fallback_end += 3
        snippet_matches.append((fallback_start, fallback_end))

    return snippet, snippet_matches


def _grep_search_content(
    content: str,
    pattern: str,
    ignore_case: bool,
    snippet_context_chars: int,
) -> List[Tuple[int, str, List[Tuple[int, int]]]]:
    """Collect content snippet matches from a single content blob."""
    snippet_entries: List[Tuple[int, str, List[Tuple[int, int]]]] = []
    seen_snippets = set()
    lines = [(i, line) for i, line in enumerate(content.splitlines(), 1) if line.strip()]

    for line_no, text in lines:
        matches = _find_matches(text, pattern, ignore_case)
        if matches:
            for match_start, match_end in matches:
                snippet, snippet_matches = _build_content_snippet(
                    text,
                    matches,
                    match_start,
                    match_end,
                    snippet_context_chars,
                )
                dedupe_key = (line_no, snippet)
                if dedupe_key in seen_snippets:
                    continue

                seen_snippets.add(dedupe_key)
                snippet_entries.append((line_no, snippet, snippet_matches))

    return snippet_entries


def _print_content_snippet_group(
    source_id: str,
    session_id: Optional[str],
    snippet_entries: List[Tuple[int, str, List[Tuple[int, int]]]],
    show_line_numbers: bool,
) -> None:
    """Print grouped snippets for a single turn."""
    header = Text()
    if session_id:
        header.append("[session]", style="dim")
        header.append(session_id, style="dim")
    else:
        header.append("[session]", style="dim")
        header.append(" ", style="dim")
    header.append(" | ", style="dim")
    header.append("[turn]", style="dim")
    header.append(source_id, style="dim")
    header.append(":", style="")
    console.print(header)

    for line_no, snippet, snippet_matches in snippet_entries:
        line_prefix = Text("  ", style="dim")
        if show_line_numbers:
            line_prefix.append(f"[line {line_no}] ", style="dim")
        line_prefix.append(": ", style="")
        console.print(line_prefix, end="")
        console.print(_build_highlighted_text(snippet, snippet_matches))


def _resolve_id_prefixes(db, table: str, selectors: str) -> List[str]:
    """Resolve a comma-separated list of IDs or prefixes to full IDs."""
    if not selectors:
        return []

    prefixes = [s.strip() for s in selectors.split(",") if s.strip()]
    resolved_ids = []

    # Use a raw connection to check for prefixes
    conn = db._get_connection()
    cursor = conn.cursor()

    for prefix in prefixes:
        # If it's already a full UUID (36 chars), just add it
        if len(prefix) == 36:
            resolved_ids.append(prefix)
            continue

        # Search for full IDs matching the prefix
        cursor.execute(f"SELECT id FROM {table} WHERE id LIKE ?", (f"{prefix}%",))
        matches = [row[0] for row in cursor.fetchall()]
        if matches:
            resolved_ids.extend(matches)
        else:
            # If no matches found, keep the prefix as is
            resolved_ids.append(prefix)

    return list(set(resolved_ids))


def _timestamp_to_epoch(value: Any) -> float:
    """Convert a timestamp-like value to epoch seconds for sorting."""
    if value is None:
        return 0.0

    if isinstance(value, datetime):
        return value.timestamp()

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0

    return 0.0


def search_command(
    query: str = typer.Argument(..., help="Search query (regex pattern)"),
    type: str = typer.Option(
        "all", "--type", "-t", help="Search type: all, turn, session, content"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of shown results"),
    from_index: int = typer.Option(0, "--from", help="Start offset (0-based, inclusive)"),
    to_index: Optional[int] = typer.Option(
        None, "--to", help="End offset (0-based, exclusive). Overrides --limit window size."
    ),
    ignore_case: bool = typer.Option(
        True, "-i/--case-sensitive", help="Ignore case (default: True)"
    ),
    count_only: bool = typer.Option(False, "--count", "-c", help="Only show match count"),
    line_numbers: bool = typer.Option(
        True, "-n/--no-line-numbers", help="Show line numbers (default: True)"
    ),
    snippet_context_chars: int = typer.Option(
        120,
        "--snippet-context",
        help="Context chars on each side for content snippet output.",
    ),
    sessions: Optional[str] = typer.Option(
        None,
        "--sessions",
        "-s",
        help="Limit search to specific sessions (comma-separated IDs)",
    ),
    turns: Optional[str] = typer.Option(
        None,
        "--turns",
        help="Limit content search to specific turns (comma-separated IDs)",
    ),
) -> int:
    """
    Search project history, turns, sessions, and content.

    This command uses the ReAlign SQLite database to perform regex searches.

    Search types:
      - turn: Search turn title and summary only
      - session: Search session title and summary
      - content: Search full turn content (JSONL)
      - all: Search turns and sessions (default)

    Note: 'all' does not include 'content' search as it can be very slow.
    Use '-t content' to search full JSONL dialogue history.

    Examples:
        aline search "sqlite.*migration"
        aline search -t turn "refactor"         # Search turn titles/summaries
        aline search -t content "error|bug"     # Search content snippets
        aline search -t session "migration"     # Search sessions
        aline search "pattern" -c               # Count matches only
        aline search "pattern" --from 10 --to 30  # Show a result window
        aline search -t content "error|bug" --snippet-context 40
        aline search "bug" -s abc123,def456     # Search within specific sessions
        aline search "bug" -t content --turns turn1,turn2  # Search within specific turns
    """
    try:
        type = str(_resolve_option_default(type) or "all").strip().lower()
        limit = int(_resolve_option_default(limit) or 10)
        from_index = int(_resolve_option_default(from_index) or 0)
        to_index_raw = _resolve_option_default(to_index)
        to_index = int(to_index_raw) if to_index_raw is not None else None
        ignore_case = bool(_resolve_option_default(ignore_case))
        count_only = bool(_resolve_option_default(count_only))
        line_numbers = bool(_resolve_option_default(line_numbers))
        snippet_context_chars = int(_resolve_option_default(snippet_context_chars) or 0)
        sessions = _resolve_option_default(sessions)
        turns = _resolve_option_default(turns)

        allowed_types = {"all", "turn", "session", "content"}
        if type not in allowed_types:
            console.print(
                "[red]Search failed:[/red] Unsupported search type "
                f"'{type}'. Allowed types: all, turn, session, content."
            )
            return 1

        if limit <= 0:
            console.print("[red]Search failed:[/red] --limit must be greater than 0.")
            return 1

        if from_index < 0:
            console.print("[red]Search failed:[/red] --from must be greater than or equal to 0.")
            return 1

        if to_index is not None and to_index <= from_index:
            console.print("[red]Search failed:[/red] --to must be greater than --from.")
            return 1

        if snippet_context_chars < 0:
            console.print(
                "[red]Search failed:[/red] --snippet-context must be greater than or equal to 0."
            )
            return 1

        page_start = from_index
        page_size = (to_index - from_index) if to_index is not None else limit
        page_end = page_start + page_size

        agent_id = os.environ.get("ALINE_AGENT_ID", "").strip()
        if not agent_id:
            console.print(
                "[red]Search failed:[/red] Search is unavailable outside the OneContext "
                "environment. Please run this command inside a OneContext dashboard session."
            )
            return 1

        db = get_database()

        results = {}
        total_result_items = 0
        total_turn_items = 0
        total_session_items = 0
        total_content_items = 0

        # Enforce agent scoping first to guarantee OneContext isolation.
        agent_sessions = db.get_sessions_by_agent_id(agent_id)
        # Always set agent_session_ids when agent_id exists
        # (empty list means no sessions for this agent -> empty results)
        agent_session_ids = [s.id for s in agent_sessions]

        # Parse session IDs if provided (resolve prefixes)
        session_ids = _resolve_id_prefixes(db, "sessions", sessions) or None

        # Intersect with agent sessions first (highest priority)
        if agent_session_ids is not None:
            if session_ids:
                session_ids = list(set(session_ids) & set(agent_session_ids))
            else:
                session_ids = agent_session_ids if agent_session_ids else []

        # Parse turn IDs if provided (for content search)
        turn_ids = _resolve_id_prefixes(db, "turns", turns) or None

        if type == "all":
            if session_ids is not None and len(session_ids) == 0:
                results["ordered"] = []
                results["turns"] = []
                results["sessions"] = []
            else:
                total_turn_items = db.count_search_conversations(
                    query,
                    use_regex=True,
                    ignore_case=ignore_case,
                    session_ids=session_ids if session_ids else None,
                )
                total_session_items = db.count_search_sessions(
                    query,
                    use_regex=True,
                    ignore_case=ignore_case,
                    session_ids=session_ids if session_ids else None,
                )
                total_result_items = total_turn_items + total_session_items

                turns_pool = db.search_conversations(
                    query,
                    limit=page_end,
                    offset=0,
                    use_regex=True,
                    ignore_case=ignore_case,
                    session_ids=session_ids if session_ids else None,
                )
                sessions_pool = db.search_sessions(
                    query,
                    limit=page_end,
                    offset=0,
                    use_regex=True,
                    ignore_case=ignore_case,
                    session_ids=session_ids if session_ids else None,
                )

                merged: List[Tuple[str, float, Any]] = []
                for turn_item in turns_pool:
                    merged.append(
                        ("turn", _timestamp_to_epoch(turn_item.get("timestamp")), turn_item)
                    )
                for session_item in sessions_pool:
                    merged.append(
                        ("session", _timestamp_to_epoch(session_item.updated_at), session_item)
                    )
                merged.sort(key=lambda item: item[1], reverse=True)

                page_items = merged[page_start:page_end]
                results["ordered"] = page_items
                results["turns"] = [item for kind, _, item in page_items if kind == "turn"]
                results["sessions"] = [item for kind, _, item in page_items if kind == "session"]

        elif type == "turn":
            if session_ids is not None and len(session_ids) == 0:
                results["turns"] = []
            else:
                total_turn_items = db.count_search_conversations(
                    query,
                    use_regex=True,
                    ignore_case=ignore_case,
                    session_ids=session_ids if session_ids else None,
                )
                total_result_items = total_turn_items
                results["turns"] = db.search_conversations(
                    query,
                    limit=page_size,
                    offset=page_start,
                    use_regex=True,
                    ignore_case=ignore_case,
                    session_ids=session_ids if session_ids else None,
                )

        elif type == "session":
            if session_ids is not None and len(session_ids) == 0:
                results["sessions"] = []
            else:
                total_session_items = db.count_search_sessions(
                    query,
                    use_regex=True,
                    ignore_case=ignore_case,
                    session_ids=session_ids if session_ids else None,
                )
                total_result_items = total_session_items
                results["sessions"] = db.search_sessions(
                    query,
                    limit=page_size,
                    offset=page_start,
                    use_regex=True,
                    ignore_case=ignore_case,
                    session_ids=session_ids if session_ids else None,
                )

        elif type == "content":
            if session_ids is not None and len(session_ids) == 0:
                results["content"] = []
            else:
                total_content_items = db.count_search_turn_content(
                    query,
                    use_regex=True,
                    ignore_case=ignore_case,
                    session_ids=session_ids if session_ids else None,
                    turn_ids=turn_ids,
                )
                results["content"] = db.search_turn_content(
                    query,
                    limit=total_content_items if total_content_items > 0 else 1,
                    offset=0,
                    use_regex=True,
                    ignore_case=ignore_case,
                    session_ids=session_ids if session_ids else None,
                    turn_ids=turn_ids,
                )

        if not count_only:
            console.print(f"\n[bold]Regex Search:[/bold] '{query}'")

        shown_results = 0

        if type == "all":
            ordered = results.get("ordered", [])
            shown_results = len(ordered)

            if not count_only:
                for kind, _, item in ordered:
                    if kind == "turn":
                        for field_name, field_value in [
                            ("title", item.get("title")),
                            ("summary", item.get("summary")),
                        ]:
                            if field_value:
                                matches = _find_matches(field_value, query, ignore_case)
                                if matches:
                                    _print_grep_line(
                                        item["turn_id"],
                                        1,
                                        field_value[:200],
                                        matches,
                                        line_numbers,
                                        source_category="turn",
                                        field_type=field_name,
                                        session_id=item.get("session_id"),
                                        turn_number=item.get("turn_number"),
                                    )
                    else:
                        for field_name, field_value in [
                            ("title", item.session_title),
                            ("summary", item.session_summary),
                        ]:
                            if field_value:
                                matches = _find_matches(field_value, query, ignore_case)
                                if matches:
                                    _print_grep_line(
                                        item.id,
                                        1,
                                        field_value[:200],
                                        matches,
                                        line_numbers,
                                        source_category="session",
                                        field_type=field_name,
                                    )
            if total_result_items == 0:
                total_result_items = total_turn_items + total_session_items

        elif type == "turn":
            shown_results = len(results.get("turns", []))
            if not count_only:
                for turn_item in results.get("turns", []):
                    for field_name, field_value in [
                        ("title", turn_item.get("title")),
                        ("summary", turn_item.get("summary")),
                    ]:
                        if field_value:
                            matches = _find_matches(field_value, query, ignore_case)
                            if matches:
                                _print_grep_line(
                                    turn_item["turn_id"],
                                    1,
                                    field_value[:200],
                                    matches,
                                    line_numbers,
                                    source_category="turn",
                                    field_type=field_name,
                                    session_id=turn_item.get("session_id"),
                                    turn_number=turn_item.get("turn_number"),
                                )

        elif type == "session":
            shown_results = len(results.get("sessions", []))
            if not count_only:
                for session_item in results.get("sessions", []):
                    for field_name, field_value in [
                        ("title", session_item.session_title),
                        ("summary", session_item.session_summary),
                    ]:
                        if field_value:
                            matches = _find_matches(field_value, query, ignore_case)
                            if matches:
                                _print_grep_line(
                                    session_item.id,
                                    1,
                                    field_value[:200],
                                    matches,
                                    line_numbers,
                                    source_category="session",
                                    field_type=field_name,
                                )

        elif type == "content":
            snippet_units: List[Dict[str, Any]] = []
            for content_item in results.get("content", []):
                content = content_item.get("content", "")
                if not content:
                    continue

                snippet_entries = _grep_search_content(
                    content,
                    query,
                    ignore_case,
                    snippet_context_chars,
                )
                for line_no, snippet, snippet_matches in snippet_entries:
                    snippet_units.append(
                        {
                            "turn_id": content_item["turn_id"],
                            "session_id": content_item.get("session_id"),
                            "line_no": line_no,
                            "snippet": snippet,
                            "snippet_matches": snippet_matches,
                        }
                    )

            total_result_items = len(snippet_units)
            window_units = snippet_units[page_start:page_end]
            shown_results = len(window_units)

            if not count_only and window_units:
                grouped: Dict[str, List[Tuple[int, str, List[Tuple[int, int]]]]] = {}
                turn_order: List[str] = []
                session_by_turn: Dict[str, Optional[str]] = {}
                for unit in window_units:
                    turn_id = unit["turn_id"]
                    if turn_id not in grouped:
                        grouped[turn_id] = []
                        turn_order.append(turn_id)
                        session_by_turn[turn_id] = unit.get("session_id")
                    grouped[turn_id].append(
                        (unit["line_no"], unit["snippet"], unit["snippet_matches"])
                    )

                for turn_id in turn_order:
                    _print_content_snippet_group(
                        turn_id,
                        session_by_turn.get(turn_id),
                        grouped[turn_id],
                        line_numbers,
                    )

        shown_end = page_start + shown_results

        console.print()
        if type == "all":
            console.print(
                "[dim]Matched "
                f"{total_result_items} total results "
                f"(turns={total_turn_items}, sessions={total_session_items}); "
                f"showing {shown_results} in [{page_start}, {shown_end}).[/dim]"
            )
        elif type == "turn":
            console.print(
                "[dim]Matched "
                f"{total_turn_items} total turn results; "
                f"showing {shown_results} in [{page_start}, {shown_end}).[/dim]"
            )
        elif type == "session":
            console.print(
                "[dim]Matched "
                f"{total_session_items} total session results; "
                f"showing {shown_results} in [{page_start}, {shown_end}).[/dim]"
            )
        else:
            console.print(
                "[dim]Matched "
                f"{total_result_items} total content snippet results; "
                f"showing {shown_results} in [{page_start}, {shown_end}).[/dim]"
            )

        if page_start >= total_result_items and total_result_items > 0:
            console.print(
                f"[yellow]Window start {page_start} is beyond total matches {total_result_items}. "
                "Use a smaller --from value.[/yellow]"
            )
        elif total_result_items > shown_end:
            next_from = shown_end
            next_to = min(next_from + page_size, total_result_items)
            console.print(
                "[yellow]More matches available. "
                f"Use --from {next_from} --to {next_to} to view the next window, "
                "or increase --limit.[/yellow]"
            )

        return 0

    except Exception as e:
        console.print(f"[red]Error searching: {e}[/red]")
        return 1
