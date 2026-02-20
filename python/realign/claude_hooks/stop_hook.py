#!/usr/bin/env python3
"""
Claude Code Stop Hook - 通知 watcher turn 已完成

当 Claude Code agent 完成响应时，此脚本被调用。
它会写入一个信号文件，让 watcher 立即检测到 turn 完成。

使用方式:
    此脚本通过 Claude Code 的 Stop hook 机制调用，
    接收 stdin JSON 和环境变量。

环境变量:
    CLAUDE_PROJECT_DIR - 项目根目录路径

stdin JSON 格式:
    {
        "session_id": "...",
        "transcript_path": "...",
        "cwd": "...",
        "hook_event_name": "Stop"
    }
"""

import os
import sys
import json
import time
import subprocess
import sqlite3
import uuid
from pathlib import Path

try:
    from terminal_state import update_terminal_mapping  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    update_terminal_mapping = None


def get_signal_dir() -> Path:
    """获取信号文件目录"""
    signal_dir = Path.home() / ".aline" / ".signals"
    signal_dir.mkdir(parents=True, exist_ok=True)
    return signal_dir


def _parse_config_sqlite_db_path(config_path: Path) -> str | None:
    """Best-effort parse `sqlite_db_path` from ~/.aline/config.yaml without PyYAML."""
    try:
        for raw_line in config_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if not line.startswith("sqlite_db_path:"):
                continue
            _, value = line.split(":", 1)
            value = value.strip()
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            return value.strip() or None
    except Exception:
        return None
    return None


def _resolve_sqlite_db_path() -> Path:
    """Resolve Aline sqlite DB path.

    Keep this lightweight because this script is executed as a Claude hook.
    """
    env_db_path = (
        os.getenv("REALIGN_SQLITE_DB_PATH")
        or os.getenv("REALIGN_DB_PATH")
        or os.getenv("ALINE_DB_PATH")
    )
    if env_db_path:
        return Path(env_db_path).expanduser()

    config_path = Path.home() / ".aline" / "config.yaml"
    cfg = _parse_config_sqlite_db_path(config_path) if config_path.exists() else None
    if cfg:
        return Path(cfg).expanduser()

    return Path.home() / ".aline" / "db" / "aline.db"


def _try_enqueue_session_process_job(
    *,
    session_id: str,
    session_file_path: str,
    workspace_path: str | None,
    session_type: str | None,
    source_event: str | None,
    no_track: bool,
    agent_id: str | None,
    terminal_id: str | None,
    connect_timeout_seconds: float,
) -> bool:
    """Best-effort enqueue into sqlite jobs table. Never raises."""
    try:
        db_path = _resolve_sqlite_db_path()
        if not db_path.exists():
            return False

        payload: dict = {"session_id": session_id, "session_file_path": session_file_path}
        if workspace_path is not None:
            payload["workspace_path"] = workspace_path
        if session_type:
            payload["session_type"] = session_type
        if source_event:
            payload["source_event"] = source_event
        if no_track:
            payload["no_track"] = True
        if agent_id:
            payload["agent_id"] = agent_id
        if terminal_id:
            payload["terminal_id"] = terminal_id

        job_id = str(uuid.uuid4())
        payload_json = json.dumps(payload, ensure_ascii=False)
        dedupe_key = f"session_process:{session_id}"

        conn = sqlite3.connect(str(db_path), timeout=float(connect_timeout_seconds))
        try:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, kind, dedupe_key, payload, status, priority, attempts, next_run_at,
                    locked_until, locked_by, reschedule, last_error, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, 'queued', ?, 0, datetime('now'),
                    NULL, NULL, 0, NULL, datetime('now'), datetime('now')
                )
                ON CONFLICT(dedupe_key) DO UPDATE SET
                    kind=excluded.kind,
                    payload=excluded.payload,
                    priority=MAX(COALESCE(jobs.priority, 0), COALESCE(excluded.priority, 0)),
                    attempts=CASE
                        WHEN jobs.status='retry' THEN 0
                        ELSE COALESCE(jobs.attempts, 0)
                    END,
                    updated_at=datetime('now'),
                    reschedule=CASE
                        WHEN jobs.status='processing' THEN 1
                        ELSE COALESCE(jobs.reschedule, 0)
                    END,
                    last_error=CASE
                        WHEN jobs.status='retry' THEN NULL
                        ELSE jobs.last_error
                    END,
                    status=CASE
                        WHEN jobs.status='processing' THEN jobs.status
                        WHEN jobs.status='queued' THEN jobs.status
                        WHEN jobs.status='retry' THEN 'queued'
                        WHEN jobs.status='done' THEN 'queued'
                        ELSE 'queued'
                    END,
                    next_run_at=CASE
                        WHEN jobs.status='processing' THEN jobs.next_run_at
                        WHEN jobs.next_run_at IS NULL THEN excluded.next_run_at
                        WHEN excluded.next_run_at < jobs.next_run_at THEN excluded.next_run_at
                        ELSE jobs.next_run_at
                    END
                """,
                (
                    job_id,
                    "session_process",
                    dedupe_key,
                    payload_json,
                    15,
                ),
            )
            conn.commit()
            return True
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception:
        return False


def main():
    """主函数"""
    try:
        # 读取 stdin JSON
        stdin_data = sys.stdin.read()
        try:
            data = json.loads(stdin_data) if stdin_data.strip() else {}
        except json.JSONDecodeError:
            data = {}

        session = data.get("session") or {}
        session_id = session.get("id") or data.get("session_id") or data.get("sessionId") or ""
        transcript_path = (
            session.get("transcript_path")
            or data.get("transcript_path")
            or data.get("transcriptPath")
            or ""
        )
        cwd = data.get("cwd") or ""

        # From environment
        project_dir = os.environ.get("CLAUDE_PROJECT_DIR", cwd)
        terminal_id = os.environ.get("ALINE_TERMINAL_ID", "")
        inner_socket = os.environ.get("ALINE_INNER_TMUX_SOCKET", "")
        inner_session = os.environ.get("ALINE_INNER_TMUX_SESSION", "")
        agent_id = os.environ.get("ALINE_AGENT_ID", "")

        if not terminal_id:
            try:
                terminal_id = (
                    subprocess.run(
                        ["tmux", "display-message", "-p", "#{@aline_terminal_id}"],
                        text=True,
                        capture_output=True,
                        check=False,
                    ).stdout
                    or ""
                ).strip()
            except Exception:
                terminal_id = terminal_id

        # 如果没有 session_id，尝试从 transcript_path 提取
        if not session_id and transcript_path:
            # transcript_path 通常是 ~/.claude/projects/<project>/<session_id>.jsonl
            transcript_file = Path(transcript_path)
            if transcript_file.suffix == ".jsonl":
                session_id = transcript_file.stem

        # 如果仍然没有 session_id，生成一个临时的
        if not session_id:
            session_id = f"unknown_{int(time.time() * 1000)}"

        # Check for no-track mode
        no_track = os.environ.get("ALINE_NO_TRACK", "") == "1"

        # Fast path: enqueue directly (no signal polling) if transcript_path is available.
        disable_direct_enqueue = os.environ.get("ALINE_STOP_HOOK_DISABLE_DB_ENQUEUE", "") == "1"
        connect_timeout = float(os.environ.get("ALINE_STOP_HOOK_DB_TIMEOUT", "0.2"))

        enqueued = False
        if (not disable_direct_enqueue) and transcript_path:
            enqueued = _try_enqueue_session_process_job(
                session_id=session_id,
                session_file_path=transcript_path,
                workspace_path=project_dir or None,
                session_type="claude",
                source_event="stop",
                no_track=no_track,
                agent_id=agent_id or None,
                terminal_id=terminal_id or None,
                connect_timeout_seconds=connect_timeout,
            )

        if not enqueued:
            # Fallback: write a stop signal file so the watcher can pick it up.
            signal_dir = get_signal_dir()
            timestamp_ms = int(time.time() * 1000)
            signal_file = signal_dir / f"{session_id}_{timestamp_ms}.signal"
            tmp_file = signal_dir / f"{session_id}_{timestamp_ms}.signal.tmp"
            signal_data = {
                "session_id": session_id,
                "terminal_id": terminal_id,
                "agent_id": agent_id,
                "project_dir": project_dir,
                "transcript_path": transcript_path,
                "cwd": cwd,
                "timestamp": time.time(),
                "hook_event": "Stop",
            }
            if no_track:
                signal_data["no_track"] = True

            # Write atomically to avoid watcher reading a partial JSON file.
            tmp_file.write_text(json.dumps(signal_data, indent=2))
            tmp_file.replace(signal_file)

        # Best-effort: tag the tmux "terminal tab" with the Claude session id.
        try:
            if terminal_id and inner_socket and inner_session and session_id:
                proc = subprocess.run(
                    [
                        "tmux",
                        "-L",
                        inner_socket,
                        "list-windows",
                        "-t",
                        inner_session,
                        "-F",
                        "#{window_id}\t#{@aline_terminal_id}",
                    ],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                for line in (proc.stdout or "").splitlines():
                    parts = line.split("\t", 1)
                    if len(parts) != 2:
                        continue
                    window_id, win_terminal_id = parts
                    if win_terminal_id == terminal_id:
                        subprocess.run(
                            [
                                "tmux",
                                "-L",
                                inner_socket,
                                "set-option",
                                "-w",
                                "-t",
                                window_id,
                                "@aline_provider",
                                "claude",
                            ],
                            check=False,
                        )
                        subprocess.run(
                            [
                                "tmux",
                                "-L",
                                inner_socket,
                                "set-option",
                                "-w",
                                "-t",
                                window_id,
                                "@aline_session_type",
                                "claude",
                            ],
                            check=False,
                        )
                        subprocess.run(
                            [
                                "tmux",
                                "-L",
                                inner_socket,
                                "set-option",
                                "-w",
                                "-t",
                                window_id,
                                "@aline_session_id",
                                session_id,
                            ],
                            check=False,
                        )
                        # Set attention state to notify dashboard
                        subprocess.run(
                            [
                                "tmux",
                                "-L",
                                inner_socket,
                                "set-option",
                                "-w",
                                "-t",
                                window_id,
                                "@aline_attention",
                                "stop",
                            ],
                            check=False,
                        )
                        # Set no-track flag if applicable
                        if no_track:
                            subprocess.run(
                                [
                                    "tmux",
                                    "-L",
                                    inner_socket,
                                    "set-option",
                                    "-w",
                                    "-t",
                                    window_id,
                                    "@aline_no_track",
                                    "1",
                                ],
                                check=False,
                            )
                        if transcript_path:
                            subprocess.run(
                                [
                                    "tmux",
                                    "-L",
                                    inner_socket,
                                    "set-option",
                                    "-w",
                                    "-t",
                                    window_id,
                                    "@aline_transcript_path",
                                    transcript_path,
                                ],
                                check=False,
                            )
                        break
            elif session_id:
                window_id = (
                    subprocess.run(
                        ["tmux", "display-message", "-p", "#{window_id}"],
                        text=True,
                        capture_output=True,
                        check=False,
                    ).stdout
                    or ""
                ).strip()
                should_tag = bool(terminal_id)
                if not should_tag:
                    try:
                        context_id = (
                            subprocess.run(
                                ["tmux", "display-message", "-p", "#{@aline_context_id}"],
                                text=True,
                                capture_output=True,
                                check=False,
                            ).stdout
                            or ""
                        ).strip()
                        should_tag = bool(context_id)
                    except Exception:
                        should_tag = False

                if window_id and should_tag:
                    subprocess.run(
                        [
                            "tmux",
                            "set-option",
                            "-w",
                            "-t",
                            window_id,
                            "@aline_provider",
                            "claude",
                        ],
                        check=False,
                    )
                    subprocess.run(
                        [
                            "tmux",
                            "set-option",
                            "-w",
                            "-t",
                            window_id,
                            "@aline_session_type",
                            "claude",
                        ],
                        check=False,
                    )
                    subprocess.run(
                        [
                            "tmux",
                            "set-option",
                            "-w",
                            "-t",
                            window_id,
                            "@aline_session_id",
                            session_id,
                        ],
                        check=False,
                    )
                    # Set attention state to notify dashboard
                    subprocess.run(
                        [
                            "tmux",
                            "set-option",
                            "-w",
                            "-t",
                            window_id,
                            "@aline_attention",
                            "stop",
                        ],
                        check=False,
                    )
                    # Set no-track flag if applicable
                    if no_track:
                        subprocess.run(
                            [
                                "tmux",
                                "set-option",
                                "-w",
                                "-t",
                                window_id,
                                "@aline_no_track",
                                "1",
                            ],
                            check=False,
                        )
                    if transcript_path:
                        subprocess.run(
                            [
                                "tmux",
                                "set-option",
                                "-w",
                                "-t",
                                window_id,
                                "@aline_transcript_path",
                                transcript_path,
                            ],
                            check=False,
                        )
        except Exception:
            pass

        # Best-effort: persist mapping to ~/.aline/terminal.json to survive tmux restarts.
        try:
            if update_terminal_mapping and terminal_id and session_id:
                update_terminal_mapping(
                    terminal_id=terminal_id,
                    provider="claude",
                    session_type="claude",
                    session_id=session_id,
                    transcript_path=transcript_path,
                    cwd=cwd,
                    project_dir=project_dir,
                    source="Stop",
                    agent_id=agent_id if agent_id else None,
                )
        except Exception:
            pass

        # 可选：输出调试信息到 stderr（不会显示给用户）
        # print(f"[Aline Stop Hook] Signal written: {signal_file.name}", file=sys.stderr)

        # Exit 0 表示成功，不阻止 Claude 停止
        sys.exit(0)

    except Exception as e:
        # 出错时静默失败，不影响 Claude Code 的正常运行
        # print(f"[Aline Stop Hook] Error: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
