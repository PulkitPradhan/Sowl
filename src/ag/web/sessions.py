"""Session persistence, validation, continuation context, and retention management."""

from __future__ import annotations

import json
import os
import re
import secrets
import stat
import time
from pathlib import Path
from typing import Any

from ag.web.sanitize import sanitize_web_text

SESSION_ID_PATTERN = re.compile(r"^[a-f0-9]{16,}$")


def validate_session_id(session_id: str) -> str:
    """Validate that session ID matches ^[a-f0-9]{16,}$ to prevent path traversal or injection."""
    if not isinstance(session_id, str) or not SESSION_ID_PATTERN.match(session_id):
        raise ValueError(f"Invalid session ID: {session_id!r}")
    return session_id


def build_continuation_prompt(messages: list[dict[str, Any]], new_text: str) -> str:
    """Construct bounded continuation context from previous session turn.

    Includes the most recent user task, final summary, and any pending question/answer.
    """
    clean_new_text = sanitize_web_text(new_text)

    prev_user_task = None
    prev_summary = None
    prev_question = None

    for msg in reversed(messages):
        msg_type = msg.get("type")
        if msg_type == "agent":
            if prev_summary is None:
                prev_summary = msg.get("summary")
            if prev_question is None and msg.get("stop_reason") == "ask":
                prev_question = msg.get("question")
        elif msg_type == "user" and prev_user_task is None:
            prev_user_task = msg.get("text")

    parts: list[str] = []
    if prev_user_task:
        parts.append(f"Previous task: {sanitize_web_text(prev_user_task)}")
    if prev_summary and not prev_question:
        parts.append(f"Result: {sanitize_web_text(prev_summary)}")
    if prev_question:
        parts.append(f"Question: {sanitize_web_text(prev_question)}")
        parts.append(f"Answer: {clean_new_text}")
    else:
        parts.append(f"Task: {clean_new_text}")

    return "\n".join(parts)


class SessionManager:
    """Manages persistent session logs in ~/.ag/sessions/."""

    def __init__(self, sessions_dir: Path | str | None = None, max_sessions: int = 100) -> None:
        self.sessions_dir = (
            Path(sessions_dir).expanduser()
            if sessions_dir
            else Path(os.environ.get("AG_SESSIONS_DIR", "~/.ag/sessions")).expanduser()
        )
        self.max_sessions = max_sessions
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        try:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            self.sessions_dir.chmod(0o700)
        except OSError:
            pass

    def _get_session_path(self, session_id: str) -> Path:
        valid_id = validate_session_id(session_id)
        path = (self.sessions_dir / f"{valid_id}.jsonl").resolve()
        # Traversal sanity check
        if path.parent != self.sessions_dir.resolve():
            raise ValueError(f"Path traversal detected in session ID: {session_id!r}")
        return path

    def _enforce_retention_cap(self) -> None:
        """Prune oldest session files when count exceeds max_sessions."""
        try:
            files = list(self.sessions_dir.glob("*.jsonl"))
            if len(files) < self.max_sessions:
                return

            # Sort by mtime ascending (oldest first)
            files.sort(key=lambda p: p.stat().st_mtime)
            to_remove = len(files) - self.max_sessions + 1
            for p in files[:to_remove]:
                p.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001, S110
            pass

    def create_session(self, title: str | None = None) -> str:
        """Create a new session file with 0600 mode and initial metadata."""
        self._ensure_dir()
        self._enforce_retention_cap()

        session_id = secrets.token_hex(16)
        path = self._get_session_path(session_id)

        meta = {
            "type": "meta",
            "session_id": session_id,
            "title": sanitize_web_text(title or "New Session"),
            "created_at": time.time(),
            "updated_at": time.time(),
        }

        # Write with 0600 mode
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta, separators=(",", ":")) + "\n")

        # Explicitly ensure chmod in case umask restricted it
        try:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

        return session_id

    def list_sessions(self) -> list[dict[str, Any]]:
        """List active sessions with metadata, sorted by updated_at descending."""
        self._ensure_dir()
        sessions: list[dict[str, Any]] = []

        for p in self.sessions_dir.glob("*.jsonl"):
            stem = p.stem
            try:
                validate_session_id(stem)
            except ValueError:
                continue

            try:
                with p.open("r", encoding="utf-8") as f:
                    first_line = f.readline()
                if not first_line:
                    continue
                meta = json.loads(first_line)
                meta["session_id"] = stem
                meta["mtime"] = p.stat().st_mtime
                meta["updated_at"] = meta.get("updated_at", meta["mtime"])
                sessions.append(meta)
            except Exception:  # noqa: BLE001, S112
                continue

        sessions.sort(key=lambda s: s.get("updated_at", 0), reverse=True)
        return sessions

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        """Load all turns for a session."""
        path = self._get_session_path(session_id)
        if not path.is_file():
            raise FileNotFoundError(f"Session not found: {session_id}")

        turns: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("type") != "meta":
                        turns.append(record)
                except json.JSONDecodeError:
                    continue
        return turns

    def append_turn(self, session_id: str, turn_data: dict[str, Any]) -> None:
        """Append a message turn to the session file and update mtime/updated_at."""
        path = self._get_session_path(session_id)
        if not path.is_file():
            # Auto-create if not existing
            self.create_session()

        record = {
            **turn_data,
            "timestamp": time.time(),
        }

        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

        try:
            path.touch()
        except OSError:
            pass

    def delete_session(self, session_id: str) -> bool:
        """Delete a session log file."""
        path = self._get_session_path(session_id)
        if path.is_file():
            path.unlink(missing_ok=True)
            return True
        return False
