"""Tests for session management, traversal protection, retention cap, and continuation context."""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest

from ag.web.sessions import (
    SessionManager,
    build_continuation_prompt,
    validate_session_id,
)


def test_session_id_validation():
    """Ensure session IDs match ^[a-f0-9]{16,}$ and reject traversal or invalid characters."""
    valid_id = "0123456789abcdef" * 2
    assert validate_session_id(valid_id) == valid_id

    # Less than 16 hex chars
    with pytest.raises(ValueError, match="Invalid session ID"):
        validate_session_id("0123456789abcdef"[:15])

    # Traversal attempts
    with pytest.raises(ValueError, match="Invalid session ID"):
        validate_session_id("../etc/passwd")

    with pytest.raises(ValueError, match="Invalid session ID"):
        validate_session_id("../../sessions/0123456789abcdef0123456789abcdef")

    with pytest.raises(ValueError, match="Invalid session ID"):
        validate_session_id("0123456789abcdef/subpath")

    with pytest.raises(ValueError, match="Invalid session ID"):
        validate_session_id("not-hex-chars-here!!!")


def test_session_creation_and_file_permissions(tmp_path: Path):
    """Verify session file is created with 0600 permissions and directory with 0700."""
    sessions_dir = tmp_path / "sessions"
    mgr = SessionManager(sessions_dir=sessions_dir)

    session_id = mgr.create_session(title="Test Session")
    session_file = sessions_dir / f"{session_id}.jsonl"

    assert session_file.is_file()

    # Verify directory permissions: 0700
    dir_mode = stat.S_IMODE(os.stat(sessions_dir).st_mode)
    assert dir_mode == 0o700, f"Expected 0700 dir mode, got {oct(dir_mode)}"

    # Verify file permissions: 0600
    file_mode = stat.S_IMODE(os.stat(session_file).st_mode)
    assert file_mode == 0o600, f"Expected 0600 file mode, got {oct(file_mode)}"


def test_session_append_and_load(tmp_path: Path):
    """Verify appending turns and loading session records."""
    sessions_dir = tmp_path / "sessions"
    mgr = SessionManager(sessions_dir=sessions_dir)

    session_id = mgr.create_session()
    mgr.append_turn(session_id, {"type": "user", "text": "Hello Sowl"})
    mgr.append_turn(session_id, {"type": "agent", "run_id": "run_123", "stop_reason": "done", "summary": "Hello there!"})

    records = mgr.load_session(session_id)
    assert len(records) == 2
    assert records[0]["type"] == "user"
    assert records[0]["text"] == "Hello Sowl"
    assert records[1]["type"] == "agent"
    assert records[1]["summary"] == "Hello there!"


def test_session_retention_cap(tmp_path: Path):
    """Verify retention cap prunes oldest sessions when limit exceeded."""
    sessions_dir = tmp_path / "sessions"
    mgr = SessionManager(sessions_dir=sessions_dir, max_sessions=3)

    s1 = mgr.create_session()
    time.sleep(0.02)
    s2 = mgr.create_session()
    time.sleep(0.02)
    s3 = mgr.create_session()
    time.sleep(0.02)

    active_sessions = [s["session_id"] for s in mgr.list_sessions()]
    assert set(active_sessions) == {s1, s2, s3}

    # Creating 4th session should prune s1 (the oldest)
    s4 = mgr.create_session()
    active_after = [s["session_id"] for s in mgr.list_sessions()]
    assert len(active_after) == 3
    assert s1 not in active_after
    assert set(active_after) == {s2, s3, s4}
    assert not (sessions_dir / f"{s1}.jsonl").exists()


def test_session_delete(tmp_path: Path):
    """Verify deleting a session removes the file."""
    sessions_dir = tmp_path / "sessions"
    mgr = SessionManager(sessions_dir=sessions_dir)

    session_id = mgr.create_session()
    assert (sessions_dir / f"{session_id}.jsonl").exists()

    mgr.delete_session(session_id)
    assert not (sessions_dir / f"{session_id}.jsonl").exists()

    # Traversal delete attempt raises error
    with pytest.raises(ValueError):
        mgr.delete_session("../secrets")


def test_build_continuation_prompt():
    """Verify bounded continuation context formatting."""
    history = [
        {"type": "user", "text": "Find notes in workspace"},
        {"type": "agent", "summary": "Found notes.txt with project roadmap"},
        {"type": "user", "text": "Email team about the roadmap"},
        {"type": "agent", "stop_reason": "ask", "question": "What is the team email address?"},
    ]

    prompt = build_continuation_prompt(history, "team@company.com")
    assert "Previous task: Email team about the roadmap" in prompt
    assert "Question: What is the team email address?" in prompt
    assert "Answer: team@company.com" in prompt
