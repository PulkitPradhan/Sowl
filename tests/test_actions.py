"""Tests for planner actions, discriminated union, placeholder checks, and action hashing."""

from __future__ import annotations

import pytest

from ag.planner.actions import (
    AskAction,
    DoneAction,
    PlanAction,
    ToolAction,
    compute_action_hash,
    parse_action,
    validate_action_arguments,
)


def test_action_discriminated_union() -> None:
    tool_act = parse_action({"kind": "tool", "tool": "files.read", "args": {"path": "a.txt"}})
    assert isinstance(tool_act, ToolAction)
    assert tool_act.tool == "files.read"

    ask_act = parse_action({"kind": "ask", "question": "What is the file name?"})
    assert isinstance(ask_act, AskAction)
    assert ask_act.question == "What is the file name?"

    done_act = parse_action({"kind": "done", "summary": "Task complete."})
    assert isinstance(done_act, DoneAction)
    assert done_act.summary == "Task complete."


def test_plan_action_in_multi_step() -> None:
    plan_act = parse_action({
        "kind": "plan",
        "steps": [
            {"kind": "tool", "tool": "files.list", "args": {"path": "."}},
            {"kind": "tool", "tool": "files.read", "args": {"path": "test.txt"}},
        ],
    })
    assert isinstance(plan_act, PlanAction)
    assert len(plan_act.steps) == 2


def test_reject_placeholder_args() -> None:
    # Reject strings like "<output>", "$step1", "{{result}}", "[TODO]"
    placeholders = [
        "<output>",
        "$step1",
        "{{result}}",
        "[TODO]",
        "<placeholder>",
        "TODO: replace this",
    ]
    for ph in placeholders:
        with pytest.raises(ValueError, match="Placeholder"):
            validate_action_arguments({"path": ph})

    # Legitimate values must pass
    valid = {"path": "src/main.py", "url": "https://example.com", "limit": 10}
    assert validate_action_arguments(valid) == valid


def test_action_hash_consistency() -> None:
    h1 = compute_action_hash("files.read", {"path": "a.txt", "encoding": "utf-8"})
    h2 = compute_action_hash("files.read", {"encoding": "utf-8", "path": "a.txt"})
    # Order of dict keys must not change hash
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex string

    h3 = compute_action_hash("files.read", {"path": "b.txt"})
    assert h1 != h3
