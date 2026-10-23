"""Tests for planner prompt formatting, KV-cache ordering, observation wrapper, and context budget."""

from __future__ import annotations

import pytest

from ag.models.profile import ModelProfile
from ag.models.provider import FakeModelProvider, ModelResponse, SchemaRejectedError
from ag.planner.planner import (
    Planner,
    TooManyToolsError,
    format_tool_table,
    wrap_observation,
)
from ag.skills.manifest import ArgField, ToolManifest


def test_tool_table_compact_args_signature() -> None:
    tools = {
        "files.read": ToolManifest(
            name="files.read",
            summary="Read a file",
            args={
                "path": ArgField(type="string", description="Relative path", required=True),
                "offset": ArgField(type="integer", description="Byte offset", required=False),
            },
            risk="R0",
        ),
        "browser.open_url": ToolManifest(
            name="browser.open_url",
            summary="Open URL in browser",
            args={
                "url": ArgField(type="string", description="Target URL", required=True),
            },
            risk="R1",
        ),
    }

    table_text = format_tool_table(tools)
    assert "files.read" in table_text
    assert "path: string, offset: integer" in table_text
    assert "browser.open_url" in table_text
    assert "url: string" in table_text
    assert '"path": {' not in table_text  # Compact signature, not nested JSON


def test_fail_loudly_if_more_than_12_tools() -> None:
    tools = {
        f"mock.tool_{i}": ToolManifest(
            name=f"mock.tool_{i}",
            summary="Dummy tool",
            args={},
            risk="R0",
        )
        for i in range(13)
    }
    with pytest.raises(TooManyToolsError, match="Too many tools"):
        format_tool_table(tools)


def test_observation_wrapper_random_delimiter() -> None:
    token = "test_rand_token_123"
    raw_data = f"File content with {token} inside it"
    wrapped = wrap_observation("files.read", raw_data, delimiter_token=token)

    assert f"--- OBSERVATION {token} ---" in wrapped
    assert "[Observation (UNTRUSTED DATA from files.read)]" in wrapped
    # The occurrence of the delimiter inside raw_data must be sanitized/stripped
    assert "File content with [STRIPPED_DELIMITER] inside it" in wrapped or token not in wrapped.split("--- OBSERVATION")[1]


def test_planner_kv_cache_prompt_ordering() -> None:
    profile = ModelProfile(name="test", tier=1, context_window=4096)
    planner = Planner(profile)
    tools = {
        "files.read": ToolManifest(
            name="files.read",
            summary="Read file",
            args={"path": ArgField(type="string")},
            risk="R0",
        )
    }

    messages = planner.build_messages(
        task="Read notes.txt",
        tools=tools,
        history=[
            ({"kind": "tool", "tool": "files.read", "args": {"path": "notes.txt"}}, "File content 1"),
        ],
        run_delimiter="deliv_token",
    )

    # 1. System message first: static prompt + tool table
    assert messages[0]["role"] == "system"
    assert "You control a computer by choosing tools." in messages[0]["content"]
    assert "files.read" in messages[0]["content"]
    # No timestamps or run ids in system prompt
    assert "deliv_token" not in messages[0]["content"]

    # 2. User task
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Read notes.txt"

    # 3. History: Assistant action, followed by user/tool observation (NEVER system)
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] in ("user", "tool")
    assert messages[3]["role"] != "system"
    assert "File content 1" in messages[3]["content"]


@pytest.mark.anyio
async def test_planner_schema_rejected_fallback() -> None:
    profile = ModelProfile(name="test", tier=1, context_window=4096, structured_output=True)

    class RejectionFakeProvider(FakeModelProvider):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        async def chat(self, messages, schema=None):
            self.attempts += 1
            if self.attempts == 1 and schema is not None:
                raise SchemaRejectedError("Endpoint does not support json_schema")
            # Fallback call without schema
            return ModelResponse(
                content='{"kind": "done", "summary": "Finished via fallback"}',
                role="assistant",
                finish_reason="stop",
            )

    provider = RejectionFakeProvider()
    planner = Planner(profile, provider=provider)
    action, _response = await planner.next_action("My task", tools={})

    assert action.kind == "done"
    assert action.summary == "Finished via fallback"
    assert provider.attempts == 2


def test_context_budget_replaces_oldest_observations_with_stubs() -> None:
    # Context window of 200 tokens: large observations will exceed 80% (160 tokens)
    profile = ModelProfile(name="test", tier=1, context_window=200)
    planner = Planner(profile)
    tools = {}

    long_text = "This is a very long text line that repeats " * 50
    history = [
        ({"kind": "tool", "tool": "files.read", "args": {"path": "1.txt"}}, long_text),
        ({"kind": "tool", "tool": "files.read", "args": {"path": "2.txt"}}, "Short observation"),
    ]

    messages = planner.build_messages("Short task", tools, history, run_delimiter="tok123")

    # The oldest observation must be converted to a stub, while preserving task and system prompt
    obs_1 = messages[3]["content"]
    assert "Observation truncated for context budget" in obs_1
    # User task preserved
    assert messages[1]["content"] == "Short task"
