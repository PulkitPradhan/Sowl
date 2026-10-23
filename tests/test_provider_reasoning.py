"""Tests for provider reasoning, thinking mode, token usage parsing, and caching."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ag.models.profile import ModelProfile
from ag.models.provider import OpenAIProvider


@pytest.fixture
def base_profile() -> ModelProfile:
    return ModelProfile(
        name="test_profile",
        tier=1,
        model="test-model",
        endpoint="http://localhost:11434/v1",
        context_window=4096,
        thinking="off",
    )


@pytest.mark.anyio
async def test_provider_sends_reasoning_effort_none_when_thinking_off(base_profile: ModelProfile) -> None:
    captured_payload: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        captured_payload = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    provider = OpenAIProvider(base_profile, transport=transport)
    res = await provider.chat([{"role": "user", "content": "hi"}])

    assert captured_payload.get("reasoning_effort") == "none"
    assert res.content == "hello"


@pytest.mark.anyio
async def test_provider_ignores_reasoning_field_and_parses_tokens(base_profile: ModelProfile) -> None:
    # Fixture required by specification:
    # {"message":{"role":"assistant","content":"56","reasoning":"Thinking Process: ..."},"finish_reason":"stop"}
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "56",
                            "reasoning": "Thinking Process: The user wants to know...",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 112,
                    "total_tokens": 132,
                    "completion_tokens_details": {
                        "reasoning_tokens": 100,
                    },
                    "prompt_tokens_details": {
                        "cached_tokens": 8,
                    },
                },
            },
        )

    transport = httpx.MockTransport(handler)
    provider = OpenAIProvider(base_profile, transport=transport)
    res = await provider.chat([{"role": "user", "content": "Calculate 7 * 8"}])

    # Must extract content only, never reasoning
    assert res.content == "56"
    assert res.usage.prompt_tokens == 20
    assert res.usage.completion_tokens == 112
    assert res.usage.reasoning_tokens == 100
    assert res.usage.cached_tokens == 8


@pytest.mark.anyio
async def test_provider_defaults_finish_reason_if_missing(base_profile: ModelProfile) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "done"},
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    transport = httpx.MockTransport(handler)
    provider = OpenAIProvider(base_profile, transport=transport)
    res = await provider.chat([{"role": "user", "content": "ping"}])
    assert res.finish_reason == "stop"
