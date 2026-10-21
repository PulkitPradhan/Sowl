"""Tests for OpenAI-compatible ModelProvider and FakeModelProvider."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from ag.models.profile import ModelProfile
from ag.models.provider import (
    FakeModelProvider,
    OpenAIProvider,
    SchemaRejectedError,
    SchemaValidationError,
)


class SampleSchema(BaseModel):
    name: str
    score: int


@pytest.mark.anyio
async def test_fake_model_provider():
    fake = FakeModelProvider(responses=["Hello from fake!"])
    res = await fake.chat([{"role": "user", "content": "hi"}])
    assert res.content == "Hello from fake!"
    assert len(fake.history) == 1
    assert fake.history[0][0]["content"] == "hi"


@pytest.mark.anyio
async def test_fake_model_provider_with_schema():
    fake = FakeModelProvider(responses=['{"name": "Alice", "score": 95}'])
    res = await fake.chat([{"role": "user", "content": "give score"}], schema=SampleSchema)
    assert res.parsed is not None
    assert isinstance(res.parsed, SampleSchema)
    assert res.parsed.name == "Alice"
    assert res.parsed.score == 95


@pytest.mark.anyio
async def test_openai_provider_chat_standard():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "test-model"
        assert body["messages"] == [{"role": "user", "content": "hello"}]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "world"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 5,
                    "total_tokens": 10,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    profile = ModelProfile(name="test", model="test-model", tier=1, endpoint="http://fake.local/v1")
    provider = OpenAIProvider(profile=profile, transport=transport)

    res = await provider.chat([{"role": "user", "content": "hello"}])
    assert res.content == "world"
    assert res.role == "assistant"
    assert res.finish_reason == "stop"
    assert res.usage.prompt_tokens == 5
    assert res.usage.completion_tokens == 5


@pytest.mark.anyio
async def test_openai_provider_structured_output_success():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert "response_format" in body
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["name"] == "SampleSchema"

        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"name": "Bob", "score": 42}),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )

    transport = httpx.MockTransport(handler)
    profile = ModelProfile(
        name="test",
        model="test-model",
        tier=1,
        endpoint="http://fake.local/v1",
        structured_output=True,
    )
    provider = OpenAIProvider(profile=profile, transport=transport)

    res = await provider.chat([{"role": "user", "content": "data"}], schema=SampleSchema)
    assert res.parsed is not None
    assert isinstance(res.parsed, SampleSchema)
    assert res.parsed.name == "Bob"
    assert res.parsed.score == 42


@pytest.mark.anyio
async def test_openai_provider_schema_rejected_by_server():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "Invalid parameter: response_format json_schema not supported"}},
        )

    transport = httpx.MockTransport(handler)
    profile = ModelProfile(name="test", model="test-model", tier=1, endpoint="http://fake.local/v1")
    provider = OpenAIProvider(profile=profile, transport=transport)

    with pytest.raises(SchemaRejectedError) as exc_info:
        await provider.chat([{"role": "user", "content": "data"}], schema=SampleSchema)
    assert "not supported" in str(exc_info.value)


@pytest.mark.anyio
async def test_openai_provider_schema_validation_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"name": "Bob", "score": "not_an_int"}),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )

    transport = httpx.MockTransport(handler)
    profile = ModelProfile(name="test", model="test-model", tier=1, endpoint="http://fake.local/v1")
    provider = OpenAIProvider(profile=profile, transport=transport)

    with pytest.raises(SchemaValidationError):
        await provider.chat([{"role": "user", "content": "data"}], schema=SampleSchema)
