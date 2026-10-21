"""Done-condition test: the exact same task runs on two models by changing only the profile."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from ag.models.profile import load_profile
from ag.models.provider import OpenAIProvider


@pytest.mark.anyio
async def test_same_task_runs_on_two_models_by_changing_only_profile():
    fixtures_dir = Path(__file__).parent / "fixtures"

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content.decode("utf-8"))
        model_name = body["model"]
        user_message = body["messages"][0]["content"]

        if model_name == "fake-model-alpha":
            reply = f"[Alpha Output] Handled '{user_message}'"
        elif model_name == "fake-model-beta":
            reply = f"[Beta Output] Handled '{user_message}'"
        else:
            reply = f"[Unknown] Handled '{user_message}'"

        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": reply}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )

    transport = httpx.MockTransport(handler)

    task_messages = [{"role": "user", "content": "Analyze system performance."}]

    # Run on Profile Alpha
    profile_alpha = load_profile(fixtures_dir / "fake_alpha.yaml")
    provider_alpha = OpenAIProvider(profile=profile_alpha, transport=transport)
    result_alpha = await provider_alpha.chat(task_messages)

    assert result_alpha.content == "[Alpha Output] Handled 'Analyze system performance.'"

    # Run on Profile Beta (ONLY the profile changes)
    profile_beta = load_profile(fixtures_dir / "fake_beta.yaml")
    provider_beta = OpenAIProvider(profile=profile_beta, transport=transport)
    result_beta = await provider_beta.chat(task_messages)

    assert result_beta.content == "[Beta Output] Handled 'Analyze system performance.'"
