"""OpenAI-compatible ModelProvider and FakeModelProvider implementations."""

from __future__ import annotations

import abc
import time
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from ag.models.profile import ModelProfile


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0


class ModelResponse(BaseModel):
    content: str
    role: str = "assistant"
    finish_reason: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    ttft_ms: float = 0.0
    tokens_per_sec: float = 0.0
    reasoning_tokens_per_sec: float = 0.0
    parsed: Any | None = None


class ModelError(Exception):
    """Base error for model inference failures."""


class SchemaRejectedError(ModelError):
    """Raised when the inference server rejects a json_schema response_format."""


class SchemaValidationError(ModelError):
    """Raised when the model output fails Pydantic schema validation."""


class ModelProvider(abc.ABC):
    """Abstract interface for all model providers."""

    @abc.abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel] | dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Send chat messages to the model, optionally requesting structured output."""


class FakeModelProvider(ModelProvider):
    """Deterministic fake provider for fast in-memory testing without network calls."""

    def __init__(
        self,
        responses: list[str] | None = None,
        default_response: str = "Fake model response",
    ) -> None:
        self.responses = list(responses) if responses else []
        self.default_response = default_response
        self.history: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel] | dict[str, Any] | None = None,
    ) -> ModelResponse:
        self.history.append(messages)
        content = self.responses.pop(0) if self.responses else self.default_response

        parsed = None
        if schema is not None:
            from pydantic import TypeAdapter
            try:
                adapter = TypeAdapter(schema)
                try:
                    parsed = adapter.validate_json(content)
                except ValidationError:
                    import json
                    raw_data = json.loads(content)
                    if isinstance(raw_data, dict) and "kind" not in raw_data:
                        if "tool" in raw_data:
                            raw_data["kind"] = "tool"
                        elif "question" in raw_data:
                            raw_data["kind"] = "ask"
                        elif "summary" in raw_data:
                            raw_data["kind"] = "done"
                        elif "steps" in raw_data:
                            raw_data["kind"] = "plan"
                        parsed = adapter.validate_python(raw_data)
                    else:
                        raise
            except Exception as err:
                raise SchemaValidationError(f"Fake response failed schema validation: {err}") from err

        usage = TokenUsage(
            prompt_tokens=10,
            completion_tokens=len(content.split()),
            total_tokens=10 + len(content.split()),
        )
        return ModelResponse(
            content=content,
            role="assistant",
            finish_reason="stop",
            usage=usage,
            ttft_ms=1.0,
            tokens_per_sec=100.0,
            parsed=parsed,
        )


class OpenAIProvider(ModelProvider):
    """OpenAI-compatible client targeting any standard local or remote inference server."""

    def __init__(
        self,
        profile: ModelProfile,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.profile = profile
        self.transport = transport
        self.endpoint = profile.endpoint.rstrip("/")
        if not self.endpoint.endswith("/v1") and not self.endpoint.endswith(("/v1", "/v1/chat/completions")):
            self.endpoint = f"{self.endpoint}/v1"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel] | dict[str, Any] | None = None,
    ) -> ModelResponse:
        url = f"{self.endpoint}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.profile.model,
            "messages": messages,
            "temperature": self.profile.temperature,
        }

        if self.profile.thinking == "off":
            payload["reasoning_effort"] = "none"

        request_has_schema = False
        if schema is not None and self.profile.structured_output:
            request_has_schema = True
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                json_schema_dict = schema.model_json_schema()
                schema_name = schema.__name__
            elif isinstance(schema, dict):
                json_schema_dict = schema
                schema_name = "response_schema"
            else:
                from pydantic import TypeAdapter
                json_schema_dict = TypeAdapter(schema).json_schema()
                schema_name = "response_schema"

            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": json_schema_dict,
                    "strict": True,
                },
            }

        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = self.profile.get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        timeout = httpx.Timeout(
            connect=self.profile.timeout_connect,
            read=self.profile.timeout_read,
            write=30.0,
            pool=10.0,
        )

        start_time = time.perf_counter()
        async with httpx.AsyncClient(transport=self.transport, timeout=timeout) as client:
            try:
                resp = await client.post(url, json=payload, headers=headers)
            except httpx.ConnectTimeout as err:
                raise ModelError(f"Connection timeout after {self.profile.timeout_connect}s to {url}") from err
            except httpx.ReadTimeout as err:
                raise ModelError(f"Read timeout after {self.profile.timeout_read}s from {url}") from err
            except httpx.RequestError as err:
                raise ModelError(f"HTTP request failed: {err}") from err

        elapsed = time.perf_counter() - start_time
        ttft_ms = elapsed * 1000.0

        if resp.status_code == 400:
            error_body = resp.text
            if request_has_schema or "json_schema" in error_body or "response_format" in error_body:
                raise SchemaRejectedError(f"Server rejected schema: {error_body}")
            raise ModelError(f"Server returned 400 Bad Request: {error_body}")

        if resp.status_code != 200:
            raise ModelError(f"Model request failed [{resp.status_code}]: {resp.text}")

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise ModelError(f"Empty choices in model response: {data}")

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        role = message.get("role", "assistant")
        finish_reason = choice.get("finish_reason") or "stop"

        raw_usage = data.get("usage", {})
        comp_details = raw_usage.get("completion_tokens_details") or {}
        reasoning_tokens = comp_details.get("reasoning_tokens", 0)
        prompt_details = raw_usage.get("prompt_tokens_details") or {}
        cached_tokens = prompt_details.get("cached_tokens", 0)

        usage = TokenUsage(
            prompt_tokens=raw_usage.get("prompt_tokens", 0),
            completion_tokens=raw_usage.get("completion_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
        )

        content_tokens = max(0, usage.completion_tokens - reasoning_tokens)
        tokens_per_sec = (content_tokens / elapsed) if elapsed > 0 else 0.0
        reasoning_tokens_per_sec = (reasoning_tokens / elapsed) if elapsed > 0 else 0.0

        parsed = None
        if schema is not None:
            from pydantic import TypeAdapter
            try:
                adapter = TypeAdapter(schema)
                try:
                    parsed = adapter.validate_json(content)
                except ValidationError:
                    import json
                    raw_data = json.loads(content)
                    if isinstance(raw_data, dict) and "kind" not in raw_data:
                        if "tool" in raw_data:
                            raw_data["kind"] = "tool"
                        elif "question" in raw_data:
                            raw_data["kind"] = "ask"
                        elif "summary" in raw_data:
                            raw_data["kind"] = "done"
                        elif "steps" in raw_data:
                            raw_data["kind"] = "plan"
                        parsed = adapter.validate_python(raw_data)
                    else:
                        raise
            except Exception as err:
                raise SchemaValidationError(f"Failed to validate response against schema: {err}") from err

        return ModelResponse(
            content=content,
            role=role,
            finish_reason=finish_reason,
            usage=usage,
            ttft_ms=round(ttft_ms, 2),
            tokens_per_sec=round(tokens_per_sec, 2),
            reasoning_tokens_per_sec=round(reasoning_tokens_per_sec, 2),
            parsed=parsed,
        )
