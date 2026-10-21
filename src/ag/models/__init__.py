"""Model layer: OpenAI-compatible client, profiles, and hardware doctor."""

from ag.models.doctor import (
    BenchmarkResult,
    SystemSpecs,
    benchmark_endpoint,
    get_system_specs,
    list_server_models,
    recommend_tier,
)
from ag.models.profile import (
    ModelProfile,
    get_active_profile,
    list_profiles,
    load_profile,
    set_active_profile,
)
from ag.models.provider import (
    FakeModelProvider,
    ModelError,
    ModelProvider,
    ModelResponse,
    OpenAIProvider,
    SchemaRejectedError,
    SchemaValidationError,
    TokenUsage,
)

__all__ = [
    "BenchmarkResult",
    "FakeModelProvider",
    "ModelError",
    "ModelProfile",
    "ModelProvider",
    "ModelResponse",
    "OpenAIProvider",
    "SchemaRejectedError",
    "SchemaValidationError",
    "SystemSpecs",
    "TokenUsage",
    "benchmark_endpoint",
    "get_active_profile",
    "get_system_specs",
    "list_profiles",
    "list_server_models",
    "load_profile",
    "recommend_tier",
    "set_active_profile",
]
