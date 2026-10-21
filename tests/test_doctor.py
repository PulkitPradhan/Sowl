"""Tests for ag doctor hardware detection, benchmarking, and tier recommendation."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from ag.models.doctor import (
    BenchmarkResult,
    benchmark_endpoint,
    get_system_specs,
    list_server_models,
    recommend_tier,
)


@pytest.mark.parametrize(
    ("vram_gb", "ram_gb", "cores", "expected_tier"),
    [
        # Low RAM, no GPU -> Tier 0 (stub / rule fallback)
        (None, 4.0, 2, 0),
        (0.0, 6.0, 4, 0),
        (None, 7.9, 4, 0),
        # Decent RAM, no GPU -> Tier 1 (CPU 1-4B)
        (None, 8.0, 4, 1),
        (None, 16.0, 8, 1),
        (None, 32.0, 16, 1),
        (0.0, 16.0, 8, 1),
        # Dedicated GPU with 6GB-10GB VRAM (e.g. RTX 5060 8GB) -> Tier 2
        (6.0, 16.0, 8, 2),
        (8.0, 16.0, 8, 2),
        (10.0, 32.0, 16, 2),
        # High VRAM (14GB+) -> Tier 3
        (14.0, 32.0, 16, 3),
        (16.0, 32.0, 16, 3),
        (24.0, 64.0, 24, 3),
    ],
)
def test_recommend_tier_pure_function(
    vram_gb: float | None,
    ram_gb: float,
    cores: int,
    expected_tier: int,
):
    tier = recommend_tier(vram_gb=vram_gb, ram_gb=ram_gb, cores=cores)
    assert tier == expected_tier


def test_get_system_specs_missing_nvidia_smi():
    with (
        patch("shutil.which", return_value=None),
        patch("psutil.virtual_memory") as mock_vm,
        patch("psutil.cpu_count", return_value=8),
    ):
        mock_vm.return_value.total = 16 * 1024 * 1024 * 1024  # 16 GB
        specs = get_system_specs()
        assert specs.ram_gb == pytest.approx(16.0, 0.1)
        assert specs.cpu_cores == 8
        assert specs.gpu_name == "no NVIDIA GPU found"
        assert specs.vram_gb is None


def test_get_system_specs_with_nvidia_smi():
    fake_smi_output = "NVIDIA GeForce RTX 5060 Laptop GPU, 8151\n"
    with (
        patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch("subprocess.check_output", return_value=fake_smi_output),
        patch("psutil.virtual_memory") as mock_vm,
        patch("psutil.cpu_count", return_value=16),
    ):
        mock_vm.return_value.total = 24 * 1024 * 1024 * 1024
        specs = get_system_specs()
        assert specs.gpu_name == "NVIDIA GeForce RTX 5060 Laptop GPU"
        assert specs.vram_gb == pytest.approx(7.96, 0.1)


@pytest.mark.anyio
async def test_list_server_models_mock_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        data = {
            "data": [
                {"id": "qwen2.5:7b"},
                {"id": "llama3.2:3b"},
            ]
        }
        return httpx.Response(200, json=data)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://fake.local") as client:
        models = await list_server_models(client=client)
        assert models == ["qwen2.5:7b", "llama3.2:3b"]


@pytest.mark.anyio
async def test_benchmark_endpoint_mock_transport():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = json.loads(request.content.decode("utf-8"))
        assert body["temperature"] == 0.0
        assert body["max_tokens"] == 100

        res = {
            "choices": [{"message": {"role": "assistant", "content": "1\n2\n3\n4\n5"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
        return httpx.Response(200, json=res)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://fake.local") as client:
        res = await benchmark_endpoint(client=client, model="fake-model")
        assert isinstance(res, BenchmarkResult)
        assert res.tokens_generated == 20
        assert res.ttft_ms >= 0.0
        assert res.tokens_per_sec > 0.0
        # Check warm-up was called first (so total 2 requests: 1 warm-up, 1 benchmark)
        assert call_count == 2
