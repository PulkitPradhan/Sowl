"""ag doctor hardware detection, benchmarking, and tier recommendation."""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any

import httpx
import psutil
from pydantic import BaseModel


class SystemSpecs(BaseModel):
    cpu_cores: int
    ram_gb: float
    gpu_name: str
    vram_gb: float | None = None


class BenchmarkResult(BaseModel):
    ttft_ms: float
    tokens_per_sec: float
    tokens_generated: int
    server_tokens_per_sec: float | None = None
    server_usage: dict[str, Any] | None = None


def recommend_tier(vram_gb: float | None, ram_gb: float, cores: int) -> int:
    """Pure function recommending model tier based on system hardware specifications.

    Tiers:
    - Tier 0: No-LLM / rule-based fallback (<8GB RAM, no GPU)
    - Tier 1: CPU 1-4B (>=8GB RAM, no GPU)
    - Tier 2: 8GB VRAM (6-10GB VRAM, e.g. RTX 5060 8GB)
    - Tier 3: 16GB+ VRAM (>=14GB VRAM)
    """
    if vram_gb is not None and vram_gb >= 14.0:
        return 3
    if vram_gb is not None and vram_gb >= 6.0:
        return 2
    if ram_gb >= 8.0:
        return 1
    return 0


def get_system_specs() -> SystemSpecs:
    """Detect RAM, CPU cores, and NVIDIA GPU/VRAM cross-platform via psutil and nvidia-smi."""
    vm = psutil.virtual_memory()
    ram_gb = round(vm.total / (1024**3), 2)
    cpu_cores = psutil.cpu_count(logical=True) or 1

    gpu_name = "no NVIDIA GPU found"
    vram_gb: float | None = None

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            output = subprocess.check_output(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=3.0,
                stderr=subprocess.DEVNULL,
            ).strip()
            if output:
                # Format: "NVIDIA GeForce RTX 5060 Laptop GPU, 8151"
                parts = [p.strip() for p in output.splitlines()[0].split(",")]
                if len(parts) >= 2:
                    gpu_name = parts[0]
                    vram_mb = float(parts[1])
                    vram_gb = round(vram_mb / 1024.0, 2)
        except (subprocess.SubprocessError, OSError, ValueError):
            # Missing nvidia-smi, driver error, or timeout: never crash
            gpu_name = "no NVIDIA GPU found"
            vram_gb = None

    return SystemSpecs(
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
    )


async def list_server_models(
    endpoint: str = "http://localhost:11434/v1",
    client: httpx.AsyncClient | None = None,
) -> list[str]:
    """Query /v1/models on the server to list installed models."""
    url = f"{endpoint.rstrip('/')}/models"
    if not url.endswith("/v1/models") and not url.endswith("/models"):
        url = f"{endpoint.rstrip('/')}/v1/models"

    async def _fetch(c: httpx.AsyncClient) -> list[str]:
        try:
            resp = await c.get(url, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("id") for m in data.get("data", []) if m.get("id")]
        except (httpx.HTTPError, ValueError):
            return []
        return []

    if client is not None:
        return await _fetch(client)

    async with httpx.AsyncClient() as c:
        return await _fetch(c)


async def benchmark_endpoint(
    endpoint: str = "http://localhost:11434/v1",
    model: str = "",
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> BenchmarkResult:
    """Benchmark model endpoint with warm-up, fixed prompt, temperature 0, and token timing."""
    url = f"{endpoint.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    fixed_prompt = "Count from 1 to 30, one number per line."

    async def _run_benchmark(c: httpx.AsyncClient) -> BenchmarkResult:
        # 1. Warm-up run (discarded)
        warmup_payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "1"}],
            "max_tokens": 5,
            "temperature": 0.0,
        }
        try:
            await c.post(url, json=warmup_payload, headers=headers, timeout=10.0)
        except Exception:  # noqa: BLE001, S110
            pass

        # 2. Benchmark run
        bench_payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": fixed_prompt}],
            "max_tokens": 100,
            "temperature": 0.0,
        }

        start_time = time.perf_counter()
        resp = await c.post(url, json=bench_payload, headers=headers, timeout=60.0)
        total_time = time.perf_counter() - start_time

        if resp.status_code != 200:
            raise RuntimeError(f"Benchmark call failed [{resp.status_code}]: {resp.text}")

        data = resp.json()
        usage_data = data.get("usage", {})
        completion_tokens = usage_data.get("completion_tokens", 0)

        # Estimate TTFT as initial response time portion
        ttft_ms = round(total_time * 1000.0 * 0.25, 2)  # conservative estimate
        tokens_per_sec = round(completion_tokens / total_time, 2) if total_time > 0 else 0.0

        return BenchmarkResult(
            ttft_ms=ttft_ms,
            tokens_per_sec=tokens_per_sec,
            tokens_generated=completion_tokens,
            server_usage=usage_data,
        )

    if client is not None:
        return await _run_benchmark(client)

    async with httpx.AsyncClient() as c:
        return await _run_benchmark(c)
