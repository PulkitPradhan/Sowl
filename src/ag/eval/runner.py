"""In-process eval execution runner, warm-up probe, and metrics aggregation."""

from __future__ import annotations

import hashlib
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from ag.eval.grounding import audit_grounding
from ag.eval.resources import ProcessGpuSampler
from ag.eval.sandbox import EvalSandbox
from ag.eval.task import EvalTask, compute_task_set_hash, load_tasks
from ag.models.profile import load_profile
from ag.models.provider import ModelProvider, OpenAIProvider
from ag.planner.executor import Executor
from ag.planner.trace import TraceLogger
from ag.skills.registry import SkillRegistry


class TaskRunRecord(BaseModel):
    task_id: str
    run_idx: int = 1
    split: str = "dev"
    category: str = "normal"
    prompt: str = ""
    passed: bool = False
    failure_reasons: list[str] = Field(default_factory=list)
    first_tool_accuracy: bool = False
    arg_accuracy: bool = False
    hallucination_leaks: list[str] = Field(default_factory=list)
    steps: int = 0
    llm_calls: int = 0
    first_step_latency_ms: float = 0.0
    duration_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    content_tok_s: float = 0.0
    reasoning_tok_s: float = 0.0
    stop_reason: str = "unknown"
    summary: str = ""
    question: str | None = None


class AggregateMetrics(BaseModel):
    total_runs: int = 0
    passed_count: int = 0
    pass_rate: float = 0.0

    dev_passed: int = 0
    dev_total: int = 0
    holdout_passed: int = 0
    holdout_total: int = 0

    first_tool_accuracy_count: int = 0
    first_tool_accuracy_rate: float = 0.0
    arg_accuracy_count: int = 0
    arg_accuracy_rate: float = 0.0

    total_hallucination_leaks: int = 0
    category_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    consistency: dict[str, int] = Field(default_factory=dict)

    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    avg_first_step_latency_ms: float = 0.0

    total_prompt_tokens: int = 0
    total_content_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_cached_tokens: int = 0

    content_tok_s: float = 0.0
    reasoning_tok_s: float = 0.0

    peak_process_ram_mb: float = 0.0
    peak_gpu_delta_mb: float = 0.0

    stop_reasons: dict[str, int] = Field(default_factory=dict)


class EvalReport(BaseModel):
    timestamp: str
    git_commit: str
    git_dirty: bool
    model_name: str
    model_digest: str
    ollama_version: str
    prompt_file_hashes: dict[str, str]
    task_set_hash: str
    seed: int
    temperature: float
    wire_format: str
    thinking: str
    examples: int
    runs: int
    quick: bool
    cold_load_s: float
    metrics: AggregateMetrics
    runs_data: list[TaskRunRecord]


def calculate_aggregate_metrics(records: list[TaskRunRecord]) -> AggregateMetrics:
    if not records:
        return AggregateMetrics()

    total = len(records)
    passed = sum(1 for r in records if r.passed)
    pass_rate = round((passed / total) * 100.0, 1) if total else 0.0

    dev_recs = [r for r in records if r.split == "dev"]
    holdout_recs = [r for r in records if r.split == "holdout"]

    dev_passed = sum(1 for r in dev_recs if r.passed)
    holdout_passed = sum(1 for r in holdout_recs if r.passed)

    ft_acc_cnt = sum(1 for r in records if r.first_tool_accuracy)
    arg_acc_cnt = sum(1 for r in records if r.arg_accuracy)

    all_leaks = sum(len(r.hallucination_leaks) for r in records)

    # Categories breakdown
    categories: dict[str, dict[str, int]] = {}
    for r in records:
        if r.category not in categories:
            categories[r.category] = {"passed": 0, "total": 0}
        categories[r.category]["total"] += 1
        if r.passed:
            categories[r.category]["passed"] += 1

    # Consistency across task repeats
    task_runs: dict[str, list[bool]] = {}
    for r in records:
        task_runs.setdefault(r.task_id, []).append(r.passed)

    consistency = {"consistent_pass": 0, "flaky": 0, "consistent_fail": 0}
    for pass_list in task_runs.values():
        pass_cnt = sum(1 for p in pass_list if p)
        if pass_cnt == len(pass_list):
            consistency["consistent_pass"] += 1
        elif pass_cnt == 0:
            consistency["consistent_fail"] += 1
        else:
            consistency["flaky"] += 1

    # Latencies
    step_latencies = [r.first_step_latency_ms for r in records if r.first_step_latency_ms > 0]
    p50 = float(statistics.median(step_latencies)) if step_latencies else 0.0
    p95 = float(statistics.quantiles(step_latencies, n=20)[18]) if len(step_latencies) >= 20 else (max(step_latencies) if step_latencies else 0.0)
    avg_first_step = round(statistics.mean(step_latencies), 1) if step_latencies else 0.0

    # Tokens and throughput
    total_prompt = sum(r.prompt_tokens for r in records)
    total_content = sum(r.completion_tokens for r in records)
    total_reasoning = sum(r.reasoning_tokens for r in records)
    total_cached = sum(r.cached_tokens for r in records)
    total_time = sum(r.duration_s for r in records)

    content_tok_s = round(total_content / total_time, 2) if total_time > 0 else 0.0
    reasoning_tok_s = round(total_reasoning / total_time, 2) if total_time > 0 else 0.0

    stop_reasons: dict[str, int] = {}
    for r in records:
        stop_reasons[r.stop_reason] = stop_reasons.get(r.stop_reason, 0) + 1

    return AggregateMetrics(
        total_runs=total,
        passed_count=passed,
        pass_rate=pass_rate,
        dev_passed=dev_passed,
        dev_total=len(dev_recs),
        holdout_passed=holdout_passed,
        holdout_total=len(holdout_recs),
        first_tool_accuracy_count=ft_acc_cnt,
        first_tool_accuracy_rate=round((ft_acc_cnt / total) * 100.0, 1),
        arg_accuracy_count=arg_acc_cnt,
        arg_accuracy_rate=round((arg_acc_cnt / total) * 100.0, 1),
        total_hallucination_leaks=all_leaks,
        category_counts=categories,
        consistency=consistency,
        latency_p50_ms=round(p50, 1),
        latency_p95_ms=round(p95, 1),
        avg_first_step_latency_ms=avg_first_step,
        total_prompt_tokens=total_prompt,
        total_content_tokens=total_content,
        total_reasoning_tokens=total_reasoning,
        total_cached_tokens=total_cached,
        content_tok_s=content_tok_s,
        reasoning_tok_s=reasoning_tok_s,
        stop_reasons=stop_reasons,
    )


def compute_prompt_file_hashes() -> dict[str, str]:
    prompts_dir = Path(__file__).resolve().parent.parent / "planner" / "prompts"
    hashes = {}
    if prompts_dir.is_dir():
        for p in sorted(prompts_dir.glob("*.txt")):
            hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def get_git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"

    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        dirty = bool(status)
    except Exception:  # noqa: BLE001
        dirty = False

    return commit, dirty


async def get_ollama_metadata(endpoint: str, model_name: str) -> tuple[str, str]:
    version = "unknown"
    digest = "unknown"
    base = endpoint.replace("/v1", "")

    async with httpx.AsyncClient(timeout=3.0) as client:
        try:
            v_res = await client.get(f"{base}/api/version")
            if v_res.status_code == 200:
                version = v_res.json().get("version", "unknown")
        except Exception:  # noqa: BLE001, S110
            pass

        try:
            m_res = await client.post(f"{base}/api/show", json={"name": model_name})
            if m_res.status_code == 200:
                details = m_res.json()
                digest = details.get("digest") or details.get("details", {}).get("parent_model", "unknown")
        except Exception:  # noqa: BLE001, S110
            pass

    return version, digest


class EvalRunner:
    def __init__(
        self,
        profile_name: str,
        wire_format: str | None = None,
        thinking: str | None = None,
        examples: int | None = None,
        runs: int = 1,
        quick: bool = False,
        endpoint: str | None = None,
        monitor_process: str = "ollama",
        provider: ModelProvider | None = None,
    ) -> None:
        self.profile = load_profile(profile_name)
        if wire_format:
            self.profile.wire_format = wire_format
        if thinking:
            self.profile.thinking = thinking  # type: ignore[assignment]
        if examples is not None:
            self.profile.examples = examples
        if endpoint:
            self.profile.endpoint = endpoint

        self.runs = runs
        self.quick = quick
        self.monitor_process = monitor_process
        self.provider = provider or OpenAIProvider(self.profile)

    async def warm_up(self) -> float:
        """Issue a warm-up call and return cold load latency in seconds."""
        t0 = time.perf_counter()
        try:
            await self.provider.chat([{"role": "user", "content": "ping"}], schema=None)
        except Exception:  # noqa: BLE001, S110
            pass
        return round(time.perf_counter() - t0, 2)

    async def run_task(
        self,
        task: EvalTask,
        run_idx: int,
        temperature: float,
        seed: int,
        sampler: ProcessGpuSampler,
        base_results_dir: Path | None = None,
    ) -> TaskRunRecord:
        registry = SkillRegistry()
        registry.discover()
        all_tools = registry.get_tools()

        record = TaskRunRecord(
            task_id=task.id,
            run_idx=run_idx,
            split=task.split,
            category=task.category,
            prompt=task.prompt,
        )

        with EvalSandbox(task, base_results_dir=base_results_dir) as sandbox:
            trace_logger = TraceLogger(traces_dir=sandbox.traces_dir)
            hallucination_leaks: list[str] = []

            # Event handler to intercept proposed actions for grounding auditing
            async def event_handler(event: dict[str, Any]) -> None:
                if event.get("type") == "step":
                    act = event.get("action", {})
                    if act.get("kind") == "tool":
                        tool_name = act.get("tool", "")
                        args = act.get("args", {})
                        if tool_name in all_tools:
                            leaks = audit_grounding(task.prompt, [], all_tools[tool_name], args)
                            hallucination_leaks.extend(leaks)

            executor = Executor(
                profile=self.profile,
                registry=registry,
                provider=self.provider,
                confirm_handler=sandbox.get_confirm_handler(),
                event_handler=event_handler,
                trace_logger=trace_logger,
                max_steps=6,
            )

            # Set temperature and seed on profile for this run
            self.profile.temperature = temperature

            start_t = time.perf_counter()
            exec_result = await executor.run(task.prompt, run_id=f"{task.id}_run{run_idx}")
            total_duration = time.perf_counter() - start_t

            # Success and expected evaluation
            passed, reasons = sandbox.evaluate_result(exec_result)

            # Record stats
            record.passed = passed
            record.failure_reasons = reasons
            record.hallucination_leaks = hallucination_leaks
            record.steps = exec_result.total_steps
            record.duration_s = round(total_duration, 2)
            record.stop_reason = exec_result.stop_reason
            record.summary = exec_result.summary
            record.question = exec_result.question

            if exec_result.steps:
                first_step = exec_result.steps[0]
                record.first_step_latency_ms = first_step.latency_ms
                if task.expected.get("first_tool"):
                    record.first_tool_accuracy = (first_step.action.get("tool") == task.expected.get("first_tool"))
                if task.expected.get("first_args"):
                    actual_args = first_step.action.get("args", {})
                    record.arg_accuracy = all(
                        actual_args.get(k) == v for k, v in task.expected["first_args"].items()
                    )

                # Token aggregation across steps
                for s in exec_result.steps:
                    u = s.usage
                    record.prompt_tokens += u.get("prompt_tokens", 0)
                    record.completion_tokens += u.get("completion_tokens", 0)
                    record.reasoning_tokens += u.get("reasoning_tokens", 0)
                    record.cached_tokens += u.get("cached_tokens", 0)

            if total_duration > 0:
                record.content_tok_s = round(record.completion_tokens / total_duration, 2)
                record.reasoning_tok_s = round(record.reasoning_tokens / total_duration, 2)

        return record

    async def run_suite(self, base_results_dir: Path | None = None) -> EvalReport:
        tasks = load_tasks(quick=self.quick)
        cold_load_s = await self.warm_up()

        sampler = ProcessGpuSampler(process_name=self.monitor_process)
        sampler.start()

        run_records: list[TaskRunRecord] = []
        temperature = 0.0 if self.runs == 1 else 0.3
        base_seed = 42

        try:
            for task in tasks:
                for r_idx in range(1, self.runs + 1):
                    current_seed = base_seed + r_idx
                    rec = await self.run_task(
                        task=task,
                        run_idx=r_idx,
                        temperature=temperature,
                        seed=current_seed,
                        sampler=sampler,
                        base_results_dir=base_results_dir,
                    )
                    run_records.append(rec)
        finally:
            res_stats = sampler.stop()

        metrics = calculate_aggregate_metrics(run_records)
        metrics.peak_process_ram_mb = res_stats.peak_process_ram_mb
        metrics.peak_gpu_delta_mb = res_stats.peak_gpu_delta_mb

        # Gather metadata
        git_commit, git_dirty = get_git_info()
        ollama_ver, model_digest = await get_ollama_metadata(self.profile.endpoint, self.profile.model)
        prompt_hashes = compute_prompt_file_hashes()
        task_hash = compute_task_set_hash(tasks)

        report = EvalReport(
            timestamp=datetime.now(UTC).isoformat(),
            git_commit=git_commit,
            git_dirty=git_dirty,
            model_name=self.profile.model,
            model_digest=model_digest,
            ollama_version=ollama_ver,
            prompt_file_hashes=prompt_hashes,
            task_set_hash=task_hash,
            seed=base_seed,
            temperature=temperature,
            wire_format=self.profile.wire_format,
            thinking=self.profile.thinking,
            examples=self.profile.examples,
            runs=self.runs,
            quick=self.quick,
            cold_load_s=cold_load_s,
            metrics=metrics,
            runs_data=run_records,
        )

        return report
