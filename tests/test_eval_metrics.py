"""Tests for eval metric calculations, latency percentiles, and compare noise thresholds."""

from ag.eval.report import compare_summaries
from ag.eval.runner import TaskRunRecord, calculate_aggregate_metrics


def test_metric_calculations() -> None:
    records = [
        TaskRunRecord(
            task_id="t1",
            run_idx=1,
            split="dev",
            category="normal",
            passed=True,
            first_tool_accuracy=True,
            arg_accuracy=True,
            first_step_latency_ms=100.0,
            duration_s=1.0,
            prompt_tokens=50,
            completion_tokens=20,
            reasoning_tokens=0,
            cached_tokens=10,
            stop_reason="done",
        ),
        TaskRunRecord(
            task_id="t2",
            run_idx=1,
            split="holdout",
            category="impossible",
            passed=False,
            first_tool_accuracy=False,
            arg_accuracy=False,
            first_step_latency_ms=300.0,
            duration_s=2.0,
            prompt_tokens=60,
            completion_tokens=40,
            reasoning_tokens=20,
            cached_tokens=0,
            stop_reason="ask",
        ),
    ]

    metrics = calculate_aggregate_metrics(records)
    assert metrics.total_runs == 2
    assert metrics.passed_count == 1
    assert metrics.pass_rate == 50.0

    # Dev and holdout counts separate
    assert metrics.dev_passed == 1
    assert metrics.dev_total == 1
    assert metrics.holdout_passed == 0
    assert metrics.holdout_total == 1

    # Tokens and rates
    assert metrics.total_content_tokens == 60
    assert metrics.total_reasoning_tokens == 20
    assert metrics.content_tok_s > 0


def test_compare_inconclusive_threshold() -> None:
    # 30 tasks total. Config A passes 20 tasks, Config B passes 22 tasks.
    # Net diff = 2 tasks (<= 3 tasks), which is within noise threshold.
    passes_a = {f"t{i}": True for i in range(20)}
    passes_b = {f"t{i}": True for i in range(22)}

    res = compare_summaries(
        config_a_name="json",
        config_b_name="toon_in",
        results_a=passes_a,
        results_b=passes_b,
        total_tasks=30,
    )
    assert res["inconclusive"] is True
    assert "noise threshold" in res["verdict"].lower()

    # Net diff = 5 tasks (> 3 tasks): statistically significant difference
    passes_c = {f"t{i}": True for i in range(26)}
    res_sig = compare_summaries(
        config_a_name="json",
        config_b_name="toon_in",
        results_a=passes_a,
        results_b=passes_c,
        total_tasks=30,
    )
    assert res_sig["inconclusive"] is False
    assert "toon_in" in res_sig["verdict"]
