"""Eval reporting, terminal tables, comparison with noise threshold, and failure inspector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ag.eval.runner import EvalReport


def get_default_results_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent / "evals" / "results"


def save_report(report: EvalReport, results_dir: Path | None = None) -> Path:
    rdir = results_dir or get_default_results_dir()
    rdir.mkdir(parents=True, exist_ok=True)
    clean_ts = report.timestamp.replace(":", "-").replace(".", "_")
    filename = f"{report.model_name.replace(':', '_')}_{report.wire_format}_{clean_ts}.json"
    dest = rdir / filename
    dest.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return dest


def load_report(path: Path | str) -> EvalReport:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Report file not found: {p}")
    return EvalReport.model_validate_json(p.read_text(encoding="utf-8"))


def format_eval_summary_table(report: EvalReport) -> str:
    m = report.metrics
    lines = []
    lines.append("=" * 75)
    lines.append(f"EVAL REPORT: {report.model_name} ({report.wire_format}) | Thinking: {report.thinking} | Examples: {report.examples}")
    lines.append(f"Timestamp:   {report.timestamp} | Commit: {report.git_commit[:8]} (dirty: {report.git_dirty})")
    lines.append(f"Model Digest:{report.model_digest} | Ollama: {report.ollama_version}")
    lines.append("-" * 75)
    lines.append(f"Pass Rate:          {m.passed_count}/{m.total_runs} ({m.pass_rate}%)")
    lines.append(f"  - Dev Split:      {m.dev_passed}/{m.dev_total} ({round((m.dev_passed/m.dev_total)*100.0, 1) if m.dev_total else 0}%)")
    lines.append(f"  - Holdout Split:  {m.holdout_passed}/{m.holdout_total} ({round((m.holdout_passed/m.holdout_total)*100.0, 1) if m.holdout_total else 0}%)")
    lines.append(f"First-Tool Accuracy:{m.first_tool_accuracy_count}/{m.total_runs} ({m.first_tool_accuracy_rate}%)")
    lines.append(f"Arg Accuracy:       {m.arg_accuracy_count}/{m.total_runs} ({m.arg_accuracy_rate}%)")
    lines.append(f"Laya Gate Mode:     {getattr(report, 'laya_mode', 'rules')}")
    lines.append(f"Laya Catches:       {getattr(m, 'laya_catches', 0)} | Over-block Rate: {getattr(m, 'over_block_rate', 0.0) * 100:.1f}%")
    lines.append(f"Auditor Breakdown:  Invented {getattr(m, 'total_invented', m.total_hallucination_leaks)} | Tainted {getattr(m, 'total_tainted', 0)} | Grounded {getattr(m, 'total_grounded', 0)}")
    lines.append(f"Hallucination Leaks:{m.total_hallucination_leaks}")
    lines.append("-" * 75)
    lines.append("Category Breakdown:")
    for cat, counts in sorted(m.category_counts.items()):
        p = counts["passed"]
        tot = counts["total"]
        pct = round((p / tot) * 100.0, 1) if tot else 0.0
        lines.append(f"  {cat.ljust(15)} {p}/{tot} ({pct}%)")
    lines.append("-" * 75)
    lines.append(f"Consistency:        Pass {m.consistency.get('consistent_pass', 0)} | Flaky {m.consistency.get('flaky', 0)} | Fail {m.consistency.get('consistent_fail', 0)}")
    lines.append(f"First Step Latency: Avg {m.avg_first_step_latency_ms}ms | p50 {m.latency_p50_ms}ms | p95 {m.latency_p95_ms}ms")
    lines.append(f"Throughput:         Content {m.content_tok_s} tok/s | Reasoning {m.reasoning_tok_s} tok/s")
    lines.append(f"Peak Process RAM:   {m.peak_process_ram_mb} MB | Peak GPU VRAM Delta: {m.peak_gpu_delta_mb} MB")
    lines.append(f"Stop Reasons:       {m.stop_reasons}")
    lines.append(f"Cold Load Latency:  {report.cold_load_s}s")
    lines.append("=" * 75)
    return "\n".join(lines)


def compare_summaries(
    config_a_name: str,
    config_b_name: str,
    results_a: dict[str, bool],
    results_b: dict[str, bool],
    total_tasks: int,
) -> dict[str, Any]:
    """Compare two configuration results per task and check for noise threshold."""
    a_passed = {tid for tid, p in results_a.items() if p}
    b_passed = {tid for tid, p in results_b.items() if p}

    a_only = a_passed - b_passed
    b_only = b_passed - a_passed
    net_diff = abs(len(a_only) - len(b_only))

    # Noise threshold: net difference <= 3 tasks (< 10 points on 30 tasks) is inconclusive
    is_inconclusive = net_diff <= 3

    if is_inconclusive:
        verdict = f"INCONCLUSIVE (net difference {net_diff} tasks is within noise threshold <=3 tasks)"
    elif len(b_passed) > len(a_passed):
        verdict = f"{config_b_name} outperforms {config_a_name} by +{net_diff} tasks (statistically significant)"
    else:
        verdict = f"{config_a_name} outperforms {config_b_name} by +{net_diff} tasks (statistically significant)"

    return {
        "config_a": config_a_name,
        "config_b": config_b_name,
        "total_tasks": total_tasks,
        "a_passed_count": len(a_passed),
        "b_passed_count": len(b_passed),
        "a_only_tasks": sorted(a_only),
        "b_only_tasks": sorted(b_only),
        "net_diff": net_diff,
        "inconclusive": is_inconclusive,
        "verdict": verdict,
    }


def compare_reports(rep_a: EvalReport, rep_b: EvalReport) -> str:
    # Build per-task passed maps
    res_a: dict[str, bool] = {}
    for r in rep_a.runs_data:
        res_a[r.task_id] = res_a.get(r.task_id, False) or r.passed

    res_b: dict[str, bool] = {}
    for r in rep_b.runs_data:
        res_b[r.task_id] = res_b.get(r.task_id, False) or r.passed

    name_a = f"{rep_a.model_name} ({rep_a.wire_format}, th={rep_a.thinking}, ex={rep_a.examples})"
    name_b = f"{rep_b.model_name} ({rep_b.wire_format}, th={rep_b.thinking}, ex={rep_b.examples})"

    cmp_res = compare_summaries(
        config_a_name=name_a,
        config_b_name=name_b,
        results_a=res_a,
        results_b=res_b,
        total_tasks=len(set(res_a.keys()) | set(res_b.keys())),
    )

    ma = rep_a.metrics
    mb = rep_b.metrics

    lines = []
    lines.append("=" * 80)
    lines.append("EVAL CONFIGURATION COMPARISON")
    lines.append("=" * 80)
    lines.append(f"Config A: {name_a}")
    lines.append(f"Config B: {name_b}")
    lines.append("-" * 80)
    lines.append(f"{'Metric':<25} {'Config A':<25} {'Config B':<25}")
    lines.append("-" * 80)
    lines.append(f"{'Pass Count':<25} {f'{ma.passed_count}/{ma.total_runs} ({ma.pass_rate}%)':<25} {f'{mb.passed_count}/{mb.total_runs} ({mb.pass_rate}%)':<25}")
    lines.append(f"{'  Dev Split':<25} {f'{ma.dev_passed}/{ma.dev_total}':<25} {f'{mb.dev_passed}/{mb.dev_total}':<25}")
    lines.append(f"{'  Holdout Split':<25} {f'{ma.holdout_passed}/{ma.holdout_total}':<25} {f'{mb.holdout_passed}/{mb.holdout_total}':<25}")
    lines.append(f"{'First-Tool Acc':<25} {f'{ma.first_tool_accuracy_rate}%':<25} {f'{mb.first_tool_accuracy_rate}%':<25}")
    lines.append(f"{'Arg Accuracy':<25} {f'{ma.arg_accuracy_rate}%':<25} {f'{mb.arg_accuracy_rate}%':<25}")
    lines.append(f"{'Hallucination Leaks':<25} {f'{ma.total_hallucination_leaks}':<25} {f'{mb.total_hallucination_leaks}':<25}")
    lines.append(f"{'First Step Latency':<25} {f'{ma.avg_first_step_latency_ms} ms':<25} {f'{mb.avg_first_step_latency_ms} ms':<25}")
    lines.append(f"{'Content Tok/s':<25} {f'{ma.content_tok_s}':<25} {f'{mb.content_tok_s}':<25}")
    lines.append(f"{'Total Prompt Tokens':<25} {f'{ma.total_prompt_tokens}':<25} {f'{mb.total_prompt_tokens}':<25}")
    lines.append("-" * 80)
    lines.append(f"A-only passes ({len(cmp_res['a_only_tasks'])}): {', '.join(cmp_res['a_only_tasks']) or 'none'}")
    lines.append(f"B-only passes ({len(cmp_res['b_only_tasks'])}): {', '.join(cmp_res['b_only_tasks']) or 'none'}")
    lines.append(f"Net Difference: {cmp_res['net_diff']} tasks")
    lines.append(f"Verdict: {cmp_res['verdict']}")
    lines.append("=" * 80)
    return "\n".join(lines)


def format_failures(report: EvalReport) -> str:
    failures = [r for r in report.runs_data if not r.passed]
    if not failures:
        return f"\nNo failures! All {len(report.runs_data)} runs passed successfully.\n"

    lines = []
    lines.append("=" * 80)
    lines.append(f"EVAL FAILURES ({len(failures)}/{len(report.runs_data)}): {report.model_name} ({report.wire_format})")
    lines.append("=" * 80)
    for r in failures:
        lines.append(f"\n[Task: {r.task_id}] (Category: {r.category}, Split: {r.split})")
        lines.append(f"  Prompt:   {r.prompt}")
        lines.append(f"  Stop:     {r.stop_reason}")
        lines.append(f"  Reasons:  {', '.join(r.failure_reasons)}")
        if r.summary:
            lines.append(f"  Summary:  {r.summary}")
        if r.question:
            lines.append(f"  Question: {r.question}")
        if r.hallucination_leaks:
            lines.append(f"  Leaks:    {r.hallucination_leaks}")
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)
