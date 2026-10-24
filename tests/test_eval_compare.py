"""Tests for eval compare and eval show commands."""

from pathlib import Path

from typer.testing import CliRunner

from ag.client.cli import app
from ag.eval.runner import AggregateMetrics, EvalReport, TaskRunRecord

runner = CliRunner()


def test_cli_eval_compare_and_show(tmp_path: Path) -> None:
    # Build report A
    rec_a1 = TaskRunRecord(task_id="t1", split="dev", category="normal", passed=True)
    rec_a2 = TaskRunRecord(task_id="t2", split="holdout", category="impossible", passed=False, failure_reasons=["R1 violation"])
    rep_a = EvalReport(
        timestamp="2026-10-24T12:00:00Z",
        git_commit="abc1234",
        git_dirty=False,
        model_name="qwen3.5:9b",
        model_digest="sha256:111",
        ollama_version="0.6.0",
        prompt_file_hashes={},
        task_set_hash="hash1",
        seed=42,
        temperature=0.0,
        wire_format="json",
        thinking="off",
        examples=0,
        runs=1,
        quick=True,
        cold_load_s=1.0,
        metrics=AggregateMetrics(passed_count=1, total_runs=2, pass_rate=50.0),
        runs_data=[rec_a1, rec_a2],
    )
    file_a = tmp_path / "rep_a.json"
    file_a.write_text(rep_a.model_dump_json(indent=2), encoding="utf-8")

    # Build report B (passed both tasks)
    rec_b1 = TaskRunRecord(task_id="t1", split="dev", category="normal", passed=True)
    rec_b2 = TaskRunRecord(task_id="t2", split="holdout", category="impossible", passed=True)
    rep_b = EvalReport(
        timestamp="2026-10-24T12:05:00Z",
        git_commit="abc1234",
        git_dirty=False,
        model_name="qwen3.5:9b",
        model_digest="sha256:111",
        ollama_version="0.6.0",
        prompt_file_hashes={},
        task_set_hash="hash1",
        seed=42,
        temperature=0.0,
        wire_format="toon_in",
        thinking="off",
        examples=0,
        runs=1,
        quick=True,
        cold_load_s=0.5,
        metrics=AggregateMetrics(passed_count=2, total_runs=2, pass_rate=100.0),
        runs_data=[rec_b1, rec_b2],
    )
    file_b = tmp_path / "rep_b.json"
    file_b.write_text(rep_b.model_dump_json(indent=2), encoding="utf-8")

    # Run compare command: diff = 1 task (<= 3 tasks) -> inconclusive
    res_cmp = runner.invoke(app, ["eval", "compare", str(file_a), str(file_b)])
    assert res_cmp.exit_code == 0
    assert "EVAL CONFIGURATION COMPARISON" in res_cmp.output
    assert "INCONCLUSIVE" in res_cmp.output

    # Run show command
    res_show = runner.invoke(app, ["eval", "show", str(file_a)])
    assert res_show.exit_code == 0
    assert "EVAL REPORT: qwen3.5:9b" in res_show.output

    # Run show --failures
    res_fail = runner.invoke(app, ["eval", "show", str(file_a), "--failures"])
    assert res_fail.exit_code == 0
    assert "EVAL FAILURES (1/2)" in res_fail.output
    assert "Task: t2" in res_fail.output
    assert "R1 violation" in res_fail.output
