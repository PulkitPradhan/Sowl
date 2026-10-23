"""Tests for trace logger, permissions, 200 file retention, and ag trace last."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ag.client.cli import app
from ag.planner.trace import TraceLogger, list_traces


def test_trace_permissions_and_retention(tmp_path: Path) -> None:
    traces_dir = tmp_path / "traces"
    logger = TraceLogger(traces_dir=traces_dir)

    # Write a trace
    logger.log_init(run_id="run_1", task="Test task 1", profile_name="test_p")
    logger.log_step(run_id="run_1", step_idx=1, action={"kind": "tool", "tool": "files.read"}, observation_summary="Read 50 bytes", latency_ms=10.0, usage={"total_tokens": 15})
    logger.log_done(run_id="run_1", stop_reason="done", summary="All good", total_steps=1, duration_s=0.1)

    # Check directory permissions: 0700
    stat_dir = traces_dir.stat()
    assert (stat_dir.st_mode & 0o777) == 0o700

    # Check file permissions: 0600
    trace_file = traces_dir / "run_1.jsonl"
    assert trace_file.is_file()
    stat_file = trace_file.stat()
    assert (stat_file.st_mode & 0o777) == 0o600

    # Test retention: write 210 traces -> only the latest 200 are kept
    for i in range(2, 212):
        logger.log_init(run_id=f"run_{i}", task=f"Task {i}", profile_name="test_p")
        logger.log_done(run_id=f"run_{i}", stop_reason="done", summary="Done", total_steps=1, duration_s=0.01)

    all_files = list_traces(traces_dir)
    assert len(all_files) == 200
    # Oldest run_1 should have been pruned
    assert not trace_file.exists()


def test_trace_cli_show_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    traces_dir = tmp_path / "traces"
    logger = TraceLogger(traces_dir=traces_dir)
    logger.log_init(run_id="last_run", task="Read secret config", profile_name="tier1_cpu")
    logger.log_step(run_id="last_run", step_idx=1, action={"kind": "done", "summary": "Done"}, observation_summary="", latency_ms=5.0, usage={"total_tokens": 5})
    logger.log_done(run_id="last_run", stop_reason="done", summary="Finished successfully", total_steps=1, duration_s=0.2)

    monkeypatch.setenv("AG_WORKSPACE_DIR", str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(app, ["trace", "last"])
    assert result.exit_code == 0
    assert "last_run" in result.output
    assert "Finished successfully" in result.output
