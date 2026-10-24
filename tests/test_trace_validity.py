"""Tests for trace record validity, JSON compliance, and confirmation source logging."""

import json
from pathlib import Path

from ag.planner.trace import TraceLogger


def test_trace_json_validity_and_fields(tmp_path: Path) -> None:
    logger = TraceLogger(traces_dir=tmp_path)
    run_id = "test_run_123"

    logger.log_init(
        run_id=run_id,
        task="Test task",
        profile_name="tier1_cpu",
        wire_format="toon_in",
        endpoint="http://localhost:11434/v1",
    )

    logger.log_confirm_request(
        run_id=run_id,
        step_idx=1,
        tool="browser.open_url",
        args={"url": "https://example.com"},
        risk="R1",
        action_hash="hash123",
    )

    logger.log_confirm_decision(
        run_id=run_id,
        step_idx=1,
        tool="browser.open_url",
        action_hash="hash123",
        approved=True,
        source="test_runner",
    )

    logger.log_step(
        run_id=run_id,
        step_idx=1,
        action={"kind": "tool", "tool": "browser.open_url", "args": {"url": "https://example.com"}},
        observation_summary="Opened URL",
        latency_ms=12.5,
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        finish_reason="stop",
    )

    logger.log_done(
        run_id=run_id,
        stop_reason="done",
        summary="Finished successfully",
        total_steps=1,
        duration_s=0.5,
    )

    trace_file = tmp_path / f"{run_id}.jsonl"
    assert trace_file.is_file()

    # Verify EVERY line is strictly valid JSON
    lines = trace_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5

    parsed_records = []
    for line in lines:
        rec = json.loads(line)  # Will raise JSONDecodeError if invalid
        parsed_records.append(rec)

    # Check init line
    init_rec = parsed_records[0]
    assert init_rec["type"] == "init"
    assert init_rec["wire_format"] == "toon_in"
    assert init_rec["endpoint"] == "http://localhost:11434/v1"

    # Check confirm_request
    req_rec = parsed_records[1]
    assert req_rec["type"] == "confirm_request"
    assert req_rec["tool"] == "browser.open_url"
    assert req_rec["risk"] == "R1"

    # Check confirm_decision
    dec_rec = parsed_records[2]
    assert dec_rec["type"] == "confirm_decision"
    assert dec_rec["approved"] is True
    assert dec_rec["source"] == "test_runner"
