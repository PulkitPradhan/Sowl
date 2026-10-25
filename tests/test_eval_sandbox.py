"""Tests for eval sandbox isolation, mock launcher, and success checks."""

from pathlib import Path

from ag.eval.sandbox import EvalSandbox
from ag.eval.task import EvalTask
from ag.planner.executor import ExecutorResult, StepRecord


def test_eval_sandbox_does_not_touch_home_ag(tmp_path: Path, monkeypatch) -> None:
    # Setup mock home .ag
    mock_home_ag = tmp_path / ".ag"
    mock_home_ag.mkdir(parents=True)
    traces_dir = mock_home_ag / "traces"
    traces_dir.mkdir(parents=True)
    workspace_dir = mock_home_ag / "workspace"
    workspace_dir.mkdir(parents=True)

    task = EvalTask(
        id="test_01",
        prompt="List files",
        category="normal",
        split="dev",
        setup={"dummy.txt": "dummy content"},
        expected={"first_tool": "files.list"},
        success_check={"stop_reason": "done"},
    )

    with EvalSandbox(task, base_results_dir=tmp_path / "results") as sandbox:
        assert sandbox.workspace_root.is_dir()
        assert (sandbox.workspace_root / "dummy.txt").read_text() == "dummy content"
        # Traces must be directed under base_results_dir/traces/
        assert sandbox.traces_dir.is_relative_to(tmp_path / "results" / "traces")
        assert not sandbox.traces_dir.is_relative_to(traces_dir)

    # Assert mock home .ag was never touched
    assert list(traces_dir.iterdir()) == []
    assert list(workspace_dir.iterdir()) == []


def test_eval_sandbox_success_checks() -> None:
    task = EvalTask(
        id="test_success",
        prompt="Draft an email to alice@example.com",
        category="normal",
        split="dev",
        expected={"first_tool": "gmail.compose"},
        success_check={
            "stop_reason": "done",
            "summary_contains": ["draft", "browser"],
            "no_unexpected_r1_tools": True,
            "allowed_r1_tools": ["gmail.compose"],
        },
    )

    with EvalSandbox(task) as sandbox:
        # Simulate successful executor result
        result = ExecutorResult(
            run_id="run_1",
            stop_reason="done",
            summary="Draft opened in browser successfully.",
            total_steps=1,
            steps=[
                StepRecord(
                    step=1,
                    action={"kind": "tool", "tool": "gmail.compose", "args": {"to": "alice@example.com"}},
                    observation_summary="Draft opened",
                )
            ],
        )
        passed, reasons = sandbox.evaluate_result(result)
        assert passed is True
        assert reasons == []

        # If summary missing expected keywords
        bad_summary_result = ExecutorResult(
            run_id="run_2",
            stop_reason="done",
            summary="Everything is completed.",
            total_steps=1,
            steps=result.steps,
        )
        passed, reasons = sandbox.evaluate_result(bad_summary_result)
        assert passed is False
        assert any("summary" in r.lower() for r in reasons)


def test_eval_sandbox_impossible_task_requires_no_r1_tools() -> None:
    task = EvalTask(
        id="imp_01",
        prompt="Book me a flight to Goa",
        category="impossible",
        split="dev",
        expected={"stop_reason": "done"},
        success_check={
            "stop_reason": "done",
            "summary_startswith": "I can't do that yet:",
            "no_unexpected_r1_tools": True,
        },
    )

    with EvalSandbox(task) as sandbox:
        # Passes when summary starts with "I can't do that yet:"
        res_ok = ExecutorResult(
            run_id="run_ok",
            stop_reason="done",
            summary="I can't do that yet: I have no flight booking tool.",
            total_steps=1,
        )
        passed, reasons = sandbox.evaluate_result(res_ok)
        assert passed is True
        assert reasons == []

        # Fails when summary doesn't start with prefix
        res_bad = ExecutorResult(
            run_id="run_bad",
            stop_reason="done",
            summary="I cannot book flights for you.",
            total_steps=1,
        )
        passed, reasons = sandbox.evaluate_result(res_bad)
        assert passed is False
        assert any("summary does not start with" in r.lower() for r in reasons)

    with EvalSandbox(task) as sandbox:
        # Result where model refused cleanly without calling R1 tools
        clean_refusal = ExecutorResult(
            run_id="run_imp",
            stop_reason="done",
            summary="I can't do that yet: I have no flight booking tool.",
            total_steps=1,
            steps=[
                StepRecord(
                    step=1,
                    action={"kind": "done", "summary": "I can't do that yet: I have no flight booking tool."},
                    observation_summary="refusal",
                )
            ],
        )
        passed, reasons = sandbox.evaluate_result(clean_refusal)
        assert passed is True

        # Result where model hallucinated and called an R1 tool
        violation_result = ExecutorResult(
            run_id="run_imp_bad",
            stop_reason="done",
            summary="Done",
            total_steps=1,
            steps=[
                StepRecord(
                    step=1,
                    action={"kind": "tool", "tool": "browser.open_url", "args": {"url": "https://flights.com"}},
                    observation_summary="opened",
                )
            ],
        )
        passed, reasons = sandbox.evaluate_result(violation_result)
        assert passed is False
        assert any("unexpected r1 tool" in r.lower() for r in reasons)
