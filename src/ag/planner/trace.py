"""Trace logging, 200-run retention, file permissions, and trace inspector."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ag.models.profile import get_default_workspace_dir


def get_default_traces_dir(workspace_dir: Path | None = None) -> Path:
    """Return default trace directory ~/.ag/traces/ with 0700 permissions."""
    base = workspace_dir or get_default_workspace_dir()
    traces_dir = base / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    try:
        traces_dir.chmod(0o700)
    except OSError:
        pass
    return traces_dir


def prune_old_traces(traces_dir: Path, max_keep: int = 200) -> None:
    """Retain only the latest max_keep trace files based on mtime."""
    if not traces_dir.is_dir():
        return
    trace_files = sorted(
        traces_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old_file in trace_files[max_keep:]:
        try:
            old_file.unlink(missing_ok=True)
        except OSError:
            pass


def list_traces(traces_dir: Path | None = None) -> list[Path]:
    """List all trace files sorted newest first."""
    tdir = traces_dir or get_default_traces_dir()
    if not tdir.is_dir():
        return []
    return sorted(
        tdir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def load_last_trace(traces_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load records from the most recent trace file."""
    files = list_traces(traces_dir)
    if not files:
        return []
    return load_trace_file(files[0])


def load_trace_file(path: Path) -> list[dict[str, Any]]:
    """Parse JSONL records from a trace file."""
    records = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


class TraceLogger:
    """Appends execution steps and summary records to a secure trace file."""

    def __init__(self, traces_dir: Path | None = None) -> None:
        self.traces_dir = traces_dir or get_default_traces_dir()
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.traces_dir.chmod(0o700)
        except OSError:
            pass

    def _get_trace_file(self, run_id: str) -> Path:
        return self.traces_dir / f"{run_id}.jsonl"

    def _write_record(self, run_id: str, record: dict[str, Any]) -> None:
        trace_file = self._get_trace_file(run_id)
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        with trace_file.open("a", encoding="utf-8") as f:
            f.write(line)
        try:
            trace_file.chmod(0o600)
        except OSError:
            pass

    def log_init(self, run_id: str, task: str, profile_name: str) -> None:
        prune_old_traces(self.traces_dir, max_keep=199)
        record = {
            "type": "init",
            "run_id": run_id,
            "task": task,
            "profile": profile_name,
            "timestamp": time.time(),
        }
        self._write_record(run_id, record)

    def log_step(
        self,
        run_id: str,
        step_idx: int,
        action: dict[str, Any],
        observation_summary: str,
        latency_ms: float,
        usage: dict[str, Any],
        finish_reason: str | None = None,
    ) -> None:
        record = {
            "type": "step",
            "run_id": run_id,
            "step": step_idx,
            "action": action,
            "observation_summary": observation_summary,
            "latency_ms": round(latency_ms, 2),
            "usage": usage,
            "finish_reason": finish_reason,
            "timestamp": time.time(),
        }
        self._write_record(run_id, record)

    def log_done(
        self,
        run_id: str,
        stop_reason: str,
        summary: str,
        total_steps: int,
        duration_s: float,
    ) -> None:
        record = {
            "type": "done",
            "run_id": run_id,
            "stop_reason": stop_reason,
            "summary": summary,
            "total_steps": total_steps,
            "duration_s": round(duration_s, 2),
            "timestamp": time.time(),
        }
        self._write_record(run_id, record)
