"""Eval task definition schema, YAML loading, and quick subset selection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


def get_default_tasks_dir() -> Path:
    """Return default tasks directory evals/tasks/."""
    return Path(__file__).resolve().parent.parent.parent.parent / "evals" / "tasks"


class EvalTask(BaseModel):
    """Scenario specification for the evaluation harness."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Unique task identifier, e.g. norm_01")
    prompt: str = Field(description="Natural language instruction for the agent")
    category: Literal[
        "normal", "multi_step", "ambiguous", "adversarial", "impossible", "conversational"
    ] = Field(
        description="Scenario category"
    )
    split: Literal["dev", "holdout"] = Field(
        default="dev", description="Evaluation split (dev or holdout)"
    )
    setup: dict[str, str] = Field(
        default_factory=dict, description="Files to create in task workspace (rel_path -> content)"
    )
    expected: dict[str, Any] = Field(
        default_factory=dict, description="Expected initial action or stop reason conditions"
    )
    success_check: dict[str, Any] = Field(
        default_factory=dict, description="Programmatic end-state checks"
    )
    policy: Literal["auto_approve", "auto_deny"] = Field(
        default="auto_approve", description="Confirmation decision policy for non-R0 tools"
    )
    teardown: list[str] = Field(
        default_factory=list, description="Files to clean up"
    )


def compute_task_set_hash(tasks: list[EvalTask]) -> str:
    """Compute deterministic SHA256 hash of task set for audit tracking."""
    sorted_tasks = sorted(tasks, key=lambda t: t.id)
    content = "".join(f"{t.id}:{t.category}:{t.split}:{t.prompt}" for t in sorted_tasks)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_tasks(tasks_dir: Path | None = None, quick: bool = False) -> list[EvalTask]:
    """Load and validate all scenario YAML tasks from the tasks directory.

    If quick=True, deterministically selects exactly 20 tasks:
    - 10 normal
    - 4 multi_step
    - 2 ambiguous
    - 2 adversarial
    - 2 impossible
    """
    tdir = tasks_dir or get_default_tasks_dir()
    if not tdir.is_dir():
        return []

    tasks: list[EvalTask] = []
    for f in sorted(tdir.glob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            task = EvalTask.model_validate(data)
            tasks.append(task)
        except Exception as err:
            raise ValueError(f"Error loading eval task from {f}: {err}") from err

    if not quick:
        return tasks

    # Deterministic quick selection of exactly 20 tasks
    # 10 normal, 4 multi_step, 2 ambiguous, 2 adversarial, 2 impossible
    categories_target = {
        "normal": 10,
        "multi_step": 4,
        "ambiguous": 2,
        "adversarial": 2,
        "impossible": 2,
    }

    selected: list[EvalTask] = []
    for cat, limit in categories_target.items():
        cat_tasks = [t for t in tasks if t.category == cat]
        selected.extend(cat_tasks[:limit])

    return selected
