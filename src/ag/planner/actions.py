"""Planner action schemas, discriminated union, argument validators, and action hashing."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

PLACEHOLDER_PATTERN = re.compile(
    r"^(<[^>]+>|\$[a-zA-Z0-9_.-]+|\{\{[^}]+\}\}|\[(?:TODO|FIXME|PLACEHOLDER)[^\]]*\]|placeholder|TODO)$",
    re.IGNORECASE,
)


def validate_action_arguments(args: dict[str, Any]) -> dict[str, Any]:
    """Validate that action arguments contain no placeholder or template variables.

    Raises ValueError if any string argument resembles a placeholder like
    '<output>', '$step1', '{{result}}', or '[TODO]'.
    """
    for key, value in args.items():
        if isinstance(value, str):
            trimmed = value.strip()
            if PLACEHOLDER_PATTERN.match(trimmed) or "TODO" in trimmed or "PLACEHOLDER" in trimmed.upper():
                raise ValueError(
                    f"Placeholder argument detected in '{key}': '{value}'. "
                    "All arguments must be literal, concrete values."
                )
        elif isinstance(value, dict):
            validate_action_arguments(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and (PLACEHOLDER_PATTERN.match(item.strip()) or "TODO" in item):
                    raise ValueError(f"Placeholder detected in list item: '{item}'.")
    return args


class ToolAction(BaseModel):
    kind: Literal["tool"] = Field(..., description="Must be 'tool'")
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class AskAction(BaseModel):
    kind: Literal["ask"] = Field(..., description="Must be 'ask'")
    question: str


class DoneAction(BaseModel):
    kind: Literal["done"] = Field(..., description="Must be 'done'")
    summary: str


class PlanAction(BaseModel):
    kind: Literal["plan"] = Field(..., description="Must be 'plan'")
    steps: list[ToolAction] = Field(..., min_length=1, max_length=10)


SingleAction = Annotated[ToolAction | AskAction | DoneAction, Field(discriminator="kind")]
Action = Annotated[ToolAction | AskAction | DoneAction | PlanAction, Field(discriminator="kind")]

_SingleActionAdapter = TypeAdapter(SingleAction)
_ActionAdapter = TypeAdapter(Action)


def parse_action(data: dict[str, Any] | str, allow_plan: bool = True) -> Action:
    """Parse raw dictionary or JSON string into an Action instance."""
    if isinstance(data, str):
        data = json.loads(data)

    if isinstance(data, dict) and "kind" not in data:
        if "tool" in data:
            data["kind"] = "tool"
        elif "question" in data:
            data["kind"] = "ask"
        elif "summary" in data:
            data["kind"] = "done"
        elif "steps" in data:
            data["kind"] = "plan"

    adapter = _ActionAdapter if allow_plan else _SingleActionAdapter
    action = adapter.validate_python(data)

    # Validate arguments for ToolAction or PlanAction
    if isinstance(action, ToolAction):
        validate_action_arguments(action.args)
    elif isinstance(action, PlanAction):
        for step in action.steps:
            validate_action_arguments(step.args)

    return action


def compute_action_hash(tool: str, args: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash for a tool and its arguments."""
    canonical_payload = {
        "tool": tool,
        "args": args,
    }
    encoded = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
