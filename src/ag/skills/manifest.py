"""Skill and tool manifest schemas, TOML parsing, and validation."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ManifestValidationError(Exception):
    """Raised when a manifest file is malformed or fails schema validation."""


class ArgField(BaseModel):
    """Schema for a single tool argument."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(description="Argument type, e.g. string, integer, boolean")
    kind: Literal["identifier", "text", "other"] = Field(
        default="other", description="Semantic kind of argument for grounding auditing"
    )
    description: str = Field(default="", description="Description of the argument")
    required: bool = Field(default=False, description="Whether the argument is required")
    default: Any = Field(default=None, description="Default value if not provided")


class ToolManifest(BaseModel):
    """Schema for a single tool inside a skill manifest."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Fully qualified tool name, e.g. files.read")
    summary: str = Field(description="Brief summary of what the tool does (<= 120 chars)")
    risk: Literal["R0", "R1", "R2", "R3"] = Field(description="Tool risk tier (R0-R3)")
    handler: str | None = Field(default=None, description="Allowlisted handler function, e.g. ag.skills.builtin.files:read_file")
    args: dict[str, ArgField] = Field(default_factory=dict, description="Argument specifications")
    examples: list[str] = Field(default_factory=list, description="Example prompt queries for this tool")
    pre_conditions: list[str] = Field(default_factory=list, description="Pre-conditions required for execution")
    post_conditions: list[str] = Field(default_factory=list, description="Post-conditions verified after execution")

    @field_validator("name")
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        if not re.match(r"^[a-z][a-z0-9_-]{1,31}\.[a-z][a-z0-9_-]{1,31}$", v):
            raise ValueError(f"Tool name '{v}' must be fully qualified like 'skill_name.tool_name' matching ^[a-z][a-z0-9_-]{{1,31}}$")
        return v

    @field_validator("summary")
    @classmethod
    def sanitize_summary(cls, v: str) -> str:
        sanitized = re.sub(r"[\r\n\t]+", " ", v).strip()
        sanitized = re.sub(r"\s+", " ", sanitized)
        if len(sanitized) > 120:
            raise ValueError(f"Tool summary exceeds 120 characters limit ({len(sanitized)} chars)")
        return sanitized

    def to_json_schema(self) -> dict[str, Any]:
        """Convert arg fields to standard JSON Schema for the LLM planner."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        type_map = {
            "string": "string",
            "str": "string",
            "integer": "integer",
            "int": "integer",
            "boolean": "boolean",
            "bool": "boolean",
            "number": "number",
            "float": "number",
            "array": "array",
            "list": "array",
            "object": "object",
            "dict": "object",
        }

        for arg_name, field_def in self.args.items():
            prop: dict[str, Any] = {
                "type": type_map.get(field_def.type.lower(), "string"),
                "description": field_def.description,
            }
            if field_def.default is not None:
                prop["default"] = field_def.default
            properties[arg_name] = prop

            if field_def.required:
                required.append(arg_name)

        return {
            "name": self.name,
            "description": self.summary,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }

    def validate_args(self, args: dict[str, Any]) -> None:
        """Validate argument dictionary against tool schema with extra='forbid'."""
        allowed_args = set(self.args.keys())
        provided_args = set(args.keys())
        extra_args = provided_args - allowed_args
        if extra_args:
            raise ValueError(f"Extra argument(s) forbidden: {sorted(extra_args)}")

        missing_args = [name for name, field_def in self.args.items() if field_def.required and name not in args]
        if missing_args:
            raise ValueError(f"Missing required argument(s): {sorted(missing_args)}")


class SkillManifest(BaseModel):
    """Schema for a complete skill manifest (manifest.toml)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Skill name matching ^[a-z][a-z0-9_-]{1,31}$")
    summary: str = Field(description="One-line summary of skill capabilities (<= 120 chars)")
    triggers: list[str] = Field(default_factory=list, description="Trigger keywords or phrases")
    examples: list[str] = Field(default_factory=list, description="User prompt examples")
    tools: list[ToolManifest] = Field(default_factory=list, description="List of tools provided by skill")

    @field_validator("name")
    @classmethod
    def validate_skill_name(cls, v: str) -> str:
        if not re.match(r"^[a-z][a-z0-9_-]{1,31}$", v):
            raise ValueError(f"Skill name '{v}' must match regex ^[a-z][a-z0-9_-]{{1,31}}$")
        return v

    @field_validator("summary")
    @classmethod
    def sanitize_summary(cls, v: str) -> str:
        sanitized = re.sub(r"[\r\n\t]+", " ", v).strip()
        sanitized = re.sub(r"\s+", " ", sanitized)
        if len(sanitized) > 120:
            raise ValueError(f"Skill summary exceeds 120 characters limit ({len(sanitized)} chars)")
        return sanitized

    @property
    def risk(self) -> str:
        """Calculate skill risk tier as the maximum risk among its tools."""
        if not self.tools:
            return "R0"
        risk_order = {"R0": 0, "R1": 1, "R2": 2, "R3": 3}
        max_level = max(risk_order.get(t.risk, 0) for t in self.tools)
        reverse_order = {0: "R0", 1: "R1", 2: "R2", 3: "R3"}
        return reverse_order[max_level]


def parse_manifest_data(data: dict[str, Any], source_label: str = "manifest") -> SkillManifest:
    """Validate dictionary against SkillManifest, raising ManifestValidationError on error."""
    try:
        return SkillManifest.model_validate(data)
    except Exception as err:
        field = "unknown"
        if hasattr(err, "errors"):
            err_list = err.errors()
            if err_list:
                loc = err_list[0].get("loc", [])
                field = ".".join(str(x) for x in loc) if loc else "unknown"
        exc = ManifestValidationError(f"Validation error in {source_label}: field '{field}' - {err}")
        exc.field = field
        raise exc from err


def validate_manifest_file(path: Path) -> SkillManifest:
    """Parse and validate a manifest.toml file, raising informative ManifestValidationError on errors."""
    if not path.is_file():
        raise ManifestValidationError(f"Manifest file not found: {path}")

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as err:
        raise ManifestValidationError(f"Could not read manifest at {path}: {err}") from err

    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError as err:
        raise ManifestValidationError(f"TOML syntax error in {path}: {err}") from err

    return parse_manifest_data(data, source_label=str(path))
