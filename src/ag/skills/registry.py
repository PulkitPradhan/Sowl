"""Skill registry discovering manifests, managing collision errors, and executing allowlisted tools."""

from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ag.skills.manifest import (
    ManifestValidationError,
    SkillManifest,
    ToolManifest,
    validate_manifest_file,
)
from ag.wire.wire import to_llm


class DuplicateSkillError(Exception):
    """Raised when two skills share the exact same name across search paths."""


def get_default_package_skills_dir() -> Path:
    """Return project root skills directory relative to package."""
    return Path(__file__).resolve().parent.parent.parent.parent / "skills"


def get_default_workspace_skills_dir() -> Path:
    """Return user workspace skills directory ~/.ag/workspace/skills."""
    env_dir = os.environ.get("AG_WORKSPACE_DIR")
    if env_dir:
        return Path(env_dir).expanduser() / "skills"
    return Path("~/.ag/workspace/skills").expanduser()


class Skill(BaseModel):
    """Loaded skill instance with on-demand instructions."""

    manifest: SkillManifest
    path: Path
    skill_md_path: Path | None = None

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def summary(self) -> str:
        return self.manifest.summary

    @property
    def risk(self) -> str:
        return self.manifest.risk

    @property
    def tools(self) -> list[ToolManifest]:
        return self.manifest.tools

    def get_instructions(self) -> str:
        """Load SKILL.md instructions into memory strictly on demand."""
        if self.skill_md_path and self.skill_md_path.is_file():
            try:
                return self.skill_md_path.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001
                return f"# Skill {self.name}\n{self.summary}"
        return f"# Skill {self.name}\n{self.summary}"


class SkillRegistry:
    """Discovers, validates, and manages skills and their executable tools."""

    def __init__(
        self,
        search_dirs: list[Path] | None = None,
        workspace_dir: Path | None = None,
    ) -> None:
        if search_dirs is not None:
            self.search_dirs = [Path(p) for p in search_dirs]
        elif workspace_dir is not None:
            self.search_dirs = [
                get_default_package_skills_dir(),
                Path(workspace_dir).expanduser() / "skills",
            ]
        else:
            self.search_dirs = [
                get_default_package_skills_dir(),
                get_default_workspace_skills_dir(),
            ]

        self.skills: dict[str, Skill] = {}
        self.broken_manifests: list[dict[str, str]] = []
        self.workspace_dir: Path = Path(workspace_dir).expanduser() if workspace_dir is not None else Path.cwd()

    def discover(self) -> None:
        """Scan search directories for skills and manifest.toml files."""
        self.skills.clear()
        self.broken_manifests.clear()

        for search_dir in self.search_dirs:
            if not search_dir.is_dir():
                continue

            for folder in sorted(search_dir.iterdir()):
                if not folder.is_dir():
                    continue

                manifest_file = folder / "manifest.toml"
                if not manifest_file.is_file():
                    continue

                try:
                    manifest = validate_manifest_file(manifest_file)
                except ManifestValidationError as err:
                    self.broken_manifests.append({
                        "file": str(manifest_file),
                        "field": getattr(err, "field", "unknown"),
                        "error": str(err),
                    })
                    continue

                # Collision detection: error, not override
                if manifest.name in self.skills:
                    existing = self.skills[manifest.name]
                    raise DuplicateSkillError(
                        f"Duplicate skill '{manifest.name}' collision: found at {folder} and {existing.path}"
                    )

                skill_md = folder / "SKILL.md"
                self.skills[manifest.name] = Skill(
                    manifest=manifest,
                    path=folder,
                    skill_md_path=skill_md if skill_md.is_file() else None,
                )

    def get_skill(self, name: str) -> Skill | None:
        """Retrieve a skill by name."""
        return self.skills.get(name)

    def get_tools(self) -> dict[str, ToolManifest]:
        """Return a mapping of fully-qualified tool name to ToolManifest."""
        tools: dict[str, ToolManifest] = {}
        for skill in self.skills.values():
            for tool in skill.tools:
                tools[tool.name] = tool
        return tools

    def list_skills(self) -> list[Skill]:
        """Return list of all successfully loaded skills."""
        return list(self.skills.values())

    def get_compact_table(self, fmt: str = "toon_in") -> str:
        """Generate a compact table of (name, summary, risk) for LLM prompts."""
        items = [
            {"name": s.name, "summary": s.summary, "risk": s.risk}
            for s in self.list_skills()
        ]
        return to_llm({"skills": items}, fmt=fmt)

    def find_tool(self, tool_name: str) -> ToolManifest | None:
        """Look up tool definition across all registered skills."""
        for skill in self.skills.values():
            for tool in skill.tools:
                if tool.name == tool_name:
                    return tool
        return None

    async def execute_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute allowlisted tool with 10s timeout, arg validation, and exception safety."""
        tool = self.find_tool(tool_name)
        if tool is None:
            return {"ok": False, "data": None, "error": f"Tool '{tool_name}' not found in registry"}

        if tool.risk == "R3":
            return {"ok": False, "data": None, "error": f"Execution blocked: R3 tools cannot be executed ('{tool_name}')"}

        if not tool.handler:
            return {"ok": False, "data": None, "error": f"Tool '{tool_name}' has no handler (instruction-only)"}

        # Allowlist check: must be under ag.skills.builtin.
        if not tool.handler.startswith("ag.skills.builtin."):
            return {
                "ok": False,
                "data": None,
                "error": f"Security violation: handler '{tool.handler}' is not allowlisted under 'ag.skills.builtin.'",
            }

        # Validate arguments: extra="forbid"
        allowed_args = set(tool.args.keys())
        provided_args = set(args.keys())
        extra_args = provided_args - allowed_args
        if extra_args:
            return {
                "ok": False,
                "data": None,
                "error": f"Extra argument(s) forbidden: {sorted(extra_args)}",
            }

        missing_args = [name for name, field_def in tool.args.items() if field_def.required and name not in args]
        if missing_args:
            return {
                "ok": False,
                "data": None,
                "error": f"Missing required argument(s): {sorted(missing_args)}",
            }

        # Dynamically import handler
        try:
            mod_path, func_name = tool.handler.split(":", 1)
            module = importlib.import_module(mod_path)
            func = getattr(module, func_name)
        except Exception as err:  # noqa: BLE001
            return {"ok": False, "data": None, "error": f"Could not load handler '{tool.handler}': {err}"}

        # Execute with 10s timeout via asyncio.to_thread
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(func, **args),
                timeout=10.0,
            )
            if isinstance(result, dict) and "ok" in result:
                return result
            return {"ok": True, "data": result, "error": None}
        except TimeoutError:
            return {"ok": False, "data": None, "error": f"Tool '{tool_name}' timed out after 10 seconds"}
        except Exception as err:  # noqa: BLE001
            return {"ok": False, "data": None, "error": f"Tool '{tool_name}' execution error: {err}"}
