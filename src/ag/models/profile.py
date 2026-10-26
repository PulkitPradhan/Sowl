"""Model profile definition, loading, and active profile state."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


def get_default_workspace_dir() -> Path:
    """Return default workspace directory ~/.ag, falling back to /tmp/.ag in read-only environments."""
    env_dir = os.environ.get("AG_WORKSPACE_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    target = Path("~/.ag").expanduser()
    try:
        if not target.exists() and not os.access(target.parent, os.W_OK):
            return Path("/tmp/.ag")
    except OSError:
        return Path("/tmp/.ag")
    return target


def get_default_profiles_dir() -> Path:
    """Return project root profiles directory."""
    # If running from repo, project root has profiles/
    env_dir = os.environ.get("AG_PROFILES_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path(__file__).resolve().parent.parent.parent.parent / "profiles"


class ModelProfile(BaseModel):
    """Configuration for a specific model tier / server."""

    name: str
    tier: int = Field(ge=0, le=4, description="Tier 0 (no-LLM), 1 (CPU), 2 (8GB VRAM), 3 (16GB+), 4 (Cloud)")
    model: str = Field(default="", description="Model identifier on server, user-configurable")
    endpoint: str = Field(default="http://localhost:11434/v1", description="OpenAI-compatible base URL")
    context_window: int = Field(default=4096, description="Context window token limit")
    wire_format: Literal["json", "toon_in", "toon_both"] = "toon_in"
    planning_mode: str = "single_pass"
    max_plan_steps: int = 5
    structured_output: bool = True
    thinking: Literal["off", "on"] = "off"
    examples: int = Field(default=0, description="Include one example call per tool if 1")
    laya_judge: Literal["off", "high_risk", "always"] = "off"
    laya_model: str = ""
    run_timeout: float = 120.0
    planner_max_tokens: int = 400
    timeout_connect: float = 3.0
    timeout_read: float = 120.0
    retries: int = 2
    temperature: float = 0.0
    api_key_env: str | None = Field(default=None, description="Environment variable holding the API key")

    def get_api_key(self) -> str | None:
        """Resolve API key strictly from environment variable if configured."""
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


def load_profile(path_or_name: Path | str, profiles_dir: Path | None = None) -> ModelProfile:
    """Load a ModelProfile from a YAML file path or profile name."""
    path = Path(path_or_name)
    if not path.is_file():
        base_dir = profiles_dir or get_default_profiles_dir()
        candidate = base_dir / f"{path_or_name}.yaml"
        if candidate.is_file():
            path = candidate
        else:
            candidate_raw = base_dir / str(path_or_name)
            if candidate_raw.is_file():
                path = candidate_raw
            else:
                raise FileNotFoundError(f"Profile '{path_or_name}' not found at {path} or {candidate}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return ModelProfile.model_validate(data)


def list_profiles(profiles_dir: Path | None = None) -> list[ModelProfile]:
    """List all available profiles in the profiles directory."""
    base_dir = profiles_dir or get_default_profiles_dir()
    if not base_dir.is_dir():
        return []

    profiles: list[ModelProfile] = []
    for file in sorted(base_dir.glob("*.yaml")):
        try:
            profile = load_profile(file)
            profiles.append(profile)
        except (yaml.YAMLError, OSError, ValueError):
            continue
    return profiles


def get_active_profile(workspace_dir: Path | None = None) -> str | None:
    """Retrieve the name of the currently active profile."""
    ws = workspace_dir or get_default_workspace_dir()
    active_file = ws / "active_profile.json"
    if not active_file.is_file():
        # Also check fallback /tmp/.ag if standard location doesn't have it
        fallback_file = Path("/tmp/.ag/active_profile.json")
        if fallback_file.is_file():
            active_file = fallback_file
        else:
            return None
    try:
        data = json.loads(active_file.read_text(encoding="utf-8"))
        return data.get("active_profile")
    except (json.JSONDecodeError, OSError):
        return None


def set_active_profile(profile_name: str, workspace_dir: Path | None = None) -> None:
    """Set the active profile name in the workspace."""
    ws = workspace_dir or get_default_workspace_dir()
    try:
        ws.mkdir(parents=True, exist_ok=True)
    except OSError:
        ws = Path("/tmp/.ag")
        ws.mkdir(parents=True, exist_ok=True)

    try:
        ws.chmod(0o700)
    except OSError:
        pass

    active_file = ws / "active_profile.json"
    payload = {"active_profile": profile_name}
    active_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        active_file.chmod(0o600)
    except OSError:
        pass
