"""Tests for profile loading, validation, and switching."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ag.models.profile import (
    ModelProfile,
    get_active_profile,
    list_profiles,
    load_profile,
    set_active_profile,
)


def test_model_profile_defaults():
    profile = ModelProfile(name="test", tier=1)
    assert profile.name == "test"
    assert profile.tier == 1
    assert profile.endpoint == "http://localhost:11434/v1"
    assert profile.structured_output is True
    assert profile.timeout_connect == 3.0
    assert profile.timeout_read == 120.0
    assert profile.api_key_env is None


def test_model_profile_cloud_reads_key_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    profile = ModelProfile(name="cloud", tier=4, api_key_env="OPENAI_API_KEY")
    assert profile.get_api_key() == "secret-test-key"

    # Ensure api_key itself is never serialized
    data = profile.model_dump()
    assert "api_key" not in data
    assert data["api_key_env"] == "OPENAI_API_KEY"


def test_load_fixture_profile():
    fixture_path = Path(__file__).parent / "fixtures" / "fake_alpha.yaml"
    profile = load_profile(fixture_path)
    assert profile.name == "fake-alpha"
    assert profile.model == "fake-model-alpha"
    assert profile.tier == 1
    assert profile.context_window == 4096


def test_load_nonexistent_profile_raises():
    with pytest.raises(FileNotFoundError):
        load_profile("nonexistent_profile_xyz")


def test_active_profile_persistence(tmp_path: Path):
    workspace_dir = tmp_path / ".ag"
    set_active_profile("fake-alpha", workspace_dir=workspace_dir)

    active_file = workspace_dir / "active_profile.json"
    assert active_file.exists()

    content = json.loads(active_file.read_text(encoding="utf-8"))
    assert content["active_profile"] == "fake-alpha"
    # Ensure no secrets in active_profile.json
    assert "api_key" not in content

    loaded = get_active_profile(workspace_dir=workspace_dir)
    assert loaded == "fake-alpha"


def test_list_profiles_finds_yaml(tmp_path: Path):
    prof1 = tmp_path / "p1.yaml"
    prof1.write_text("name: p1\ntier: 1\n", encoding="utf-8")
    prof2 = tmp_path / "p2.yaml"
    prof2.write_text("name: p2\ntier: 2\n", encoding="utf-8")

    profiles = list_profiles(profiles_dir=tmp_path)
    assert len(profiles) == 2
    names = {p.name for p in profiles}
    assert names == {"p1", "p2"}


def test_production_profiles_valid():
    repo_profiles = list_profiles()
    names = {p.name for p in repo_profiles}
    assert {"tier0_none", "tier1_cpu", "tier2_gpu", "tier3_heavy", "tier4_cloud"}.issubset(names)

    cloud = load_profile("tier4_cloud")
    assert cloud.api_key_env == "OPENAI_API_KEY"
    assert cloud.timeout_connect >= 3.0
    assert cloud.timeout_read >= 120.0

    tier0 = load_profile("tier0_none")
    assert tier0.tier == 0
    assert tier0.structured_output is False

