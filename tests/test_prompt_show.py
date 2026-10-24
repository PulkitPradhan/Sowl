"""Tests for ag prompt show CLI command and formatting."""

from pathlib import Path

from typer.testing import CliRunner

from ag.client.cli import app
from ag.models.profile import set_active_profile

runner = CliRunner()


def test_cli_prompt_show_basic(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AG_WORKSPACE_DIR", str(tmp_path))
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "test_p.yaml").write_text(
        "name: test_p\ntier: 1\ncontext_window: 4096\nwire_format: toon_in\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AG_PROFILES_DIR", str(profiles_dir))
    set_active_profile("test_p", workspace_dir=tmp_path)

    result = runner.invoke(app, ["prompt", "show", "List my files"])
    assert result.exit_code == 0
    assert "Wire format: toon_in" in result.output
    assert "Approximate tokens:" in result.output
    assert "Role: system" in result.output
    assert "Role: user" in result.output
    assert "List my files" in result.output


def test_cli_prompt_show_sample_observation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AG_WORKSPACE_DIR", str(tmp_path))
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "tier2_gpu.yaml").write_text(
        "name: tier2_gpu\ntier: 2\ncontext_window: 8192\nwire_format: toon_in\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AG_PROFILES_DIR", str(profiles_dir))

    result = runner.invoke(
        app,
        ["prompt", "show", "List my files", "--profile", "tier2_gpu", "--sample-observation"],
    )
    assert result.exit_code == 0
    assert "files.list" in result.output
    assert "files.read" in result.output
    # Must format a fake files.list result (~5 entries) through real format_observation path
    assert "data[" in result.output or "file_" in result.output
