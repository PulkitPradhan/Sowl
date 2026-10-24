"""Tests for endpoint loopback verification and security controls."""

from pathlib import Path

from typer.testing import CliRunner

from ag.client.cli import app, is_loopback_endpoint

runner = CliRunner()


def test_is_loopback_endpoint() -> None:
    assert is_loopback_endpoint("http://localhost:11434/v1") is True
    assert is_loopback_endpoint("http://127.0.0.1:11434/v1") is True
    assert is_loopback_endpoint("http://[::1]:11434/v1") is True
    assert is_loopback_endpoint("http://127.0.0.2:8000") is True

    assert is_loopback_endpoint("http://192.168.1.50:11434/v1") is False
    assert is_loopback_endpoint("https://api.openai.com/v1") is False
    assert is_loopback_endpoint("http://remote-server:11434/v1") is False


def test_cli_run_rejects_remote_endpoint_without_flag(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AG_WORKSPACE_DIR", str(tmp_path))
    result = runner.invoke(app, ["run", "hello", "--endpoint", "http://192.168.1.50:11434/v1"])
    assert result.exit_code != 0
    assert "loopback" in result.output.lower() or "remote" in result.output.lower()
