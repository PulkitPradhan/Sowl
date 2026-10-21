"""Tests for the ag CLI using typer.testing.CliRunner."""

import asyncio
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ag.client.cli import app
from ag.daemon.server import AgentDaemon, register_default_handlers

runner = CliRunner()


@pytest.fixture
def test_socket(tmp_path: Path) -> Path:
    return tmp_path / "cli_test.sock"


@pytest.fixture
def cli_daemon(test_socket: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    test_ws = tmp_path / "ws"
    monkeypatch.setenv("AG_WORKSPACE_DIR", str(test_ws))
    daemon = AgentDaemon(test_socket)
    register_default_handlers(daemon, workspace_dir=test_ws)

    loop = asyncio.new_event_loop()
    ready_evt = threading.Event()

    def run_daemon_loop():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(daemon.start())
        ready_evt.set()
        loop.run_until_complete(daemon._stop_event.wait())
        loop.run_until_complete(daemon.stop())
        loop.close()

    thread = threading.Thread(target=run_daemon_loop, daemon=True)
    thread.start()
    assert ready_evt.wait(timeout=5.0), "Daemon failed to start within 5s"

    try:
        yield daemon
    finally:
        loop.call_soon_threadsafe(daemon._stop_event.set)
        thread.join(timeout=5.0)


def test_cli_ping(cli_daemon: AgentDaemon, test_socket: Path):
    result = runner.invoke(app, ["--socket-path", str(test_socket), "ping"])
    assert result.exit_code == 0
    assert "pong" in result.stdout


def test_cli_run_echo(cli_daemon: AgentDaemon, test_socket: Path):
    result = runner.invoke(app, ["--socket-path", str(test_socket), "run", "hello CLI"])
    assert result.exit_code == 0
    assert "hello CLI" in result.stdout


def test_cli_model_list(cli_daemon: AgentDaemon, test_socket: Path):
    result = runner.invoke(app, ["--socket-path", str(test_socket), "model", "list"])
    assert result.exit_code == 0
    assert "tier1_cpu" in result.stdout
    assert "tier2_gpu" in result.stdout


def test_cli_model_use(cli_daemon: AgentDaemon, test_socket: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("AG_WORKSPACE_DIR", str(tmp_path / ".ag"))
    result = runner.invoke(app, ["--socket-path", str(test_socket), "model", "use", "tier2_gpu"])
    assert result.exit_code == 0
    assert "Switched active profile to: tier2_gpu" in result.stdout


def test_cli_doctor(test_socket: Path):
    result = runner.invoke(app, ["--socket-path", str(test_socket), "doctor", "--no-bench"])
    assert result.exit_code == 0
    assert "System Diagnostics" in result.stdout
    assert "Recommended Tier:" in result.stdout

