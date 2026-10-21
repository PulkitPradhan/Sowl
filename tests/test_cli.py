"""Tests for the ag CLI using typer.testing.CliRunner."""

import asyncio
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ag.client.cli import app
from ag.daemon.server import AgentDaemon

runner = CliRunner()


@pytest.fixture
def test_socket(tmp_path: Path) -> Path:
    return tmp_path / "cli_test.sock"


@pytest.fixture
def cli_daemon(test_socket: Path):
    daemon = AgentDaemon(test_socket)
    daemon.register("ping", lambda params: "pong")
    daemon.register("run", lambda params: {"echo": params})

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
