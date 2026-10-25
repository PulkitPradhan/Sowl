"""Tests for ag web CLI command and port collision handling."""

from __future__ import annotations

import socket
from pathlib import Path
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from ag.client.cli import get_app
from ag.skills.launcher import MockLauncherBackend, set_launcher
from ag.web.server import run_web_server


def test_ag_web_no_open_prints_url(tmp_path: Path):
    """ag web --no-open must print the URL containing token in fragment and not invoke browser."""
    runner = CliRunner()
    app = get_app()
    mock_launcher = MockLauncherBackend()
    set_launcher(mock_launcher)

    # Pick an ephemeral free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]

    with (
        patch("ag.web.server.ensure_agentd_running", new_callable=AsyncMock),
        patch("ag.web.server.run_web_server", side_effect=lambda **kwargs: run_web_server(**{**kwargs, "serve_forever": False})),
    ):
        result = runner.invoke(app, ["web", "--port", str(free_port), "--no-open"])
        assert result.exit_code == 0
        assert f"http://127.0.0.1:{free_port}/#token=" in result.stdout
        assert len(mock_launcher.opened_urls) == 0


def test_ag_web_port_in_use_exits_with_clear_message(tmp_path: Path):
    """When target port is in use, ag web must exit with code 1 and a clear error message."""
    runner = CliRunner()
    app = get_app()

    # Bind a socket to occupy a port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]

    try:
        result = runner.invoke(app, ["web", "--port", str(port), "--no-open"])
        assert result.exit_code != 0
        assert f"Error: Port {port} is already in use" in result.output
    finally:
        sock.close()
