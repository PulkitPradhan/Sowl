"""agentd daemon CLI runner."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer

from ag.daemon.server import AgentDaemon, get_default_socket_path, register_default_handlers

app = typer.Typer(name="agentd", help="agentd: Background daemon for ag")


@app.command()
def start(
    socket_path: Annotated[
        Path | None,
        typer.Option(
            "--socket-path",
            "-s",
            help="Path to Unix domain socket",
        ),
    ] = None,
) -> None:
    """Start the agentd daemon process."""
    target_path = socket_path or get_default_socket_path()
    daemon = AgentDaemon(target_path)
    register_default_handlers(daemon)

    typer.echo(f"Starting agentd (pid: {os.getpid()}) on socket: {target_path}")

    try:
        asyncio.run(daemon.serve_forever())
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        typer.echo("\nagentd stopped.")


if __name__ == "__main__":
    app()
