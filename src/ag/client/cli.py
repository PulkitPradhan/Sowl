"""ag CLI client powered by Typer with fast-path execution for ping."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Annotated, Any

from ag.wire.rpc import get_default_socket_path

_app = None


def get_app():
    """Lazily construct and return the Typer app to keep CLI startup fast."""
    global _app
    if _app is not None:
        return _app

    import asyncio

    import typer

    from ag.client.rpc import RpcClient, RpcRemoteError

    # Make typer accessible in module globals for annotation string evaluation
    globals()["typer"] = typer

    app = typer.Typer(name="ag", help="ag: Local-first personal AI agent CLI")
    state = {"socket_path": get_default_socket_path()}

    @app.callback()
    def common(
        socket_path: Annotated[
            Path | None,
            typer.Option(
                "--socket-path",
                "-s",
                help="Path to agentd Unix domain socket",
            ),
        ] = None,
    ) -> None:
        if socket_path is not None:
            state["socket_path"] = socket_path

    @app.command()
    def ping() -> None:
        """Check connectivity to agentd."""
        async def _ping():
            async with RpcClient(state["socket_path"]) as client:
                return await client.call("ping")

        try:
            res = asyncio.run(_ping())
            typer.echo(res)
        except FileNotFoundError:
            typer.secho(
                f"Error: agentd socket not found at {state['socket_path']}. Is agentd running?",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        except ConnectionRefusedError:
            typer.secho(
                f"Error: Connection refused at {state['socket_path']}. agentd may have terminated.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        except RpcRemoteError as err:
            typer.secho(f"RPC Error: {err}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    @app.command()
    def run(
        text: Annotated[str, typer.Argument(help="Text to send to agentd")],
    ) -> None:
        """Send text to agentd (Phase 1 echo)."""
        async def _run():
            async with RpcClient(state["socket_path"]) as client:
                return await client.call("run", {"text": text})

        try:
            res = asyncio.run(_run())
            if isinstance(res, (dict, list)):
                typer.echo(json.dumps(res, indent=2))
            else:
                typer.echo(str(res))
        except FileNotFoundError:
            typer.secho(
                f"Error: agentd socket not found at {state['socket_path']}. Is agentd running?",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        except ConnectionRefusedError:
            typer.secho(
                f"Error: Connection refused at {state['socket_path']}. agentd may have terminated.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        except RpcRemoteError as err:
            typer.secho(f"RPC Error: {err}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    @app.command()
    def status() -> None:
        """Get status from agentd."""
        async def _status():
            async with RpcClient(state["socket_path"]) as client:
                return await client.call("status")

        try:
            res = asyncio.run(_status())
            typer.echo(json.dumps(res, indent=2))
        except (FileNotFoundError, ConnectionRefusedError):
            typer.secho("agentd: stopped (socket inactive)", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        except RpcRemoteError as err:
            typer.secho(f"RPC Error: {err}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    _app = app
    return _app


def __getattr__(name: str) -> Any:
    if name == "app":
        return get_app()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def _try_fast_ping(args: list[str]) -> int | None:
    """Fast-path for `ag ping` to avoid heavy CLI framework import latency."""
    socket_path_str = os.environ.get("AG_SOCKET_PATH")
    i = 0
    is_ping = False

    while i < len(args):
        arg = args[i]
        if arg in ("-s", "--socket-path") and i + 1 < len(args):
            socket_path_str = args[i + 1]
            i += 2
        elif arg.startswith("--socket-path="):
            socket_path_str = arg.split("=", 1)[1]
            i += 1
        elif arg == "ping":
            is_ping = True
            i += 1
        else:
            return None  # Fall back to full Typer parser for any other argument/help
    if not is_ping:
        return None

    target = Path(socket_path_str).expanduser() if socket_path_str else get_default_socket_path()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(2.0)
        sock.connect(str(target))
        sock.sendall(b'{"jsonrpc":"2.0","method":"ping","id":1}\n')
        raw = sock.recv(1024)
        data = json.loads(raw.decode("utf-8"))
        if "result" in data:
            sys.stdout.write(f"{data['result']}\n")
            return 0
        if "error" in data:
            sys.stderr.write(f"RPC Error: {data['error'].get('message', 'Unknown error')}\n")
            return 1
    except FileNotFoundError:
        sys.stderr.write(f"Error: agentd socket not found at {target}. Is agentd running?\n")
        return 1
    except ConnectionRefusedError:
        sys.stderr.write(
            f"Error: Connection refused at {target}. agentd may have terminated.\n"
        )
        return 1
    except Exception as err:  # noqa: BLE001
        sys.stderr.write(f"Error: {err}\n")
        return 1
    finally:
        sock.close()
    return 0


def main() -> None:
    """Entry point for the ag CLI command."""
    ret = _try_fast_ping(sys.argv[1:])
    if ret is not None:
        sys.exit(ret)
    get_app()()


if __name__ == "__main__":
    main()
