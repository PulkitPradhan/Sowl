"""agentd daemon server: asyncio Unix domain socket server with JSON-RPC 2.0 dispatch."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ag.wire.rpc import (
    INTERNAL_ERROR,
    METHOD_NOT_FOUND,
    JsonRpcError,
    RpcNotification,
    RpcRequest,
    format_error,
    format_response,
    get_default_socket_path,
    parse_message,
)

Handler = Callable[[Any], Any] | Callable[[Any], Awaitable[Any]]

# Re-export for convenience
__all__ = ["AgentDaemon", "get_default_socket_path"]


class AgentDaemon:
    """The agentd daemon process managing client connections and method dispatch."""

    def __init__(self, socket_path: Path | str | None = None) -> None:
        self.socket_path = (
            Path(socket_path).expanduser() if socket_path else get_default_socket_path()
        )
        self.handlers: dict[str, Handler] = {}
        self.server: asyncio.Server | None = None
        self._stop_event = asyncio.Event()
        self._active_writers: set[asyncio.StreamWriter] = set()

    def register(self, method: str, handler: Handler) -> None:
        """Register an RPC method handler."""
        self.handlers[method] = handler

    async def _check_and_clean_stale_socket(self) -> None:
        """Clean up stale socket file if daemon is not currently running."""
        if not self.socket_path.exists():
            return

        try:
            _, writer = await asyncio.open_unix_connection(str(self.socket_path))
            writer.close()
            await writer.wait_closed()
            raise RuntimeError(f"agentd is already running on {self.socket_path}")
        except (ConnectionRefusedError, FileNotFoundError):
            # Socket file exists on filesystem but no active process is listening: safe to unlink
            self.socket_path.unlink(missing_ok=True)

    async def start(self) -> asyncio.Server:
        """Start the Unix domain socket server."""
        if self.server is not None:
            return self.server

        # Ensure parent directory exists (e.g. ~/.ag with 0700 permissions)
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.socket_path.parent.chmod(0o700)
        except OSError:
            pass

        await self._check_and_clean_stale_socket()

        self.server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self.socket_path),
        )

        # Secure socket permissions: read/write by owner only (0600)
        try:
            self.socket_path.chmod(0o600)
        except OSError:
            pass

        return self.server

    async def stop(self) -> None:
        """Stop the server, terminate client connections, and remove socket file."""
        self._stop_event.set()

        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

        # Close all active client connections
        for writer in list(self._active_writers):
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionError, BrokenPipeError):
                pass
        self._active_writers.clear()

        # Remove the socket file
        if self.socket_path.exists():
            self.socket_path.unlink(missing_ok=True)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Read and dispatch JSON-RPC 2.0 messages from a client stream."""
        self._active_writers.add(writer)
        try:
            while not self._stop_event.is_set():
                line = await reader.readline()
                if not line:
                    break  # Client disconnected

                try:
                    msg = parse_message(line)
                except JsonRpcError as err:
                    response_bytes = format_error(
                        code=err.code,
                        message=err.message,
                        data=err.data,
                        msg_id=err.msg_id,
                    )
                    writer.write(response_bytes)
                    await writer.drain()
                    continue
                except Exception as err:  # noqa: BLE001
                    response_bytes = format_error(
                        code=INTERNAL_ERROR,
                        message=f"Internal error while parsing: {err}",
                    )
                    writer.write(response_bytes)
                    await writer.drain()
                    continue

                if isinstance(msg, RpcNotification):
                    handler = self.handlers.get(msg.method)
                    if handler is not None:
                        try:
                            if inspect.iscoroutinefunction(handler):
                                await handler(msg.params)
                            else:
                                handler(msg.params)
                        except Exception:  # noqa: BLE001, S110
                            pass  # Per JSON-RPC 2.0: notifications never return errors or responses
                    continue

                if isinstance(msg, RpcRequest):
                    handler = self.handlers.get(msg.method)
                    if handler is None:
                        response_bytes = format_error(
                            code=METHOD_NOT_FOUND,
                            message=f"Method '{msg.method}' not found",
                            msg_id=msg.id,
                        )
                    else:
                        try:
                            if inspect.iscoroutinefunction(handler):
                                result = await handler(msg.params)
                            else:
                                result = handler(msg.params)
                            response_bytes = format_response(result=result, msg_id=msg.id)
                        except JsonRpcError as err:
                            response_bytes = format_error(
                                code=err.code,
                                message=err.message,
                                data=err.data,
                                msg_id=msg.id,
                            )
                        except Exception as err:  # noqa: BLE001
                            response_bytes = format_error(
                                code=INTERNAL_ERROR,
                                message=f"Internal error in '{msg.method}': {err}",
                                msg_id=msg.id,
                            )
                    writer.write(response_bytes)
                    await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            self._active_writers.discard(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionError, BrokenPipeError):
                pass

    async def serve_forever(self) -> None:
        """Run the daemon server until stopped."""
        await self.start()
        loop = asyncio.get_running_loop()
        for sig in ("SIGINT", "SIGTERM"):
            import signal

            if hasattr(signal, sig):
                try:
                    loop.add_signal_handler(
                        getattr(signal, sig),
                        lambda: asyncio.create_task(self.stop()),
                    )
                except NotImplementedError:
                    pass
        await self._stop_event.wait()
