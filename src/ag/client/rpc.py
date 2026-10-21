"""Async JSON-RPC 2.0 client over Unix domain socket."""

from __future__ import annotations

import asyncio
import json
import types
from pathlib import Path
from typing import Any, Self

from ag.wire.rpc import format_notification, format_request


class RpcRemoteError(Exception):
    """Exception raised when the remote JSON-RPC daemon returns an error."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.data = data


class RpcClient:
    """Client for connecting to agentd over a Unix domain socket."""

    def __init__(self, socket_path: Path | str, timeout: float = 5.0) -> None:
        self.socket_path = Path(socket_path).expanduser()
        self.timeout = timeout
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self._next_id = 1

    async def connect(self) -> None:
        """Establish connection to the daemon socket."""
        if self.writer is not None:
            return
        self.reader, self.writer = await asyncio.open_unix_connection(str(self.socket_path))

    async def close(self) -> None:
        """Close connection to the daemon socket."""
        if self.writer is not None:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except (ConnectionError, BrokenPipeError):
                pass
            finally:
                self.reader = None
                self.writer = None

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        await self.close()

    async def call(self, method: str, params: Any = None) -> Any:
        """Send a JSON-RPC request and wait for the response."""
        if self.writer is None or self.reader is None:
            await self.connect()
        assert self.writer is not None and self.reader is not None

        msg_id = self._next_id
        self._next_id += 1

        payload = format_request(method=method, params=params, msg_id=msg_id)
        self.writer.write(payload)
        await self.writer.drain()

        try:
            line = await asyncio.wait_for(self.reader.readline(), timeout=self.timeout)
        except TimeoutError as err:
            raise TimeoutError(f"RPC call to '{method}' timed out after {self.timeout}s") from err

        if not line:
            raise ConnectionResetError("Connection closed by agentd daemon")

        data = json.loads(line.decode("utf-8"))
        if "error" in data:
            err = data["error"]
            raise RpcRemoteError(
                code=err.get("code", -32603),
                message=err.get("message", "Unknown error"),
                data=err.get("data"),
            )
        return data.get("result")

    async def notify(self, method: str, params: Any = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if self.writer is None or self.reader is None:
            await self.connect()
        assert self.writer is not None

        payload = format_notification(method=method, params=params)
        self.writer.write(payload)
        await self.writer.drain()
