"""Newline-delimited JSON-RPC 2.0 wire protocol specification and serialization."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Standard JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def get_default_socket_path() -> Path:
    """Return the default Unix domain socket path for agentd."""
    env_path = os.environ.get("AG_SOCKET_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / ".ag" / "ag.sock"


class JsonRpcError(Exception):
    """Exception raised when JSON-RPC parsing or validation fails."""

    def __init__(
        self,
        code: int,
        message: str,
        data: Any = None,
        msg_id: int | str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data
        self.msg_id = msg_id


@dataclass(slots=True)
class RpcRequest:
    """A JSON-RPC 2.0 Request object."""

    method: str
    params: Any
    id: int | str
    jsonrpc: str = "2.0"


@dataclass(slots=True)
class RpcNotification:
    """A JSON-RPC 2.0 Notification object (no id)."""

    method: str
    params: Any
    jsonrpc: str = "2.0"
    id: None = None


@dataclass(slots=True)
class RpcResponse:
    """A JSON-RPC 2.0 Success Response object."""

    result: Any
    id: int | str
    jsonrpc: str = "2.0"


def format_request(method: str, params: Any = None, msg_id: int | str = 1) -> bytes:
    """Format a JSON-RPC 2.0 request as newline-terminated UTF-8 bytes."""
    payload: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
        "id": msg_id,
    }
    if params is not None:
        payload["params"] = params
    return json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"


def format_notification(method: str, params: Any = None) -> bytes:
    """Format a JSON-RPC 2.0 notification as newline-terminated UTF-8 bytes."""
    payload: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        payload["params"] = params
    return json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"


def format_response(result: Any, msg_id: int | str) -> bytes:
    """Format a JSON-RPC 2.0 success response as newline-terminated UTF-8 bytes."""
    payload = {
        "jsonrpc": "2.0",
        "result": result,
        "id": msg_id,
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"


def format_error(
    code: int,
    message: str,
    data: Any = None,
    msg_id: int | str | None = None,
) -> bytes:
    """Format a JSON-RPC 2.0 error response as newline-terminated UTF-8 bytes."""
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    payload = {
        "jsonrpc": "2.0",
        "error": err,
        "id": msg_id,
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"


def parse_message(raw: str | bytes) -> RpcRequest | RpcNotification:
    """Parse a single newline-delimited JSON-RPC 2.0 message."""
    if isinstance(raw, bytes):
        try:
            raw_str = raw.decode("utf-8").strip()
        except UnicodeDecodeError as err:
            raise JsonRpcError(PARSE_ERROR, "Invalid UTF-8 encoding") from err
    else:
        raw_str = raw.strip()

    if not raw_str:
        raise JsonRpcError(INVALID_REQUEST, "Empty request")

    try:
        data = json.loads(raw_str)
    except json.JSONDecodeError as err:
        raise JsonRpcError(PARSE_ERROR, f"Parse error: {err.msg}") from err

    if not isinstance(data, dict):
        raise JsonRpcError(INVALID_REQUEST, "Request must be a JSON object")

    if data.get("jsonrpc") != "2.0":
        raise JsonRpcError(INVALID_REQUEST, "Invalid or missing jsonrpc version (must be '2.0')")

    method = data.get("method")
    if not isinstance(method, str) or not method:
        raise JsonRpcError(INVALID_REQUEST, "Missing or invalid 'method' field")

    params = data.get("params")
    if "id" in data:
        msg_id = data["id"]
        if msg_id is not None and not isinstance(msg_id, (str, int, float)):
            raise JsonRpcError(
                INVALID_REQUEST, "Field 'id' must be string, number, or null", msg_id=None
            )
        return RpcRequest(method=method, params=params, id=msg_id)
    return RpcNotification(method=method, params=params)
