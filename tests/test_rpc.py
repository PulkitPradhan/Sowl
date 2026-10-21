"""Tests for newline-delimited JSON-RPC 2.0 wire protocol."""

import json

import pytest

from ag.wire.rpc import (
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    JsonRpcError,
    RpcNotification,
    RpcRequest,
    format_error,
    format_notification,
    format_request,
    format_response,
    parse_message,
)


def test_format_and_parse_request():
    payload = format_request("ping", {"hello": "world"}, msg_id=42)
    assert payload.endswith(b"\n")

    msg = parse_message(payload)
    assert isinstance(msg, RpcRequest)
    assert msg.jsonrpc == "2.0"
    assert msg.method == "ping"
    assert msg.params == {"hello": "world"}
    assert msg.id == 42


def test_format_and_parse_notification():
    payload = format_notification("heartbeat", {"tick": 1})
    assert payload.endswith(b"\n")

    msg = parse_message(payload)
    assert isinstance(msg, RpcNotification)
    assert msg.jsonrpc == "2.0"
    assert msg.method == "heartbeat"
    assert msg.params == {"tick": 1}
    assert not hasattr(msg, "id") or msg.id is None


def test_format_and_parse_response():
    payload = format_response({"status": "ok"}, msg_id=42)
    assert payload.endswith(b"\n")

    data = json.loads(payload.decode("utf-8"))
    assert data == {"jsonrpc": "2.0", "result": {"status": "ok"}, "id": 42}


def test_format_error_standard_codes():
    payload = format_error(METHOD_NOT_FOUND, "Method not found", msg_id=1)
    data = json.loads(payload.decode("utf-8"))
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == 1
    assert data["error"]["code"] == -32601
    assert data["error"]["message"] == "Method not found"


def test_parse_invalid_json():
    with pytest.raises(JsonRpcError) as exc_info:
        parse_message(b"this is not json\n")
    assert exc_info.value.code == PARSE_ERROR


def test_parse_invalid_rpc_structure():
    # Not a dictionary
    with pytest.raises(JsonRpcError) as exc_info:
        parse_message(b'"just a string"\n')
    assert exc_info.value.code == INVALID_REQUEST

    # Missing jsonrpc="2.0"
    with pytest.raises(JsonRpcError) as exc_info:
        parse_message(b'{"method": "ping", "id": 1}\n')
    assert exc_info.value.code == INVALID_REQUEST

    # Missing method
    with pytest.raises(JsonRpcError) as exc_info:
        parse_message(b'{"jsonrpc": "2.0", "id": 1}\n')
    assert exc_info.value.code == INVALID_REQUEST
