"""ag wire: JSON-RPC 2.0 framing and TOON token serialization."""

from ag.wire.rpc import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    JsonRpcError,
    RpcNotification,
    RpcRequest,
    RpcResponse,
    format_error,
    format_notification,
    format_request,
    format_response,
    get_default_socket_path,
    parse_message,
)
from ag.wire.wire import (
    TruncatedOutputError,
    WireError,
    count_tokens,
    from_llm,
    to_llm,
)

__all__ = [
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "PARSE_ERROR",
    "JsonRpcError",
    "RpcNotification",
    "RpcRequest",
    "RpcResponse",
    "TruncatedOutputError",
    "WireError",
    "count_tokens",
    "format_error",
    "format_notification",
    "format_request",
    "format_response",
    "from_llm",
    "get_default_socket_path",
    "parse_message",
    "to_llm",
]
