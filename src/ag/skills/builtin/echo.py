"""Safe echo skill handlers for testing."""

from __future__ import annotations

from typing import Any


def echo(text: str) -> dict[str, Any]:
    """Echo text input for safe testing."""
    return {
        "ok": True,
        "data": {"echo": text},
        "error": None,
    }


def failing_tool() -> dict[str, Any]:
    """Intentionally raise an exception to verify error handling."""
    raise RuntimeError("Intentional tool failure for testing")
