"""Wire format serialization (TOON and compact JSON), stripping, and token counting."""

from __future__ import annotations

import json
import re
from typing import Any

import toon_format


class WireError(Exception):
    """Base error for wire serialization and deserialization."""


class TruncatedOutputError(WireError):
    """Raised when declared [N] does not match actual row count or finish_reason='length'."""


def _strip_fences_and_prose(text: str) -> str:
    """Extract code block if enclosed in markdown code fences, or return stripped text."""
    # Look for ```[lang]\n...\n``` block
    fence_pattern = re.compile(r"```(?:[a-zA-Z0-9_\-]+)?\s*\n(.*?)\n```", re.DOTALL)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()

    # Also handle leading and trailing ``` on single/multiline without interior newlines
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```") and len(stripped) >= 6:
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()

    return stripped


def to_llm(obj: Any, fmt: str = "toon_in") -> str:
    """Encode Python structure into compact JSON or TOON for LLM consumption.

    fmt options:
    - 'json': compact JSON without extra whitespace.
    - 'toon_in' / 'toon_both': TOON format (Token-Oriented Object Notation).
    """
    if fmt not in ("json", "toon_in", "toon_both"):
        raise WireError(f"Invalid wire format '{fmt}'. Expected 'json', 'toon_in', or 'toon_both'.")

    if fmt == "json":
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

    try:
        return toon_format.dumps(obj)
    except Exception as err:
        raise WireError(f"TOON encoding error: {err}") from err


def from_llm(text: str, fmt: str = "toon_in", finish_reason: str | None = None) -> Any:
    """Decode LLM response (stripping markdown code fences and prose).

    fmt options:
    - 'json' or 'toon_in': expects JSON first, falls back to TOON.
    - 'toon_both': expects TOON first, falls back to JSON.
    """
    if fmt not in ("json", "toon_in", "toon_both"):
        raise WireError(f"Invalid wire format '{fmt}'. Expected 'json', 'toon_in', or 'toon_both'.")

    if len(text.encode("utf-8")) > 1_000_000:
        raise WireError("Input exceeds maximum size of 1 MB")

    if finish_reason == "length":
        raise TruncatedOutputError("Model response was truncated (finish_reason='length')")

    cleaned = _strip_fences_and_prose(text)

    def _try_json(s: str) -> tuple[bool, Any]:
        try:
            return True, json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return False, None

    def _try_toon(s: str) -> tuple[bool, Any, Exception | None]:
        try:
            return True, toon_format.loads(s), None
        except Exception as err:  # noqa: BLE001
            return False, None, err

    # If the text is clearly enclosed in JSON object/array braces, check JSON
    looks_like_json = (
        (cleaned.startswith("{") and cleaned.endswith("}"))
        or (cleaned.startswith("[") and cleaned.endswith("]") and not re.match(r"^\[\d+\]", cleaned))
    )

    if fmt == "toon_both":
        # If looks like JSON, parse JSON first
        if looks_like_json:
            ok_json, val_json = _try_json(cleaned)
            if ok_json:
                return val_json

        # Try TOON first
        ok_toon, val_toon, err_toon = _try_toon(cleaned)
        if ok_toon:
            return val_toon

        # If declared row count mismatch occurred, raise TruncatedOutputError directly
        if isinstance(err_toon, toon_format.ToonDecodeError):
            err_msg = str(err_toon).lower()
            if "declared length" in err_msg or "found" in err_msg or "expected" in err_msg:
                raise TruncatedOutputError(f"Declared length mismatch: {err_toon}") from err_toon

        # Fall back to JSON
        ok_json, val_json = _try_json(cleaned)
        if ok_json:
            return val_json

        raise WireError(f"Failed to decode message as TOON or JSON: {err_toon}") from err_toon

    # fmt is 'json' or 'toon_in': Expected format is JSON, fallback to TOON
    ok_json, val_json = _try_json(cleaned)
    if ok_json:
        return val_json

    ok_toon, val_toon, err_toon = _try_toon(cleaned)
    if ok_toon:
        return val_toon

    if isinstance(err_toon, toon_format.ToonDecodeError):
        err_msg = str(err_toon).lower()
        if "declared length" in err_msg or "found" in err_msg or "expected" in err_msg:
            raise TruncatedOutputError(f"Declared length mismatch: {err_toon}") from err_toon

    raise WireError(f"Failed to decode message as JSON or TOON: {err_toon}") from err_toon


def count_tokens(text: str) -> tuple[int, str]:
    """Count tokens in text string.

    Returns (count, label).
    Uses tiktoken (cl100k_base) if available, falling back to a character heuristic.
    Always labeled 'approx (cl100k)' per specification.
    """
    label = "approx (cl100k)"
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text)), label
    except Exception:  # noqa: BLE001
        token_est = max(1, int(len(text) / 3.8))
        return token_est, label
