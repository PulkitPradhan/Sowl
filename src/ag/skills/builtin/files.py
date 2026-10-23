"""Secure file reading and listing tools with strict containment and deny patterns."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any

# Default deny patterns protecting sensitive credentials and keys
DEFAULT_DENY_PATTERNS = [
    ".env*",
    "*.pem",
    "id_rsa*",
    ".ssh*",
    "*.key",
]

MAX_READ_BYTES = 100_000  # 100 KB read cap


def get_default_workspace_root() -> Path:
    """Return default workspace directory ~/.ag/workspace/ or fallback."""
    env_dir = os.environ.get("AG_WORKSPACE_DIR")
    if env_dir:
        candidate = Path(env_dir).expanduser()
        if (candidate / "workspace").is_dir():
            return candidate / "workspace"
        return candidate
    target = Path("~/.ag/workspace").expanduser()
    return target


def _is_denied(path: Path) -> bool:
    """Check if any path component or file name matches deny patterns."""
    for part in path.parts:
        for pattern in DEFAULT_DENY_PATTERNS:
            if fnmatch.fnmatch(part.lower(), pattern.lower()):
                return True
    return False


def _resolve_and_contain(path_str: str, root: Path | None = None) -> tuple[bool, Path | None, str | None]:
    """Resolve target path and verify containment within the configured root directory."""
    target_root = (root or get_default_workspace_root()).resolve()
    target_path = Path(path_str)

    # If relative path, join with root
    if not target_path.is_absolute():
        resolved_path = (target_root / target_path).resolve()
    else:
        resolved_path = target_path.resolve()

    # Containment check via is_relative_to
    if not resolved_path.is_relative_to(target_root):
        return False, None, f"Path traversal rejected: '{path_str}' escapes root '{target_root}'"

    # Check sensitive deny patterns
    if _is_denied(resolved_path):
        return False, None, f"Access denied: '{path_str}' matches sensitive file protection pattern"

    return True, resolved_path, None


def read_file(path: str, root: Path | None = None) -> dict[str, Any]:
    """Read contents of a text file within root, capped at 100 KB."""
    ok, target, err = _resolve_and_contain(path, root)
    if not ok or target is None:
        return {"ok": False, "data": None, "error": err}

    if not target.is_file():
        return {"ok": False, "data": None, "error": f"File not found: '{path}'"}

    try:
        # Check for binary file first (first 4096 bytes)
        with target.open("rb") as f:
            chunk = f.read(4096)
            if b"\x00" in chunk:
                return {"ok": False, "data": None, "error": f"Refusing to read binary file: '{path}'"}

            f.seek(0)
            raw = f.read(MAX_READ_BYTES + 1)

        truncated = len(raw) > MAX_READ_BYTES
        content_bytes = raw[:MAX_READ_BYTES]
        content_str = content_bytes.decode("utf-8", errors="replace")

        return {
            "ok": True,
            "data": {
                "path": str(target),
                "content": content_str,
                "truncated": truncated,
                "bytes_read": len(content_bytes),
            },
            "error": None,
        }
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "data": None, "error": f"Error reading file '{path}': {err}"}


def list_files(path: str = ".", root: Path | None = None) -> dict[str, Any]:
    """List directory contents within root."""
    ok, target, err = _resolve_and_contain(path, root)
    if not ok or target is None:
        return {"ok": False, "data": None, "error": err}

    if not target.is_dir():
        return {"ok": False, "data": None, "error": f"Directory not found: '{path}'"}

    try:
        entries: list[dict[str, Any]] = []
        for item in sorted(target.iterdir()):
            if _is_denied(item):
                continue
            entries.append({
                "name": item.name,
                "is_dir": item.is_dir(),
                "size_bytes": item.stat().st_size if item.is_file() else None,
            })
        return {
            "ok": True,
            "data": {
                "directory": str(target),
                "entries": entries,
            },
            "error": None,
        }
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "data": None, "error": f"Error listing directory '{path}': {err}"}
