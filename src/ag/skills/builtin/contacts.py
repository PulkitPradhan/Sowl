"""Contacts skill: Lookup contact details in contacts.toml."""

from __future__ import annotations

import os
import stat
import tomllib
from pathlib import Path
from typing import Any


def get_default_contacts_path(workspace_dir: Path | None = None) -> Path:
    """Resolve contacts.toml path from workspace or fallback to ~/.ag/workspace."""
    if workspace_dir:
        return Path(workspace_dir) / "contacts.toml"
    env_dir = os.environ.get("AG_WORKSPACE_DIR")
    if env_dir:
        return Path(env_dir).expanduser() / "workspace" / "contacts.toml"
    return Path("~/.ag/workspace/contacts.toml").expanduser()


def find_contact(name: str, workspace_dir: Path | None = None) -> dict[str, Any]:
    """Search contacts.toml for matching name (case-insensitive full or first name).

    Returns up to 5 matching candidates. Warns if contacts.toml is group- or world-writable.
    """
    contacts_path = get_default_contacts_path(workspace_dir)
    warning: str | None = None

    if not contacts_path.is_file():
        return {
            "found": False,
            "count": 0,
            "candidates": [],
            "error": f"Contacts file not found at {contacts_path}",
        }

    # Security check: check file permissions for group/world-writable
    try:
        mode = contacts_path.stat().st_mode
        if mode & (stat.S_IWGRP | stat.S_IWOTH):
            warning = f"Security warning: contacts file {contacts_path} is group- or world-writable"
    except OSError:
        pass

    try:
        content = contacts_path.read_text(encoding="utf-8")
        data = tomllib.loads(content)
    except Exception as err:  # noqa: BLE001
        return {
            "found": False,
            "count": 0,
            "candidates": [],
            "error": f"Failed to parse contacts.toml: {err}",
            "warning": warning,
        }

    contacts_list = data.get("contact") or data.get("contacts") or []
    if isinstance(contacts_list, dict):
        contacts_list = [contacts_list]

    target = name.strip().lower()
    matches: list[dict[str, Any]] = []

    for c in contacts_list:
        if not isinstance(c, dict):
            continue
        c_name = str(c.get("name", "")).strip().lower()
        c_first = c_name.split()[0] if c_name else ""
        if target == c_name or target == c_first:
            matches.append(c)

    count = len(matches)
    capped_matches = matches[:5]

    res: dict[str, Any] = {
        "found": count > 0,
        "count": count,
        "candidates": capped_matches,
    }
    if warning:
        res["warning"] = warning

    if count == 1:
        res["match"] = capped_matches[0]
        res["ambiguous"] = False
    elif count > 1:
        res["ambiguous"] = True

    return res
