"""Browser opening tool restricted strictly to http and https schemes."""

from __future__ import annotations

import urllib.parse
from typing import Any

from ag.skills.launcher import get_launcher

ALLOWED_SCHEMES = {"http", "https"}


def open_url(url: str) -> dict[str, Any]:
    """Open URL in default web browser, strictly restricted to http/https schemes."""
    if not url or not isinstance(url, str):
        return {"ok": False, "data": None, "error": "Invalid URL: must be a non-empty string"}

    try:
        parsed = urllib.parse.urlparse(url.strip())
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "data": None, "error": f"Failed to parse URL '{url}': {err}"}

    scheme = parsed.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        return {
            "ok": False,
            "data": None,
            "error": f"Disallowed URL scheme '{scheme}'. Only http and https schemes are permitted.",
        }

    try:
        get_launcher().open(url.strip())
        return {
            "ok": True,
            "data": {
                "url": url.strip(),
                "action": "opened",
                "note": "URL opened in default web browser",
            },
            "error": None,
        }
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "data": None, "error": f"Failed to open browser for '{url}': {err}"}
