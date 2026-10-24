"""Pluggable launcher backend: real OS web browser vs safe test/eval mock."""

from __future__ import annotations

import os
import urllib.parse
import webbrowser
from typing import Any, Protocol


class LauncherBackend(Protocol):
    """Protocol for launching external web URLs and draft applications."""

    def open(self, url: str) -> bool:
        """Open target URL in external application."""
        ...


class RealLauncherBackend:
    """Production launcher calling Python webbrowser. Refuses execution if AG_EVAL=1."""

    def open(self, url: str) -> bool:
        if os.environ.get("AG_EVAL") == "1":
            raise RuntimeError("Real launcher execution strictly forbidden when AG_EVAL=1")
        return bool(webbrowser.open(url))


class MockLauncherBackend:
    """Mock launcher capturing opened URLs and structured Gmail draft payloads."""

    def __init__(self) -> None:
        self.opened_urls: list[str] = []

    def open(self, url: str) -> bool:
        self.opened_urls.append(url)
        return True

    def get_gmail_calls(self) -> list[dict[str, Any]]:
        """Parse captured Gmail compose URLs into structured draft records."""
        calls = []
        for u in self.opened_urls:
            if "mail.google.com" in u and "view=cm" in u:
                parsed = urllib.parse.urlparse(u)
                qs = urllib.parse.parse_qs(parsed.query)
                calls.append({
                    "url": u,
                    "to": qs.get("to", [""])[0],
                    "subject": qs.get("su", [""])[0],
                    "body": qs.get("body", [""])[0],
                })
        return calls


_current_launcher: LauncherBackend = RealLauncherBackend()


def get_launcher() -> LauncherBackend:
    """Return active launcher backend."""
    return _current_launcher


def set_launcher(launcher: LauncherBackend) -> None:
    """Override active launcher backend."""
    global _current_launcher
    _current_launcher = launcher
