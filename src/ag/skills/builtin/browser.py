"""Browser automation skill handlers powered by Playwright CDP and accessibility snapshots."""

from __future__ import annotations

import asyncio
import re
import urllib.parse
from typing import TYPE_CHECKING, Any

from ag.browser.session import get_browser_session
from ag.skills.launcher import get_launcher

if TYPE_CHECKING:
    from playwright.async_api import Page

ALLOWED_SCHEMES = {"http", "https"}


# --- Default Browser Tool (Uncontrolled System Browser) ---

def open_url(url: str) -> dict[str, Any]:
    """Open URL in default system web browser, strictly restricted to http/https schemes."""
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
                "note": "URL opened in default system web browser",
            },
            "error": None,
        }
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "data": None, "error": f"Failed to open browser for '{url}': {err}"}


# --- Controlled Browser Tools (Attached Dedicated Profile over CDP) ---

async def goto(url: str) -> dict[str, Any]:
    """Navigate the controlled browser page to an HTTP/HTTPS URL."""
    try:
        session = await get_browser_session()
        return await session.goto(url)
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "data": None, "error": f"Failed to navigate: {err}"}


async def snapshot() -> dict[str, Any]:
    """Capture URL, title, and a compact TOON table of visible interactive elements."""
    try:
        session = await get_browser_session()
        page = await session.get_page()
        res = await session.snapshot_engine.take_snapshot(page)
        return {
            "ok": True,
            "data": {
                "url": res.url,
                "title": res.title,
                "snapshot": res.toon_table,
            },
            "error": None,
        }
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "data": None, "error": f"Snapshot failed: {err}"}


async def read_text(ref: str | None = None, max_chars: int = 2000) -> dict[str, Any]:
    """Extract visible text content from the active page or a specific element ref."""
    try:
        session = await get_browser_session()
        page = await session.get_page()
        cap = min(max(max_chars, 100), 8000)

        if ref:
            elem = session.snapshot_engine.get_element(ref)
            if not elem:
                return {
                    "ok": False,
                    "error": f"Invalid or stale ref '{ref}'. Please call browser.snapshot to refresh element refs.",
                }
            locator = page.locator(elem.selector).first
            text = await locator.inner_text()
        else:
            text = await page.evaluate("() => document.body ? document.body.innerText : ''")

        text = re.sub(r"\s+", " ", text or "").strip()
        if len(text) > cap:
            text = text[:cap] + "…"

        return {
            "ok": True,
            "data": {
                "text": text,
                "length": len(text),
            },
            "error": None,
        }
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "data": None, "error": f"read_text failed: {err}"}


async def extract_links(
    page: Page, patterns: list[str] | None = None, max_results: int = 100
) -> dict[str, Any]:
    """Extract deduplicated href links from page matching any substring patterns."""
    raw_hrefs: list[str] = await page.evaluate(
        """() => Array.from(document.querySelectorAll('a[href]')).map(a => a.href)"""
    )
    seen: set[str] = set()
    matched: list[str] = []

    for href in raw_hrefs:
        if not href or href in seen:
            continue
        if patterns:
            if any(pat in href for pat in patterns):
                seen.add(href)
                matched.append(href)
        else:
            seen.add(href)
            matched.append(href)

        if len(matched) >= max_results:
            break

    return {
        "ok": True,
        "data": {
            "links": matched,
            "count": len(matched),
        },
        "error": None,
    }


async def links(patterns: list[str] | None = None, max: int = 100) -> dict[str, Any]:
    """Return deduplicated hrefs from the controlled page matching substring patterns."""
    try:
        session = await get_browser_session()
        page = await session.get_page()
        return await extract_links(page, patterns=patterns, max_results=max)
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "data": None, "error": f"links failed: {err}"}


async def scroll(direction: str = "down", amount: str = "page") -> dict[str, Any]:
    """Scroll the active controlled page up or down."""
    try:
        session = await get_browser_session()
        return await session.scroll(direction=direction, amount=amount)
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "data": None, "error": f"scroll failed: {err}"}


async def click(ref: str) -> dict[str, Any]:
    """Click an interactive element identified by ref from the latest snapshot."""
    try:
        session = await get_browser_session()
        return await session.click_ref(ref)
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "data": None, "error": f"click failed: {err}"}


async def type_text(ref: str, text: str) -> dict[str, Any]:
    """Type text into an editable element ref, strictly refusing password and payment fields."""
    try:
        session = await get_browser_session()
        return await session.type_text(ref, text)
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "data": None, "error": f"type failed: {err}"}


async def back() -> dict[str, Any]:
    """Navigate backward in browser history."""
    try:
        session = await get_browser_session()
        page = await session.get_page()
        await page.go_back(wait_until="domcontentloaded", timeout=10000)
        return {
            "ok": True,
            "data": {
                "url": page.url,
                "title": await page.title(),
            },
            "error": None,
        }
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "data": None, "error": f"back failed: {err}"}


async def wait(seconds: float = 1.0) -> dict[str, Any]:
    """Wait for a bounded duration (up to 10 seconds) for asynchronous page updates."""
    sec = min(max(float(seconds), 0.1), 10.0)
    await asyncio.sleep(sec)
    return {
        "ok": True,
        "data": {"waited_seconds": sec},
        "error": None,
    }
