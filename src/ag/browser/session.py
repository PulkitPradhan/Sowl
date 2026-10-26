"""Browser session management over Playwright CDP with safety constraints, URL verification, and ref resolution."""

from __future__ import annotations

import asyncio
import logging
import urllib.parse
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from ag.browser.config import BrowserConfig, load_browser_config
from ag.browser.process import get_browser_status
from ag.browser.snapshot import SnapshotEngine

logger = logging.getLogger(__name__)

DISALLOWED_SCHEMES = {"file", "chrome", "view-source", "data", "javascript"}


class BrowserSession:
    """Singleton-style active browser session attached via Playwright CDP."""

    def __init__(self, cdp_url: str = "http://127.0.0.1:9222", config: BrowserConfig | None = None) -> None:
        self.cdp_url = cdp_url
        self.config = config or load_browser_config()
        self.playwright: Any = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.current_page: Page | None = None
        self.snapshot_engine = SnapshotEngine()
        self.action_count: int = 0
        self.scroll_count: int = 0
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Connect Playwright to running Chrome CDP instance."""
        if self.browser and self.browser.is_connected():
            return

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.connect_over_cdp(self.cdp_url)

        if not self.browser.contexts:
            self.context = await self.browser.new_context()
        else:
            self.context = self.browser.contexts[0]

        # Ensure active page
        pages = self.context.pages
        if pages:
            self.current_page = pages[0]
        else:
            self.current_page = await self.context.new_page()

        self._attach_page_guards(self.current_page)

    def _attach_page_guards(self, page: Page) -> None:
        """Attach dialog and security handlers to page."""
        # Dismiss and log JS dialogs
        page.on("dialog", lambda dialog: asyncio.create_task(self._handle_dialog(dialog)))

    async def _handle_dialog(self, dialog: Any) -> None:
        logger.warning("Dismissed unexpected JS dialog: type=%s, message=%s", dialog.type, dialog.message)
        try:
            await dialog.dismiss()
        except Exception:  # noqa: BLE001, S110
            pass

    async def get_page(self) -> Page:
        """Retrieve current active page while enforcing tab limits."""
        await self.connect()
        assert self.context is not None

        # Enforce tab cap (~5 tabs)
        pages = self.context.pages
        while len(pages) > self.config.max_tabs:
            extra = pages[-1]
            try:
                await extra.close()
            except Exception:  # noqa: BLE001, S110
                pass
            pages = self.context.pages

        if not self.current_page or self.current_page.is_closed():
            self.current_page = pages[0] if pages else await self.context.new_page()
            self._attach_page_guards(self.current_page)

        return self.current_page

    def check_url_safety(self, url: str) -> tuple[bool, str]:
        """Check URL scheme safety. Block file:, chrome:, data:, javascript:, etc."""
        try:
            parsed = urllib.parse.urlparse(url)
            scheme = parsed.scheme.lower()
            if scheme in DISALLOWED_SCHEMES:
                return False, f"Disallowed URL scheme '{scheme}:'. Execution is blocked."
            if scheme not in ("http", "https"):
                return False, f"Only http and https protocols are permitted (got '{scheme}:')."
        except Exception as err:  # noqa: BLE001
            return False, f"Invalid URL format: {err}"
        return True, ""

    async def apply_inter_action_delay(self, url: str) -> None:
        """Enforce domain-specific delay for page settling."""
        delay_s = self.config.get_delay_for_url(url)
        if delay_s > 0:
            await asyncio.sleep(delay_s)

    async def goto(self, url: str) -> dict[str, Any]:
        """Navigate to URL with safety checks and limits."""
        async with self._lock:
            self.action_count += 1
            if self.action_count > self.config.max_actions:
                return {"ok": False, "error": f"Max actions cap reached ({self.config.max_actions})"}

            safe, err = self.check_url_safety(url)
            if not safe:
                return {"ok": False, "error": err}

            page = await self.get_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await self.apply_inter_action_delay(page.url)

            return {
                "ok": True,
                "data": {
                    "url": page.url,
                    "title": await page.title(),
                },
                "error": None,
            }

    async def click_ref(self, ref: str, engine: SnapshotEngine | None = None) -> dict[str, Any]:
        """Re-resolve element by ref, verify role and name against snapshot, then click."""
        async with self._lock:
            self.action_count += 1
            if self.action_count > self.config.max_actions:
                return {"ok": False, "error": f"Max actions cap reached ({self.config.max_actions})"}

            eng = engine or self.snapshot_engine
            elem = eng.get_element(ref)
            if not elem:
                return {
                    "ok": False,
                    "error": f"Invalid or stale ref '{ref}'. Please call browser.snapshot to refresh element refs.",
                }

            page = await self.get_page()

            # Re-resolve element on page
            locator = page.locator(elem.selector).first
            count = await locator.count()
            if count == 0:
                return {
                    "ok": False,
                    "error": f"Stale ref '{ref}': element not found on current page. Call browser.snapshot to re-capture.",
                }

            # Verify visible
            if not await locator.is_visible():
                return {
                    "ok": False,
                    "error": f"Element '{ref}' is no longer visible. Call browser.snapshot to re-capture.",
                }

            # Check fresh role/name mismatch
            try:
                fresh_text = (await locator.inner_text()).strip()
            except Exception:  # noqa: BLE001
                fresh_text = ""

            if elem.name and fresh_text and elem.name.lower() not in fresh_text.lower() and fresh_text.lower() not in elem.name.lower():
                return {
                    "ok": False,
                    "error": f"Element mismatch on '{ref}': expected '{elem.name}', found '{fresh_text}'. Call browser.snapshot to re-capture.",
                }

            # Perform click
            url_before = page.url
            await locator.click(timeout=10000)
            await self.apply_inter_action_delay(page.url)
            url_after = page.url

            return {
                "ok": True,
                "data": {
                    "ref": ref,
                    "action": "clicked",
                    "url_before": url_before,
                    "url_after": url_after,
                    "page_changed": (url_before != url_after),
                },
                "error": None,
            }

    async def type_text(self, ref: str, text: str, engine: SnapshotEngine | None = None) -> dict[str, Any]:
        """Type text into element after verifying it is not a password or payment field."""
        async with self._lock:
            self.action_count += 1
            if self.action_count > self.config.max_actions:
                return {"ok": False, "error": f"Max actions cap reached ({self.config.max_actions})"}

            eng = engine or self.snapshot_engine
            elem = eng.get_element(ref)
            if not elem:
                return {
                    "ok": False,
                    "error": f"Invalid or stale ref '{ref}'. Please call browser.snapshot to refresh element refs.",
                }

            if elem.is_password:
                return {"ok": False, "error": "Security refusal: typing into password fields is prohibited."}
            if elem.is_card:
                return {"ok": False, "error": "Security refusal: typing into payment/card fields is prohibited."}
            if elem.is_otp:
                return {"ok": False, "error": "Security refusal: typing into one-time-code fields is prohibited."}

            page = await self.get_page()
            locator = page.locator(elem.selector).first
            if await locator.count() == 0 or not await locator.is_visible():
                return {"ok": False, "error": f"Element '{ref}' is no longer visible or present."}

            await locator.fill(text, timeout=10000)
            await self.apply_inter_action_delay(page.url)

            # Post-check: verify element value matches typed text
            typed_val = await locator.input_value()
            return {
                "ok": True,
                "data": {
                    "ref": ref,
                    "typed": text,
                    "value_verified": (typed_val == text),
                },
                "error": None,
            }

    async def scroll(self, direction: str = "down", amount: str | int = "page") -> dict[str, Any]:
        """Scroll page up or down."""
        async with self._lock:
            self.scroll_count += 1
            if self.scroll_count > self.config.max_scrolls:
                return {"ok": False, "error": f"Max scrolls cap reached ({self.config.max_scrolls})"}

            page = await self.get_page()
            pixels = 500
            if isinstance(amount, int):
                pixels = min(max(amount, 100), 1000)
            elif amount == "page":
                pixels = 800
            elif amount == "half_page":
                pixels = 400
            elif amount == "small":
                pixels = 200

            delta_y = pixels if direction.lower() == "down" else -pixels
            await page.evaluate(f"window.scrollBy(0, {delta_y})")
            await self.apply_inter_action_delay(page.url)

            scroll_y = await page.evaluate("window.scrollY")
            return {
                "ok": True,
                "data": {
                    "direction": direction,
                    "pixels": delta_y,
                    "scroll_y": scroll_y,
                },
                "error": None,
            }

    async def close(self) -> None:
        """Close browser CDP connection."""
        if self.browser:
            try:
                await self.browser.close()
            except Exception:  # noqa: BLE001, S110
                pass
            self.browser = None
        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception:  # noqa: BLE001, S110
                pass
            self.playwright = None


_global_session: BrowserSession | None = None


async def reset_browser_session() -> None:
    """Close and clear the global browser session."""
    global _global_session
    if _global_session is not None:
        try:
            await _global_session.close()
        except Exception:  # noqa: BLE001, S110
            pass
        _global_session = None


async def get_browser_session(cdp_url: str | None = None) -> BrowserSession:
    """Retrieve or create singleton browser session."""
    global _global_session
    needs_reconnect = (
        _global_session is None
        or _global_session.browser is None
        or not _global_session.browser.is_connected()
        or (cdp_url is not None and _global_session.cdp_url != cdp_url)
    )

    if needs_reconnect:
        if _global_session is not None:
            try:
                await _global_session.close()
            except Exception:  # noqa: BLE001, S110
                pass
            _global_session = None

        url = cdp_url
        if not url:
            status = await get_browser_status()
            if status["running"] and status["cdp_url"]:
                url = status["cdp_url"]
            else:
                url = "http://127.0.0.1:9222"
        _global_session = BrowserSession(cdp_url=url)
        await _global_session.connect()
    return _global_session
