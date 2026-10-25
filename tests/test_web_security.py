"""Tests for Web server security middleware, headers, host/origin validation, and token authentication."""

from __future__ import annotations

import asyncio

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from ag.web.server import create_web_app


class TestWebSecurity(AioHTTPTestCase):
    async def get_application(self) -> web.Application:
        self.startup_token = "test_startup_token_128bit_minimum_entropy"
        self.app = create_web_app(
            startup_token=self.startup_token,
            port=3180,
            token_ttl=60.0,
        )
        return self.app

    async def test_wrong_host_header_returns_403(self):
        """Requests with non-loopback Host header must return 403 Forbidden."""
        resp = await self.client.get(
            "/",
            headers={"Host": "attacker.com"},
        )
        assert resp.status == 403
        text = await resp.text()
        assert "Forbidden: Invalid Host header" in text

    async def test_wrong_origin_header_returns_403(self):
        """State-changing requests with non-loopback Origin header must return 403 Forbidden."""
        resp = await self.client.post(
            "/api/auth",
            headers={"Host": "127.0.0.1:3180", "Origin": "http://evil.com"},
            json={"token": self.startup_token},
        )
        assert resp.status == 403
        text = await resp.text()
        assert "Forbidden: Invalid Origin header" in text

    async def test_security_headers_present(self):
        """All responses must carry strict defense-in-depth security headers."""
        resp = await self.client.get(
            "/",
            headers={"Host": "127.0.0.1:3180"},
        )
        headers = resp.headers

        assert headers.get("X-Frame-Options") == "DENY"
        assert headers.get("X-Content-Type-Options") == "nosniff"
        assert headers.get("Referrer-Policy") == "no-referrer"

        csp = headers.get("Content-Security-Policy", "")
        assert "default-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "base-uri 'none'" in csp
        assert "form-action 'none'" in csp
        assert "object-src 'none'" in csp
        assert "'unsafe-inline'" not in csp

    async def test_api_cache_control_no_store(self):
        """API responses must specify Cache-Control: no-store."""
        resp = await self.client.get(
            "/api/sessions",
            headers={"Host": "127.0.0.1:3180"},
        )
        assert resp.headers.get("Cache-Control") == "no-store"

    async def test_token_exchange_sets_cookie_and_is_single_use(self):
        """Startup token must exchange for an HttpOnly SameSite=Strict session cookie and be single-use."""
        # 1. Invalid token fails
        fail_resp = await self.client.post(
            "/api/auth",
            headers={"Host": "127.0.0.1:3180", "Origin": "http://127.0.0.1:3180"},
            json={"token": "wrong_token"},
        )
        assert fail_resp.status == 401

        # 2. Valid token succeeds and sets cookie
        success_resp = await self.client.post(
            "/api/auth",
            headers={"Host": "127.0.0.1:3180", "Origin": "http://127.0.0.1:3180"},
            json={"token": self.startup_token},
        )
        assert success_resp.status == 200
        cookie_header = success_resp.headers.get("Set-Cookie", "")
        assert "sowl_session=" in cookie_header
        assert "HttpOnly" in cookie_header
        assert "SameSite=Strict" in cookie_header

        # 3. Single-use: Replaying the same startup token must fail (401)
        replay_resp = await self.client.post(
            "/api/auth",
            headers={"Host": "127.0.0.1:3180", "Origin": "http://127.0.0.1:3180"},
            json={"token": self.startup_token},
        )
        assert replay_resp.status == 401

    async def test_token_expiry(self):
        """Expired startup token (>60s) must return 401."""
        expired_app = create_web_app(
            startup_token="expired_token",
            port=3199,
            token_ttl=0.01,  # 10ms expiry
        )
        # Fast-forward time
        await asyncio.sleep(0.02)

        # Make test request against expired_app runner
        runner = web.AppRunner(expired_app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 3199)
        await site.start()

        try:
            import httpx

            async with httpx.AsyncClient(base_url="http://127.0.0.1:3199") as client:
                resp = await client.post(
                    "/api/auth",
                    headers={"Host": "127.0.0.1:3199", "Origin": "http://127.0.0.1:3199"},
                    json={"token": "expired_token"},
                )
                assert resp.status_code == 401
        finally:
            await runner.cleanup()

    async def test_unauthenticated_api_request_returns_401(self):
        """Accessing API endpoints without a valid session cookie returns 401."""
        resp = await self.client.get(
            "/api/sessions",
            headers={"Host": "127.0.0.1:3180"},
        )
        assert resp.status == 401
