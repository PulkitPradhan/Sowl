"""Tests for Phase 9: Browser control via Playwright CDP, snapshot engine, safety escalation, and taint."""

from __future__ import annotations

import functools
import http.server
import socket
import threading
from pathlib import Path

import pytest
import toon_format

from ag.browser.config import (
    is_engagement_action,
    is_escalation_label,
    load_browser_config,
)
from ag.browser.process import (
    find_chrome_executable,
    get_browser_status,
    start_browser,
    stop_browser,
)
from ag.browser.session import (
    get_browser_session,
    reset_browser_session,
)
from ag.browser.snapshot import SnapshotEngine
from ag.laya.gate import LayaGate
from ag.laya.provenance import ProvenanceStore
from ag.skills.manifest import ArgField, ToolManifest

# --- Fixture Server ---

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture(scope="module")
def fixture_server():
    """Spin up local loopback HTTP server serving tests/fixtures/browser/."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "browser"
    handler = functools.partial(QuietHandler, directory=str(fixtures_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def has_chrome() -> bool:
    return find_chrome_executable() is not None


chrome_required = pytest.mark.skipif(not has_chrome(), reason="Chrome/Chromium binary not found")


# --- Unit Tests: Configuration & Escalation Matching ---

def test_escalation_labels_matching() -> None:
    # Test case-insensitive, word-boundary, normalized labels
    assert is_escalation_label("Send")
    assert is_escalation_label("send message")
    assert is_escalation_label("SUBMIT FORM")
    assert is_escalation_label("Place Order")
    assert is_escalation_label("Checkout")
    assert is_escalation_label("Delete Account")
    assert is_escalation_label("Pay Now")
    assert is_escalation_label("Transfer funds")
    assert is_escalation_label("Confirm changes")
    assert is_escalation_label("Buy tickets")
    assert is_escalation_label("Donate $10")
    assert is_escalation_label("Agree and continue")
    assert is_escalation_label("Accept terms")

    # Non-escalating labels
    assert not is_escalation_label("View details")
    assert not is_escalation_label("Refresh")
    assert not is_escalation_label("Next page")
    assert not is_escalation_label("Ascending")  # Contains "send" as substring, but word boundary prevents match


def test_engagement_guard_rules() -> None:
    # Social domains engagement check
    social_domain = "instagram.com"
    non_social_domain = "news.ycombinator.com"

    for action in ["like", "follow", "comment", "reply", "share", "repost", "post", "message", "subscribe"]:
        assert is_engagement_action(action, social_domain) is True
        assert is_engagement_action(action, non_social_domain) is False

    assert is_engagement_action("view", social_domain) is False


def test_load_browser_config(tmp_path: Path) -> None:
    config_file = tmp_path / "browser.toml"
    config_file.write_text(
        """
min_delay_ms = 300
wait_cap_s = 8.0
social_domains = ["instagram.com", "custom.social"]
[domain_delays_ms]
"instagram.com" = 2000
"custom.social" = 1500
""",
        encoding="utf-8",
    )
    cfg = load_browser_config(config_file)
    assert cfg.min_delay_ms == 300
    assert cfg.wait_cap_s == 8.0
    assert "custom.social" in cfg.social_domains
    assert cfg.get_delay_for_url("https://instagram.com/p/123") == 2.0
    assert cfg.get_delay_for_url("https://example.com") == 0.3


# --- Unit Tests: Laya Gate Browser Risk Escalation ---

def test_laya_escalation_for_browser_click() -> None:
    gate = LayaGate()
    store = ProvenanceStore()
    store.add_user_text("Click @e1 or @e2 or @e3")

    manifest_click = ToolManifest(
        name="browser.click",
        summary="Click ref",
        risk="R1",
        args={"ref": ArgField(type="string", kind="identifier")},
    )

    # 1. Normal button click -> R1 (confirm by default)
    action_normal = {
        "tool": "browser.click",
        "args": {"ref": "@e1"},
        "_element_label": "View Details",
        "_page_url": "https://example.com",
    }
    dec_normal = gate.evaluate(action_normal, manifest_click, store, Path("/workspace"))
    assert dec_normal.action == "confirm"
    assert dec_normal.never_auto_approve is False

    # 2. Escalation button click -> R2 (never auto-approve)
    action_send = {
        "tool": "browser.click",
        "args": {"ref": "@e2"},
        "_element_label": "Send Message",
        "_page_url": "https://example.com",
    }
    dec_send = gate.evaluate(action_send, manifest_click, store, Path("/workspace"))
    assert dec_send.action == "confirm"
    assert dec_send.never_auto_approve is True
    assert any("escalated to R2" in r for r in dec_send.reasons)

    # 3. Social engagement click on social domain -> R3 (Blocked!)
    action_like = {
        "tool": "browser.click",
        "args": {"ref": "@e3"},
        "_element_label": "Like",
        "_page_url": "https://instagram.com/reel/123",
    }
    dec_like = gate.evaluate(action_like, manifest_click, store, Path("/workspace"))
    assert dec_like.action == "block"
    assert any("engagement" in r.lower() for r in dec_like.reasons)


def test_laya_type_password_and_engagement_guard() -> None:
    gate = LayaGate()
    store = ProvenanceStore()
    store.add_user_text("Type into @e1, @e2, @e3")

    manifest_type = ToolManifest(
        name="browser.type",
        summary="Type text",
        risk="R1",
        args={
            "ref": ArgField(type="string", kind="identifier"),
            "text": ArgField(type="string", kind="text"),
        },
    )

    # 1. Type into password field -> Blocked!
    action_pwd = {
        "tool": "browser.type",
        "args": {"ref": "@e1", "text": "secret123"},
        "_element_type": "password",
        "_page_url": "https://example.com/login",
    }
    dec_pwd = gate.evaluate(action_pwd, manifest_type, store, Path("/workspace"))
    assert dec_pwd.action == "block"
    assert any("password" in r.lower() for r in dec_pwd.reasons)

    # 2. Type into OTP / credit card -> Blocked!
    action_card = {
        "tool": "browser.type",
        "args": {"ref": "@e2", "text": "4111222233334444"},
        "_element_autocomplete": "cc-number",
        "_page_url": "https://example.com/checkout",
    }
    dec_card = gate.evaluate(action_card, manifest_type, store, Path("/workspace"))
    assert dec_card.action == "block"

    # 3. Type into comment box on social domain -> R3 Blocked!
    action_comment = {
        "tool": "browser.type",
        "args": {"ref": "@e3", "text": "Nice photo!"},
        "_element_role": "comment_box",
        "_page_url": "https://instagram.com/p/123",
    }
    dec_comment = gate.evaluate(action_comment, manifest_type, store, Path("/workspace"))
    assert dec_comment.action == "block"
    assert any("comment/reply" in r.lower() or "engagement" in r.lower() for r in dec_comment.reasons)


# --- Playwright Live Integration Tests (against local fixture server) ---

@chrome_required
@pytest.mark.anyio
async def test_browser_lifecycle_and_cdp_port(tmp_path: Path) -> None:
    profile_dir = tmp_path / "chrome-profile"
    pid_file = tmp_path / "browser.json"

    # Start browser with dedicated profile
    info = await start_browser(profile_dir=profile_dir, pid_file=pid_file, idle_timeout_s=300)
    try:
        assert info["running"] is True
        assert info["pid"] > 0
        assert info["port"] > 1024
        assert profile_dir.stat().st_mode & 0o777 == 0o700
        assert pid_file.stat().st_mode & 0o777 == 0o600

        # Check status
        status = await get_browser_status(pid_file=pid_file)
        assert status["running"] is True
        assert status["port"] == info["port"]

        # Verify CDP binds only 127.0.0.1
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        res = sock.connect_ex(("127.0.0.1", info["port"]))
        sock.close()
        assert res == 0

    finally:
        await stop_browser(pid_file=pid_file)
        status_after = await get_browser_status(pid_file=pid_file)
        assert status_after["running"] is False


@chrome_required
@pytest.mark.anyio
async def test_browser_snapshot_hidden_elements(fixture_server: str, tmp_path: Path) -> None:
    profile_dir = tmp_path / "chrome-profile"
    pid_file = tmp_path / "browser.json"

    info = await start_browser(profile_dir=profile_dir, pid_file=pid_file)
    try:
        session = await get_browser_session(cdp_url=f"http://127.0.0.1:{info['port']}")
        page = await session.get_page()
        await page.goto(f"{fixture_server}/hidden_text.html")

        engine = SnapshotEngine()
        snapshot = await engine.take_snapshot(page)

        # Verify TOON elements table format
        assert "elements[" in snapshot.toon_table
        assert "@e" in snapshot.toon_table

        # Visible button must be present
        names = [el.name for el in snapshot.elements.values()]
        assert "Visible Button" in names
        assert "Visible Input" in names

        # Hidden elements must NOT appear
        assert "Should Not Appear" not in names
        assert "Invisible Button" not in names
        assert "Aria Hidden Button" not in names
        assert "Tiny Zero-Size Button" not in names
        assert "Off-screen Button" not in names

        # Decode TOON check
        decoded = toon_format.loads(snapshot.toon_table)
        assert "elements" in decoded

    finally:
        await reset_browser_session()
        await stop_browser(pid_file=pid_file)


@chrome_required
@pytest.mark.anyio
async def test_browser_stale_ref_handling(fixture_server: str, tmp_path: Path) -> None:
    profile_dir = tmp_path / "chrome-profile"
    pid_file = tmp_path / "browser.json"

    info = await start_browser(profile_dir=profile_dir, pid_file=pid_file)
    try:
        session = await get_browser_session(cdp_url=f"http://127.0.0.1:{info['port']}")
        page = await session.get_page()
        await page.goto(f"{fixture_server}/stale_ref.html")

        engine = SnapshotEngine()
        snap1 = await engine.take_snapshot(page)
        btn_ref = next(k for k, v in snap1.elements.items() if v.name == "Original Action")

        # Trigger DOM modification that removes button
        await page.evaluate("replaceButton()")

        # Attempt to click using stale ref -> must detect mismatch and return error
        res = await session.click_ref(btn_ref, engine=engine)
        assert res["ok"] is False
        assert "stale" in res["error"].lower() or "mismatch" in res["error"].lower()

    finally:
        await reset_browser_session()
        await stop_browser(pid_file=pid_file)


@chrome_required
@pytest.mark.anyio
async def test_browser_links_substring_dedupe(fixture_server: str, tmp_path: Path) -> None:
    profile_dir = tmp_path / "chrome-profile"
    pid_file = tmp_path / "browser.json"

    info = await start_browser(profile_dir=profile_dir, pid_file=pid_file)
    try:
        session = await get_browser_session(cdp_url=f"http://127.0.0.1:{info['port']}")
        page = await session.get_page()
        await page.goto(f"{fixture_server}/infinite_scroll.html")

        # Extract links matching substrings /p/ and /reel/
        from ag.skills.builtin.browser import extract_links
        links_res = await extract_links(page, patterns=["/p/", "/reel/"], max_results=10)
        assert links_res["ok"] is True
        urls = links_res["data"]["links"]
        assert len(urls) == len(set(urls))  # Deduplicated
        assert any("/p/" in u for u in urls)
        assert any("/reel/" in u for u in urls)

    finally:
        await reset_browser_session()
        await stop_browser(pid_file=pid_file)
