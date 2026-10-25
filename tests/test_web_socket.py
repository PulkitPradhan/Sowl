"""Tests for WebSocket bridge to agentd: limits, message size cap, action_hash single-use, and run flow."""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiohttp
import pytest
from aiohttp import ClientSession, web

from ag.daemon.server import AgentDaemon, register_default_handlers
from ag.models.provider import FakeModelProvider
from ag.web.server import create_web_app


@pytest.fixture
async def web_and_daemon(tmp_path: Path):
    socket_path = tmp_path / "ag.sock"
    daemon = AgentDaemon(socket_path=socket_path)

    # Fake provider returning single step done action
    fake_provider = FakeModelProvider([
        '{"kind": "done", "summary": "Task completed successfully"}',
    ])

    register_default_handlers(
        daemon=daemon,
        workspace_dir=tmp_path / "workspace",
        fake_provider=fake_provider,
        sessions_dir=tmp_path / "sessions",
    )
    await daemon.start()

    # Find ephemeral free port
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    startup_token = "valid_test_token_128bit_min"
    web_app = create_web_app(
        startup_token=startup_token,
        port=port,
        daemon_socket_path=socket_path,
        sessions_dir=tmp_path / "sessions",
    )

    runner = web.AppRunner(web_app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    yield {
        "port": port,
        "token": startup_token,
        "daemon": daemon,
    }

    await runner.cleanup()
    await daemon.stop()


@pytest.mark.anyio
async def test_ws_unauthenticated_rejected(web_and_daemon):
    """Connecting to WebSocket without authentication cookie or token must fail."""
    port = web_and_daemon["port"]
    async with ClientSession() as session:
        try:
            async with session.ws_connect(f"http://127.0.0.1:{port}/ws"):
                assert False, "Unauthenticated WS should have failed"
        except Exception as err:  # noqa: BLE001
            # Expected: 401 Unauthorized during handshake
            assert "401" in str(err) or "Invalid response status" in str(err)


@pytest.mark.anyio
async def test_ws_authenticated_run_flow(web_and_daemon):
    """Authenticated client can start a run and receive streaming events over WebSocket."""
    port = web_and_daemon["port"]
    token = web_and_daemon["token"]

    async with ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as session:
        # 1. Exchange token for cookie
        auth_resp = await session.post(
            f"http://127.0.0.1:{port}/api/auth",
            headers={"Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}"},
            json={"token": token},
        )
        assert auth_resp.status == 200

        # 2. Connect to WS
        async with session.ws_connect(
            f"http://127.0.0.1:{port}/ws",
            headers={"Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}"},
        ) as ws:
            # 3. Create a session first via HTTP API
            s_resp = await session.post(f"http://127.0.0.1:{port}/api/sessions", json={"title": "WS Test"})
            assert s_resp.status == 200
            s_data = await s_resp.json()
            session_id = s_data["session_id"]

            # 4. Send task over WS
            await ws.send_json({
                "type": "run",
                "task": "Test task",
                "session_id": session_id,
            })

            # 5. Read events until done
            received_done = False
            for _ in range(10):
                msg = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
                if msg.get("type") == "done":
                    received_done = True
                    assert msg.get("summary") == "Task completed successfully"
                    break

            assert received_done, "Did not receive done event"


@pytest.mark.anyio
async def test_ws_task_size_cap(web_and_daemon):
    """Tasks larger than ~8 KB are rejected with an error event."""
    port = web_and_daemon["port"]

    async with ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as session:
        # Authenticate via token
        auth_resp = await session.post(
            f"http://127.0.0.1:{port}/api/auth",
            headers={"Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}"},
            json={"token": web_and_daemon["token"]},
        )
        assert auth_resp.status == 200

        async with session.ws_connect(
            f"http://127.0.0.1:{port}/ws",
            headers={"Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}"},
        ) as ws:
            # Send message with task > 8192 bytes
            huge_task = "x" * 9000
            await ws.send_json({
                "type": "run",
                "task": huge_task,
            })

            msg = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
            assert msg.get("type") == "error"
            assert "exceeds maximum allowed length" in msg.get("message", "")


@pytest.mark.anyio
async def test_ws_confirm_flow_and_action_hash_single_use(tmp_path: Path):
    """Verify tool confirmation approval and that replayed action_hash is rejected."""
    socket_path = tmp_path / "ag.sock"
    daemon = AgentDaemon(socket_path=socket_path)

    # Fake provider returning R1 tool call followed by done
    fake_provider = FakeModelProvider([
        '{"kind": "tool", "tool": "browser.open_url", "args": {"url": "https://docs.python.org"}}',
        '{"kind": "done", "summary": "Opened documentation"}',
    ])

    register_default_handlers(
        daemon=daemon,
        workspace_dir=tmp_path / "workspace",
        fake_provider=fake_provider,
        sessions_dir=tmp_path / "sessions",
    )
    await daemon.start()

    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    startup_token = "valid_confirm_token_128bit"
    web_app = create_web_app(
        startup_token=startup_token,
        port=port,
        daemon_socket_path=socket_path,
        sessions_dir=tmp_path / "sessions",
    )

    runner = web.AppRunner(web_app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    try:
        async with ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as session:
            auth_resp = await session.post(
                f"http://127.0.0.1:{port}/api/auth",
                headers={"Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}"},
                json={"token": startup_token},
            )
            assert auth_resp.status == 200

            async with session.ws_connect(
                f"http://127.0.0.1:{port}/ws",
                headers={"Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}"},
            ) as ws:
                await ws.send_json({"type": "run", "task": "Open docs"})

                # Wait for confirm event
                confirm_evt = None
                for _ in range(10):
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
                    if msg.get("type") == "confirm":
                        confirm_evt = msg
                        break

                assert confirm_evt is not None
                action_hash = confirm_evt["action_hash"]
                assert confirm_evt["tool"] == "browser.open_url"

                # Send approval
                await ws.send_json({
                    "type": "confirm",
                    "action_hash": action_hash,
                    "approve": True,
                })

                # Receive done event
                received_done = False
                for _ in range(10):
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
                    if msg.get("type") == "done":
                        received_done = True
                        break
                assert received_done

                # Replay the same action_hash a second time: must be refused!
                await ws.send_json({
                    "type": "confirm",
                    "action_hash": action_hash,
                    "approve": True,
                })

                err_msg = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
                assert err_msg.get("type") == "error"
                assert "Unknown or already consumed action_hash" in err_msg.get("message", "")
    finally:
        await runner.cleanup()
        await daemon.stop()
