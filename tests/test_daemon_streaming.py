"""Tests for daemon streaming protocol, confirmation exchange, action hash, and lock."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from ag.daemon.server import AgentDaemon, register_default_handlers
from ag.models.provider import FakeModelProvider
from ag.wire.rpc import format_request


@pytest.mark.anyio
async def test_daemon_socket_permissions(tmp_path: Path) -> None:
    socket_path = tmp_path / "ag.sock"
    daemon = AgentDaemon(socket_path=socket_path)
    register_default_handlers(daemon, workspace_dir=tmp_path)
    await daemon.start()

    try:
        assert socket_path.exists()
        stat_sock = socket_path.stat()
        assert (stat_sock.st_mode & 0o777) == 0o600
    finally:
        await daemon.stop()


@pytest.mark.anyio
async def test_daemon_streaming_run_and_confirm(tmp_path: Path) -> None:
    socket_path = tmp_path / "ag.sock"
    daemon = AgentDaemon(socket_path=socket_path)

    # 1. browser.open_url (requires confirm) -> 2. done
    responses = [
        '{"kind": "tool", "tool": "browser.open_url", "args": {"url": "https://example.com"}}',
        '{"kind": "done", "summary": "Opened webpage"}',
    ]
    fake_provider = FakeModelProvider(responses=responses)
    register_default_handlers(daemon, workspace_dir=tmp_path, fake_provider=fake_provider)
    await daemon.start()

    try:
        reader, writer = await asyncio.open_unix_connection(str(socket_path))

        # Send run request
        run_req = format_request("run", {"task": "Open example.com"}, msg_id=1)
        writer.write(run_req)
        await writer.drain()

        # Read streaming lines
        events: list[dict[str, Any]] = []
        final_result = None

        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=3.0)
            if not line:
                break
            msg = json.loads(line.decode("utf-8"))
            if "method" in msg and msg["method"] == "run.event":
                evt = msg["params"]
                events.append(evt)
                if evt["type"] == "confirm":
                    # Respond with run.confirm
                    action_hash = evt["action_hash"]
                    confirm_req = format_request(
                        "run.confirm",
                        {
                            "run_id": evt["run_id"],
                            "step": evt["step"],
                            "action_hash": action_hash,
                            "approve": True,
                        },
                        msg_id=2,
                    )
                    writer.write(confirm_req)
                    await writer.drain()
            elif "id" in msg and msg["id"] == 1:
                final_result = msg.get("result")
                break

        writer.close()
        await writer.wait_closed()

        assert final_result is not None
        assert final_result["stop_reason"] == "done"
        types = [e["type"] for e in events]
        assert "confirm" in types
        assert "observation" in types
    finally:
        await daemon.stop()


@pytest.mark.anyio
async def test_daemon_confirm_rejected_from_different_connection(tmp_path: Path) -> None:
    socket_path = tmp_path / "ag.sock"
    daemon = AgentDaemon(socket_path=socket_path)

    responses = [
        '{"kind": "tool", "tool": "browser.open_url", "args": {"url": "https://example.com"}}',
        '{"kind": "done", "summary": "Done"}',
    ]
    fake_provider = FakeModelProvider(responses=responses)
    register_default_handlers(daemon, workspace_dir=tmp_path, fake_provider=fake_provider)
    await daemon.start()

    try:
        reader1, writer1 = await asyncio.open_unix_connection(str(socket_path))
        reader2, writer2 = await asyncio.open_unix_connection(str(socket_path))

        # Client 1 starts run
        writer1.write(format_request("run", {"task": "Open example.com"}, msg_id=10))
        await writer1.drain()

        # Wait for confirm on client 1
        confirm_evt = None
        while True:
            line = await asyncio.wait_for(reader1.readline(), timeout=3.0)
            msg = json.loads(line.decode("utf-8"))
            if msg.get("params", {}).get("type") == "confirm":
                confirm_evt = msg["params"]
                break

        assert confirm_evt is not None

        # Client 2 tries to confirm client 1's action -> must be rejected!
        writer2.write(
            format_request(
                "run.confirm",
                {
                    "run_id": confirm_evt["run_id"],
                    "step": confirm_evt["step"],
                    "action_hash": confirm_evt["action_hash"],
                    "approve": True,
                },
                msg_id=20,
            )
        )
        await writer2.drain()

        # Reader 2 gets error
        line2 = await asyncio.wait_for(reader2.readline(), timeout=3.0)
        resp2 = json.loads(line2.decode("utf-8"))
        assert "error" in resp2

        writer1.close()
        writer2.close()
        await writer1.wait_closed()
        await writer2.wait_closed()
    finally:
        await daemon.stop()
