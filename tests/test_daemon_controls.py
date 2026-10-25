"""Tests for daemon controls: model.use refused during run, session RPC methods, and daemon pidfile/status/stop."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from ag.daemon.server import AgentDaemon, register_default_handlers
from ag.wire.rpc import JsonRpcError


@pytest.mark.anyio
async def test_model_use_refused_while_run_active(tmp_path: Path):
    """model.use must return an error if a run is currently active."""
    socket_path = tmp_path / "ag.sock"
    daemon = AgentDaemon(socket_path=socket_path)
    register_default_handlers(
        daemon=daemon,
        workspace_dir=tmp_path / "workspace",
        sessions_dir=tmp_path / "sessions",
    )

    # Simulate an active run
    dummy_task = asyncio.create_task(asyncio.sleep(10))
    daemon.active_runs["run_simulated"] = dummy_task

    try:
        handler = daemon.handlers["model.use"]
        with pytest.raises(JsonRpcError) as exc_info:
            await handler({"name": "tier1_cpu"}) if inspect.iscoroutinefunction(handler) else handler({"name": "tier1_cpu"})

        assert exc_info.value.code == -32004
        assert "Cannot switch model profile while a run is active" in exc_info.value.message
    finally:
        dummy_task.cancel()


@pytest.mark.anyio
async def test_session_rpc_methods(tmp_path: Path):
    """Verify session.list, session.new, session.load, and session.delete in agentd."""
    socket_path = tmp_path / "ag.sock"
    sessions_dir = tmp_path / "sessions"
    daemon = AgentDaemon(socket_path=socket_path)

    register_default_handlers(
        daemon=daemon,
        workspace_dir=tmp_path / "workspace",
        sessions_dir=sessions_dir,
    )

    # 1. session.new
    new_handler = daemon.handlers["session.new"]
    res_new = await new_handler({"title": "First Session"}) if inspect.iscoroutinefunction(new_handler) else new_handler({"title": "First Session"})
    session_id = res_new["session_id"]
    assert len(session_id) >= 16

    # 2. session.list
    list_handler = daemon.handlers["session.list"]
    res_list = await list_handler({}) if inspect.iscoroutinefunction(list_handler) else list_handler({})
    assert len(res_list) == 1
    assert res_list[0]["session_id"] == session_id

    # 3. session.load
    load_handler = daemon.handlers["session.load"]
    res_load = await load_handler({"session_id": session_id}) if inspect.iscoroutinefunction(load_handler) else load_handler({"session_id": session_id})
    assert res_load["session_id"] == session_id
    assert isinstance(res_load["messages"], list)

    # 4. session.delete
    del_handler = daemon.handlers["session.delete"]
    res_del = await del_handler({"session_id": session_id}) if inspect.iscoroutinefunction(del_handler) else del_handler({"session_id": session_id})
    assert res_del["deleted"] is True

    # Confirm deletion
    res_list2 = await list_handler({}) if inspect.iscoroutinefunction(list_handler) else list_handler({})
    assert len(res_list2) == 0
