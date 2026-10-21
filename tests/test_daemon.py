"""Integration tests for agentd and client RPC communication."""

import asyncio
import threading
import time
from pathlib import Path

import pytest

from ag.client.rpc import RpcClient, RpcRemoteError
from ag.daemon.server import AgentDaemon
from ag.wire.rpc import METHOD_NOT_FOUND


def run_async(coro):
    return asyncio.run(coro)


@pytest.fixture
def socket_path(tmp_path: Path) -> Path:
    return tmp_path / "agentd_test.sock"


@pytest.fixture
def running_daemon(socket_path: Path):
    daemon = AgentDaemon(socket_path)
    daemon.register("ping", lambda params: "pong")
    daemon.register("run", lambda params: {"echo": params})

    loop = asyncio.new_event_loop()
    ready_evt = threading.Event()

    def run_daemon_loop():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(daemon.start())
        ready_evt.set()
        loop.run_until_complete(daemon._stop_event.wait())
        loop.run_until_complete(daemon.stop())
        loop.close()

    thread = threading.Thread(target=run_daemon_loop, daemon=True)
    thread.start()
    assert ready_evt.wait(timeout=5.0), "Daemon failed to start within 5s"

    try:
        yield daemon
    finally:
        loop.call_soon_threadsafe(daemon._stop_event.set)
        thread.join(timeout=5.0)


def test_ping_takes_under_50ms(running_daemon: AgentDaemon, socket_path: Path):
    async def _test():
        async with RpcClient(socket_path) as client:
            start_time = time.perf_counter()
            result = await client.call("ping")
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            assert result == "pong"
            assert elapsed_ms < 50.0, f"Ping took {elapsed_ms:.2f} ms (expected < 50 ms)"

    run_async(_test())


def test_run_echo(running_daemon: AgentDaemon, socket_path: Path):
    async def _test():
        async with RpcClient(socket_path) as client:
            result = await client.call("run", {"message": "hello Sowl"})
            assert result == {"echo": {"message": "hello Sowl"}}

    run_async(_test())


def test_two_clients_work_concurrently(running_daemon: AgentDaemon, socket_path: Path):
    async def _test():
        async with RpcClient(socket_path) as client1, RpcClient(socket_path) as client2:
            async def work(client: RpcClient, tag: str):
                res1 = await client.call("ping")
                res2 = await client.call("run", {"tag": tag})
                return res1, res2

            (res1_c1, res2_c1), (res1_c2, res2_c2) = await asyncio.gather(
                work(client1, "client_1"),
                work(client2, "client_2"),
            )

            assert res1_c1 == "pong"
            assert res2_c1 == {"echo": {"tag": "client_1"}}
            assert res1_c2 == "pong"
            assert res2_c2 == {"echo": {"tag": "client_2"}}

    run_async(_test())


def test_malformed_input_returns_error_without_crashing(
    running_daemon: AgentDaemon, socket_path: Path
):
    async def _test():
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        try:
            # 1. Invalid JSON (parse error -32700)
            writer.write(b"not valid json at all\n")
            await writer.drain()

            err_line = await reader.readline()
            assert b"-32700" in err_line

            # 2. Invalid Request structure (-32600)
            writer.write(b'{"foo": "bar"}\n')
            await writer.drain()

            err_line2 = await reader.readline()
            assert b"-32600" in err_line2

            # 3. Method not found (-32601)
            writer.write(b'{"jsonrpc": "2.0", "method": "non_existent_method", "id": 99}\n')
            await writer.drain()

            err_line3 = await reader.readline()
            assert b"-32601" in err_line3

            # 4. Daemon is still healthy and responds to a valid ping on the same connection!
            writer.write(b'{"jsonrpc": "2.0", "method": "ping", "id": 100}\n')
            await writer.drain()

            valid_line = await reader.readline()
            assert b'"pong"' in valid_line
        finally:
            writer.close()
            await writer.wait_closed()

    run_async(_test())


def test_client_handles_remote_error(running_daemon: AgentDaemon, socket_path: Path):
    async def _test():
        async with RpcClient(socket_path) as client:
            with pytest.raises(RpcRemoteError) as exc_info:
                await client.call("unknown_method")
            assert exc_info.value.code == METHOD_NOT_FOUND

    run_async(_test())
