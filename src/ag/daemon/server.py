"""agentd daemon server: asyncio Unix domain socket server with JSON-RPC 2.0 dispatch."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ag.wire.rpc import (
    INTERNAL_ERROR,
    METHOD_NOT_FOUND,
    JsonRpcError,
    RpcNotification,
    RpcRequest,
    format_error,
    format_response,
    get_default_socket_path,
    parse_message,
)

Handler = Callable[[Any], Any] | Callable[[Any], Awaitable[Any]]

# Re-export for convenience
__all__ = ["AgentDaemon", "get_default_socket_path", "register_default_handlers"]


def is_loopback_endpoint(endpoint_url: str) -> bool:
    """Check if an endpoint URL resolves to a local loopback address."""
    try:
        import ipaddress
        from urllib.parse import urlparse
        parsed = urlparse(endpoint_url)
        host = parsed.hostname or ""
        if host.lower() in ("localhost", "127.0.0.1", "::1"):
            return True
        ip = ipaddress.ip_address(host)
        return ip.is_loopback
    except Exception:  # noqa: BLE001
        return False


def register_default_handlers(
    daemon: AgentDaemon,
    workspace_dir: Path | None = None,
    profiles_dir: Path | None = None,
    fake_provider: Any | None = None,
) -> None:
    """Register built-in RPC method handlers for agentd."""
    from ag.models.doctor import get_system_specs, list_server_models, recommend_tier
    from ag.models.profile import (
        ModelProfile,
        get_active_profile,
        list_profiles,
        load_profile,
        set_active_profile,
    )

    start_time = time.time()

    daemon.register("ping", lambda params: "pong")

    def _status_handler(params: Any) -> dict[str, Any]:
        active_name = get_active_profile(workspace_dir)
        return {
            "status": "running",
            "pid": os.getpid(),
            "uptime_seconds": round(time.time() - start_time, 2),
            "socket": str(daemon.socket_path),
            "active_profile": active_name,
        }

    daemon.register("status", _status_handler)

    def _model_list_handler(params: Any) -> list[dict[str, Any]]:
        profiles = list_profiles(profiles_dir)
        active_name = get_active_profile(workspace_dir)
        return [
            {
                "name": p.name,
                "tier": p.tier,
                "model": p.model,
                "endpoint": p.endpoint,
                "active": (p.name == active_name),
            }
            for p in profiles
        ]

    daemon.register("model.list", _model_list_handler)

    def _model_use_handler(params: Any) -> dict[str, Any]:
        name = params.get("name") if isinstance(params, dict) else str(params)
        profile = load_profile(name, profiles_dir=profiles_dir)
        set_active_profile(profile.name, workspace_dir=workspace_dir)
        return {"active_profile": profile.name}

    daemon.register("model.use", _model_use_handler)

    def _model_active_handler(params: Any) -> dict[str, Any] | None:
        active_name = get_active_profile(workspace_dir)
        if not active_name:
            return None
        try:
            profile = load_profile(active_name, profiles_dir=profiles_dir)
            return profile.model_dump()
        except FileNotFoundError:
            return None

    daemon.register("model.active", _model_active_handler)

    async def _doctor_handler(params: Any) -> dict[str, Any]:
        specs = get_system_specs()
        recommended = recommend_tier(specs.vram_gb, specs.ram_gb, specs.cpu_cores)
        installed = await list_server_models()
        return {
            "specs": specs.model_dump(),
            "recommended_tier": recommended,
            "installed_models": installed,
        }

    daemon.register("doctor", _doctor_handler)

    async def _run_confirm_handler(params: Any, writer: asyncio.StreamWriter | None = None) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise JsonRpcError(code=-32602, message="Invalid params for run.confirm")
        run_id = params.get("run_id")
        step = params.get("step")
        action_hash = params.get("action_hash")
        approve = params.get("approve", False)

        key = (run_id, step)
        if key not in daemon.pending_confirms:
            raise JsonRpcError(code=-32001, message=f"No pending confirmation found for run '{run_id}', step {step}")

        orig_writer, expected_hash, fut = daemon.pending_confirms[key]
        if writer is not None and orig_writer is not None and writer is not orig_writer:
            raise JsonRpcError(code=-32002, message="Confirmation only accepted from originating connection")

        if action_hash != expected_hash:
            raise JsonRpcError(code=-32003, message="action_hash mismatch")

        if not fut.done():
            fut.set_result(bool(approve))
        return {"confirmed": True}

    daemon.register("run.confirm", _run_confirm_handler)

    async def _run_cancel_handler(params: Any, writer: asyncio.StreamWriter | None = None) -> dict[str, Any]:
        run_id = params.get("run_id") if isinstance(params, dict) else str(params)
        task = daemon.active_runs.get(run_id)
        if task and not task.done():
            task.cancel()
            return {"cancelled": True}
        return {"cancelled": False, "message": "No active run found"}

    daemon.register("run.cancel", _run_cancel_handler)

    async def _run_handler(params: Any, writer: asyncio.StreamWriter | None = None) -> Any:
        import secrets

        from ag.planner.executor import Executor
        from ag.skills.registry import SkillRegistry

        task_text = ""
        run_id = None
        if isinstance(params, dict):
            task_text = params.get("task") or params.get("text") or params.get("message") or ""
            run_id = params.get("run_id")
        elif isinstance(params, str):
            task_text = params

        run_id = run_id or f"run_{secrets.token_hex(6)}"

        if daemon.model_lock.locked() and writer is not None:
            queued_line = json.dumps({
                "jsonrpc": "2.0",
                "method": "run.event",
                "params": {"run_id": run_id, "type": "queued"},
            }, separators=(",", ":")) + "\n"
            writer.write(queued_line.encode("utf-8"))
            await writer.drain()

        active_name = get_active_profile(workspace_dir)
        if not active_name and fake_provider is None:
            return {"echo": task_text, "summary": task_text, "stop_reason": "done", "total_steps": 1, "duration_s": 0.0}

        try:
            profile = load_profile(active_name or "tier1_cpu", profiles_dir=profiles_dir)
        except FileNotFoundError:
            profile = ModelProfile(name="default", tier=1)

        endpoint_override = None
        allow_remote = False
        if isinstance(params, dict):
            endpoint_override = params.get("endpoint")
            allow_remote = bool(params.get("allow_remote", False))

        if endpoint_override:
            if not allow_remote and not is_loopback_endpoint(endpoint_override):
                raise JsonRpcError(
                    code=-32602,
                    message="Endpoint must be a loopback address unless allow_remote is True",
                )
            profile.endpoint = endpoint_override

        registry = SkillRegistry(workspace_dir=workspace_dir)

        async def confirm_handler(confirm_event: dict[str, Any]) -> dict[str, Any]:
            step = confirm_event["step"]
            action_hash = confirm_event["action_hash"]
            fut = asyncio.get_running_loop().create_future()
            daemon.pending_confirms[(run_id, step)] = (writer, action_hash, fut)
            try:
                approved = await asyncio.wait_for(fut, timeout=60.0)
                return {"approved": approved, "source": "user"}
            except TimeoutError:
                return {"approved": False, "source": "timeout"}
            finally:
                daemon.pending_confirms.pop((run_id, step), None)

        async def event_handler(event_data: dict[str, Any]) -> None:
            if writer is not None:
                evt_line = json.dumps({
                    "jsonrpc": "2.0",
                    "method": "run.event",
                    "params": event_data,
                }, separators=(",", ":")) + "\n"
                writer.write(evt_line.encode("utf-8"))
                await writer.drain()

        executor = Executor(
            profile=profile,
            registry=registry,
            provider=fake_provider,
            confirm_handler=confirm_handler,
            event_handler=event_handler,
        )

        current_task = asyncio.current_task()
        if current_task:
            daemon.active_runs[run_id] = current_task

        async with daemon.model_lock:
            try:
                res = await executor.run(task_text, run_id=run_id)
                return res.model_dump()
            finally:
                daemon.active_runs.pop(run_id, None)

    daemon.register("run", _run_handler)


class AgentDaemon:
    """The agentd daemon process managing client connections and method dispatch."""

    def __init__(self, socket_path: Path | str | None = None) -> None:
        self.socket_path = (
            Path(socket_path).expanduser() if socket_path else get_default_socket_path()
        )
        self.handlers: dict[str, Handler] = {}
        self.server: asyncio.Server | None = None
        self._stop_event = asyncio.Event()
        self._active_writers: set[asyncio.StreamWriter] = set()
        self.model_lock = asyncio.Lock()
        self.active_runs: dict[str, asyncio.Task[Any]] = {}
        self.pending_confirms: dict[tuple[str, int], tuple[asyncio.StreamWriter, str, asyncio.Future[bool]]] = {}

    def register(self, method: str, handler: Handler) -> None:
        """Register an RPC method handler."""
        self.handlers[method] = handler

    async def _check_and_clean_stale_socket(self) -> None:
        """Clean up stale socket file if daemon is not currently running."""
        if not self.socket_path.exists():
            return

        try:
            _, writer = await asyncio.open_unix_connection(str(self.socket_path))
            writer.close()
            await writer.wait_closed()
            raise RuntimeError(f"agentd is already running on {self.socket_path}")
        except (ConnectionRefusedError, FileNotFoundError):
            # Socket file exists on filesystem but no active process is listening: safe to unlink
            self.socket_path.unlink(missing_ok=True)

    async def start(self) -> asyncio.Server:
        """Start the Unix domain socket server."""
        if self.server is not None:
            return self.server

        # Ensure parent directory exists (e.g. ~/.ag with 0700 permissions)
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.socket_path.parent.chmod(0o700)
        except OSError:
            pass

        await self._check_and_clean_stale_socket()

        self.server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self.socket_path),
        )

        # Secure socket permissions: read/write by owner only (0600)
        try:
            self.socket_path.chmod(0o600)
        except OSError:
            pass

        return self.server

    async def stop(self) -> None:
        """Stop the server, terminate client connections, and remove socket file."""
        self._stop_event.set()

        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

        # Close all active client connections
        for writer in list(self._active_writers):
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionError, BrokenPipeError):
                pass
        self._active_writers.clear()

        # Remove the socket file
        if self.socket_path.exists():
            self.socket_path.unlink(missing_ok=True)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Read and dispatch JSON-RPC 2.0 messages from a client stream."""
        self._active_writers.add(writer)
        try:
            while not self._stop_event.is_set():
                line = await reader.readline()
                if not line:
                    break  # Client disconnected

                try:
                    msg = parse_message(line)
                except JsonRpcError as err:
                    response_bytes = format_error(
                        code=err.code,
                        message=err.message,
                        data=err.data,
                        msg_id=err.msg_id,
                    )
                    writer.write(response_bytes)
                    await writer.drain()
                    continue
                except Exception as err:  # noqa: BLE001
                    response_bytes = format_error(
                        code=INTERNAL_ERROR,
                        message=f"Internal error while parsing: {err}",
                    )
                    writer.write(response_bytes)
                    await writer.drain()
                    continue

                if isinstance(msg, RpcNotification):
                    handler = self.handlers.get(msg.method)
                    if handler is not None:
                        try:
                            if inspect.iscoroutinefunction(handler):
                                await handler(msg.params)
                            else:
                                handler(msg.params)
                        except Exception:  # noqa: BLE001, S110
                            pass  # Per JSON-RPC 2.0: notifications never return errors or responses
                    continue

                if isinstance(msg, RpcRequest):
                    asyncio.create_task(self._dispatch_request(msg, writer))
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            self._active_writers.discard(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionError, BrokenPipeError):
                pass

    async def _dispatch_request(
        self,
        msg: RpcRequest,
        writer: asyncio.StreamWriter,
    ) -> None:
        handler = self.handlers.get(msg.method)
        if handler is None:
            response_bytes = format_error(
                code=METHOD_NOT_FOUND,
                message=f"Method '{msg.method}' not found",
                msg_id=msg.id,
            )
        else:
            try:
                sig = inspect.signature(handler)
                if "writer" in sig.parameters:
                    res_val = handler(msg.params, writer=writer)
                else:
                    res_val = handler(msg.params)

                if inspect.isawaitable(res_val):
                    result = await res_val
                else:
                    result = res_val
                response_bytes = format_response(result=result, msg_id=msg.id)
            except JsonRpcError as err:
                response_bytes = format_error(
                    code=err.code,
                    message=err.message,
                    data=err.data,
                    msg_id=msg.id,
                )
            except Exception as err:  # noqa: BLE001
                response_bytes = format_error(
                    code=INTERNAL_ERROR,
                    message=f"Internal error in '{msg.method}': {err}",
                    msg_id=msg.id,
                )
        try:
            writer.write(response_bytes)
            await writer.drain()
        except (ConnectionError, BrokenPipeError):
            pass

    async def serve_forever(self) -> None:
        """Run the daemon server until stopped."""
        await self.start()
        loop = asyncio.get_running_loop()
        for sig in ("SIGINT", "SIGTERM"):
            import signal

            if hasattr(signal, sig):
                try:
                    loop.add_signal_handler(
                        getattr(signal, sig),
                        lambda: asyncio.create_task(self.stop()),
                    )
                except NotImplementedError:
                    pass
        await self._stop_event.wait()
