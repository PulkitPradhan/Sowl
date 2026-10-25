"""Interactive web harness server powered by aiohttp.

Bridges local web UI to agentd JSON-RPC daemon with strict loopback binding,
DNS rebinding protection, single-use token auth, SameSite=Strict cookies,
and pure textContent rendering.
"""

from __future__ import annotations

import asyncio
import errno
import hmac
import json
import os
import secrets
import socket
import sys
import time
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from ag.skills.launcher import get_launcher
from ag.skills.registry import SkillRegistry
from ag.web.sessions import SessionManager, validate_session_id
from ag.wire.rpc import format_request, get_default_socket_path

STATIC_DIR = Path(__file__).parent / "static"


class ServerState:
    def __init__(
        self,
        startup_token: str | None,
        port: int,
        token_ttl: float,
        daemon_socket_path: Path,
        session_mgr: SessionManager,
    ) -> None:
        self.startup_token = startup_token
        self.startup_token_created_at = time.time()
        self.port = port
        self.token_ttl = token_ttl
        self.daemon_socket_path = daemon_socket_path
        self.session_mgr = session_mgr
        self.valid_sessions: set[str] = set()
        self.active_websockets: set[web.WebSocketResponse] = set()
        self.auth_failures: dict[str, list[float]] = {}


STATE_KEY = web.AppKey("state", ServerState)


def _attach_security_headers(response: web.StreamResponse, port: int, path: str) -> web.StreamResponse:
    headers = response.headers
    headers["X-Frame-Options"] = "DENY"
    headers["X-Content-Type-Options"] = "nosniff"
    headers["Referrer-Policy"] = "no-referrer"
    headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "img-src 'self' data:; "
        f"connect-src http://127.0.0.1:{port} ws://127.0.0.1:{port}; "
        "frame-ancestors 'none'; "
        "base-uri 'none'; "
        "form-action 'none'; "
        "object-src 'none'"
    )
    if path.startswith("/api/") or path == "/ws":
        headers["Cache-Control"] = "no-store"
        headers["Pragma"] = "no-cache"
    return response


def create_web_app(
    startup_token: str,
    port: int = 3180,
    token_ttl: float = 60.0,
    daemon_socket_path: Path | str | None = None,
    sessions_dir: Path | None = None,
) -> web.Application:
    """Create and configure the secure aiohttp web application."""
    app = web.Application()

    daemon_sock = (
        Path(daemon_socket_path).expanduser()
        if daemon_socket_path
        else get_default_socket_path()
    )
    state = ServerState(
        startup_token=startup_token,
        port=port,
        token_ttl=token_ttl,
        daemon_socket_path=daemon_sock,
        session_mgr=SessionManager(sessions_dir=sessions_dir),
    )
    app[STATE_KEY] = state

    # ---------------- Security Middleware ----------------
    @web.middleware
    async def security_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
        path = request.path

        # 1. Host header validation (DNS rebinding protection)
        host = request.headers.get("Host", "").strip()
        allowed_hosts = {
            f"127.0.0.1:{state.port}",
            f"localhost:{state.port}",
            "127.0.0.1",
            "localhost",
        }
        if host not in allowed_hosts:
            resp = web.Response(
                status=403,
                text=f"Forbidden: Invalid Host header '{host}'. Loopback only.",
            )
            return _attach_security_headers(resp, state.port, path)

        # 2. Origin validation on WebSocket and state-changing requests (CSWSH & CSRF protection)
        origin = request.headers.get("Origin")
        if origin is not None:
            allowed_origins = {
                f"http://127.0.0.1:{state.port}",
                f"http://localhost:{state.port}",
            }
            if origin not in allowed_origins:
                resp = web.Response(
                    status=403,
                    text=f"Forbidden: Invalid Origin header '{origin}'.",
                )
                return _attach_security_headers(resp, state.port, path)

        # 3. Authentication Check
        is_public = (
            path == "/"
            or path == "/index.html"
            or path.startswith("/static/")
            or path == "/api/auth"
        )

        if not is_public:
            # Check session cookie
            cookie_val = request.cookies.get("sowl_session")
            if not cookie_val or cookie_val not in state.valid_sessions:
                resp = web.json_response({"error": "Unauthorized"}, status=401)
                return _attach_security_headers(resp, state.port, path)

        # 4. Invoke Handler
        response = await handler(request)

        # 5. Attach defense-in-depth security headers
        return _attach_security_headers(response, state.port, path)

    app.middlewares.append(security_middleware)

    # ---------------- Auth Endpoint ----------------
    async def handle_auth(request: web.Request) -> web.Response:
        client_ip = request.remote or "127.0.0.1"
        now = time.time()

        # Rate limiting: max 5 failed attempts per minute
        failures = [t for t in state.auth_failures.get(client_ip, []) if now - t < 60.0]
        if len(failures) >= 5:
            return web.Response(status=429, text="Too Many Requests")

        try:
            body = await request.json()
            provided_token = str(body.get("token", ""))
        except Exception:  # noqa: BLE001
            return web.Response(status=400, text="Bad Request")

        current_token = state.startup_token
        created_at = state.startup_token_created_at

        # Verify token existence, TTL (60s), and cryptographic equality
        if (
            not current_token
            or (now - created_at > state.token_ttl)
            or not hmac.compare_digest(provided_token, current_token)
        ):
            failures.append(now)
            state.auth_failures[client_ip] = failures
            return web.Response(status=401, text="Unauthorized: Invalid or expired token")

        # Invalidate startup token (single-use guarantee)
        state.startup_token = None

        # Issue separate random session cookie
        session_cookie = secrets.token_hex(32)
        state.valid_sessions.add(session_cookie)

        resp = web.json_response({"ok": True})
        resp.set_cookie(
            "sowl_session",
            session_cookie,
            path="/",
            httponly=True,
            samesite="Strict",
            max_age=86400,
        )
        return resp

    # ---------------- Static Routes ----------------
    async def handle_index(request: web.Request) -> web.FileResponse:
        index_file = STATIC_DIR / "index.html"
        return web.FileResponse(index_file)

    # ---------------- Session API Endpoints ----------------
    session_mgr: SessionManager = state.session_mgr

    async def api_list_sessions(request: web.Request) -> web.Response:
        sessions = session_mgr.list_sessions()
        return web.json_response(sessions)

    async def api_create_session(request: web.Request) -> web.Response:
        title = None
        if request.can_read_body:
            try:
                body = await request.json()
                title = body.get("title")
            except Exception:  # noqa: BLE001, S110
                pass
        session_id = session_mgr.create_session(title=title)
        return web.json_response({"session_id": session_id})

    async def api_load_session(request: web.Request) -> web.Response:
        session_id = request.match_info.get("session_id", "")
        try:
            turns = session_mgr.load_session(session_id)
            return web.json_response({"session_id": session_id, "messages": turns})
        except Exception as err:  # noqa: BLE001
            return web.json_response({"error": str(err)}, status=400)

    async def api_delete_session(request: web.Request) -> web.Response:
        session_id = request.match_info.get("session_id", "")
        try:
            deleted = session_mgr.delete_session(session_id)
            return web.json_response({"deleted": deleted})
        except Exception as err:  # noqa: BLE001
            return web.json_response({"error": str(err)}, status=400)

    # ---------------- System / Daemon API Endpoints ----------------
    async def _call_daemon(method: str, params: Any = None) -> Any:
        from ag.client.rpc import RpcClient

        async with RpcClient(state.daemon_socket_path) as client:
            return await client.call(method, params)

    async def api_get_status(request: web.Request) -> web.Response:
        try:
            status_data = await _call_daemon("status")
            doctor_data = await _call_daemon("doctor")
            return web.json_response({
                "daemon": status_data,
                "doctor": doctor_data,
            })
        except Exception as err:  # noqa: BLE001
            return web.json_response({"error": f"Failed to reach daemon: {err}"}, status=503)

    async def api_list_models(request: web.Request) -> web.Response:
        try:
            models = await _call_daemon("model.list")
            return web.json_response(models)
        except Exception as err:  # noqa: BLE001
            return web.json_response({"error": str(err)}, status=503)

    async def api_use_model(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            model_name = body.get("name")
            res = await _call_daemon("model.use", {"name": model_name})
            return web.json_response(res)
        except Exception as err:  # noqa: BLE001
            return web.json_response({"error": str(err)}, status=400)

    async def api_list_skills(request: web.Request) -> web.Response:
        try:
            registry = SkillRegistry()
            skills_data = []
            for skill in registry.list_skills():
                skills_data.append({
                    "name": skill.name,
                    "summary": skill.summary,
                    "risk": skill.risk,
                    "tools": [
                        {
                            "name": t.name,
                            "summary": t.summary,
                            "risk": t.risk,
                        }
                        for t in skill.tools
                    ],
                })
            return web.json_response(skills_data)
        except Exception as err:  # noqa: BLE001
            return web.json_response({"error": str(err)}, status=500)

    # ---------------- WebSocket Bridge ----------------
    async def handle_ws(request: web.Request) -> web.WebSocketResponse:
        if len(state.active_websockets) >= 5:
            return web.Response(status=429, text="Too many active WebSocket connections")

        ws = web.WebSocketResponse(
            heartbeat=30.0,
            max_msg_size=65536,  # 64 KB cap
        )
        await ws.prepare(request)
        state.active_websockets.add(ws)

        pending_confirms: dict[str, tuple[str, int, socket.socket, float]] = {}

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        await ws.send_json({"type": "error", "message": "Invalid JSON"})
                        continue

                    msg_type = data.get("type")

                    if msg_type == "run":
                        task = data.get("task", "")
                        if not isinstance(task, str):
                            await ws.send_json({"type": "error", "message": "Task must be a string"})
                            continue

                        # Cap task text at 8192 chars (~8 KB)
                        if len(task) > 8192:
                            await ws.send_json({
                                "type": "error",
                                "message": f"Task length ({len(task)}) exceeds maximum allowed length of 8192 characters",
                            })
                            continue

                        session_id = data.get("session_id")
                        if session_id:
                            try:
                                validate_session_id(session_id)
                            except ValueError as err:
                                await ws.send_json({"type": "error", "message": str(err)})
                                continue

                        # Launch background task connecting to daemon
                        asyncio.create_task(
                            _execute_run_over_ws(
                                ws=ws,
                                task=task,
                                session_id=session_id,
                                daemon_socket_path=state.daemon_socket_path,
                                pending_confirms=pending_confirms,
                            )
                        )

                    elif msg_type == "confirm":
                        action_hash = data.get("action_hash")
                        approve = bool(data.get("approve", False))

                        # Single-use check and expiry check (60s)
                        if not action_hash or action_hash not in pending_confirms:
                            await ws.send_json({
                                "type": "error",
                                "message": "Unknown or already consumed action_hash",
                            })
                            continue

                        run_id, step, daemon_sock, created_at = pending_confirms.pop(action_hash)

                        if time.time() - created_at > 60.0:
                            await ws.send_json({
                                "type": "error",
                                "message": "Confirmation expired (60s timeout)",
                            })
                            continue

                        confirm_req = format_request(
                            "run.confirm",
                            {
                                "run_id": run_id,
                                "step": step,
                                "action_hash": action_hash,
                                "approve": approve,
                            },
                            msg_id=2,
                        )
                        try:
                            daemon_sock.sendall(confirm_req)
                        except Exception as err:  # noqa: BLE001
                            await ws.send_json({"type": "error", "message": f"Failed to forward confirmation: {err}"})

                    elif msg_type == "cancel":
                        run_id = data.get("run_id")
                        if run_id:
                            from ag.client.rpc import RpcClient

                            try:
                                async with RpcClient(state.daemon_socket_path) as client:
                                    await client.call("run.cancel", {"run_id": run_id})
                            except Exception:  # noqa: BLE001, S110
                                pass

                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            state.active_websockets.discard(ws)
            for _, _, sock, _ in pending_confirms.values():
                try:
                    sock.close()
                except Exception:  # noqa: BLE001, S110
                    pass

        return ws

    async def _execute_run_over_ws(
        ws: web.WebSocketResponse,
        task: str,
        session_id: str | None,
        daemon_socket_path: Path,
        pending_confirms: dict[str, tuple[str, int, socket.socket, float]],
    ) -> None:
        run_id = f"run_{secrets.token_hex(6)}"
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(str(daemon_socket_path))
        except Exception as err:  # noqa: BLE001
            await ws.send_json({"type": "error", "message": f"Cannot connect to agentd daemon: {err}"})
            return

        run_params: dict[str, Any] = {"task": task, "run_id": run_id}
        if session_id:
            run_params["session_id"] = session_id

        req_bytes = format_request("run", run_params, msg_id=1)
        sock.sendall(req_bytes)

        buffer = b""
        loop = asyncio.get_running_loop()

        try:
            while not ws.closed:
                chunk = await loop.run_in_executor(None, sock.recv, 4096)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    msg = json.loads(line.decode("utf-8"))

                    if msg.get("method") == "run.event":
                        evt = msg.get("params", {})
                        evt_type = evt.get("type")

                        if evt_type == "confirm":
                            act_hash = evt.get("action_hash")
                            if act_hash:
                                pending_confirms[act_hash] = (
                                    evt.get("run_id"),
                                    evt.get("step"),
                                    sock,
                                    time.time(),
                                )
                        await ws.send_json(evt)

                    elif msg.get("id") == 1:
                        if "error" in msg:
                            await ws.send_json({"type": "error", "message": msg["error"].get("message")})
                        else:
                            res = msg.get("result", {})
                            await ws.send_json({"type": "done", **res})
                        return
        except Exception as err:  # noqa: BLE001
            if not ws.closed:
                await ws.send_json({"type": "error", "message": str(err)})
        finally:
            sock.close()

    # Register URLs
    app.router.add_get("/", handle_index)
    app.router.add_get("/index.html", handle_index)
    app.router.add_post("/api/auth", handle_auth)
    app.router.add_get("/api/sessions", api_list_sessions)
    app.router.add_post("/api/sessions", api_create_session)
    app.router.add_get("/api/sessions/{session_id}", api_load_session)
    app.router.add_delete("/api/sessions/{session_id}", api_delete_session)
    app.router.add_get("/api/status", api_get_status)
    app.router.add_get("/api/models", api_list_models)
    app.router.add_post("/api/models/use", api_use_model)
    app.router.add_get("/api/skills", api_list_skills)
    app.router.add_get("/ws", handle_ws)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.router.add_static("/static/", path=str(STATIC_DIR), name="static")

    return app


async def ensure_agentd_running(socket_path: Path | str | None = None) -> None:
    """Check if agentd is running, or spawn it in background and wait for socket activation."""
    target_path = Path(socket_path).expanduser() if socket_path else get_default_socket_path()

    # Try connecting
    for _ in range(2):
        try:
            _, writer = await asyncio.open_unix_connection(str(target_path))
            writer.close()
            await writer.wait_closed()
            return  # Already running!
        except (FileNotFoundError, ConnectionRefusedError):
            pass

    # Not running: spawn background process
    import subprocess

    target_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(os.environ.get("AG_LOGS_DIR", "~/.ag/logs")).expanduser()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_dir.chmod(0o700)
    except OSError:
        pass

    log_file = log_dir / "agentd.log"
    out_fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)

    try:
        subprocess.Popen(  # noqa: ASYNC220
            [sys.executable, "-m", "ag.daemon.cli", "--socket-path", str(target_path)],
            stdout=out_fd,
            stderr=out_fd,
            start_new_session=True,
        )
    finally:
        os.close(out_fd)

    # Wait up to 3 seconds for socket to become active
    for _ in range(60):
        await asyncio.sleep(0.05)
        try:
            _, writer = await asyncio.open_unix_connection(str(target_path))
            writer.close()
            await writer.wait_closed()
            return
        except (FileNotFoundError, ConnectionRefusedError):
            continue

    raise RuntimeError(f"Failed to start agentd daemon at {target_path}")


def run_web_server(
    port: int = 3180,
    no_open: bool = False,
    sessions_dir: Path | None = None,
    daemon_socket_path: Path | None = None,
    serve_forever: bool = True,
) -> None:
    """Run the web harness server synchronously."""
    startup_token = secrets.token_urlsafe(32)
    app = create_web_app(
        startup_token=startup_token,
        port=port,
        sessions_dir=sessions_dir,
        daemon_socket_path=daemon_socket_path,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Ensure agentd is up
    try:
        loop.run_until_complete(ensure_agentd_running(daemon_socket_path))
    except Exception as err:  # noqa: BLE001
        sys.stderr.write(f"Warning: could not verify agentd: {err}\n")

    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())

    # Bind-then-handle EADDRINUSE without reuse_port
    try:
        site = web.TCPSite(runner, host="127.0.0.1", port=port, reuse_port=False)
        loop.run_until_complete(site.start())
    except OSError as err:
        loop.run_until_complete(runner.cleanup())
        if err.errno == errno.EADDRINUSE:
            sys.stderr.write(f"Error: Port {port} is already in use. Please specify a different port with --port.\n")
            sys.exit(1)
        raise

    url = f"http://127.0.0.1:{port}/#token={startup_token}"
    print(f"\nSowl web interface running at: {url}")
    print("Press Ctrl-C to stop web server.\n")

    if not no_open:
        get_launcher().open(url)

    if not serve_forever:
        loop.run_until_complete(runner.cleanup())
        loop.close()
        return

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(runner.cleanup())
        loop.close()
