"""ag CLI client powered by Typer with fast-path execution for ping."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Annotated, Any

from ag.daemon.server import is_loopback_endpoint
from ag.wire.rpc import get_default_socket_path

_app = None


def get_app():
    """Lazily construct and return the Typer app to keep CLI startup fast."""
    global _app
    if _app is not None:
        return _app

    import asyncio

    import typer

    from ag.client.rpc import RpcClient, RpcRemoteError

    # Make typer accessible in module globals for annotation string evaluation
    globals()["typer"] = typer

    app = typer.Typer(name="ag", help="ag: Local-first personal AI agent CLI")
    state = {"socket_path": get_default_socket_path()}

    @app.callback()
    def common(
        socket_path: Annotated[
            Path | None,
            typer.Option(
                "--socket-path",
                "-s",
                help="Path to agentd Unix domain socket",
            ),
        ] = None,
    ) -> None:
        if socket_path is not None:
            state["socket_path"] = socket_path

    @app.command()
    def ping() -> None:
        """Check connectivity to agentd."""
        async def _ping():
            async with RpcClient(state["socket_path"]) as client:
                return await client.call("ping")

        try:
            res = asyncio.run(_ping())
            typer.echo(res)
        except FileNotFoundError:
            typer.secho(
                f"Error: agentd socket not found at {state['socket_path']}. Is agentd running?",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        except ConnectionRefusedError:
            typer.secho(
                f"Error: Connection refused at {state['socket_path']}. agentd may have terminated.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        except RpcRemoteError as err:
            typer.secho(f"RPC Error: {err}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    @app.command()
    def run(
        text: Annotated[str, typer.Argument(help="Task or instruction to execute")],
        endpoint: Annotated[str | None, typer.Option("--endpoint", help="Custom OpenAI endpoint override")] = None,
        allow_remote: Annotated[bool, typer.Option("--allow-remote", help="Allow non-loopback endpoints")] = False,
    ) -> None:
        """Execute an autonomous agent task with streaming steps, confirmation, and trace logging."""
        if endpoint and not allow_remote and not is_loopback_endpoint(endpoint):
            typer.secho(
                "Error: --endpoint must be a loopback address (e.g. 127.0.0.1, localhost) unless --allow-remote is set.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        import secrets
        import socket

        from ag.wire.rpc import format_request

        target = Path(state["socket_path"]).expanduser()
        current_task = text

        while True:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(str(target))
            except (FileNotFoundError, ConnectionRefusedError):
                typer.secho(
                    f"Error: agentd socket not found at {target}. Is agentd running? Start it with 'ag daemon start'.",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)

            run_id = f"run_{secrets.token_hex(6)}"
            run_payload = {"task": current_task, "run_id": run_id}
            if endpoint:
                run_payload["endpoint"] = endpoint
                run_payload["allow_remote"] = allow_remote
            req_bytes = format_request("run", run_payload, msg_id=1)
            sock.sendall(req_bytes)

            buffer = b""
            final_result = None

            try:
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if not line.strip():
                            continue
                        msg = json.loads(line.decode("utf-8"))

                        # Handle notifications (streaming events)
                        if msg.get("method") == "run.event":
                            evt = msg.get("params", {})
                            evt_type = evt.get("type")

                            if evt_type == "queued":
                                typer.secho("Agent is queued waiting for model lock...", fg=typer.colors.YELLOW)
                            elif evt_type == "step":
                                act = evt.get("action", {})
                                kind = act.get("kind", "")
                                tool_name = act.get("tool", "")
                                if kind == "tool":
                                    typer.secho(f"\n[Step {evt.get('step')}] Tool: {tool_name}", fg=typer.colors.CYAN)
                                    args_str = json.dumps(act.get("args", {}), indent=2)
                                    typer.echo(f"  Args: {args_str}")
                                elif kind == "done":
                                    typer.secho(f"\n[Done] {act.get('summary')}", fg=typer.colors.GREEN)
                                elif kind == "ask":
                                    typer.secho(f"\n[Ask] {act.get('question')}", fg=typer.colors.MAGENTA)
                            elif evt_type == "observation":
                                obs = evt.get("observation", "")
                                preview = obs[:200] + "..." if len(obs) > 200 else obs
                                typer.echo(f"  ↳ Obs: {preview}")
                            elif evt_type == "confirm":
                                tool = evt.get("tool")
                                risk = evt.get("risk", "R1")
                                host = evt.get("hostname")
                                host_info = f" ({host})" if host else ""
                                typer.secho(f"\n⚠️  [Confirmation Required] Tool: {tool} [{risk}]{host_info}", fg=typer.colors.YELLOW, bold=True)
                                typer.echo(f"  Arguments: {json.dumps(evt.get('args', {}), indent=2)}")
                                prov_list = evt.get("provenance", [])
                                if prov_list:
                                    typer.secho("  Provenance:", fg=typer.colors.CYAN)
                                    for p in prov_list:
                                        p_field = p.get("field", "")
                                        p_val = p.get("value", "")
                                        p_src = p.get("source", "")
                                        p_orig = p.get("origin", "")
                                        typer.echo(f"    • {p_field}: '{p_val}' [source: {p_src}, origin: {p_orig}]")
                                resp = typer.prompt("  Approve execution? [y/N]", default="n").strip().lower()
                                approved = resp in ("y", "yes")

                                confirm_req = format_request(
                                    "run.confirm",
                                    {
                                        "run_id": evt.get("run_id"),
                                        "step": evt.get("step"),
                                        "action_hash": evt.get("action_hash"),
                                        "approve": approved,
                                    },
                                    msg_id=2,
                                )
                                sock.sendall(confirm_req)

                        # Handle final response
                        elif msg.get("id") == 1:
                            if "error" in msg:
                                typer.secho(f"\nRun Error: {msg['error'].get('message')}", fg=typer.colors.RED)
                                raise typer.Exit(code=1)
                            final_result = msg.get("result")
                            break
                    if final_result is not None:
                        break

            except KeyboardInterrupt:
                typer.secho("\nCancelling run...", fg=typer.colors.YELLOW)
                cancel_req = format_request("run.cancel", {"run_id": run_id}, msg_id=99)
                try:
                    sock.sendall(cancel_req)
                except Exception:  # noqa: BLE001, S110
                    pass
                raise typer.Exit(code=130)
            finally:
                sock.close()

            if not final_result:
                break

            stop_reason = final_result.get("stop_reason")
            summary = final_result.get("summary")

            if stop_reason == "ask":
                question = final_result.get("question") or summary
                typer.secho(f"\nAgent asked: {question}", fg=typer.colors.MAGENTA, bold=True)
                answer = typer.prompt("Your Answer").strip()
                if not answer:
                    break
                current_task = f"{current_task}\nQuestion: {question}\nAnswer: {answer}"
                typer.echo("Continuing task with your answer...")
                continue

            typer.echo("\n" + "=" * 50)
            typer.secho(f"Stop Reason: {stop_reason}", fg=typer.colors.GREEN if stop_reason == "done" else typer.colors.YELLOW)
            typer.echo(f"Summary:     {summary}")
            typer.echo(f"Total Steps: {final_result.get('total_steps')} in {final_result.get('duration_s')}s")
            typer.echo("=" * 50)
            break

    @app.command()
    def status() -> None:
        """Get status from agentd."""
        async def _status():
            async with RpcClient(state["socket_path"]) as client:
                return await client.call("status")

        try:
            res = asyncio.run(_status())
            typer.echo(json.dumps(res, indent=2))
        except (FileNotFoundError, ConnectionRefusedError):
            typer.secho("agentd: stopped (socket inactive)", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        except RpcRemoteError as err:
            typer.secho(f"RPC Error: {err}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    @app.command()
    def web(
        port: Annotated[int, typer.Option("--port", "-p", help="Port to listen on")] = 3180,
        no_open: Annotated[bool, typer.Option("--no-open", help="Do not automatically open the browser")] = False,
    ) -> None:
        """Start the local web harness interface."""
        from ag.web.server import run_web_server

        run_web_server(
            port=port,
            no_open=no_open,
            daemon_socket_path=state["socket_path"],
        )

    daemon_app = typer.Typer(name="daemon", help="Manage background agentd daemon")
    app.add_typer(daemon_app, name="daemon")

    @daemon_app.command(name="start")
    def daemon_start(
        detach: Annotated[bool, typer.Option("--detach/--no-detach", help="Run daemon in background")] = True,
    ) -> None:
        """Start the agentd daemon process."""
        import subprocess

        from ag.daemon.server import (
            AgentDaemon,
            get_default_socket_path,
            get_log_dir,
            register_default_handlers,
        )

        sock_path = state["socket_path"] or get_default_socket_path()
        if detach:
            log_dir = get_log_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = open(log_dir / "agentd.log", "a", encoding="utf-8")  # noqa: SIM115
            cmd = [sys.executable, "-m", "ag.daemon.cli", "--socket-path", str(sock_path)]
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )
            typer.secho(f"agentd started in background (PID {proc.pid}) on socket: {sock_path}", fg=typer.colors.GREEN)
        else:
            daemon = AgentDaemon(sock_path)
            register_default_handlers(daemon)
            typer.echo(f"Starting agentd (pid: {os.getpid()}) on socket: {sock_path}")
            try:
                asyncio.run(daemon.serve_forever())
            except (KeyboardInterrupt, SystemExit):
                pass
            finally:
                typer.echo("\nagentd stopped.")

    @daemon_app.command(name="restart")
    def daemon_restart() -> None:
        """Restart running agentd daemon."""
        daemon_stop()
        daemon_start(detach=True)

    @daemon_app.command(name="status")
    def daemon_status() -> None:
        """Check status of agentd daemon."""
        status()

    @daemon_app.command(name="stop")
    def daemon_stop() -> None:
        """Stop running agentd daemon."""
        import signal

        from ag.daemon.server import get_log_dir

        pid_file = get_log_dir() / "agentd.pid"
        if not pid_file.is_file():
            typer.secho("agentd: stopped (no pidfile found)", fg=typer.colors.YELLOW)
            return

        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            os.kill(pid, signal.SIGTERM)
            typer.secho(f"Sent SIGTERM to agentd (PID {pid}).", fg=typer.colors.GREEN)
        except ProcessLookupError:
            typer.secho("agentd process was not running (stale pidfile cleaned).", fg=typer.colors.YELLOW)
            pid_file.unlink(missing_ok=True)
        except Exception as err:  # noqa: BLE001
            typer.secho(f"Failed to stop agentd: {err}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    browser_app = typer.Typer(name="browser", help="Manage dedicated controlled browser instance")
    app.add_typer(browser_app, name="browser")

    @browser_app.command(name="start")
    def browser_start(
        port: Annotated[int | None, typer.Option("--port", "-p", help="Remote debugging port")] = None,
    ) -> None:
        """Start the dedicated Chrome browser instance with loopback CDP."""
        from ag.browser.process import start_browser

        try:
            info = asyncio.run(start_browser(port=port))
            typer.secho(
                f"Dedicated browser started (PID {info['pid']}) on {info['cdp_url']}",
                fg=typer.colors.GREEN,
            )
        except Exception as err:  # noqa: BLE001
            typer.secho(f"Failed to start browser: {err}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    @browser_app.command(name="status")
    def browser_status_cmd() -> None:
        """Check status of dedicated Chrome browser instance."""
        from ag.browser.process import get_browser_status

        info = asyncio.run(get_browser_status())
        if info["running"]:
            typer.secho(
                f"Browser is running: PID {info['pid']}, CDP: {info['cdp_url']} ({info.get('browser', 'Chrome')})",
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho("Browser is stopped.", fg=typer.colors.YELLOW)

    @browser_app.command(name="stop")
    def browser_stop_cmd() -> None:
        """Stop dedicated Chrome browser instance."""
        from ag.browser.process import stop_browser
        from ag.browser.session import reset_browser_session

        asyncio.run(reset_browser_session())
        stopped = asyncio.run(stop_browser())
        if stopped:
            typer.secho("Browser stopped.", fg=typer.colors.GREEN)
        else:
            typer.secho("Browser was not running.", fg=typer.colors.YELLOW)

    session_app = typer.Typer(name="session", help="Manage persistent sessions")
    app.add_typer(session_app, name="session")

    @session_app.command(name="list")
    def session_list() -> None:
        """List active sessions."""
        async def _list():
            async with RpcClient(state["socket_path"]) as client:
                return await client.call("session.list")

        try:
            sessions = asyncio.run(_list())
            if not sessions:
                typer.echo("No active sessions.")
                return
            for s in sessions:
                typer.echo(f"ID: {s.get('session_id')} | Title: {s.get('title')} | Updated: {s.get('updated_at')}")
        except Exception as err:  # noqa: BLE001
            typer.secho(f"Error: {err}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    @session_app.command(name="new")
    def session_new(
        title: Annotated[str | None, typer.Option("--title", "-t", help="Session title")] = None,
    ) -> None:
        """Create a new session."""
        async def _new():
            async with RpcClient(state["socket_path"]) as client:
                return await client.call("session.new", {"title": title})

        try:
            res = asyncio.run(_new())
            typer.echo(f"Created session: {res.get('session_id')}")
        except Exception as err:  # noqa: BLE001
            typer.secho(f"Error: {err}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    @session_app.command(name="show")
    def session_show(session_id: str) -> None:
        """Show session history."""
        async def _show():
            async with RpcClient(state["socket_path"]) as client:
                return await client.call("session.load", {"session_id": session_id})

        try:
            res = asyncio.run(_show())
            typer.echo(json.dumps(res, indent=2))
        except Exception as err:  # noqa: BLE001
            typer.secho(f"Error: {err}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    @session_app.command(name="delete")
    def session_delete(session_id: str) -> None:
        """Delete a session."""
        async def _delete():
            async with RpcClient(state["socket_path"]) as client:
                return await client.call("session.delete", {"session_id": session_id})

        try:
            asyncio.run(_delete())
            typer.echo(f"Deleted session: {session_id}")
        except Exception as err:  # noqa: BLE001
            typer.secho(f"Error: {err}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    model_app = typer.Typer(name="model", help="Manage model profiles")
    app.add_typer(model_app, name="model")

    @model_app.command(name="list")
    def model_list() -> None:
        """List all available profiles."""
        async def _list():
            async with RpcClient(state["socket_path"]) as client:
                return await client.call("model.list")

        profiles = None
        try:
            profiles = asyncio.run(_list())
        except (FileNotFoundError, ConnectionRefusedError):
            # Fall back to reading profiles directly from filesystem if agentd is offline
            from ag.models.profile import get_active_profile, list_profiles
            active = get_active_profile()
            profiles = [
                {
                    "name": p.name,
                    "tier": p.tier,
                    "model": p.model,
                    "endpoint": p.endpoint,
                    "active": (p.name == active),
                }
                for p in list_profiles()
            ]
        except RpcRemoteError as err:
            typer.secho(f"RPC Error: {err}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        typer.echo(f"{'NAME':<15} {'TIER':<6} {'MODEL':<18} {'ENDPOINT':<28} {'ACTIVE'}")
        typer.echo("-" * 75)
        for p in profiles:
            active_mark = "*" if p.get("active") else ""
            model_display = p.get("model") or "(not set)"
            endpoint_display = p.get("endpoint") or "(none)"
            typer.echo(
                f"{p['name']:<15} {p['tier']:<6} {model_display:<18} {endpoint_display:<28} {active_mark}"
            )

    @model_app.command(name="use")
    def model_use(
        name: Annotated[str, typer.Argument(help="Name of the profile to activate")],
    ) -> None:
        """Activate a profile for agentd."""
        from ag.models.profile import load_profile, set_active_profile

        try:
            profile = load_profile(name)
        except FileNotFoundError:
            typer.secho(f"Error: Profile '{name}' not found.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        set_active_profile(profile.name)

        async def _use():
            async with RpcClient(state["socket_path"]) as client:
                return await client.call("model.use", {"name": profile.name})

        try:
            asyncio.run(_use())
        except (FileNotFoundError, ConnectionRefusedError):
            pass  # Saved to workspace; will be picked up next time agentd starts

        typer.secho(f"Switched active profile to: {profile.name} (Tier {profile.tier})", fg=typer.colors.GREEN)

    @app.command(name="doctor")
    def doctor(
        no_bench: Annotated[
            bool,
            typer.Option("--no-bench", help="Skip model performance benchmark"),
        ] = False,
    ) -> None:
        """Run system diagnostics, list server models, and recommend a profile."""
        from ag.models.doctor import (
            benchmark_endpoint,
            get_system_specs,
            list_server_models,
            recommend_tier,
        )
        from ag.models.profile import get_active_profile, load_profile

        specs = get_system_specs()
        rec_tier = recommend_tier(specs.vram_gb, specs.ram_gb, specs.cpu_cores)

        tier_profile_map = {
            0: "tier0_none",
            1: "tier1_cpu",
            2: "tier2_gpu",
            3: "tier3_heavy",
            4: "tier4_cloud",
        }
        rec_profile = tier_profile_map.get(rec_tier, "tier1_cpu")

        # Determine endpoint to query
        active_name = get_active_profile()
        endpoint = "http://localhost:11434/v1"
        active_model = ""
        if active_name:
            try:
                p = load_profile(active_name)
                if p.endpoint:
                    endpoint = p.endpoint
                active_model = p.model
            except (FileNotFoundError, ValueError):
                pass

        installed_models = asyncio.run(list_server_models(endpoint))

        typer.echo("\n[ag doctor] System Diagnostics")
        typer.echo("=" * 50)
        typer.echo(f"  CPU Cores:         {specs.cpu_cores}")
        typer.echo(f"  System RAM:        {specs.ram_gb:.2f} GB")
        vram_str = f" ({specs.vram_gb:.2f} GB VRAM)" if specs.vram_gb else ""
        typer.echo(f"  GPU:               {specs.gpu_name}{vram_str}")
        typer.echo(f"  Active Profile:    {active_name or '(none)'}")
        typer.echo(f"  Server Endpoint:   {endpoint}")

        if installed_models:
            typer.echo(f"  Installed Models:  {', '.join(installed_models)}")
        else:
            typer.echo("  Installed Models:  None found (server offline or no models loaded)")

        typer.echo("-" * 50)
        typer.echo(f"  Recommended Tier:    Tier {rec_tier}")
        typer.echo(f"  Recommended Profile: {rec_profile}")

        # Benchmark if requested and server models available
        if not no_bench:
            if not active_name:
                typer.echo("\n  Benchmark skipped: No active profile selected. Select one with 'ag model use <name>'.")
            elif not active_model:
                typer.echo(f"\n  Benchmark skipped: Active profile '{active_name}' has no model specified.")
            elif active_model not in installed_models:
                typer.secho(f"\n  Warning: Active profile model '{active_model}' is not in installed models: {', '.join(installed_models)}", fg=typer.colors.YELLOW)
            else:
                typer.echo(f"\nRunning benchmark on active model '{active_model}' (warm-up + fixed prompt, ~100 tokens)...")
                try:
                    active_prof = load_profile(active_name)
                    bench = asyncio.run(benchmark_endpoint(
                        endpoint=endpoint,
                        model=active_model,
                        tier=active_prof.tier,
                        profile_context=active_prof.context_window,
                    ))
                    typer.echo(f"  Estimated TTFT:         {bench.ttft_ms:.2f} ms")
                    typer.echo(f"  Total Throughput:       {bench.tokens_per_sec:.2f} tokens/sec")
                    typer.echo(f"  Content Throughput:     {bench.content_tokens_per_sec:.2f} tokens/sec")
                    if bench.reasoning_tokens > 0:
                        typer.echo(f"  Reasoning Throughput:   {bench.reasoning_tokens_per_sec:.2f} tokens/sec ({bench.reasoning_tokens} reasoning tokens)")
                    typer.echo(f"  Tokens Generated:       {bench.tokens_generated}")
                    if bench.cpu_warning:
                        typer.secho("  ⚠️ Warning: Model throughput is unusually low; it may be running on CPU rather than GPU.", fg=typer.colors.YELLOW)
                    if bench.context_warning:
                        typer.secho(f"  ⚠️ Warning: {bench.context_warning}", fg=typer.colors.YELLOW)
                except Exception as err:  # noqa: BLE001
                    typer.secho(f"  Benchmark error: {err}", fg=typer.colors.YELLOW)
        typer.echo("=" * 50 + "\n")

    @app.command(name="wire")
    def wire_cmd(
        file: Annotated[
            Path | None,
            typer.Argument(help="JSON file to analyze (defaults to standard fixtures)"),
        ] = None,
        count: Annotated[
            bool,
            typer.Option("--count", "-c", help="Print token counts for JSON vs TOON"),
        ] = False,
    ) -> None:
        """Analyze wire format (TOON vs JSON) token efficiency."""
        from ag.wire.wire import count_tokens, to_llm

        def _analyze_file(target: Path) -> dict[str, Any]:
            raw_text = target.read_text(encoding="utf-8")
            data = json.loads(raw_text)
            indented = json.dumps(data, indent=2)
            compact = json.dumps(data, separators=(",", ":"))
            toon = to_llm(data, fmt="toon_both")

            t_indented, label = count_tokens(indented)
            t_compact, _ = count_tokens(compact)
            t_toon, _ = count_tokens(toon)

            saving_compact = ((t_compact - t_toon) / t_compact * 100) if t_compact > 0 else 0.0
            saving_indented = ((t_indented - t_toon) / t_indented * 100) if t_indented > 0 else 0.0

            return {
                "name": target.name,
                "t_indented": t_indented,
                "t_compact": t_compact,
                "t_toon": t_toon,
                "saving_compact": saving_compact,
                "saving_indented": saving_indented,
                "label": label,
            }

        files_to_check: list[Path] = []
        if file is not None:
            if not file.is_file():
                typer.secho(f"Error: File '{file}' not found.", fg=typer.colors.RED)
                raise typer.Exit(code=1)
            files_to_check.append(file)
        else:
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            fixture_dir = repo_root / "tests" / "fixtures"
            for name in ("tools.json", "snapshot.json", "plan.json"):
                candidate = fixture_dir / name
                if candidate.is_file():
                    files_to_check.append(candidate)

        if not files_to_check:
            typer.secho("Error: No JSON file specified and fixtures not found.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        typer.echo("\n[ag wire] Token Efficiency Report (TOON vs JSON)")
        typer.echo("=" * 70)
        typer.echo(f"{'FIXTURE / FILE':<18} {'COMPACT JSON':<14} {'TOON':<10} {'SAVINGS vs COMPACT'}")
        typer.echo("-" * 70)

        total_compact = 0
        total_toon = 0
        last_label = ""

        for f in files_to_check:
            stats = _analyze_file(f)
            total_compact += stats["t_compact"]
            total_toon += stats["t_toon"]
            last_label = stats["label"]
            typer.echo(
                f"{stats['name']:<18} {stats['t_compact']:<14} {stats['t_toon']:<10} {stats['saving_compact']:>6.1f}%"
            )

        typer.echo("=" * 70)
        headline_saving = (
            ((total_compact - total_toon) / total_compact * 100) if total_compact > 0 else 0.0
        )
        typer.secho(
            f"Headline Savings vs Compact JSON: {headline_saving:.1f}% ({last_label})",
            fg=typer.colors.GREEN,
            bold=True,
        )
        typer.echo(f"Total Tokens: Compact JSON = {total_compact} | TOON = {total_toon}\n")

    skills_app = typer.Typer(name="skills", help="Manage and inspect agent skills")
    app.add_typer(skills_app, name="skills")

    @skills_app.command(name="list")
    def skills_list() -> None:
        """List all discovered skills and their risk levels."""
        from ag.skills.registry import SkillRegistry

        registry = SkillRegistry()
        registry.discover()
        skills = registry.list_skills()

        typer.echo(f"\n{'NAME':<15} {'RISK':<6} {'TOOLS':<6} {'SUMMARY'}")
        typer.echo("-" * 75)
        for s in skills:
            typer.echo(f"{s.name:<15} {s.risk:<6} {len(s.tools):<6} {s.summary}")

        if registry.broken_manifests:
            typer.secho("\nBroken Manifests Found:", fg=typer.colors.YELLOW, bold=True)
            for bm in registry.broken_manifests:
                typer.secho(f"  • File:  {bm['file']}", fg=typer.colors.RED)
                typer.secho(f"    Field: {bm['field']}", fg=typer.colors.YELLOW)
                typer.secho(f"    Error: {bm['error']}\n", fg=typer.colors.RED)
        typer.echo()

    @skills_app.command(name="show")
    def skills_show(
        name: Annotated[str, typer.Argument(help="Name of the skill to inspect")],
    ) -> None:
        """Show full details, tool signatures, and SKILL.md for a skill."""
        from ag.skills.registry import SkillRegistry

        registry = SkillRegistry()
        registry.discover()
        skill = registry.get_skill(name)

        if not skill:
            typer.secho(f"Error: Skill '{name}' not found.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        typer.echo(f"\n[Skill: {skill.name}] (Risk: {skill.risk})")
        typer.echo("=" * 60)
        typer.echo(f"Summary:  {skill.summary}")
        if skill.manifest.triggers:
            typer.echo(f"Triggers: {', '.join(skill.manifest.triggers)}")
        if skill.manifest.examples:
            typer.echo("Examples:")
            for ex in skill.manifest.examples:
                typer.echo(f"  - {ex}")

        typer.echo("\nTools:")
        for t in skill.tools:
            typer.echo(f"  • {t.name} [{t.risk}]: {t.summary}")
            if t.args:
                for arg_name, arg_def in t.args.items():
                    req = "(required)" if arg_def.required else f"(optional, default={arg_def.default})"
                    typer.echo(f"      - {arg_name} ({arg_def.type}) {req}: {arg_def.description}")

        typer.echo("\n--- SKILL.md Instructions ---")
        typer.echo(skill.get_instructions())
        typer.echo("=" * 60 + "\n")

    # -------------------------------------------------------------------------
    # ag trace
    # -------------------------------------------------------------------------
    trace_app = typer.Typer(name="trace", help="Inspect agent execution traces")
    app.add_typer(trace_app)

    @trace_app.command("last")
    def trace_last() -> None:
        """Display the most recent execution trace."""
        from ag.planner.trace import get_default_traces_dir, list_traces, load_last_trace

        traces_dir = get_default_traces_dir()
        files = list_traces(traces_dir)
        if not files:
            typer.echo("No traces found.")
            return

        records = load_last_trace(traces_dir)
        typer.echo(f"\n[Trace: {files[0].stem}]")
        typer.echo("=" * 60)
        for r in records:
            r_type = r.get("type")
            if r_type == "init":
                typer.echo(f"Task:      {r.get('task')}")
                typer.echo(f"Profile:   {r.get('profile')}")
            elif r_type == "step":
                act = r.get("action", {})
                typer.echo(f"\nStep {r.get('step')}: {act.get('kind')} {act.get('tool', '')}")
                obs_sum = r.get("observation_summary", "")
                if obs_sum:
                    typer.echo(f"  Obs:     {obs_sum}")
                usage = r.get("usage", {})
                typer.echo(f"  Latency: {r.get('latency_ms')}ms | Usage: {usage}")
            elif r_type == "done":
                typer.echo("\n" + "-" * 60)
                typer.echo(f"Stop Reason: {r.get('stop_reason')}")
                typer.echo(f"Summary:     {r.get('summary')}")
                typer.echo(f"Total Steps: {r.get('total_steps')} | Duration: {r.get('duration_s')}s")
        typer.echo("=" * 60 + "\n")

    # -------------------------------------------------------------------------
    # ag prompt
    # -------------------------------------------------------------------------
    prompt_app = typer.Typer(name="prompt", help="Inspect prompt assembly and token counts")
    app.add_typer(prompt_app)

    @prompt_app.command("show")
    def prompt_show(
        task: Annotated[str, typer.Argument(help="Task prompt to inspect")],
        profile_name: Annotated[str | None, typer.Option("--profile", "-p", help="Model profile to load")] = None,
        sample_observation: Annotated[bool, typer.Option("--sample-observation", help="Include sample files.list observation in history")] = False,
    ) -> None:
        """Display exact messages, active wire_format, and token counts without calling model."""
        from ag.models.profile import get_active_profile, load_profile
        from ag.planner.planner import Planner, format_observation
        from ag.skills.registry import SkillRegistry
        from ag.wire import count_tokens

        active = profile_name or get_active_profile() or "tier1_cpu"
        profile = load_profile(active)

        registry = SkillRegistry()
        registry.discover()
        tools = registry.get_tools()

        planner = Planner(profile=profile)
        history: list[tuple[Any, str]] = []

        if sample_observation:
            fake_files = {
                "ok": True,
                "data": [
                    {"name": "file_1.txt", "size": 1024, "type": "file"},
                    {"name": "file_2.py", "size": 2048, "type": "file"},
                    {"name": "notes.md", "size": 512, "type": "file"},
                    {"name": "data.csv", "size": 4096, "type": "file"},
                    {"name": "readme.rst", "size": 256, "type": "file"},
                ],
                "error": None,
            }
            obs_formatted = format_observation(
                fake_files,
                wire_format=profile.wire_format,
                context_window=profile.context_window,
            )
            history.append(
                (
                    {"kind": "tool", "tool": "files.list", "args": {"path": "."}},
                    obs_formatted,
                )
            )

        messages = planner.build_messages(
            task=task,
            tools=tools,
            history=history,
            run_delimiter="sample_run_delimiter",
        )

        all_text = " ".join(m["content"] for m in messages)
        approx_tokens, label = count_tokens(all_text)

        typer.secho(f"\n[Prompt Inspection: {profile.name}]", fg=typer.colors.CYAN, bold=True)
        typer.echo(f"Wire format: {profile.wire_format}")
        typer.echo(f"Approximate tokens: {approx_tokens} ({label})\n")

        for idx, msg in enumerate(messages, 1):
            role = msg["role"]
            typer.secho(f"--- Message {idx} [Role: {role}] ---", fg=typer.colors.YELLOW)
            typer.echo(msg["content"] + "\n")

    # -------------------------------------------------------------------------
    # ag eval
    # -------------------------------------------------------------------------
    eval_app = typer.Typer(name="eval", help="Agent evaluation harness and benchmarking", invoke_without_command=True)
    app.add_typer(eval_app)

    @eval_app.callback(invoke_without_command=True)
    def eval_run(
        ctx: typer.Context,
        profile_name: Annotated[str | None, typer.Option("--profile", "-p", help="Profile to evaluate")] = None,
        wire_format: Annotated[str | None, typer.Option("--format", "-f", help="Wire format override (json, toon_in, toon_both)")] = None,
        thinking: Annotated[str | None, typer.Option("--thinking", help="Thinking mode (on or off)")] = None,
        examples: Annotated[int | None, typer.Option("--examples", help="Include few-shot examples (0 or 1)")] = None,
        runs: Annotated[int, typer.Option("--runs", "-r", help="Number of runs per task")] = 1,
        quick: Annotated[bool, typer.Option("--quick", help="Run quick 20-task subset")] = False,
        laya: Annotated[str, typer.Option("--laya", help="Laya gate mode: off, rules, rules+judge")] = "rules",
        endpoint: Annotated[str | None, typer.Option("--endpoint", help="Custom OpenAI endpoint override")] = None,
        monitor_process: Annotated[str, typer.Option("--monitor-process", help="Process name to monitor memory for")] = "ollama",
    ) -> None:
        """Run in-process evaluation benchmark suite across defined scenarios."""
        if ctx.invoked_subcommand is not None:
            return

        import asyncio

        from ag.eval.report import format_eval_summary_table, save_report
        from ag.eval.runner import EvalRunner
        from ag.models.profile import get_active_profile

        active = profile_name or get_active_profile() or "tier2_gpu"

        # Validate endpoint loopback
        if endpoint and not is_loopback_endpoint(endpoint):
            typer.secho("Error: --endpoint must be a loopback address unless authorized.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        runner_inst = EvalRunner(
            profile_name=active,
            wire_format=wire_format,
            thinking=thinking,
            examples=examples,
            runs=runs,
            quick=quick,
            endpoint=endpoint,
            monitor_process=monitor_process,
            laya_mode=laya,
        )

        typer.secho(f"\nStarting evaluation on profile '{active}' ({runner_inst.profile.model})...", fg=typer.colors.CYAN, bold=True)
        report = asyncio.run(runner_inst.run_suite())

        # Print formatted report table
        typer.echo(format_eval_summary_table(report))

        # Save to evals/results/<timestamp>.json
        saved_path = save_report(report)
        typer.secho(f"\nResults saved to: {saved_path}\n", fg=typer.colors.GREEN)

    @eval_app.command("compare")
    def eval_compare(
        file_a: Annotated[Path, typer.Argument(help="Path to first report JSON")],
        file_b: Annotated[Path, typer.Argument(help="Path to second report JSON")],
    ) -> None:
        """Compare two evaluation report JSON files side-by-side with noise threshold check."""
        from ag.eval.report import compare_reports, load_report

        rep_a = load_report(file_a)
        rep_b = load_report(file_b)
        typer.echo("\n" + compare_reports(rep_a, rep_b) + "\n")

    @eval_app.command("show")
    def eval_show(
        file_path: Annotated[Path, typer.Argument(help="Path to report JSON")],
        failures: Annotated[bool, typer.Option("--failures", help="Show only failed task details")] = False,
    ) -> None:
        """Display summary or failure details from a saved evaluation report JSON file."""
        from ag.eval.report import format_eval_summary_table, format_failures, load_report

        report = load_report(file_path)
        if failures:
            typer.echo(format_failures(report))
        else:
            typer.echo(format_eval_summary_table(report))

    _app = app
    return _app


def __getattr__(name: str) -> Any:
    if name == "app":
        return get_app()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def _try_fast_ping(args: list[str]) -> int | None:
    """Fast-path for `ag ping` to avoid heavy CLI framework import latency."""
    socket_path_str = os.environ.get("AG_SOCKET_PATH")
    i = 0
    is_ping = False

    while i < len(args):
        arg = args[i]
        if arg in ("-s", "--socket-path") and i + 1 < len(args):
            socket_path_str = args[i + 1]
            i += 2
        elif arg.startswith("--socket-path="):
            socket_path_str = arg.split("=", 1)[1]
            i += 1
        elif arg == "ping":
            is_ping = True
            i += 1
        else:
            return None  # Fall back to full Typer parser for any other argument/help
    if not is_ping:
        return None

    target = Path(socket_path_str).expanduser() if socket_path_str else get_default_socket_path()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(2.0)
        sock.connect(str(target))
        sock.sendall(b'{"jsonrpc":"2.0","method":"ping","id":1}\n')
        raw = sock.recv(1024)
        data = json.loads(raw.decode("utf-8"))
        if "result" in data:
            sys.stdout.write(f"{data['result']}\n")
            return 0
        if "error" in data:
            sys.stderr.write(f"RPC Error: {data['error'].get('message', 'Unknown error')}\n")
            return 1
    except FileNotFoundError:
        sys.stderr.write(f"Error: agentd socket not found at {target}. Is agentd running?\n")
        return 1
    except ConnectionRefusedError:
        sys.stderr.write(
            f"Error: Connection refused at {target}. agentd may have terminated.\n"
        )
        return 1
    except Exception as err:  # noqa: BLE001
        sys.stderr.write(f"Error: {err}\n")
        return 1
    finally:
        sock.close()
    return 0


def main() -> None:
    """Entry point for the ag CLI command."""
    ret = _try_fast_ping(sys.argv[1:])
    if ret is not None:
        sys.exit(ret)
    get_app()()


if __name__ == "__main__":
    main()
