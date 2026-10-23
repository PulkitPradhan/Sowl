"""ag CLI client powered by Typer with fast-path execution for ping."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Annotated, Any

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
        text: Annotated[str, typer.Argument(help="Text to send to agentd")],
    ) -> None:
        """Send text to agentd (Phase 1 echo)."""
        async def _run():
            async with RpcClient(state["socket_path"]) as client:
                return await client.call("run", {"text": text})

        try:
            res = asyncio.run(_run())
            if isinstance(res, (dict, list)):
                typer.echo(json.dumps(res, indent=2))
            else:
                typer.echo(str(res))
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
            bench_model = active_model or (installed_models[0] if installed_models else None)
            if bench_model and installed_models:
                typer.echo(f"\nRunning benchmark on model '{bench_model}' (warm-up + fixed prompt, ~100 tokens)...")
                try:
                    bench = asyncio.run(benchmark_endpoint(endpoint=endpoint, model=bench_model))
                    typer.echo(f"  Estimated TTFT:    {bench.ttft_ms:.2f} ms")
                    typer.echo(f"  Throughput:        {bench.tokens_per_sec:.2f} tokens/sec")
                    typer.echo(f"  Tokens Generated:  {bench.tokens_generated}")
                except Exception as err:  # noqa: BLE001
                    typer.secho(f"  Benchmark skipped: {err}", fg=typer.colors.YELLOW)
            else:
                typer.echo("\n  Benchmark skipped: No model server reachable or no models installed.")
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
