"""Laya Gate: Unified decision model evaluating Layer A rules and Layer B safety judge."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ag.laya.judge import LayaJudge
from ag.laya.provenance import (
    ProvenanceSource,
    ProvenanceStore,
)
from ag.laya.url_check import check_url
from ag.skills.manifest import ToolManifest

ANSI_REGEX = re.compile(r"""\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])""")
CONTROL_CHAR_REGEX = re.compile(r"""[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]""")


def sanitize_display(val: Any) -> str:
    """Strip ANSI sequences and control characters for safe logging and display."""
    s = str(val)
    s = ANSI_REGEX.sub("", s)
    s = CONTROL_CHAR_REGEX.sub("", s)
    return s


class LayaDecision(BaseModel):
    """Result of Laya safety evaluation for a proposed action."""

    model_config = ConfigDict(extra="ignore")

    action: Literal["allow", "confirm", "block"] = "allow"
    reasons: list[str] = Field(default_factory=list)
    provenance: list[dict[str, str]] = Field(default_factory=list)
    is_tainted: bool = False
    never_auto_approve: bool = False
    guidance: str = ""
    layer_a_verdict: str = "allow"
    layer_b_verdict: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


def get_default_laya_log_path() -> Path:
    """Return default laya decision log path ~/.ag/laya-log.jsonl."""
    env_dir = os.environ.get("AG_WORKSPACE_DIR")
    if env_dir:
        base = Path(env_dir).expanduser()
    else:
        base = Path("~/.ag").expanduser()
    try:
        if not base.exists():
            base.mkdir(parents=True, exist_ok=True)
            base.chmod(0o700)
    except OSError:
        base = Path("/tmp/.ag")
        base.mkdir(parents=True, exist_ok=True)
    return base / "laya-log.jsonl"


class LayaGate:
    """Evaluates actions against security rules (Layer A) and optional decision model (Layer B)."""

    def __init__(self, judge: LayaJudge | None = None) -> None:
        self.judge = judge

    def evaluate(
        self,
        action: dict[str, Any],
        manifest: ToolManifest,
        provenance: ProvenanceStore,
        workspace_root: Path,
        web_port: int = 3180,
        auto_approve_tools: set[str] | None = None,
    ) -> LayaDecision:
        """Evaluate Layer A pure security rules."""
        tool_name = str(action.get("tool", ""))
        args = action.get("args", {})
        reasons: list[str] = []
        prov_list: list[dict[str, str]] = []
        is_tainted = False
        never_auto_approve = False
        is_blocked = False
        guidance = ""

        # 1. R3 extreme risk check
        if manifest.risk == "R3":
            return LayaDecision(
                action="block",
                reasons=[f"Tool '{tool_name}' is classified as R3 extreme risk; execution is permanently blocked."],
                layer_a_verdict="block",
                payload=dict(args),
            )

        # 2. Check identifiers in args
        for arg_name, arg_val in args.items():
            field_def = manifest.args.get(arg_name)
            if not field_def or getattr(field_def, "kind", "other") != "identifier":
                continue

            if not isinstance(arg_val, str):
                continue

            clean_val = sanitize_display(arg_val)
            prov = provenance.resolve(clean_val, field_name=arg_name)
            prov_entry = {
                "field": arg_name,
                "value": clean_val,
                "source": prov.source.value,
                "origin": prov.origin or "unknown",
            }
            prov_list.append(prov_entry)

            # Check if identifier has no provenance (NONE)
            if prov.source == ProvenanceSource.NONE:
                # Whitelisted relative paths (., ./) are trusted
                if prov.normalized not in (".", "./", "", "/"):
                    is_blocked = True
                    reasons.append(
                        f"Identifier '{clean_val}' in argument '{arg_name}' has no trusted provenance (NONE). Never guess identifiers."
                    )
                    guidance = (
                        f"Laya Guidance: Argument '{arg_name}' has unknown origin. Do not guess it; ask the user for clarification."
                    )

            elif prov.source == ProvenanceSource.TOOL_OUTPUT:
                is_tainted = True
                reasons.append(
                    f"Argument '{arg_name}' value '{clean_val}' came from tool output ({prov.origin}), not directly from the user."
                )

        # 3. URL safety checks
        url_target = None
        if "url" in args and isinstance(args["url"], str):
            url_target = args["url"]
        elif tool_name == "browser.open_url" and "url" in args:
            url_target = str(args["url"])

        if url_target:
            url_blocked, url_warnings, _hostname = check_url(url_target, web_port=web_port)
            if url_blocked:
                is_blocked = True
                reasons.extend(url_warnings)
            elif url_warnings:
                reasons.extend(url_warnings)
                never_auto_approve = True

        if is_blocked:
            decision = LayaDecision(
                action="block",
                reasons=reasons,
                provenance=prov_list,
                is_tainted=is_tainted,
                never_auto_approve=True,
                guidance=guidance,
                layer_a_verdict="block",
                payload=dict(args),
            )
            self._log_decision(decision, tool_name, args)
            return decision

        # 4. Risk-based determination
        final_action: Literal["allow", "confirm", "block"] = "allow"

        # R0: Allowed. Taint propagates, but does NOT trigger confirm for R0 inside workspace
        if manifest.risk == "R0":
            final_action = "allow"

        # R1: Confirm by default. Allowed if auto_approved and not tainted
        elif manifest.risk == "R1":
            if is_tainted or never_auto_approve:
                final_action = "confirm"
                never_auto_approve = True
            elif auto_approve_tools and tool_name in auto_approve_tools:
                final_action = "allow"
            else:
                final_action = "confirm"

        # R2: Always confirms with exact untruncated payload
        elif manifest.risk == "R2":
            final_action = "confirm"
            never_auto_approve = True

        decision = LayaDecision(
            action=final_action,
            reasons=reasons,
            provenance=prov_list,
            is_tainted=is_tainted,
            never_auto_approve=never_auto_approve,
            guidance=guidance,
            layer_a_verdict=final_action,
            payload=dict(args),
        )
        self._log_decision(decision, tool_name, args)
        return decision

    async def evaluate_full(
        self,
        action: dict[str, Any],
        manifest: ToolManifest,
        provenance: ProvenanceStore,
        workspace_root: Path,
        web_port: int = 3180,
        auto_approve_tools: set[str] | None = None,
    ) -> LayaDecision:
        """Run Layer A rules and then optionally adjudicate with Layer B judge."""
        decision = self.evaluate(
            action,
            manifest,
            provenance,
            workspace_root=workspace_root,
            web_port=web_port,
            auto_approve_tools=auto_approve_tools,
        )

        if not self.judge or not self.judge.provider:
            return decision

        # Call Layer B judge
        features = {
            "tool": action.get("tool"),
            "risk": manifest.risk,
            "args": action.get("args"),
        }
        adjudicated = await self.judge.adjudicate(decision, features, provenance)
        self._log_decision(adjudicated, str(action.get("tool")), action.get("args", {}))
        return adjudicated

    def _log_decision(self, decision: LayaDecision, tool_name: str, args: dict[str, Any]) -> None:
        """Append decision telemetry to ~/.ag/laya-log.jsonl (mode 0600). No file contents."""
        log_path = get_default_laya_log_path()
        try:
            entry = {
                "timestamp": time.time(),
                "tool": tool_name,
                "args_keys": list(args.keys()),
                "action": decision.action,
                "reasons": decision.reasons,
                "layer_a": decision.layer_a_verdict,
                "layer_b": decision.layer_b_verdict,
                "is_tainted": decision.is_tainted,
                "provenance": decision.provenance,
            }
            line = json.dumps(entry) + "\n"
            if not log_path.exists():
                log_path.write_text(line, encoding="utf-8")
                try:
                    log_path.chmod(0o600)
                except OSError:
                    pass
            else:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:  # noqa: BLE001, S110
            pass
