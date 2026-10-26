"""Agent executor loop: Action validation, safety confirmation gate, loop detection, and tool execution."""

from __future__ import annotations

import asyncio
import re
import secrets
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from ag.laya.gate import LayaDecision, LayaGate
from ag.laya.judge import LayaJudge
from ag.laya.post_check import verify_post_conditions
from ag.laya.provenance import ProvenanceStore
from ag.models.profile import ModelProfile
from ag.models.provider import ModelProvider, OpenAIProvider
from ag.planner.actions import (
    AskAction,
    DoneAction,
    PlanAction,
    compute_action_hash,
)
from ag.planner.planner import (
    Planner,
    format_observation,
)
from ag.planner.trace import TraceLogger
from ag.skills.manifest import ToolManifest
from ag.skills.registry import SkillRegistry
from ag.wire import TruncatedOutputError

ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def strip_ansi_and_control(text: str) -> str:
    """Strip ANSI escape sequences and non-printable control characters."""
    without_ansi = ANSI_ESCAPE_PATTERN.sub("", text)
    return CONTROL_CHARS_PATTERN.sub("", without_ansi)


def sanitize_value(val: Any) -> Any:
    """Recursively strip ANSI and control characters from string values."""
    if isinstance(val, str):
        return strip_ansi_and_control(val)
    if isinstance(val, dict):
        return {k: sanitize_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [sanitize_value(v) for v in val]
    return val


def extract_hostname_if_url(args: dict[str, Any]) -> str | None:
    """Extract and parse hostname if any argument contains an http/https URL."""
    for val in args.values():
        if isinstance(val, str) and val.startswith(("http://", "https://")):
            try:
                parsed = urlparse(val)
                if parsed.hostname:
                    return parsed.hostname
            except Exception:  # noqa: BLE001, S110
                pass
    return None


class StepRecord(BaseModel):
    step: int
    action: dict[str, Any]
    observation_summary: str
    latency_ms: float = 0.0
    usage: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = None


class ExecutorResult(BaseModel):
    run_id: str
    stop_reason: str
    summary: str
    question: str | None = None
    steps: list[StepRecord] = Field(default_factory=list)
    total_steps: int = 0
    duration_s: float = 0.0


ConfirmHandler = Callable[[dict[str, Any]], Any]  # Can be sync or async
EventHandler = Callable[[dict[str, Any]], Any]


class Executor:
    """Executes the autonomous agent loop with strict confirmation, validation, and safety boundaries."""

    def __init__(
        self,
        profile: ModelProfile,
        registry: SkillRegistry,
        provider: ModelProvider | None = None,
        confirm_handler: ConfirmHandler | None = None,
        event_handler: EventHandler | None = None,
        trace_logger: TraceLogger | None = None,
        max_steps: int = 6,
        confirm_timeout: float = 60.0,
    ) -> None:
        self.profile = profile
        self.registry = registry
        self.provider = provider or OpenAIProvider(profile)
        self.confirm_handler = confirm_handler
        self.event_handler = event_handler
        self.trace_logger = trace_logger or TraceLogger()
        self.max_steps = max_steps
        self.confirm_timeout = confirm_timeout
        self.planner = Planner(profile=self.profile, provider=self.provider)
        self.allow_tainted_auto_approve = False

        judge: LayaJudge | None = None
        judge_setting = getattr(self.profile, "laya_judge", "off")
        if judge_setting in ("high_risk", "always"):
            judge_model = getattr(self.profile, "laya_model", "") or self.profile.model
            judge_profile = self.profile.model_copy(update={"model": judge_model, "thinking": "off"})
            judge = LayaJudge(provider=OpenAIProvider(judge_profile))

        self.gate = LayaGate(judge=judge)

    async def _emit_event(self, event_type: str, run_id: str, data: dict[str, Any]) -> None:
        if self.event_handler is not None:
            payload = {"run_id": run_id, "type": event_type, **data}
            res = self.event_handler(payload)
            if asyncio.iscoroutine(res):
                await res

    async def _request_confirmation(
        self,
        run_id: str,
        step_idx: int,
        tool: ToolManifest,
        args: dict[str, Any],
        decision: LayaDecision | None = None,
    ) -> bool:
        """Prompt confirmation gate for non-R0 tools."""
        clean_args = sanitize_value(args)
        hostname = extract_hostname_if_url(clean_args)
        action_hash = compute_action_hash(tool.name, clean_args)

        confirm_data = {
            "run_id": run_id,
            "step": step_idx,
            "action_hash": action_hash,
            "tool": tool.name,
            "args": clean_args,
            "risk": tool.risk,
            "hostname": hostname,
            "provenance": decision.provenance if decision else [],
            "is_tainted": decision.is_tainted if decision else False,
            "never_auto_approve": decision.never_auto_approve if decision else False,
        }

        await self._emit_event("confirm", run_id, confirm_data)

        self.trace_logger.log_confirm_request(
            run_id=run_id,
            step_idx=step_idx,
            tool=tool.name,
            args=clean_args,
            risk=tool.risk,
            action_hash=action_hash,
        )

        if self.confirm_handler is None:
            # Default policy: without a handler, non-R0 is denied
            self.trace_logger.log_confirm_decision(
                run_id=run_id,
                step_idx=step_idx,
                tool=tool.name,
                action_hash=action_hash,
                approved=False,
                source="policy",
            )
            return False

        try:
            res = self.confirm_handler(confirm_data)
            if asyncio.iscoroutine(res):
                res_val = await asyncio.wait_for(res, timeout=self.confirm_timeout)
            else:
                res_val = res

            if isinstance(res_val, dict):
                approval = bool(res_val.get("approved", False))
                source = str(res_val.get("source", "user"))
                allow_tainted = bool(res_val.get("allow_tainted", False))
            else:
                approval = bool(res_val)
                source = "user"
                allow_tainted = False

            if (
                decision
                and decision.is_tainted
                and not allow_tainted
                and not self.allow_tainted_auto_approve
                and (source in ("policy", "auto_approve") or not isinstance(res_val, dict))
            ):
                approval = False

            self.trace_logger.log_confirm_decision(
                run_id=run_id,
                step_idx=step_idx,
                tool=tool.name,
                action_hash=action_hash,
                approved=approval,
                source=source,
            )
            return approval
        except TimeoutError:
            self.trace_logger.log_confirm_decision(
                run_id=run_id,
                step_idx=step_idx,
                tool=tool.name,
                action_hash=action_hash,
                approved=False,
                source="timeout",
            )
            return False
        except Exception:  # noqa: BLE001
            self.trace_logger.log_confirm_decision(
                run_id=run_id,
                step_idx=step_idx,
                tool=tool.name,
                action_hash=action_hash,
                approved=False,
                source="policy",
            )
            return False

    async def run(self, task: str, run_id: str | None = None) -> ExecutorResult:
        """Run the complete agent loop for a task."""
        rid = run_id or f"run_{secrets.token_hex(6)}"
        start_time = time.time()
        self.registry.discover()
        tools = self.registry.get_tools()

        provenance = ProvenanceStore(workspace_root=self.registry.workspace_dir)
        provenance.add_user_text(task)
        consecutive_blocks = 0

        self.trace_logger.log_init(
            rid,
            task,
            self.profile.name,
            wire_format=self.profile.wire_format,
            endpoint=self.profile.endpoint,
        )
        await self._emit_event("init", rid, {"task": task, "profile": self.profile.name})

        # Tier 0 check
        if self.profile.tier == 0:
            summary = "No model configured for Tier 0 (rule engine only)."
            self.trace_logger.log_done(rid, "no_model", summary, 0, time.time() - start_time)
            await self._emit_event("done", rid, {"stop_reason": "no_model", "summary": summary})
            return ExecutorResult(
                run_id=rid,
                stop_reason="no_model",
                summary=summary,
                total_steps=0,
                duration_s=time.time() - start_time,
            )

        run_delimiter = secrets.token_hex(8)
        history: list[tuple[Any, str]] = []
        step_records: list[StepRecord] = []
        action_history_hashes: set[str] = set()

        invalid_action_count = 0
        step_idx = 0
        current_max_tokens = self.profile.planner_max_tokens

        try:
            while step_idx < self.max_steps:
                step_idx += 1
                loop_start = time.perf_counter()

                # Check run timeout
                if (time.time() - start_time) > self.profile.run_timeout:
                    return self._finish(
                        rid, "timeout", "Run exceeded configured profile run_timeout",
                        step_records, start_time
                    )

                # Call planner
                try:
                    action, response = await self.planner.next_action(
                        task=task,
                        tools=tools,
                        history=history,
                        run_delimiter=run_delimiter,
                        max_tokens=current_max_tokens,
                    )
                except TruncatedOutputError:
                    # Retry once with doubled token budget if length truncation occurred
                    if current_max_tokens == self.profile.planner_max_tokens:
                        current_max_tokens *= 2
                        try:
                            action, response = await self.planner.next_action(
                                task=task,
                                tools=tools,
                                history=history,
                                run_delimiter=run_delimiter,
                                max_tokens=current_max_tokens,
                            )
                        except Exception as err:  # noqa: BLE001
                            return self._finish(rid, "error", f"Model output truncated: {err}", step_records, start_time)
                    else:
                        return self._finish(rid, "error", "Model output truncated twice.", step_records, start_time)
                except Exception as err:  # noqa: BLE001
                    invalid_action_count += 1
                    if invalid_action_count >= 3:
                        return self._finish(
                            rid, "error",
                            f"Exceeded invalid action limit of 3. Last error: {err}",
                            step_records, start_time,
                        )
                    history.append(({"kind": "error"}, f"Invalid action format: {err}. Please return valid JSON action."))
                    continue

                latency = (time.perf_counter() - loop_start) * 1000.0

                # Handle DoneAction
                if isinstance(action, DoneAction):
                    rec = StepRecord(
                        step=step_idx,
                        action=action.model_dump(),
                        observation_summary=action.summary,
                        latency_ms=latency,
                        usage=response.usage.model_dump(),
                        finish_reason=response.finish_reason,
                    )
                    step_records.append(rec)
                    self.trace_logger.log_step(
                        rid, step_idx, action.model_dump(), action.summary,
                        latency, response.usage.model_dump(), response.finish_reason
                    )
                    await self._emit_event("step", rid, {"step": step_idx, "action": action.model_dump()})
                    return self._finish(rid, "done", action.summary, step_records, start_time)

                # Handle AskAction
                if isinstance(action, AskAction):
                    rec = StepRecord(
                        step=step_idx,
                        action=action.model_dump(),
                        observation_summary=action.question,
                        latency_ms=latency,
                        usage=response.usage.model_dump(),
                        finish_reason=response.finish_reason,
                    )
                    step_records.append(rec)
                    self.trace_logger.log_step(
                        rid, step_idx, action.model_dump(), action.question,
                        latency, response.usage.model_dump(), response.finish_reason
                    )
                    await self._emit_event("step", rid, {"step": step_idx, "action": action.model_dump()})
                    return self._finish(
                        rid, "ask", f"Model requested user input: {action.question}",
                        step_records, start_time, question=action.question
                    )

                # Process ToolAction or PlanAction
                tool_steps = action.steps if isinstance(action, PlanAction) else [action]

                plan_failed = False
                for tool_act in tool_steps:
                    action_dict = tool_act.model_dump()
                    await self._emit_event("step", rid, {"step": step_idx, "action": action_dict})

                    # Loop detection
                    act_hash = compute_action_hash(tool_act.tool, tool_act.args)
                    if act_hash in action_history_hashes:
                        rec = StepRecord(
                            step=step_idx,
                            action=action_dict,
                            observation_summary="Loop detected: same action called twice",
                            latency_ms=latency,
                            usage=response.usage.model_dump(),
                            finish_reason=response.finish_reason,
                        )
                        step_records.append(rec)
                        return self._finish(
                            rid, "loop",
                            f"Execution stopped: loop detected on '{tool_act.tool}'",
                            step_records, start_time,
                        )
                    action_history_hashes.add(act_hash)

                    # Validate tool exists
                    if tool_act.tool not in tools:
                        invalid_action_count += 1
                        err_msg = f"Tool '{tool_act.tool}' does not exist. Available tools: {', '.join(tools.keys())}."
                        if invalid_action_count >= 3:
                            return self._finish(
                                rid, "error",
                                f"Exceeded invalid action limit of 3. Unknown tool: {tool_act.tool}",
                                step_records, start_time,
                            )
                        history.append((action_dict, err_msg))
                        plan_failed = True
                        break

                    tool_manifest = tools[tool_act.tool]

                    # Validate args against schema
                    try:
                        tool_manifest.validate_args(tool_act.args)
                    except Exception as err:  # noqa: BLE001
                        invalid_action_count += 1
                        err_msg = f"Invalid arguments for '{tool_act.tool}': {err}."
                        if invalid_action_count >= 3:
                            return self._finish(
                                rid, "error",
                                f"Exceeded invalid action limit of 3. Bad args: {err}",
                                step_records, start_time,
                            )
                        history.append((action_dict, err_msg))
                        plan_failed = True
                        break

                    # Laya gate evaluation (Layer A rules + Layer B judge)
                    decision = await self.gate.evaluate_full(
                        action=action_dict,
                        manifest=tool_manifest,
                        provenance=provenance,
                        workspace_root=self.registry.workspace_dir,
                        web_port=3180,
                    )

                    self.trace_logger.log_laya(
                        rid,
                        step_idx,
                        tool_act.tool,
                        decision.action,
                        decision.reasons,
                        decision.provenance,
                    )

                    if decision.action == "block":
                        consecutive_blocks += 1
                        if consecutive_blocks >= 2:
                            return self._finish(
                                rid, "blocked",
                                f"Laya gate blocked execution: {'; '.join(decision.reasons)}",
                                step_records, start_time,
                            )
                        reason_msg = f"Laya Blocked: {'; '.join(decision.reasons)}."
                        if decision.guidance:
                            reason_msg += f" {decision.guidance}"
                        history.append((action_dict, reason_msg))
                        plan_failed = True
                        break

                    consecutive_blocks = 0

                    if decision.action == "confirm":
                        approved = await self._request_confirmation(
                            rid, step_idx, tool_manifest, tool_act.args, decision=decision
                        )
                        if not approved:
                            return self._finish(
                                rid, "denied",
                                f"User denied execution of tool '{tool_act.tool}'",
                                step_records, start_time,
                            )

                    # Execute tool via registry
                    exec_result = await self.registry.execute_tool(tool_act.tool, tool_act.args)

                    # Post-checks verification
                    post_ok, post_err = verify_post_conditions(
                        tool_act.tool,
                        tool_manifest,
                        tool_act.args,
                        exec_result.get("data"),
                        workspace_root=self.registry.workspace_dir,
                    )
                    if not post_ok:
                        history.append((action_dict, f"Post-check failure: {post_err}"))
                        plan_failed = True
                        break

                    # Update provenance store with tool output
                    provenance.add_tool_observation(
                        tool_act.tool,
                        exec_result.get("data") or exec_result,
                        origin=f"step_{step_idx}_{tool_act.tool}",
                        is_tainted_input=decision.is_tainted,
                    )
                    if (
                        tool_act.tool == "contacts.find"
                        and isinstance(exec_result.get("data"), dict)
                        and exec_result["data"].get("found")
                        and exec_result["data"].get("count") == 1
                    ):
                        matched = exec_result["data"].get("match", {})
                        if "email" in matched:
                            provenance.add_trusted_identifier(matched["email"], origin="contacts.find")
                    formatted_obs = format_observation(
                        exec_result,
                        wire_format=self.profile.wire_format,
                        context_window=self.profile.context_window,
                    )

                    await self._emit_event(
                        "observation", rid,
                        {"step": step_idx, "tool": tool_act.tool, "observation": formatted_obs}
                    )

                    history.append((action_dict, formatted_obs))

                    obs_summary = (
                        formatted_obs[:100] + "..." if len(formatted_obs) > 100 else formatted_obs
                    )
                    rec = StepRecord(
                        step=step_idx,
                        action=action_dict,
                        observation_summary=obs_summary,
                        latency_ms=latency,
                        usage=response.usage.model_dump(),
                        finish_reason=response.finish_reason,
                    )
                    step_records.append(rec)
                    self.trace_logger.log_step(
                        rid, step_idx, action_dict, obs_summary,
                        latency, response.usage.model_dump(), response.finish_reason
                    )

                    # If tool execution returned failure, abort remaining plan steps and re-plan
                    if not exec_result.get("ok"):
                        plan_failed = True
                        break

                if plan_failed:
                    continue

            # Max steps exceeded
            return self._finish(
                rid, "max_steps",
                f"Agent exceeded maximum allowed steps ({self.max_steps}).",
                step_records, start_time,
            )

        except asyncio.CancelledError:
            return self._finish(rid, "cancelled", "Run was cancelled by client.", step_records, start_time)

    def _finish(
        self,
        run_id: str,
        stop_reason: str,
        summary: str,
        steps: list[StepRecord],
        start_time: float,
        question: str | None = None,
    ) -> ExecutorResult:
        duration = time.time() - start_time
        self.trace_logger.log_done(
            run_id, stop_reason, summary, len(steps), duration
        )
        return ExecutorResult(
            run_id=run_id,
            stop_reason=stop_reason,
            summary=summary,
            question=question,
            steps=steps,
            total_steps=len(steps),
            duration_s=round(duration, 2),
        )
