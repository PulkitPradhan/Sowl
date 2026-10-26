"""Planner: Prompt formatting, KV-cache ordering, context budgeting, and action generation."""

from __future__ import annotations

import json
import logging
import secrets
from pathlib import Path
from typing import Any

from ag.models.profile import ModelProfile
from ag.models.provider import (
    ModelProvider,
    ModelResponse,
    OpenAIProvider,
    SchemaRejectedError,
)
from ag.planner.actions import (
    Action,
    SingleAction,
    parse_action,
)
from ag.skills.manifest import ToolManifest
from ag.wire import count_tokens, from_llm, to_llm

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


class TooManyToolsError(RuntimeError):
    """Raised when more than 12 tools are registered before the router exists."""


def format_tool_signature(tool: ToolManifest) -> str:
    """Format a compact signature string for a tool's arguments."""
    if not tool.args:
        return "none"
    parts = []
    for arg_name, arg_field in tool.args.items():
        parts.append(f"{arg_name}: {arg_field.type}")
    return ", ".join(parts)


def format_tool_table(tools: dict[str, ToolManifest], max_tools: int = 12) -> str:
    """Format tools into a compact TOON table for prompt inclusion.

    Columns: name, risk, args, summary.
    Fails loudly if there are more than max_tools (default 12).
    """
    if len(tools) > max_tools:
        raise TooManyToolsError(
            f"Too many tools ({len(tools)}) registered for planner. "
            f"Maximum supported tools is {max_tools}."
        )

    rows = []
    for tool_name in sorted(tools.keys()):
        tool = tools[tool_name]
        rows.append({
            "name": tool_name,
            "risk": tool.risk,
            "args": format_tool_signature(tool),
            "summary": tool.summary,
        })

    if not rows:
        return "No tools available."

    return to_llm(rows, fmt="toon_in")


def format_observation(
    result: dict[str, Any],
    wire_format: str = "toon_in",
    context_window: int = 4096,
) -> str:
    """Format tool execution result via wire.to_llm with dynamic token budget truncation.

    Truncates DATA (rows or string length) before wire.to_llm and sets truncated: true
    if size exceeds ~10-15% of context_window in tokens.
    Observations always pass through wire.to_llm(result, fmt=wire_format).
    """
    res = dict(result)
    max_tokens = max(100, int(context_window * 0.12))  # ~12% of context window

    data = res.get("data")
    if data is not None:
        if isinstance(data, (list, str)):
            raw_str = to_llm(res, fmt=wire_format)
            tok_count, _ = count_tokens(raw_str)
            if tok_count > max_tokens:
                low = 0
                high = len(data)
                best = 0
                while low <= high:
                    mid = (low + high) // 2
                    test_res = dict(res)
                    test_res["data"] = data[:mid]
                    test_res["truncated"] = True
                    t_count, _ = count_tokens(to_llm(test_res, fmt=wire_format))
                    if t_count <= max_tokens:
                        best = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                res["data"] = data[:best]
                res["truncated"] = True
        elif isinstance(data, dict):
            raw_str = to_llm(res, fmt=wire_format)
            tok_count, _ = count_tokens(raw_str)
            if tok_count > max_tokens:
                res["data"] = str(data)[:max_tokens * 3]
                res["truncated"] = True

    return to_llm(res, fmt=wire_format)


def wrap_observation(tool_name: str, raw_data: str, delimiter_token: str) -> str:
    """Wrap raw tool output with untrusted labels and a unique run delimiter."""
    delimiter = f"--- OBSERVATION {delimiter_token} ---"
    # Sanitize delimiter token if it occurs inside data
    sanitized_data = raw_data.replace(delimiter_token, "[STRIPPED_DELIMITER]")
    return (
        f"[Observation (UNTRUSTED DATA from {tool_name})]\n"
        f"{delimiter}\n"
        f"{sanitized_data}\n"
        f"{delimiter}"
    )


class Planner:
    """Assembles prompt context and generates validated agent actions."""

    def __init__(
        self,
        profile: ModelProfile,
        provider: ModelProvider | None = None,
    ) -> None:
        self.profile = profile
        self.provider = provider or OpenAIProvider(profile)

    def _load_system_prompt(self) -> str:
        if self.profile.planning_mode == "multi_step":
            prompt_path = PROMPTS_DIR / "planner_multi.txt"
            if not prompt_path.is_file():
                prompt_path = PROMPTS_DIR / "planner.txt"
            template = prompt_path.read_text(encoding="utf-8").strip()
            return template.replace("{max_plan_steps}", str(self.profile.max_plan_steps))

        prompt_path = PROMPTS_DIR / "planner_single.txt"
        if not prompt_path.is_file():
            prompt_path = PROMPTS_DIR / "planner.txt"
        return prompt_path.read_text(encoding="utf-8").strip()

    def build_messages(
        self,
        task: str,
        tools: dict[str, ToolManifest],
        history: list[tuple[Any, str]],
        run_delimiter: str,
        skills: list[Any] | None = None,
    ) -> list[dict[str, str]]:
        """Construct KV-cache friendly messages: static prefix first, task, then history."""
        # 1. Static prefix: system prompt + tool table (no run IDs or timestamps)
        sys_prompt = self._load_system_prompt()
        max_tools = getattr(self.profile, "max_tools", 20)
        tool_table = format_tool_table(tools, max_tools=max_tools)
        examples_block = ""
        if getattr(self.profile, "examples", 0) == 1:
            all_examples = []
            for tool in tools.values():
                for ex in getattr(tool, "examples", []):
                    if ex not in all_examples:
                        all_examples.append(ex)
            if all_examples:
                examples_lines = "\n".join(f"- {ex}" for ex in all_examples)
                examples_block = f"\n\nExamples:\n{examples_lines}"

        system_content = f"{sys_prompt}\n\nAvailable Tools:\n{tool_table}{examples_block}"

        # Match on-demand skill instructions (loaded into user turn to preserve system prefix cache)
        user_content = task
        if skills:
            t_lower = task.lower()
            skill_blocks = []
            for s in skills:
                name_match = getattr(s, "name", "").lower() in t_lower
                manifest = getattr(s, "manifest", None)
                triggers = getattr(manifest, "triggers", []) if manifest else []
                trigger_match = any(trig.lower() in t_lower for trig in triggers)
                if (name_match or trigger_match) and hasattr(s, "get_instructions"):
                    inst = s.get_instructions().strip()
                    if inst:
                        skill_blocks.append(f"\n\n[Active Skill '{s.name}' Instructions]:\n{inst}")
            if skill_blocks:
                user_content += "".join(skill_blocks)

        # Prepare editable observation list for context budgeting
        observations = [obs for _, obs in history]

        # Context budget trimming: estimate * 1.25 + reserved output (500 tokens)
        reserved_output = 500

        def _estimate_total() -> int:
            text_blocks = [system_content, user_content]
            for act, obs in zip([a for a, _ in history], observations, strict=False):
                act_str = json.dumps(act) if isinstance(act, dict) else str(act)
                text_blocks.extend([act_str, obs])
            raw_count, _ = count_tokens(" ".join(text_blocks))
            return int(raw_count * 1.25) + reserved_output

        # Trim oldest observations to stubs if budget exceeded
        context_limit = int(self.profile.context_window * 0.8)
        for i in range(len(observations)):
            if _estimate_total() <= context_limit:
                break
            if "Observation truncated for context budget" not in observations[i]:
                observations[i] = "[Observation truncated for context budget: summary of previous step]"

        # Assemble final messages
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        for (action, _), obs_content in zip(history, observations, strict=False):
            # Assistant action
            if isinstance(action, dict):
                act_json = json.dumps(action, separators=(",", ":"))
            elif hasattr(action, "model_dump_json"):
                act_json = action.model_dump_json()
            else:
                act_json = str(action)

            messages.append({"role": "assistant", "content": act_json})

            # Observation wrapped in user/tool message, NEVER system
            tool_name = "tool"
            if isinstance(action, dict):
                tool_name = action.get("tool", "tool")
            elif hasattr(action, "tool"):
                tool_name = action.tool

            wrapped = wrap_observation(tool_name, obs_content, delimiter_token=run_delimiter)
            messages.append({"role": "user", "content": wrapped})

        return messages

    async def next_action(
        self,
        task: str,
        tools: dict[str, ToolManifest],
        history: list[tuple[Any, str]] | None = None,
        run_delimiter: str | None = None,
        max_tokens: int | None = None,
        skills: list[Any] | None = None,
    ) -> tuple[Action, ModelResponse]:
        """Generate the next validated agent action from the model."""
        hist = history or []
        delimiter = run_delimiter or secrets.token_hex(8)
        messages = self.build_messages(task, tools, hist, run_delimiter=delimiter, skills=skills)

        # Estimate prompt tokens for silent undercount check
        all_text = " ".join(m["content"] for m in messages)
        est_tokens, _ = count_tokens(all_text)

        schema = Action if self.profile.planning_mode == "multi_step" else SingleAction

        try:
            # Try structured output first if enabled
            response = await self.provider.chat(messages, schema=schema)
        except SchemaRejectedError:
            # Fall back to prompt-guided JSON
            from pydantic import TypeAdapter
            fallback_instruction = (
                "\n\nRespond with valid JSON matching this schema: "
                f"{TypeAdapter(schema).json_schema()}"
            )
            messages_fb = list(messages)
            messages_fb[-1] = {
                "role": messages_fb[-1]["role"],
                "content": messages_fb[-1]["content"] + fallback_instruction,
            }
            response = await self.provider.chat(messages_fb, schema=None)

        # Check for silent server truncation
        if response.usage.prompt_tokens > 0 and response.usage.prompt_tokens < (est_tokens * 0.75):
            logger.warning(
                "Potential silent prompt truncation detected! Server prompt_tokens=%d, estimated=%d",
                response.usage.prompt_tokens,
                est_tokens,
            )

        # Parse action from response content (pass finish_reason to detect length truncation)
        allow_plan = self.profile.planning_mode == "multi_step"
        decoded = from_llm(
            response.content,
            fmt=self.profile.wire_format,
            finish_reason=response.finish_reason,
        )
        action = parse_action(decoded, allow_plan=allow_plan)

        return action, response
