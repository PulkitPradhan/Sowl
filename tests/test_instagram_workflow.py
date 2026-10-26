"""Tests for Instagram browser workflow: Autonomous initial navigation, login detection, and provenance grounding."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ag.laya.gate import LayaGate
from ag.laya.provenance import ProvenanceSource, ProvenanceStore
from ag.models.profile import ModelProfile
from ag.models.provider import ModelResponse, TokenUsage
from ag.planner.actions import AskAction, ToolAction
from ag.planner.planner import Planner
from ag.skills.manifest import ArgField, ToolManifest
from ag.skills.registry import SkillRegistry


def test_provenance_service_domain_matching() -> None:
    """When user mentions 'Instagram', URLs like 'https://www.instagram.com/' should be grounded."""
    store = ProvenanceStore()
    store.add_user_text("Open my Instagram saved page and count my saved reels and posts")

    prov = store.resolve("https://www.instagram.com/", field_name="url")
    assert prov.source in (ProvenanceSource.USER, ProvenanceSource.TRUSTED)

    # Laya gate evaluation on browser.goto must not be blocked for provenance
    gate = LayaGate()
    manifest_goto = ToolManifest(
        name="browser.goto",
        risk="R1",
        summary="Navigate to URL",
        args={"url": ArgField(type="string", kind="identifier")},
    )
    decision = gate.evaluate(
        {"tool": "browser.goto", "args": {"url": "https://www.instagram.com/"}},
        manifest_goto,
        store,
        Path.cwd(),
    )
    assert decision.action == "confirm"
    assert not any("no trusted provenance" in r for r in decision.reasons)


def test_provenance_registers_skill_instructions_urls() -> None:
    """URLs defined in active skill instructions should be registered as TRUSTED."""
    store = ProvenanceStore()
    store.add_skill_text("instagram", "Navigate first to: browser.goto(url='https://www.instagram.com/')")

    prov = store.resolve("https://www.instagram.com/", field_name="url")
    assert prov.source == ProvenanceSource.TRUSTED
    assert "skill:instagram" in prov.origin


@pytest.mark.anyio
async def test_planner_step_1_opens_instagram_first_without_asking() -> None:
    """Step 1 MUST navigate to Instagram directly, not ask for username upfront."""
    profile = ModelProfile(name="test_profile", tier=1, planning_mode="single_pass", wire_format="toon_in")
    
    # Mock provider returning browser.goto
    mock_provider = AsyncMock()
    mock_provider.chat.return_value = ModelResponse(
        content='{"kind": "tool", "tool": "browser.goto", "args": {"url": "https://www.instagram.com/"}}',
        usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        finish_reason="stop",
    )

    planner = Planner(profile=profile, provider=mock_provider)
    registry = SkillRegistry()
    registry.discover()
    tools = registry.get_tools(attached_browser=True)
    skills = registry.list_skills()

    task = "Open my Instagram saved page and count my saved reels and posts"
    action, _ = await planner.next_action(task=task, tools=tools, history=[], skills=skills)

    assert isinstance(action, ToolAction)
    assert action.tool == "browser.goto"
    assert "instagram.com" in action.args.get("url", "")


@pytest.mark.anyio
async def test_planner_detects_login_page_and_asks_user() -> None:
    """When snapshot indicates user is logged out, the agent must ask the user to log in."""
    profile = ModelProfile(name="test_profile", tier=1, planning_mode="single_pass", wire_format="toon_in")
    
    mock_provider = AsyncMock()
    mock_provider.chat.return_value = ModelResponse(
        content='{"kind": "ask", "question": "Please log in to your Instagram account in the open browser window, then let me know once you are logged in."}',
        usage=TokenUsage(prompt_tokens=100, completion_tokens=30, total_tokens=130),
        finish_reason="stop",
    )

    planner = Planner(profile=profile, provider=mock_provider)
    registry = SkillRegistry()
    registry.discover()
    tools = registry.get_tools(attached_browser=True)
    skills = registry.list_skills()

    task = "Open my Instagram saved page and count my saved reels and posts"
    # History showing goto and snapshot of login screen
    history = [
        ({"kind": "tool", "tool": "browser.goto", "args": {"url": "https://www.instagram.com/"}}, "ok: true"),
        ({"kind": "tool", "tool": "browser.snapshot", "args": {}}, "elements[2]{ref,role,name}\n  @e1 button 'Log In'\n  @e2 button 'Sign Up'"),
    ]
    action, _ = await planner.next_action(task=task, tools=tools, history=history, skills=skills)

    assert isinstance(action, AskAction)
    assert "log in" in action.question.lower()
