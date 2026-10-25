"""Tests for KV-cache prefix stability, prompt rules, and examples."""

from ag.models.profile import ModelProfile
from ag.planner.planner import Planner
from ag.skills.manifest import ArgField, ToolManifest


def test_first_message_byte_identical_across_runs() -> None:
    profile = ModelProfile(name="test_p", tier=1, context_window=4096)
    planner = Planner(profile=profile)
    tools = {
        "files.read": ToolManifest(
            name="files.read",
            summary="Read file contents",
            risk="R0",
            args={"path": ArgField(type="string", required=True)},
        )
    }

    # Run 1: task A, delimiter 1
    msgs1 = planner.build_messages(
        task="First user query",
        tools=tools,
        history=[],
        run_delimiter="rand_delim_111",
    )

    # Run 2: task B, delimiter 2
    msgs2 = planner.build_messages(
        task="Completely different query",
        tools=tools,
        history=[],
        run_delimiter="rand_delim_222",
    )

    # First message MUST be completely byte-identical
    assert msgs1[0] == msgs2[0]
    assert msgs1[0]["role"] == "system"
    # Random delimiter must NEVER appear in the system message
    assert "rand_delim_111" not in msgs1[0]["content"]
    assert "rand_delim_222" not in msgs2[0]["content"]


def test_prompt_rules_and_refusal_example() -> None:
    profile = ModelProfile(name="test_p", tier=1, context_window=4096)
    planner = Planner(profile=profile)
    sys_prompt = planner._load_system_prompt()

    # Rule (a): subject/body/summary vs identifiers
    assert "subject" in sys_prompt.lower()
    # Rule (b): ask for email address
    assert "email" in sys_prompt.lower()
    # Rule (c): refuse with done action if no tool ("Order me a pizza")
    assert "Order me a pizza" in sys_prompt
    # Never use eval/smoke tasks as prompt examples
    assert "Delete all my files" not in sys_prompt

    # Rule 1: Conversational / no-tool questions answered directly with done
    assert "what model are you?" in sys_prompt.lower()
    assert "what can you do?" in sys_prompt.lower()
    assert "list available tools" in sys_prompt.lower() or "list the available tools" in sys_prompt.lower()

    # Rule 2: Refusals start with "I can't do that yet:" and name missing capability; never use ask for refusal
    assert "I can't do that yet:" in sys_prompt
    assert "never use ask for a refusal" in sys_prompt.lower() or "never use an ask action for a refusal" in sys_prompt.lower()


def test_multi_step_prompt_rules_conversational_and_refusal() -> None:
    profile = ModelProfile(name="test_multi", tier=2, planning_mode="multi_step", max_plan_steps=4)
    planner = Planner(profile=profile)
    sys_prompt = planner._load_system_prompt()

    assert "what model are you?" in sys_prompt.lower()
    assert "what can you do?" in sys_prompt.lower()
    assert "I can't do that yet:" in sys_prompt
    assert "never use ask for a refusal" in sys_prompt.lower() or "never use an ask action for a refusal" in sys_prompt.lower()


def test_few_shot_examples_from_manifests_only() -> None:
    profile_no_examples = ModelProfile(name="no_ex", tier=1, examples=0)
    profile_with_examples = ModelProfile(name="with_ex", tier=1, examples=1)

    tools = {
        "files.read": ToolManifest(
            name="files.read",
            summary="Read file contents",
            risk="R0",
            args={"path": ArgField(type="string", required=True)},
        )
    }

    planner_no = Planner(profile=profile_no_examples)
    planner_with = Planner(profile=profile_with_examples)

    msgs_no = planner_no.build_messages("test", tools, [], run_delimiter="d1")
    msgs_with = planner_with.build_messages("test", tools, [], run_delimiter="d2")

    # With examples=0, no Examples section
    assert "Examples:" not in msgs_no[0]["content"]
    assert msgs_with[0]["role"] == "system"
