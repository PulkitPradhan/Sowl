"""Tests for Phase 4 Skills system: manifests, registry, security, and CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from ag.client.cli import app
from ag.skills.builtin.browser import open_url
from ag.skills.builtin.files import read_file
from ag.skills.builtin.gmail import compose_email
from ag.skills.manifest import (
    ManifestValidationError,
    SkillManifest,
    ToolManifest,
    parse_manifest_data,
    validate_manifest_file,
)
from ag.skills.registry import DuplicateSkillError, SkillRegistry

runner = CliRunner()


# ---------------------------------------------------------------------------
# 1. Manifest Parsing & Validation Tests
# ---------------------------------------------------------------------------


def test_valid_manifest_parsing(tmp_path: Path):
    manifest_file = tmp_path / "manifest.toml"
    manifest_file.write_text(
        """
name = "demo_skill"
summary = "A demonstration skill"
triggers = ["demo", "test"]
examples = ["run demo"]

[[tools]]
name = "demo_skill.run"
summary = "Run demo tool"
risk = "R0"
handler = "ag.skills.builtin.echo:echo"
pre_conditions = ["file_exists"]
post_conditions = ["success"]

[tools.args.text]
type = "string"
description = "Text to echo"
required = true
""",
        encoding="utf-8",
    )

    manifest = validate_manifest_file(manifest_file)
    assert manifest.name == "demo_skill"
    assert manifest.summary == "A demonstration skill"
    assert len(manifest.tools) == 1
    assert manifest.tools[0].name == "demo_skill.run"
    assert manifest.tools[0].risk == "R0"
    assert "text" in manifest.tools[0].args
    assert manifest.tools[0].args["text"].type == "string"


def test_manifest_invalid_name_regex():
    with pytest.raises(ManifestValidationError) as exc:
        parse_manifest_data({
            "name": "123_invalid",
            "summary": "Bad name",
            "triggers": [],
            "examples": [],
            "tools": [],
        })
    assert "name" in str(exc.value).lower()


def test_manifest_summary_capped_and_sanitized():
    # Long summary (> 120 chars) should be rejected
    long_summary = "A" * 150
    with pytest.raises(ManifestValidationError) as exc:
        parse_manifest_data({
            "name": "valid_name",
            "summary": long_summary,
            "triggers": [],
            "examples": [],
            "tools": [],
        })
    assert "summary" in str(exc.value).lower()

    # Newlines and control characters must be stripped
    manifest = SkillManifest(
        name="valid_name",
        summary="Hello\n\r\tWorld",
        triggers=[],
        examples=[],
        tools=[],
    )
    assert "\n" not in manifest.summary
    assert "\r" not in manifest.summary
    assert manifest.summary == "Hello World"


def test_manifest_invalid_toml_file(tmp_path: Path):
    bad_toml = tmp_path / "manifest.toml"
    bad_toml.write_text("name = [unclosed bracket", encoding="utf-8")
    with pytest.raises(ManifestValidationError) as exc:
        validate_manifest_file(bad_toml)
    assert "manifest.toml" in str(exc.value)


def test_manifest_missing_required_field(tmp_path: Path):
    bad_manifest = tmp_path / "manifest.toml"
    bad_manifest.write_text("name = 'test'\n", encoding="utf-8")  # Missing summary, tools
    with pytest.raises(ManifestValidationError) as exc:
        validate_manifest_file(bad_manifest)
    assert "summary" in str(exc.value) or "tools" in str(exc.value)


def test_manifest_invalid_risk_level():
    with pytest.raises(ManifestValidationError) as exc:
        parse_manifest_data({
            "name": "test_skill",
            "summary": "Test skill",
            "triggers": [],
            "examples": [],
            "tools": [
                {
                    "name": "test_skill.tool",
                    "summary": "Test tool",
                    "risk": "R5",  # Invalid risk
                    "args": {},
                }
            ],
        })
    assert "risk" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# 2. Registry Discovery & Collision Tests
# ---------------------------------------------------------------------------


def test_duplicate_skill_collision_raises_error(tmp_path: Path):
    dir1 = tmp_path / "dir1" / "skill_a"
    dir1.mkdir(parents=True)
    (dir1 / "manifest.toml").write_text(
        'name = "skill_a"\nsummary = "Summary 1"\ntriggers = []\nexamples = []\ntools = []\n',
        encoding="utf-8",
    )

    dir2 = tmp_path / "dir2" / "skill_a"
    dir2.mkdir(parents=True)
    (dir2 / "manifest.toml").write_text(
        'name = "skill_a"\nsummary = "Summary 2"\ntriggers = []\nexamples = []\ntools = []\n',
        encoding="utf-8",
    )

    registry = SkillRegistry(search_dirs=[tmp_path / "dir1", tmp_path / "dir2"])
    with pytest.raises(DuplicateSkillError) as exc:
        registry.discover()
    assert "skill_a" in str(exc.value)


def test_drop_in_new_skill_folder_appears_without_code_changes(tmp_path: Path):
    skills_root = tmp_path / "skills"
    skills_root.mkdir()

    registry = SkillRegistry(search_dirs=[skills_root])
    assert len(registry.list_skills()) == 0

    # Drop in a new skill folder
    new_skill = skills_root / "dynamic_skill"
    new_skill.mkdir()
    (new_skill / "manifest.toml").write_text(
        'name = "dynamic_skill"\nsummary = "Dynamic skill"\ntriggers = []\nexamples = []\ntools = []\n',
        encoding="utf-8",
    )
    (new_skill / "SKILL.md").write_text("# Dynamic Skill\nOne line.\n", encoding="utf-8")

    registry.discover()
    assert len(registry.list_skills()) == 1
    assert registry.get_skill("dynamic_skill") is not None


def test_malformed_manifest_does_not_crash_registry(tmp_path: Path):
    skills_root = tmp_path / "skills"
    skills_root.mkdir()

    # Good skill
    good = skills_root / "good"
    good.mkdir()
    (good / "manifest.toml").write_text(
        'name = "good"\nsummary = "Good skill"\ntriggers = []\nexamples = []\ntools = []\n',
        encoding="utf-8",
    )

    # Broken skill
    broken = skills_root / "broken"
    broken.mkdir()
    (broken / "manifest.toml").write_text("invalid toml syntax {{{", encoding="utf-8")

    registry = SkillRegistry(search_dirs=[skills_root])
    registry.discover()

    # Registry still loads the good skill and notes the broken manifest
    assert len(registry.list_skills()) == 1
    assert len(registry.broken_manifests) == 1
    broken_info = registry.broken_manifests[0]
    assert "broken" in broken_info["file"]


def test_skill_risk_equals_max_of_tools():
    tool0 = ToolManifest(name="skill.tool0", summary="t0", risk="R0", args={})
    tool2 = ToolManifest(name="skill.tool2", summary="t2", risk="R2", args={})
    tool1 = ToolManifest(name="skill.tool1", summary="t1", risk="R1", args={})

    manifest = SkillManifest(
        name="mixed_skill",
        summary="Mixed risk tools",
        triggers=[],
        examples=[],
        tools=[tool0, tool2, tool1],
    )
    assert manifest.risk == "R2"


# ---------------------------------------------------------------------------
# 3. Tool Execution, Sandboxing & Handlers Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_execute_tool_handler_allowlist_check(tmp_path: Path):
    # A manifest attempting to load a non-builtin handler must be rejected
    skill_dir = tmp_path / "evil"
    skill_dir.mkdir()
    (skill_dir / "manifest.toml").write_text(
        """
name = "evil"
summary = "Evil skill"
triggers = []
examples = []

[[tools]]
name = "evil.hack"
summary = "Hack system"
risk = "R1"
handler = "os:system"
args = {}
""",
        encoding="utf-8",
    )

    registry = SkillRegistry(search_dirs=[tmp_path])
    registry.discover()

    res = await registry.execute_tool("evil.hack", {})
    assert res["ok"] is False
    assert "allowlist" in res["error"].lower()


@pytest.mark.anyio
async def test_execute_tool_r3_blocked(tmp_path: Path):
    skill_dir = tmp_path / "nuke"
    skill_dir.mkdir()
    (skill_dir / "manifest.toml").write_text(
        """
name = "nuke"
summary = "Nuke system"
triggers = []
examples = []

[[tools]]
name = "nuke.destroy"
summary = "Destroy system"
risk = "R3"
handler = "ag.skills.builtin.echo:echo"
args = {}
""",
        encoding="utf-8",
    )

    registry = SkillRegistry(search_dirs=[tmp_path])
    registry.discover()

    res = await registry.execute_tool("nuke.destroy", {})
    assert res["ok"] is False
    assert "r3" in res["error"].lower()


@pytest.mark.anyio
async def test_execute_tool_extra_arg_forbidden(tmp_path: Path):
    skill_dir = tmp_path / "echo_skill"
    skill_dir.mkdir()
    (skill_dir / "manifest.toml").write_text(
        """
name = "echo_skill"
summary = "Echo skill"
triggers = []
examples = []

[[tools]]
name = "echo_skill.echo"
summary = "Echo text"
risk = "R0"
handler = "ag.skills.builtin.echo:echo"

[tools.args.text]
type = "string"
description = "Text to echo"
required = true
""",
        encoding="utf-8",
    )

    registry = SkillRegistry(search_dirs=[tmp_path])
    registry.discover()

    # Pass unexpected extra argument
    res = await registry.execute_tool("echo_skill.echo", {"text": "hi", "unexpected_arg": 123})
    assert res["ok"] is False
    assert "extra" in res["error"].lower() or "unexpected" in res["error"].lower()


@pytest.mark.anyio
async def test_execute_tool_timeout_and_exceptions(tmp_path: Path):
    # Test handling of exception in handler
    skill_dir = tmp_path / "failing"
    skill_dir.mkdir()
    (skill_dir / "manifest.toml").write_text(
        """
name = "failing"
summary = "Failing skill"
triggers = []
examples = []

[[tools]]
name = "failing.fail"
summary = "Fail tool"
risk = "R0"
handler = "ag.skills.builtin.echo:failing_tool"
args = {}
""",
        encoding="utf-8",
    )

    registry = SkillRegistry(search_dirs=[tmp_path])
    registry.discover()

    res = await registry.execute_tool("failing.fail", {})
    assert res["ok"] is False
    assert "error" in res


# ---------------------------------------------------------------------------
# 4. Security Tests for Starter Skills (files, browser, gmail)
# ---------------------------------------------------------------------------


def test_files_path_traversal_and_symlink_escape(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    (secret_dir / "password.txt").write_text("classified", encoding="utf-8")

    # Path traversal attempt
    res = read_file("../secret/password.txt", root=root)
    assert res["ok"] is False
    assert "containment" in res["error"].lower() or "escape" in res["error"].lower()

    # Symlink escape attempt
    symlink = root / "leak_link"
    symlink.symlink_to(secret_dir / "password.txt")

    res_sym = read_file("leak_link", root=root)
    assert res_sym["ok"] is False
    assert "containment" in res_sym["error"].lower() or "escape" in res_sym["error"].lower()


def test_files_deny_patterns(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()

    for forbidden in (".env", "id_rsa", "server.pem", "id_rsa.pub", "priv.key"):
        file_path = root / forbidden
        file_path.write_text("secret", encoding="utf-8")
        res = read_file(forbidden, root=root)
        assert res["ok"] is False
        assert "denied" in res["error"].lower() or "forbidden" in res["error"].lower()


def test_files_refuse_binary_file(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    binary_file = root / "test.bin"
    binary_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe")

    res = read_file("test.bin", root=root)
    assert res["ok"] is False
    assert "binary" in res["error"].lower()


def test_files_read_cap_100kb(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    large_file = root / "large.txt"
    large_file.write_text("A" * 150_000, encoding="utf-8")

    res = read_file("large.txt", root=root)
    assert res["ok"] is True
    assert res["data"]["truncated"] is True
    assert len(res["data"]["content"]) == 100_000


@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(path_str=st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789/.-_\\", max_size=30))
def test_files_hypothesis_random_paths(tmp_path: Path, path_str: str):
    root = tmp_path / "sandbox_root"
    root.mkdir(exist_ok=True)
    res = read_file(path_str, root=root)
    if not res["ok"]:
        assert res["error"] is not None


def test_browser_url_scheme_rejection():
    with patch("webbrowser.open") as mock_open:
        # Rejected schemes
        for bad_url in (
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/html,<b>hello</b>",
            "chrome://settings",
        ):
            res = open_url(bad_url)
            assert res["ok"] is False
            assert "scheme" in res["error"].lower()
            mock_open.assert_not_called()

        # Allowed schemes
        res_http = open_url("https://example.com/docs")
        assert res_http["ok"] is True
        mock_open.assert_called_once_with("https://example.com/docs")


def test_gmail_compose_url_encoding_and_capping():
    with patch("webbrowser.open") as mock_open:
        res = compose_email(
            to="user@example.com",
            subject="Hello, Sowl! & Test",
            body="Line 1\nLine 2 with Unicode: 🚀 ñ 世界",
        )
        assert res["ok"] is True
        url = res["data"]["url"]
        assert url.startswith("https://mail.google.com/mail/?view=cm&fs=1")
        assert "to=user%40example.com" in url
        assert "%0A" in url  # Newline encoded
        assert "%F0%9F%9A%80" in url or "🚀" not in url  # Properly percent-encoded
        mock_open.assert_called_once_with(url)


def test_gmail_compose_body_capping():
    with patch("webbrowser.open"):
        long_body = "x" * 5000
        res = compose_email(to="a@b.com", subject="Test", body=long_body)
        assert res["ok"] is True
        assert res["data"]["truncated"] is True


# ---------------------------------------------------------------------------
# 5. Compact TOON Table & Starter Skills Integration
# ---------------------------------------------------------------------------


def test_compact_skills_table():
    registry = SkillRegistry()
    registry.discover()
    table = registry.get_compact_table(fmt="toon_in")
    assert "skills[" in table
    assert "name,summary,risk" in table
    assert "files" in table
    assert "browser" in table
    assert "gmail" in table


def test_cli_skills_list():
    result = runner.invoke(app, ["skills", "list"])
    assert result.exit_code == 0
    assert "files" in result.stdout
    assert "browser" in result.stdout
    assert "gmail" in result.stdout


def test_cli_skills_show():
    result = runner.invoke(app, ["skills", "show", "files"])
    assert result.exit_code == 0
    assert "files" in result.stdout
    assert "files.read" in result.stdout
    assert "files.list" in result.stdout
