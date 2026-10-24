"""Tests for ArgField kind tags and LauncherBackend mock enforcement."""

from pathlib import Path

import pytest

from ag.skills.launcher import (
    MockLauncherBackend,
    RealLauncherBackend,
    get_launcher,
    set_launcher,
)
from ag.skills.manifest import ArgField, validate_manifest_file


def test_arg_field_kind_tags() -> None:
    f_ident = ArgField(type="string", kind="identifier")
    assert f_ident.kind == "identifier"

    f_text = ArgField(type="string", kind="text")
    assert f_text.kind == "text"

    f_other = ArgField(type="string")
    assert f_other.kind == "other"


def test_builtin_manifests_have_kind_tags() -> None:
    skills_dir = Path(__file__).resolve().parent.parent / "skills"
    
    # files
    files_manifest = validate_manifest_file(skills_dir / "files" / "manifest.toml")
    read_tool = next(t for t in files_manifest.tools if t.name == "files.read")
    assert read_tool.args["path"].kind == "identifier"

    # gmail
    gmail_manifest = validate_manifest_file(skills_dir / "gmail" / "manifest.toml")
    compose_tool = next(t for t in gmail_manifest.tools if t.name == "gmail.compose")
    assert compose_tool.args["to"].kind == "identifier"
    assert compose_tool.args["subject"].kind == "text"
    assert compose_tool.args["body"].kind == "text"

    # browser
    browser_manifest = validate_manifest_file(skills_dir / "browser" / "manifest.toml")
    open_tool = next(t for t in browser_manifest.tools if t.name == "browser.open_url")
    assert open_tool.args["url"].kind == "identifier"


def test_real_launcher_refuses_when_ag_eval_set(monkeypatch) -> None:
    launcher = RealLauncherBackend()
    monkeypatch.setenv("AG_EVAL", "1")
    with pytest.raises(RuntimeError, match="Real launcher execution strictly forbidden"):
        launcher.open("https://example.com")


def test_mock_launcher_records_urls() -> None:
    mock = MockLauncherBackend()
    set_launcher(mock)
    try:
        assert get_launcher() is mock
        mock.open("https://example.com/test")
        mock.open("https://mail.google.com/mail/?view=cm&to=alice%40example.com&su=Hello&body=World")
        assert len(mock.opened_urls) == 2
        
        gmail_calls = mock.get_gmail_calls()
        assert len(gmail_calls) == 1
        assert gmail_calls[0]["to"] == "alice@example.com"
        assert gmail_calls[0]["subject"] == "Hello"
        assert gmail_calls[0]["body"] == "World"
    finally:
        set_launcher(RealLauncherBackend())
