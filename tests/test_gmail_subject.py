"""Tests for Gmail compose tool subject handling and manifest schema."""

from pathlib import Path

from ag.skills.builtin.gmail import compose_email
from ag.skills.manifest import validate_manifest_file


def test_gmail_compose_with_subject(monkeypatch) -> None:
    opened_urls = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url))

    res = compose_email(to="alice@example.com", subject="Meeting Update", body="See you at 5pm")
    assert res["ok"] is True
    assert len(opened_urls) == 1
    url = opened_urls[0]
    assert "to=alice%40example.com" in url
    assert "su=Meeting%20Update" in url
    assert "body=See%20you%20at%205pm" in url


def test_gmail_compose_empty_subject_omits_su(monkeypatch) -> None:
    opened_urls = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url))

    # Empty string subject
    res = compose_email(to="alice@example.com", subject="", body="Quick note")
    assert res["ok"] is True
    assert len(opened_urls) == 1
    url = opened_urls[0]
    assert "to=alice%40example.com" in url
    assert "su=" not in url
    assert "no%20subject" not in url.lower()
    assert "(no subject)" not in url.lower()


def test_gmail_compose_omitted_subject(monkeypatch) -> None:
    opened_urls = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url))

    # Default/omitted subject
    res = compose_email(to="bob@example.com", body="Hello Bob")
    assert res["ok"] is True
    assert len(opened_urls) == 1
    url = opened_urls[0]
    assert "to=bob%40example.com" in url
    assert "su=" not in url


def test_gmail_manifest_subject_optional() -> None:
    manifest_path = Path(__file__).resolve().parent.parent / "skills" / "gmail" / "manifest.toml"
    manifest = validate_manifest_file(manifest_path)
    tool = next(t for t in manifest.tools if t.name == "gmail.compose")
    # subject argument should be optional
    assert "subject" in tool.args
    assert tool.args["subject"].required is False
