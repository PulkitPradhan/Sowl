"""Tests for Grounding Auditor detecting hallucination leaks and avoiding false positives."""

from ag.eval.grounding import audit_grounding, normalize_identifier
from ag.skills.manifest import ArgField, ToolManifest


def test_normalize_identifier() -> None:
    assert normalize_identifier("mailto:Alice@Example.com,") == "alice@example.com"
    assert normalize_identifier('"https%3A%2F%2Fexample.com%2Fpath"') == "https://example.com/path"
    assert normalize_identifier("./notes.txt.") == "./notes.txt"
    assert normalize_identifier(".") == "."
    assert normalize_identifier("") == ""


def test_audit_grounding_allows_grounded_identifiers() -> None:
    tool = ToolManifest(
        name="files.read",
        summary="Read file",
        risk="R0",
        args={
            "path": ArgField(type="string", kind="identifier", required=True),
        },
    )

    prompt = "Please read notes.txt and summarize it."
    history = []
    # Grounded path: notes.txt is in prompt
    leaks = audit_grounding(prompt, history, tool, {"path": "notes.txt"})
    assert leaks == []

    # Whitelisted path: "."
    leaks = audit_grounding(prompt, history, tool, {"path": "."})
    assert leaks == []


def test_audit_grounding_catches_hallucinated_email() -> None:
    tool = ToolManifest(
        name="gmail.compose",
        summary="Draft email",
        risk="R1",
        args={
            "to": ArgField(type="string", kind="identifier", required=True),
            "subject": ArgField(type="string", kind="text", required=False),
            "body": ArgField(type="string", kind="text", required=False),
        },
    )

    prompt = "Draft an email to Bob saying the meeting is moved."
    history = []
    # Invented email: bob@example.com was not in prompt
    leaks = audit_grounding(prompt, history, tool, {
        "to": "bob@example.com",
        "subject": "Meeting Update",
        "body": "The meeting is moved.",
    })
    assert leaks == ["bob@example.com"]


def test_audit_grounding_ignores_text_fields_avoiding_false_positives() -> None:
    tool = ToolManifest(
        name="gmail.compose",
        summary="Draft email",
        risk="R1",
        args={
            "to": ArgField(type="string", kind="identifier", required=True),
            "subject": ArgField(type="string", kind="text", required=False),
            "body": ArgField(type="string", kind="text", required=False),
        },
    )

    prompt = "Draft an email to alice@example.com about project updates."
    history = []
    # Model invents creative subject and body: MUST NOT be flagged as leaks because kind="text"!
    leaks = audit_grounding(prompt, history, tool, {
        "to": "alice@example.com",
        "subject": "Q3 Progress and Roadmap",
        "body": "Hi Alice, here are the updates on our milestone goals.",
    })
    assert leaks == []


def test_audit_grounding_allows_identifiers_from_prior_observations() -> None:
    tool = ToolManifest(
        name="files.read",
        summary="Read file",
        risk="R0",
        args={
            "path": ArgField(type="string", kind="identifier", required=True),
        },
    )

    prompt = "Check what files are in the workspace."
    history = [
        ({"kind": "tool", "tool": "files.list", "args": {"path": "."}}, "file_1.txt\nreport_2026.pdf"),
    ]
    # report_2026.pdf was discovered in prior tool observation
    leaks = audit_grounding(prompt, history, tool, {"path": "report_2026.pdf"})
    assert leaks == []
