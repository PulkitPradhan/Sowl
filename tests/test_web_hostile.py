"""Tests for hostile content sanitization: ANSI codes, control chars, and BiDi overrides."""

from __future__ import annotations

import pytest

from ag.web.sanitize import sanitize_web_text


@pytest.mark.parametrize(
    ("input_text", "expected_output"),
    [
        (
            "Hello \x1b[31;1mRed Alert\x1b[0m World",
            "Hello Red Alert World",
        ),
        (
            "Normal text with \x00 null byte and \x07 bell \x1f unit separator",
            "Normal text with  null byte and  bell  unit separator",
        ),
        (
            "Fake file name \u202Etxt.exe\u202C with bidi override",
            "Fake file name txt.exe with bidi override",
        ),
        (
            "BiDi isolates \u2066left-to-right\u2069 and \u2067right-to-left\u2068",
            "BiDi isolates left-to-right and right-to-left",
        ),
        (
            "<script>alert('xss')</script>",
            "<script>alert('xss')</script>",  # Not transformed by string sanitizer; client renders via textContent
        ),
        (
            "Line 1\nLine 2\r\nLine 3\tTabbed",
            "Line 1\nLine 2\r\nLine 3\tTabbed",  # Newlines and tabs preserved
        ),
    ],
)
def test_sanitize_web_text(input_text: str, expected_output: str):
    """Ensure sanitize_web_text thoroughly strips ANSI, control codes, and bidi overrides."""
    result = sanitize_web_text(input_text)
    assert result == expected_output
