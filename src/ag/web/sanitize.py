"""Text sanitization for Web UI: ANSI, control chars, and BiDi override stripping."""

from __future__ import annotations

import re

ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
BIDI_OVERRIDE_PATTERN = re.compile(r"[\u202A-\u202E\u2066-\u2069]")


def sanitize_web_text(text: str) -> str:
    """Sanitize text destined for the web interface.

    Strips ANSI escape sequences, non-printable control characters (excluding tab/newline),
    and Unicode BiDi direction override characters that can disguise filenames or text.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Strip ANSI escape sequences
    s = ANSI_ESCAPE_PATTERN.sub("", text)

    # 2. Strip non-printable control codes (preserves \t, \n, \r)
    s = CONTROL_CHARS_PATTERN.sub("", s)

    # 3. Strip BiDi overrides/isolates
    s = BIDI_OVERRIDE_PATTERN.sub("", s)

    return s
