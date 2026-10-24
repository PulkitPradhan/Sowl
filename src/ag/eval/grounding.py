"""Grounding auditor: Pure function detecting ungrounded identifier hallucination leaks."""

from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Any

from ag.skills.manifest import ToolManifest


def normalize_identifier(val: str) -> str:
    """Normalize identifier: strip mailto:, URL decode, lowercase, and strip punctuation."""
    if not isinstance(val, str):
        return ""
    s = val.strip()
    if s.lower().startswith("mailto:"):
        s = s[7:]
    try:
        s = urllib.parse.unquote(s)
    except Exception:  # noqa: BLE001, S110
        pass
    s = s.lower().strip("\"'")
    if s != "." and s != "./":
        s = s.rstrip(".,;:!?")
    return s


def audit_grounding(
    prompt: str,
    history: list[tuple[Any, str]],
    tool: ToolManifest,
    args: dict[str, Any],
) -> list[str]:
    """Audit tool arguments for ungrounded identifiers at proposal time.

    Audits only argument fields defined with kind="identifier" in the tool manifest.
    Text fields (kind="text") and non-identifiers are ignored to avoid false positives.
    Whitelists '.', empty string, and '/'.
    Returns list of ungrounded raw argument values.
    """
    leaks: list[str] = []

    # Build corpus of all trusted text: prompt + all prior observation contents
    corpus_parts = [prompt.lower()]
    for _action, obs in history:
        corpus_parts.append(str(obs).lower())
    text_corpus = " ".join(corpus_parts)

    whitelist = {".", "", "/", "./"}

    for arg_name, arg_val in args.items():
        field_def = tool.args.get(arg_name)
        if not field_def or getattr(field_def, "kind", "other") != "identifier":
            continue

        if not isinstance(arg_val, str):
            continue

        norm = normalize_identifier(arg_val)
        if norm in whitelist:
            continue

        # Check if normalized identifier or its basename is present in text_corpus
        if norm in text_corpus:
            continue

        # For file paths, check if file basename is present
        try:
            base_name = Path(norm).name.lower()
            if base_name and base_name in text_corpus:
                continue
        except Exception:  # noqa: BLE001, S110
            pass

        leaks.append(arg_val)

    return leaks
