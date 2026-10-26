"""Grounding auditor: Detects invented (hallucinated), tainted, and grounded identifiers."""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ag.skills.manifest import ToolManifest


@dataclass
class AuditResult(list):
    """Result of grounding audit.

    Inherits from list (containing invented/hallucinated leaks) for backwards compatibility,
    while exposing separate invented, tainted, and grounded sets.
    """

    invented: list[str] = field(default_factory=list)
    tainted: list[str] = field(default_factory=list)
    grounded: list[str] = field(default_factory=list)

    def __init__(
        self,
        invented: list[str] | None = None,
        tainted: list[str] | None = None,
        grounded: list[str] | None = None,
    ) -> None:
        inv = invented or []
        super().__init__(inv)
        self.invented = inv
        self.tainted = tainted or []
        self.grounded = grounded or []


def normalize_identifier(val: str, workspace_root: Path | None = None) -> str:
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
    if s not in (".", "./", "/"):
        s = s.rstrip(".,;:!?")

    if workspace_root:
        try:
            p = Path(s)
            ws = Path(workspace_root).resolve()
            if p.is_absolute() and p.resolve().is_relative_to(ws):
                return str(p.resolve().relative_to(ws)).lower()
        except Exception:  # noqa: BLE001, S110
            pass

    return s


def audit_grounding(
    prompt: str,
    history: list[tuple[Any, str]],
    tool: ToolManifest,
    args: dict[str, Any],
    workspace_root: Path | None = None,
) -> AuditResult:
    """Audit tool arguments for ungrounded identifiers at proposal time.

    Audits only argument fields defined with kind="identifier" in the tool manifest.
    Text fields (kind="text") and non-identifiers are ignored to avoid false positives.
    Categorizes identifiers into:
      - grounded: present in user prompt or trusted whitelist
      - tainted: present in prior tool observations
      - invented: not found anywhere (hallucination leak)
    """
    invented: list[str] = []
    tainted: list[str] = []
    grounded: list[str] = []

    user_text = prompt.lower()
    obs_texts = [str(obs).lower() for _action, obs in history]
    whitelist = {".", "", "/", "./"}

    if workspace_root:
        whitelist.add(str(workspace_root).lower())

    for arg_name, arg_val in args.items():
        field_def = tool.args.get(arg_name)
        if not field_def or getattr(field_def, "kind", "other") != "identifier":
            continue

        if not isinstance(arg_val, str):
            continue

        norm = normalize_identifier(arg_val, workspace_root)
        if norm in whitelist:
            grounded.append(arg_val)
            continue

        base_name = ""
        try:
            base_name = Path(norm).name.lower()
        except Exception:  # noqa: BLE001, S110
            pass

        # Check if in user prompt
        if norm in user_text or (base_name and base_name in user_text):
            grounded.append(arg_val)
            continue

        # Check if in prior tool observations
        found_in_obs = False
        for o in obs_texts:
            if norm in o or (base_name and base_name in o):
                tainted.append(arg_val)
                found_in_obs = True
                break

        if not found_in_obs:
            invented.append(arg_val)

    return AuditResult(invented=invented, tainted=tainted, grounded=grounded)
