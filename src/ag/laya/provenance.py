"""Provenance store and taint tracking for identifiers (email, URL, path)."""

from __future__ import annotations

import re
import urllib.parse
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class ProvenanceSource(str, Enum):
    USER = "USER"
    TRUSTED = "TRUSTED"
    TOOL_OUTPUT = "TOOL_OUTPUT"
    NONE = "NONE"


class IdentifierProvenance(BaseModel):
    """Provenance record for a single identifier."""

    model_config = ConfigDict(extra="ignore")

    field: str = ""
    value: str = ""
    normalized: str = ""
    source: ProvenanceSource = ProvenanceSource.NONE
    origin: str = ""


# Regexes for extracting identifiers from arbitrary text
EMAIL_REGEX = re.compile(r"""[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+""")
URL_REGEX = re.compile(r"""https?://[^\s"'<>]+""")
FILENAME_REGEX = re.compile(r"""\b[a-zA-Z0-9_.-]+\.[a-zA-Z0-9_]{1,8}\b""")


def normalize_identifier(val: str, workspace_root: Path | None = None) -> str:
    """Normalize identifier: strip mailto:, unquote URL, strip punctuation, lowercase."""
    if not isinstance(val, str):
        return ""
    s = val.strip()
    if s.lower().startswith("mailto:"):
        s = s[7:]
    try:
        s = urllib.parse.unquote(s)
    except Exception:  # noqa: BLE001, S110
        pass
    s = s.strip("\"'").lower()
    if s not in (".", "./", "/"):
        s = s.rstrip(".,;:!?")

    # If it is an absolute path within workspace_root, map it to relative representation
    if workspace_root:
        try:
            p = Path(s)
            ws = Path(workspace_root).resolve()
            if p.is_absolute() and p.resolve().is_relative_to(ws):
                rel = p.resolve().relative_to(ws)
                return str(rel).lower()
        except Exception:  # noqa: BLE001, S110
            pass

    return s


class ProvenanceStore:
    """Tracks provenance and taint for identifiers encountered during agent execution."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = workspace_root.resolve() if workspace_root else Path.cwd()
        self.identifiers: dict[str, IdentifierProvenance] = {}
        self.user_corpus: list[str] = []
        self.observation_corpus: list[tuple[str, str]] = []  # (text, origin)

        # Pre-seed workspace root as TRUSTED
        ws_str = str(self.workspace_root).lower()
        self.identifiers[ws_str] = IdentifierProvenance(
            field="path",
            value=str(self.workspace_root),
            normalized=ws_str,
            source=ProvenanceSource.TRUSTED,
            origin="workspace root",
        )
        self.identifiers["."] = IdentifierProvenance(
            field="path", value=".", normalized=".", source=ProvenanceSource.TRUSTED, origin="workspace root"
        )
        self.identifiers["./"] = IdentifierProvenance(
            field="path", value="./", normalized="./", source=ProvenanceSource.TRUSTED, origin="workspace root"
        )
        self.identifiers[""] = IdentifierProvenance(
            field="path", value="", normalized="", source=ProvenanceSource.TRUSTED, origin="empty"
        )

    def add_user_text(self, text: str) -> None:
        """Register text from the user (prompt or answer to an ask action)."""
        if not text:
            return
        self.user_corpus.append(text.lower())
        for match in EMAIL_REGEX.findall(text):
            norm = normalize_identifier(match, self.workspace_root)
            self.identifiers[norm] = IdentifierProvenance(
                field="identifier",
                value=match,
                normalized=norm,
                source=ProvenanceSource.USER,
                origin="user prompt",
            )
        for match in URL_REGEX.findall(text):
            norm = normalize_identifier(match, self.workspace_root)
            self.identifiers[norm] = IdentifierProvenance(
                field="identifier",
                value=match,
                normalized=norm,
                source=ProvenanceSource.USER,
                origin="user prompt",
            )
        for match in FILENAME_REGEX.findall(text):
            norm = normalize_identifier(match, self.workspace_root)
            self.identifiers[norm] = IdentifierProvenance(
                field="identifier",
                value=match,
                normalized=norm,
                source=ProvenanceSource.USER,
                origin="user prompt",
            )

    def add_trusted_identifier(self, val: str, origin: str = "contacts.toml") -> None:
        """Register an identifier as trusted (e.g. unique contact match)."""
        norm = normalize_identifier(val, self.workspace_root)
        self.identifiers[norm] = IdentifierProvenance(
            field="identifier",
            value=val,
            normalized=norm,
            source=ProvenanceSource.TRUSTED,
            origin=origin,
        )

    def add_skill_text(self, skill_name: str, text: str) -> None:
        """Register identifiers found in active skill instructions/manifest as TRUSTED."""
        if not text:
            return
        origin = f"skill:{skill_name}"
        for match in URL_REGEX.findall(text):
            norm = normalize_identifier(match, self.workspace_root)
            self.identifiers[norm] = IdentifierProvenance(
                field="url",
                value=match,
                normalized=norm,
                source=ProvenanceSource.TRUSTED,
                origin=origin,
            )
        for match in EMAIL_REGEX.findall(text):
            norm = normalize_identifier(match, self.workspace_root)
            self.identifiers[norm] = IdentifierProvenance(
                field="email",
                value=match,
                normalized=norm,
                source=ProvenanceSource.TRUSTED,
                origin=origin,
            )
        for match in FILENAME_REGEX.findall(text):
            norm = normalize_identifier(match, self.workspace_root)
            self.identifiers[norm] = IdentifierProvenance(
                field="path",
                value=match,
                normalized=norm,
                source=ProvenanceSource.TRUSTED,
                origin=origin,
            )

    def add_tool_observation(
        self,
        tool_name: str,
        data: Any,
        origin: str = "",
        is_tainted_input: bool = False,
    ) -> None:
        """Register tool observation data (marks extracted identifiers as TOOL_OUTPUT / tainted)."""
        text = str(data)
        eff_origin = origin or tool_name
        self.observation_corpus.append((text.lower(), eff_origin))

        for match in EMAIL_REGEX.findall(text):
            norm = normalize_identifier(match, self.workspace_root)
            if norm not in self.identifiers or self.identifiers[norm].source == ProvenanceSource.NONE:
                self.identifiers[norm] = IdentifierProvenance(
                    field="identifier",
                    value=match,
                    normalized=norm,
                    source=ProvenanceSource.TOOL_OUTPUT,
                    origin=f"{eff_origin}",
                )

        for match in URL_REGEX.findall(text):
            norm = normalize_identifier(match, self.workspace_root)
            if norm not in self.identifiers or self.identifiers[norm].source == ProvenanceSource.NONE:
                self.identifiers[norm] = IdentifierProvenance(
                    field="identifier",
                    value=match,
                    normalized=norm,
                    source=ProvenanceSource.TOOL_OUTPUT,
                    origin=f"{eff_origin}",
                )

        for match in FILENAME_REGEX.findall(text):
            norm = normalize_identifier(match, self.workspace_root)
            if norm not in self.identifiers or self.identifiers[norm].source == ProvenanceSource.NONE:
                self.identifiers[norm] = IdentifierProvenance(
                    field="identifier",
                    value=match,
                    normalized=norm,
                    source=ProvenanceSource.TOOL_OUTPUT,
                    origin=f"{eff_origin}",
                )

    def resolve(self, val: str, field_name: str = "") -> IdentifierProvenance:
        """Resolve provenance of an argument identifier."""
        if not isinstance(val, str):
            return IdentifierProvenance(
                field=field_name,
                value=str(val),
                normalized=str(val),
                source=ProvenanceSource.NONE,
                origin="",
            )

        norm = normalize_identifier(val, self.workspace_root)

        # Check exact match in store
        if norm in self.identifiers:
            rec = self.identifiers[norm]
            return IdentifierProvenance(
                field=field_name or rec.field,
                value=val,
                normalized=norm,
                source=rec.source,
                origin=rec.origin,
            )

        # Check if basename of file matches a registered user/trusted identifier
        try:
            base = Path(norm).name.lower()
            if base in self.identifiers:
                rec = self.identifiers[base]
                return IdentifierProvenance(
                    field=field_name,
                    value=val,
                    normalized=norm,
                    source=rec.source,
                    origin=rec.origin,
                )
        except Exception:  # noqa: BLE001, S110
            pass

        # Check presence in user corpus
        for u_text in self.user_corpus:
            if norm in u_text or (base and base in u_text):
                return IdentifierProvenance(
                    field=field_name,
                    value=val,
                    normalized=norm,
                    source=ProvenanceSource.USER,
                    origin="user text",
                )

        # Check if URL service/domain name appears in user prompt
        if norm.startswith(("http://", "https://")):
            try:
                parsed = urllib.parse.urlparse(norm)
                hostname = parsed.hostname or ""
                parts = hostname.split(".")
                for part in parts:
                    if len(part) >= 3 and part not in ("www", "com", "org", "net", "edu", "gov", "io", "co", "app"):
                        for u_text in self.user_corpus:
                            if re.search(r"\b" + re.escape(part) + r"\b", u_text):
                                return IdentifierProvenance(
                                    field=field_name,
                                    value=val,
                                    normalized=norm,
                                    source=ProvenanceSource.USER,
                                    origin=f"user prompt ({part})",
                                )
            except Exception:  # noqa: BLE001, S110
                pass

        # Check presence in tool observations
        for obs_text, origin in self.observation_corpus:
            if norm in obs_text or (base and base in obs_text):
                return IdentifierProvenance(
                    field=field_name,
                    value=val,
                    normalized=norm,
                    source=ProvenanceSource.TOOL_OUTPUT,
                    origin=origin,
                )

        # Whitelist fallback for empty or current directory
        if norm in (".", "./", "", "/"):
            return IdentifierProvenance(
                field=field_name,
                value=val,
                normalized=norm,
                source=ProvenanceSource.TRUSTED,
                origin="default",
            )

        return IdentifierProvenance(
            field=field_name,
            value=val,
            normalized=norm,
            source=ProvenanceSource.NONE,
            origin="",
        )
