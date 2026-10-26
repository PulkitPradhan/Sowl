"""Layer B decision model: LLM safety judge evaluating structured features."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ag.laya.provenance import (
    EMAIL_REGEX,
    FILENAME_REGEX,
    URL_REGEX,
    ProvenanceSource,
    ProvenanceStore,
)
from ag.models.provider import ModelProvider


class LayaJudgeVerdict(BaseModel):
    """Schema-constrained response from the Laya safety decision model."""

    model_config = ConfigDict(extra="ignore")

    verdict: Literal["proceed", "confirm", "block"] = "proceed"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: str = Field(default="", description="Brief rationale, at most 20 words")
    guidance: str = Field(default="", description="Safety guidance for the planner")


JUDGE_SYSTEM_PROMPT = """You are Laya, an independent AI safety decision model.
Evaluate whether the proposed tool call is safe to proceed, requires confirmation, or must be blocked.
You receive structured telemetry ONLY: tool name, declared risk, argument provenance, and safety flags.
Reply with a single JSON object strictly conforming to:
{
  "verdict": "proceed" | "confirm" | "block",
  "confidence": 0.0 to 1.0,
  "reason": "short explanation in 20 words or fewer",
  "guidance": "optional guidance for the agent"
}
"""


class LayaJudge:
    """Invokes and validates the Layer B safety decision model."""

    def __init__(self, provider: ModelProvider | None = None) -> None:
        self.provider = provider

    async def adjudicate(
        self,
        layer_a_decision: Any,
        features: dict[str, Any],
        provenance: ProvenanceStore,
    ) -> Any:
        """Call Layer B judge and apply monotonic strictness enforcement."""
        if not self.provider:
            return layer_a_decision

        # Format structured input message (no raw file or page content)
        clean_features = {
            "tool": features.get("tool"),
            "risk": features.get("risk"),
            "args": features.get("args"),
            "layer_a_verdict": layer_a_decision.action,
            "layer_a_reasons": layer_a_decision.reasons,
            "provenance": [
                {"field": p.get("field"), "source": p.get("source"), "origin": p.get("origin")}
                for p in layer_a_decision.provenance
            ],
            "is_tainted": layer_a_decision.is_tainted,
        }

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(clean_features, indent=2)},
        ]

        try:
            resp = await self.provider.chat(
                messages,
                temperature=0.0,
                structured=True,
                response_format=LayaJudgeVerdict,
            )
            raw_text = getattr(resp, "content", getattr(resp, "text", "")).strip()
            # Parse JSON safely
            data = json.loads(raw_text)
            verdict_obj = LayaJudgeVerdict.model_validate(data)
        except Exception:  # noqa: BLE001
            # Any model error or invalid output is ignored; Layer A holds
            return layer_a_decision

        # Sanitize guidance: strip any identifier that is not in the provenance store
        sanitized_guidance = self._sanitize_guidance(verdict_obj.guidance, provenance)

        # Monotonic strictness: Layer B can ONLY make decisions stricter
        # Hierarchy: "allow"/"proceed" (0) < "confirm" (1) < "block" (2)
        rank_map = {
            "allow": 0,
            "proceed": 0,
            "confirm": 1,
            "block": 2,
        }

        a_rank = rank_map.get(layer_a_decision.action, 0)
        b_rank = rank_map.get(verdict_obj.verdict, 0)

        # If judge is strictly looser, ignore judge verdict and retain Layer A
        if b_rank < a_rank:
            layer_a_decision.layer_b_verdict = verdict_obj.verdict
            if sanitized_guidance:
                layer_a_decision.guidance = sanitized_guidance
            return layer_a_decision

        # Stricter or equal: upgrade action if necessary
        new_action = layer_a_decision.action
        if b_rank > a_rank and verdict_obj.verdict in ("block", "confirm"):
            new_action = verdict_obj.verdict

        new_reasons = list(layer_a_decision.reasons)
        if verdict_obj.reason:
            new_reasons.append(f"Layer B Judge: {verdict_obj.reason}")

        layer_a_decision.action = new_action
        layer_a_decision.reasons = new_reasons
        layer_a_decision.layer_b_verdict = verdict_obj.verdict
        if sanitized_guidance:
            layer_a_decision.guidance = sanitized_guidance

        return layer_a_decision

    def _sanitize_guidance(self, text: str, provenance: ProvenanceStore) -> str:
        """Strip any foreign identifiers from judge guidance text that do not exist in provenance."""
        if not text:
            return ""

        words = text
        for email in EMAIL_REGEX.findall(text):
            if provenance.resolve(email).source == ProvenanceSource.NONE:
                words = words.replace(email, "[REDACTED_IDENTIFIER]")

        for url in URL_REGEX.findall(text):
            if provenance.resolve(url).source == ProvenanceSource.NONE:
                words = words.replace(url, "[REDACTED_URL]")

        for fn in FILENAME_REGEX.findall(text):
            if provenance.resolve(fn).source == ProvenanceSource.NONE:
                words = words.replace(fn, "[REDACTED_PATH]")

        return words
