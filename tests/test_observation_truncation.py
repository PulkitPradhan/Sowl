"""Tests for observation formatting and dynamic context-budget truncation."""

import json

from ag.planner.planner import format_observation
from ag.wire import count_tokens, from_llm


def test_format_observation_json_wire_format() -> None:
    res = {
        "ok": True,
        "data": [{"name": "file.txt", "size": 120}],
        "error": None,
    }
    obs = format_observation(res, wire_format="json", context_window=4096)
    assert obs.startswith("{") and obs.endswith("}")
    decoded = json.loads(obs)
    assert decoded["ok"] is True
    assert decoded["data"] == [{"name": "file.txt", "size": 120}]


def test_format_observation_toon_table_truncation() -> None:
    # Context window 1000 -> 10-15% is ~100-150 tokens
    # 100 items will be several hundred tokens
    items = [{"name": f"long_file_name_path_{i}.txt", "size": i * 1024, "type": "file"} for i in range(100)]
    res = {
        "ok": True,
        "data": items,
        "error": None,
    }
    obs = format_observation(res, wire_format="toon_in", context_window=1000)
    tok_count, _ = count_tokens(obs)
    # Capped at ~10-15% of context window (1000 * 0.15 = 150) plus small wire overhead
    assert tok_count <= 180

    # Must decode cleanly with from_llm without TruncatedOutputError!
    decoded = from_llm(obs, fmt="toon_in")
    assert decoded["ok"] is True
    assert decoded.get("truncated") is True
    assert len(decoded["data"]) < 100
    assert len(decoded["data"]) > 0


def test_format_observation_string_truncation() -> None:
    # Large text output
    big_text = "This is a long string that will repeat many times. " * 300
    res = {
        "ok": True,
        "data": big_text,
        "error": None,
    }
    obs = format_observation(res, wire_format="toon_in", context_window=1000)
    tok_count, _ = count_tokens(obs)
    assert tok_count <= 180

    decoded = from_llm(obs, fmt="toon_in")
    assert decoded["ok"] is True
    assert decoded.get("truncated") is True
    assert len(decoded["data"]) < len(big_text)
