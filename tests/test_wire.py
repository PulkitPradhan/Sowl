"""Tests for wire format (TOON and JSON), fence stripping, truncation, and CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from ag.client.cli import app
from ag.wire.wire import (
    TruncatedOutputError,
    WireError,
    count_tokens,
    from_llm,
    to_llm,
)

runner = CliRunner()
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# 1. Hypothesis round-trip test on 200 random structures
json_primitives = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-100000, max_value=100000)
    | st.floats(allow_nan=False, allow_infinity=False, min_value=-100000, max_value=100000).map(
        lambda f: round(f, 4)
    )
    | st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=25)
)

json_values = st.recursive(
    json_primitives,
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(
        st.text(min_size=1, max_size=8, alphabet="abcdefghijklmnopqrstuvwxyz_"),
        children,
        max_size=4,
    ),
    max_leaves=12,
)


@settings(max_examples=200, deadline=None)
@given(val=json_values)
def test_hypothesis_roundtrip_200_random_structures(val):
    # Round-trip through TOON
    encoded_toon = to_llm(val, fmt="toon_both")
    decoded_toon = from_llm(encoded_toon, fmt="toon_both")
    assert decoded_toon == val

    # Round-trip through JSON
    encoded_json = to_llm(val, fmt="json")
    decoded_json = from_llm(encoded_json, fmt="json")
    assert decoded_json == val


# 2. Tabular uniform array encoding
def test_uniform_array_tabular_format():
    data = {
        "users": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
    }
    encoded = to_llm(data, fmt="toon_in")
    assert "users[2]{name,age}:" in encoded
    assert "Alice,30" in encoded
    assert "Bob,25" in encoded
    assert from_llm(encoded, fmt="toon_both") == data


def test_quote_values_with_commas_colons_newlines():
    data = {
        "items": [
            {"desc": "hello, world", "note": "key: value", "multiline": "line1\nline2"}
        ]
    }
    encoded = to_llm(data, fmt="toon_in")
    assert '"hello, world"' in encoded
    assert '"key: value"' in encoded
    assert from_llm(encoded, fmt="toon_both") == data


# 3. Explicit edge-case tests
@pytest.mark.parametrize(
    "val",
    [
        "true",
        "null",
        "123",
        "007",
        "",
        " padded ",
        "hello, world",
        "key: value",
        "first\nsecond",
        {"key with spaces": "val"},
        [],
        {},
        "unicode: 🚀 ñ 世界",
        None,
    ],
)
def test_edge_case_lookalikes_roundtrip(val):
    encoded = to_llm(val, fmt="toon_both")
    decoded = from_llm(encoded, fmt="toon_both")
    assert decoded == val
    assert type(decoded) is type(val), f"Type mismatch for {val!r}: got {type(decoded)}"


# 4. Code fence and surrounding prose stripping
def test_from_llm_markdown_fence_stripping_json():
    raw = """Here is the resulting data:
```json
{
  "status": "success",
  "code": 200
}
```
Let me know if you need anything else!"""
    decoded = from_llm(raw, fmt="toon_in")
    assert decoded == {"status": "success", "code": 200}


def test_from_llm_markdown_fence_stripping_toon():
    raw = """Check out this table:
```toon
users[2]{id,name}:
  1,Alice
  2,Bob
```
End of output."""
    decoded = from_llm(raw, fmt="toon_both")
    assert decoded == {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}


def test_from_llm_generic_fence():
    raw = "```\n[1, 2, 3]\n```"
    decoded = from_llm(raw, fmt="json")
    assert decoded == [1, 2, 3]


# 5. Row count mismatch [N] and finish_reason='length' truncation
def test_row_count_mismatch_raises_truncated_output_error():
    corrupted = "users[3]{id,name}:\n  1,Alice\n  2,Bob\n"
    with pytest.raises(TruncatedOutputError) as exc_info:
        from_llm(corrupted, fmt="toon_both")
    assert "length" in str(exc_info.value).lower() or "declared" in str(exc_info.value).lower()


def test_finish_reason_length_raises_truncated_output_error():
    content = '{"key": "incomplete...'
    with pytest.raises(TruncatedOutputError):
        from_llm(content, fmt="json", finish_reason="length")


# 6. Input size cap (1 MB)
def test_input_size_cap_exceeded_raises_wire_error():
    huge_input = "a" * (1_000_000 + 10)
    with pytest.raises(WireError) as exc_info:
        from_llm(huge_input, fmt="toon_in")
    assert "1 MB" in str(exc_info.value)


# 7. Library wrapping: only WireError or TruncatedOutputError raised
def test_invalid_syntax_raises_wire_error():
    invalid_text = "users[2]{a,b:\n  invalid"
    with pytest.raises(WireError) as exc_info:
        from_llm(invalid_text, fmt="toon_both")
    assert isinstance(exc_info.value, WireError)


# 8. Token counting
def test_count_tokens():
    text = "Hello world from Sowl personal AI agent!"
    count, label = count_tokens(text)
    assert count > 0
    assert "cl100k" in label


# 9. CLI command `ag wire --count`
def test_cli_wire_count_single_file():
    target = FIXTURES_DIR / "tools.json"
    result = runner.invoke(app, ["wire", "--count", str(target)])
    assert result.exit_code == 0
    assert "tools.json" in result.stdout
    assert "Savings vs Compact JSON:" in result.stdout
    assert "approx (cl100k)" in result.stdout


def test_cli_wire_count_all_fixtures():
    # Calling `ag wire --count` without arguments reports all three fixtures
    result = runner.invoke(app, ["wire", "--count"])
    assert result.exit_code == 0
    assert "tools.json" in result.stdout
    assert "snapshot.json" in result.stdout
    assert "plan.json" in result.stdout
    assert "Headline Savings" in result.stdout
