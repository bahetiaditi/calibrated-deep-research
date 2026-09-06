"""Tests for defensive JSON parsing.

Each case is damage actually seen from LLM structured output. Repairing
locally is free; re-prompting costs quota we do not have.
"""
import pytest

from src.llm.structured import (
    StructuredOutputError,
    extract_json,
    require_keys,
)


def test_clean_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_markdown_fenced():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_fenced_without_language():
    assert extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_prose_preamble():
    raw = 'Sure! Here is the plan you asked for:\n{"sub_questions": ["x"]}\nHope that helps.'
    assert extract_json(raw) == {"sub_questions": ["x"]}


def test_trailing_comma():
    assert extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_trailing_comma_in_array():
    assert extract_json('{"a": [1, 2, 3,]}') == {"a": [1, 2, 3]}


def test_top_level_array():
    assert extract_json('[{"id": 1}, {"id": 2}]') == [{"id": 1}, {"id": 2}]


def test_brace_inside_string_does_not_close_object():
    raw = '{"note": "use {curly} braces", "n": 2}'
    assert extract_json(raw) == {"note": "use {curly} braces", "n": 2}


def test_escaped_quote_inside_string():
    raw = '{"q": "he said \\"hi\\"", "n": 1}'
    assert extract_json(raw)["n"] == 1


def test_truncated_object_is_closed():
    """Cut off at max_output_tokens — partial content still beats a retry."""
    raw = '{"sub_questions": [{"id": "sq1", "text": "what is X"}, {"id": "sq2"'
    parsed = extract_json(raw)
    assert parsed["sub_questions"][0]["id"] == "sq1"


def test_truncated_mid_string_is_closed():
    raw = '{"rationale": "the evidence suggests that'
    assert "rationale" in extract_json(raw)


def test_empty_response_raises():
    with pytest.raises(StructuredOutputError):
        extract_json("   ")


def test_unparseable_raises_with_raw_attached():
    with pytest.raises(StructuredOutputError) as exc:
        extract_json("this is not json at all")
    assert exc.value.raw == "this is not json at all"


def test_require_keys_passes():
    require_keys({"a": 1, "b": 2}, ["a"])


def test_require_keys_reports_missing():
    with pytest.raises(StructuredOutputError, match="missing required keys"):
        require_keys({"a": 1}, ["a", "b"])


def test_require_keys_rejects_non_dict():
    with pytest.raises(StructuredOutputError, match="expected a JSON object"):
        require_keys([1, 2], ["a"])