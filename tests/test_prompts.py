"""Tests for the prompt loader and every shipped prompt.

Prompts contain JSON examples, so a literal brace must be doubled or
str.format raises — at call time, mid-run, after the retrieval has already
been paid for. These tests discover each prompt's placeholders by
introspection rather than by a hand-maintained list, so adding a prompt
cannot silently skip the check.
"""
import string

import pytest

from src.prompts import PROMPT_DIR, PromptNotFound, load, render

PROMPTS = sorted(p.stem for p in PROMPT_DIR.glob("*.txt"))


def placeholders(name: str) -> set[str]:
    return {
        field
        for _, field, _, _ in string.Formatter().parse(load(name))
        if field
    }


def test_prompts_exist():
    assert PROMPTS, "no prompt templates found"


@pytest.mark.parametrize("name", PROMPTS)
def test_prompt_is_not_empty(name):
    assert len(load(name)) > 100, f"{name} looks like a stub"


@pytest.mark.parametrize("name", PROMPTS)
def test_prompt_renders_with_its_own_placeholders(name):
    """Catches an undoubled literal brace — the failure mode that otherwise
    surfaces mid-run, after retrieval has been paid for."""
    rendered = render(name, **{field: f"<{field}>" for field in placeholders(name)})
    assert rendered
    for field in placeholders(name):
        assert f"<{field}>" in rendered, f"{name} dropped {field}"


@pytest.mark.parametrize("name", PROMPTS)
def test_json_example_survives_rendering(name):
    """A prompt that asks for JSON must still show well-formed JSON after
    formatting; doubled braces collapse to single ones here."""
    text = load(name)
    if '{{"' not in text:
        pytest.skip("prompt declares no JSON shape")
    rendered = render(name, **{f: f"<{f}>" for f in placeholders(name)})
    assert '{"' in rendered
    assert '{{' not in rendered


def test_missing_prompt_lists_what_is_available():
    with pytest.raises(PromptNotFound, match="available"):
        load("does_not_exist")


def test_load_is_cached():
    assert load(PROMPTS[0]) is load(PROMPTS[0])