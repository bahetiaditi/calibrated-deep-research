"""Versioned prompt templates.

Prompts live in .txt files rather than Python string literals for three
reasons. A prompt change shows up as a readable diff rather than an escaped
blob. The writeup can quote the exact text that produced the reported
numbers. And a prompt-hash cache key (C3) changes when the file changes,
which is correct — a different prompt is a different experiment.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).parent


class PromptNotFound(FileNotFoundError):
    pass


@lru_cache(maxsize=64)
def load(name: str) -> str:
    path = PROMPT_DIR / f"{name}.txt"
    if not path.is_file():
        available = sorted(p.stem for p in PROMPT_DIR.glob("*.txt"))
        raise PromptNotFound(
            f"no prompt named {name!r}; available: {available}"
        )
    return path.read_text(encoding="utf-8").strip()


def render(name: str, **fields: object) -> str:
    """Load a prompt and substitute {placeholders}.

    Uses str.format, so a literal brace in a prompt must be doubled. JSON
    examples in prompts therefore need {{ }} — easy to get wrong, so the
    tests render every shipped prompt.
    """
    return load(name).format(**fields)