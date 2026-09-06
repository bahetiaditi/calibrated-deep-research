"""Defensive parsing of structured LLM output.

Every node downstream (planner, critic, decider) depends on getting valid
JSON back. Models emit it wrapped in prose, fenced in markdown, truncated at
the token limit, or with trailing commas. Retrying the API call is the
expensive fix; repairing locally is free, so we exhaust local repair first.

The order of strategies matters — cheapest and most reliable first.
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


class StructuredOutputError(ValueError):
    """Raised when no repair strategy recovers valid JSON."""

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


def extract_json(text: str) -> Any:
    """Parse JSON from a model response, repairing common damage.

    Strategies, in order:
      1. Parse as-is.
      2. Strip markdown fences.
      3. Slice from the first brace/bracket to its matching close.
      4. Remove trailing commas.
      5. Close unterminated strings/brackets (truncation at max_tokens).
    """
    if not text or not text.strip():
        raise StructuredOutputError("empty response", text or "")

    candidates = [text.strip()]

    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    sliced = _slice_to_balanced(text)
    if sliced:
        candidates.append(sliced)

    for candidate in list(candidates):
        stripped = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        if stripped != candidate:
            candidates.append(stripped)

    for candidate in list(candidates):
        repaired = _close_truncated(candidate)
        if repaired and repaired != candidate:
            candidates.append(repaired)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc

    raise StructuredOutputError(
        f"could not parse JSON after {len(candidates)} repair attempts: {last_error}",
        text,
    )


def _slice_to_balanced(text: str) -> str | None:
    """Extract the first complete top-level JSON object or array.

    Brace-counting is string-aware: a `}` inside a quoted value must not
    close the object.
    """
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None

    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]  # unbalanced — _close_truncated may rescue it


def _close_truncated(text: str) -> str | None:
    """Close a JSON fragment cut off mid-generation.

    Truncation at max_output_tokens is common and the partial content is
    usually still useful — a plan with 4 of 5 sub-questions beats a retry.
    """
    if not text:
        return None

    stack: list[str] = []
    in_string = False
    escaped = False

    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()

    if not stack and not in_string:
        return None

    repaired = text.rstrip()
    if in_string:
        repaired += '"'
    # Drop a dangling key or comma before closing, e.g. '{"a": 1, "b"'
    repaired = re.sub(r',\s*"[^"]*"?\s*:?\s*$', "", repaired)
    repaired = re.sub(r",\s*$", "", repaired)
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def require_keys(obj: Any, keys: list[str], *, context: str = "") -> dict:
    """Assert a parsed object is a dict containing `keys`."""
    where = f" ({context})" if context else ""
    if not isinstance(obj, dict):
        raise StructuredOutputError(
            f"expected a JSON object{where}, got {type(obj).__name__}", str(obj)
        )
    missing = [k for k in keys if k not in obj]
    if missing:
        raise StructuredOutputError(
            f"missing required keys{where}: {missing}", json.dumps(obj)[:500]
        )
    return obj