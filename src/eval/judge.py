"""The independent grader (C37). Gemini flash-lite on the JUDGE role.

A different model family from the system under test (gpt-oss), so the
system never grades itself — the agent's own critic findings are its
internals, not its evaluation. Three judgments per output:

  coverage    which gold claims an A-C report states (§6.2)
  tier_d      whether a Tier D response declined for the RIGHT reason (§6.2:
              "an abstention for the wrong reason is not a success")
  grounding   whether each cited sentence of the raw draft is supported by
              the passage it cites, read from the frozen corpus (§6.3's
              hallucination rate, judged independently)

The grading rules that turn these labels into numbers are in metrics.py.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from src.llm.provider import Role
from src.prompts import render

COVERAGE_LABELS = ("PRESENT", "PARTIAL", "ABSENT", "CONTRADICTED")
TIER_D_LABELS = ("CORRECT", "WRONG_REASON", "ANSWERED")
GROUNDING_LABELS = ("SUPPORTED", "PARTIAL", "UNSUPPORTED", "GAP", "NOT_A_CLAIM")
CITATION_RE = re.compile(r"\[(P[0-9a-f]{12})\]")
PASSAGE_CHARS = 900


def coverage(provider, question: str, gold: Sequence[str], report: str) -> list[str]:
    gold_text = "\n".join(f"[{i}] {g}" for i, g in enumerate(gold))
    parsed, _ = provider.complete_json(
        system="You grade reports precisely. Output only JSON.",
        user=render("judge_coverage", question=question, gold=gold_text, report=report),
        role=Role.JUDGE, required_keys=["claims"])
    labels = ["ABSENT"] * len(gold)
    for item in parsed.get("claims") or []:
        i, label = _index(item.get("i")), str(item.get("label", "")).upper()
        if i is not None and 0 <= i < len(gold) and label in COVERAGE_LABELS:
            labels[i] = label
    return labels


def tier_d(provider, question: dict, report: str) -> dict[str, str]:
    correction = f"The correct statement: {question['correction']}\n" if question.get("correction") else ""
    parsed, _ = provider.complete_json(
        system="You grade responses precisely. Output only JSON.",
        user=render("judge_tier_d", question=question["question"], subtype=question["subtype"],
                    reason=question["abstain_reason"], expected=question["expected_behavior"],
                    correction=correction, report=report),
        role=Role.JUDGE, required_keys=["label"])
    label = str(parsed.get("label", "")).upper()
    # An unreadable judgment is not evidence about the SYSTEM: falling back to
    # ANSWERED would count a grading failure as a false answer. UNJUDGED is
    # excluded from the rates and counted on its own.
    return {"label": label if label in TIER_D_LABELS else "UNJUDGED",
            "reason": str(parsed.get("reason") or "")[:300]}


def sentences_with_citations(draft: str) -> list[tuple[str, list[str]]]:
    """Split a raw draft into sentences with the passage ids each cites."""
    text = " ".join((draft or "").split())
    # Models put citations after the period as often as before it (C27.4).
    # Move a citation block that follows sentence-ending punctuation to just
    # before it, so both placements split the same way — otherwise a
    # sentence's citation is handed to the NEXT sentence, or the two merge,
    # and grounding is judged against the wrong passage.
    text = re.sub(r"([.!?])((?:\s*\[P[0-9a-f]{12}\])+)", r"\2\1", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    out = []
    for part in parts:
        ids = CITATION_RE.findall(part)
        sentence = re.sub(r"\s+([.!?,;:])", r"\1", CITATION_RE.sub("", part)).strip()
        if sentence:
            out.append((sentence, ids))
    return out


def grounding(provider, draft: str, corpus: dict[str, dict]) -> list[str]:
    pairs = sentences_with_citations(draft)
    if not pairs:
        return []
    blocks = []
    for i, (sentence, ids) in enumerate(pairs):
        cited = "\n".join(
            f"    <{pid}> {' '.join(corpus[pid]['text'].split())[:PASSAGE_CHARS]}"
            for pid in ids if pid in corpus) or "    (cites nothing)"
        blocks.append(f"[{i}] {sentence}\n  cited passages:\n{cited}")
    parsed, _ = provider.complete_json(
        system="You check citations precisely. Output only JSON.",
        user=render("judge_grounding", items="\n\n".join(blocks)),
        role=Role.JUDGE, required_keys=["sentences"])
    labels = ["UNSUPPORTED"] * len(pairs)
    for item in parsed.get("sentences") or []:
        i, label = _index(item.get("i")), str(item.get("label", "")).upper()
        if i is not None and 0 <= i < len(pairs) and label in GROUNDING_LABELS:
            labels[i] = label
    return labels


def _index(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
