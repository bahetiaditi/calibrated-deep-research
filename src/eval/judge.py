"""The independent grader (C37). Gemini flash-lite on the JUDGE role.

A different model family from the system under test (gpt-oss), so the
system never grades itself — the agent's own critic findings are its
internals, not its evaluation. Three judgments per output:

  coverage    which gold claims an A-C report states (§6.2)
  tier_d      whether a Tier D response declined for the RIGHT reason (§6.2:
              "an abstention for the wrong reason is not a success")
  grounding   whether each sentence of the raw draft is supported by the
              passages it cites, read in full from the frozen corpus (§6's
              hallucination rate, judged independently)

The grading rules that turn these labels into numbers are in metrics.py.

C37.2 corrections (REFERENCE, C37.2 entry; made before any test result):
  - A label the judge does not return is UNJUDGED, never a verdict. C37
    defaulted missing grounding labels to UNSUPPORTED and missing coverage
    labels to ABSENT — a grading failure counted as the system's. One
    re-ask for the missing items first.
  - The judge reads whole passages. C37 cut each at 900 characters
    mid-sentence (C30.1's defect class: a cut can remove or invert the
    supporting sentence). Only if a batch would not fit the judge's prompt
    budget are passages trimmed, at sentence boundaries, with a marker, and
    the trim is counted.
  - Long drafts are judged in batches, so no single answer has to label more
    sentences than fits comfortably in the output cap.
  - Grounding returns per-sentence items (sentence, cited ids, label), so a
    judgment can be audited and the hallucination rate decomposed into
    uncited claims and unsupported citations.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from src.llm.provider import Role, estimate_tokens
from src.nodes.premise_checker import _trim_at_sentence as trim_at_sentence
from src.prompts import render

# Stamped into every judged file. score.py re-judges a cached judgment made
# under an older version instead of silently mixing two graders' labels.
JUDGE_VERSION = "C37.2"

COVERAGE_LABELS = ("PRESENT", "PARTIAL", "ABSENT", "CONTRADICTED")
TIER_D_LABELS = ("CORRECT", "WRONG_REASON", "ANSWERED")
GROUNDING_LABELS = ("SUPPORTED", "PARTIAL", "UNSUPPORTED", "GAP", "NOT_A_CLAIM")
UNJUDGED = "UNJUDGED"
CITATION_RE = re.compile(r"\[(P[0-9a-f]{12})\]")
# Sentences per grounding call. Labels cost ~20 output tokens each, so 15 is
# ~300 of the 1,024-token cap: room for the model's JSON habits without ever
# approaching truncation. Nearly every draft is one call.
GROUNDING_BATCH = 15
# Floor when fitting passages to the budget: below this a passage is too
# short to judge against, and the call is made anyway (and will be counted).
MIN_PASSAGE_CHARS = 400
JUDGE_OUTPUT_TOKENS = 1024


# -- coverage -------------------------------------------------------------------


def coverage(provider, question: str, gold: Sequence[str], report: str) -> list[str]:
    def ask(indices: Sequence[int]) -> dict[int, str]:
        gold_text = "\n".join(f"[{i}] {gold[i]}" for i in indices)
        parsed, _ = provider.complete_json(
            system="You grade reports precisely. Output only JSON.",
            user=render("judge_coverage", question=question, gold=gold_text, report=report),
            role=Role.JUDGE, required_keys=["claims"])
        return _labels(parsed.get("claims"), set(indices), COVERAGE_LABELS)

    return _with_reask(len(gold), ask)


# -- tier D -----------------------------------------------------------------------


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
    return {"label": label if label in TIER_D_LABELS else UNJUDGED,
            "reason": str(parsed.get("reason") or "")[:300]}


# -- grounding ----------------------------------------------------------------------


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


def grounding(provider, draft: str, corpus: dict[str, dict],
              batch_size: int = GROUNDING_BATCH) -> dict[str, Any]:
    """Per-sentence grounding judgments for a raw draft.

    Returns {"items": [{"i", "sentence", "cites", "label"}], "trimmed": n}.
    `cites` holds only ids that resolve in the corpus — an id that does not
    is not evidence the judge can read, so the sentence is judged as citing
    nothing, which is what a reader holding the report would face too.
    """
    pairs = sentences_with_citations(draft)
    items = [{"i": i, "sentence": s, "cites": [p for p in ids if p in corpus], "label": None}
             for i, (s, ids) in enumerate(pairs)]
    budget = _budget(provider)
    trimmed = 0
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]

        def ask(indices: Sequence[int]) -> dict[int, str]:
            nonlocal trimmed
            chosen = [it for it in batch if it["i"] in set(indices)]
            prompt, n_trimmed = _fit_grounding_prompt(chosen, corpus, budget)
            trimmed += n_trimmed
            parsed, _ = provider.complete_json(
                system="You check citations precisely. Output only JSON.",
                user=prompt, role=Role.JUDGE, required_keys=["sentences"])
            return _labels(parsed.get("sentences"), set(indices), GROUNDING_LABELS)

        labels = _with_reask_indices([it["i"] for it in batch], ask)
        for it in batch:
            it["label"] = labels[it["i"]]
    return {"items": items, "trimmed": trimmed}


def _render_items(items: Sequence[dict], corpus: dict, cap: int | None) -> tuple[str, int]:
    blocks, trimmed = [], 0
    for it in items:
        cited = []
        for pid in it["cites"]:
            text = " ".join(corpus[pid]["text"].split())
            if cap is not None and len(text) > cap:
                text = trim_at_sentence(text, cap)
                trimmed += 1
            cited.append(f"    <{pid}> {text}")
        body = "\n".join(cited) or "    (cites nothing)"
        blocks.append(f"[{it['i']}] {it['sentence']}\n  cited passages:\n{body}")
    return "\n\n".join(blocks), trimmed


def _fit_grounding_prompt(items, corpus, budget: int | None) -> tuple[str, int]:
    """Whole passages unless the batch would not fit; then the largest
    per-passage cap that fits, trimmed at sentence ends."""
    system = "You check citations precisely. Output only JSON."
    text, _ = _render_items(items, corpus, None)
    prompt = render("judge_grounding", items=text)
    if budget is None or estimate_tokens(system, prompt, 0) <= budget:
        return prompt, 0
    longest = max((len(" ".join(corpus[p]["text"].split())) for it in items for p in it["cites"]),
                  default=0)
    cap, n = longest, 0
    while cap > MIN_PASSAGE_CHARS:
        cap = max(MIN_PASSAGE_CHARS, int(cap * 0.85))
        text, n = _render_items(items, corpus, cap)
        prompt = render("judge_grounding", items=text)
        if estimate_tokens(system, prompt, 0) <= budget:
            return prompt, n
    return prompt, n


def _budget(provider) -> int | None:
    fn = getattr(provider, "prompt_token_budget", None)
    try:
        return fn(Role.JUDGE) if fn else None
    except Exception:  # a fake provider in tests, or a chain without TPM
        return None


# -- shared -----------------------------------------------------------------------


def _labels(raw: Any, wanted: set[int], allowed: Sequence[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        i, label = _index(item.get("i")), str(item.get("label", "")).upper()
        if i in wanted and label in allowed and i not in out:
            out[i] = label
    return out


def _with_reask(n: int, ask) -> list[str]:
    labels = _with_reask_indices(list(range(n)), ask)
    return [labels[i] for i in range(n)]


def _with_reask_indices(indices: list[int], ask) -> dict[int, str]:
    """Ask once; re-ask ONCE for whatever came back unlabelled; the rest is
    UNJUDGED. The re-ask carries only the missing items, so it is a
    different prompt (a cached identical prompt would return the same gap)."""
    if not indices:
        return {}
    got = ask(indices)
    missing = [i for i in indices if i not in got]
    if missing:
        got.update(ask(missing))
    return {i: got.get(i, UNJUDGED) for i in indices}


def _index(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
