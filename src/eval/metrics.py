"""Grading rules and metrics (C37). Pure functions; no LLM calls.

Every number in the writeup is computed here, from judged records. The rules
(REFERENCE, C37 entry) in one place:

  A-C correct      weighted coverage of gold claims >= MIN_COVERAGE (PARTIAL
                   counts half) AND no gold claim contradicted
  answered         ANSWER or PARTIAL; ABSTAIN is an abstention; a
                   system_failure is neither, and is reported separately
  Tier D correct   the judge's behaviour label is CORRECT — declining for the
                   wrong reason is not a success (§6.2)
  hallucination    UNSUPPORTED sentences / factual sentences (SUPPORTED,
                   PARTIAL, UNSUPPORTED); GAP and NOT_A_CLAIM excluded
  selective        "correct if answered": A-C correctness of the ANSWER-MODE
  prediction       draft; answering any Tier D item is an error. Risk-coverage
                   sweeps the configuration's score over that.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

MIN_COVERAGE = 0.5
ANSWERED = ("ANSWER", "PARTIAL")


def weighted_coverage(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    return sum(1.0 if l == "PRESENT" else 0.5 if l == "PARTIAL" else 0.0
               for l in labels) / len(labels)


def correct_ac(labels: Sequence[str], min_coverage: float = MIN_COVERAGE) -> bool:
    return bool(labels) and "CONTRADICTED" not in labels and \
        weighted_coverage(labels) >= min_coverage


def hallucination_rate(labels: Iterable[str]) -> float | None:
    factual = [l for l in labels if l in ("SUPPORTED", "PARTIAL", "UNSUPPORTED")]
    if not factual:
        return None
    return sum(l == "UNSUPPORTED" for l in factual) / len(factual)


@dataclass
class Outcome:
    """One question under one configuration, graded."""
    id: str
    tier: str
    kind: str                       # answered | abstained | failure | pending
    correct: bool                   # did the right thing
    correct_if_answered: bool       # for selective prediction
    score: float | None
    coverage: float | None = None   # delivered report, A-C answered only
    grounding: tuple[str, ...] = ()
    tier_d_label: str | None = None
    llm_calls: int | None = None


def grade(record: dict, judged: dict) -> Outcome:
    tier = record.get("tier")
    decision = record.get("terminal_decision")
    rule = record.get("rule")
    if rule == "threshold_pending":
        kind = "pending"            # B3 before its tau exists (C33)
    elif rule == "system_failure" or decision is None:
        kind = "failure"
    elif decision in ANSWERED:
        kind = "answered"
    else:
        kind = "abstained"

    if tier == "D":
        label = judged.get("tier_d", {}).get("label")
        correct = kind != "failure" and label == "CORRECT"
        if_answered = False
        coverage = None
    else:
        answer_labels = judged.get("coverage_answer") or []
        delivered = judged.get("coverage_delivered") or answer_labels
        if_answered = correct_ac(answer_labels)
        correct = kind == "answered" and correct_ac(delivered)
        coverage = weighted_coverage(delivered) if kind == "answered" else None
        label = None
    budget = record.get("budget") or {}
    return Outcome(id=record["id"], tier=tier, kind=kind, correct=correct,
                   correct_if_answered=if_answered, score=record.get("score"),
                   coverage=coverage, grounding=tuple(judged.get("grounding") or ()),
                   tier_d_label=label, llm_calls=budget.get("llm_calls_used"))


# -- selective prediction ---------------------------------------------------


def risk_coverage(scores: Sequence[float], correct: Sequence[bool]) -> list[tuple[float, float]]:
    """(coverage, risk) answering the top-k by score, for k = 1..n.

    Ties are broken by the worse outcome first — the pessimistic curve, so a
    score that cannot rank two items gets no credit for an order it did not
    produce.
    """
    items = sorted(zip(scores, correct), key=lambda x: (-x[0], x[1]))
    n, wrong, points = len(items), 0, []
    for k, (_, ok) in enumerate(items, 1):
        wrong += not ok
        points.append((k / n, wrong / k))
    return points


def aurc(scores: Sequence[float], correct: Sequence[bool]) -> float | None:
    points = risk_coverage(scores, correct)
    return sum(r for _, r in points) / len(points) if points else None


def accuracy_at_coverage(scores, correct, coverage: float = 0.8) -> float | None:
    points = risk_coverage(scores, correct)
    eligible = [r for c, r in points if c >= coverage - 1e-9]
    return 1.0 - eligible[0] if eligible else None


def select_threshold(scores, correct, target_risk: float) -> float | None:
    """The lowest threshold whose answered set has risk <= target — the most
    coverage the target allows. None when no threshold achieves it. Used to
    choose tau on the calibration split (C33), for the agent and for B3."""
    best = None
    for tau in sorted(set(scores), reverse=True):
        answered = [ok for s, ok in zip(scores, correct) if s >= tau]
        if answered and (1 - sum(answered) / len(answered)) <= target_risk:
            best = tau
    return best


def ece(probs: Sequence[float], outcomes: Sequence[bool], bins: int = 5) -> float | None:
    """Expected calibration error, equal-width bins. Few bins: n is small."""
    if not probs:
        return None
    total, n = 0.0, len(probs)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if idx:
            conf = sum(probs[i] for i in idx) / len(idx)
            acc = sum(outcomes[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(conf - acc)
    return total


# -- aggregate ---------------------------------------------------------------


def summarise(outcomes: Sequence[Outcome]) -> dict:
    ac = [o for o in outcomes if o.tier in ("A", "B", "C")]
    d = [o for o in outcomes if o.tier == "D"]
    decided = lambda xs: [o for o in xs if o.kind in ("answered", "abstained")]
    ac_ok, d_ok = decided(ac), decided(d)
    answered_ac = [o for o in ac_ok if o.kind == "answered"]
    d_judged = [o for o in d_ok if o.tier_d_label not in (None, "UNJUDGED")]
    scored = [o for o in outcomes if o.kind != "failure" and o.score is not None]
    scores = [o.score for o in scored]
    cia = [o.correct_if_answered for o in scored]
    halluc = [l for o in decided(outcomes) for l in o.grounding]
    calls = [o.llm_calls for o in outcomes if o.llm_calls is not None]
    rate = lambda num, den: (num / den) if den else None
    return {
        "n": len(outcomes),
        "failures": sum(o.kind == "failure" for o in outcomes),
        "ac_answered": len(answered_ac),
        "ac_accuracy_answered": rate(sum(o.correct for o in answered_ac), len(answered_ac)),
        "ac_false_abstention": rate(sum(o.kind == "abstained" for o in ac_ok), len(ac_ok)),
        "ac_mean_coverage": rate(sum(o.coverage for o in answered_ac), len(answered_ac)),
        "d_correct": rate(sum(o.correct for o in d_judged), len(d_judged)),
        "d_false_answer": rate(sum(o.tier_d_label == "ANSWERED" for o in d_judged), len(d_judged)),
        "d_wrong_reason": rate(sum(o.tier_d_label == "WRONG_REASON" for o in d_judged), len(d_judged)),
        "unjudged": sum(o.tier_d_label in (None, "UNJUDGED") for o in d_ok),
        "hallucination": hallucination_rate(halluc),
        "aurc": aurc(scores, cia) if scores else None,
        "acc_at_80_coverage": accuracy_at_coverage(scores, cia) if scores else None,
        "ece": ece(scores, cia) if scores else None,
        "mean_llm_calls": rate(sum(calls), len(calls)),
    }


def fmt(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.3f}" if isinstance(value, float) else str(value)
