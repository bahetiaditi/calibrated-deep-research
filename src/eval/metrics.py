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
                   PARTIAL, UNSUPPORTED); GAP and NOT_A_CLAIM excluded.
                   Reported with its two parts (C37.2): `uncited` (factual
                   sentences citing nothing, judged UNSUPPORTED, / factual)
                   and `miscitation` (UNSUPPORTED among factual sentences
                   that DO cite / those sentences) — an agent that asserts
                   without citing and one that cites wrongly fail differently
  unjudged         a label the judge did not return (C37.2). Never a
                   verdict: the item is excluded from every rate that needs
                   it and counted. An A-C question with any unjudged gold
                   claim is not graded partially
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


FACTUAL = ("SUPPORTED", "PARTIAL", "UNSUPPORTED")
UNJUDGED = "UNJUDGED"


def hallucination_rate(labels: Iterable[str]) -> float | None:
    factual = [l for l in labels if l in FACTUAL]
    if not factual:
        return None
    return sum(l == "UNSUPPORTED" for l in factual) / len(factual)


def hallucination_parts(labels: Sequence[str], cited: Sequence[bool]
                        ) -> tuple[float | None, float | None]:
    """(uncited, miscitation). `uncited` shares the total's denominator, so
    uncited + (cited UNSUPPORTED / factual) = the total; `miscitation` is over
    cited factual sentences only — how often a citation fails its claim."""
    factual = [(l, c) for l, c in zip(labels, cited) if l in FACTUAL]
    if not factual:
        return None, None
    uncited = sum(l == "UNSUPPORTED" and not c for l, c in factual) / len(factual)
    with_cite = [l for l, c in factual if c]
    mis = (sum(l == "UNSUPPORTED" for l in with_cite) / len(with_cite)) if with_cite else None
    return uncited, mis


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
    grounding_cited: tuple[bool, ...] = ()   # per sentence: cites a corpus passage
    tier_d_label: str | None = None
    llm_calls: int | None = None
    unjudged: bool = False                   # the main grade could not be computed
    selective_unjudged: bool = False         # correct_if_answered could not be


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

    decided = kind in ("answered", "abstained")
    if tier == "D":
        label = judged.get("tier_d", {}).get("label")
        correct = kind != "failure" and label == "CORRECT"
        if_answered = False
        coverage = None
        unjudged = decided and label in (None, UNJUDGED)
        selective_unjudged = False      # answering a D item is an error, judged or not
    else:
        answer_labels = judged.get("coverage_answer") or []
        delivered = judged.get("coverage_delivered") or answer_labels
        selective_unjudged = UNJUDGED in answer_labels
        unjudged = kind == "answered" and UNJUDGED in delivered
        if_answered = correct_ac(answer_labels) and not selective_unjudged
        correct = kind == "answered" and not unjudged and correct_ac(delivered)
        coverage = weighted_coverage(delivered) if kind == "answered" and not unjudged else None
        label = None
    items = judged.get("grounding_items")
    if items:
        grounding = tuple(it.get("label") or UNJUDGED for it in items)
        cited = tuple(bool(it.get("cites")) for it in items)
    else:                               # judged before C37.2: labels only
        grounding = tuple(judged.get("grounding") or ())
        cited = ()
    budget = record.get("budget") or {}
    return Outcome(id=record["id"], tier=tier, kind=kind, correct=correct,
                   correct_if_answered=if_answered, score=record.get("score"),
                   coverage=coverage, grounding=grounding, grounding_cited=cited,
                   tier_d_label=label, llm_calls=budget.get("llm_calls_used"),
                   unjudged=unjudged, selective_unjudged=selective_unjudged)


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
    answered_ac = [o for o in ac_ok if o.kind == "answered" and not o.unjudged]
    d_judged = [o for o in d_ok if not o.unjudged]
    scored = [o for o in outcomes if o.kind != "failure" and o.score is not None
              and not o.selective_unjudged]
    scores = [o.score for o in scored]
    cia = [o.correct_if_answered for o in scored]
    halluc = [l for o in decided(outcomes) for l in o.grounding]
    # The decomposition needs per-sentence citation flags; outcomes judged
    # before C37.2 have none and are left out of the parts (not the total).
    parts = [(l, c) for o in decided(outcomes) if o.grounding_cited
             for l, c in zip(o.grounding, o.grounding_cited)]
    uncited, miscitation = hallucination_parts([l for l, _ in parts], [c for _, c in parts])
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
        "unjudged": sum(o.unjudged for o in decided(outcomes)),
        "hallucination": hallucination_rate(halluc),
        "halluc_uncited": uncited,
        "miscitation": miscitation,
        "grounding_unjudged": sum(l == UNJUDGED for l in halluc),
        "aurc": aurc(scores, cia) if scores else None,
        "acc_at_80_coverage": accuracy_at_coverage(scores, cia) if scores else None,
        "ece": ece(scores, cia) if scores else None,
        "mean_llm_calls": rate(sum(calls), len(calls)),
    }


def fmt(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.3f}" if isinstance(value, float) else str(value)
