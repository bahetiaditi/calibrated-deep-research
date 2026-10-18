"""Evidence-sufficiency features (§5.2).

These are the inputs to the calibrated model that decides whether to answer.
The project's first claim — that a retrieval-grounded signal is better
calibrated than the model's verbalized confidence — stands or falls on
whether this vector carries information, so the rules here are conservative:

**Missing is not zero.** f6 needs the critic (C23) and f5 needs contradiction
scanning, so at C20 several features are genuinely unavailable. They return
`None`, and `to_vector()` emits an explicit missingness indicator alongside
the imputed value. Silently imputing 0.0 would tell the model "no
contradictions were found" when the truth is "nobody looked" — two very
different states that would be collapsed into one, and the coefficient fit at
C33 would be meaningless.

**Dimensional consistency is a hard requirement.** `FEATURE_NAMES` fixes the
order; every extraction returns every key. A vector whose length depends on
what happened cannot be stacked into a design matrix, and the failure would
appear at C33 as a shape error after the eval had already been paid for.

**Units differ and that is fine here.** f1-f3 are logits (unbounded, roughly
-11 to +11), f4 and f8 are counts, f5/f6/f7/f10 are in [0,1]. Standardisation
belongs in the model pipeline at C26, not here — scaling at extraction time
would mean the stored features depend on the batch they were extracted with,
and a feature logged during a run could not be compared to one logged a week
later.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Sequence

from src.state import Passage, ResearchState, SubQuestion, evidence_for

log = logging.getLogger(__name__)

# Order is fixed. Changing it invalidates every persisted feature vector and
# any model fitted on them.
FEATURE_NAMES: tuple[str, ...] = (
    "f1_max_rerank",
    "f2_mean_top3",
    "f3_score_gap",
    "f4_independent_sources",
    "f5_contradiction_rate",
    "f6_entailment_strength",
    "f7_premise_grounding",
    "f8_rounds_consumed",
    "f9_route_escalation",
    "f10_recency",
)

RUN_FEATURE_NAMES: tuple[str, ...] = (
    "r1_mean_sufficiency",
    "r2_min_sufficiency",
    "r3_fraction_sufficient",
    "r4_plan_revisions",
    "r5_budget_fraction",
)

# How far the retriever had to go. The ordering is cost, not quality: an
# answer found in an abstract required less work than one that needed the
# full paper, and needing the expensive route is weak evidence that the
# question was hard.
ROUTE_DEPTH = {
    "evidence_store": 0,
    "arxiv_meta": 1,
    "web": 1,
    "arxiv_fulltext": 2,
}
MAX_ROUTE_DEPTH = 2

# Verdict → entailment strength for f6.
VERDICT_STRENGTH = {
    "SUPPORTED": 1.0,
    "PARTIAL": 0.5,
    "UNSUPPORTED": 0.0,
    "CONTRADICTED": 0.0,
}

# f10 saturates here: evidence older than this is uniformly stale for a
# question about current ML research, and distinguishing 9 years from 12
# would be false precision.
RECENCY_HORIZON_YEARS = 8.0

# Values used when a feature is missing. Each is the neutral reading, and
# each is paired with a `*_missing` indicator so the model can learn that
# missingness itself predicts something.
IMPUTE = {
    "f1_max_rerank": 0.0,
    "f2_mean_top3": 0.0,
    "f3_score_gap": 0.0,
    "f4_independent_sources": 0.0,
    "f5_contradiction_rate": 0.0,
    "f6_entailment_strength": 0.5,
    "f7_premise_grounding": 0.5,
    "f8_rounds_consumed": 0.0,
    "f9_route_escalation": 0.0,
    "f10_recency": 0.5,
}


# ---------------------------------------------------------------------------
# Per-sub-question features
# ---------------------------------------------------------------------------


def extract_features(
    sub_question: SubQuestion,
    passages: Sequence[Passage],
    *,
    contradictions: Sequence[dict[str, Any]] | None = None,
    contradictions_scanned: bool = False,
    premise_check: dict[str, Any] | None = None,
    critic_findings: Sequence[dict[str, Any]] | None = None,
    today: date | None = None,
) -> dict[str, float | None]:
    """f1-f10 for one sub-question. Always returns every key.

    `contradictions_scanned` defaults to False: nothing wires the C12
    detector into the graph yet, so claiming a clean scan would be a lie.
    C21 flips it where the scan actually runs.
    """
    scores = sorted(
        (
            float(p["rerank_score"])
            for p in passages
            if p.get("rerank_score") is not None
        ),
        reverse=True,
    )

    return {
        "f1_max_rerank": scores[0] if scores else None,
        "f2_mean_top3": (
            sum(scores[:3]) / len(scores[:3]) if scores else None
        ),
        # None rather than 0.0 below five results: a zero gap is a strong
        # claim ("the evidence is uniformly flat") that four passages do not
        # license. Established at C11 and preserved here.
        "f3_score_gap": (scores[0] - scores[4]) if len(scores) >= 5 else None,
        "f4_independent_sources": _independent_sources(passages),
        "f5_contradiction_rate": _contradiction_rate(
            passages, contradictions, scanned=contradictions_scanned
        ),
        "f6_entailment_strength": _entailment_strength(
            sub_question, critic_findings
        ),
        "f7_premise_grounding": _premise_grounding(premise_check),
        "f8_rounds_consumed": float(sub_question.get("rounds", 0) or 0),
        "f9_route_escalation": _route_escalation(sub_question),
        "f10_recency": _recency(passages, today),
    }


def _independent_sources(passages: Sequence[Passage]) -> float | None:
    """f4. Distinct origins — two chunks of one paper are one source.

    Returning None on empty evidence rather than 0.0: "no sources" and "zero
    independent sources among the ones found" are the same number but not the
    same situation, and f1 already encodes the former.
    """
    if not passages:
        return None
    return float(len({p.get("source_id") for p in passages if p.get("source_id")}))


def _contradiction_rate(
    passages: Sequence[Passage],
    contradictions: Sequence[dict[str, Any]] | None,
    *,
    scanned: bool = True,
) -> float | None:
    """f5. Fraction of scorable passage pairs found to contradict.

    `None` when scanning has not run — distinct from 0.0, which asserts that
    sources agree.

    `scanned` exists because the two states are indistinguishable from the
    data alone: `ResearchState` initialises `contradictions` to `[]`, so an
    unscanned run and a scanned-and-clean run both present an empty list.
    A live C20 run reported f5 = 0.00 on all ten questions for exactly this
    reason — the feature was asserting agreement that nothing had checked,
    and at C33 the model would have been fit on a constant mislabelled as an
    observation.
    """
    if contradictions is None or not scanned:
        return None
    ids = {p["id"] for p in passages}
    if len(ids) < 2:
        return 0.0
    relevant = [
        c for c in contradictions
        if c.get("passage_a") in ids and c.get("passage_b") in ids
    ]
    total_pairs = len(ids) * (len(ids) - 1) / 2
    return len(relevant) / total_pairs if total_pairs else 0.0


def _entailment_strength(
    sub_question: SubQuestion,
    critic_findings: Sequence[dict[str, Any]] | None,
) -> float | None:
    """f6. Mean critic verdict for this sub-question's claims.

    `None` before the critic runs (C23), which is most of a run's life — the
    retriever computes features long before any draft exists.
    """
    if critic_findings is None:
        return None
    mine = [
        f for f in critic_findings
        if f.get("sub_question_id") == sub_question["id"]
    ]
    if not mine:
        return None
    strengths = [VERDICT_STRENGTH.get(str(f.get("verdict")), 0.5) for f in mine]
    return sum(strengths) / len(strengths)


def _premise_grounding(premise_check: dict[str, Any] | None) -> float | None:
    """f7. From the premise checker (C15).

    Run-level rather than per-sub-question — the presuppositions belong to
    the parent question — but it varies between questions, which is what the
    model needs.
    """
    if not premise_check or not premise_check.get("ran"):
        return None
    value = premise_check.get("grounding_score")
    return float(value) if value is not None else None


def _route_escalation(sub_question: SubQuestion) -> float | None:
    """f9. How expensive a route the sub-question needed, normalised."""
    routes = sub_question.get("attempted_routes") or []
    if not routes:
        return None
    depth = max(ROUTE_DEPTH.get(str(r), 1) for r in routes)
    return depth / MAX_ROUTE_DEPTH


def _recency(passages: Sequence[Passage], today: date | None) -> float | None:
    """f10. Evidence recency, 1.0 = published today, 0.0 = at the horizon.

    **Deviation from §5.2, recorded deliberately.** The spec describes "gap
    between the question's implied timeframe and source dates". Extracting a
    question's implied timeframe needs an LLM call per question, and the
    premise checker already costs two — the guard would start costing more
    than the research.

    This measures evidence recency instead, which is a proxy rather than the
    thing itself. It still carries the signal Tier D4 needs: a question whose
    answer has moved will retrieve old evidence, and old evidence on a
    fast-moving topic is a reason to hedge. It will NOT catch a recent
    question answered with recent but superseded evidence. Worth stating in
    the writeup rather than implying the full version was built.
    """
    today = today or date.today()
    dates = []
    for passage in passages:
        published = passage.get("published")
        if not published:
            continue
        try:
            dates.append(date.fromisoformat(str(published)[:10]))
        except ValueError:
            continue
    if not dates:
        return None

    median = sorted(dates)[len(dates) // 2]
    years = (today - median).days / 365.25
    return max(0.0, min(1.0, 1.0 - years / RECENCY_HORIZON_YEARS))


# ---------------------------------------------------------------------------
# Run-level aggregates
# ---------------------------------------------------------------------------


def extract_run_features(
    state: ResearchState, *, budget: Any = None
) -> dict[str, float | None]:
    """Aggregates over the whole plan (§5.2). Always returns every key."""
    plan = list(state.get("plan") or [])
    scored = [
        float(sq["sufficiency"]) for sq in plan
        if sq.get("sufficiency") is not None
    ]

    budget_fraction = None
    if budget is not None:
        budget_fraction = budget.fraction_consumed()
    elif state.get("budget"):
        b = state["budget"]
        caps = b.get("llm_calls_max", 0) + b.get("retrieval_calls_max", 0)
        if caps:
            used = b.get("llm_calls_used", 0) + b.get("retrieval_calls_used", 0)
            budget_fraction = used / caps

    return {
        "r1_mean_sufficiency": sum(scored) / len(scored) if scored else None,
        # The minimum, not the mean, is what a reader should care about: a
        # report resting on one unsupported sub-question is unsafe however
        # well the others went, and averaging hides exactly that.
        "r2_min_sufficiency": min(scored) if scored else None,
        "r3_fraction_sufficient": (
            sum(1 for sq in plan if sq.get("status") == "sufficient") / len(plan)
            if plan else None
        ),
        "r4_plan_revisions": float(state.get("plan_revisions", 0) or 0),
        "r5_budget_fraction": budget_fraction,
    }


# ---------------------------------------------------------------------------
# Vectorisation
# ---------------------------------------------------------------------------


def to_vector(
    features: dict[str, float | None],
    *,
    names: Sequence[str] = FEATURE_NAMES,
    with_missing_indicators: bool = True,
) -> tuple[list[float], list[str]]:
    """Dense vector plus column names, with explicit missingness.

    Each missing feature contributes its neutral imputation AND a `_missing`
    indicator set to 1.0. Imputing without the indicator would tell the model
    "no contradictions were found" when the truth is "nobody looked" — two
    different states collapsed into one. With the indicator, the model can
    learn that not having looked predicts something, which it plausibly does:
    a sub-question whose critic never ran is one the system gave up on.
    """
    values: list[float] = []
    columns: list[str] = []

    for name in names:
        raw = features.get(name)
        missing = raw is None
        values.append(IMPUTE.get(name, 0.0) if missing else float(raw))
        columns.append(name)
        if with_missing_indicators:
            values.append(1.0 if missing else 0.0)
            columns.append(f"{name}_missing")

    return values, columns


def feature_vector_length(
    *, names: Sequence[str] = FEATURE_NAMES, with_missing_indicators: bool = True
) -> int:
    return len(names) * (2 if with_missing_indicators else 1)


def completeness(features: dict[str, float | None]) -> float:
    """Fraction of features actually observed. For diagnostics, not the model."""
    if not features:
        return 0.0
    return sum(1 for v in features.values() if v is not None) / len(features)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def features_for_sub_question(
    state: ResearchState,
    sub_question_id: str,
    *,
    contradictions_scanned: bool = False,
    today: date | None = None,
) -> dict[str, float | None]:
    """Extract features from state for one sub-question."""
    from src.state import find_sub_question

    sub_question = find_sub_question(state, sub_question_id)
    if sub_question is None:
        return {name: None for name in FEATURE_NAMES}

    return extract_features(
        sub_question,
        evidence_for(state, sub_question_id),
        contradictions=state.get("contradictions"),
        contradictions_scanned=contradictions_scanned,
        premise_check=state.get("premise_check"),
        critic_findings=state.get("critic_findings") or None,
        today=today,
    )