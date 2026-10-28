"""Fixtures shared by the scorer and decider tests.

Synthetic but not arbitrary: stage 1 learns that strong rerank evidence means
a sufficient sub-question, stage 2 learns that a high mean stage-1
probability means an answerable question. That is enough for forced feature
vectors to land in each decision band, which is the C27 criterion.
"""
from __future__ import annotations

import random
from functools import lru_cache

from src.state import new_state, new_sub_question
from src.sufficiency.model import (
    RUN_FEATURE_NAMES,
    STAGE1_FEATURE_NAMES,
    SufficiencyModel,
)

STRONG, MEDIUM, WEAK = 7.0, 3.0, -2.0


@lru_cache(maxsize=1)
def fitted_models() -> tuple[SufficiencyModel, SufficiencyModel]:
    rng = random.Random(7)
    rows, labels = [], []
    for _ in range(90):
        label = rng.random() < 0.5
        center = STRONG if label else WEAK
        row = {n: rng.gauss(0, 1) for n in STAGE1_FEATURE_NAMES}
        row["f1_max_rerank"] = rng.gauss(center, 1.5)
        row["f2_mean_top3"] = rng.gauss(center - 1, 1.5)
        rows.append(row)
        labels.append(int(label))
    stage1 = SufficiencyModel(STAGE1_FEATURE_NAMES, with_missing_indicators=False)
    stage1.fit(rows, labels, calibrate="never")

    run_rows, run_labels = [], []
    for _ in range(80):
        mean_p = rng.random()
        run_rows.append({
            "mean_sub_question_p": mean_p,
            "min_sub_question_p": mean_p * rng.uniform(0.6, 1.0),
            "premise_grounding": rng.uniform(0.5, 1.0),
            "fraction_sufficient": min(1.0, max(0.0, mean_p + rng.gauss(0, 0.2))),
        })
        run_labels.append(int(rng.random() < mean_p))
    stage2 = SufficiencyModel(RUN_FEATURE_NAMES, with_missing_indicators=False)
    stage2.fit(run_rows, run_labels, calibrate="never")
    return stage1, stage2


def passage(pid, sqid, score, source="p1"):
    return {
        "id": pid, "source_type": "arxiv", "source_id": source,
        "source_domain": "arxiv.org", "title": "t", "text": f"text {pid}",
        "section": "Results", "published": "2025-01-01",
        "sub_question_id": sqid, "retrieval_score": 0.5, "rerank_score": score,
    }


def decision_state(*strengths, statuses=None, verdicts=None, draft="A report [P1].",
                   draft_mode="answer", premise=None):
    """A finished run: one sub-question per strength, critic already run."""
    state = new_state("How do X and Y differ?")
    state["premise_check"] = premise or {"ran": True, "failed": False,
                                         "grounding_score": 1.0}
    plan, evidence = [], []
    for i, strength in enumerate(strengths):
        sqid = f"sq{i + 1}"
        sq = new_sub_question(f"sub-question {i + 1}", sub_question_id=sqid)
        sq["status"] = (statuses or {}).get(
            sqid, "sufficient" if strength >= MEDIUM else "insufficient")
        sq["rounds"] = 1
        sq["attempted_routes"] = ["arxiv_meta"]
        ps = [passage(f"P{sqid}{j}", sqid, strength - j * 0.3, source=f"s{j}")
              for j in range(5)]
        sq["evidence_ids"] = [p["id"] for p in ps]
        plan.append(sq)
        evidence.extend(ps)
    state["plan"] = plan
    state["evidence"] = evidence
    state["draft"] = draft
    state["draft_mode"] = draft_mode
    state["critic_findings"] = [
        {"claim": f"claim about {sqid}", "sub_question_id": sqid,
         "cited_passage_ids": [], "verdict": v, "reasoning": "",
         "action_taken": "ACCEPT" if v == "SUPPORTED" else "UNRESOLVED"}
        for sqid, v in (verdicts or {}).items()
    ]
    return state
