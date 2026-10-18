"""Tests for sufficiency feature extraction.

The project's first claim rests on this vector carrying information, so the
properties that matter are: missing is never silently zero, the vector is
dimensionally stable, and each feature measures what its name says.
"""
from datetime import date

import pytest

from src.state import Passage, new_state, new_sub_question
from src.sufficiency.features import (
    FEATURE_NAMES,
    IMPUTE,
    RUN_FEATURE_NAMES,
    completeness,
    extract_features,
    extract_run_features,
    feature_vector_length,
    features_for_sub_question,
    to_vector,
)

TODAY = date(2026, 9, 6)


def passage(pid, *, score=7.0, source="paperA", published="2024-01-01"):
    return Passage(
        id=pid, source_type="arxiv", source_id=source,
        source_domain="arxiv.org", title="t", text=f"text {pid}",
        section="Results", published=published, sub_question_id="sq1",
        retrieval_score=0.5, rerank_score=score,
    )


def sub_question(**over):
    sq = new_sub_question("a sub-question", sub_question_id="sq1")
    sq.update(over)
    return sq


# --- dimensional stability (the C20 criterion) -----------------------------


def test_every_key_is_always_present():
    """A vector whose length depends on what happened cannot be stacked into
    a design matrix, and the failure would surface at C33 as a shape error
    after the eval had been paid for."""
    for passages in ([], [passage("P1")], [passage(f"P{i}") for i in range(9)]):
        assert set(extract_features(sub_question(), passages)) == set(FEATURE_NAMES)


def test_vector_length_is_constant():
    short = to_vector(extract_features(sub_question(), []))[0]
    long = to_vector(extract_features(
        sub_question(rounds=3, attempted_routes=["arxiv_fulltext"]),
        [passage(f"P{i}", score=9 - i) for i in range(8)],
        contradictions=[], premise_check={"ran": True, "grounding_score": 1.0},
    ))[0]
    assert len(short) == len(long) == feature_vector_length()


def test_column_names_are_stable():
    a = to_vector(extract_features(sub_question(), []))[1]
    b = to_vector(extract_features(sub_question(), [passage("P1")]))[1]
    assert a == b


def test_run_features_always_complete():
    assert set(extract_run_features(new_state("q"))) == set(RUN_FEATURE_NAMES)


# --- missing is not zero ---------------------------------------------------


def test_unavailable_features_are_none_not_zero():
    """Imputing 0.0 for f5 would tell the model 'no contradictions were
    found' when the truth is 'nobody looked'."""
    features = extract_features(sub_question(), [passage("P1")])
    assert features["f5_contradiction_rate"] is None     # scanning not run
    assert features["f6_entailment_strength"] is None    # critic not run
    assert features["f7_premise_grounding"] is None      # checker not run


def test_contradiction_rate_zero_differs_from_none():
    with_scan = extract_features(
        sub_question(), [passage("P1"), passage("P2", source="paperB")],
        contradictions=[],
    )
    without = extract_features(
        sub_question(), [passage("P1"), passage("P2", source="paperB")]
    )
    assert with_scan["f5_contradiction_rate"] == 0.0
    assert without["f5_contradiction_rate"] is None


def test_missing_indicator_is_emitted():
    values, columns = to_vector(extract_features(sub_question(), []))
    index = columns.index("f5_contradiction_rate_missing")
    assert values[index] == 1.0
    assert values[columns.index("f5_contradiction_rate")] == IMPUTE[
        "f5_contradiction_rate"
    ]


def test_present_feature_has_indicator_zero():
    values, columns = to_vector(
        extract_features(sub_question(), [passage("P1", score=4.2)])
    )
    assert values[columns.index("f1_max_rerank")] == pytest.approx(4.2)
    assert values[columns.index("f1_max_rerank_missing")] == 0.0


def test_indicators_can_be_suppressed():
    values, columns = to_vector(
        extract_features(sub_question(), []), with_missing_indicators=False
    )
    assert len(values) == len(FEATURE_NAMES)
    assert not any(c.endswith("_missing") for c in columns)


def test_completeness_reports_observation_rate():
    assert completeness(extract_features(sub_question(), [])) < 0.5
    full = extract_features(
        sub_question(rounds=1, attempted_routes=["web"]),
        [passage(f"P{i}", score=9 - i) for i in range(6)],
        contradictions=[], premise_check={"ran": True, "grounding_score": 0.9},
        critic_findings=[{"sub_question_id": "sq1", "verdict": "SUPPORTED"}],
    )
    assert completeness(full) == 1.0


# --- f1-f3: rerank scores (logits, after the C17 fix) ----------------------


def test_f1_is_the_top_logit():
    f = extract_features(sub_question(),
                         [passage("P1", score=3.4), passage("P2", score=9.1)])
    assert f["f1_max_rerank"] == pytest.approx(9.1)


def test_f2_averages_the_top_three():
    scores = [9.0, 8.0, 7.0, 1.0]
    f = extract_features(
        sub_question(),
        [passage(f"P{i}", score=s) for i, s in enumerate(scores)],
    )
    assert f["f2_mean_top3"] == pytest.approx(8.0)


def test_f3_is_none_below_five_results():
    """A zero gap is a strong claim — 'the evidence is uniformly flat' —
    that four passages do not license. Established at C11."""
    f = extract_features(
        sub_question(), [passage(f"P{i}", score=9 - i) for i in range(4)]
    )
    assert f["f3_score_gap"] is None


def test_f3_measures_peak_sharpness():
    sharp = extract_features(sub_question(), [
        passage("P0", score=9.0), *[passage(f"P{i}", score=1.0) for i in range(1, 6)]
    ])
    flat = extract_features(sub_question(), [
        passage(f"P{i}", score=5.0) for i in range(6)
    ])
    assert sharp["f3_score_gap"] > 7.0
    assert flat["f3_score_gap"] == pytest.approx(0.0)


def test_passages_without_rerank_scores_are_ignored():
    p = passage("P1")
    p["rerank_score"] = None
    assert extract_features(sub_question(), [p])["f1_max_rerank"] is None


# --- f4: independent sources ----------------------------------------------


def test_f4_counts_distinct_papers_not_chunks():
    f = extract_features(sub_question(), [
        passage("P1", source="paperA"), passage("P2", source="paperA"),
        passage("P3", source="paperB"),
    ])
    assert f["f4_independent_sources"] == 2.0


def test_f4_is_none_with_no_evidence():
    """'No sources' and 'zero independent among those found' are the same
    number but not the same situation."""
    assert extract_features(sub_question(), [])["f4_independent_sources"] is None


# --- f5: contradiction rate ------------------------------------------------


def test_f5_counts_only_pairs_among_these_passages():
    contradictions = [
        {"passage_a": "P1", "passage_b": "P2", "score": 0.9},
        {"passage_a": "P8", "passage_b": "P9", "score": 0.9},   # another sq
    ]
    f = extract_features(
        sub_question(),
        [passage("P1"), passage("P2", source="B"), passage("P3", source="C")],
        contradictions=contradictions,
    )
    assert f["f5_contradiction_rate"] == pytest.approx(1 / 3)


def test_f5_zero_with_a_single_passage():
    f = extract_features(sub_question(), [passage("P1")], contradictions=[])
    assert f["f5_contradiction_rate"] == 0.0


# --- f6: entailment strength ----------------------------------------------


def test_f6_averages_this_sub_questions_verdicts():
    f = extract_features(sub_question(), [passage("P1")], critic_findings=[
        {"sub_question_id": "sq1", "verdict": "SUPPORTED"},
        {"sub_question_id": "sq1", "verdict": "PARTIAL"},
        {"sub_question_id": "sq2", "verdict": "UNSUPPORTED"},   # not ours
    ])
    assert f["f6_entailment_strength"] == pytest.approx(0.75)


def test_f6_none_when_no_claims_belong_to_this_sub_question():
    f = extract_features(sub_question(), [passage("P1")], critic_findings=[
        {"sub_question_id": "sq2", "verdict": "SUPPORTED"},
    ])
    assert f["f6_entailment_strength"] is None


def test_contradicted_scores_as_zero_like_unsupported():
    f = extract_features(sub_question(), [passage("P1")], critic_findings=[
        {"sub_question_id": "sq1", "verdict": "CONTRADICTED"},
    ])
    assert f["f6_entailment_strength"] == 0.0


# --- f7-f9 -----------------------------------------------------------------


def test_f7_comes_from_the_premise_checker():
    f = extract_features(sub_question(), [passage("P1")],
                         premise_check={"ran": True, "grounding_score": 0.75})
    assert f["f7_premise_grounding"] == 0.75


def test_f7_none_when_the_checker_did_not_run():
    f = extract_features(sub_question(), [passage("P1")],
                         premise_check={"ran": False})
    assert f["f7_premise_grounding"] is None


def test_f8_counts_rounds():
    assert extract_features(sub_question(rounds=3), [])["f8_rounds_consumed"] == 3.0


def test_f9_measures_how_expensive_a_route_was_needed():
    cheap = extract_features(
        sub_question(attempted_routes=["arxiv_meta"]), [])["f9_route_escalation"]
    expensive = extract_features(
        sub_question(attempted_routes=["arxiv_meta", "arxiv_fulltext"]),
        [])["f9_route_escalation"]
    assert cheap < expensive == 1.0


def test_f9_none_before_any_retrieval():
    assert extract_features(sub_question(), [])["f9_route_escalation"] is None


# --- f10: recency ----------------------------------------------------------


def test_f10_recent_evidence_scores_high():
    f = extract_features(sub_question(),
                         [passage("P1", published="2026-06-01")], today=TODAY)
    assert f["f10_recency"] > 0.9


def test_f10_old_evidence_scores_low():
    f = extract_features(sub_question(),
                         [passage("P1", published="2017-06-01")], today=TODAY)
    assert f["f10_recency"] < 0.2


def test_f10_uses_the_median_not_the_newest():
    """One recent citation should not make a corpus of old evidence look
    current."""
    passages = [passage(f"P{i}", published="2016-01-01") for i in range(4)]
    passages.append(passage("Pnew", published="2026-09-01"))
    assert extract_features(sub_question(), passages, today=TODAY)["f10_recency"] < 0.3


def test_f10_none_without_dates():
    f = extract_features(sub_question(), [passage("P1", published=None)])
    assert f["f10_recency"] is None


def test_f10_tolerates_malformed_dates():
    f = extract_features(sub_question(), [passage("P1", published="n.d.")])
    assert f["f10_recency"] is None


# --- run-level aggregates --------------------------------------------------


def test_run_features_summarise_the_plan():
    state = new_state("q")
    state["plan"] = [
        new_sub_question("a", sub_question_id="sq1"),
        new_sub_question("b", sub_question_id="sq2"),
    ]
    state["plan"][0]["sufficiency"] = 0.9
    state["plan"][0]["status"] = "sufficient"
    state["plan"][1]["sufficiency"] = 0.3
    state["plan"][1]["status"] = "insufficient"
    state["plan_revisions"] = 1

    r = extract_run_features(state)
    assert r["r1_mean_sufficiency"] == pytest.approx(0.6)
    assert r["r2_min_sufficiency"] == pytest.approx(0.3)
    assert r["r3_fraction_sufficient"] == pytest.approx(0.5)
    assert r["r4_plan_revisions"] == 1.0


def test_minimum_is_reported_separately_from_the_mean():
    """A report resting on one unsupported sub-question is unsafe however
    well the others went, and averaging hides exactly that."""
    state = new_state("q")
    state["plan"] = [
        new_sub_question(f"s{i}", sub_question_id=f"sq{i}") for i in range(4)
    ]
    for i, value in enumerate([0.95, 0.95, 0.95, 0.05]):
        state["plan"][i]["sufficiency"] = value
    r = extract_run_features(state)
    assert r["r1_mean_sufficiency"] > 0.7
    assert r["r2_min_sufficiency"] == pytest.approx(0.05)


def test_budget_fraction_from_a_budget_object():
    from src.budget import Budget

    budget = Budget(llm_calls_max=20, retrieval_calls_max=20,
                    wall_clock_max_s=900)
    budget.spend("llm", 10)
    r = extract_run_features(new_state("q"), budget=budget)
    assert r["r5_budget_fraction"] == pytest.approx(0.25)


def test_empty_plan_yields_none_not_zero():
    r = extract_run_features(new_state("q"))
    assert r["r1_mean_sufficiency"] is None
    assert r["r3_fraction_sufficient"] is None


# --- state integration -----------------------------------------------------


def test_features_from_state():
    state = new_state("q")
    state["plan"] = [new_sub_question("a", sub_question_id="sq1")]
    state["plan"][0]["evidence_ids"] = ["P1", "P2"]
    state["plan"][0]["rounds"] = 2
    state["plan"][0]["attempted_routes"] = ["arxiv_meta", "arxiv_fulltext"]
    state["evidence"] = [passage("P1", score=8.0), passage("P2", score=3.0)]
    state["premise_check"] = {"ran": True, "grounding_score": 1.0}

    f = features_for_sub_question(state, "sq1", today=TODAY)
    assert f["f1_max_rerank"] == 8.0
    assert f["f8_rounds_consumed"] == 2.0
    assert f["f9_route_escalation"] == 1.0
    assert f["f7_premise_grounding"] == 1.0


def test_unknown_sub_question_returns_all_none():
    f = features_for_sub_question(new_state("q"), "nope")
    assert set(f) == set(FEATURE_NAMES)
    assert all(v is None for v in f.values())