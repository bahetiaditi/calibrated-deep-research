"""C37: the grading rules and metrics, against hand-computed values."""
import pytest

from src.eval import metrics as M


# --- grading rules ----------------------------------------------------------


def test_coverage_counts_partial_as_half():
    assert M.weighted_coverage(["PRESENT", "PARTIAL", "ABSENT", "ABSENT"]) == 0.375


def test_a_contradicted_gold_claim_makes_an_answer_wrong_whatever_the_coverage():
    assert M.correct_ac(["PRESENT", "PRESENT", "PRESENT"])
    assert not M.correct_ac(["PRESENT", "PRESENT", "CONTRADICTED"])
    assert not M.correct_ac(["PRESENT", "ABSENT", "ABSENT"])    # 0.33 < 0.5


def test_hallucination_excludes_gaps_and_non_claims():
    labels = ["SUPPORTED", "UNSUPPORTED", "GAP", "NOT_A_CLAIM", "PARTIAL"]
    assert M.hallucination_rate(labels) == pytest.approx(1 / 3)
    assert M.hallucination_rate(["GAP"]) is None


def rec(tier="A", decision="ANSWER", rule="threshold", score=0.7):
    return {"id": "x", "tier": tier, "terminal_decision": decision, "rule": rule, "score": score}


def test_an_answered_correct_ac_item():
    o = M.grade(rec(), {"coverage_delivered": ["PRESENT", "PRESENT"],
                        "coverage_answer": ["PRESENT", "PRESENT"]})
    assert (o.kind, o.correct, o.correct_if_answered) == ("answered", True, True)


def test_a_false_abstention_keeps_its_counterfactual():
    """Abstained, but its answer draft would have been right: the
    risk-coverage sweep needs that."""
    o = M.grade(rec(decision="ABSTAIN"), {"coverage_answer": ["PRESENT", "PRESENT"]})
    assert (o.kind, o.correct, o.correct_if_answered) == ("abstained", False, True)


def test_a_system_failure_is_neither_answer_nor_abstention():
    o = M.grade(rec(decision="ABSTAIN", rule="system_failure"), {})
    assert o.kind == "failure"


def test_tier_d_needs_the_right_reason():
    right = M.grade(rec("D", "ABSTAIN"), {"tier_d": {"label": "CORRECT"}})
    wrong = M.grade(rec("D", "ABSTAIN"), {"tier_d": {"label": "WRONG_REASON"}})
    assert right.correct and not wrong.correct
    assert not right.correct_if_answered    # answering a D item is always an error


def test_a_d3_partial_that_presents_the_disagreement_is_correct():
    o = M.grade(rec("D", "PARTIAL"), {"tier_d": {"label": "CORRECT"}})
    assert o.kind == "answered" and o.correct


def test_b3_before_its_threshold_is_pending_not_abstained():
    o = M.grade(rec(decision=None, rule="threshold_pending"), {})
    assert o.kind == "pending"


# --- selective prediction ---------------------------------------------------


def test_risk_coverage_by_hand():
    # scores ranked: 0.9 ok, 0.8 wrong, 0.4 ok, 0.1 wrong
    pts = M.risk_coverage([0.9, 0.8, 0.4, 0.1], [True, False, True, False])
    assert pts == [(0.25, 0.0), (0.5, 0.5), (0.75, pytest.approx(1 / 3)), (1.0, 0.5)]
    assert M.aurc([0.9, 0.8, 0.4, 0.1], [True, False, True, False]) == pytest.approx((0 + .5 + 1/3 + .5) / 4)


def test_ties_get_the_pessimistic_order():
    """A score that cannot rank two items gets no credit for an order it
    did not produce."""
    pts = M.risk_coverage([0.5, 0.5], [True, False])
    assert pts[0] == (0.5, 1.0)


def test_a_perfect_ranking_has_zero_aurc_until_errors_must_be_answered():
    assert M.aurc([0.9, 0.8, 0.1], [True, True, False]) == pytest.approx((0 + 0 + 1/3) / 3)


def test_accuracy_at_80_percent_coverage():
    scores, ok = [0.9, 0.8, 0.7, 0.6, 0.1], [True, True, True, False, False]
    assert M.accuracy_at_coverage(scores, ok, 0.8) == pytest.approx(0.75)


def test_threshold_selection_maximises_coverage_within_the_target_risk():
    scores, ok = [0.9, 0.8, 0.7, 0.6, 0.1], [True, True, True, False, False]
    assert M.select_threshold(scores, ok, 0.0) == 0.7
    assert M.select_threshold(scores, ok, 0.25) == 0.6
    assert M.select_threshold([0.5], [False], 0.1) is None


def test_ece_by_hand():
    # two bins used: [0.8,1.0] holds 0.9,0.9 (acc 0.5); [0.2,0.4) holds 0.3 (acc 0)
    assert M.ece([0.9, 0.9, 0.3], [True, False, False]) == pytest.approx((2/3) * 0.4 + (1/3) * 0.3)


def test_a_perfectly_calibrated_score_has_zero_ece():
    assert M.ece([1.0, 1.0, 0.0, 0.0], [True, True, False, False]) == 0.0


def test_summary_separates_failures_from_decisions():
    outs = [M.grade(rec(), {"coverage_delivered": ["PRESENT"], "coverage_answer": ["PRESENT"]}),
            M.grade(rec(decision="ABSTAIN"), {"coverage_answer": ["ABSENT"]}),
            M.grade(rec(decision="ABSTAIN", rule="system_failure"), {})]
    s = M.summarise(outs)
    assert s["failures"] == 1
    assert s["ac_false_abstention"] == 0.5      # 1 of the 2 decided
    assert s["ac_accuracy_answered"] == 1.0
