"""Tests for the D5 policy (pure function).

The C27 criterion is that forced inputs produce each of the three decisions.
The tests beyond it guard the properties the metrics depend on: the premise
rule bypasses tau however high the score, a crash is never an epistemic
abstention, and the static policy keeps the decision it overrode.
"""
import pytest

from src.sufficiency.policy import (
    ABSTAIN,
    ANSWER,
    PARTIAL,
    RULE_NO_EVIDENCE,
    RULE_PREMISE,
    RULE_STATIC,
    RULE_SYSTEM_FAILURE,
    RULE_THRESHOLD,
    DecisionInputs,
    Thresholds,
    decide,
    enumerate_gaps,
    gap_for,
    static_decision,
)

T = Thresholds(answer=0.65, partial=0.40, source="test")


def inputs(score, **over):
    base = dict(score=score, thresholds=T, calibrated=True, scorer="calibrated")
    base.update(over)
    return DecisionInputs(**base)


# --- the C27 criterion -----------------------------------------------------


@pytest.mark.parametrize("score,expected", [
    (0.90, ANSWER), (0.52, PARTIAL), (0.10, ABSTAIN),
])
def test_each_decision_is_reachable(score, expected):
    decision = decide(inputs(score))
    assert decision.terminal == expected
    assert decision.rule == RULE_THRESHOLD


@pytest.mark.parametrize("score,expected", [
    (0.65, ANSWER), (0.6499, PARTIAL), (0.40, PARTIAL), (0.3999, ABSTAIN),
])
def test_boundaries_are_inclusive_from_above(score, expected):
    assert decide(inputs(score)).terminal == expected


# --- hard rules ------------------------------------------------------------


def test_refuted_premise_abstains_whatever_the_score():
    """Categorical, not probabilistic (§5.4)."""
    decision = decide(inputs(0.99, premise_failed=True,
                             premise_correction="Mamba-2 uses no positional encoding"))
    assert (decision.terminal, decision.rule) == (ABSTAIN, RULE_PREMISE)
    assert "Mamba-2 uses no positional encoding" in decision.rationale
    assert decision.epistemic


def test_premise_rule_can_be_switched_off():
    decision = decide(inputs(0.99, premise_failed=True,
                             abstain_on_failed_premise=False))
    assert decision.terminal == ANSWER


def test_a_crash_is_not_an_epistemic_abstention():
    """Both abstain, and on Tier D both would be graded correct. Only one of
    them is a judgment about the evidence."""
    decision = decide(inputs(None, has_draft=False, error="synthesizer returned no text"))
    assert (decision.terminal, decision.rule) == (ABSTAIN, RULE_SYSTEM_FAILURE)
    assert not decision.epistemic


def test_an_error_with_a_draft_still_goes_to_the_threshold():
    """A critic failure leaves a draft whose claims are all UNSUPPORTED; the
    score reflects that. It is not a system failure in the rule sense."""
    decision = decide(inputs(0.9, error="critic unavailable"))
    assert decision.rule == RULE_THRESHOLD


def test_no_evidence_abstains():
    decision = decide(inputs(0.9, has_evidence=False))
    assert (decision.terminal, decision.rule) == (ABSTAIN, RULE_NO_EVIDENCE)


def test_nothing_scorable_abstains():
    decision = decide(inputs(None))
    assert decision.rule == RULE_NO_EVIDENCE


def test_premise_outranks_system_failure():
    """The refuted premise is the more informative reason."""
    decision = decide(inputs(None, premise_failed=True, has_draft=False, error="x"))
    assert decision.rule == RULE_PREMISE


# --- static policy (B2) ----------------------------------------------------


def test_static_policy_answers_and_keeps_the_counterfactual():
    decision = static_decision(inputs(0.10))
    assert (decision.terminal, decision.rule) == (ANSWER, RULE_STATIC)
    assert decision.counterfactual["terminal"] == ABSTAIN
    assert decision.counterfactual["score"] == 0.10


def test_static_policy_answers_even_a_refuted_premise():
    """That is what 'abstention disabled' means; B2 measures exactly this."""
    assert static_decision(inputs(0.9, premise_failed=True)).terminal == ANSWER


# --- thresholds ------------------------------------------------------------


@pytest.mark.parametrize("answer,partial", [(0.4, 0.6), (1.2, 0.5), (0.5, -0.1)])
def test_invalid_thresholds_are_rejected(answer, partial):
    with pytest.raises(ValueError):
        Thresholds(answer=answer, partial=partial)


def test_thresholds_load_from_config_with_their_source():
    from src.config import Config

    t = Thresholds.from_config(Config({"sufficiency": {"thresholds": {
        "tau_answer": 0.7, "tau_partial": 0.3, "source": "C33 calibration split"}}}))
    assert (t.answer, t.partial, t.source) == (0.7, 0.3, "C33 calibration split")


def test_the_real_config_thresholds_are_valid():
    from src.config import get_config

    Thresholds.from_config(get_config(reload=True))


# --- the record ------------------------------------------------------------


def test_rationale_states_the_numbers_that_decided():
    """A template, not an LLM: the explanation cannot disagree with its cause."""
    decision = decide(inputs(0.52, resolved=["a"], unresolved=["what b is"]))
    assert "0.52" in decision.rationale
    assert "0.40" in decision.rationale and "0.65" in decision.rationale
    assert "1 of 2" in decision.rationale
    assert "what b is" in decision.rationale


def test_margin_is_the_distance_to_the_nearest_threshold():
    assert decide(inputs(0.62)).margin == pytest.approx(0.03)
    assert decide(inputs(0.95)).margin == pytest.approx(0.30)


def test_uncalibrated_decisions_say_so():
    decision = decide(inputs(0.9, calibrated=False, scorer="heuristic"))
    assert not decision.calibrated
    assert any("UNCALIBRATED" in n for n in decision.notes)
    assert decision.to_dict()["calibrated"] is False


def test_record_is_serialisable():
    import json

    json.dumps(static_decision(inputs(0.3)).to_dict())


# --- gaps ------------------------------------------------------------------


def test_gap_format_matches_the_planner():
    from src.nodes.planner import _gap
    from src.state import new_sub_question

    sq = new_sub_question("What rank does LoRA use by default.")
    assert gap_for(sq["text"]) == _gap(sq)


def test_unresolved_sub_questions_get_gaps_without_duplicating_recorded_ones():
    """The adaptive retriever abandons without writing a gap; status is the
    authoritative record of what was not established."""
    recorded = ["could not establish: claim X", gap_for("sub-question two")]
    gaps = enumerate_gaps(recorded, ["sub-question two", "sub-question three"])
    assert gaps == [
        "could not establish: claim X",
        gap_for("sub-question two"),
        gap_for("sub-question three"),
    ]
