"""Tests for the run-level scorers.

The C27 criterion — forced feature vectors produce each of the three
decisions — is tested here through the real two-stage model, with
thresholds placed on the fitted model's own scale. Placing them by hand at
the config placeholders would test the placeholders, not the model: a hard
regularised fit compresses its probabilities toward the base rate, and the
synthetic model here never reaches 0.65 at all.
"""
import pytest

from src.config import Config
from src.sufficiency.model import (
    LabelLeakage,
    SufficiencyModel,
    STAGE1_FEATURE_NAMES,
)
from src.sufficiency.features import FEATURE_NAMES
from src.sufficiency.policy import (
    ABSTAIN,
    ANSWER,
    PARTIAL,
    DecisionInputs,
    Thresholds,
    decide,
)
from src.sufficiency.scorer import (
    CalibratedScorer,
    HeuristicScorer,
    ScorerUnavailable,
    get_scorer,
    sub_question_scores,
)
from tests.decision_fixtures import (
    MEDIUM,
    STRONG,
    WEAK,
    decision_state,
    fitted_models,
)

CFG = Config({"sufficiency": {"heuristic": {"min_entailment": 0.5}}})


def calibrated(thresholds=Thresholds(0.65, 0.40)):
    stage1, stage2 = fitted_models()
    return CalibratedScorer(stage1, stage2, thresholds)


# --- the C27 criterion -----------------------------------------------------


def test_forced_feature_vectors_produce_each_decision():
    scorer = calibrated()
    strong = scorer.score(decision_state(STRONG, STRONG)).value
    mixed = scorer.score(decision_state(STRONG, WEAK)).value
    weak = scorer.score(decision_state(WEAK, WEAK)).value
    # The model must order them, or no threshold could separate them.
    assert strong > mixed > weak

    t = Thresholds(answer=(strong + mixed) / 2, partial=(mixed + weak) / 2,
                   source="test: midpoints on the fitted scale")
    got = [decide(DecisionInputs(score=v, thresholds=t, calibrated=True)).terminal
           for v in (strong, mixed, weak)]
    assert got == [ANSWER, PARTIAL, ABSTAIN]


def test_stage1_probability_rises_with_evidence_strength():
    run = calibrated().score(decision_state(STRONG, MEDIUM, WEAK))
    p = [s.p for s in run.sub_questions]
    assert p[0] > p[1] > p[2]


def test_the_min_matters_not_only_the_mean():
    """§5.2's r2: one unsupported sub-question should cost more than its
    share of the average."""
    scorer = calibrated()
    even = scorer.score(decision_state(MEDIUM, MEDIUM)).run_features
    lopsided = scorer.score(decision_state(STRONG, WEAK)).run_features
    assert lopsided["min_sub_question_p"] < even["min_sub_question_p"]


# --- the leakage guard -----------------------------------------------------


def test_stage1_feature_set_excludes_f6():
    assert "f6_entailment_strength" in FEATURE_NAMES
    assert "f6_entailment_strength" not in STAGE1_FEATURE_NAMES


def test_a_stage1_model_fitted_on_f6_is_refused():
    _, stage2 = fitted_models()
    leaky = SufficiencyModel(FEATURE_NAMES)
    leaky._pipeline = object()          # fitted or not, it must never decide
    with pytest.raises(LabelLeakage):
        CalibratedScorer(leaky, stage2, Thresholds(0.65, 0.4))


def test_f6_on_critic_labels_fits_noise_almost_perfectly():
    """The effect that justifies the guard, kept as a regression test: with
    every feature pure noise and labels from random verdicts, f6 alone
    separates the classes — because it is computed from the label."""
    import random

    from src.sufficiency.features import VERDICT_STRENGTH
    from src.sufficiency.model import derive_sub_question_labels

    rng = random.Random(0)
    rows, labels = [], []
    for _ in range(70):
        verdicts = [rng.choice(["SUPPORTED", "SUPPORTED", "PARTIAL", "UNSUPPORTED"])
                    for _ in range(rng.randint(1, 3))]
        row = {n: rng.gauss(0, 1) for n in FEATURE_NAMES}
        row["f6_entailment_strength"] = (
            sum(VERDICT_STRENGTH[v] for v in verdicts) / len(verdicts))
        rows.append(row)
        labels.append(derive_sub_question_labels(
            [{"id": "s"}], [{"sub_question_id": "s", "verdict": v} for v in verdicts]
        )["s"])

    leaky = SufficiencyModel(FEATURE_NAMES).fit(rows, labels)
    clean = SufficiencyModel(STAGE1_FEATURE_NAMES).fit(rows, labels)
    assert leaky.cv_auc > 0.9
    assert clean.cv_auc < 0.7
    assert "f6_entailment_strength" in SufficiencyModel(FEATURE_NAMES).fit(
        rows, labels).coefficients


def test_unfitted_models_are_refused():
    """An unfitted model returns 0.5 for everything, which under the
    placeholder thresholds makes every run PARTIAL."""
    with pytest.raises(ScorerUnavailable):
        CalibratedScorer(SufficiencyModel(STAGE1_FEATURE_NAMES),
                         fitted_models()[1], Thresholds(0.65, 0.4))


# --- the heuristic ---------------------------------------------------------


def test_heuristic_is_the_fraction_resolved_with_its_own_thresholds():
    run = HeuristicScorer(CFG).score(decision_state(STRONG, STRONG, WEAK))
    assert run.value == pytest.approx(2 / 3)
    assert (run.thresholds.answer, run.thresholds.partial) == (1.0, pytest.approx(1 / 3))
    assert not run.calibrated


@pytest.mark.parametrize("strengths,expected", [
    ((STRONG, STRONG), ANSWER),
    ((STRONG, WEAK), PARTIAL),
    ((WEAK, WEAK), ABSTAIN),
])
def test_heuristic_reads_all_any_none(strengths, expected):
    run = HeuristicScorer(CFG).score(decision_state(*strengths))
    decision = decide(DecisionInputs(score=run.value, thresholds=run.thresholds))
    assert decision.terminal == expected


def test_a_sufficient_sub_question_whose_claims_failed_is_not_resolved():
    """The reason D5 runs after the critic."""
    state = decision_state(STRONG, STRONG,
                           verdicts={"sq1": "SUPPORTED", "sq2": "UNSUPPORTED"})
    run = HeuristicScorer(CFG).score(state)
    assert [s.resolved for s in run.sub_questions] == [True, False]
    assert "entailment" in run.sub_questions[1].note


def test_abandoned_sub_question_counts_as_unresolved():
    state = decision_state(STRONG, WEAK, statuses={"sq2": "abandoned"})
    run = HeuristicScorer(CFG).score(state)
    assert run.sub_questions[1].included and not run.sub_questions[1].resolved


def test_a_merged_sub_question_is_not_counted_against_the_run():
    state = decision_state(STRONG, WEAK, statuses={"sq2": "abandoned"})
    state["plan"][1]["rationale"] = "find X | covered by sq1: same ground"
    run = HeuristicScorer(CFG).score(state)
    assert not run.sub_questions[1].included
    assert run.value == 1.0


def test_a_merge_into_a_missing_sibling_still_counts():
    state = decision_state(STRONG, WEAK, statuses={"sq2": "abandoned"})
    state["plan"][1]["rationale"] = "covered by sq9: gone"
    assert HeuristicScorer(CFG).score(state).sub_questions[1].included


def test_features_use_all_evidence_not_the_last_round():
    """The retriever's stored features describe one round. The decision
    rests on everything gathered."""
    state = decision_state(STRONG)
    state["plan"][0]["sufficiency_features"] = {"f1_max_rerank": -5.0}
    sub = sub_question_scores(state)[0]
    assert sub.features["f1_max_rerank"] == pytest.approx(STRONG)


def test_verdicts_are_captured_per_sub_question():
    state = decision_state(STRONG, STRONG,
                           verdicts={"sq1": "SUPPORTED", "sq2": "PARTIAL"})
    subs = sub_question_scores(state)
    assert [s.verdicts for s in subs] == [["SUPPORTED"], ["PARTIAL"]]


# --- selection -------------------------------------------------------------


def _cfg(tmp_path, mode):
    return Config({"sufficiency": {
        "scorer": mode,
        "stage1_model_path": str(tmp_path / "s1.joblib"),
        "stage2_model_path": str(tmp_path / "s2.joblib"),
        "thresholds": {"tau_answer": 0.65, "tau_partial": 0.4},
    }})


def test_auto_falls_back_to_the_heuristic_and_says_why(tmp_path):
    scorer = get_scorer(_cfg(tmp_path, "auto"))
    assert isinstance(scorer, HeuristicScorer)
    assert "no fitted model" in scorer.reason


def test_calibrated_mode_refuses_to_fall_back(tmp_path):
    """The held-out run must not silently become 24 heuristic decisions."""
    with pytest.raises(ScorerUnavailable):
        get_scorer(_cfg(tmp_path, "calibrated"))


def test_auto_loads_fitted_models_from_disk(tmp_path):
    stage1, stage2 = fitted_models()
    stage1.save(tmp_path / "s1.joblib")
    stage2.save(tmp_path / "s2.joblib")
    scorer = get_scorer(_cfg(tmp_path, "auto"))
    assert isinstance(scorer, CalibratedScorer)
    assert scorer.score(decision_state(STRONG)).calibrated


def test_a_leaky_model_on_disk_raises_even_in_auto(tmp_path):
    """Falling back from a broken model would hide the failure worth seeing."""
    import random

    rng = random.Random(1)
    rows = [{n: rng.gauss(0, 1) for n in FEATURE_NAMES} for _ in range(30)]
    leaky = SufficiencyModel(FEATURE_NAMES)
    leaky.fit(rows, [i % 2 for i in range(30)], calibrate="never")
    leaky.save(tmp_path / "s1.joblib")
    fitted_models()[1].save(tmp_path / "s2.joblib")
    with pytest.raises(LabelLeakage):
        get_scorer(_cfg(tmp_path, "auto"))


def test_unknown_mode_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        get_scorer(_cfg(tmp_path, "vibes"))
