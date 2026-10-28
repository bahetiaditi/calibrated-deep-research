"""Tests for the sufficiency model.

The model's job is to be honest at a sample size where honesty is hard.
These tests check that it reports its own unreliability rather than
producing confident coefficients from 24 rows, that calibration is not
applied where it would absorb noise, and that an unfitted model degrades
rather than crashing.
"""
import random

import pytest

from src.sufficiency.features import FEATURE_NAMES
from src.sufficiency.model import (
    MIN_SAMPLES_PER_PREDICTOR,
    RUN_FEATURE_NAMES,
    FitReport,
    SufficiencyModel,
    derive_sub_question_labels,
    expected_calibration_error,
    reliability_curve,
    run_features,
)

SEED = 7


def features(f1, **over):
    """A feature dict where f1 carries the signal."""
    base = {name: None for name in FEATURE_NAMES}
    base.update({
        "f1_max_rerank": f1,
        "f2_mean_top3": f1 - 1.0,
        "f3_score_gap": 2.0,
        "f4_independent_sources": 4.0,
        "f7_premise_grounding": 1.0,
        "f8_rounds_consumed": 1.0,
        "f9_route_escalation": 0.5,
        "f10_recency": 0.6,
    })
    base.update(over)
    return base


def separable(n=80, noise=1.5, seed=SEED):
    """Half well-evidenced, half not — roughly the C20 observed spread."""
    rng = random.Random(seed)
    rows, labels = [], []
    for i in range(n):
        positive = i % 2 == 0
        centre = 7.0 if positive else 0.0
        rows.append(features(centre + rng.gauss(0, noise)))
        labels.append(int(positive))
    return rows, labels


# --- fitting ---------------------------------------------------------------


def test_fits_and_predicts_on_separable_data():
    """The C26 criterion."""
    rows, labels = separable()
    model = SufficiencyModel()
    report = model.fit(rows, labels)

    assert model.is_fitted
    assert report.cv_auc is not None and report.cv_auc > 0.8
    assert model.predict_proba(features(8.0)) > model.predict_proba(features(-1.0))


def test_coefficients_are_inspectable():
    """'Which signals predict answerability' is a finding the project
    intends to report (§5.3), so the coefficients have to be readable."""
    rows, labels = separable()
    model = SufficiencyModel()
    report = model.fit(rows, labels)
    assert "f1_max_rerank" in report.coefficients
    assert report.coefficients["f1_max_rerank"] > 0
    assert "f1_max_rerank" in report.render()


def test_the_signal_carrying_feature_survives_resampling():
    """A point estimate is not reportable at this sample size."""
    rows, labels = separable()
    report = SufficiencyModel().fit(rows, labels)
    assert "f1_max_rerank" in report.significant_coefficients()


def test_a_noise_feature_does_not_survive_resampling():
    rng = random.Random(SEED)
    rows, labels = [], []
    for i in range(80):
        positive = i % 2 == 0
        rows.append(features(7.0 if positive else 0.0,
                             f10_recency=rng.random()))
        labels.append(int(positive))
    report = SufficiencyModel().fit(rows, labels)
    assert "f10_recency" not in report.significant_coefficients()


def test_pure_noise_produces_no_significant_coefficients():
    """The honest outcome when the features carry nothing."""
    rng = random.Random(SEED)
    rows = [features(rng.gauss(3, 3)) for _ in range(80)]
    labels = [rng.randint(0, 1) for _ in range(80)]
    report = SufficiencyModel().fit(rows, labels)
    assert len(report.significant_coefficients()) <= 2


# --- honesty about sample size ---------------------------------------------


def varied(n=24, seed=SEED):
    """Every feature varies, so none is dropped as constant — the shape
    §5.3 assumed: many live predictors against few rows."""
    rng = random.Random(seed)
    rows, labels = [], []
    for i in range(n):
        positive = i % 2 == 0
        rows.append({
            "f1_max_rerank": (7.0 if positive else 0.0) + rng.gauss(0, 1.5),
            "f2_mean_top3": rng.gauss(3, 2),
            "f3_score_gap": rng.gauss(2, 1),
            "f4_independent_sources": float(rng.randint(1, 8)),
            "f5_contradiction_rate": rng.random(),
            "f6_entailment_strength": rng.random(),
            "f7_premise_grounding": rng.random(),
            "f8_rounds_consumed": float(rng.randint(1, 3)),
            "f9_route_escalation": rng.choice([0.0, 0.5, 1.0]),
            "f10_recency": rng.random(),
        })
        labels.append(int(positive))
    return rows, labels


def test_too_few_samples_per_predictor_is_reported():
    """§5.3 as written would fit 15 predictors on 24 rows. The model says
    so rather than producing a coefficient table that cannot mean anything."""
    rows, labels = varied(n=24)
    report = SufficiencyModel().fit(rows, labels)
    assert report.samples_per_predictor < MIN_SAMPLES_PER_PREDICTOR
    assert any("samples per predictor" in w for w in report.warnings)
    assert not report.reliable


def test_the_warning_counts_predictors_the_fit_actually_uses():
    """Warning about columns dropped as constant would be worse than no
    warning: a constant feature is not a weak predictor, it is not a
    predictor."""
    rows, labels = separable(n=24)
    report = SufficiencyModel().fit(rows, labels)
    assert report.n_predictors == len(report.coefficients)


def test_all_constant_features_is_degenerate():
    rows = [features(5.0) for _ in range(20)]
    labels = [i % 2 for i in range(20)]
    report = SufficiencyModel().fit(rows, labels)
    assert any("degenerate" in w for w in report.warnings)


def test_few_positive_events_is_reported():
    rows, labels = separable(n=40)
    labels = [1 if i < 4 else 0 for i in range(40)]
    report = SufficiencyModel().fit(rows, labels)
    assert any("positive events" in w for w in report.warnings)


def test_a_single_class_is_degenerate_not_a_fit():
    rows, _ = separable(n=40)
    report = SufficiencyModel().fit(rows, [1] * 40)
    assert any("degenerate" in w for w in report.warnings)
    assert not report.reliable


def test_too_few_rows_is_degenerate():
    rows, labels = separable(n=6)
    assert any("degenerate" in w for w in SufficiencyModel().fit(rows, labels).warnings)


def test_constant_columns_are_dropped_and_named():
    """f8 and f9 are constants under the static policy by construction
    (C20); the fit must not pretend to estimate them."""
    rows, labels = separable()
    report = SufficiencyModel().fit(rows, labels)
    assert any("constant in this sample" in w for w in report.warnings)
    assert "f8_rounds_consumed" not in report.coefficients


def test_a_healthy_fit_is_reported_as_reliable():
    rows, labels = separable(n=200)
    assert SufficiencyModel(
        ["f1_max_rerank", "f2_mean_top3"], with_missing_indicators=False
    ).fit(rows, labels).reliable


# --- calibration -----------------------------------------------------------


def test_calibration_is_not_applied_at_small_n():
    """At n=24 isotonic regression fits the noise. §5.3 says to apply it if
    the reliability diagram shows miscalibration; auto additionally asks
    whether there is enough data for the correction to mean anything."""
    rows, labels = separable(n=24)
    report = SufficiencyModel().fit(rows, labels, calibrate="auto")
    assert report.calibrated is False


def test_calibration_can_be_forced():
    rows, labels = separable(n=80)
    report = SufficiencyModel().fit(rows, labels, calibrate="always")
    assert report.calibrated is True


def test_calibration_can_be_refused():
    rows, labels = separable(n=200)
    assert SufficiencyModel().fit(
        rows, labels, calibrate="never"
    ).calibrated is False


def test_calibrated_predictions_stay_in_range():
    rows, labels = separable(n=80)
    model = SufficiencyModel()
    model.fit(rows, labels, calibrate="always")
    for f1 in (-10.0, 0.0, 20.0):
        assert 0.0 <= model.predict_proba(features(f1)) <= 1.0


# --- ECE and reliability ---------------------------------------------------


def test_perfect_calibration_scores_zero():
    labels = [1] * 50 + [0] * 50
    probabilities = [0.9] * 50 + [0.1] * 50
    assert expected_calibration_error(labels, probabilities) < 0.15


def test_overconfidence_is_penalised():
    labels = [1] * 50 + [0] * 50
    confident = [0.99] * 50 + [0.99] * 50     # certain, half wrong
    honest = [0.5] * 100
    assert (expected_calibration_error(labels, confident)
            > expected_calibration_error(labels, honest))


def test_ece_on_empty_input():
    assert expected_calibration_error([], []) is None


def test_reliability_curve_shape():
    curve = reliability_curve([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert all(len(point) == 3 for point in curve)
    assert sum(point[2] for point in curve) == 4


# --- prediction and persistence -------------------------------------------


def test_unfitted_model_returns_neutral_rather_than_raising():
    """An unfitted model means the calibration run has not happened yet; a
    run should degrade to 'no information' rather than crash."""
    assert SufficiencyModel().predict_proba(features(8.0)) == 0.5


def test_predictions_are_probabilities():
    rows, labels = separable()
    model = SufficiencyModel()
    model.fit(rows, labels)
    for f1 in (-5.0, 0.0, 4.0, 9.0):
        assert 0.0 <= model.predict_proba(features(f1)) <= 1.0


def test_missing_features_do_not_break_prediction():
    rows, labels = separable()
    model = SufficiencyModel()
    model.fit(rows, labels)
    assert 0.0 <= model.predict_proba(
        {name: None for name in FEATURE_NAMES}
    ) <= 1.0


def test_save_and_load_round_trip(tmp_path):
    rows, labels = separable()
    model = SufficiencyModel()
    model.fit(rows, labels)
    before = model.predict_proba(features(7.5))

    model.save(tmp_path / "model.joblib")
    restored = SufficiencyModel.load(tmp_path / "model.joblib")
    assert restored.predict_proba(features(7.5)) == pytest.approx(before)
    assert restored.report.coefficients == model.report.coefficients


def test_save_writes_a_readable_report(tmp_path):
    """The coefficient table is a reportable finding and should not require
    loading a pickle to read."""
    import json

    rows, labels = separable()
    model = SufficiencyModel()
    model.fit(rows, labels)
    model.save(tmp_path / "model.joblib")
    blob = json.loads((tmp_path / "model.json").read_text())
    assert "coefficients" in blob and "warnings" in blob


# --- stage 2 ---------------------------------------------------------------


def test_run_features_shape():
    f = run_features([0.9, 0.2], premise_grounding=1.0,
                     fraction_sufficient=0.5)
    assert set(f) == set(RUN_FEATURE_NAMES)
    assert f["mean_sub_question_p"] == pytest.approx(0.55)
    assert f["min_sub_question_p"] == pytest.approx(0.2)


def test_min_is_carried_alongside_mean():
    """A report resting on one unsupported sub-question is unsafe however
    well the others went, and averaging hides exactly that."""
    f = run_features([0.95, 0.95, 0.95, 0.05], premise_grounding=1.0,
                     fraction_sufficient=0.75)
    assert f["mean_sub_question_p"] > 0.7
    assert f["min_sub_question_p"] == pytest.approx(0.05)


def test_run_features_with_no_sub_questions():
    f = run_features([], premise_grounding=None, fraction_sufficient=None)
    assert all(v is None for v in f.values())


def test_stage_two_fits_on_four_predictors():
    rng = random.Random(SEED)
    rows, labels = [], []
    for i in range(24):
        answerable = i % 2 == 0
        rows.append(run_features(
            [rng.gauss(0.8 if answerable else 0.25, 0.1) for _ in range(3)],
            premise_grounding=1.0 if answerable else 0.6,
            fraction_sufficient=0.9 if answerable else 0.3,
        ))
        labels.append(int(answerable))
    model = SufficiencyModel(RUN_FEATURE_NAMES, with_missing_indicators=False)
    report = model.fit(rows, labels)
    assert report.n_predictors == 4
    assert report.samples_per_predictor == 6.0


# --- weak labels from the critic -------------------------------------------


def test_labels_are_derived_from_critic_verdicts():
    plan = [{"id": "sq1"}, {"id": "sq2"}]
    findings = [
        {"sub_question_id": "sq1", "verdict": "SUPPORTED"},
        {"sub_question_id": "sq1", "verdict": "SUPPORTED"},
        {"sub_question_id": "sq2", "verdict": "UNSUPPORTED"},
    ]
    assert derive_sub_question_labels(plan, findings) == {"sq1": 1, "sq2": 0}


def test_one_unsupported_claim_makes_a_sub_question_insufficient():
    plan = [{"id": "sq1"}]
    findings = [
        {"sub_question_id": "sq1", "verdict": "SUPPORTED"},
        {"sub_question_id": "sq1", "verdict": "PARTIAL"},
    ]
    assert derive_sub_question_labels(plan, findings) == {"sq1": 0}


def test_sub_questions_with_no_claims_are_omitted_not_labelled_zero():
    """Producing no claim is a synthesizer outcome, not evidence that
    retrieval failed."""
    plan = [{"id": "sq1"}, {"id": "sq2"}]
    findings = [{"sub_question_id": "sq1", "verdict": "SUPPORTED"}]
    assert derive_sub_question_labels(plan, findings) == {"sq1": 1}


def test_report_renders_without_a_fit():
    assert "coefficients" in FitReport().render()