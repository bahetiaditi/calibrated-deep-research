"""The calibrated sufficiency model (§5.3).

This is the machinery; fitting happens at C33 with the calibration split.

A correction to §5.3, made before fitting rather than after
-----------------------------------------------------------
The spec says "logistic regression on the ~10 sub-question features and ~5
run-level aggregates", fit on 24 calibration questions. That does not work,
and it is better to say so now than to report a coefficient table that
cannot mean anything:

    features          n    p    n/p
    f1-f10 + aggregates + missing indicators   24   30   0.8
    f1-f10 + aggregates                        24   15   1.6
    aggregates only                            24    5   4.8

The usual rule for logistic regression is ten *events* per predictor. With
Tier D at ~40% of the benchmark, 24 runs supply roughly ten positive events
— which licenses about one predictor. Fitting fifteen would produce
coefficients that are noise with standard errors attached, and the headline
claim (§1.4, claim 1) would rest on them.

**Two stages instead.** The sub-question features are *per sub-question*, and
24 questions decompose into roughly 70 sub-questions. So:

  stage 1  sub-question model   f1-f10  ->  P(this sub-question is answered)
           n ≈ 70, p = 20 (with indicators). Still tight; regularised hard.

  stage 2  run model            a few aggregates of stage-1 output, plus
           premise grounding and budget  ->  P(the question is answerable)
           n = 24, p = 4.

Stage 2 is deliberately kept to four predictors. Its inputs are summaries of
stage 1 rather than raw features, so the information is not lost — it is
pooled where there is enough data to estimate it.

Labels for stage 1 come free from the critic: a sub-question whose claims
the critic SUPPORTED was sufficiently evidenced. That is a weak label and is
documented as one; it supervises retrieval sufficiency, not truth.

What this module does NOT do
----------------------------
It does not decide anything. `predict_proba` returns a probability; the
threshold that turns it into answer / partial / abstain is C27's, chosen at
C33 against a target risk. Keeping them apart is what lets the same model be
re-thresholded without refitting.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.sufficiency.features import FEATURE_NAMES, to_vector

log = logging.getLogger(__name__)

# Stage 2's predictors. Four, deliberately — see the module docstring.
RUN_FEATURE_NAMES: tuple[str, ...] = (
    "mean_sub_question_p",
    "min_sub_question_p",
    "premise_grounding",
    "fraction_sufficient",
)

# Stage 1's predictors: f1-f10 WITHOUT f6 (C27).
#
# Stage 1 is supervised by the critic: a sub-question is labelled 1 when all
# of its claims were SUPPORTED. f6 is the mean critic verdict strength for
# the same claims, so f6 == 1.0 exactly when the label is 1. Fitting on it is
# fitting on the label. Measured before the fix: on pure-noise features with
# random verdicts, the model with f6 reached CV AUC 0.979 and flagged f6 as
# significant; without it, 0.478 and nothing significant. At C33 the first
# number would have been reported as "stage 1 predicts sufficiency almost
# perfectly". f6 stays in the logged feature vector for analysis — it is a
# legitimate input wherever the label does not come from the critic.
LABEL_LEAKING_FEATURES: tuple[str, ...] = ("f6_entailment_strength",)
STAGE1_FEATURE_NAMES: tuple[str, ...] = tuple(
    name for name in FEATURE_NAMES if name not in LABEL_LEAKING_FEATURES
)


class LabelLeakage(ValueError):
    """A model was built on a feature computed from its own label."""


def assert_no_label_leakage(feature_names: Sequence[str]) -> None:
    leaking = [n for n in feature_names if n in LABEL_LEAKING_FEATURES]
    if leaking:
        raise LabelLeakage(
            f"{', '.join(leaking)} is derived from the critic verdicts that "
            f"label stage 1; a model fitted on it predicts its own label"
        )


# Below this ratio of samples to predictors the fit is reported as unreliable
# rather than quietly used.
MIN_SAMPLES_PER_PREDICTOR = 3.0

# L2 strengths searched. Small C is strong regularisation, which is what a
# short, wide design matrix needs.
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)

BOOTSTRAP_ITERATIONS = 400
CALIBRATION_BINS = 5


@dataclass
class FitReport:
    """What a fit produced, including the reasons to distrust it."""

    n_samples: int = 0
    n_predictors: int = 0
    n_positive: int = 0
    cv_auc: float | None = None
    cv_brier: float | None = None
    ece: float | None = None
    chosen_c: float | None = None
    calibrated: bool = False
    coefficients: dict[str, float] = field(default_factory=dict)
    coefficient_ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def samples_per_predictor(self) -> float:
        return self.n_samples / self.n_predictors if self.n_predictors else 0.0

    @property
    def reliable(self) -> bool:
        return (
            self.samples_per_predictor >= MIN_SAMPLES_PER_PREDICTOR
            and self.n_positive >= 5
            and not any("degenerate" in w for w in self.warnings)
        )

    def significant_coefficients(self) -> dict[str, float]:
        """Coefficients whose bootstrap interval excludes zero.

        The point estimates alone are not reportable at this sample size.
        "Which signals predict answerability" is a finding the project
        intends to make (§5.3), and it has to survive resampling to count.
        """
        out = {}
        for name, value in self.coefficients.items():
            low, high = self.coefficient_ci.get(name, (0.0, 0.0))
            if low > 0.0 or high < 0.0:
                out[name] = value
        return out

    def render(self) -> str:
        lines = [
            f"n={self.n_samples}  predictors={self.n_predictors}  "
            f"positives={self.n_positive}  "
            f"n/p={self.samples_per_predictor:.1f}",
            f"CV AUC={_fmt(self.cv_auc)}  Brier={_fmt(self.cv_brier)}  "
            f"ECE={_fmt(self.ece)}  C={self.chosen_c}  "
            f"calibrated={self.calibrated}",
        ]
        if self.warnings:
            lines.append("warnings:")
            lines.extend(f"  ! {w}" for w in self.warnings)
        significant = self.significant_coefficients()
        lines.append(
            "coefficients (* = bootstrap CI excludes zero):"
            if self.coefficients else "coefficients: (none)"
        )
        for name, value in sorted(
            self.coefficients.items(), key=lambda kv: -abs(kv[1])
        ):
            low, high = self.coefficient_ci.get(name, (float("nan"),) * 2)
            mark = "*" if name in significant else " "
            lines.append(
                f"  {mark} {name:<28}{value:>8.3f}  "
                f"[{low:>7.3f}, {high:>7.3f}]"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SufficiencyModel:
    """Regularised logistic regression with optional calibration.

    Used twice: once over sub-question features, once over run aggregates.
    Both are the same shape of problem — a short design matrix where the
    honest answer may be "this does not predict anything".
    """

    def __init__(
        self,
        feature_names: Sequence[str] = FEATURE_NAMES,
        *,
        with_missing_indicators: bool = True,
        seed: int = 20260805,
    ) -> None:
        self.feature_names = tuple(feature_names)
        self.with_missing_indicators = with_missing_indicators
        self.seed = seed
        self._pipeline: Any = None
        self._calibrator: Any = None
        self.columns: list[str] = []
        self.report = FitReport()

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None

    # -- fitting ------------------------------------------------------------

    def fit(
        self,
        feature_dicts: Sequence[dict[str, float | None]],
        labels: Sequence[int],
        *,
        calibrate: str = "auto",
    ) -> FitReport:
        """Fit on feature dicts and binary labels.

        `calibrate`: "auto" applies isotonic only if the reliability diagram
        shows systematic miscalibration AND there is enough data to fit it;
        "never" and "always" force the choice. §5.3 says to apply it "if the
        reliability diagram shows systematic miscalibration" — at n=24,
        isotonic regression will happily fit the noise, so auto is
        deliberately conservative.
        """
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, columns = self._design_matrix(feature_dicts)
        y = np.asarray(list(labels), dtype=int)
        self.columns = columns

        report = FitReport(
            n_samples=len(y),
            n_predictors=len(columns),
            n_positive=int(y.sum()),
        )

        if len(y) < 8 or report.n_positive < 2 or report.n_positive == len(y):
            report.warnings.append(
                "degenerate sample: too few examples or only one class present"
            )
            self.report = report
            return report

        # Drop columns with no variance FIRST. They cannot contribute, they
        # make the scaler and the bootstrap misbehave, and judging the
        # sample size before dropping them would warn about predictors the
        # fit never uses. A constant feature is not a weak predictor; it is
        # not a predictor. f8 and f9 are constants by construction under the
        # static policy (C20), so this fires on every ablation run.
        keep = [i for i in range(X.shape[1]) if np.std(X[:, i]) > 1e-12]
        dropped = [columns[i] for i in range(X.shape[1]) if i not in keep]
        if dropped:
            report.warnings.append(
                f"constant in this sample, dropped: {', '.join(dropped[:6])}"
                + (f" (+{len(dropped) - 6} more)" if len(dropped) > 6 else "")
            )
        X = X[:, keep]
        self.columns = [columns[i] for i in keep]
        report.n_predictors = len(self.columns)

        if not self.columns:
            report.warnings.append(
                "degenerate sample: every feature is constant"
            )
            self.report = report
            return report

        if report.samples_per_predictor < MIN_SAMPLES_PER_PREDICTOR:
            report.warnings.append(
                f"{report.samples_per_predictor:.1f} samples per predictor "
                f"(want >= {MIN_SAMPLES_PER_PREDICTOR}); coefficients are not "
                f"individually interpretable"
            )
        if report.n_positive < 10:
            report.warnings.append(
                f"only {report.n_positive} positive events; the usual rule is "
                f"10 per predictor"
            )

        chosen_c, cv_auc, cv_brier = self._select_c(X, y)
        report.chosen_c = chosen_c
        report.cv_auc = cv_auc
        report.cv_brier = cv_brier

        self._pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(
                l1_ratio=0.0, C=chosen_c, max_iter=2000,
                random_state=self.seed, class_weight="balanced",
            )),
        ])
        self._pipeline.fit(X, y)

        coefficients = self._pipeline.named_steps["lr"].coef_[0]
        report.coefficients = dict(zip(self.columns, map(float, coefficients)))
        report.coefficient_ci = self._bootstrap_ci(X, y, chosen_c)

        probabilities = self._pipeline.predict_proba(X)[:, 1]
        report.ece = expected_calibration_error(y, probabilities)
        self._calibrator = self._maybe_calibrate(
            X, y, calibrate, report
        )
        report.calibrated = self._calibrator is not None

        self.report = report
        return report

    def _design_matrix(self, feature_dicts) -> tuple[Any, list[str]]:
        import numpy as np

        rows, columns = [], []
        for features in feature_dicts:
            values, columns = to_vector(
                features, names=self.feature_names,
                with_missing_indicators=self.with_missing_indicators,
            )
            rows.append(values)
        return np.asarray(rows, dtype=float), list(columns)

    def _select_c(self, X, y) -> tuple[float, float | None, float | None]:
        """Pick L2 strength by cross-validation.

        Stratified k-fold with k bounded by the smaller class, because at
        this size a fold can otherwise contain one class only.
        """
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        minority = int(min(np.bincount(y)))
        folds = max(2, min(5, minority))
        cv = StratifiedKFold(n_splits=folds, shuffle=True,
                             random_state=self.seed)

        best = (C_GRID[0], -1.0, None)
        for c in C_GRID:
            pipeline = Pipeline([
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(
                    l1_ratio=0.0, C=c, max_iter=2000,
                    random_state=self.seed, class_weight="balanced",
                )),
            ])
            try:
                predicted = cross_val_predict(
                    pipeline, X, y, cv=cv, method="predict_proba"
                )[:, 1]
            except Exception as exc:  # noqa: BLE001
                log.debug("CV failed at C=%s: %s", c, exc)
                continue
            auc = _safe_auc(y, predicted)
            if auc is not None and auc > best[1]:
                best = (c, auc, float(np.mean((predicted - y) ** 2)))

        if best[1] < 0:
            return C_GRID[2], None, None
        return best[0], best[1], best[2]

    def _bootstrap_ci(self, X, y, c: float) -> dict[str, tuple[float, float]]:
        """Percentile intervals on each coefficient.

        At this sample size a point estimate is not reportable on its own.
        "Which signals predict answerability" is a finding the project
        intends to make, and it has to survive resampling to count.
        """
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(self.seed)
        draws: list[Any] = []
        for _ in range(BOOTSTRAP_ITERATIONS):
            index = rng.integers(0, len(y), len(y))
            if len(set(y[index])) < 2:
                continue
            pipeline = Pipeline([
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(
                    l1_ratio=0.0, C=c, max_iter=2000,
                    random_state=self.seed, class_weight="balanced",
                )),
            ])
            try:
                pipeline.fit(X[index], y[index])
            except Exception:  # noqa: BLE001
                continue
            draws.append(pipeline.named_steps["lr"].coef_[0])

        if len(draws) < 50:
            return {name: (float("nan"), float("nan")) for name in self.columns}

        stacked = np.vstack(draws)
        low = np.percentile(stacked, 2.5, axis=0)
        high = np.percentile(stacked, 97.5, axis=0)
        return {
            name: (float(low[i]), float(high[i]))
            for i, name in enumerate(self.columns)
        }

    def _maybe_calibrate(self, X, y, mode: str, report: FitReport) -> Any:
        """Isotonic calibration, applied conservatively.

        §5.3 says to apply it "if the reliability diagram shows systematic
        miscalibration". At n=24 isotonic will happily fit the noise, so
        `auto` additionally requires enough data for the correction to mean
        something.
        """
        if mode == "never":
            return None
        enough_data = len(y) >= 40 and int(y.sum()) >= 10
        miscalibrated = (report.ece or 0.0) > 0.1

        if mode == "always" or (mode == "auto" and enough_data and miscalibrated):
            from sklearn.isotonic import IsotonicRegression

            raw = self._pipeline.predict_proba(X)[:, 1]
            calibrator = IsotonicRegression(out_of_bounds="clip",
                                            y_min=0.0, y_max=1.0)
            calibrator.fit(raw, y)
            return calibrator

        if mode == "auto" and miscalibrated and not enough_data:
            report.warnings.append(
                f"ECE={report.ece:.3f} suggests miscalibration, but n={len(y)} "
                f"is too small to fit isotonic without absorbing noise; "
                f"left uncalibrated"
            )
        return None

    # -- prediction ---------------------------------------------------------

    def predict_proba(self, features: dict[str, float | None]) -> float:
        """P(sufficient) for one feature dict.

        Returns the neutral 0.5 when unfitted rather than raising: an
        unfitted model means the calibration run has not happened yet, and
        a run should degrade to "no information" rather than crash.
        """
        if not self.is_fitted:
            return 0.5
        import numpy as np

        values, columns = to_vector(
            features, names=self.feature_names,
            with_missing_indicators=self.with_missing_indicators,
        )
        lookup = dict(zip(columns, values))
        row = np.asarray([[lookup.get(name, 0.0) for name in self.columns]])
        raw = float(self._pipeline.predict_proba(row)[0, 1])
        if self._calibrator is not None:
            raw = float(self._calibrator.predict([raw])[0])
        return max(0.0, min(1.0, raw))

    def predict_many(
        self, feature_dicts: Sequence[dict[str, float | None]]
    ) -> list[float]:
        return [self.predict_proba(f) for f in feature_dicts]

    def coefficients(self) -> dict[str, float]:
        return dict(self.report.coefficients)

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self._pipeline,
            "calibrator": self._calibrator,
            "columns": self.columns,
            "feature_names": list(self.feature_names),
            "with_missing_indicators": self.with_missing_indicators,
            "seed": self.seed,
            "report": self.report.to_dict(),
        }, path)
        # A human-readable sibling, because the coefficient table is a
        # reportable finding and should not require loading a pickle.
        path.with_suffix(".json").write_text(
            json.dumps(self.report.to_dict(), indent=2, default=str)
        )

    @classmethod
    def load(cls, path: str | Path) -> "SufficiencyModel":
        import joblib

        blob = joblib.load(Path(path))
        model = cls(
            blob["feature_names"],
            with_missing_indicators=blob["with_missing_indicators"],
            seed=blob["seed"],
        )
        model._pipeline = blob["pipeline"]
        model._calibrator = blob["calibrator"]
        model.columns = blob["columns"]
        model.report = FitReport(**blob["report"])
        return model


# ---------------------------------------------------------------------------
# Run-level aggregation (stage 2)
# ---------------------------------------------------------------------------


def run_features(
    sub_question_probabilities: Sequence[float],
    *,
    premise_grounding: float | None,
    fraction_sufficient: float | None,
) -> dict[str, float | None]:
    """Stage 2's four predictors.

    `min` is carried alongside `mean` because a report resting on one
    unsupported sub-question is unsafe however well the others went, and
    averaging hides exactly that (§5.2's r2 rationale).
    """
    probabilities = list(sub_question_probabilities)
    return {
        "mean_sub_question_p": (
            sum(probabilities) / len(probabilities) if probabilities else None
        ),
        "min_sub_question_p": min(probabilities) if probabilities else None,
        "premise_grounding": premise_grounding,
        "fraction_sufficient": fraction_sufficient,
    }


def derive_sub_question_labels(
    plan: Sequence[dict[str, Any]],
    critic_findings: Sequence[dict[str, Any]],
) -> dict[str, int]:
    """Weak labels for stage 1, free from the critic.

    A sub-question is labelled sufficient when the claims drawing on its
    evidence were SUPPORTED. This supervises **retrieval sufficiency**, not
    truth: the critic judges whether a passage supports a claim, not whether
    the claim is correct. That distinction belongs in the writeup — the
    stage-1 model predicts "was enough evidence found", which is what D3 and
    D5 need, and not "is the answer right", which the benchmark's gold
    claims measure separately.

    Sub-questions with no claims at all are omitted rather than labelled 0:
    producing no claim is a synthesizer outcome, not evidence that retrieval
    failed.
    """
    labels: dict[str, int] = {}
    by_sub_question: dict[str, list[str]] = {}
    for finding in critic_findings:
        sqid = finding.get("sub_question_id")
        if sqid:
            by_sub_question.setdefault(sqid, []).append(
                str(finding.get("verdict"))
            )

    for sub_question in plan:
        verdicts = by_sub_question.get(sub_question["id"])
        if not verdicts:
            continue
        labels[sub_question["id"]] = int(
            all(v == "SUPPORTED" for v in verdicts)
        )
    return labels


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    bins: int = CALIBRATION_BINS,
) -> float | None:
    """ECE: mean gap between stated confidence and observed accuracy.

    Few bins on purpose. At n=24 a ten-bin ECE is mostly a report on which
    bins happened to be empty.
    """
    import numpy as np

    y = np.asarray(list(labels), dtype=float)
    p = np.asarray(list(probabilities), dtype=float)
    if len(y) == 0:
        return None

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (p > low) & (p <= high) if low > 0 else (p >= low) & (p <= high)
        if not mask.any():
            continue
        total += (mask.sum() / len(y)) * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def reliability_curve(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    bins: int = CALIBRATION_BINS,
) -> list[tuple[float, float, int]]:
    """(mean predicted, observed rate, count) per bin, for the C39 plot."""
    import numpy as np

    y = np.asarray(list(labels), dtype=float)
    p = np.asarray(list(probabilities), dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (p > low) & (p <= high) if low > 0 else (p >= low) & (p <= high)
        if mask.any():
            out.append((float(p[mask].mean()), float(y[mask].mean()),
                        int(mask.sum())))
    return out


def _safe_auc(y, predicted) -> float | None:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y, predicted))
    except ValueError:
        return None


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"