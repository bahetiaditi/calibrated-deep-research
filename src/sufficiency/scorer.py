"""Turning a finished run into one number D5 can threshold.

Two scorers behind one interface, and a third arrives at C34 (B3: the
model's verbalized confidence) so that the calibration comparison in claim 1
swaps exactly one component.

  CalibratedScorer   the C26 two-stage model. Stage 1 per sub-question,
                     stage 2 over four aggregates. The reportable one.
  HeuristicScorer    fraction of sub-questions resolved. Exists so the
                     graph runs end to end before C33 has fitted anything.

Why the fallback is a different scorer and not the unfitted model
-----------------------------------------------------------------
An unfitted `SufficiencyModel` returns 0.5. Through stage 2 that is 0.5 for
every question, which sits between the placeholder thresholds (0.40, 0.65),
so every run would come out PARTIAL — a plausible-looking decision that
depends on nothing. The heuristic at least depends on the evidence, uses its
own thresholds rather than borrowing ones meant for probabilities, and every
decision it produces says `calibrated: false`.

`sufficiency.scorer: calibrated` refuses to fall back at all. The held-out
test run (C36) should use it, so a missing model file is an error rather
than 24 quietly uncalibrated decisions.

Which sub-questions count
-------------------------
All of them except one merged into a sibling that is still in the plan:
that sub-question's ground is covered by the sibling, and counting it as a
failure would penalise the planner for a correct merge. A sub-question the
retriever or planner abandoned for any other reason COUNTS, as unresolved —
leaving it out would let a question whose hardest part was given up on score
as if that part had never been asked.

The merge is detected from the rationale text the planner writes ("covered
by sqX: ..."). That is fragile and is the only such parse in the module; a
structured `covered_by` field on SubQuestion would be better, and is noted
for C28.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.config import Config, get_config
from src.state import ResearchState, SubQuestion
from src.sufficiency.features import features_for_sub_question
from src.sufficiency.model import (
    RUN_FEATURE_NAMES,
    SufficiencyModel,
    assert_no_label_leakage,
    run_features,
)
from src.sufficiency.policy import Thresholds

log = logging.getLogger(__name__)

COVERED_RE = re.compile(r"covered by (\S+?):")


class ScorerUnavailable(RuntimeError):
    """`sufficiency.scorer: calibrated` was requested and no fitted model exists."""


@dataclass
class SubQuestionScore:
    id: str
    text: str
    status: str
    included: bool          # False only when merged into a live sibling
    kind: str               # core | background (C27.8)
    resolved: bool
    p: float | None         # stage-1 probability; None under the heuristic
    features: dict[str, float | None]
    verdicts: list[str]     # critic verdicts for this sub-question, at decision time
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunScore:
    value: float | None
    thresholds: Thresholds
    kind: str
    calibrated: bool
    sub_questions: list[SubQuestionScore] = field(default_factory=list)
    run_features: dict[str, float | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def included(self) -> list[SubQuestionScore]:
        return [s for s in self.sub_questions if s.included]

    def scored(self) -> list[SubQuestionScore]:
        """What D5 decides on: included CORE sub-questions (C27.8)."""
        return [s for s in self.included() if s.kind == "core"]

    def background(self) -> list[SubQuestionScore]:
        return [s for s in self.included() if s.kind == "background"]

    def resolved(self) -> list[SubQuestionScore]:
        return [s for s in self.scored() if s.resolved]

    def unresolved(self) -> list[SubQuestionScore]:
        return [s for s in self.scored() if not s.resolved]


class Scorer(Protocol):
    kind: str
    calibrated: bool

    def score(self, state: ResearchState) -> RunScore: ...


# ---------------------------------------------------------------------------
# Shared: what each sub-question looks like at decision time
# ---------------------------------------------------------------------------


def sub_question_scores(
    state: ResearchState, *, min_entailment: float = 0.5
) -> list[SubQuestionScore]:
    """Features, verdicts and resolution per sub-question, from final state.

    Features are recomputed from ALL evidence the sub-question gathered,
    not read from `sufficiency_features` — the retriever writes those per
    round from that round's passages only (C17), so they describe the last
    round rather than what the decision rests on.

    `contradictions_scanned` is False because scanning is still not wired
    into the graph (see the C27 log). f5 is therefore None, honestly.

    Resolved means: retrieval called it sufficient AND, if the critic judged
    claims drawing on it, their mean entailment reached `min_entailment`.
    Retrieval alone is not enough — the critic router exists because a
    sub-question can look well-evidenced and still fail verification.
    """
    plan = list(state.get("plan") or [])
    live_ids = {sq["id"] for sq in plan if sq.get("status") != "abandoned"}
    findings = state.get("critic_findings") or []

    out: list[SubQuestionScore] = []
    for sq in plan:
        features = features_for_sub_question(state, sq["id"])
        verdicts = [
            str(f.get("verdict")) for f in findings
            if f.get("sub_question_id") == sq["id"]
        ]
        covered = _covered_by(sq, live_ids)
        f6 = features.get("f6_entailment_strength")
        resolved = sq.get("status") == "sufficient" and (
            f6 is None or f6 >= min_entailment
        )
        note = f"merged into {covered}" if covered else ""
        if sq.get("status") == "sufficient" and not resolved:
            note = f"retrieval sufficient but entailment {f6:.2f} < {min_entailment}"
        out.append(SubQuestionScore(
            id=sq["id"], text=sq.get("text", ""), status=str(sq.get("status")),
            included=covered is None,
            kind="background" if sq.get("kind") == "background" else "core",
            resolved=resolved, p=None,
            features=features, verdicts=verdicts, note=note,
        ))
    return out


def _covered_by(sq: SubQuestion, live_ids: set[str]) -> str | None:
    if sq.get("status") != "abandoned":
        return None
    # The structured field since C28; the rationale parse remains only for
    # plans written before it.
    into = sq.get("covered_by")
    if into:
        return into if into in live_ids else None
    match = COVERED_RE.search(sq.get("rationale") or "")
    if match and match.group(1) in live_ids:
        return match.group(1)
    return None


def _premise_grounding(state: ResearchState) -> float | None:
    premise = state.get("premise_check") or {}
    if not premise.get("ran"):
        return None
    value = premise.get("grounding_score")
    return None if value is None else float(value)


# ---------------------------------------------------------------------------
# Heuristic (pre-C33)
# ---------------------------------------------------------------------------


class HeuristicScorer:
    """Fraction of included sub-questions resolved. Not a probability.

    Its thresholds are its own — ANSWER when all parts are resolved, PARTIAL
    when any is, ABSTAIN when none is — expressed as (1.0, 1/n) so the same
    policy applies. Borrowing tau_answer = 0.65 would be a category error:
    that number is meant for a calibrated probability, and with three
    sub-questions it would ANSWER with one of them unresolved.
    """

    kind = "heuristic"
    calibrated = False

    def __init__(self, config: Config | None = None, *, reason: str = "") -> None:
        self.cfg = config or get_config()
        self.reason = reason
        self.min_entailment = float(
            self.cfg.get("sufficiency.heuristic.min_entailment", 0.5)
        )

    def score(self, state: ResearchState) -> RunScore:
        subs = sub_question_scores(state, min_entailment=self.min_entailment)
        # Core only (C27.8). A background sub-question resolves whatever the
        # question presupposes; counting it let a false-premise question
        # reach PARTIAL on "what is RoPE?" alone, a report that explains
        # RoPE beside the premise it never questioned.
        scored = [s for s in subs if s.included and s.kind == "core"]
        n = len(scored)
        value = (sum(s.resolved for s in scored) / n) if n else None
        thresholds = Thresholds(
            answer=1.0, partial=(1.0 / n) if n else 1.0,
            source="heuristic: all / any / none resolved",
        )
        notes = ["uncalibrated heuristic scorer"]
        if self.reason:
            notes.append(self.reason)
        return RunScore(
            value=value, thresholds=thresholds, kind=self.kind,
            calibrated=False, sub_questions=subs,
            run_features={"fraction_resolved": value,
                          "premise_grounding": _premise_grounding(state)},
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Calibrated (C26 models, fitted at C33)
# ---------------------------------------------------------------------------


class CalibratedScorer:
    kind = "calibrated"
    calibrated = True

    def __init__(
        self,
        stage1: SufficiencyModel,
        stage2: SufficiencyModel,
        thresholds: Thresholds,
        *,
        min_entailment: float = 0.5,
    ) -> None:
        # Refuse a leaky stage 1 at construction, not at prediction: a model
        # fitted on its own label must not produce a single decision.
        assert_no_label_leakage(stage1.feature_names)
        if not (stage1.is_fitted and stage2.is_fitted):
            raise ScorerUnavailable(
                "CalibratedScorer needs both stages fitted; an unfitted model "
                "returns 0.5 for everything"
            )
        if tuple(stage2.feature_names) != tuple(RUN_FEATURE_NAMES):
            raise ValueError(
                f"stage 2 must use {RUN_FEATURE_NAMES}, got {stage2.feature_names}"
            )
        self.stage1 = stage1
        self.stage2 = stage2
        self.thresholds = thresholds
        self.min_entailment = min_entailment

    def score(self, state: ResearchState) -> RunScore:
        subs = sub_question_scores(state, min_entailment=self.min_entailment)
        # Stage-1 probabilities for every included sub-question — they are
        # recorded, and C33 trains on them — but stage 2 aggregates CORE ones
        # only (C27.8); see HeuristicScorer.score.
        for s in subs:
            if s.included:
                s.p = round(self.stage1.predict_proba(s.features), 4)
        included = [s for s in subs if s.included and s.kind == "core"]

        if not included:
            return RunScore(value=None, thresholds=self.thresholds,
                            kind=self.kind, calibrated=True, sub_questions=subs,
                            notes=["no sub-question to score"])

        # fraction_sufficient is retrieval status, not "resolved": stage 2 was
        # specified on status at C26, and redefining an input after the
        # model is fitted would silently change what its coefficient means.
        sufficient = sum(1 for s in included if s.status == "sufficient")
        features = run_features(
            [s.p for s in included if s.p is not None],
            premise_grounding=_premise_grounding(state),
            fraction_sufficient=sufficient / len(included),
        )
        value = self.stage2.predict_proba(features)
        return RunScore(
            value=value, thresholds=self.thresholds, kind=self.kind,
            calibrated=True, sub_questions=subs, run_features=features,
        )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def get_scorer(config: Config | None = None) -> Scorer:
    """auto | calibrated | heuristic, from `sufficiency.scorer`.

    auto uses the fitted models when both files exist and the heuristic
    otherwise. A leaky or malformed model file raises in every mode — falling
    back from a broken model would hide exactly the failure worth seeing.
    """
    cfg = config or get_config()
    mode = str(cfg.get("sufficiency.scorer", "auto"))
    min_entailment = float(cfg.get("sufficiency.heuristic.min_entailment", 0.5))
    if mode == "heuristic":
        return HeuristicScorer(cfg, reason="heuristic requested in config")
    if mode not in ("auto", "calibrated"):
        raise ValueError(f"sufficiency.scorer must be auto|calibrated|heuristic, got {mode!r}")

    paths = [
        Path(cfg.path("sufficiency.stage1_model_path")),
        Path(cfg.path("sufficiency.stage2_model_path")),
    ]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        if mode == "calibrated":
            raise ScorerUnavailable(f"no fitted model at {', '.join(missing)}")
        log.warning("no fitted sufficiency model; using the uncalibrated heuristic")
        return HeuristicScorer(cfg, reason=f"no fitted model at {', '.join(missing)}")

    stage1 = SufficiencyModel.load(paths[0])
    stage2 = SufficiencyModel.load(paths[1])
    return CalibratedScorer(stage1, stage2, Thresholds.from_config(cfg),
                            min_entailment=min_entailment)
