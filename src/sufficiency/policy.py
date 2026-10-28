"""The terminal decision policy — D5 (§5.4).

A pure function from a score and a handful of facts to ANSWER / PARTIAL /
ABSTAIN. No state, no I/O, no model calls: everything the decision rests on
arrives as an argument, so it is exhaustively testable and the rationale can
state exactly what produced it.

Why the rationale is a template, not an LLM call
------------------------------------------------
§8 lists "rationale generation" under C27, and the obvious build is to ask
the model to explain the decision. That would produce a fluent account of a
decision the model did not make — the score and thresholds made it — and an
explanation generated after the fact can disagree with its own cause without
anyone noticing. The template states the numbers that actually decided. It
is also free, which matters on the ABSTAIN path where the run has already
spent its budget finding out it cannot answer.

The rules, in order
-------------------
    1. premise_refuted   ABSTAIN. Categorical, bypasses tau (§5.4).
    2. system_failure    ABSTAIN. No draft and an error: nothing was written
                         because something broke, not because the evidence
                         was thin.
    3. no_evidence       ABSTAIN. Nothing was retrieved, or nothing is left
                         to score.
    4. threshold         score >= tau_answer -> ANSWER; >= tau_partial ->
                         PARTIAL; else ABSTAIN.

Rule 2 exists for the metrics, not the user. Both it and rule 4 produce an
abstention, and on a Tier D question both would be graded correct. A crash
is not a correct abstention, and C37 must be able to tell them apart without
reading traces — so the rule that fired is part of the decision.

Static counterpart (D5 disabled — baseline B2): always ANSWER. The adaptive
decision is still computed and recorded as a counterfactual, so the B2 run
carries the score it would have been thresholded on.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

ANSWER = "ANSWER"
PARTIAL = "PARTIAL"
ABSTAIN = "ABSTAIN"
TERMINALS = (ANSWER, PARTIAL, ABSTAIN)

RULE_PREMISE = "premise_refuted"
RULE_SYSTEM_FAILURE = "system_failure"
RULE_NO_EVIDENCE = "no_evidence"
RULE_THRESHOLD = "threshold"
RULE_STATIC = "static_always_answer"

# Rules whose abstentions are judgments about the evidence. A system_failure
# abstention is not in this set and must never be scored as a correct
# abstention, whatever tier the question is in.
EPISTEMIC_RULES = frozenset({RULE_PREMISE, RULE_NO_EVIDENCE, RULE_THRESHOLD})

GAP_PREFIX = "could not establish: "
MAX_GAP_TEXT = 160


@dataclass(frozen=True)
class Thresholds:
    """tau_answer and tau_partial, with where they came from.

    `source` is recorded in every decision. A placeholder threshold from
    config.yaml and one selected at C33 against a target risk are different
    experiments, and a decision record that cannot say which it used cannot
    be reported.
    """

    answer: float
    partial: float
    source: str = "config"

    def __post_init__(self) -> None:
        if not (0.0 <= self.partial <= self.answer <= 1.0):
            raise ValueError(
                f"thresholds must satisfy 0 <= partial <= answer <= 1, got "
                f"partial={self.partial}, answer={self.answer}"
            )

    @classmethod
    def from_config(cls, cfg: Any) -> "Thresholds":
        return cls(
            answer=float(cfg.get("sufficiency.thresholds.tau_answer")),
            partial=float(cfg.get("sufficiency.thresholds.tau_partial")),
            source=str(cfg.get("sufficiency.thresholds.source", "config")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionInputs:
    score: float | None
    thresholds: Thresholds
    premise_failed: bool = False
    premise_rationale: str = ""
    premise_correction: str = ""
    abstain_on_failed_premise: bool = True
    has_evidence: bool = True
    has_draft: bool = True
    error: str | None = None
    scorer: str = "heuristic"
    calibrated: bool = False
    resolved: Sequence[str] = ()       # CORE sub-question texts
    unresolved: Sequence[str] = ()
    background: int = 0                # background sub-questions, not scored


@dataclass
class Decision:
    terminal: str
    rule: str
    score: float | None
    thresholds: dict[str, Any]
    margin: float | None              # distance to the nearest threshold
    rationale: str
    scorer: str
    calibrated: bool
    counterfactual: dict[str, Any] | None = None   # set under the static policy
    notes: list[str] = field(default_factory=list)

    @property
    def epistemic(self) -> bool:
        return self.rule in EPISTEMIC_RULES

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["epistemic"] = self.epistemic
        return out


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


def decide(inputs: DecisionInputs) -> Decision:
    """The adaptive policy. Rules apply in the order the module lists."""
    t = inputs.thresholds

    if inputs.premise_failed and inputs.abstain_on_failed_premise:
        return _decision(inputs, ABSTAIN, RULE_PREMISE, _premise_rationale(inputs))

    if not inputs.has_draft and inputs.error:
        return _decision(
            inputs, ABSTAIN, RULE_SYSTEM_FAILURE,
            f"Abstained because the run failed before a report could be "
            f"written ({inputs.error[:160]}). This is a system failure, not "
            f"a judgment about the evidence.",
        )

    if not inputs.has_evidence or inputs.score is None:
        why = ("no evidence was retrieved" if not inputs.has_evidence
               else "no sub-question could be scored")
        return _decision(
            inputs, ABSTAIN, RULE_NO_EVIDENCE,
            f"Abstained: {why}, so there is nothing to answer from.",
        )

    if inputs.score >= t.answer:
        terminal = ANSWER
    elif inputs.score >= t.partial:
        terminal = PARTIAL
    else:
        terminal = ABSTAIN
    return _decision(inputs, terminal, RULE_THRESHOLD, _threshold_rationale(
        inputs, terminal
    ))


def static_decision(inputs: DecisionInputs) -> Decision:
    """D5 frozen: always ANSWER — with the adaptive decision kept beside it."""
    counterfactual = decide(inputs)
    decision = _decision(
        inputs, ANSWER, RULE_STATIC,
        "Answered unconditionally: the terminal decision is disabled "
        f"(D5 ablation). The adaptive policy would have chosen "
        f"{counterfactual.terminal} ({counterfactual.rule}).",
    )
    decision.counterfactual = counterfactual.to_dict()
    return decision


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------


def gap_for(text: str) -> str:
    """The gap line for a sub-question. Same format the planner writes."""
    return f"{GAP_PREFIX}{(text or '')[:MAX_GAP_TEXT].rstrip(' .')}"


def enumerate_gaps(
    recorded: Sequence[str], unresolved_texts: Sequence[str]
) -> list[str]:
    """Recorded gaps plus one per unresolved sub-question not yet covered.

    Recorded gaps are not enough on their own. The planner and the critic
    router write gaps, but the adaptive retriever abandons a hopeless
    sub-question without one — so a run could reach PARTIAL with its
    biggest hole unlisted. Status is the authoritative record of what was
    not established; the recorded gaps add the claim-level detail.
    """
    out: list[str] = []
    seen: set[str] = set()
    for gap in list(recorded) + [gap_for(t) for t in unresolved_texts]:
        if gap and gap not in seen:
            seen.add(gap)
            out.append(gap)
    return out


# ---------------------------------------------------------------------------
# Rationale
# ---------------------------------------------------------------------------


def _decision(
    inputs: DecisionInputs, terminal: str, rule: str, rationale: str
) -> Decision:
    t = inputs.thresholds
    margin = None
    if inputs.score is not None and rule == RULE_THRESHOLD:
        margin = round(min(abs(inputs.score - t.answer),
                           abs(inputs.score - t.partial)), 4)
    notes = []
    if not inputs.calibrated:
        notes.append(
            "UNCALIBRATED: decided by the heuristic scorer, not a fitted "
            "model. Not reportable as a result of the abstention layer."
        )
    return Decision(
        terminal=terminal, rule=rule,
        score=None if inputs.score is None else round(float(inputs.score), 4),
        thresholds=t.to_dict(), margin=margin, rationale=rationale,
        scorer=inputs.scorer, calibrated=inputs.calibrated, notes=notes,
    )


def _premise_rationale(inputs: DecisionInputs) -> str:
    text = "Abstained: the question rests on a premise the evidence contradicts"
    if inputs.premise_rationale:
        text += f" ({inputs.premise_rationale.removeprefix('false premise: ')})"
    text += "."
    if inputs.premise_correction:
        text += f" What the evidence says instead: {inputs.premise_correction}"
    return text


def _threshold_rationale(inputs: DecisionInputs, terminal: str) -> str:
    t = inputs.thresholds
    score = f"{inputs.score:.2f}"
    kind = "calibrated" if inputs.calibrated else "uncalibrated heuristic"
    n_res, n_unres = len(inputs.resolved), len(inputs.unresolved)
    parts = f"{n_res} of {n_res + n_unres} core sub-question(s) resolved"
    if inputs.background:
        parts += f", {inputs.background} background not scored"

    if terminal == ANSWER:
        head = f"Answered: score {score} >= tau_answer {t.answer:.2f}"
    elif terminal == PARTIAL:
        head = (f"Partial answer: score {score} is between tau_partial "
                f"{t.partial:.2f} and tau_answer {t.answer:.2f}")
    else:
        head = f"Abstained: score {score} < tau_partial {t.partial:.2f}"

    text = f"{head} ({kind}; {parts})."
    if inputs.unresolved and terminal != ANSWER:
        listed = "; ".join(u[:90] for u in list(inputs.unresolved)[:3])
        more = len(inputs.unresolved) - 3
        text += f" Not established: {listed}" + (f" (+{more} more)" if more > 0 else "") + "."
    return text
