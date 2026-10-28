"""The decider — D5, the terminal decision (C27).

    answer / partial answer with declared gaps / abstain

§2.2 calls this "the highest form of agency in the system: deciding not to
act". Its static counterpart is "always answer", which is baseline B2.

Where it sits, and what that costs
----------------------------------
After the critic and its router, not before synthesis. The original
component map scored sufficiency before writing, but the decision should be
able to use what verification found: a sub-question that retrieval called
sufficient can still have every claim drawing on it fail the critic, and the
router then marks it unresolved. Deciding before that would ignore the one
check that looks at whether the evidence supports what was said.

The cost is real and stated: an ABSTAIN arrives after the run has paid for a
draft and its critique, roughly three LLM calls it then withholds. If C33's
data shows the critic never changes a decision, an early exit before
synthesis becomes worth adding — and that is measurable, because the
decision record keeps the per-sub-question verdicts.

What the node changes, not just records
---------------------------------------
  ABSTAIN  the draft is withheld: moved into the decision record and cleared,
           so nothing downstream can show it. Kept rather than discarded
           because C38 needs it — "was the withheld report actually right?"
           is how a false abstention is diagnosed.
  PARTIAL  an answer-mode draft is withheld and cleared, which sends the run
           back through the synthesizer in partial mode (C18's second prompt)
           and the critic. A report written to answer the whole question
           reads as an answer even after the gaps are listed beneath it.
           Skipped only when the remaining budget cannot pay for the rewrite
           and its critique; the record then says so, for C28 to handle.
  ANSWER   the draft stands.

The decision record is the reason this node exists for the eval, not only
for the user: the continuous score (C37 sweeps it for the risk-coverage
curve), the rule that fired (C37 separates system failures from
abstentions), and per-sub-question features and verdicts **as they were
when the decision was made**. C33 fits stage 1 on those. Reading them from
the final state instead would mean that on every PARTIAL run the
unresolved sub-questions — the negatives — had vanished, since the partial
rewrite makes no claims about them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.config import Config, get_config
from src.state import ResearchState
from src.sufficiency.policy import (
    ABSTAIN,
    ANSWER,
    PARTIAL,
    TERMINALS,
    Decision,
    DecisionInputs,
    decide,
    enumerate_gaps,
    static_decision,
)
from src.sufficiency.scorer import RunScore, Scorer, get_scorer

log = logging.getLogger(__name__)

REWRITE_REQUESTED = "requested"
REWRITE_UNAFFORDABLE = "unaffordable"
REWRITE_DISABLED = "disabled"
REWRITE_NOT_NEEDED = "not_needed"


@dataclass
class DeciderResult:
    decision: Decision
    run: RunScore
    update: dict[str, Any] = field(default_factory=dict)


class Decider:
    def __init__(
        self,
        config: Config | None = None,
        *,
        scorer: Scorer | None = None,
        tracer: Any = None,
        budget: Any = None,
    ) -> None:
        self.cfg = config or get_config()
        self._scorer = scorer
        self.tracer = tracer
        self.budget = budget
        self.adaptive = bool(self.cfg.get("agency.d5_terminal_decision", True))
        self.abstain_on_failed_premise = bool(
            self.cfg.get("sufficiency.hard_rules.abstain_on_failed_premise", True)
        )
        self.rewrite_enabled = bool(
            self.cfg.get("sufficiency.partial_rewrite.enabled", True)
        )
        self.rewrite_llm_calls = int(
            self.cfg.get("sufficiency.partial_rewrite.llm_calls", 3)
        )

    @property
    def scorer(self) -> Scorer:
        # Lazy: loading a model from disk is not something a test that
        # passes its own scorer should pay for.
        if self._scorer is None:
            self._scorer = get_scorer(self.cfg)
        return self._scorer

    # -- main entry point ---------------------------------------------------

    def decide(self, state: ResearchState) -> DeciderResult:
        run = self.scorer.score(state)
        premise = state.get("premise_check") or {}
        draft = (state.get("draft") or "").strip()

        inputs = DecisionInputs(
            score=run.value,
            thresholds=run.thresholds,
            premise_failed=bool(premise.get("failed")),
            premise_rationale=str(premise.get("rationale") or ""),
            premise_correction=str(premise.get("correction") or ""),
            abstain_on_failed_premise=self.abstain_on_failed_premise,
            has_evidence=bool(state.get("evidence")),
            has_draft=bool(draft),
            error=state.get("error"),
            scorer=run.kind,
            calibrated=run.calibrated,
            resolved=[s.text for s in run.resolved()],
            unresolved=[s.text for s in run.unresolved()],
            background=len(run.background()),
        )
        decision = decide(inputs) if self.adaptive else static_decision(inputs)
        assert decision.terminal in TERMINALS

        gaps = enumerate_gaps(
            state.get("unresolved_gaps") or [],
            [s.text for s in run.unresolved()],
        )
        update, record_extra = self._apply(state, decision, draft)

        record = {
            **decision.to_dict(),
            "adaptive": self.adaptive,
            "gaps": gaps,
            "sub_questions": [s.to_dict() for s in run.sub_questions],
            "run_features": run.run_features,
            "scorer_notes": run.notes,
            "error": state.get("error"),
            **record_extra,
        }
        update.update({
            "terminal_decision": decision.terminal,
            "decision_rationale": decision.rationale,
            "decision": record,
            "unresolved_gaps": gaps,     # merge_gaps reducer: appends the new ones
        })
        plan = self._plan_with_probabilities(state, run)
        if plan is not None:
            update["plan"] = plan

        self._trace(decision, run, gaps, record_extra)
        return DeciderResult(decision=decision, run=run, update=update)

    # -- effects ------------------------------------------------------------

    def _apply(
        self, state: ResearchState, decision: Decision, draft: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        update: dict[str, Any] = {}
        extra: dict[str, Any] = {
            "withheld_draft": "",
            "withheld_draft_mode": "",
            "partial_rewrite": REWRITE_NOT_NEEDED,
        }

        def withhold() -> None:
            extra["withheld_draft"] = draft
            extra["withheld_draft_mode"] = state.get("draft_mode") or ""
            update["draft"] = ""

        if decision.terminal == ABSTAIN and draft:
            withhold()

        elif decision.terminal == PARTIAL:
            already_partial = state.get("draft_mode") == "partial"
            if not draft or already_partial:
                # No draft: the controller sends the run to the synthesizer,
                # which writes in partial mode because terminal_decision is
                # PARTIAL. Already partial: nothing to redo.
                extra["partial_rewrite"] = (
                    REWRITE_REQUESTED if not draft and not state.get("error")
                    else REWRITE_NOT_NEEDED
                )
            elif not self.rewrite_enabled:
                extra["partial_rewrite"] = REWRITE_DISABLED
            elif not self._can_afford_rewrite():
                # Keep the answer-mode draft rather than leave nothing. The
                # finalizer (C28) must then present it as partial: the gaps
                # are in the record, and the claims routed UNRESOLVED or
                # DROP_CLAIM are in the findings.
                extra["partial_rewrite"] = REWRITE_UNAFFORDABLE
            else:
                withhold()
                extra["partial_rewrite"] = REWRITE_REQUESTED

        return update, extra

    def _can_afford_rewrite(self) -> bool:
        if self.budget is None:
            return True
        return self.budget.remaining("llm") >= self.rewrite_llm_calls

    def _plan_with_probabilities(
        self, state: ResearchState, run: RunScore
    ) -> list[dict[str, Any]] | None:
        """Fill `SubQuestion.sufficiency` with the stage-1 probability.

        Only a calibrated probability goes there. The field has been in the
        state since C14 and `extract_run_features` averages it; writing the
        heuristic's 1/0 resolution bits into it would let a later reader take
        them for probabilities.
        """
        if not run.calibrated:
            return None
        p_by_id = {s.id: s.p for s in run.sub_questions if s.p is not None}
        if not p_by_id:
            return None
        return [
            {**sq, "sufficiency": p_by_id[sq["id"]]} if sq["id"] in p_by_id else sq
            for sq in (state.get("plan") or [])
        ]

    # -- tracing ------------------------------------------------------------

    def _trace(
        self,
        decision: Decision,
        run: RunScore,
        gaps: list[str],
        extra: dict[str, Any],
    ) -> None:
        if self.tracer is None:
            return
        from src.tracing.tracer import DecisionSurface

        self.tracer.decision(
            DecisionSurface.D5_TERMINAL_DECISION,
            chosen=decision.terminal,
            alternatives=list(TERMINALS),
            rationale=decision.rationale,
            inputs={
                "rule": decision.rule,
                "score": decision.score,
                "thresholds": decision.thresholds,
                "margin": decision.margin,
                "scorer": decision.scorer,
                "calibrated": decision.calibrated,
                "resolved": len(run.resolved()),
                "unresolved": len(run.unresolved()),
                "gaps": len(gaps),
                "partial_rewrite": extra.get("partial_rewrite"),
                "counterfactual": (decision.counterfactual or {}).get("terminal"),
            },
            was_adaptive=self.adaptive,
        )


def decider_node(decider: Decider):
    """LangGraph node: finished research -> terminal decision."""

    def node(state: dict) -> dict:
        return decider.decide(state).update  # type: ignore[arg-type]

    return node
