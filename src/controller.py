"""The controller — the only component that decides what runs next.

§3.1 calls this the architectural centrepiece, and the reason is specific:
it is what makes the graph a graph rather than a pipeline with branches
sprinkled through it. Every worker node does one job and returns; **routing
lives here and nowhere else**.

Three properties follow from that, and all three are load-bearing:

**Auditable.** Every routing decision is logged with the state that produced
it. C38's MAST annotation replays a failed run and asks why the system did
what it did; that question is answerable only if the answer is in one place.
Routing scattered across conditional edges would make "why did it stop
researching?" a question about four files.

**Deterministic and LLM-free.** The controller makes no model calls. Given a
state it returns the same node every time, so it is testable exhaustively
without an API key, and a routing bug cannot hide behind sampling noise. The
*decisions* are agentic (D1-D5 live in the nodes); the *dispatch* is not, and
conflating the two would make the agency claim unmeasurable.

**Degrades by node availability.** Nodes arrive across Phases 2 and 3, so the
controller knows the full routing table from the start and skips stages whose
node is not yet registered. C19 runs premise_checker → planner → retriever →
synthesizer; adding the critic at C23 changes no routing code, only the
registry. Writing the table once avoids the usual outcome where the routing
grows a branch per commit and nobody can state the whole policy.
"""
from __future__ import annotations

import logging
from typing import Any

from src.config import Config, get_config
from src.state import ResearchState, all_terminal, pending_sub_questions

log = logging.getLogger(__name__)

# Node names. The graph registers a subset; the controller knows them all.
PREMISE_CHECKER = "premise_checker"
PLANNER = "planner"
RETRIEVER = "retriever"
SUFFICIENCY_SCORER = "sufficiency_scorer"   # C20/C26
SYNTHESIZER = "synthesizer"
CRITIC = "critic"                           # C23
DECIDER = "decider"                         # C27
FINALIZER = "finalizer"                     # C28
FINISH = "__end__"

ALL_NODES = (
    PREMISE_CHECKER, PLANNER, RETRIEVER, SUFFICIENCY_SCORER,
    SYNTHESIZER, CRITIC, DECIDER, FINALIZER,
)


class Controller:
    """Pure routing. No LLM calls, no state mutation."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        available: set[str] | None = None,
        budget: Any = None,
        tracer: Any = None,
    ) -> None:
        self.cfg = config or get_config()
        self.available = set(available) if available is not None else set(ALL_NODES)
        self.budget = budget
        self.tracer = tracer
        self.skip_research_on_failed_premise = bool(
            self.cfg.get("agency.skip_research_on_failed_premise", True)
        )
        self.max_rounds = int(
            self.cfg.get("budget.max_rounds_per_subquestion", 3)
        )

    # -- the routing table --------------------------------------------------

    def route(self, state: ResearchState) -> str:
        """Return the next node name, or FINISH."""
        target, reason = self._decide(state)
        self._trace(state, target, reason)
        return target

    def _decide(self, state: ResearchState) -> tuple[str, str]:
        if state.get("error"):
            return self._next_after_research(state, "unrecoverable error")

        # 1. Premise check runs before planning. A failed premise is the
        #    cheapest abstention signal available (§3.3).
        if self._enabled(PREMISE_CHECKER) and not (state.get("premise_check") or {}):
            return PREMISE_CHECKER, "premise check not run"

        premise = state.get("premise_check") or {}
        if (
            premise.get("failed")
            and self.skip_research_on_failed_premise
            and not state.get("plan")
        ):
            # The question rests on something the corpus refutes, and the
            # premise checker already retrieved the correction. Researching
            # it anyway would spend the whole budget elaborating a premise we
            # know to be false, and the decider would abstain regardless
            # (§5.4's hard rule). Skipping straight there is the cheap,
            # correct move — and because the correction is already in state,
            # the abstention can still explain itself.
            return self._next_after_research(state, "premise refuted")

        # 2. Plan.
        if self._enabled(PLANNER) and not state.get("plan"):
            return PLANNER, "plan is empty"

        # 3. Research, while there is budget.
        pending = pending_sub_questions(state)
        if pending and self._enabled(RETRIEVER):
            if self._has_retrieval_budget():
                return RETRIEVER, (
                    f"{len(pending)} sub-question(s) pending, budget remains"
                )
            return self._next_after_research(
                state, "retrieval budget exhausted with work pending"
            )

        # 4. Plan revision (D1). Live from C25; before that the planner is
        #    registered but has no revision mode, so the guard keeps it from
        #    looping on a plan it cannot change.
        if self._should_revise(state):
            return PLANNER, "sub-question insufficient; revising plan (D1)"

        return self._next_after_research(state, "all sub-questions terminal")

    def _next_after_research(
        self, state: ResearchState, reason: str
    ) -> tuple[str, str]:
        """Everything downstream of retrieval, in order, skipping absentees."""
        premise_failed = bool((state.get("premise_check") or {}).get("failed"))

        if self._enabled(SUFFICIENCY_SCORER) and not self._sufficiency_done(state):
            return SUFFICIENCY_SCORER, reason

        # On a refuted premise there is nothing to write. The synthesizer
        # already declines to run on the ABSTAIN path (C18), but routing
        # around it is clearer than relying on the node to no-op.
        if (
            self._enabled(SYNTHESIZER)
            and not state.get("draft")
            and state.get("terminal_decision") != "ABSTAIN"
            and not (premise_failed and self.skip_research_on_failed_premise)
            # No evidence means no report. Beyond being correct, this closes
            # a live loop: the synthesizer returns an empty draft when it has
            # nothing to work with, `not state["draft"]` stays true, and the
            # controller sends it straight back. Found by a test that
            # registered the retriever without a planner.
            and state.get("evidence")
            # And never re-enter after a failed attempt. `draft` stays falsy
            # on failure, so without this the same loop reappears whenever
            # the model returns nothing — at the cost of an LLM call per
            # iteration until the recursion limit fires.
            and not state.get("error")
        ):
            return SYNTHESIZER, reason

        if (
            self._enabled(CRITIC)
            and state.get("draft")
            and not state.get("critic_findings")
        ):
            return CRITIC, "draft written, critic not run"

        if self._enabled(DECIDER) and state.get("terminal_decision") is None:
            return DECIDER, reason

        if self._enabled(FINALIZER) and not state.get("final_report"):
            return FINALIZER, "decision made, formatting output"

        return FINISH, reason

    # -- predicates ---------------------------------------------------------

    def _enabled(self, node: str) -> bool:
        return node in self.available

    def _has_retrieval_budget(self) -> bool:
        if self.budget is None:
            return True
        return self.budget.can_spend("retrieval") and self.budget.can_spend("llm")

    def _sufficiency_done(self, state: ResearchState) -> bool:
        plan = state.get("plan") or []
        return bool(plan) and all(
            sq.get("sufficiency") is not None for sq in plan
        )

    def _should_revise(self, state: ResearchState) -> bool:
        """D1 revision, guarded three ways.

        The ceiling matters: without it a sub-question that cannot be
        satisfied sends the controller back to the planner forever, which is
        §2.4's "the loop counter is the only thing that changes" anti-pattern
        with the counter removed.
        """
        if not self._enabled(PLANNER):
            return False
        if self.budget is not None and not self.budget.can_revise_plan():
            return False
        if self.budget is not None and not self._has_retrieval_budget():
            return False
        if not self.cfg.get("agency.d1_plan_revision", True):
            return False
        return any(
            sq.get("status") == "insufficient"
            and int(sq.get("rounds", 0)) >= 1
            for sq in (state.get("plan") or [])
        )

    # -- tracing ------------------------------------------------------------

    def _trace(self, state: ResearchState, target: str, reason: str) -> None:
        if self.tracer is None:
            return
        self.tracer.route(target, reason=reason, state_summary=self.summary(state))

    def summary(self, state: ResearchState) -> dict[str, Any]:
        """The fields routing actually reads.

        §3.3 asks routing events to carry "the state snapshot that produced
        it". Dumping full state is not viable — evidence passages run to
        kilobytes and a run makes tens of routing decisions — so this is the
        subset `_decide` consults, and nothing else. A trace too large to
        read is a trace nobody reads.
        """
        plan = state.get("plan") or []
        premise = state.get("premise_check") or {}
        summary = {
            "premise_ran": bool(premise),
            "premise_failed": bool(premise.get("failed")),
            "plan_size": len(plan),
            "pending": [sq["id"] for sq in pending_sub_questions(state)],
            "statuses": {sq["id"]: sq.get("status") for sq in plan},
            "plan_revisions": state.get("plan_revisions", 0),
            "evidence": len(state.get("evidence") or []),
            "has_draft": bool(state.get("draft")),
            "critic_run": bool(state.get("critic_findings")),
            "terminal_decision": state.get("terminal_decision"),
            "all_terminal": all_terminal(state),
        }
        if self.budget is not None:
            summary["budget"] = {
                "llm": f"{self.budget.llm_calls_used}/{self.budget.llm_calls_max}",
                "retrieval": (
                    f"{self.budget.retrieval_calls_used}/"
                    f"{self.budget.retrieval_calls_max}"
                ),
            }
        return summary