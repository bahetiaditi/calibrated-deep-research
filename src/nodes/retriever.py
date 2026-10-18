"""Retriever — static version (C17). D2 and D3 are deliberately OFF.

This is not a placeholder to be replaced. It is the **static counterpart**
that ablations A2 and A3 restore (§6.5), so it has to be a fair opponent:
a deliberately crippled baseline would make the adaptive version look good
for the wrong reason, and the agency claim would be unfalsifiable in exactly
the way §2.1 exists to prevent.

So the static policy is the sensible one a competent engineer would write
without an agent:

  route    fixed, from `agency.static.route_rule`
  query    the sub-question text, verbatim
  depth    one round, fixed top-k, always stop

What C21 adds is *conditioning*: route chosen from what previous attempts
returned, query reformulated after a miss, and a stop/deeper/wider judgment
instead of always stopping. If those three changes do not beat this file, D2
and D3 were decorative and the project reports that.

Bookkeeping that matters either way
-----------------------------------
Even with the decisions frozen, this node records `attempted_routes`,
`attempted_queries`, `rounds` and `evidence_ids`. Two reasons. The trace of
an A2/A3 run must be comparable to a full run, or C35 cannot tell a frozen
decision from a missing one. And C21 inherits this plumbing rather than
reimplementing it, which keeps the only difference between static and
adaptive the decision itself.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence

from src.config import Config, get_config
from src.state import (
    Passage,
    ResearchState,
    Route,
    SubQuestion,
    find_sub_question,
    pending_sub_questions,
)

log = logging.getLogger(__name__)

# Routes this node can execute. `evidence_store` is a search over material
# already gathered; it needs no new fetch and so is handled by the same call.
EXECUTABLE_ROUTES: tuple[Route, ...] = (
    "arxiv_meta", "arxiv_fulltext", "web", "evidence_store",
)


@dataclass
class RetrievalOutcome:
    sub_question_id: str
    route: Route
    query: str
    passages: list[Passage]
    rerank_scores: list[float]
    latency_s: float
    error: str | None = None

    @property
    def found(self) -> bool:
        return bool(self.passages)


class StaticRetriever:
    """Fixed route, verbatim query, one round. The A2/A3 baseline."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        retriever: Any = None,
        reranker: Any = None,
        ingestor: Any = None,
        tracer: Any = None,
        budget: Any = None,
        now: Any = time.time,
    ) -> None:
        self.cfg = config or get_config()
        self._retriever = retriever
        self._reranker = reranker
        self._ingestor = ingestor
        self.tracer = tracer
        self.budget = budget
        self._now = now

        self.top_k = int(self.cfg.get("agency.static.fixed_top_k", 8))
        self.candidates = int(
            self.cfg.get("retrieval.fusion.candidates_per_retriever", 50)
        )
        self.static_route: Route = _coerce_route(
            self.cfg.get("agency.static.route_rule", "arxiv_meta")
        )
        self.use_planner_route = (
            str(self.cfg.get("agency.static.route_rule", "")).lower() == "planner"
        )

    # -- lazy components ----------------------------------------------------

    @property
    def retriever(self) -> Any:
        if self._retriever is None:
            from src.rag.fusion import HybridRetriever

            self._retriever = HybridRetriever(self.cfg, tracer=self.tracer)
        return self._retriever

    @property
    def reranker(self) -> Any:
        if self._reranker is None:
            from src.rag.rerank import Reranker

            self._reranker = Reranker(self.cfg, tracer=self.tracer)
        return self._reranker

    # -- policy (frozen here, live at C21) ----------------------------------

    def choose_route(self, sub_question: SubQuestion) -> Route:
        """D2, static half. Ignores history entirely — that is the point."""
        if self.use_planner_route:
            return _coerce_route(sub_question.get("suggested_route"))
        return self.static_route

    def formulate_query(self, sub_question: SubQuestion) -> str:
        """D2, other half. The sub-question verbatim."""
        return sub_question["text"]

    def decide_next(self, sub_question: SubQuestion, outcome: RetrievalOutcome) -> str:
        """D3, static. Always stop after one round.

        The status still depends on whether anything was found — that is
        deterministic bookkeeping, not a judgment. A sub-question that
        retrieved nothing is `insufficient`, which is information D5 needs
        and which would be lost if everything were marked `sufficient`.
        """
        return "sufficient" if outcome.found else "insufficient"

    # -- execution ----------------------------------------------------------

    def retrieve_one(
        self, state: ResearchState, sub_question_id: str
    ) -> dict[str, Any]:
        """One retrieval round for one sub-question. Returns a state update."""
        sub_question = find_sub_question(state, sub_question_id)
        if sub_question is None:
            return {}

        route = self.choose_route(sub_question)
        query = self.formulate_query(sub_question)
        outcome = self._execute(sub_question_id, route, query)

        updated = dict(sub_question)
        updated["rounds"] = int(sub_question.get("rounds", 0)) + 1
        updated["attempted_routes"] = [*sub_question.get("attempted_routes", []), route]
        updated["attempted_queries"] = [
            *sub_question.get("attempted_queries", []), query
        ]
        updated["evidence_ids"] = _merge_ids(
            sub_question.get("evidence_ids") or [],
            [p["id"] for p in outcome.passages],
        )
        updated["status"] = self.decide_next(sub_question, outcome)

        # Features are recomputed every round, not only at the end. D3 reads
        # them at C21 to decide whether to stop, so they have to exist while
        # the sub-question is still in flight; and recording them per round
        # means the trace shows how the evidence improved (or did not),
        # which is what distinguishes a useful second round from a retry.
        from src.sufficiency.features import extract_features

        updated["sufficiency_features"] = extract_features(
            updated,
            outcome.passages,
            contradictions=state.get("contradictions"),
            # Nothing wires the C12 detector into the graph yet, so a clean
            # list means "not scanned", not "no disagreement". C21 flips this
            # where the scan actually runs.
            contradictions_scanned=False,
            premise_check=state.get("premise_check"),
            critic_findings=state.get("critic_findings") or None,
        )

        self._trace(sub_question, route, query, outcome, updated["status"],
                    features=updated["sufficiency_features"])

        plan = [
            updated if sq["id"] == sub_question_id else sq
            for sq in (state.get("plan") or [])
        ]
        return {"plan": plan, "evidence": outcome.passages}

    def retrieve_all(self, state: ResearchState) -> ResearchState:
        """Run every pending sub-question. Used before the graph exists (C19).

        Applies updates in sequence rather than returning one patch, because
        each round's `plan` replaces the previous one.
        """
        from src.state import merge_evidence

        working = dict(state)
        while True:
            pending = pending_sub_questions(working)  # type: ignore[arg-type]
            if not pending:
                break
            if self.budget is not None and not self.budget.can_spend("retrieval"):
                log.info("retrieval budget exhausted with %d sub-questions "
                         "pending", len(pending))
                break
            update = self.retrieve_one(working, pending[0]["id"])  # type: ignore[arg-type]
            if not update:
                break
            working["plan"] = update["plan"]
            working["evidence"] = merge_evidence(
                working.get("evidence"), update.get("evidence")
            )
        return working  # type: ignore[return-value]

    # -- the actual retrieval ----------------------------------------------

    def _execute(
        self, sub_question_id: str, route: Route, query: str
    ) -> RetrievalOutcome:
        started = self._now()
        try:
            results = self.retriever.retrieve(query, top_k=self.candidates)
            results = self.reranker.rerank(query, results, top_k=self.top_k)
        except Exception as exc:  # noqa: BLE001
            # Same contract as C6/C7: a retrieval failure is information the
            # agent acts on, not a reason to crash the run.
            log.warning("retrieval failed for %r: %s", query[:60], exc)
            return RetrievalOutcome(
                sub_question_id, route, query, [], [],
                self._now() - started, error=f"{type(exc).__name__}: {exc}",
            )

        self._charge_retrieval()

        passages: list[Passage] = []
        scores: list[float] = []
        for result in results:
            passage = dict(result.passage)
            # Stamp the sub-question that retrieved it. Note this records
            # only which sub-question found it FIRST once merge_evidence
            # dedupes; SubQuestion.evidence_ids is the authoritative mapping.
            passage["sub_question_id"] = sub_question_id
            passages.append(passage)  # type: ignore[arg-type]
            score = passage.get("rerank_score")
            if score is not None:
                scores.append(float(score))

        return RetrievalOutcome(
            sub_question_id, route, query, passages, scores,
            self._now() - started,
        )

    # -- plumbing -----------------------------------------------------------

    def _charge_retrieval(self) -> None:
        if self.budget is not None and self.budget.can_spend("retrieval"):
            self.budget.spend("retrieval", 1)

    def _trace(
        self,
        sub_question: SubQuestion,
        route: Route,
        query: str,
        outcome: RetrievalOutcome,
        status: str,
        features: dict | None = None,
    ) -> None:
        if self.tracer is None:
            return
        from src.tracing.tracer import DecisionSurface

        # Recorded as NON-adaptive. The trace of an A2/A3 run must be
        # comparable to a full run, so the decision points are logged either
        # way — what differs is `was_adaptive`, which is exactly what C35
        # counts.
        self.tracer.decision(
            DecisionSurface.D2_ROUTE_AND_QUERY,
            chosen={"route": route, "query": query},
            alternatives=[],
            rationale="static policy: fixed route, verbatim query",
            inputs={
                "sub_question_id": sub_question["id"],
                "attempted_routes": sub_question.get("attempted_routes", []),
            },
            was_adaptive=False,
        )
        self.tracer.decision(
            DecisionSurface.D3_DEPTH_AND_STOPPING,
            chosen=status,
            alternatives=[],
            rationale="static policy: one round, always stop",
            inputs={
                "rounds": sub_question.get("rounds", 0) + 1,
                "n_results": len(outcome.passages),
                "max_rerank": max(outcome.rerank_scores, default=None),
            },
            was_adaptive=False,
        )
        self.tracer.retrieval(
            route=route,
            query=query,
            n_results=len(outcome.passages),
            latency_s=outcome.latency_s,
            sub_question_id=sub_question["id"],
            error=outcome.error,
            features=features,
        )


def retriever_node(retriever: StaticRetriever):
    """LangGraph node: retrieve for the next pending sub-question.

    One sub-question per invocation, because the controller (C19) re-enters
    this node and chooses which sub-question to work on next — that choice is
    D4's, and doing the whole plan here would take it away.
    """

    def node(state: dict) -> dict:
        pending = pending_sub_questions(state)  # type: ignore[arg-type]
        if not pending:
            return {}
        return retriever.retrieve_one(state, pending[0]["id"])  # type: ignore[arg-type]

    return node


def _coerce_route(value: Any) -> Route:
    text = str(value or "").strip().lower()
    return text if text in EXECUTABLE_ROUTES else "arxiv_meta"  # type: ignore[return-value]


def _merge_ids(existing: Sequence[str], incoming: Sequence[str]) -> list[str]:
    """Append new ids, preserving rank order and dropping repeats."""
    out = list(existing)
    seen = set(out)
    for pid in incoming:
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out