"""Adaptive retriever — D2 and D3 live (C21).

What changes from C17
---------------------
Three decisions stop being constants:

  D3  stop / deeper / wider / give up, from the evidence actually retrieved
  D2a which route to try next, conditioned on what has already been tried
  D2b how to rephrase the query, conditioned on what came back

Everything else — the retrieval mechanics, the bookkeeping, the feature
extraction — is inherited from `StaticRetriever` unchanged. That is
deliberate: §6.5's A2 and A3 ablations restore the static decisions while
holding everything else fixed, and the comparison is only clean if the
*only* difference is the decision.

Where the LLM is, and where it is not
-------------------------------------
D3 is decided from features, not by a model. A sub-question whose best
passage scores 8.6 is plainly answered; one scoring -1.9 plainly is not.
Spending an LLM call to be told that would cost roughly 2,500 tokens per
sub-question per round against a daily budget of 200,000, and it would make
the decision less auditable rather than more — a threshold on f1 can be
inspected, a model's "I think that's enough" cannot.

This is still adaptive in the sense §2.1 requires: the decision is
*conditioned on the evidence retrieved*, so freezing it changes behaviour,
which is what A3 measures. "Adaptive" means the outcome depends on what
happened, not that an LLM was involved.

The LLM is used for the one part that genuinely needs language: writing a
better query after a failure. A threshold cannot tell you that "what
perplexity does the model reach" should become "WikiText-103 perplexity
results table", and that rewrite is where the recovery actually comes from.

**Cost: one LLM call per escalation, and none at all for a sub-question
answered on the first attempt.** Round one needs no conditioning because
there is no history to condition on, so the common case is free.

Thresholds
----------
`sufficient_min_rerank: 4.0` sits in the gap a live C20 run measured:
well-covered questions averaged f1 7.2, corpus gaps averaged 0.0, with the
worst good question at 4.74 and the best bad one at 2.58. The threshold is
fitted to observed data rather than guessed, and it is in config so C33 can
revisit it once the calibration split exists.
"""
from __future__ import annotations

import logging
from typing import Any

from src.config import Config
from src.llm.provider import Role
from src.nodes.retriever import RetrievalOutcome, StaticRetriever, _coerce_route
from src.prompts import render
from src.state import Route, SubQuestion

log = logging.getLogger(__name__)

# What "deeper" and "wider" mean as route moves. Deeper reads further into
# material already found; wider looks somewhere else entirely.
DEEPER_FROM: dict[str, Route] = {
    "arxiv_meta": "arxiv_fulltext",
    "web": "arxiv_fulltext",
    "evidence_store": "arxiv_meta",
    "arxiv_fulltext": "arxiv_fulltext",
}
# Depth is earned, never assumed. A first attempt starts shallow even when
# the planner asked for the expensive route; D3 escalates if the shallow
# attempt is not good enough, which is the point of having D3 at all.
CHEAP_FIRST: dict[str, Route] = {
    "arxiv_fulltext": "arxiv_meta",
}

WIDER_FROM: dict[str, Route] = {
    "arxiv_meta": "web",
    "arxiv_fulltext": "web",
    "web": "arxiv_meta",
    "evidence_store": "web",
}

SUFFICIENT = "sufficient"
# "another round is wanted" and "finished, without enough evidence" are
# different states and must not share a status. `insufficient` is in
# TERMINAL_STATUSES, so returning it to mean "try again" stopped the loop
# before any second round ever ran — the adaptive retriever behaved exactly
# like the static one and the smoke test measured a zero delta.
IN_PROGRESS = "in_progress"
INSUFFICIENT = "insufficient"
ABANDONED = "abandoned"


class AdaptiveRetriever(StaticRetriever):
    def __init__(self, config: Config | None = None, *, provider: Any = None, **kw):
        super().__init__(config, **kw)
        adaptive = self.cfg.section("retrieval.adaptive")
        self.sufficient_min_rerank = float(adaptive.get("sufficient_min_rerank", 4.0))
        self.min_sources = int(adaptive.get("min_independent_sources", 2))
        self.hopeless_max_rerank = float(adaptive.get("hopeless_max_rerank", 0.0))
        self.reformulate_with_llm = bool(adaptive.get("reformulate_with_llm", True))
        self.max_rounds = int(
            self.cfg.get("budget.max_rounds_per_subquestion", 3)
        )

        # Agency flags: these are what A2 and A3 flip.
        self.d2_live = bool(self.cfg.get("agency.d2_adaptive_route_and_query", True))
        self.d3_live = bool(self.cfg.get("agency.d3_adaptive_depth_and_stopping", True))

        self.min_improvement = float(
            self.cfg.get("retrieval.adaptive.min_improvement", 0.5)
        )
        self._provider = provider
        # NO per-sub-question state on the instance (C27.2). The pending
        # route/query and the best f1 so far live on the SubQuestion
        # (`next_route`, `next_query`, `best_f1`): this object is built once
        # and reused across questions whose sub-questions share ids, and
        # instance dicts keyed by id leaked one question's reformulations
        # into the next. `decide_next` writes those fields onto the
        # sub-question it is given; `step` hands it a copy.

    @property
    def provider(self) -> Any:
        if self._provider is None:
            from src.llm.provider import LLMProvider, Role

            self._provider = LLMProvider(self.cfg, tracer=self.tracer)
        return self._provider

    # -- D2: route and query ------------------------------------------------

    def choose_route(self, sub_question: SubQuestion) -> Route:
        if not self.d2_live:
            return super().choose_route(sub_question)

        attempted = sub_question.get("attempted_routes") or []
        if not attempted:
            # First attempt: the planner's suggestion, but **clamped to a
            # cheap route**. §3.3 is explicit that the reader is "invoked
            # only when the retriever decides depth is needed (D3), which
            # makes depth an EARNED cost rather than a default" — and a
            # planner that suggests `arxiv_fulltext` up front bypasses that
            # entirely.
            #
            # Measured consequence, from the first working C21 run: the
            # planner suggested arxiv_fulltext for all three sub-questions,
            # so every one started inside paper bodies. Independent sources
            # fell from 12 to 2 against the static baseline, because
            # full-text chunks concentrate in the few papers already
            # ingested while abstracts span every paper the search returns.
            # Depth bought precision (f1 3.07 -> 4.79) and cost breadth
            # (f4 12 -> 2), which is a bad trade to make before the cheap
            # route has even been tried.
            suggested = _coerce_route(sub_question.get("suggested_route"))
            return CHEAP_FIRST.get(suggested, suggested)

        route = sub_question.get("next_route")
        if route:
            return _coerce_route(route)

        # No LLM decision recorded (reformulation disabled or it failed):
        # escalate deterministically rather than repeating a failed route.
        return self._escalate_route(attempted[-1], "wider")

    def formulate_query(self, sub_question: SubQuestion) -> str:
        if not self.d2_live:
            return super().formulate_query(sub_question)

        query = (sub_question.get("next_query") or "").strip()
        attempted = sub_question.get("attempted_queries") or []
        if query and query not in attempted:
            return query
        return sub_question["text"]

    def _escalate_route(self, last: str, action: str) -> Route:
        table = DEEPER_FROM if action == "deeper" else WIDER_FROM
        return table.get(str(last), "web")  # type: ignore[return-value]

    # -- D3: depth and stopping ---------------------------------------------

    def decide_next(self, sub_question: SubQuestion, outcome: RetrievalOutcome) -> str:
        if not self.d3_live:
            return super().decide_next(sub_question, outcome)

        rounds = int(sub_question.get("rounds", 0)) + 1
        features = outcome_features(outcome)
        f1 = features["f1"]
        sources = features["sources"]

        previous = sub_question.get("best_f1")
        if f1 is not None:
            sub_question["best_f1"] = max(f1, previous if previous is not None else f1)
        # This round has consumed whatever was pending; a stale query must
        # not be replayed if the sub-question is reopened later.
        sub_question["next_route"] = None
        sub_question["next_query"] = None

        verdict, rationale = self._judge(rounds, f1, sources, previous)

        if verdict in (SUFFICIENT, ABANDONED, INSUFFICIENT):
            self._record_decision(sub_question, verdict, rationale, features, None)
            return verdict

        # Escalating: this is where the LLM earns its call.
        decision = self._reformulate(sub_question, outcome, features)
        if decision is None or decision.get("action") == "give_up":
            reason = (decision or {}).get("reasoning", "no viable reformulation")
            self._record_decision(sub_question, ABANDONED, reason, features, decision)
            return ABANDONED

        sub_question["next_route"] = decision.get("route") or None
        sub_question["next_query"] = (decision.get("query") or "").strip() or None
        self._record_decision(
            sub_question, IN_PROGRESS,
            f"{decision.get('action')}: {decision.get('reasoning', '')}",
            features, decision,
        )
        # IN_PROGRESS is non-terminal, so the controller comes back to this
        # sub-question with the reformulated query recorded above.
        return IN_PROGRESS

    def _judge(
        self,
        rounds: int,
        f1: float | None,
        sources: int,
        previous_best: float | None = None,
    ) -> tuple[str, str]:
        """The stopping policy. Deterministic given the evidence, and
        therefore inspectable — which an LLM's "that seems enough" is not.

        Two tests, and the second matters more than the first.

        The **absolute** test asks whether the evidence is good enough. The
        **marginal** test asks whether the last round made it better. A
        retriever with only the absolute test escalates until it runs out of
        rounds whenever the corpus simply cannot clear the bar: a live C21
        run spent 9 retrieval calls, 6 LLM calls and 71 seconds to reach
        exactly the f1 the static retriever reached in 3 calls and 8 seconds,
        because no round after the first improved anything and nothing
        noticed.

        That is not adaptivity, it is a fixed schedule with a threshold
        attached — §2.4's "the loop counter is the only thing that changes
        between iterations". Stopping when a round stops helping is what
        makes the number of rounds depend on the evidence.
        """
        if f1 is None:
            if rounds >= self.max_rounds:
                return ABANDONED, f"nothing retrieved in {rounds} rounds"
            return IN_PROGRESS, "nothing retrieved; trying another approach"

        # 1. Good enough, corroborated. Stop.
        if f1 >= self.sufficient_min_rerank and sources >= self.min_sources:
            return SUFFICIENT, (
                f"f1={f1:.2f} >= {self.sufficient_min_rerank} across "
                f"{sources} sources"
            )

        # 2. Out of rounds, or this round stopped helping. Either way no
        #    further work is coming, and the LEVEL of the evidence decides
        #    what to call it.
        plateaued = (
            previous_best is not None
            and f1 - previous_best < self.min_improvement
        )
        exhausted = rounds >= self.max_rounds

        if plateaued or exhausted:
            why = (
                f"round {rounds} improved f1 by {f1 - previous_best:+.2f} "
                f"(< {self.min_improvement})"
                if plateaued else f"round limit reached"
            )
            return self._label(f1), f"{why}; f1={f1:.2f}"

        # 3. Still improving and rounds remain. Keep going — including from
        #    deep below the bar. A live run abandoned a sub-question that
        #    had gone from f1 -4.21 to -0.29, eight times `min_improvement`,
        #    purely for still being below zero. The absolute level says
        #    where the evidence IS; the improvement says where it is GOING,
        #    and only the second justifies stopping early.
        return IN_PROGRESS, f"f1={f1:.2f} below {self.sufficient_min_rerank}"

    def _label(self, f1: float) -> str:
        """What to call a sub-question no further work will improve."""
        if f1 >= self.sufficient_min_rerank:
            return SUFFICIENT
        if f1 <= self.hopeless_max_rerank:
            return ABANDONED
        return INSUFFICIENT

    # -- reformulation (the LLM call) ---------------------------------------

    def _reformulate(
        self,
        sub_question: SubQuestion,
        outcome: RetrievalOutcome,
        features: dict[str, Any],
    ) -> dict[str, Any] | None:
        last_route = (sub_question.get("attempted_routes") or ["arxiv_meta"])[-1]

        if not self.reformulate_with_llm or not self.d2_live:
            return {
                "action": "wider",
                "route": self._escalate_route(last_route, "wider"),
                "query": sub_question["text"],
                "reasoning": "deterministic escalation (LLM reformulation off)",
            }

        if self.budget is not None and not self.budget.can_spend("llm"):
            log.info("no LLM budget for reformulation; escalating route only")
            return {
                "action": "wider",
                "route": self._escalate_route(last_route, "wider"),
                "query": sub_question["text"],
                "reasoning": "budget exhausted; route escalation only",
            }

        attempted = "\n".join(
            f"- route={r}  query={q!r}"
            for r, q in zip(
                sub_question.get("attempted_routes") or [],
                sub_question.get("attempted_queries") or [],
            )
        ) or "(nothing yet)"

        summary = _summarise(outcome, features)

        try:
            parsed, _ = self.provider.complete_json(
                system=render(
                    "retriever_reformulate",
                    sub_question=sub_question["text"],
                    question=sub_question.get("rationale") or sub_question["text"],
                    attempted=attempted,
                    evidence_summary=summary,
                ),
                user="Decide how to retry.",
                # MECHANICAL, as C2 allocated query reformulation. The call
                # passed no role and inherited complete_json's default,
                # JUDGMENT, so every reformulation spent the Groq chain's
                # scarce daily tokens (~1.1K each, ~6 per question) and added
                # to the minute-window pressure that pushed reports onto
                # fallback models (C27.6).
                role=Role.MECHANICAL,
                required_keys=["action"],
            )
        except Exception as exc:  # noqa: BLE001
            # Same contract as everywhere else: a failed decision degrades to
            # a deterministic one rather than crashing the run.
            log.warning("reformulation failed: %s", exc)
            return {
                "action": "wider",
                "route": self._escalate_route(last_route, "wider"),
                "query": sub_question["text"],
                "reasoning": f"reformulation failed ({type(exc).__name__})",
            }

        if self.budget is not None and self.budget.can_spend("llm"):
            self.budget.spend("llm", 1)

        action = str(parsed.get("action") or "wider").lower()
        if action not in ("deeper", "wider", "give_up"):
            action = "wider"

        route = parsed.get("route")
        if not route:
            route = self._escalate_route(last_route, action)

        return {
            "action": action,
            "route": _coerce_route(route),
            "query": str(parsed.get("query") or "").strip(),
            "reasoning": str(parsed.get("reasoning") or "")[:300],
        }

    # -- tracing ------------------------------------------------------------

    def _record_decision(
        self,
        sub_question: SubQuestion,
        verdict: str,
        rationale: str,
        features: dict[str, Any],
        decision: dict[str, Any] | None,
    ) -> None:
        if self.tracer is None:
            return
        from src.tracing.tracer import DecisionSurface

        self.tracer.decision(
            DecisionSurface.D3_DEPTH_AND_STOPPING,
            chosen=verdict,
            alternatives=[SUFFICIENT, IN_PROGRESS, INSUFFICIENT, ABANDONED],
            rationale=rationale,
            inputs={
                "rounds": int(sub_question.get("rounds", 0)) + 1,
                "f1_max_rerank": features.get("f1"),
                "independent_sources": features.get("sources"),
                "n_results": features.get("n_results"),
            },
            was_adaptive=self.d3_live,
        )
        if decision is not None:
            self.tracer.decision(
                DecisionSurface.D2_ROUTE_AND_QUERY,
                chosen={"route": decision.get("route"),
                        "query": decision.get("query")},
                alternatives=[r for r in ("arxiv_meta", "arxiv_fulltext", "web")
                              if r != decision.get("route")],
                rationale=decision.get("reasoning", ""),
                inputs={
                    "attempted_routes": sub_question.get("attempted_routes", []),
                    "attempted_queries": sub_question.get("attempted_queries", []),
                    "action": decision.get("action"),
                },
                was_adaptive=self.d2_live,
            )


def outcome_features(outcome: RetrievalOutcome) -> dict[str, Any]:
    """The subset of the feature vector D3 reads, from one round's results."""
    scores = sorted(outcome.rerank_scores, reverse=True)
    return {
        "f1": scores[0] if scores else None,
        "sources": len({
            p.get("source_id") for p in outcome.passages if p.get("source_id")
        }),
        "n_results": len(outcome.passages),
    }


def _summarise(outcome: RetrievalOutcome, features: dict[str, Any]) -> str:
    """What the reformulation prompt is shown.

    Titles and sections rather than full text: the model needs to see what
    *kind* of thing came back to judge whether the query found the wrong
    topic or the wrong granularity, and full passages would cost thousands
    of tokens to say the same thing.
    """
    if not outcome.passages:
        return "Nothing was returned."

    lines = [
        f"{features['n_results']} passages from {features['sources']} "
        f"source(s); best relevance score {features['f1']:.2f} "
        f"(above ~4 means the passage answers the query)."
    ]
    for passage in outcome.passages[:5]:
        section = passage.get("section") or "—"
        lines.append(
            f"- [{section}] {passage.get('source_id', '?')}: "
            f"{(passage.get('text') or '')[:140]}"
        )
    return "\n".join(lines)


def adaptive_retriever_node(retriever: AdaptiveRetriever, allocator: Any = None):
    from src.nodes.retriever import retriever_node

    return retriever_node(retriever, allocator)