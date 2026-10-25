"""Budget allocation — D4 (C22).

The decision
------------
A run has a finite retrieval budget and several sub-questions competing for
it. §2.2 defines D4 as spending "more on the ones that are close to
sufficiency and cutting losses on the ones that are not". The static
counterpart (C5's `UniformAllocator`, which *is* ablation A4) splits evenly
and ignores what has been learned.

The shape of marginal value
---------------------------
The value of one more retrieval round is **not** monotone in how badly a
sub-question is doing. It peaks in the middle:

    already sufficient      0.0   nothing left to buy
    just below the bar      high  one more round plausibly closes it
    far below the bar       low   the corpus does not appear to have it
    never attempted         high  unknown, and unknown must be tried

That last row is a floor, not a formula, and it matters. A sub-question that
has never been retrieved for has no evidence *because nobody looked*, so any
score-based estimate of its prospects is a guess dressed as a measurement.
`min_calls_per_subquestion` guarantees every sub-question at least one
attempt before allocation touches it — otherwise D4 is not allocating, it is
pre-judging, and it would systematically starve exactly the sub-questions
whose difficulty the planner failed to anticipate.

Plausibility
------------
Rounds already spent without success lower the estimate, which is f8's
rationale from §5.2 ("many rounds to reach thin evidence is a bad sign")
used as a decision rather than only as a feature. A sub-question on its third
round at f1 1.0 is a worse bet than one on its first round at f1 1.0, even
though their evidence is identical.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

from src.budget import Budget, _distribute, _sq_id
from src.config import Config, get_config

log = logging.getLogger(__name__)

# How fast prospects decay with rounds already spent. At 0.5, a second round
# is worth ~0.67 of a first and a third ~0.5 — steep enough to cut losses,
# gentle enough that one bad round does not condemn a sub-question.
ROUND_DECAY = 0.5

# Floor on the value of a sub-question that has been attempted. Nothing is
# ever certainly hopeless; a floor keeps a surviving sub-question eligible if
# budget remains after the promising ones are funded.
MIN_VALUE = 0.05


class MarginalValueAllocator:
    """Distributes budget by expected value of one more round (D4)."""

    name = "marginal_value"

    def __init__(self, config: Config | None = None, *, tracer: Any = None) -> None:
        self.cfg = config or get_config()
        adaptive = self.cfg.section("retrieval.adaptive")
        self.sufficient_min_rerank = float(adaptive.get("sufficient_min_rerank", 4.0))
        self.hopeless_max_rerank = float(adaptive.get("hopeless_max_rerank", 0.0))
        self.tracer = tracer

    # -- value --------------------------------------------------------------

    def value(self, sub_question: dict[str, Any]) -> float:
        """Expected value of granting this sub-question one more round."""
        status = sub_question.get("status")
        if status in ("sufficient", "abandoned"):
            return 0.0

        rounds = int(sub_question.get("rounds", 0) or 0)
        if rounds == 0:
            # Never attempted. No evidence exists because nobody looked, so
            # any score-based estimate would be a guess dressed as a
            # measurement. Fund it and find out.
            return 1.0

        f1 = (sub_question.get("sufficiency_features") or {}).get("f1_max_rerank")
        if f1 is None:
            # Attempted and retrieved nothing at all. Worse than a low score,
            # which at least means the route found the right neighbourhood.
            return max(MIN_VALUE, 0.3 * self._plausibility(rounds))

        proximity = self._proximity(float(f1))
        return max(MIN_VALUE, proximity * self._plausibility(rounds))

    def _proximity(self, f1: float) -> float:
        """How close this is to clearing the bar, in [0, 1]."""
        span = self.sufficient_min_rerank - self.hopeless_max_rerank
        if span <= 0:
            return 0.5
        return max(0.0, min(1.0, (f1 - self.hopeless_max_rerank) / span))

    def _plausibility(self, rounds: int) -> float:
        """Prospects decay with rounds already spent without success.

        f8's rationale (§5.2) used as a decision rather than a feature: a
        sub-question on its third round at f1 1.0 is a worse bet than one on
        its first round at f1 1.0, though their evidence is identical.
        """
        return 1.0 / (1.0 + ROUND_DECAY * rounds)

    # -- allocation ---------------------------------------------------------

    def allocate(
        self, budget: Budget, sub_questions: Iterable[Any]
    ) -> dict[str, int]:
        items = list(sub_questions)
        ids = [_sq_id(sq) for sq in items]
        pool = budget.allocatable()

        if not ids:
            return {}
        if pool <= 0:
            allocation = {sq: 0 for sq in ids}
            budget.allocation = allocation
            return allocation

        values = {
            _sq_id(sq): self.value(sq if isinstance(sq, dict) else {})
            for sq in items
        }

        # Guarantee a first attempt before allocation decides anything. A
        # sub-question starved before it has been tried is pre-judged, not
        # allocated — and it is exactly the one whose difficulty the planner
        # failed to anticipate.
        floor = budget.min_calls_per_subquestion
        untried = [i for i in ids if values[i] >= 1.0]
        guaranteed = {i: floor for i in untried}
        remaining = pool - sum(guaranteed.values())

        if remaining <= 0:
            allocation = _distribute(pool, ids, floor)
            budget.allocation = allocation
            self._trace(allocation, values, pool, "budget covers only first attempts")
            return allocation

        contenders = {i: v for i, v in values.items() if v > 0.0}
        total = sum(contenders.values())
        allocation = {i: guaranteed.get(i, 0) for i in ids}

        if total <= 0:
            # Everything is terminal; nothing left worth funding.
            budget.allocation = allocation
            self._trace(allocation, values, pool, "no sub-question has value")
            return allocation

        # Largest-remainder apportionment, so the shares sum exactly to the
        # pool rather than losing up to n-1 calls to truncation.
        exact = {i: remaining * v / total for i, v in contenders.items()}
        for i, share in exact.items():
            allocation[i] += int(share)
        shortfall = remaining - sum(
            allocation[i] - guaranteed.get(i, 0) for i in ids
        )
        for i, _ in sorted(
            exact.items(), key=lambda kv: (kv[1] - int(kv[1])), reverse=True
        ):
            if shortfall <= 0:
                break
            allocation[i] += 1
            shortfall -= 1

        budget.allocation = allocation
        self._trace(allocation, values, pool, "marginal expected value")
        return allocation

    def next_sub_question(self, sub_questions: Sequence[Any]) -> str | None:
        """Which sub-question to work on now — the highest-value open one.

        §3.3's routing table says the controller picks "highest-value SQ per
        D4". Without this the retriever takes the first pending one in plan
        order, and D4's allocation is a number nobody reads.
        """
        scored = [
            (self.value(sq), position, _sq_id(sq))
            for position, sq in enumerate(sub_questions)
            if isinstance(sq, dict) and self.value(sq) > 0.0
        ]
        if not scored:
            return None
        # Ties break toward PLAN ORDER, not lexical id order. Every untried
        # sub-question scores exactly 1.0, so the first choice of a run is
        # always a tie — and sorting on id meant a live run opened by
        # retrieving for "sq2", the hopeless one, purely because "sq2" sorts
        # after "sq1". Plan order carries the planner's reasoning about
        # sequence; id order carries nothing.
        return min(scored, key=lambda item: (-item[0], item[1]))[2]

    # -- tracing ------------------------------------------------------------

    def _trace(
        self,
        allocation: dict[str, int],
        values: dict[str, float],
        pool: int,
        rationale: str,
    ) -> None:
        if self.tracer is None:
            return
        from src.tracing.tracer import DecisionSurface

        self.tracer.decision(
            DecisionSurface.D4_BUDGET_ALLOCATION,
            chosen=allocation,
            alternatives=[{i: pool // max(1, len(allocation)) for i in allocation}],
            rationale=rationale,
            inputs={
                "allocatable": pool,
                "values": {i: round(v, 3) for i, v in values.items()},
            },
            was_adaptive=True,
        )


def get_allocator(config: Config | None = None, *, tracer: Any = None) -> Any:
    """Pick the allocator the agency flags call for.

    With `agency.d4_budget_allocation: false` the uniform policy is used and
    its decisions are logged as non-adaptive, which is what makes the A4
    ablation verifiable from the trace alone (C35).
    """
    cfg = config or get_config()
    if not cfg.get("agency.d4_budget_allocation", True):
        from src.budget import UniformAllocator

        return UniformAllocator()
    return MarginalValueAllocator(cfg, tracer=tracer)