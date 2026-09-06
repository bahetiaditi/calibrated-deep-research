"""Per-run resource budget — the substrate D4 allocates over.

Budget vs. quota: two different things
--------------------------------------
`llm/rate_limit.py` tracks **provider quota**: what this machine may spend
against Groq and Gemini today, persisted across runs and days. It answers
"am I allowed to make this call at all?"

This module tracks a **single run's budget**: how many LLM and retrieval
calls one benchmark question may consume before the system must stop
researching and decide. It answers "should I keep working on this question?"

They are deliberately separate. Quota is an external constraint imposed on
us; budget is an internal one we impose on ourselves, and it exists because
an agent with unlimited budget never has to make the interesting decision.
D4 (§2.2) is only a real decision surface if the budget can actually run out
— an agent that never faces scarcity is not allocating, it is just spending.

The reserve
-----------
`reserve_fraction` of the LLM budget is withheld from allocation. The critic
and decider run *after* retrieval finishes, and a system that spends its
entire budget researching and then cannot afford to verify or decide has
failed in the most embarrassing way available: it did all the work and
produced nothing. D4 allocates over `allocatable()`, never over the total.

Exhaustion
----------
Two paths, deliberately:
  - `can_spend()` — what the controller consults to route to the decider
    gracefully when the budget is nearly gone. This is the normal path.
  - `spend()` — raises `BudgetExhausted` if a caller tries to overspend.
    This is the guard: it should never fire in correct operation, and if it
    does, a node bypassed the controller's check and that is a bug worth
    hearing about loudly rather than a run that silently overruns.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Protocol

from src.config import Config, get_config

ResourceKind = Literal["llm", "retrieval"]


class BudgetExhausted(RuntimeError):
    """A caller tried to spend past a hard cap without checking first."""

    def __init__(self, kind: str, used: int, cap: int, requested: int) -> None:
        super().__init__(
            f"{kind} budget exhausted: {used}/{cap} used, requested {requested}"
        )
        self.kind = kind
        self.used = used
        self.cap = cap
        self.requested = requested


@dataclass
class Budget:
    """Accounting for one run."""

    llm_calls_max: int
    retrieval_calls_max: int
    wall_clock_max_s: float
    reserve_fraction: float = 0.20
    min_calls_per_subquestion: int = 1
    max_rounds_per_subquestion: int = 3
    max_plan_revisions: int = 2

    llm_calls_used: int = 0
    retrieval_calls_used: int = 0
    plan_revisions: int = 0

    allocation: dict[str, int] = field(default_factory=dict)
    spend_by_sub_question: dict[str, int] = field(default_factory=dict)

    _started: float = field(default_factory=time.time)
    _now: Callable[[], float] = field(default=time.time, repr=False)
    tracer: Any = field(default=None, repr=False)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Config | None = None,
        *,
        now: Callable[[], float] = time.time,
        tracer: Any = None,
    ) -> "Budget":
        cfg = config or get_config()
        section = cfg.section("budget")
        return cls(
            llm_calls_max=int(section["llm_calls_max"]),
            retrieval_calls_max=int(section["retrieval_calls_max"]),
            wall_clock_max_s=float(section["wall_clock_max_s"]),
            reserve_fraction=float(section.get("reserve_fraction", 0.20)),
            min_calls_per_subquestion=int(section.get("min_calls_per_subquestion", 1)),
            max_rounds_per_subquestion=int(section.get("max_rounds_per_subquestion", 3)),
            max_plan_revisions=int(section.get("max_plan_revisions", 2)),
            _started=now(),
            _now=now,
            tracer=tracer,
        )

    # -- accounting ---------------------------------------------------------

    def _cap(self, kind: ResourceKind) -> int:
        return self.llm_calls_max if kind == "llm" else self.retrieval_calls_max

    def _used(self, kind: ResourceKind) -> int:
        return self.llm_calls_used if kind == "llm" else self.retrieval_calls_used

    def remaining(self, kind: ResourceKind) -> int:
        return max(0, self._cap(kind) - self._used(kind))

    def can_spend(self, kind: ResourceKind = "llm", n: int = 1) -> bool:
        """Non-raising check. This is what the controller consults."""
        return self._used(kind) + n <= self._cap(kind) and not self.out_of_time()

    def spend(
        self, kind: ResourceKind = "llm", n: int = 1, *, sub_question_id: str | None = None
    ) -> None:
        used, cap = self._used(kind), self._cap(kind)
        if used + n > cap:
            raise BudgetExhausted(kind, used, cap, n)
        if kind == "llm":
            self.llm_calls_used += n
        else:
            self.retrieval_calls_used += n
        if sub_question_id is not None:
            self.spend_by_sub_question[sub_question_id] = (
                self.spend_by_sub_question.get(sub_question_id, 0) + n
            )
        if self.tracer is not None:
            self.tracer.note(
                "budget_spend",
                kind=kind, n=n, sub_question_id=sub_question_id,
                llm_used=self.llm_calls_used, llm_max=self.llm_calls_max,
                retrieval_used=self.retrieval_calls_used,
                retrieval_max=self.retrieval_calls_max,
            )

    # -- time ---------------------------------------------------------------

    def elapsed_s(self) -> float:
        return self._now() - self._started

    def out_of_time(self) -> bool:
        return self.elapsed_s() >= self.wall_clock_max_s

    # -- the reserve --------------------------------------------------------

    def reserved(self) -> int:
        """LLM calls withheld from allocation for the critic and decider."""
        return int(round(self.llm_calls_max * self.reserve_fraction))

    def allocatable(self) -> int:
        """LLM calls D4 may distribute across sub-questions, right now."""
        return max(0, self.llm_calls_max - self.reserved() - self.llm_calls_used)

    def in_reserve(self) -> bool:
        """True once research has eaten everything but the reserve.

        The controller reads this to stop retrieving and start verifying.
        """
        return self.allocatable() <= 0

    # -- plan revisions (D1's hard ceiling) ---------------------------------

    def can_revise_plan(self) -> bool:
        return self.plan_revisions < self.max_plan_revisions

    def record_plan_revision(self) -> None:
        self.plan_revisions += 1

    # -- reporting ----------------------------------------------------------

    def fraction_consumed(self) -> float:
        """Run-level sufficiency feature f-aggregate (§5.2)."""
        caps = self.llm_calls_max + self.retrieval_calls_max
        return (self.llm_calls_used + self.retrieval_calls_used) / caps if caps else 0.0

    def to_state(self) -> dict[str, Any]:
        """Serialise into ResearchState['budget'] (§3.2 BudgetState)."""
        return {
            "llm_calls_used": self.llm_calls_used,
            "llm_calls_max": self.llm_calls_max,
            "retrieval_calls_used": self.retrieval_calls_used,
            "retrieval_calls_max": self.retrieval_calls_max,
            "allocation": dict(self.allocation),
        }

    def report(self) -> str:
        return (
            f"llm {self.llm_calls_used}/{self.llm_calls_max} "
            f"(reserve {self.reserved()}, allocatable {self.allocatable()}) | "
            f"retrieval {self.retrieval_calls_used}/{self.retrieval_calls_max} | "
            f"{self.elapsed_s():.0f}s/{self.wall_clock_max_s:.0f}s | "
            f"revisions {self.plan_revisions}/{self.max_plan_revisions}"
        )


# ---------------------------------------------------------------------------
# Allocation (D4)
# ---------------------------------------------------------------------------


class Allocator(Protocol):
    """Distributes the allocatable budget across sub-questions.

    Two implementations exist by design. `UniformAllocator` is the static
    policy and is literally ablation A4 (§6.5). The adaptive allocator lands
    at C22 and must beat it, or D4 was not a real decision surface.
    """

    name: str

    def allocate(self, budget: Budget, sub_questions: Iterable[Any]) -> dict[str, int]:
        ...


def _distribute(pool: int, ids: list[str], minimum: int) -> dict[str, int]:
    """Split `pool` across `ids`, giving each at least `minimum`.

    Remainder is handed out one call at a time rather than discarded, so
    allocations always sum to exactly the pool — integer division would
    silently lose up to len(ids)-1 calls, which at a 25-call budget is a
    material fraction.
    """
    if not ids:
        return {}
    if pool <= 0:
        return {sq: 0 for sq in ids}

    if pool < minimum * len(ids):
        # Cannot honour the floor for everyone. Fund as many as possible in
        # order rather than starving all of them equally.
        out = {sq: 0 for sq in ids}
        for sq in ids:
            if pool < minimum:
                break
            out[sq] = minimum
            pool -= minimum
        for sq in ids:  # any dregs
            if pool <= 0:
                break
            out[sq] += 1
            pool -= 1
        return out

    base, remainder = divmod(pool, len(ids))
    out = {sq: base for sq in ids}
    for sq in ids[:remainder]:
        out[sq] += 1
    return out


class UniformAllocator:
    """Even split. The static counterpart of D4 — this IS ablation A4."""

    name = "uniform"

    def allocate(self, budget: Budget, sub_questions: Iterable[Any]) -> dict[str, int]:
        ids = [_sq_id(sq) for sq in sub_questions]
        alloc = _distribute(budget.allocatable(), ids, budget.min_calls_per_subquestion)
        budget.allocation = alloc
        if budget.tracer is not None:
            from src.tracing.tracer import DecisionSurface

            budget.tracer.decision(
                DecisionSurface.D4_BUDGET_ALLOCATION,
                chosen=alloc,
                alternatives=[],
                rationale="static uniform split (A4 / D4 disabled)",
                inputs={"allocatable": budget.allocatable(), "n": len(ids)},
                was_adaptive=False,   # must not count toward adaptive decisions
            )
        return alloc


def _sq_id(sub_question: Any) -> str:
    if isinstance(sub_question, str):
        return sub_question
    if isinstance(sub_question, dict):
        return str(sub_question["id"])
    return str(getattr(sub_question, "id"))


def get_allocator(config: Config | None = None) -> Allocator:
    """Pick the allocator the agency flags call for.

    With `agency.d4_budget_allocation: false` the uniform policy is used and
    its decisions are logged as non-adaptive, which is what makes the A4
    ablation verifiable from the trace alone (C35).
    """
    cfg = config or get_config()
    if not cfg.get("agency.d4_budget_allocation", True):
        return UniformAllocator()
    # Adaptive allocator arrives at C22. Until then uniform is the behaviour,
    # and it is honest to say so rather than pretend D4 is live.
    return UniformAllocator()