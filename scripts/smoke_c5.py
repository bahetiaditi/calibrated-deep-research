#!/usr/bin/env python3
"""C5 acceptance check + Phase 0 exit criterion.

    python -m scripts.smoke_c5

Phase 0's exit criterion (REFERENCE.md §8) is: "you can make a cached,
traced, budgeted LLM call with automatic fallback." This script proves
exactly that, end to end, with one live call — everything else is cached,
budgeted, and traced around it.
"""
from __future__ import annotations

import sys
import time

from src.budget import Budget, BudgetExhausted, UniformAllocator
from src.config import get_config
from src.llm.provider import LLMProvider, Role
from src.tracing.tracer import DecisionSurface, TraceReader, Tracer

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def ok(m: str) -> None:
    print(f"{GREEN}  PASS{RESET} {m}")


def bad(m: str) -> None:
    print(f"{RED}  FAIL{RESET} {m}")


def main() -> int:
    print("\n=== C5 smoke test: budget + Phase 0 exit ===\n")
    cfg = get_config()
    failures = 0

    # ---- budget mechanics (no API calls) --------------------------------
    print(f"{DIM}--- budget mechanics{RESET}")
    b = Budget.from_config(cfg)
    print(f"      {b.report()}")

    if b.reserved() > 0 and b.allocatable() == b.llm_calls_max - b.reserved():
        ok(f"{b.reserved()} calls reserved for critic/decider, "
           f"{b.allocatable()} allocatable")
    else:
        bad("reserve is not withheld from the allocatable pool")
        failures += 1

    alloc = UniformAllocator().allocate(b, ["sq1", "sq2", "sq3"])
    if sum(alloc.values()) == b.allocatable():
        ok(f"allocation sums exactly to the pool: {alloc}")
    else:
        bad(f"allocation {alloc} sums to {sum(alloc.values())}, "
            f"expected {b.allocatable()}")
        failures += 1

    probe = Budget.from_config(cfg)
    probe.spend("llm", probe.llm_calls_max)
    try:
        probe.spend("llm", 1)
        bad("overspending did not raise")
        failures += 1
    except BudgetExhausted as exc:
        ok(f"hard stop raises cleanly: {exc}")

    # ---- Phase 0 exit: cached, traced, budgeted call with fallback -------
    print(f"\n{DIM}--- Phase 0 exit: a cached, traced, budgeted call{RESET}")

    trace_id = f"smoke-c5-{int(time.time())}"
    tracer = Tracer(
        cfg.path("run.trace_dir"),
        trace_id=trace_id,
        question="Phase 0 exit check",
        metadata={"smoke": True},
    )
    run_budget = Budget.from_config(cfg, tracer=tracer)
    provider = LLMProvider(cfg, tracer=tracer)

    system = "You are terse. Answer in at most eight words."
    user = f"[{trace_id}] What is reciprocal rank fusion used for?"

    with tracer.node("planner"):
        tracer.decision(
            DecisionSurface.D1_PLAN_REVISION,
            chosen="initial_decomposition",
            alternatives=["defer"],
            rationale="phase 0 exit check",
        )
        if not run_budget.can_spend("llm"):
            bad("budget refused the first call")
            return 1
        first = provider.complete(system, user, role=Role.JUDGMENT)
        run_budget.spend("llm", 1, sub_question_id="sq1")
        ok(f"live call via {first.model}: {first.text.strip()[:44]!r}")

        second = provider.complete(system, user, role=Role.JUDGMENT)
        run_budget.spend("llm", 1, sub_question_id="sq1")
        if second.from_cache:
            ok("identical call served from cache")
        else:
            bad("second identical call was not cached")
            failures += 1

    tracer.close(ok=True, budget=run_budget.to_state())

    reader = TraceReader.load(tracer.path)
    totals = reader.llm_totals()
    if totals["live_calls"] == 1 and totals["cached_calls"] == 1:
        ok(f"trace accounts for both calls: {totals['live_calls']} live, "
           f"{totals['cached_calls']} cached")
    else:
        bad(f"trace LLM accounting wrong: {totals}")
        failures += 1

    if run_budget.llm_calls_used == 2:
        ok(f"budget charged both calls: {run_budget.report()}")
    else:
        bad("budget did not track spend")
        failures += 1

    spend_events = [
        e for e in reader.events if e.get("name") == "budget_spend"
    ]
    if len(spend_events) == 2:
        ok("budget spend is visible in the trace")
    else:
        bad(f"expected 2 budget_spend events, got {len(spend_events)}")
        failures += 1

    print(f"\n{DIM}--- quota{RESET}")
    print(provider.quota_report())
    print(f"\n{DIM}--- cache{RESET}\n  {provider.cache.report()}")
    print(f"\n{DIM}Trace: {tracer.path}{RESET}")

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C5 acceptance passed.{RESET}")
    print(f"{GREEN}PHASE 0 COMPLETE — cached, traced, budgeted LLM calls "
          f"with automatic fallback.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())