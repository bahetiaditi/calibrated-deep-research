#!/usr/bin/env python3
"""C4 acceptance check — dummy three-node run produces a well-formed trace.

    python -m scripts.smoke_c4

Costs ZERO API calls. Everything is simulated: the point is to prove the
trace format, the decision-surface accounting, and the reader — not to talk
to a model. The shape below deliberately mirrors a real run: plan, retrieve
with a route escalation, decide.
"""
from __future__ import annotations

import sys
import time

from src.config import get_config
from src.tracing.tracer import DecisionSurface, EventType, TraceReader, Tracer

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def ok(m: str) -> None:
    print(f"{GREEN}  PASS{RESET} {m}")


def bad(m: str) -> None:
    print(f"{RED}  FAIL{RESET} {m}")


def simulate(tracer: Tracer) -> None:
    """Three nodes, including one deliberate route escalation and one error."""
    tracer.route("planner", reason="plan is empty", state_summary={"plan_size": 0})
    with tracer.node("planner"):
        tracer.llm_call(
            role="judgment", model="openai/gpt-oss-120b", provider="groq",
            prompt_tokens=420, completion_tokens=180, total_tokens=600,
            latency_s=1.1, from_cache=False,
        )
        tracer.decision(
            DecisionSurface.D1_PLAN_REVISION,
            chosen="initial_decomposition",
            alternatives=["defer"],
            rationale="no plan exists yet",
            inputs={"question_len": 68},
        )
        time.sleep(0.01)

    tracer.route(
        "retriever",
        reason="sq1 pending and budget remains",
        state_summary={"plan_size": 3, "pending": ["sq1"], "budget_used": 0.12},
        from_node="planner",
    )
    with tracer.node("retriever"):
        tracer.decision(
            DecisionSurface.D2_ROUTE_AND_QUERY,
            chosen="arxiv_meta",
            alternatives=["web", "arxiv_fulltext"],
            rationale="sub-question names a paper",
            inputs={"attempted_routes": []},
        )
        tracer.retrieval(route="arxiv_meta", query="mamba-2 SSM position embeddings",
                         n_results=0, latency_s=0.8, sub_question_id="sq1")
        # Route escalation: the shape that proves D2/D3 are live, not static.
        tracer.decision(
            DecisionSurface.D3_DEPTH_AND_STOPPING,
            chosen="wider",
            alternatives=["sufficient", "deeper", "give_up"],
            rationale="zero results; abstract search too narrow",
            inputs={"rounds": 1, "max_rerank": None},
        )
        tracer.decision(
            DecisionSurface.D2_ROUTE_AND_QUERY,
            chosen="web",
            alternatives=["arxiv_meta", "arxiv_fulltext"],
            rationale="arxiv metadata returned nothing; reformulating",
            inputs={"attempted_routes": ["arxiv_meta"],
                    "attempted_queries": ["mamba-2 SSM position embeddings"]},
        )
        tracer.retrieval(route="web", query="does Mamba-2 use rotary embeddings",
                         n_results=6, latency_s=0.5, sub_question_id="sq1")
        tracer.llm_call(
            role="mechanical", model="gemma-4-31b-it", provider="gemini",
            prompt_tokens=900, completion_tokens=120, total_tokens=1020,
            latency_s=2.4, from_cache=False,
        )
        tracer.llm_call(
            role="mechanical", model="gemma-4-31b-it", provider="gemini",
            prompt_tokens=900, completion_tokens=120, total_tokens=1020,
            latency_s=0.0, from_cache=True,
        )

    tracer.route("decider", reason="all sub-questions terminal", from_node="retriever")
    with tracer.node("decider"):
        # Ablation shape: a surface exercised by a STATIC policy, so it must
        # not count toward adaptive decisions.
        tracer.decision(
            DecisionSurface.D4_BUDGET_ALLOCATION,
            chosen="uniform", rationale="ablation A4: static split",
            was_adaptive=False,
        )
        tracer.decision(
            DecisionSurface.D5_TERMINAL_DECISION,
            chosen="ABSTAIN",
            alternatives=["ANSWER", "PARTIAL"],
            rationale="premise check failed: Mamba-2 does not use RoPE",
            inputs={"run_sufficiency": 0.21, "tau_answer": 0.65},
        )
    try:
        with tracer.node("finalizer"):
            raise RuntimeError("simulated node failure")
    except RuntimeError:
        pass


def main() -> int:
    print("\n=== C4 smoke test: structured tracing ===\n")
    cfg = get_config()
    trace_dir = cfg.path("run.trace_dir")

    tracer = Tracer(
        trace_dir,
        trace_id=f"smoke-c4-{int(time.time())}",
        question="Why does Mamba-2 use rotary position embeddings in its SSM blocks?",
        metadata={"smoke": True},
    )
    simulate(tracer)
    tracer.close(ok=True, terminal_decision="ABSTAIN")

    failures = 0
    reader = TraceReader.load(tracer.path)

    if reader.trace_id == tracer.trace_id:
        ok(f"trace written and parsed: {tracer.path.name}")
    else:
        bad("trace_id missing from run_start")
        failures += 1

    raw_lines = [l for l in tracer.path.read_text().splitlines() if l.strip()]
    if len(raw_lines) == len(reader.events):
        ok(f"all {len(raw_lines)} lines are well-formed JSON")
    else:
        bad(f"{len(raw_lines) - len(reader.events)} unparseable lines")
        failures += 1

    nodes = reader.nodes_visited()
    if nodes == ["planner", "retriever", "decider", "finalizer"]:
        ok(f"node sequence recorded: {' -> '.join(nodes)}")
    else:
        bad(f"unexpected node sequence: {nodes}")
        failures += 1

    if len(reader.routes()) == 3:
        ok("3 routing decisions recorded with reasons and state summaries")
    else:
        bad(f"expected 3 routes, got {len(reader.routes())}")
        failures += 1

    counts = reader.adaptive_decision_counts()
    print(f"\n{DIM}--- adaptive decisions per surface (the C35 ablation check){RESET}")
    for surface, n in counts.items():
        print(f"      {surface:28s} {n}")

    if counts["D2_route_and_query"] == 2:
        ok("D2 fired twice — route escalation is visible in the trace")
    else:
        bad("route escalation not captured")
        failures += 1

    if counts["D4_budget_allocation"] == 0:
        ok("D4 was static (was_adaptive=False) and counts as zero — "
           "an A4 ablation would verify exactly this way")
    else:
        bad("a static decision was counted as adaptive")
        failures += 1

    if reader.errors():
        ok(f"node failure captured: {reader.errors()[0]['payload']['error']!r}")
    else:
        bad("simulated node failure was not recorded")
        failures += 1

    totals = reader.llm_totals()
    if totals["live_calls"] == 2 and totals["cached_calls"] == 1:
        ok(f"LLM accounting: {totals['live_calls']} live, "
           f"{totals['cached_calls']} cached, {totals['total_tokens']:,} tokens")
    else:
        bad(f"LLM accounting wrong: {totals}")
        failures += 1

    print(f"\n{DIM}--- run summary (what C38 reads for MAST annotation){RESET}")
    for k, v in reader.summary().items():
        print(f"      {k:16s} {v}")

    print(f"\n{DIM}Trace: {tracer.path}{RESET}")
    print(f"{DIM}Inspect with:  cat {tracer.path.name} | jq -c '.type, .name'{RESET}")

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C4 acceptance: all checks passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())