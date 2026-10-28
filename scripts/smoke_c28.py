#!/usr/bin/env python3
"""C28 acceptance check — the full graph, finalizer included. Closes Phase 3.

    python -m scripts.smoke_c28

Costs roughly 15-30 judgment calls and ~10 mechanical calls.

The C28 criterion: "full graph runs to completion on an answerable question
and on a false-premise question, producing appropriately different outputs."

The false-premise question is chosen so the corpus can REFUTE it, not merely
fail to support it: the Mamba abstract in the C13 corpus states the
architecture is "without attention or even MLP blocks". Every earlier
false-premise run used Mamba-2 + RoPE, which the corpus is silent on, so the
premise checker could only say UNVERIFIABLE and the hard rule (§5.4) never
fired in a full run. This question should fire it.

Prints each final report in full, then checks what a reader depends on.
"""
from __future__ import annotations

import re
import sys
import time

from src.config import get_config
from src.tracing.tracer import TraceReader

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)

ANSWERABLE = ("How do FlashAttention-2 and PagedAttention differ in their "
              "memory access patterns?")
FALSE_PREMISE = ("Why does Mamba interleave multi-head attention layers between "
                 "its selective SSM blocks?")

INTERNAL_ID = re.compile(r"P[0-9a-f]{12}")


def ok(m): print(f"{GREEN}  PASS{RESET} {m}")
def warn(m): print(f"{YELLOW}  WARN{RESET} {m}")
def bad(m): print(f"{RED}  FAIL{RESET} {m}")


def components(cfg):
    from scripts.smoke_c27 import wire
    from src.nodes.finalizer import Finalizer, finalizer_node

    comps = wire(cfg)
    comps.finalizer = finalizer_node(Finalizer(cfg))
    return comps


def run(comps, cfg, question, label):
    from src.graph import run_research

    thread = f"smoke-c28-{label}-{int(time.time())}"
    print(f"\n{BOLD}{'=' * 78}\n{question}{RESET}")
    t0 = time.time()
    result = run_research(question, components=comps, config=cfg,
                          thread_id=thread, trace_id=thread)
    print(f"{DIM}{time.time() - t0:.0f}s  {' -> '.join(result.node_visits)}")
    if result.budget:
        print(f"budget: {result.budget.report()}{RESET}")
    report = result.state.get("final_report") or ""
    print(f"\n{report}")
    return result


def check(result, label) -> int:
    failures = 0
    state = result.state
    report = state.get("final_report") or ""
    record = state.get("decision") or {}
    decision = state.get("terminal_decision")

    if result.error:
        bad(f"run failed: {result.error}")
        return 1
    if not report:
        bad("no final report")
        return 1
    ok(f"completed: {decision} via {record.get('rule')}")

    if result.node_visits[-1:] == ["finalizer"]:
        ok("finalizer ran last")
    else:
        bad(f"last node was {result.node_visits[-1:]}")
        failures += 1

    if INTERNAL_ID.search(report):
        bad("internal passage ids reached the report")
        failures += 1
    else:
        ok("no internal ids in the report")

    if decision == "ABSTAIN":
        withheld = (record.get("withheld_draft") or "")[:60]
        if "## Answer" in report or (withheld and withheld in report):
            bad("an abstention shows a draft")
            failures += 1
        else:
            ok("abstention shows no draft")
    else:
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", report.split("## Sources")[0])}
        sources = len(re.findall(r"^\[\d+\] ", report, flags=re.M))
        if cited and max(cited) <= sources:
            ok(f"{len(cited)} footnote(s), all resolve to the {sources} source(s)")
        elif cited:
            bad(f"footnote [{max(cited)}] but only {sources} source(s)")
            failures += 1

    subs = [s for s in record.get("sub_questions") or [] if s.get("features")]
    f1 = [s["features"].get("f1_max_rerank") for s in subs]
    kinds = [s.get("kind") for s in subs]
    print(f"{DIM}   f1 per sub-question: {[None if v is None else round(v, 2) for v in f1]}"
          f"  kinds: {kinds}{RESET}")
    distinct = {round(v, 4) for v in f1 if v is not None}
    if len([v for v in f1 if v is not None]) > 1 and len(distinct) == 1:
        warn("every sub-question has the same f1 — the C27.9 symptom")
    elif len(distinct) > 1:
        ok("f1 varies across sub-questions (C27.9)")

    if label == "premise":
        premise = state.get("premise_check") or {}
        verdicts = [p.get("verdict") for p in premise.get("presuppositions") or []]
        print(f"{DIM}   premise verdicts: {verdicts}{RESET}")
        if record.get("rule") == "premise_refuted":
            ok("the hard rule fired: refuted premise, research skipped")
            if "retriever" in result.node_visits:
                bad("research ran despite the refuted premise")
                failures += 1
            if "What the evidence says instead" in report or "contradicts" in report:
                ok("the report states the premise the evidence contradicts")
        else:
            warn(f"premise not refuted (verdicts {verdicts}) — the corpus states "
                 f"Mamba is attention-free; check the premise check's reasoning")

    events = TraceReader.load(result.trace_path).of_type(
        __import__("src.tracing.tracer", fromlist=["EventType"]).EventType.NOTE)
    notes = [e for e in events if e.get("name") == "finalize"]
    if notes:
        p = notes[-1].get("payload", {})
        print(f"{DIM}   finalize: footnotes={p.get('footnotes')} dropped={p.get('dropped')} "
              f"marked={p.get('marked')} stray={p.get('stray_markers')} gaps={p.get('gaps')}{RESET}")
    return failures


def main() -> int:
    print("\n=== C28 smoke test: finalizer + full graph ===")
    cfg = get_config()
    comps = components(cfg)
    failures = 0

    answer = run(comps, cfg, ANSWERABLE, "answerable")
    failures += check(answer, "answerable")
    premise = run(comps, cfg, FALSE_PREMISE, "premise")
    failures += check(premise, "premise")

    print(f"\n{BOLD}Appropriately different?{RESET}")
    a, p = answer.state.get("terminal_decision"), premise.state.get("terminal_decision")
    if a != "ABSTAIN" and p == "ABSTAIN":
        ok(f"answerable -> {a}, false premise -> {p}")
    else:
        warn(f"answerable -> {a}, false premise -> {p}: not the expected split")

    print(f"\n{BOLD}{'all checks passed' if not failures else f'{failures} failure(s)'}"
          f"{RESET}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
