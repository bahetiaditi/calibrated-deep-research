#!/usr/bin/env python3
"""C16 acceptance check — planner on ten sample questions.

    python -m scripts.smoke_c16

Costs 10 LLM calls, no retrieval.

The C16 criterion: "valid structured output on ten sample questions with no
parse failures."

Beyond parsing, this measures the thing §6.7 names as an expected failure
mode: **over-decomposition**. The question set deliberately mixes one-hop
lookups with genuinely multi-part questions, so a planner that emits five
sub-questions for everything is visible rather than merely suspected.
"""
from __future__ import annotations

import sys
import time

from src.budget import Budget
from src.config import get_config
from src.nodes.planner import Planner

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)

# (question, expected_shape) — "shape" is how many sub-questions a careful
# human would write, used to spot over-decomposition rather than to grade.
QUESTIONS = [
    ("What context length does Llama 3.1 405B support?", 1),
    ("What rank did the LoRA authors use in their main experiments?", 1),
    ("What fraction of theoretical peak FLOPs does FlashAttention-2 reach?", 1),
    ("How do FlashAttention-2 and PagedAttention differ in their memory "
     "access patterns?", 3),
    ("Compare LoRA and QLoRA in terms of memory requirements and training "
     "speed.", 4),
    ("How does Mamba's selective state space approach differ from standard "
     "transformer attention?", 3),
    ("What are the main critiques of the Chinchilla scaling laws?", 4),
    ("What are the current approaches to reducing hallucination in RAG "
     "systems?", 4),
    ("How has the approach to RLHF evolved from InstructGPT to DPO to GRPO?", 4),
    ("Under what conditions does parameter-efficient finetuning match full "
     "finetuning?", 4),
]


def ok(m): print(f"{GREEN}  PASS{RESET} {m}")
def warn(m): print(f"{YELLOW}  WARN{RESET} {m}")
def bad(m): print(f"{RED}  FAIL{RESET} {m}")


def main() -> int:
    print("\n=== C16 smoke test: planner ===\n")
    cfg = get_config()
    budget = Budget.from_config(cfg)
    planner = Planner(cfg, budget=budget)

    failures = 0
    parse_failures = 0
    counts: list[tuple[int, int]] = []
    route_use: dict[str, int] = {}
    missing_rationale = 0
    vague = 0

    for question, expected in QUESTIONS:
        print(f"{BOLD}{question[:74]}{RESET}")
        t0 = time.time()
        try:
            plan = planner.plan(question)
        except Exception as exc:  # noqa: BLE001
            bad(f"parse/provider failure: {type(exc).__name__}: {exc}")
            parse_failures += 1
            print()
            continue

        for sq in plan:
            route = sq.get("suggested_route", "?")
            route_use[route] = route_use.get(route, 0) + 1
            print(f"{DIM}   {sq['id']}  [{route:<15}] {sq['text'][:62]}{RESET}")
            print(f"{DIM}        why: {sq['rationale'][:66]}{RESET}")
            if not sq["rationale"]:
                missing_rationale += 1
            # A sub-question is sent to a search engine with no memory of the
            # parent, so a pronoun or a bare "the method" retrieves nothing.
            lowered = sq["text"].lower()
            if any(w in lowered.split() for w in ("it", "this", "that", "they")):
                vague += 1

        counts.append((len(plan), expected))
        print(f"{DIM}   {len(plan)} sub-questions (a careful human: ~{expected}) "
              f"in {time.time() - t0:.1f}s{RESET}\n")

    print(f"{BOLD}Checks{RESET}")

    if parse_failures == 0:
        ok(f"{len(QUESTIONS)}/{len(QUESTIONS)} questions parsed — the C16 "
           f"criterion")
    else:
        bad(f"{parse_failures} parse failures")
        failures += 1

    if missing_rationale == 0:
        ok("every sub-question carries a rationale — C25 can act on failures")
    else:
        bad(f"{missing_rationale} sub-questions have no rationale; D1 "
            f"revision degenerates into a retry loop without them")
        failures += 1

    if vague == 0:
        ok("no sub-question relies on a pronoun or bare reference")
    else:
        warn(f"{vague} sub-questions contain 'it'/'this'/'that' — they are "
             f"sent to search with no memory of the parent question")

    # Over-decomposition: §6.7's named failure mode.
    single_hop = [n for n, expected in counts if expected == 1]
    multi = [n for n, expected in counts if expected > 1]
    print(f"\n{DIM}--- decomposition size{RESET}")
    print(f"{DIM}       one-hop questions  -> {single_hop}{RESET}")
    print(f"{DIM}       multi-part         -> {multi}{RESET}")

    if single_hop and max(single_hop) <= 3:
        ok(f"one-hop questions stay small (max {max(single_hop)})")
    elif single_hop:
        warn(f"a one-hop question produced {max(single_hop)} sub-questions — "
             f"over-decomposition (§6.7). Each costs retrieval budget from a "
             f"pool of ~20.")

    if multi and single_hop and min(multi) >= max(single_hop):
        ok("the planner scales decomposition with question complexity")
    elif multi and single_hop:
        warn("decomposition size does not track complexity — the planner may "
             "be emitting a fixed shape regardless of the question")

    print(f"\n{DIM}--- route choice{RESET}")
    for route, count in sorted(route_use.items(), key=lambda kv: -kv[1]):
        print(f"{DIM}       {route:<16} {count}{RESET}")
    if len(route_use) > 1:
        ok("routes vary by sub-question")
    else:
        warn("every sub-question got the same route — D2 has less to escalate "
             "from, though it may still override")

    total = sum(n for n, _ in counts)
    print(f"\n{DIM}--- cost{RESET}")
    print(f"       {budget.llm_calls_used} LLM calls, {total} sub-questions "
          f"({total / max(1, len(counts)):.1f} per question)")
    print(f"{DIM}       retrieval budget is ~20 calls per run, so a plan of "
          f"{total / max(1, len(counts)):.1f}\n       sub-questions leaves "
          f"~{20 / max(1.0, total / max(1, len(counts))):.0f} retrieval "
          f"calls each{RESET}")

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C16 acceptance: all checks passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())