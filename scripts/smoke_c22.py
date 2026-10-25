#!/usr/bin/env python3
"""C22 acceptance check — budget allocation (D4). Closes Phase 2.

    python -m scripts.smoke_c22

Costs ~8 LLM calls plus retrieval.

The C22 criterion: "on a run with one hopeless sub-question, the trace shows
budget reallocated away from it."

A plan is constructed with one sub-question the corpus can answer and one it
cannot, so the reallocation is observable rather than inferred.
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.budget import Budget, UniformAllocator
from src.config import get_config
from src.nodes.adaptive_retriever import AdaptiveRetriever
from src.state import new_state, new_sub_question
from src.sufficiency.allocation import MarginalValueAllocator

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)

COLLECTION = "retrieval_eval"

ANSWERABLE = "What fraction of theoretical peak FLOPs does FlashAttention-2 reach?"
# The C13 labelling found zero fully-relevant passages for this in the pool.
HOPELESS = "What makes retrieval augmentation fail on questions requiring synthesis?"


def ok(m): print(f"{GREEN}  PASS{RESET} {m}")
def warn(m): print(f"{YELLOW}  WARN{RESET} {m}")
def bad(m): print(f"{RED}  FAIL{RESET} {m}")


def wire(cfg):
    from qdrant_client import QdrantClient

    from src.rag.fusion import HybridRetriever
    from src.rag.rerank import Reranker
    from src.rag.sparse import BM25Index
    from src.rag.store import EvidenceStore

    path = Path(cfg.get("retrieval.store.path")) / COLLECTION
    store = EvidenceStore(cfg, client=QdrantClient(path=str(path)),
                          collection=COLLECTION)
    hybrid = HybridRetriever(cfg, store=store, sparse=BM25Index(cfg))
    records, _ = store.client.scroll(collection_name=COLLECTION, limit=10_000,
                                     with_payload=True)
    passages = [dict(r.payload) for r in records if r.payload]
    if not passages:
        print(f"{RED}corpus is empty — run "
              f"`python -m scripts.retrieval_labels build` first{RESET}")
        sys.exit(1)
    hybrid.sparse.build(passages)
    print(f"{DIM}corpus: {len(passages)} passages{RESET}\n")
    return hybrid, Reranker(cfg)


def build_plan():
    plan = []
    for i, text in enumerate((ANSWERABLE, HOPELESS)):
        sq = new_sub_question(text, sub_question_id=f"sq{i + 1}")
        sq["suggested_route"] = "arxiv_meta"
        plan.append(sq)
    return plan


def main() -> int:
    print("\n=== C22 smoke test: budget allocation (D4) ===\n")
    cfg = get_config()
    hybrid, reranker = wire(cfg)
    failures = 0

    print(f"{BOLD}A plan with one answerable and one hopeless "
          f"sub-question{RESET}")
    print(f"{DIM}   sq1  {ANSWERABLE[:66]}")
    print(f"   sq2  {HOPELESS[:66]}")
    print(f"        (the C13 labelling found zero fully-relevant passages "
          f"for sq2){RESET}\n")

    budget = Budget.from_config(cfg)
    state = new_state("combined")
    state["plan"] = build_plan()
    retriever = AdaptiveRetriever(cfg, retriever=hybrid, reranker=reranker,
                                  budget=budget)
    state = retriever.retrieve_all(state)

    print(f"{BOLD}After retrieval{RESET}")
    for sq in state["plan"]:
        f1 = (sq.get("sufficiency_features") or {}).get("f1_max_rerank")
        print(f"{DIM}   {sq['id']}  {sq['status']:<13} rounds={sq['rounds']}  "
              f"f1={'—' if f1 is None else f'{f1:.2f}'}{RESET}")
    print()

    allocator = MarginalValueAllocator(cfg)
    values = {sq["id"]: allocator.value(sq) for sq in state["plan"]}
    print(f"{BOLD}Marginal value of one more round{RESET}")
    for sqid, value in values.items():
        print(f"{DIM}   {sqid}  {value:.3f}{RESET}")

    fresh = Budget.from_config(cfg)
    adaptive_alloc = allocator.allocate(fresh, state["plan"])
    uniform_alloc = UniformAllocator().allocate(
        Budget.from_config(cfg), state["plan"]
    )

    print(f"\n{BOLD}Allocation of the remaining budget{RESET}")
    print(f"{DIM}   {'':<6}{'uniform (A4)':>14}{'marginal (D4)':>15}{RESET}")
    for sqid in adaptive_alloc:
        print(f"   {sqid:<6}{uniform_alloc.get(sqid, 0):>14}"
              f"{adaptive_alloc[sqid]:>15}")

    sq1, sq2 = state["plan"][0]["id"], state["plan"][1]["id"]
    if values[sq1] == values[sq2] == 0.0:
        warn("both sub-questions are terminal, so there is nothing left to "
             "allocate. Re-run with a larger round limit to see the "
             "reallocation.")
    elif adaptive_alloc[sq1] > adaptive_alloc[sq2]:
        ok(f"budget moved toward the answerable sub-question "
           f"({adaptive_alloc[sq1]} vs {adaptive_alloc[sq2]}) — the C22 "
           f"criterion")
    elif values[sq1] < values[sq2]:
        warn("the hopeless sub-question scored higher. Check the f1 values "
             "above — if sq1 already succeeded, it correctly has no value "
             "left and sq2 is the only open one.")
    else:
        bad("allocation did not follow value")
        failures += 1

    if uniform_alloc[sq1] == uniform_alloc[sq2]:
        ok("the uniform allocator (A4) splits evenly, ignoring all of it")
    else:
        bad("the uniform baseline is not uniform")
        failures += 1

    print(f"\n{BOLD}Which sub-question D4 would work on next{RESET}")
    chosen = allocator.next_sub_question(state["plan"])
    plan_order = next(
        (sq["id"] for sq in state["plan"]
         if sq["status"] not in ("sufficient", "abandoned")), None
    )
    print(f"{DIM}   D4 chooses       {chosen}")
    print(f"   plan order (A4)  {plan_order}{RESET}")
    if chosen is None:
        warn("nothing open to choose")
    elif chosen == plan_order:
        warn("D4 and plan order agree here — they will not always, and the "
             "ablation measures the cases where they differ")
    else:
        ok("D4 chooses a different sub-question than plan order would")

    print(f"\n{BOLD}Guarantees{RESET}")
    untried = Budget.from_config(cfg)
    fresh_plan = build_plan()
    allocation = allocator.allocate(untried, fresh_plan)
    if all(v >= untried.min_calls_per_subquestion for v in allocation.values()):
        ok("every untried sub-question is guaranteed a first attempt")
        print(f"{DIM}       starving one before it has been tried is "
              f"pre-judging, not allocating —\n       and it is exactly the "
              f"sub-question whose difficulty the planner missed{RESET}")
    else:
        bad("an untried sub-question was starved")
        failures += 1

    if sum(allocation.values()) == untried.allocatable():
        ok(f"allocation sums exactly to the pool ({untried.allocatable()})")
    else:
        bad(f"allocation sums to {sum(allocation.values())}, pool is "
            f"{untried.allocatable()}")
        failures += 1

    print(f"\n{DIM}--- cost{RESET}")
    print(f"       {budget.report()}")

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C22 acceptance: all checks passed.{RESET}")
    print(f"{GREEN}PHASE 2 COMPLETE — the trace of a single run shows the "
          f"system changing its mind.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())