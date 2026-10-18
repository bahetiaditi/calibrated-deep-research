#!/usr/bin/env python3
"""C17 acceptance check — static retriever over a real plan.

    python -m scripts.smoke_c17

Costs 1 LLM call (the planner) plus retrieval against the C13 corpus.

The C17 criterion: "retrieves for every sub-question in a plan."

It also records the baseline C21 has to beat. The static policy is a fair
opponent on purpose — a crippled baseline would make the adaptive retriever
look good for the wrong reason and leave the agency claim unfalsifiable.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from src.budget import Budget
from src.config import get_config
from src.nodes.planner import Planner
from src.nodes.retriever import StaticRetriever
from src.state import evidence_for, independent_sources, new_state

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)

COLLECTION = "retrieval_eval"
QUESTION = ("How do FlashAttention-2 and PagedAttention differ in their "
            "memory access patterns?")


def ok(m): print(f"{GREEN}  PASS{RESET} {m}")
def warn(m): print(f"{YELLOW}  WARN{RESET} {m}")
def bad(m): print(f"{RED}  FAIL{RESET} {m}")


def build_components(cfg, budget):
    from qdrant_client import QdrantClient

    from src.rag.fusion import HybridRetriever
    from src.rag.rerank import Reranker
    from src.rag.sparse import BM25Index
    from src.rag.store import EvidenceStore

    path = Path(cfg.get("retrieval.store.path")) / COLLECTION
    store = EvidenceStore(cfg, client=QdrantClient(path=str(path)),
                          collection=COLLECTION)
    hybrid = HybridRetriever(cfg, store=store, sparse=BM25Index(cfg))

    records, _ = store.client.scroll(
        collection_name=COLLECTION, limit=10_000, with_payload=True
    )
    passages = [dict(r.payload) for r in records if r.payload]
    if not passages:
        print(f"{RED}corpus is empty — run "
              f"`python -m scripts.retrieval_labels build` first{RESET}")
        sys.exit(1)
    hybrid.sparse.build(passages)
    print(f"{DIM}corpus: {len(passages)} passages{RESET}\n")

    return StaticRetriever(cfg, retriever=hybrid, reranker=Reranker(cfg),
                           budget=budget)


def main() -> int:
    print("\n=== C17 smoke test: static retriever ===")
    print(f"{DIM}D2 and D3 are OFF. This is the A2/A3 baseline that C21 "
          f"must beat.{RESET}\n")

    cfg = get_config()
    budget = Budget.from_config(cfg)
    retriever = build_components(cfg, budget)
    failures = 0

    print(f"{BOLD}{QUESTION}{RESET}\n")
    plan = Planner(cfg, budget=budget).plan(QUESTION)
    for sq in plan:
        print(f"{DIM}   {sq['id']}  {sq['text'][:66]}{RESET}")
    print()

    state = new_state(QUESTION)
    state["plan"] = plan

    t0 = time.time()
    state = retriever.retrieve_all(state)
    elapsed = time.time() - t0

    print(f"{BOLD}Retrieval{RESET}")
    covered = 0
    for sq in state["plan"]:
        evidence = evidence_for(state, sq["id"])
        scores = [p["rerank_score"] for p in evidence
                  if p.get("rerank_score") is not None]
        colour = GREEN if sq["status"] == "sufficient" else YELLOW
        print(f"   {colour}{sq['status']:<13}{RESET}{sq['id']}  "
              f"{len(evidence)} passages  "
              f"max_rerank={max(scores, default=float('nan')):.2f}  "
              f"sources={independent_sources(evidence)}")
        print(f"{DIM}        route={sq['attempted_routes']}  "
              f"rounds={sq['rounds']}{RESET}")
        print(f"{DIM}        query={sq['attempted_queries'][0][:62]!r}{RESET}")
        if evidence:
            covered += 1

    print()
    if all(sq["status"] != "pending" for sq in state["plan"]):
        ok(f"every sub-question processed ({len(state['plan'])}) — the C17 "
           f"criterion")
    else:
        bad("some sub-questions were never retrieved for")
        failures += 1

    if covered == len(state["plan"]):
        ok("every sub-question found evidence")
    else:
        warn(f"{len(state['plan']) - covered} sub-question(s) found nothing — "
             f"with D2 frozen there is no second query to try, which is "
             f"precisely the gap C21 exists to close")

    print(f"\n{BOLD}Static policy held{RESET}")
    routes = {r for sq in state["plan"] for r in sq["attempted_routes"]}
    if len(routes) == 1:
        ok(f"one route used throughout: {routes.pop()} (D2 frozen)")
    else:
        bad(f"route varied: {routes} — D2 should be off here")
        failures += 1

    if all(sq["rounds"] == 1 for sq in state["plan"]):
        ok("exactly one round per sub-question (D3 frozen)")
    else:
        bad("more than one round ran — D3 should be off here")
        failures += 1

    verbatim = all(
        sq["attempted_queries"][0] == sq["text"] for sq in state["plan"]
        if sq["attempted_queries"]
    )
    if verbatim:
        ok("queries were the sub-questions verbatim (no reformulation)")
    else:
        bad("a query was reformulated — D2 should be off here")
        failures += 1

    print(f"\n{BOLD}Baseline for C21 to beat{RESET}")
    all_evidence = state["evidence"]
    all_scores = [p["rerank_score"] for p in all_evidence
                  if p.get("rerank_score") is not None]
    print(f"{DIM}   sub-questions          {len(state['plan'])}")
    print(f"   with evidence          {covered}")
    print(f"   unique passages        {len(all_evidence)}")
    print(f"   independent sources    {independent_sources(all_evidence)}")
    if all_scores:
        top = sorted(all_scores, reverse=True)
        print(f"   max rerank (f1)        {top[0]:.3f}")
        print(f"   mean top-3 (f2)        {sum(top[:3]) / len(top[:3]):.3f}")
        if len(top) >= 5:
            print(f"   gap 1-5 (f3)           {top[0] - top[4]:.3f}")
    print(f"   retrieval calls        {budget.retrieval_calls_used}")
    print(f"   wall clock             {elapsed:.1f}s{RESET}")
    print(f"\n{DIM}   Record these. If C21's adaptive retriever does not "
          f"improve them,\n   D2 and D3 were decorative and the project "
          f"reports that (§2.1).{RESET}")

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C17 acceptance: all checks passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())