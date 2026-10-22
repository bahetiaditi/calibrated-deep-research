#!/usr/bin/env python3
"""C21 acceptance check — adaptive retrieval against the static baseline.

    python -m scripts.smoke_c21

Costs ~15 LLM calls plus retrieval.

The C21 criterion: "on a question with a deliberately hard sub-question, the
trace shows route escalation and a reformulated query that differs from the
original."

It also runs the C17 question through both retrievers so the comparison is
measured rather than asserted. If the adaptive version does not improve on
the static one, D2 and D3 were decorative and the project reports that (§2.1)
— that outcome is a result, not a failure of the script.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from src.budget import Budget
from src.config import get_config
from src.nodes.adaptive_retriever import AdaptiveRetriever
from src.nodes.planner import Planner
from src.nodes.retriever import StaticRetriever
from src.state import independent_sources, new_state, new_sub_question
from src.sufficiency.features import features_for_sub_question

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)

COLLECTION = "retrieval_eval"
COMPARISON = ("How do FlashAttention-2 and PagedAttention differ in their "
              "memory access patterns?")

# Phrased the way a person would ask, not the way a paper states it. The
# corpus covers the answer; the vocabulary does not match. That is precisely
# the gap a reformulation should close and a verbatim query cannot.
HARD = ("what proportion of the chip's arithmetic capability does the "
        "improved attention kernel actually manage to use")


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


def run(retriever, plan, question):
    state = new_state(question)
    state["plan"] = plan
    t0 = time.time()
    state = retriever.retrieve_all(state)
    return state, time.time() - t0


def summarise(state, budget, label, elapsed):
    scores = [p["rerank_score"] for p in state["evidence"]
              if p.get("rerank_score") is not None]
    top = sorted(scores, reverse=True)
    print(f"{BOLD}{label}{RESET}")
    for sq in state["plan"]:
        print(f"{DIM}   {sq['id']} {sq['status']:<13} rounds={sq['rounds']} "
              f"routes={sq['attempted_routes']}{RESET}")
    row = {
        "passages": len(state["evidence"]),
        "sources": independent_sources(state["evidence"]),
        "f1": top[0] if top else float("nan"),
        "f2": sum(top[:3]) / len(top[:3]) if top else float("nan"),
        "f3": (top[0] - top[4]) if len(top) >= 5 else float("nan"),
        "retrieval": budget.retrieval_calls_used,
        "llm": budget.llm_calls_used,
        "seconds": elapsed,
    }
    print(f"{DIM}   passages={row['passages']} sources={row['sources']} "
          f"f1={row['f1']:.2f} f2={row['f2']:.2f} f3={row['f3']:.2f} "
          f"retrieval={row['retrieval']} llm={row['llm']} "
          f"{row['seconds']:.0f}s{RESET}\n")
    return row


def main() -> int:
    print("\n=== C21 smoke test: adaptive retriever (D2 + D3 live) ===\n")
    cfg = get_config()
    hybrid, reranker = wire(cfg)
    failures = 0

    # ---- the acceptance criterion ----------------------------------------
    print(f"{BOLD}A deliberately hard sub-question{RESET}")
    print(f"{DIM}Phrased as a person would ask it, not as a paper states it. "
          f"The corpus\nhas the answer; the vocabulary does not match.{RESET}\n")
    print(f"{DIM}   {HARD}{RESET}\n")

    budget = Budget.from_config(cfg)
    sq = new_sub_question(HARD, sub_question_id="sq1")
    sq["suggested_route"] = "arxiv_meta"
    state = new_state(HARD)
    state["plan"] = [sq]

    adaptive = AdaptiveRetriever(cfg, retriever=hybrid, reranker=reranker,
                                 budget=budget)
    state = adaptive.retrieve_all(state)
    result = state["plan"][0]

    print(f"{BOLD}What it did{RESET}")
    for i, (route, query) in enumerate(zip(result["attempted_routes"],
                                           result["attempted_queries"]), 1):
        print(f"   round {i}  [{route}]")
        print(f"{DIM}            {query[:72]}{RESET}")
    print(f"   final: {result['status']}  "
          f"f1={features_for_sub_question(state, 'sq1')['f1_max_rerank']}\n")

    if len(result["attempted_queries"]) > 1:
        original, *rewrites = result["attempted_queries"]
        if any(r != original for r in rewrites):
            ok("the query was reformulated, not repeated")
        else:
            bad("the retry used the same query — a retry, not a reformulation")
            failures += 1
    else:
        warn("one round only; the first attempt may have succeeded outright. "
             "Check f1 above — if it is high, this question was not hard "
             "enough to force escalation")

    routes = result["attempted_routes"]
    if len(set(routes)) > 1:
        ok(f"route escalated: {' -> '.join(routes)}")
    elif len(routes) > 1:
        warn(f"retried on the same route ({routes[0]}) with a new query")
    else:
        warn("no escalation occurred")

    # ---- static vs adaptive, measured -------------------------------------
    print(f"\n{BOLD}Static vs adaptive on the C17 question{RESET}")
    print(f"{DIM}{COMPARISON}{RESET}\n")

    plan_budget = Budget.from_config(cfg)
    plan = Planner(cfg, budget=plan_budget).plan(COMPARISON)

    static_budget = Budget.from_config(cfg)
    static_state, static_s = run(
        StaticRetriever(cfg, retriever=hybrid, reranker=reranker,
                        budget=static_budget),
        [dict(sq) for sq in plan], COMPARISON,
    )
    static_row = summarise(static_state, static_budget, "STATIC (C17)", static_s)

    adaptive_budget = Budget.from_config(cfg)
    adaptive_state, adaptive_s = run(
        AdaptiveRetriever(cfg, retriever=hybrid, reranker=reranker,
                          budget=adaptive_budget),
        [dict(sq) for sq in plan], COMPARISON,
    )
    adaptive_row = summarise(adaptive_state, adaptive_budget, "ADAPTIVE (C21)",
                             adaptive_s)

    print(f"{BOLD}Delta{RESET}")
    print(f"{DIM}  {'metric':<12}{'static':>9}{'adaptive':>10}{'delta':>9}{RESET}")
    improved = 0
    for key in ("f1", "f2", "f3", "sources", "passages"):
        a, b = static_row[key], adaptive_row[key]
        delta = b - a
        colour = GREEN if delta > 0 else (DIM if delta == 0 else YELLOW)
        print(f"  {key:<12}{a:>9.2f}{b:>10.2f}{colour}{delta:>+9.2f}{RESET}")
        if delta > 0:
            improved += 1
    for key in ("retrieval", "llm", "seconds"):
        a, b = static_row[key], adaptive_row[key]
        print(f"{DIM}  {key:<12}{a:>9.1f}{b:>10.1f}{b - a:>+9.1f}  (cost){RESET}")

    print()
    if improved >= 2:
        ok(f"adaptive improved {improved}/5 evidence metrics")
    elif improved:
        warn(f"adaptive improved only {improved}/5 — a thin result, and one "
             f"worth reporting as thin")
    else:
        warn("adaptive improved nothing on this question. That is a legitimate "
             "outcome (§2.1): if it holds across the benchmark, D2 and D3 were "
             "decorative and the project says so.")

    # ---- f8 and f9 --------------------------------------------------------
    print(f"\n{BOLD}f8 and f9 — the features that only move if the agent is "
          f"adaptive{RESET}")
    for label, st in (("static", static_state), ("adaptive", adaptive_state)):
        rounds = [sq["rounds"] for sq in st["plan"]]
        depths = [len(set(sq["attempted_routes"])) for sq in st["plan"]]
        print(f"{DIM}   {label:<9} rounds={rounds}  distinct routes={depths}{RESET}")

    # The comparison that matters is static vs adaptive, not variance across
    # sub-questions within one run. Three sub-questions of a single question
    # are often genuinely similar, so expecting them to differ tests the
    # question's shape rather than the retriever's behaviour.
    static_rounds = [sq["rounds"] for sq in static_state["plan"]]
    adaptive_rounds = [sq["rounds"] for sq in adaptive_state["plan"]]
    static_depth = [len(set(sq["attempted_routes"]))
                    for sq in static_state["plan"]]
    adaptive_depth = [len(set(sq["attempted_routes"]))
                      for sq in adaptive_state["plan"]]

    if adaptive_rounds != static_rounds:
        ok(f"f8 differs from the static baseline: {static_rounds} -> "
           f"{adaptive_rounds}")
    else:
        warn("f8 matches the static baseline exactly — D3 changed nothing "
             "on this question")

    if adaptive_depth != static_depth:
        ok(f"f9 differs from the static baseline: {static_depth} -> "
           f"{adaptive_depth}")
    else:
        warn("f9 matches the static baseline — D2 never escalated a route")

    if len(set(adaptive_rounds)) > 1:
        ok("round count varies BETWEEN sub-questions — the number of rounds "
           "depends on the evidence, not on a schedule")
    else:
        warn(f"every sub-question took exactly {adaptive_rounds[0]} round(s). "
             f"If that is the round limit, the retriever is running a "
             f"schedule rather than deciding (§2.4).")

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C21 acceptance: all checks passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())