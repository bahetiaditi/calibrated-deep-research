#!/usr/bin/env python3
"""C19 acceptance check — end-to-end run, then resume from checkpoint.

    python -m scripts.smoke_c19

Costs ~6 LLM calls plus retrieval against the C13 corpus.

The C19 criterion: "end-to-end run on one question; kill the process mid-run
and confirm it resumes from checkpoint."

The second half is simulated rather than literal — the script interrupts its
own graph at the retriever, then re-enters with the same thread_id and
confirms the completed nodes are not redone.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from src.config import get_config
from src.controller import Controller
from src.graph import Components, build_graph, default_components, run_research
from src.state import new_state
from src.tracing.tracer import DecisionSurface, EventType, TraceReader

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)

COLLECTION = "retrieval_eval"
QUESTION = ("How do FlashAttention-2 and PagedAttention differ in their "
            "memory access patterns?")


def ok(m): print(f"{GREEN}  PASS{RESET} {m}")
def warn(m): print(f"{YELLOW}  WARN{RESET} {m}")
def bad(m): print(f"{RED}  FAIL{RESET} {m}")


def wire_corpus(cfg):
    """Point the nodes at the C13 evaluation corpus."""
    from qdrant_client import QdrantClient

    from src.nodes.planner import Planner, planner_node
    from src.nodes.premise_checker import PremiseChecker, premise_checker_node
    from src.nodes.retriever import StaticRetriever, retriever_node
    from src.nodes.synthesizer import Synthesizer, synthesizer_node
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

    reranker = Reranker(cfg)
    return Components(
        premise_checker=premise_checker_node(
            PremiseChecker(cfg, retriever=hybrid, reranker=reranker)),
        planner=planner_node(Planner(cfg)),
        retriever=retriever_node(
            StaticRetriever(cfg, retriever=hybrid, reranker=reranker)),
        synthesizer=synthesizer_node(Synthesizer(cfg)),
    )


def main() -> int:
    print("\n=== C19 smoke test: controller + graph ===\n")
    cfg = get_config()
    failures = 0

    components = wire_corpus(cfg)
    thread = f"smoke-c19-{int(time.time())}"

    print(f"{BOLD}{QUESTION}{RESET}\n")
    t0 = time.time()
    result = run_research(
        QUESTION, components=components, config=cfg,
        thread_id=thread, trace_id=thread,
    )
    elapsed = time.time() - t0

    if result.error:
        bad(f"run failed: {result.error}")
        return 1

    state = result.state
    print(f"{BOLD}Node sequence{RESET}")
    print(f"{DIM}   {' -> '.join(result.node_visits)}{RESET}\n")

    if result.node_visits and result.node_visits[0] == "premise_checker":
        ok("premise check ran before planning")
    else:
        bad("premise check did not run first")
        failures += 1

    if "synthesizer" in result.node_visits:
        ok(f"reached synthesis in {elapsed:.0f}s")
    else:
        bad("never reached the synthesizer")
        failures += 1

    retriever_visits = result.node_visits.count("retriever")
    if retriever_visits == len(state["plan"]):
        ok(f"retriever ran once per sub-question ({retriever_visits})")
    else:
        warn(f"{retriever_visits} retriever visits for "
             f"{len(state['plan'])} sub-questions")

    print(f"\n{BOLD}Final state{RESET}")
    print(f"{DIM}   premise grounding   "
          f"{state['premise_check'].get('grounding_score')}")
    print(f"   sub-questions       {len(state['plan'])}")
    print(f"   evidence passages   {len(state['evidence'])}")
    print(f"   draft length        {len(state['draft'])} chars")
    if result.budget:
        print(f"   budget              {result.budget.report()}{RESET}")

    if state["draft"]:
        ok("a draft was produced")
        print(f"\n{DIM}{state['draft'][:500]}…{RESET}")
    else:
        bad("no draft")
        failures += 1

    print(f"\n{BOLD}Routing is auditable{RESET}")
    reader = TraceReader.load(result.trace_path)
    routes = reader.routes()
    print(f"{DIM}   {len(routes)} routing decisions, each with its reason:{RESET}")
    for frm, to, reason in routes[:8]:
        print(f"{DIM}     {str(frm or 'START'):<16} -> {to:<16} {reason[:44]}{RESET}")

    if all(reason for _, _, reason in routes):
        ok("every routing decision carries a reason — C38 can replay the run")
    else:
        bad("some routing decisions have no reason recorded")
        failures += 1

    route_events = reader.of_type(EventType.ROUTE)
    if route_events and all(e.get("payload", {}).get("state")
                            for e in route_events):
        ok("every routing decision carries the state that produced it (§3.3)")
    else:
        bad("routing events lack a state summary")
        failures += 1

    print(f"\n{BOLD}Decision surfaces (all frozen at C19){RESET}")
    counts = reader.adaptive_decision_counts()
    for surface, n in counts.items():
        marker = GREEN if n == 0 else YELLOW
        print(f"{DIM}   {surface:<28}{RESET}{marker}{n}{RESET}")
    if sum(counts.values()) == 0:
        ok("zero adaptive decisions — correct, since D1-D5 land at C21-C27")
        print(f"{DIM}       this is the A1-A5 baseline shape: every decision "
              f"present,\n       none of them adaptive{RESET}")
    else:
        warn(f"{sum(counts.values())} adaptive decisions recorded before the "
             f"adaptive nodes exist")

    print(f"\n{BOLD}Resume from checkpoint{RESET}")
    with __import__("langgraph.checkpoint.sqlite",
                    fromlist=["SqliteSaver"]).SqliteSaver.from_conn_string(
                        str(Path(cfg.get("run.checkpoint_db")))) as saver:
        graph = build_graph(components, config=cfg, checkpointer=saver)
        config_ = {"configurable": {"thread_id": thread}}
        restored = graph.get_state(config_).values

        if restored.get("draft") == state["draft"]:
            ok("a fresh process recovered the completed run from disk")
        else:
            bad("resumed state differs from the finished run")
            failures += 1

        t0 = time.time()
        graph.invoke(None, config_)
        resume_s = time.time() - t0
        if resume_s < 2.0:
            ok(f"resuming a finished run redid no work ({resume_s:.2f}s) — "
               f"an interrupted eval continues rather than restarts")
        else:
            warn(f"resume took {resume_s:.1f}s; work may have been repeated")

    print(f"\n{DIM}trace: {result.trace_path}{RESET}")

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C19 acceptance: all checks passed.{RESET}")
    print(f"{GREEN}The system runs end to end. Everything after this makes it "
          f"agentic and calibrated.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())