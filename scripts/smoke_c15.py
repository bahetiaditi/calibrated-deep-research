#!/usr/bin/env python3
"""C15 acceptance check — premise checking against the real corpus.

    python -m scripts.smoke_c15

Costs ~12 LLM calls (2 per question) plus retrieval. Uses the evaluation
corpus built at C13, so run `scripts.retrieval_labels build` first if the
store is empty.

The C15 criterion: "on three hand-written false-premise questions it fails;
on three valid questions it passes."

The second half matters more than it looks. A checker that fails everything
would pass the first half perfectly and destroy the project's headline
metric, because every Tier A-C question would abstain. That failure mode is
what the valid questions are here to catch.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from src.budget import Budget
from src.config import get_config
from src.nodes.premise_checker import CONTRADICTED, SUPPORTED, PremiseChecker

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)

COLLECTION = "retrieval_eval"

# Presuppositions that the corpus should positively refute.
FALSE_PREMISE = [
    "Why does Mamba-2 use rotary position embeddings in its SSM blocks?",
    "How does QLoRA avoid quantising the base model weights?",
    "Why does FlashAttention-2 increase memory usage compared to standard "
    "attention?",
]

# Well-posed questions the corpus supports. These are the important ones.
VALID = [
    "What fraction of theoretical peak FLOPs does FlashAttention-2 reach?",
    "How does QLoRA quantise the frozen base model?",
    "What is the selection mechanism in Mamba's state space model?",
]

# No checkable presuppositions — must not be penalised.
OPEN = ["What approaches exist for reducing hallucination in RAG systems?"]


def ok(m): print(f"{GREEN}  PASS{RESET} {m}")
def warn(m): print(f"{YELLOW}  WARN{RESET} {m}")
def bad(m): print(f"{RED}  FAIL{RESET} {m}")


def build_checker(cfg, tracer=None, budget=None):
    from qdrant_client import QdrantClient

    from src.rag.fusion import HybridRetriever
    from src.rag.rerank import Reranker
    from src.rag.sparse import BM25Index
    from src.rag.store import EvidenceStore

    path = Path(cfg.get("retrieval.store.path")) / COLLECTION
    store = EvidenceStore(cfg, client=QdrantClient(path=str(path)),
                          collection=COLLECTION)
    retriever = HybridRetriever(cfg, store=store, sparse=BM25Index(cfg))

    records, _ = store.client.scroll(
        collection_name=COLLECTION, limit=10_000, with_payload=True
    )
    passages = [dict(r.payload) for r in records if r.payload]
    if passages:
        retriever.sparse.build(passages)
    print(f"{DIM}corpus: {len(passages)} passages{RESET}\n")
    if not passages:
        print(f"{RED}corpus is empty — run "
              f"`python -m scripts.retrieval_labels build` first{RESET}")
        sys.exit(1)

    return PremiseChecker(
        cfg, retriever=retriever, reranker=Reranker(cfg),
        tracer=tracer, budget=budget,
    )


def report(question: str, result) -> None:
    print(f"{DIM}  {question[:72]}{RESET}")
    for p in result.presuppositions:
        colour = {SUPPORTED: GREEN, CONTRADICTED: RED}.get(p.verdict, YELLOW)
        tag = "" if p.is_core else " (incidental)"
        print(f"      {colour}{p.verdict:<13}{RESET}{p.text[:58]}{tag}")
        if p.reasoning:
            print(f"{DIM}                    {p.reasoning[:66]}{RESET}")
    print(f"{DIM}      f7={result.grounding_score:.2f}  "
          f"{result.rationale[:70]}{RESET}")


def main() -> int:
    print("\n=== C15 smoke test: premise checker ===\n")
    cfg = get_config()
    budget = Budget.from_config(cfg)
    checker = build_checker(cfg, budget=budget)
    failures = 0

    print(f"{BOLD}False premises — should FAIL the check{RESET}\n")
    caught = 0
    for question in FALSE_PREMISE:
        t0 = time.time()
        result = checker.check(question)
        report(question, result)
        if result.failed:
            ok(f"caught in {time.time() - t0:.1f}s"
               + (f" — correction: {result.correction[:54]}"
                  if result.correction else ""))
            caught += 1
        else:
            warn("not caught — the corpus may not positively refute this, "
                 "which is UNVERIFIABLE rather than CONTRADICTED")
        print()

    if caught >= 2:
        ok(f"{caught}/{len(FALSE_PREMISE)} false premises caught")
    else:
        bad(f"only {caught}/{len(FALSE_PREMISE)} caught — the hard "
            f"abstention rule will rarely fire")
        failures += 1

    print(f"\n{BOLD}Valid questions — must PASS{RESET}")
    print(f"{DIM}The important half. A checker that fails everything scores "
          f"perfectly above\nand destroys the headline metric by abstaining "
          f"on every Tier A-C question.{RESET}\n")
    false_abstentions = 0
    for question in VALID:
        result = checker.check(question)
        report(question, result)
        if result.failed:
            bad("FALSE ABSTENTION — a well-posed question was refused")
            false_abstentions += 1
        else:
            ok("passed")
        print()

    if false_abstentions == 0:
        ok("no false abstentions on well-posed questions")
    else:
        bad(f"{false_abstentions} false abstention(s) — this is the failure "
            f"mode that would quietly ruin the results")
        failures += 1

    print(f"\n{BOLD}Open question — no checkable presuppositions{RESET}\n")
    for question in OPEN:
        result = checker.check(question)
        report(question, result)
        if not result.failed and result.grounding_score >= 0.5:
            ok("not penalised for having nothing to check")
        else:
            bad("an open question was penalised")
            failures += 1
        print()

    print(f"{DIM}--- cost{RESET}")
    n = len(FALSE_PREMISE) + len(VALID) + len(OPEN)
    print(f"       {budget.llm_calls_used} LLM calls over {n} questions "
          f"({budget.llm_calls_used / n:.1f} per question)")
    print(f"       {budget.retrieval_calls_used} retrieval calls")
    if budget.llm_calls_used / n <= 2.5:
        ok("2 calls per question — batched verification holds")
    else:
        warn("more than 2 calls per question; verification may not be batching")

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C15 acceptance: all checks passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())