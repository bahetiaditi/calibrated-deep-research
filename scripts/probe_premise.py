#!/usr/bin/env python3
"""Run ONLY the premise checker, and show what the verifier saw.

    python -m scripts.probe_premise
    python -m scripts.probe_premise "Why does X use Y?"

Two judgment calls per question (extract, verify), pinned to gpt-oss-120b so
a verdict is never a fallback model's. If 120b is out for the day the call
fails loudly — wait for the reset rather than reading a qwen verdict.

Written at C28.1. The C28 live run's verifier answered "the evidence does
not mention Mamba interleaving multi-head attention layers" while — by its
own citations for a sibling presupposition — Mamba passages retrieved for
that very query were in front of it. The full graph costs ~20 calls to
re-test that; this costs 2, and prints every passage with negations marked
so retrieval failure and verification failure cannot be confused.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from src.config import get_config

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)
COLLECTION = "retrieval_eval"
PIN = "openai/gpt-oss-120b"

# (question, what the premise check should conclude on this corpus)
CASES = [
    ("Why does Mamba interleave multi-head attention layers between its "
     "selective SSM blocks?", "CONTRADICTED"),
    ("Why does Mamba-2 use rotary position embeddings in its SSM blocks?",
     "UNVERIFIABLE"),     # the corpus is silent on Mamba-2 (C27.1)
    ("How do FlashAttention-2 and PagedAttention differ in their memory "
     "access patterns?", "SUPPORTED"),
]

NEGATION = re.compile(
    r"\b(without|not use[sd]?|does not|no \w+ (?:layers?|blocks?)|\w+-free|free of)\b",
    re.I,
)


def wire(cfg):
    from qdrant_client import QdrantClient

    from src.nodes.premise_checker import PremiseChecker
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
    hybrid.sparse.build([dict(r.payload) for r in records if r.payload])
    return PremiseChecker(cfg, retriever=hybrid, reranker=Reranker(cfg))


def mark(text: str) -> str:
    return NEGATION.sub(lambda m: f"{YELLOW}{BOLD}{m.group(0)}{RESET}{DIM}", text)


def main() -> int:
    cfg = get_config().with_overrides({"llm.pin.judgment": PIN})
    checker = wire(cfg)
    if "--tier-d1" in sys.argv:
        # C30: every benchmark false premise, which the corpus refutes by
        # construction. A MISS here does not invalidate the item — it is
        # still ill-posed — but it means the run will reach abstention, if
        # at all, the slow way (research, then threshold) instead of by rule.
        import json
        questions = json.loads((Path(__file__).resolve().parents[1]
                                / "benchmark" / "questions.json").read_text())["questions"]
        cases = [(q["question"], "CONTRADICTED") for q in questions if q.get("subtype") == "D1"]
    elif len(sys.argv) > 1:
        cases = [(sys.argv[1], None)]
    else:
        cases = CASES
    misses = 0

    for question, expected in cases:
        print(f"\n{BOLD}{'=' * 78}\n{question}{RESET}")
        result = checker.check(question)
        if result.error:
            print(f"{RED}premise check failed: {result.error}{RESET}")
            misses += 1
            continue

        evidence = checker._gather(result.presuppositions)  # same retrieval, free
        for pre in result.presuppositions:
            colour = {"CONTRADICTED": RED, "SUPPORTED": GREEN}.get(pre.verdict, YELLOW)
            print(f"\n  {colour}{pre.verdict:<13}{RESET} ({pre.centrality}) {pre.text}")
            print(f"  {DIM}reasoning: {pre.reasoning}{RESET}")
            if pre.correction:
                print(f"  {DIM}correction: {pre.correction}{RESET}")
            print(f"  {DIM}cited: {pre.cited_passage_ids}{RESET}")
        print(f"\n  {BOLD}evidence the verifier saw{RESET} "
              f"({len(evidence)} passages, labelled in presupposition order)")
        for label, p in evidence:
            snippet = " ".join((p.get("text") or "").split())[:260]
            print(f"  {label} {DIM}{p.get('source_id')} [{p.get('section')}] "
                  f"{mark(snippet)}{RESET}")

        verdicts = {pre.verdict for pre in result.presuppositions if pre.is_core}
        print(f"\n  failed={result.failed}  f7={result.grounding_score:.2f}")
        if expected == "CONTRADICTED":
            if result.failed:
                print(f"  {GREEN}PASS{RESET} refuted — the hard rule will fire")
            else:
                print(f"  {RED}MISS{RESET} not refuted")
                misses += 1
        elif expected == "UNVERIFIABLE":
            if result.failed:
                print(f"  {RED}MISS{RESET} refuted on silence — the C15 failure "
                      f"the prompt change must not reintroduce")
                misses += 1
            else:
                print(f"  {GREEN}PASS{RESET} not refuted (corpus is silent)")
        elif expected == "SUPPORTED":
            if result.failed or "CONTRADICTED" in verdicts:
                print(f"  {RED}MISS{RESET} a valid question was refuted")
                misses += 1
            else:
                print(f"  {GREEN}PASS{RESET} valid question passes")

    print(f"\n{BOLD}{'all as expected' if not misses else f'{misses} miss(es)'}{RESET}\n")
    return 1 if misses else 0


if __name__ == "__main__":
    sys.exit(main())
