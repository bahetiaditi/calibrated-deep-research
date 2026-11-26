#!/usr/bin/env python3
"""Run the real NLI contradiction scan on real retrieved passages.

    python -m scripts.probe_contradiction
    python -m scripts.probe_contradiction "your query"

ZERO API calls: hybrid retrieval, cross-encoder rerank and NLI all run
locally. Written at C28.5, when contradiction scanning was first wired into
the graph. The unit tests use a fake NLI model; this is the only check of
the real checkpoint's behaviour on real text — label order, the 0.60
threshold, and whether a 400-character window carries enough of a claim to
judge (the limitation the detector's docstring records).

Scores EVERY cross-source pair (threshold 0 internally) and marks those at
or above the configured threshold, so near-misses are visible too. Three
queries: two where the corpus plausibly disagrees, and a consensus control.
If the control scores as high as the contested ones, f5 is measuring NLI
noise, not disagreement — and that must be known before C33 fits on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.config import get_config

RED, GREEN, YELLOW, DIM, BOLD, RESET = (
    "\033[31m", "\033[32m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)
COLLECTION = "retrieval_eval"

CASES = [
    ("What LoRA rank is needed for fine-tuning to match full fine-tuning?",
     "contested"),
    ("What is the compute-optimal ratio of training tokens to model parameters?",
     "contested"),
    ("How does PagedAttention store the KV cache in memory?", "control"),
]


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
    hybrid.sparse.build([dict(r.payload) for r in records if r.payload])
    return hybrid, Reranker(cfg)


def snippet(p) -> str:
    return " ".join((p.get("text") or "").split())[:150]


def main() -> int:
    from src.rag.contradiction import ContradictionDetector

    cfg = get_config()
    threshold = float(cfg.get("retrieval.contradiction.contradiction_threshold", 0.6))
    top_k = int(cfg.get("retrieval.contradiction.scan_top_k", 8))
    hybrid, reranker = wire(cfg)
    # Threshold 0: every scored pair comes back with its score.
    everything = ContradictionDetector(cfg.with_overrides(
        {"retrieval.contradiction.contradiction_threshold": 0.0}))

    cases = [(sys.argv[1], "query")] if len(sys.argv) > 1 else CASES
    rates: dict[str, list[float]] = {}
    for query, kind in cases:
        print(f"\n{BOLD}{'=' * 78}\n[{kind}] {query}{RESET}")
        results = reranker.rerank(query, hybrid.retrieve(query, top_k=30), top_k=top_k)
        passages = [r.passage for r in results]
        by_id = {p["id"]: p for p in passages}
        for p in passages:
            print(f"  {DIM}{p['id']} {p.get('source_id', '')[:40]:<40} "
                  f"{snippet(p)[:90]}{RESET}")

        scan = everything.scan(passages, sub_question_id="probe")
        if scan.failed:
            print(f"  {RED}NLI failed: {scan.error}{RESET}")
            return 1
        hits = [r for r in scan.records if r.score >= threshold]
        rate = len(hits) / scan.pairs if scan.pairs else 0.0
        rates.setdefault(kind, []).append(rate)
        print(f"\n  {scan.pairs} cross-source pairs scored; "
              f"{len(hits)} at >= {threshold:.2f}  ->  f5 = {rate:.3f}")
        for r in scan.records[:6]:
            mark = f"{RED}CONTRADICTS{RESET}" if r.score >= threshold else f"{DIM}below{RESET}"
            print(f"\n  {r.score:.3f} {mark}")
            print(f"    A {by_id[r.passage_a].get('source_id', '')[:40]}: "
                  f"{DIM}{snippet(by_id[r.passage_a])}{RESET}")
            print(f"    B {by_id[r.passage_b].get('source_id', '')[:40]}: "
                  f"{DIM}{snippet(by_id[r.passage_b])}{RESET}")

    if "contested" in rates and "control" in rates:
        contested = max(rates["contested"])
        control = max(rates["control"])
        print(f"\n{BOLD}contested f5 max {contested:.3f}  vs  control f5 {control:.3f}{RESET}")
        if control >= contested:
            print(f"  {YELLOW}WARN{RESET} the control scores as high as the contested "
                  f"questions — f5 may be measuring NLI noise")
        elif control > 0.2:
            print(f"  {YELLOW}WARN{RESET} the control's f5 is high in absolute terms")
        else:
            print(f"  {GREEN}OK{RESET} contested questions score above the control")
    print("\nRead the pairs, not only the rates: a 'contradiction' between a "
          "claim and an unrelated passage is a false positive whatever its score.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
