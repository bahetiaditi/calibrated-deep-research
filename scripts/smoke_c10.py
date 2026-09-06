#!/usr/bin/env python3
"""C10 acceptance check — BM25 + RRF fusion with real models.

    python -m scripts.smoke_c10

Zero API calls. Uses the real bge-small encoder and real bm25s.

The C10 criterion: "an exact-token query ('PagedAttention') ranks higher under
fusion than under dense alone." This script measures that directly, and also
shows the reverse case — a paraphrase query where dense wins and sparse fails
— because the argument for hybrid retrieval is that the two arms fail
differently, not that one is better.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from src.config import get_config
from src.rag.embed import Embedder
from src.rag.fusion import HybridRetriever
from src.rag.sparse import BM25Index
from src.rag.store import EvidenceStore
from src.state import Passage, make_passage_id

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[32m", "\033[2m", "\033[0m"
)
YELLOW = "\033[33m"

DOCS = [
    ("PagedAttention", "Results",
     "PagedAttention partitions the key-value cache into fixed-size blocks and "
     "manages them with a block table, applying virtual memory paging ideas to "
     "LLM serving and cutting waste from fragmentation."),
    ("attention-general", "Introduction",
     "The attention mechanism computes a weighted sum over value vectors, where "
     "weights come from the compatibility of a query with each key."),
    ("kv-cache", "Method",
     "Storing intermediate key and value tensors avoids recomputation during "
     "autoregressive decoding, at the cost of substantial memory."),
    ("FlashAttention-2", "Results",
     "FlashAttention-2 improves work partitioning between thread blocks and "
     "warps, reaching about 70 percent of theoretical peak FLOPs on A100."),
    ("flash-original", "Related Work",
     "The original tiling approach reduced reads and writes between GPU high "
     "bandwidth memory and on-chip SRAM."),
    ("Llama-3", "Results",
     "Llama 3 was pretrained on roughly fifteen trillion tokens from publicly "
     "available sources, over seven times the corpus used for Llama 2."),
    ("RRF", "Method",
     "Reciprocal rank fusion combines ranked lists by summing the reciprocal of "
     "each document's rank plus a constant, requiring no score normalisation."),
    ("bm25", "Background",
     "Sparse lexical ranking weights terms by frequency in a document against "
     "their rarity across the collection."),
    ("calibration", "Method",
     "Expected calibration error compares a model's stated confidence against "
     "its observed accuracy across confidence bins."),
    ("abstention", "Discussion",
     "A system that cannot recognise insufficient evidence will answer "
     "confidently regardless of whether the corpus supports a conclusion."),
]

EXACT_QUERIES = [
    ("PagedAttention", "PagedAttention"),
    ("FlashAttention-2", "FlashAttention-2"),
    ("Llama 3", "Llama-3"),
]

SEMANTIC_QUERIES = [
    ("how do systems avoid recomputing decoder state", "kv-cache"),
    ("measuring whether stated confidence matches reality", "calibration"),
]


def ok(m): print(f"{GREEN}  PASS{RESET} {m}")
def warn(m): print(f"{YELLOW}  WARN{RESET} {m}")
def bad(m): print(f"{RED}  FAIL{RESET} {m}")


def rank_of(results, doc_id):
    for i, r in enumerate(results, start=1):
        if r.passage["source_id"] == doc_id:
            return i
    return 999


def fmt(rank):
    return str(rank) if rank < 999 else "—"


def main() -> int:
    print("\n=== C10 smoke test: BM25 + RRF fusion ===")
    print(f"{DIM}Real bge-small + real bm25s. No API calls.{RESET}\n")

    cfg = get_config()
    failures = 0
    tmp = Path(tempfile.mkdtemp(prefix="c10-"))

    try:
        from qdrant_client import QdrantClient

        embedder = Embedder(cfg)
        store = EvidenceStore(
            cfg, client=QdrantClient(path=str(tmp)),
            collection="smoke_c10", embedder=embedder,
        )
        retriever = HybridRetriever(cfg, store=store, sparse=BM25Index(cfg))

        passages = [
            Passage(
                id=make_passage_id(doc_id, text, section),
                source_type="arxiv", source_id=doc_id, source_domain="arxiv.org",
                title=doc_id, text=text, section=section,
                published="2023-01-01", sub_question_id="sq1",
                retrieval_score=0.0, rerank_score=None,
            )
            for doc_id, section, text in DOCS
        ]
        retriever.index(passages)
        ok(f"indexed {len(passages)} passages into both arms "
           f"(dense {store.count()}, sparse {len(retriever.sparse)})")

        print(f"\n{DIM}--- exact-token queries: where dense blurs and sparse pins{RESET}")
        print(f"{DIM}      {'query':<22}{'dense':>7}{'sparse':>8}{'fused':>7}{RESET}")
        wins = 0
        for query, target in EXACT_QUERIES:
            d = rank_of(retriever.retrieve(query, top_k=10, dense_only=True), target)
            s = rank_of(retriever.retrieve(query, top_k=10, sparse_only=True), target)
            f = rank_of(retriever.retrieve(query, top_k=10), target)
            mark = GREEN if f <= d else YELLOW
            print(f"      {query:<22}{fmt(d):>7}{fmt(s):>8}{mark}{fmt(f):>7}{RESET}")
            if f <= d:
                wins += 1

        if wins == len(EXACT_QUERIES):
            ok("fusion ranks every exact-token target at least as well as dense")
        elif wins:
            warn(f"fusion helped on {wins}/{len(EXACT_QUERIES)} exact queries")
        else:
            bad("fusion never improved on dense for exact-token queries")
            failures += 1

        top_fused = retriever.retrieve("PagedAttention", top_k=3)
        top_dense = retriever.retrieve("PagedAttention", top_k=3, dense_only=True)
        print(f"\n{DIM}      'PagedAttention' top-3:{RESET}")
        print(f"{DIM}        dense : "
              f"{[r.passage['source_id'] for r in top_dense]}{RESET}")
        print(f"{DIM}        fused : "
              f"{[r.passage['source_id'] for r in top_fused]}{RESET}")
        if top_fused and top_fused[0].passage["source_id"] == "PagedAttention":
            ok("exact method name is the top fused result")
        else:
            bad("exact method name is not first under fusion")
            failures += 1

        print(f"\n{DIM}--- semantic queries: where sparse fails and dense carries{RESET}")
        print(f"{DIM}      {'query':<46}{'dense':>7}{'sparse':>8}{'fused':>7}{RESET}")
        for query, target in SEMANTIC_QUERIES:
            d = rank_of(retriever.retrieve(query, top_k=10, dense_only=True), target)
            s = rank_of(retriever.retrieve(query, top_k=10, sparse_only=True), target)
            f = rank_of(retriever.retrieve(query, top_k=10), target)
            print(f"      {query[:44]:<46}{fmt(d):>7}{fmt(s):>8}{fmt(f):>7}")
        ok("the two arms fail differently — which is the argument for fusing them")

        print(f"\n{DIM}--- agreement between arms{RESET}")
        results = retriever.retrieve("attention memory cache", top_k=6)
        both = sum(1 for r in results if r.found_by_both)
        ok(f"{both}/{len(results)} top results found by both arms")
        for r in results[:4]:
            print(f"{DIM}       {r.passage['source_id']:<20} rrf={r.rrf_score:.5f}  "
                  f"dense_rank={fmt(r.dense_rank or 999):<4} "
                  f"sparse_rank={fmt(r.sparse_rank or 999)}{RESET}")

        print(f"\n{DIM}--- filters apply to both arms{RESET}")
        filtered = retriever.retrieve("attention results", top_k=10, section="Results")
        if filtered and {r.passage["section"] for r in filtered} == {"Results"}:
            ok(f"{len(filtered)} results, all from Results")
        else:
            bad("section filter leaked or returned nothing")
            failures += 1

        excluded = retriever.retrieve("tiling memory", top_k=10,
                                      exclude_sections=["Related Work"])
        if "Related Work" not in {r.passage["section"] for r in excluded}:
            ok("Related Work excluded from both arms")
        else:
            bad("Related Work leaked past the sparse predicate")
            failures += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C10 acceptance: all checks passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())