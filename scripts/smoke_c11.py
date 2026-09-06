#!/usr/bin/env python3
"""C11 acceptance check — real bge-reranker-base.

    python -m scripts.smoke_c11

Zero API calls. Downloads ~1.1 GB of model weights on first run, then CPU only.

The C11 criterion: "reranking measurably reorders on a hand-checked example;
latency is acceptable on CPU."

The hand-checked example is built so the right answer is *lexically* poor and
*semantically* right: the correct passage never uses the query's words, while
three distractors are stuffed with them. Fusion should rank it badly and the
cross-encoder should rescue it. If reranking cannot do that, it is not
earning the latency it costs.
"""
from __future__ import annotations

import statistics
import sys
import time

from src.config import get_config
from src.rag.fusion import FusedResult
from src.rag.rerank import Reranker, rerank_features, sigmoid
from src.state import Passage, make_passage_id

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
)

QUERY = "what fraction of peak FLOPs does FlashAttention-2 reach on an A100"

# The target (GOLD) answers the question without repeating its wording.
# The distractors repeat the wording without answering it.
DOCS = [
    ("GOLD", "Results",
     "The improved kernel sustains roughly 70 percent of the theoretical maximum "
     "throughput on NVIDIA A100 hardware, up from about 35 percent for the "
     "previous implementation, by rebalancing work across thread blocks."),
    ("D1", "Introduction",
     "FlashAttention-2 is discussed at length in this section. FlashAttention-2 "
     "builds on FlashAttention. Peak FLOPs and A100 utilisation are common "
     "benchmarks when evaluating attention kernels."),
    ("D2", "Related Work",
     "Prior work on FlashAttention-2 and peak FLOPs on the A100 established that "
     "attention kernels are memory bound. What fraction of peak FLOPs a kernel "
     "reaches depends on the hardware."),
    ("D3", "Background",
     "An A100 has a theoretical peak of 312 TFLOPs in bfloat16. FLOPs are a "
     "measure of arithmetic throughput. FlashAttention-2 targets the A100."),
    ("D4", "Method",
     "We describe the tiling strategy used to keep intermediate values in "
     "on-chip SRAM rather than writing them to high bandwidth memory."),
    ("D5", "Discussion",
     "Reciprocal rank fusion merges ranked lists by summing reciprocal ranks."),
    ("D6", "Conclusion",
     "We have presented a faster attention implementation and released code."),
    ("D7", "Background",
     "Expected calibration error compares stated confidence against observed "
     "accuracy across bins."),
]


def ok(m): print(f"{GREEN}  PASS{RESET} {m}")
def warn(m): print(f"{YELLOW}  WARN{RESET} {m}")
def bad(m): print(f"{RED}  FAIL{RESET} {m}")


def make_results() -> list[FusedResult]:
    """Fusion order deliberately puts the correct passage near the bottom."""
    out = []
    for rank, (doc_id, section, text) in enumerate(DOCS, start=1):
        passage = Passage(
            id=make_passage_id(doc_id, text, section),
            source_type="arxiv", source_id=doc_id, source_domain="arxiv.org",
            title=doc_id, text=text, section=section, published="2023-07-01",
            sub_question_id="sq1", retrieval_score=1.0 / (60 + rank),
            rerank_score=None,
        )
        out.append(FusedResult(passage=passage, rrf_score=1.0 / (60 + rank),
                               dense_rank=rank, sparse_rank=rank))
    # Put GOLD at position 5 to simulate lexical retrieval burying it.
    gold = out.pop(0)
    out.insert(4, gold)
    return out


def position(results, doc_id):
    for i, r in enumerate(results, start=1):
        if r.passage["source_id"] == doc_id:
            return i
    return 999


def main() -> int:
    print("\n=== C11 smoke test: cross-encoder reranking ===")
    print(f"{DIM}First run downloads ~1.1 GB of model weights. CPU only.{RESET}\n")

    cfg = get_config()
    failures = 0

    t0 = time.time()
    reranker = Reranker(cfg)
    candidates = make_results()
    fusion_position = position(candidates, "GOLD")
    print(f"{DIM}--- before reranking (fusion order){RESET}")
    for i, r in enumerate(candidates, start=1):
        mark = "*" if r.passage["source_id"] == "GOLD" else " "
        print(f"{DIM}     {mark}{i}. {r.passage['source_id']:<6} "
              f"[{r.passage['section']}]{RESET}")

    reranked = reranker.rerank(QUERY, candidates, top_k=5)
    load_and_run_s = time.time() - t0
    print(f"\n{DIM}model loaded and first rerank done in {load_and_run_s:.1f}s{RESET}")

    print(f"\n{DIM}--- after reranking{RESET}")
    for i, r in enumerate(reranked, start=1):
        logit = r.passage["rerank_score"]
        mark = "*" if r.passage["source_id"] == "GOLD" else " "
        print(f"     {mark}{i}. {r.passage['source_id']:<6} "
              f"logit={logit:>7.3f}  p={sigmoid(logit):.4f}  "
              f"(was #{r.dense_rank})")

    new_position = position(reranked, "GOLD")
    if new_position == 1:
        ok(f"correct passage moved from #{fusion_position} to #1 — the "
           f"cross-encoder read query and passage together")
    elif new_position < fusion_position:
        warn(f"moved from #{fusion_position} to #{new_position} but not to top")
    else:
        bad(f"reranking did not rescue the correct passage "
            f"(#{fusion_position} -> #{new_position})")
        failures += 1

    reordered = sum(1 for i, r in enumerate(reranked, start=1) if r.dense_rank != i)
    if reordered:
        ok(f"{reordered}/{len(reranked)} results changed position")
    else:
        bad("reranking preserved fusion order exactly — it is doing nothing")
        failures += 1

    print(f"\n{DIM}--- scores are raw logits, not probabilities{RESET}")
    logits = [r.passage["rerank_score"] for r in reranked]
    if any(abs(x) > 1.0 for x in logits):
        ok(f"logit range {min(logits):.2f} to {max(logits):.2f} — unbounded, "
           f"so f3's gap survives")
    else:
        warn("all logits within [-1,1]; check nothing squashed them")

    print(f"\n{DIM}--- features f1-f3 (§5.2){RESET}")
    features = rerank_features(reranked)
    for name, value in features.items():
        shown = f"{value:.4f}" if value is not None else "None (insufficient results)"
        print(f"       {name:<18} {shown}")
    if features["f1_max_rerank"] is not None:
        ok("f1/f2 computed from logits")
    else:
        bad("features not computed")
        failures += 1

    print(f"\n{DIM}--- CPU latency{RESET}")
    timings = []
    for _ in range(3):
        t0 = time.time()
        reranker.rerank(QUERY, candidates, top_k=5)
        timings.append(time.time() - t0)
    median = statistics.median(timings)
    per_pair = median / len(candidates)
    print(f"{DIM}       {len(candidates)} pairs: median {median * 1000:.0f}ms "
          f"({per_pair * 1000:.0f}ms/pair){RESET}")
    projected = per_pair * cfg.get("retrieval.rerank.input_candidates")
    print(f"{DIM}       projected for {cfg.get('retrieval.rerank.input_candidates')} "
          f"candidates: {projected:.2f}s per retrieval round{RESET}")
    if projected < 15.0:
        ok(f"acceptable on CPU ({projected:.1f}s per round)")
    else:
        warn(f"{projected:.1f}s per round is slow; consider lowering "
             f"retrieval.rerank.input_candidates")

    print(f"\n{DIM}--- flat-evidence case (what f3 exists to detect){RESET}")
    vague = [r for r in make_results() if r.passage["source_id"] in
             ("D5", "D6", "D7", "D4", "D3")]
    flat = reranker.rerank("what is the capital of France", vague, top_k=5)
    flat_features = rerank_features(flat)
    print(f"{DIM}       off-topic query -> f1={flat_features['f1_max_rerank']:.3f}, "
          f"gap={flat_features['f3_score_gap']}{RESET}")
    if flat_features["f1_max_rerank"] < features["f1_max_rerank"]:
        ok("off-topic query scores lower than on-topic — f1 is discriminative, "
           "which is what the sufficiency model will rely on")
    else:
        warn("off-topic query scored as high as on-topic; f1 may be weak here")

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C11 acceptance: all checks passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())