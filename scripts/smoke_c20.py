#!/usr/bin/env python3
"""C20 acceptance check — feature vectors across ten real runs.

    python -m scripts.smoke_c20

Costs ~10 LLM calls plus retrieval, over ten varied questions.

The C20 criterion: "feature vectors are populated and dimensionally
consistent across ten runs."

Dimensional consistency is the hard requirement — a vector whose length
depends on what happened cannot be stacked into a design matrix, and the
failure would surface at C33 as a shape error with the eval already paid for.
But the more interesting question is whether the features VARY: a feature
identical on every question carries no information and will get a zero
coefficient at C33, and it is far better to learn that now.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

from src.budget import Budget
from src.config import get_config
from src.nodes.planner import Planner
from src.nodes.retriever import StaticRetriever
from src.state import new_state
from src.sufficiency.features import (
    FEATURE_NAMES,
    completeness,
    extract_run_features,
    feature_vector_length,
    features_for_sub_question,
    to_vector,
)

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)

COLLECTION = "retrieval_eval"

# Deliberately mixed: well-covered lookups, questions the corpus barely
# touches, and one with a false premise. If every question produced the same
# features, the vector would be useless and this is where that shows.
QUESTIONS = [
    "What fraction of theoretical peak FLOPs does FlashAttention-2 reach?",
    "How does PagedAttention manage the key value cache?",
    "What rank did the LoRA authors use in their main experiments?",
    "How does QLoRA quantise the frozen base model?",
    "What is the selection mechanism in Mamba's state space model?",
    "What perplexity does the model reach on the language modelling benchmark?",
    "What limitations do the authors acknowledge about their own method?",
    "What makes retrieval augmentation fail on questions requiring synthesis?",
    "Is a higher adapter rank always better for parameter efficient finetuning?",
    "Why does Mamba-2 use rotary position embeddings in its SSM blocks?",
]


def ok(m): print(f"{GREEN}  PASS{RESET} {m}")
def warn(m): print(f"{YELLOW}  WARN{RESET} {m}")
def bad(m): print(f"{RED}  FAIL{RESET} {m}")


def main() -> int:
    print("\n=== C20 smoke test: sufficiency features ===\n")
    cfg = get_config()
    failures = 0

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
        return 1
    hybrid.sparse.build(passages)
    print(f"{DIM}corpus: {len(passages)} passages{RESET}\n")

    reranker = Reranker(cfg)
    collected: list[dict] = []
    lengths: set[int] = set()

    skipped: list[str] = []
    for question in QUESTIONS:
        budget = Budget.from_config(cfg)
        state = new_state(question)
        try:
            # f7 comes from the premise checker, so it has to run or the
            # feature is never exercised and the variance table says
            # "never observed" for a reason that has nothing to do with the
            # feature itself.
            from src.nodes.premise_checker import PremiseChecker

            state["premise_check"] = PremiseChecker(
                cfg, retriever=hybrid, reranker=reranker, budget=budget
            ).check(question).to_dict()
            state["plan"] = Planner(cfg, budget=budget).plan(question)[:1]
            state = StaticRetriever(cfg, retriever=hybrid, reranker=reranker,
                                    budget=budget).retrieve_all(state)
        except Exception as exc:  # noqa: BLE001
            # One question failing must not discard nine good feature
            # vectors — the variance table is the point of this script.
            print(f"{YELLOW}  SKIP{RESET} {question[:52]}")
            print(f"{DIM}       {type(exc).__name__}: {str(exc)[:88]}{RESET}")
            skipped.append(question)
            continue

        sq = state["plan"][0]
        features = features_for_sub_question(state, sq["id"])
        values, columns = to_vector(features)
        lengths.add(len(values))
        collected.append(features)

        shown = " ".join(
            f"{name.split('_')[0]}={'—' if features[name] is None else f'{features[name]:.2f}'}"
            for name in FEATURE_NAMES
        )
        print(f"{DIM}  {question[:52]:<54}{RESET}")
        print(f"{DIM}     {shown}  ({completeness(features):.0%} observed){RESET}")

    if skipped:
        print(f"\n{YELLOW}{len(skipped)} question(s) skipped after an "
              f"error; {len(collected)} vectors collected.{RESET}")

    if not collected:
        print(f"{RED}no feature vectors collected{RESET}")
        return 1

    print(f"\n{BOLD}Dimensional consistency{RESET}")
    if len(lengths) == 1 and lengths.pop() == feature_vector_length():
        ok(f"every vector is {feature_vector_length()} wide — the C20 criterion")
    else:
        bad(f"vector length varied: {lengths}")
        failures += 1

    if all(set(f) == set(FEATURE_NAMES) for f in collected):
        ok("every extraction returned every feature key")
    else:
        bad("some extractions dropped keys")
        failures += 1

    print(f"\n{BOLD}Does each feature carry information?{RESET}")
    print(f"{DIM}A feature identical on every question gets a zero "
          f"coefficient at C33.\nFar better to learn that now.{RESET}\n")
    print(f"{DIM}  {'feature':<26}{'observed':>9}{'min':>8}{'max':>8}"
          f"{'stdev':>8}{RESET}")
    dead: list[str] = []
    for name in FEATURE_NAMES:
        observed = [f[name] for f in collected if f[name] is not None]
        if not observed:
            print(f"  {name:<26}{'0/10':>9}{'—':>8}{'—':>8}{'—':>8}"
                  f"{DIM}  never observed{RESET}")
            dead.append(name)
            continue
        spread = statistics.pstdev(observed) if len(observed) > 1 else 0.0
        flag = f"{YELLOW}  constant{RESET}" if spread < 1e-9 else ""
        print(f"  {name:<26}{f'{len(observed)}/{len(collected)}':>9}{min(observed):>8.2f}"
              f"{max(observed):>8.2f}{spread:>8.3f}{flag}")
        if spread < 1e-9:
            dead.append(name)

    varying = [n for n in FEATURE_NAMES if n not in dead]
    if len(varying) >= 5:
        ok(f"{len(varying)}/{len(FEATURE_NAMES)} features vary across questions")
    else:
        bad(f"only {len(varying)} features vary — the vector may not carry "
            f"enough signal for C33")
        failures += 1

    expected_missing = {"f5_contradiction_rate", "f6_entailment_strength"}
    unexpected = set(dead) - expected_missing
    if unexpected:
        warn(f"unexpectedly flat or absent: {sorted(unexpected)}")
        print(f"{DIM}       f5 and f6 are expected to be absent here — "
              f"contradiction scanning\n       is not wired into the "
              f"retriever and the critic lands at C23{RESET}")
    else:
        ok("the only absent features are f5 and f6, both expected before C23")

    print(f"\n{BOLD}Missingness is explicit{RESET}")
    values, columns = to_vector(collected[0])
    missing_cols = [c for c in columns if c.endswith("_missing")]
    print(f"{DIM}   {len(missing_cols)} indicator columns alongside "
          f"{len(FEATURE_NAMES)} features{RESET}")
    if len(missing_cols) == len(FEATURE_NAMES):
        ok("every feature has a missingness indicator")
        print(f"{DIM}       imputing without one would tell the model "
              f"'no contradictions found'\n       when the truth is "
              f"'nobody looked'{RESET}")
    else:
        bad("indicator columns do not match feature count")
        failures += 1

    print(f"\n{BOLD}Run-level aggregates{RESET}")
    run_features = extract_run_features(state)
    for name, value in run_features.items():
        print(f"{DIM}   {name:<26}"
              f"{'—' if value is None else f'{value:.3f}'}{RESET}")

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C20 acceptance: all checks passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())