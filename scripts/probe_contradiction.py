#!/usr/bin/env python3
"""Measure the real NLI contradiction scan on real passages — and choose f5's rule.

    python -m scripts.probe_contradiction
    python -m scripts.probe_contradiction "your query"

ZERO API calls: retrieval, rerank, embeddings and NLI all run locally.

History. v1 (C28.5) compared f5 on contested queries against a consensus
control and printed "OK": 0.533 against 0.000. Reading the pairs showed the
0.533 was false. One passage ("We varied the LoRA rank over three orders of
magnitude…") "contradicted" a methods paragraph about XLM-RoBERTa, a glossary
bullet list, an explainer on what rank is, and a rule of thumb for alpha —
all at 0.995–1.000. None disagrees with it; they are about different things.
A passage that contradicts everything is a HUB: NLI models trained on MNLI
have no "unrelated" class, and often call a change of subject a
contradiction. The rate comparison passed for the wrong reason.

v2 measures what a rule needs, per pair:
  ab, ba   P(contradiction) in each direction (the detector takes the max)
  sim      cosine similarity of the two windows the NLI model sees (bge)
and reports f5 under candidate rules, plus a hub check. Two passages can
only disagree about something they are both about — that is what `sim`
tests. A real disagreement should also hold in both directions — `min`.
"""
from __future__ import annotations

import itertools
import math
import sys
from collections import Counter
from pathlib import Path

from src.config import get_config

RED, GREEN, YELLOW, DIM, BOLD, RESET = (
    "\033[31m", "\033[32m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)
COLLECTION = "retrieval_eval"
THRESHOLD = 0.60

CASES = [
    ("What LoRA rank is needed for fine-tuning to match full fine-tuning?", "contested"),
    ("Does LoRA match the performance of full fine-tuning?", "contested"),
    ("What is the compute-optimal ratio of training tokens to model parameters?", "contested"),
    ("How does PagedAttention store the KV cache in memory?", "control"),
    ("What does FlashAttention-2 change compared to FlashAttention?", "control"),
]

# Candidate rules for "this pair contradicts". All use the 0.60 threshold.
RULES = {
    "R0 max (current)": lambda ab, ba, sim: max(ab, ba) >= THRESHOLD,
    "R1 min (both directions)": lambda ab, ba, sim: min(ab, ba) >= THRESHOLD,
    "R2 max & sim>=0.6": lambda ab, ba, sim: max(ab, ba) >= THRESHOLD and sim >= 0.6,
    "R3 min & sim>=0.5": lambda ab, ba, sim: min(ab, ba) >= THRESHOLD and sim >= 0.5,
}


def wire(cfg):
    from qdrant_client import QdrantClient

    from src.rag.embed import Embedder
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
    return hybrid, Reranker(cfg), Embedder(cfg)


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def short(text: str, n: int = 140) -> str:
    return " ".join((text or "").split())[:n]


def measure(passages, detector, embedder):
    """Every cross-source pair: (a, b, ab, ba, sim)."""
    from src.rag.contradiction import _contradiction_probability, _window

    pairs = [(a, b) for a, b in itertools.combinations(passages, 2)
             if a["source_id"] != b["source_id"]][: detector.max_pairs]
    if not pairs:
        return []
    windows = {p["id"]: _window(p["text"]) for p in passages}
    inputs = [(windows[a["id"]], windows[b["id"]]) for a, b in pairs]
    inputs += [(y, x) for x, y in inputs]
    raw = detector.model.predict(inputs)
    index = detector.contradiction_index()
    probs = [_contradiction_probability(r, index) for r in raw]
    half = len(pairs)
    vectors = dict(zip(windows, embedder.embed_documents(list(windows.values()))))
    return [(a, b, probs[i], probs[i + half], cosine(vectors[a["id"]], vectors[b["id"]]))
            for i, (a, b) in enumerate(pairs)]


def main() -> int:
    from src.rag.contradiction import ContradictionDetector

    cfg = get_config()
    top_k = int(cfg.get("retrieval.contradiction.scan_top_k", 8))
    hybrid, reranker, embedder = wire(cfg)
    detector = ContradictionDetector(cfg)
    cases = [(sys.argv[1], "query")] if len(sys.argv) > 1 else CASES
    table: dict[str, dict[str, list[float]]] = {r: {} for r in RULES}

    for query, kind in cases:
        print(f"\n{BOLD}{'=' * 78}\n[{kind}] {query}{RESET}")
        results = reranker.rerank(query, hybrid.retrieve(query, top_k=30), top_k=top_k)
        passages = [r.passage for r in results]
        label = {p["id"]: chr(ord("A") + i) for i, p in enumerate(passages)}
        for p in passages:
            print(f"  {label[p['id']]} {DIM}{p.get('source_id', '')[:38]:<38} "
                  f"{short(p['text'], 95)}{RESET}")

        rows = measure(passages, detector, embedder)
        if not rows:
            print("  no cross-source pairs")
            continue

        for rule, test in RULES.items():
            hits = sum(1 for _, _, ab, ba, sim in rows if test(ab, ba, sim))
            table[rule].setdefault(kind, []).append(hits / len(rows))

        hub = Counter()
        for a, b, ab, ba, sim in rows:
            if max(ab, ba) >= THRESHOLD:
                hub[a["id"]] += 1
                hub[b["id"]] += 1
        r0_hits = sum(1 for _, _, ab, ba, _ in rows if max(ab, ba) >= THRESHOLD)
        if r0_hits >= 3:
            top_id, top_n = hub.most_common(1)[0]
            if top_n >= 0.6 * r0_hits:
                print(f"\n  {YELLOW}HUB{RESET} passage {label[top_id]} is in "
                      f"{top_n} of {r0_hits} R0 hits — likely topic mismatch, "
                      f"not disagreement")

        print(f"\n  {len(rows)} pairs.  {'pair':<6}{'ab':>6}{'ba':>6}{'sim':>6}   "
              f"R0 R1 R2 R3")
        rows.sort(key=lambda r: max(r[2], r[3]), reverse=True)
        for a, b, ab, ba, sim in rows[:8]:
            marks = " ".join("x " if test(ab, ba, sim) else ". "
                             for test in RULES.values())
            print(f"  {label[a['id']]}-{label[b['id']]:<4}{ab:>6.2f}{ba:>6.2f}{sim:>6.2f}   {marks}")
        production = detector.scan(passages, sub_question_id="probe")
        print(f"\n  production rule (C28.7): {production.pairs} pairs after hub "
              f"exclusion, {len(production.records)} contradictions, "
              f"f5 = {production.rate:.3f}, hubs = "
              f"{[label[h] for h in production.hubs] or 'none'}")
        for a, b, ab, ba, sim in rows[:3]:
            if max(ab, ba) >= THRESHOLD:
                print(f"\n  {label[a['id']]}-{label[b['id']]}  {DIM}{short(a['text'])}{RESET}")
                print(f"  {' ' * 4}  {DIM}{short(b['text'])}{RESET}")

    print(f"\n{BOLD}f5 by rule (mean over queries){RESET}")
    print(f"  {'rule':<26}{'contested':>11}{'control':>9}")
    for rule in RULES:
        cont = table[rule].get("contested") or [0.0]
        ctrl = table[rule].get("control") or [0.0]
        print(f"  {rule:<26}{sum(cont)/len(cont):>11.3f}{sum(ctrl)/len(ctrl):>9.3f}")
    print("\nA rule earns its place only if the pairs it keeps are ones a reader "
          "would call a disagreement. Read the printed pairs before the table.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
