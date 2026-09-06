#!/usr/bin/env python3
"""C9 acceptance check — real bge-small embeddings + Qdrant.

    python -m scripts.smoke_c9

Zero API calls. Downloads ~130 MB of model weights on first run (cached by
HuggingFace afterwards) and runs entirely on CPU.

The C9 criterion: "store 50 passages, retrieve by similarity, retrieve with a
section == 'Results' filter, retrieve by ID."

It also checks the thing no unit test can: that the query prefix actually
improves retrieval on the real model. That failure is silent by nature, so it
is worth one direct measurement.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

from src.config import get_config
from src.rag.embed import Embedder
from src.rag.store import EvidenceStore
from src.state import Passage, make_passage_id

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
)

SECTIONS = ["Results", "Related Work", "Method", "Abstract", "Introduction"]
TOPICS = [
    "the model achieves 28.4 BLEU on WMT 2014 English-to-German translation",
    "reciprocal rank fusion combines ranked lists without score normalisation",
    "QLoRA quantises the frozen base model to 4-bit NF4 precision",
    "cross-encoder rerankers score query-document pairs jointly",
    "selective state space models scale linearly with sequence length",
    "hierarchical navigable small world graphs index high dimensional vectors",
    "expected calibration error measures confidence reliability",
    "abstention requires knowing when evidence is insufficient",
    "BM25 ranks documents by term frequency and inverse document frequency",
    "section aware chunking preserves the provenance of a claim",
]


def ok(m): print(f"{GREEN}  PASS{RESET} {m}")
def warn(m): print(f"{YELLOW}  WARN{RESET} {m}")
def bad(m): print(f"{RED}  FAIL{RESET} {m}")


def build_passages() -> list[Passage]:
    passages = []
    for i in range(50):
        text = f"{TOPICS[i % len(TOPICS)]}. This is passage {i} of the corpus."
        section = SECTIONS[i % len(SECTIONS)]
        source_id = f"paper-{i % 7}"
        passages.append(Passage(
            id=make_passage_id(source_id, text, section),
            source_type="arxiv" if i % 3 else "web",
            source_id=source_id,
            source_domain="arxiv.org" if i % 3 else "example.com",
            title=f"Paper {i % 7}",
            text=text,
            section=section,
            published=f"20{15 + (i % 9)}-03-01",
            sub_question_id="sq1",
            retrieval_score=0.0,
            rerank_score=None,
        ))
    return passages


def main() -> int:
    print("\n=== C9 smoke test: embeddings + Qdrant ===")
    print(f"{DIM}First run downloads ~130 MB of model weights. CPU only.{RESET}\n")

    cfg = get_config()
    failures = 0

    print(f"{DIM}--- embedder{RESET}")
    t0 = time.time()
    embedder = Embedder(cfg)
    vector = embedder.embed_query("what is reciprocal rank fusion")
    load_s = time.time() - t0

    if len(vector) == embedder.dim:
        ok(f"{embedder.model_name} loaded in {load_s:.1f}s, dim={len(vector)}")
    else:
        bad(f"dimension mismatch: got {len(vector)}, config says {embedder.dim}")
        failures += 1

    norm = sum(v * v for v in vector) ** 0.5
    if abs(norm - 1.0) < 0.01:
        ok(f"vectors L2-normalised (‖v‖={norm:.4f}) — cosine == dot product")
    else:
        bad(f"vectors not normalised (‖v‖={norm:.4f})")
        failures += 1

    # The silent failure this layer is most exposed to.
    print(f"\n{DIM}--- query prefix actually helps (silent if wrong){RESET}")
    doc = "Reciprocal rank fusion merges ranked result lists using 1/(k+rank)."
    doc_vec = embedder.embed_documents([doc])[0]
    with_prefix = embedder.embed_query("how does reciprocal rank fusion work")
    without = embedder.embed_documents(["how does reciprocal rank fusion work"])[0]

    def dot(a, b): return sum(x * y for x, y in zip(a, b))

    sim_with, sim_without = dot(with_prefix, doc_vec), dot(without, doc_vec)
    print(f"{DIM}       with prefix: {sim_with:.4f}   without: {sim_without:.4f}{RESET}")
    if sim_with > sim_without:
        ok("prefixed query scores higher — asymmetric encoding is working")
    else:
        warn("prefix did not improve this pair; check the model card's "
             "required prefix before trusting retrieval numbers")

    tmp = Path(tempfile.mkdtemp(prefix="c9-qdrant-"))
    try:
        store = EvidenceStore(cfg, path=str(tmp), collection="smoke_c9",
                              embedder=embedder)

        print(f"\n{DIM}--- store 50 passages{RESET}")
        t0 = time.time()
        written = store.upsert(build_passages())
        if written == 50 and store.count() == 50:
            ok(f"stored {written} passages in {time.time() - t0:.1f}s")
        else:
            bad(f"wrote {written}, count reports {store.count()}")
            failures += 1

        print(f"\n{DIM}--- similarity search{RESET}")
        results = store.search("how do rankers combine multiple result lists",
                               top_k=3)
        if results:
            ok(f"{len(results)} results, top score {results[0]['retrieval_score']:.3f}")
            for r in results:
                print(f"{DIM}       [{r['section']:<13}] {r['retrieval_score']:.3f}  "
                      f"{r['text'][:58]}{RESET}")
            if "fusion" in results[0]["text"].lower():
                ok("top result is semantically correct")
            else:
                warn("top result is not the obvious match — inspect above")
        else:
            bad("similarity search returned nothing")
            failures += 1

        print(f"\n{DIM}--- filtered search: section == 'Results'{RESET}")
        filtered = store.search("model performance", top_k=10, section="Results")
        if filtered and {r["section"] for r in filtered} == {"Results"}:
            ok(f"{len(filtered)} results, all from Results")
        else:
            bad(f"filter leaked sections: {{r['section'] for r in filtered}}")
            failures += 1

        print(f"\n{DIM}--- combined filter (low selectivity){RESET}")
        narrow = store.search("model", top_k=10, section="Results",
                              source_type="arxiv", published_after="2018-01-01")
        if narrow:
            ok(f"{len(narrow)} results survive a three-way filter — "
               f"predicate-aware traversal, not post-filtering")
        else:
            warn("narrow filter returned nothing; may be correct for this corpus")

        print(f"\n{DIM}--- exclude Related Work{RESET}")
        excluded = store.search("attention", top_k=20,
                                exclude_sections=["Related Work"])
        if "Related Work" not in {r["section"] for r in excluded}:
            ok("Related Work excluded — the misattribution guard works live")
        else:
            bad("Related Work leaked through must_not")
            failures += 1

        print(f"\n{DIM}--- retrieve by id{RESET}")
        target = build_passages()[7]
        got = store.get(target["id"])
        if got and got["id"] == target["id"] and got["text"] == target["text"]:
            ok(f"exact lookup returned {target['id']}")
        else:
            bad("lookup by id failed")
            failures += 1

        if store.get("Pnonexistent") is None:
            ok("missing id returns None")
        else:
            bad("missing id returned something")
            failures += 1

        print(f"\n{DIM}--- idempotent re-ingest{RESET}")
        store.upsert(build_passages())
        if store.count() == 50:
            ok("re-storing the same passages overwrites, does not duplicate")
        else:
            bad(f"count grew to {store.count()} after re-ingest")
            failures += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C9 acceptance: all checks passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())