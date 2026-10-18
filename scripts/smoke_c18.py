#!/usr/bin/env python3
"""C18 acceptance check — synthesizer over a real plan.

    python -m scripts.smoke_c18

Costs 2-3 LLM calls (planner + synthesizer, plus one for the partial path)
and retrieval against the C13 corpus.

The C18 criterion: "a full run produces a draft where 100% of citation IDs
exist."
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from src.budget import Budget
from src.config import get_config
from src.nodes.planner import Planner
from src.nodes.retriever import StaticRetriever
from src.nodes.synthesizer import Synthesizer
from src.state import new_state

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)

COLLECTION = "retrieval_eval"
QUESTION = ("How do FlashAttention-2 and PagedAttention differ in their "
            "memory access patterns?")


def ok(m): print(f"{GREEN}  PASS{RESET} {m}")
def warn(m): print(f"{YELLOW}  WARN{RESET} {m}")
def bad(m): print(f"{RED}  FAIL{RESET} {m}")


def main() -> int:
    print("\n=== C18 smoke test: synthesizer ===\n")
    cfg = get_config()
    budget = Budget.from_config(cfg)
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

    state = new_state(QUESTION)
    state["plan"] = Planner(cfg, budget=budget).plan(QUESTION)
    state = StaticRetriever(cfg, retriever=hybrid, reranker=Reranker(cfg),
                            budget=budget).retrieve_all(state)

    synthesizer = Synthesizer(cfg, budget=budget)
    draft = synthesizer.synthesize(state)

    if draft.error:
        bad(f"synthesis failed: {draft.error}")
        return 1

    print(f"{BOLD}Draft{RESET}\n")
    print(draft.text[:1800])
    print()

    print(f"{BOLD}Citation integrity{RESET}")
    cited = re.findall(r"\[(P[0-9a-f]{12})\]", draft.text)
    known = set(draft.label_map.values())
    unresolved = [c for c in cited if c not in known]

    if cited and not unresolved:
        ok(f"{len(cited)} citation markers, all resolve — the C18 criterion")
    elif not cited:
        bad("the draft contains no citations at all")
        failures += 1
    else:
        bad(f"{len(unresolved)} citations do not resolve: {unresolved[:4]}")
        failures += 1

    leftover = re.findall(r"\[P\d{2}\]", draft.text)
    if leftover:
        bad(f"short labels survived into the draft: {leftover[:4]} — the "
            f"critic cannot resolve these")
        failures += 1
    else:
        ok("no short labels left unmapped")

    if draft.invalid_labels:
        warn(f"the model invented {len(draft.invalid_labels)} label(s) "
             f"({draft.invalid_labels[:4]}) — stripped, so those claims are "
             f"now uncited and the critic will mark them UNSUPPORTED")
    else:
        ok("the model invented no labels")

    print(f"\n{BOLD}Coverage{RESET}")
    print(f"{DIM}   passages offered   {len(draft.label_map)}")
    print(f"   passages cited     {draft.citation_count}")
    print(f"   citation markers   {len(cited)}")
    print(f"   uncited sentences  {draft.uncited_sentences}")
    print(f"   draft length       {len(draft.text)} chars{RESET}")

    used = draft.citation_count / max(1, len(draft.label_map))
    if used < 0.3:
        warn(f"only {used:.0%} of offered passages were cited — the report may "
             f"be leaning on a few sources")
    else:
        ok(f"{used:.0%} of offered passages cited")

    if draft.uncited_sentences == 0:
        ok("every substantial sentence carries a citation")
    else:
        warn(f"{draft.uncited_sentences} substantial sentence(s) uncited — "
             f"the critic will mark these UNSUPPORTED, which is the intended "
             f"behaviour, not a bug")

    print(f"\n{BOLD}Partial path{RESET}")
    partial = synthesizer.synthesize(
        state, mode="partial",
        gaps=["no source gives the backward-pass figure for PagedAttention"],
    )
    if partial.error:
        bad(f"partial synthesis failed: {partial.error}")
        failures += 1
    else:
        print(f"{DIM}{partial.text[:600]}{RESET}\n")
        mentions_gap = any(
            w in partial.text.lower()
            for w in ("not establish", "no source", "does not", "unresolved",
                      "could not", "unclear", "absent")
        )
        if mentions_gap:
            ok("the partial draft states its gap rather than hedging around it")
        else:
            warn("the partial draft does not visibly acknowledge the gap — "
                 "check the wording above")

    print(f"\n{DIM}--- cost{RESET}")
    print(f"       {budget.llm_calls_used} LLM calls, "
          f"{budget.retrieval_calls_used} retrieval calls")

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C18 acceptance: all checks passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())