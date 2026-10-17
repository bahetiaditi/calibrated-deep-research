#!/usr/bin/env python3
"""Retrieval labelling and evaluation (C13).

    python -m scripts.retrieval_labels build     # ingest corpus + pool candidates
    python -m scripts.retrieval_labels label     # blind labelling session
    python -m scripts.retrieval_labels stats     # progress
    python -m scripts.retrieval_labels evaluate  # the R1-R4 table

Three properties of the labelling protocol, each of which affects whether the
resulting numbers mean anything:

**Pooling.** Candidates are the union of the top-N from all four
configurations. No system gets its results labelled more thoroughly than
another, so "unjudged = irrelevant" is a fair assumption rather than a bias
toward whichever system was labelled first.

**Blind.** The labelling view shows the passage and nothing else — not which
system retrieved it, not its rank, not its score. Ranks are the strongest
anchor there is: shown "rank 1 of the reranker", a labeller agrees.

**Shuffled.** Pool order within a query is randomised with a fixed seed, so
fatigue late in a session does not systematically penalise one system.

Labelling ~600 judgments takes an afternoon. `label` saves after every
judgment and resumes where it stopped, so it can be done in pieces.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

from src.config import get_config

GREEN, RED, YELLOW, CYAN, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[2m", "\033[1m", "\033[0m"
)

QUESTIONS_PATH = Path("benchmark/retrieval_questions.json")
LABELS_PATH = Path("benchmark/retrieval_labels.json")
COLLECTION = "retrieval_eval"
POOL_DEPTH = 20
SEED = 20260805

GRADES = {
    "2": "relevant — directly answers the sub-question",
    "1": "partially relevant — related and useful, does not answer it",
    "0": "irrelevant",
}


# ---------------------------------------------------------------------------
# build: ingest a corpus and pool candidates
# ---------------------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> int:
    from qdrant_client import QdrantClient

    from src.rag.fusion import HybridRetriever
    from src.rag.rerank import Reranker
    from src.rag.sparse import BM25Index
    from src.rag.store import EvidenceStore
    from src.tools.arxiv_tool import ArxivTool, pdf_url_of

    cfg = get_config()
    if not QUESTIONS_PATH.is_file():
        print(f"{RED}missing {QUESTIONS_PATH}{RESET}")
        print("Create it first — see the template printed by `stats`.")
        return 1

    spec = json.loads(QUESTIONS_PATH.read_text())
    questions = spec["questions"]
    seed_ids = spec.get("seed_arxiv_ids", [])

    store_path = Path(cfg.get("retrieval.store.path")) / COLLECTION
    if args.rebuild and store_path.exists():
        shutil.rmtree(store_path, ignore_errors=True)
    store_path.mkdir(parents=True, exist_ok=True)

    store = EvidenceStore(cfg, client=QdrantClient(path=str(store_path)),
                          collection=COLLECTION)
    retriever = HybridRetriever(cfg, store=store, sparse=BM25Index(cfg))

    print(f"\n{BOLD}Building the evaluation corpus{RESET}")
    print(f"{DIM}{len(questions)} sub-questions, {len(seed_ids)} seed papers{RESET}\n")

    if store.count() and not args.rebuild:
        print(f"{DIM}corpus already has {store.count()} passages; "
              f"pass --rebuild to start over{RESET}")
        all_passages = []
    else:
        all_passages = _ingest_corpus(cfg, seed_ids, questions)
        if all_passages:
            retriever.index(all_passages)
        print(f"\n{GREEN}indexed {store.count()} passages{RESET}")

    # bm25 lives in memory, so rebuild it from the store on a resumed run
    if not all_passages:
        _rehydrate_sparse(store, retriever)

    print(f"\n{BOLD}Pooling candidates{RESET}")
    reranker = Reranker(cfg)
    rng = random.Random(SEED)
    pool_entries = []

    for question in questions:
        qid, text = question["id"], question["text"]
        pooled: dict[str, dict] = {}

        for config, kwargs in (
            ("R1", {"dense_only": True}),
            ("R2", {"sparse_only": True}),
            ("R3", {}),
        ):
            for result in retriever.retrieve(text, top_k=POOL_DEPTH, **kwargs):
                pooled.setdefault(result.passage["id"], result.passage)

        fused = retriever.retrieve(text, top_k=POOL_DEPTH * 2)
        for result in reranker.rerank(text, fused, top_k=POOL_DEPTH):
            pooled.setdefault(result.passage["id"], result.passage)

        candidates = list(pooled.values())
        rng.shuffle(candidates)   # fatigue must not penalise one system
        pool_entries.append({
            "id": qid,
            "query": text,
            "candidates": [
                {
                    "passage_id": p["id"],
                    "source_id": p["source_id"],
                    "section": p.get("section"),
                    "title": p.get("title", ""),
                    "text": p["text"],
                }
                for p in candidates
            ],
            "judgments": {},
        })
        print(f"  {qid:<6} {len(candidates):>3} candidates  {text[:56]}")

    _merge_and_save(pool_entries)
    total = sum(len(e["candidates"]) for e in pool_entries)
    print(f"\n{GREEN}pooled {total} candidates across {len(pool_entries)} "
          f"queries{RESET}")
    print(f"{DIM}Next: python -m scripts.retrieval_labels label{RESET}\n")
    return 0


def _ingest_corpus(cfg, seed_ids, questions) -> list:
    from src.rag.ingest import PDFIngestor
    from src.tools.arxiv_tool import ArxivTool, pdf_url_of
    from src.tools.web_search import get_search_provider

    arxiv = ArxivTool(cfg)
    ingestor = PDFIngestor(cfg)
    passages: list = []

    if seed_ids:
        print(f"{DIM}fetching {len(seed_ids)} seed papers "
              f"(3s arXiv delay each){RESET}")
        import arxiv as arxiv_lib

        results = list(arxiv.client.results(arxiv_lib.Search(id_list=seed_ids)))
        for result in results:
            short = result.get_short_id()
            url = pdf_url_of(result)
            if not url:
                continue
            full = ingestor.ingest_paper(
                url, source_id=short, sub_question_id="",
                published=result.published.date().isoformat() if result.published else None,
                title=result.title,
            )
            passages.extend(full)
            print(f"  {short:<16} {len(full):>4} passages  {result.title[:48]}")

    print(f"\n{DIM}retrieving per-question context{RESET}")
    web = get_search_provider(cfg)
    for question in questions:
        found = arxiv.search(question["text"], max_results=4,
                             sub_question_id=question["id"])
        hits = web.search(question["text"], max_results=4,
                          sub_question_id=question["id"])
        for passage in hits:
            passages.extend(ingestor.chunk_web_passage(passage))
        passages.extend(found)
        print(f"  {question['id']:<6} +{len(found)} arxiv  +{len(hits)} web")
    return passages


def _rehydrate_sparse(store, retriever) -> None:
    """BM25 is in-memory; rebuild it from the persisted store."""
    records, _ = store.client.scroll(
        collection_name=store.collection, limit=10_000, with_payload=True
    )
    passages = [dict(r.payload) for r in records if r.payload]
    if passages:
        retriever.sparse.build(passages)
        print(f"{DIM}rebuilt sparse index over {len(passages)} passages{RESET}")


def _merge_and_save(entries: list[dict]) -> None:
    """Preserve any judgments already made when re-pooling."""
    existing: dict[str, dict] = {}
    if LABELS_PATH.is_file():
        for entry in json.loads(LABELS_PATH.read_text()).get("queries", []):
            existing[entry["id"]] = entry.get("judgments", {})

    for entry in entries:
        entry["judgments"] = {
            **{c["passage_id"]: None for c in entry["candidates"]},
            **{k: v for k, v in existing.get(entry["id"], {}).items()},
        }

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(json.dumps(
        {"pool_depth": POOL_DEPTH, "seed": SEED, "queries": entries}, indent=2
    ))


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------


def cmd_label(args: argparse.Namespace) -> int:
    if not LABELS_PATH.is_file():
        print(f"{RED}no pool yet — run `build` first{RESET}")
        return 1

    data = json.loads(LABELS_PATH.read_text())
    entries = data["queries"]
    if args.query:
        entries = [e for e in entries if e["id"] == args.query]
        if not entries:
            print(f"{RED}no query with id {args.query!r}{RESET}")
            return 1

    # Query-at-a-time, deliberately. A PARTIALLY labelled query is worse than
    # an unlabelled one: unjudged counts as irrelevant, so a relevant passage
    # you never reached is scored as a miss, penalising whichever system
    # ranked it highly. Fifteen complete queries beat thirty half-done ones.
    pending = [
        (entry, candidate)
        for entry in entries
        for candidate in entry["candidates"]
        if entry["judgments"].get(candidate["passage_id"]) is None
    ]

    if not pending:
        print(f"{GREEN}every candidate is labelled.{RESET}")
        return 0

    print(f"\n{BOLD}Blind labelling{RESET}  {len(pending)} remaining")
    print(f"{DIM}You are shown the passage only — no rank, no system, no score.")
    print(f"Ranks are the strongest anchor there is; seeing one makes you agree "
          f"with it.{RESET}\n")
    for key, description in GRADES.items():
        print(f"  {BOLD}{key}{RESET}  {description}")
    print(f"  {BOLD}s{RESET}  skip   {BOLD}q{RESET}  save and quit")
    print(f"\n{DIM}Two rules that keep judgments consistent across a long "
          f"session:\n"
          f"  · Judge THIS passage alone — not what the agent could infer "
          f"from the set.\n"
          f"  · Vocabulary overlap is not relevance. A passage stuffed with "
          f"the query's\n    words that never answers it is 0; grading it 1 "
          f"hides the exact failure\n    the reranker is supposed to fix."
          f"{RESET}\n")

    done = 0
    for entry, candidate in pending[: args.limit or len(pending)]:
        in_query = sum(
            1 for c in entry["candidates"]
            if entry["judgments"].get(c["passage_id"]) is None
        )
        print("=" * 78)
        print(f"{CYAN}QUERY{RESET}  {entry['query']}")
        print(f"{DIM}       {entry['id']} · {in_query} left in this query · "
              f"{len(pending) - done} total{RESET}")
        print("-" * 78)
        section = candidate.get("section") or "—"
        print(f"{DIM}[{section}] {candidate.get('source_id', '')} · "
              f"{candidate.get('title', '')[:52]}{RESET}")
        print(_wrap(candidate["text"], 76))
        print(f"{DIM}({len(candidate['text'])} chars — full passage shown){RESET}")
        print("-" * 78)

        while True:
            answer = input("grade [2/1/0/s/q] > ").strip().lower()
            if answer in GRADES:
                entry["judgments"][candidate["passage_id"]] = int(answer)
                break
            if answer == "s":
                break
            if answer == "q":
                LABELS_PATH.write_text(json.dumps(data, indent=2))
                print(f"\n{GREEN}saved. {done} labelled this session.{RESET}\n")
                return 0
            print(f"{YELLOW}enter 2, 1, 0, s or q{RESET}")

        done += 1
        # Save after every judgment: an afternoon's work must not depend on
        # exiting cleanly.
        LABELS_PATH.write_text(json.dumps(data, indent=2))

    print(f"\n{GREEN}{done} labelled this session.{RESET}")
    return cmd_stats(args)


def _wrap(text: str, width: int) -> str:
    """Show the WHOLE passage.

    An earlier version capped this at 14 lines, which was a real bug: paper
    chunks run to 1000 characters and the answer often sits in the last
    sentence. Judging a truncated view means labelling something different
    from what the retriever actually returns, and every metric computed from
    those labels would be quietly wrong.
    """
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width))


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def cmd_stats(args: argparse.Namespace) -> int:
    if not LABELS_PATH.is_file():
        print(f"{YELLOW}no label file yet.{RESET}")
        print(f"\nCreate {QUESTIONS_PATH} shaped like:\n")
        print(json.dumps({
            "seed_arxiv_ids": ["1706.03762v7", "2312.00752v2"],
            "questions": [
                {"id": "rq01", "text": "how does FlashAttention-2 partition work"},
            ],
        }, indent=2))
        return 1

    data = json.loads(LABELS_PATH.read_text())
    total = labelled = relevant = 0
    incomplete = []

    for entry in data["queries"]:
        grades = entry["judgments"]
        total += len(grades)
        done = [g for g in grades.values() if g is not None]
        labelled += len(done)
        relevant += sum(1 for g in done if g and g >= 1)
        if len(done) < len(grades):
            incomplete.append((entry["id"], len(done), len(grades)))

    print(f"\n{BOLD}Labelling progress{RESET}")
    print(f"  queries          {len(data['queries'])}")
    print(f"  judgments        {labelled}/{total} "
          f"({labelled / total:.0%})" if total else "  judgments  0")
    print(f"  relevant so far  {relevant}")
    if incomplete:
        print(f"\n{DIM}incomplete queries:{RESET}")
        for qid, done, all_ in incomplete[:12]:
            print(f"    {qid:<6} {done}/{all_}")
    else:
        print(f"\n{GREEN}complete — run `evaluate`{RESET}")
    print()
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def cmd_evaluate(args: argparse.Namespace) -> int:
    from qdrant_client import QdrantClient

    from analysis.retrieval_eval import (
        Judgments,
        RetrievalEvaluator,
        check_expected_ordering,
        render_table,
    )
    from src.rag.fusion import HybridRetriever
    from src.rag.rerank import Reranker
    from src.rag.sparse import BM25Index
    from src.rag.store import EvidenceStore

    cfg = get_config()
    judgments = Judgments.load(LABELS_PATH)
    coverage = judgments.coverage()

    print(f"\n{BOLD}Retrieval evaluation — R1-R4{RESET}\n")
    print(f"{DIM}  {coverage['judgments']}/{coverage['pooled']} judgments "
          f"over {coverage['queries']} queries, {coverage['relevant']} relevant")
    print(f"  {coverage['queries_complete']} fully labelled, "
          f"{coverage['queries_with_relevant']} with a relevant passage")
    print(f"  {coverage['queries_usable']} USABLE (both){RESET}\n")

    if coverage["queries_complete"] < coverage["queries"]:
        skipped = coverage["queries"] - coverage["queries_complete"]
        print(f"{DIM}  {skipped} partially-labelled queries are excluded. "
              f"Unjudged counts as\n  irrelevant, so a half-labelled query "
              f"scores a passage you never reached\n  as a miss — biased "
              f"against whichever system ranked it highly.{RESET}\n")

    if coverage["queries_usable"] < 5:
        print(f"{YELLOW}fewer than 5 usable queries — numbers will be noise. "
              f"Finish a few queries completely first.{RESET}\n")
        return 1

    store_path = Path(cfg.get("retrieval.store.path")) / COLLECTION
    store = EvidenceStore(cfg, client=QdrantClient(path=str(store_path)),
                          collection=COLLECTION)
    retriever = HybridRetriever(cfg, store=store, sparse=BM25Index(cfg))
    _rehydrate_sparse(store, retriever)

    evaluator = RetrievalEvaluator(retriever, judgments, reranker=Reranker(cfg))
    started = time.time()
    scores = evaluator.evaluate()
    print(render_table(scores))
    print(f"\n{DIM}evaluated in {time.time() - started:.1f}s{RESET}")

    problems = check_expected_ordering(scores)
    print()
    if not problems:
        print(f"{GREEN}Expected ordering holds: R4 >= R3 >= max(R1, R2).{RESET}")
        print(f"{DIM}The C10 and C11 design bets are supported on this "
              f"corpus.{RESET}")
    else:
        print(f"{YELLOW}Expected ordering does NOT hold:{RESET}")
        for problem in problems:
            print(f"  · {problem}")
        print(f"\n{DIM}§8 permits this outcome — it requires an explanation, "
              f"not a fix.\nLikely causes worth checking before concluding:")
        print(f"  · too few labelled queries for the difference to show")
        print(f"  · the bge query prefix (a silent recall loss; see C9)")
        print(f"  · a corpus so small that every system finds everything{RESET}")

    output = Path(cfg.get("run.results_dir")) / "retrieval_eval.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Retrieval evaluation (R1-R4)\n\n"
        f"{coverage['queries_with_relevant']} queries with relevant judgments, "
        f"{coverage['judgments']} total judgments.\n\n"
        "```\n" + render_table(scores) + "\n```\n\n"
        + ("All expected orderings hold.\n" if not problems
           else "Deviations:\n\n" + "\n".join(f"- {p}" for p in problems) + "\n")
    )
    print(f"\n{DIM}written to {output}{RESET}\n")
    return 0


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="ingest corpus and pool candidates")
    build.add_argument("--rebuild", action="store_true",
                       help="discard the existing corpus first")
    build.set_defaults(func=cmd_build)

    label = sub.add_parser("label", help="blind labelling session")
    label.add_argument("--limit", type=int, default=0,
                       help="stop after N judgments")
    label.add_argument("--query", type=str, default="",
                       help="label only this query id (e.g. rq01)")
    label.set_defaults(func=cmd_label)

    sub.add_parser("stats", help="labelling progress").set_defaults(func=cmd_stats)
    sub.add_parser("evaluate", help="run the R1-R4 table").set_defaults(
        func=cmd_evaluate
    )

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())