#!/usr/bin/env python3
"""Snapshot the retrieval-evaluation corpus so the judgments stay usable.

    python -m scripts.corpus_snapshot export
    python -m scripts.corpus_snapshot restore

Why this exists
---------------
`benchmark/retrieval_labels.json` records ~1000 judgments against passage
ids. Those ids are content-addressed, so arXiv passages would regenerate
identically from the cached PDFs. **Web passages would not** — Tavily returns
different pages over time, and a re-run would produce different ids for the
same query.

So the label set is only as durable as `data/`, which is gitignored machine
state. Losing it orphans a share of an afternoon's labelling that cannot be
recreated. The snapshot is a few megabytes of JSON, commits cleanly, and
makes the R1-R4 result reproducible by anyone who clones the repo — which is
the difference between a number in a README and a result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.config import get_config

SNAPSHOT_PATH = Path("benchmark/retrieval_corpus.json")
COLLECTION = "retrieval_eval"

GREEN, RED, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
)


def _store(cfg):
    from qdrant_client import QdrantClient

    from src.rag.store import EvidenceStore

    path = Path(cfg.get("retrieval.store.path")) / COLLECTION
    return EvidenceStore(
        cfg, client=QdrantClient(path=str(path)), collection=COLLECTION
    )


def cmd_export(_args) -> int:
    cfg = get_config()
    store = _store(cfg)

    passages: list[dict] = []
    offset = None
    while True:
        records, offset = store.client.scroll(
            collection_name=COLLECTION, limit=1000,
            offset=offset, with_payload=True,
        )
        passages.extend(dict(r.payload) for r in records if r.payload)
        if offset is None:
            break

    if not passages:
        print(f"{RED}corpus is empty — nothing to snapshot{RESET}")
        return 1

    by_type: dict[str, int] = {}
    for p in passages:
        by_type[p.get("source_type", "?")] = by_type.get(p.get("source_type", "?"), 0) + 1

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(
        {"collection": COLLECTION, "count": len(passages), "passages": passages},
        indent=1,
    ))

    size_mb = SNAPSHOT_PATH.stat().st_size / 1_048_576
    print(f"\n{GREEN}exported {len(passages)} passages "
          f"({size_mb:.1f} MB){RESET}")
    for source_type, count in sorted(by_type.items()):
        print(f"{DIM}  {source_type:<8} {count}{RESET}")
    print(f"{DIM}  -> {SNAPSHOT_PATH}{RESET}")

    labels = Path("benchmark/retrieval_labels.json")
    if labels.is_file():
        judged = {
            pid
            for entry in json.loads(labels.read_text())["queries"]
            for pid in entry["judgments"]
        }
        have = {p["id"] for p in passages}
        missing = judged - have
        if missing:
            print(f"\n{RED}{len(missing)} judged passages are NOT in the "
                  f"corpus — those judgments cannot be scored.{RESET}")
            return 1
        print(f"{GREEN}all {len(judged)} judged passages are covered{RESET}")
    print()
    return 0


def cmd_restore(_args) -> int:
    cfg = get_config()
    if not SNAPSHOT_PATH.is_file():
        print(f"{RED}no snapshot at {SNAPSHOT_PATH}{RESET}")
        return 1

    data = json.loads(SNAPSHOT_PATH.read_text())
    passages = data["passages"]

    from src.rag.fusion import HybridRetriever
    from src.rag.sparse import BM25Index

    store = _store(cfg)
    retriever = HybridRetriever(cfg, store=store, sparse=BM25Index(cfg))

    print(f"\n{BOLD}Restoring {len(passages)} passages{RESET}")
    print(f"{DIM}re-embedding on CPU; this takes a few minutes{RESET}")
    written = retriever.index(passages)
    print(f"\n{GREEN}restored {written} passages into {COLLECTION}{RESET}")
    print(f"{DIM}run `python -m scripts.retrieval_labels evaluate` to "
          f"reproduce the table{RESET}\n")
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export", help="write the corpus snapshot").set_defaults(
        func=cmd_export
    )
    sub.add_parser("restore", help="rebuild the store from a snapshot").set_defaults(
        func=cmd_restore
    )
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())