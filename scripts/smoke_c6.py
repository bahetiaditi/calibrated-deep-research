#!/usr/bin/env python3
"""C6 acceptance check — live arXiv API.

    python -m scripts.smoke_c6

Costs zero LLM tokens; makes ~5 arXiv requests. arXiv enforces a 3-second
inter-request delay, so expect this to take ~20 seconds. That delay is
deliberate and must not be lowered.

The C6 check is "returns valid passages for three sample queries; handles a
deliberately malformed query".
"""
from __future__ import annotations

import sys
import time

from src.config import get_config
from src.state import Passage
from src.tools.arxiv_tool import ArxivTool

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"

QUERIES = [
    "Mamba selective state space models",
    "FlashAttention memory efficient exact attention",
    "QLoRA efficient finetuning quantized LLMs",
]

REQUIRED = set(Passage.__annotations__)


def ok(m: str) -> None:
    print(f"{GREEN}  PASS{RESET} {m}")


def bad(m: str) -> None:
    print(f"{RED}  FAIL{RESET} {m}")


def check_schema(passages: list[dict]) -> list[str]:
    problems = []
    for p in passages:
        missing = REQUIRED - set(p)
        if missing:
            problems.append(f"missing fields {sorted(missing)}")
        if not p.get("text"):
            problems.append("empty text")
        if not p.get("source_id"):
            problems.append("no source_id")
        if p.get("source_type") != "arxiv":
            problems.append(f"wrong source_type {p.get('source_type')!r}")
    return problems


def main() -> int:
    print("\n=== C6 smoke test: arXiv tool ===")
    print(f"{DIM}arXiv enforces a 3s inter-request delay; this takes ~20s.{RESET}\n")

    tool = ArxivTool(get_config())
    failures = 0
    all_ids: set[str] = set()

    for query in QUERIES:
        t0 = time.time()
        passages = tool.search(query, max_results=3, sub_question_id="sq1")
        elapsed = time.time() - t0

        if not passages:
            bad(f"{query[:44]!r} returned nothing")
            failures += 1
            continue

        problems = check_schema(passages)
        if problems:
            bad(f"{query[:44]!r}: {problems[:3]}")
            failures += 1
            continue

        ok(f"{len(passages)} passages in {elapsed:.1f}s — {query[:44]!r}")
        top = passages[0]
        print(f"{DIM}       {top['source_id']}  {top['published']}  "
              f"{top['title'][:56]}{RESET}")
        print(f"{DIM}       {top['id']}  section={top['section']}  "
              f"{len(top['text'])} chars{RESET}")
        all_ids.update(p["id"] for p in passages)

    print(f"\n{DIM}--- malformed query{RESET}")
    weird = tool.search('"""((( unbalanced AND AND', max_results=2)
    ok(f"malformed query degraded gracefully ({len(weird)} passages, no exception)")

    print(f"\n{DIM}--- empty query{RESET}")
    if tool.search("") == []:
        ok("empty query short-circuits without hitting the API")
    else:
        bad("empty query was sent to the API")
        failures += 1

    print(f"\n{DIM}--- id stability (re-ingest must not duplicate){RESET}")
    again = tool.search(QUERIES[0], max_results=3, sub_question_id="sq2")
    repeat_ids = {p["id"] for p in again}
    if repeat_ids and repeat_ids <= all_ids:
        ok("repeat search yields identical passage ids — ingestion is idempotent")
    elif not repeat_ids:
        bad("repeat search returned nothing")
        failures += 1
    else:
        # arXiv relevance ranking can shift between calls; only a total
        # mismatch indicates the id scheme is unstable.
        overlap = len(repeat_ids & all_ids)
        if overlap:
            ok(f"{overlap}/{len(repeat_ids)} ids matched (arXiv ranking drifts)")
        else:
            bad("no id overlap — passage ids are not content-addressed")
            failures += 1

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C6 acceptance: all checks passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())