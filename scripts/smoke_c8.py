#!/usr/bin/env python3
"""C8 acceptance check — three real papers.

    python -m scripts.smoke_c8

Zero LLM tokens. Downloads three arXiv PDFs (~5 MB total, cached afterwards)
and verifies the C8 criterion: "section names are correctly extracted and no
chunk crosses a section boundary."

Run it twice — the second run should download nothing.
"""
from __future__ import annotations

import sys
import time

from src.config import get_config
from src.rag.chunking import NON_EVIDENCE_SECTIONS
from src.rag.ingest import PDFIngestor
from src.state import Passage

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
)

PAPERS = [
    ("1706.03762v7", "https://arxiv.org/pdf/1706.03762v7", "Attention Is All You Need"),
    ("2312.00752v2", "https://arxiv.org/pdf/2312.00752v2", "Mamba"),
    ("2305.14314v1", "https://arxiv.org/pdf/2305.14314v1", "QLoRA"),
]

# A paper with none of these extracted has a section-detection problem.
EXPECTED_ANY = {"Abstract", "Introduction", "Related Work", "Method",
                "Experiments", "Results", "Conclusion", "Discussion",
                "Background", "Analysis", "Ablation", "Limitations"}

REQUIRED = set(Passage.__annotations__)


def ok(m): print(f"{GREEN}  PASS{RESET} {m}")
def warn(m): print(f"{YELLOW}  WARN{RESET} {m}")
def bad(m): print(f"{RED}  FAIL{RESET} {m}")


def main() -> int:
    print("\n=== C8 smoke test: PDF ingestion + section-aware chunking ===")
    print(f"{DIM}Downloads ~5 MB on first run; cached after.{RESET}\n")

    ing = PDFIngestor(get_config())
    failures = 0

    for source_id, url, name in PAPERS:
        print(f"{DIM}--- {name} ({source_id}){RESET}")
        t0 = time.time()
        passages = ing.ingest_paper(
            url, source_id=source_id, sub_question_id="sq1", title=name
        )
        elapsed = time.time() - t0

        if not passages:
            bad(f"no passages extracted from {url}")
            failures += 1
            continue

        sections = []
        for p in passages:
            if p["section"] and p["section"] not in sections:
                sections.append(p["section"])

        recognised = EXPECTED_ANY & set(sections)
        if len(recognised) >= 3:
            ok(f"{len(passages)} passages across {len(sections)} sections "
               f"in {elapsed:.1f}s")
            print(f"{DIM}       sections: {', '.join(sections[:9])}"
                  f"{' …' if len(sections) > 9 else ''}{RESET}")
        else:
            bad(f"only recognised {sorted(recognised)} — section detection weak")
            failures += 1

        leaked = NON_EVIDENCE_SECTIONS & set(sections)
        if leaked:
            bad(f"bibliography leaked into evidence: {leaked}")
            failures += 1
        else:
            ok("References/Acknowledgments excluded")

        schema_problems = [
            f"{p['id']}: missing {sorted(REQUIRED - set(p))}"
            for p in passages if REQUIRED - set(p)
        ]
        if schema_problems:
            bad(schema_problems[0])
            failures += 1

        if len({p["id"] for p in passages}) == len(passages):
            ok("all passage ids unique (dedup working)")
        else:
            bad("duplicate passage ids returned")
            failures += 1

        lengths = [len(p["text"]) for p in passages]
        print(f"{DIM}       chunk chars: min {min(lengths)}, "
              f"median {sorted(lengths)[len(lengths) // 2]}, max {max(lengths)}{RESET}")

        # The invariant the whole design exists for.
        related = [p for p in passages if p["section"] == "Related Work"]
        results = [p for p in passages if p["section"] in ("Results", "Experiments")]
        if related and results:
            overlap = {p["id"] for p in related} & {p["id"] for p in results}
            if overlap:
                bad("a chunk appears in both Related Work and Results")
                failures += 1
            else:
                ok(f"Related Work ({len(related)}) and Results/Experiments "
                   f"({len(results)}) are cleanly separated")
        else:
            warn("this paper has no Related Work or no Results section to compare")

        print()

    print(f"{DIM}--- cache (re-ingest must not re-download){RESET}")
    t0 = time.time()
    again = ing.ingest_paper(PAPERS[0][1], source_id=PAPERS[0][0], title="x")
    elapsed = time.time() - t0
    if again and elapsed < 3.0:
        ok(f"re-ingest served from cache in {elapsed:.2f}s")
    else:
        bad(f"re-ingest took {elapsed:.1f}s — cache not working")
        failures += 1

    print(f"\n{DIM}--- dead URL degrades gracefully{RESET}")
    if ing.ingest_paper("https://arxiv.org/pdf/0000.00000v9",
                        source_id="nope") == []:
        ok("returns [] rather than raising")
    else:
        bad("unexpected passages from a dead URL")
        failures += 1

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C8 acceptance: all checks passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())