#!/usr/bin/env python3
"""Sanity-check the relevance judgments before trusting the R1-R4 table.

    python -m scripts.check_labels

Costs nothing and takes a second. Worth running because a label set can be
100% complete and still produce a meaningless table — and the table will look
perfectly reasonable either way, which is the dangerous part.

Three things it looks for:

**Relevance rate.** Pooled sets usually land at 10-30% relevant. Much higher
means either the corpus is so narrow that everything retrieved is on-topic,
or grading drifted toward generosity. Either way the configs cannot separate:
if every system finds relevant material easily, nDCG saturates and the
comparison measures nothing.

**Per-query spread.** Queries should differ. A lookup question with a stated
answer should have a few 2s; a conceptual question should have more 1s and
0s. If every query has an identical distribution, the queries are not
probing different retrieval behaviour.

**Degenerate queries.** All-relevant or all-irrelevant queries cannot
discriminate between systems and are excluded from scoring — but it is worth
knowing how many there are, because they represent labelling effort that
bought nothing.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

LABELS_PATH = Path("benchmark/retrieval_labels.json")

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


def main() -> int:
    if not LABELS_PATH.is_file():
        print(f"{RED}no label file at {LABELS_PATH}{RESET}")
        return 1

    data = json.loads(LABELS_PATH.read_text())
    queries = data["queries"]

    overall = Counter()
    rows = []
    degenerate_all_relevant = []
    degenerate_none_relevant = []
    unlabelled = []

    for entry in queries:
        grades = [g for g in entry["judgments"].values() if g is not None]
        missing = len(entry["judgments"]) - len(grades)
        counts = Counter(grades)
        overall.update(counts)

        if missing:
            unlabelled.append((entry["id"], missing))

        n = len(grades)
        relevant = counts[2] + counts[1]
        if n and relevant == n:
            degenerate_all_relevant.append(entry["id"])
        if n and relevant == 0:
            degenerate_none_relevant.append(entry["id"])

        rows.append({
            "id": entry["id"],
            "n": n,
            "two": counts[2],
            "one": counts[1],
            "zero": counts[0],
            "rate": relevant / n if n else 0.0,
            "query": entry["query"],
        })

    total = sum(overall.values())
    relevant_total = overall[2] + overall[1]

    print(f"\n{BOLD}Per-query grade distribution{RESET}\n")
    print(f"{'query':<7}{'n':>4}{'2':>5}{'1':>5}{'0':>5}{'rel%':>7}   question")
    print("-" * 96)
    for row in rows:
        flag = ""
        if row["n"] and row["rate"] == 1.0:
            flag = f"{RED} ALL-REL{RESET}"
        elif row["n"] and row["rate"] == 0.0:
            flag = f"{RED} NONE-REL{RESET}"
        elif row["rate"] > 0.85:
            flag = f"{YELLOW} high{RESET}"
        print(
            f"{row['id']:<7}{row['n']:>4}{row['two']:>5}{row['one']:>5}"
            f"{row['zero']:>5}{row['rate']:>6.0%}   {row['query'][:48]}{flag}"
        )

    print(f"\n{BOLD}Overall{RESET}")
    print(f"  judgments            {total}")
    print(f"  grade 2 (relevant)   {overall[2]:>5}  ({overall[2] / total:.0%})"
          if total else "")
    print(f"  grade 1 (partial)    {overall[1]:>5}  ({overall[1] / total:.0%})"
          if total else "")
    print(f"  grade 0 (irrelevant) {overall[0]:>5}  ({overall[0] / total:.0%})"
          if total else "")
    print(f"  relevance rate       {relevant_total / total:.0%}" if total else "")

    print(f"\n{BOLD}Checks{RESET}")
    problems = 0

    rate = relevant_total / total if total else 0.0
    if rate > 0.50:
        print(f"{YELLOW}  ! relevance rate {rate:.0%} is high for a pooled set "
              f"(typical: 10-30%).{RESET}")
        print(f"{DIM}    Consequence: nDCG saturates and the configs cannot "
              f"separate. If nearly\n    every pooled passage counts as "
              f"relevant, R1 and R4 will score alike and\n    the table will "
              f"say nothing about fusion or reranking.{RESET}")
        print(f"{DIM}    Most likely causes: grade 1 used for 'same topic' "
              f"rather than 'useful but\n    does not answer it', or a corpus "
              f"too narrow to contain real distractors.{RESET}")
        problems += 1
    else:
        print(f"{GREEN}  relevance rate {rate:.0%} — in the usual range{RESET}")

    if overall[0] < total * 0.2:
        print(f"{YELLOW}  ! only {overall[0] / total:.0%} graded 0. A pool with "
              f"few irrelevant passages\n    gives the reranker nothing to "
              f"demote.{RESET}")
        problems += 1

    if degenerate_all_relevant:
        print(f"{YELLOW}  ! {len(degenerate_all_relevant)} queries are ALL "
              f"relevant: {', '.join(degenerate_all_relevant[:8])}{RESET}")
        print(f"{DIM}    Every system scores ~1.0 on these; they add effort, "
              f"not discrimination.{RESET}")
        problems += 1

    if degenerate_none_relevant:
        print(f"{YELLOW}  ! {len(degenerate_none_relevant)} queries have NO "
              f"relevant passage: "
              f"{', '.join(degenerate_none_relevant[:8])}{RESET}")
        print(f"{DIM}    Excluded from scoring. This is a corpus gap — the "
              f"seed papers do not\n    cover these questions.{RESET}")
        problems += 1

    if unlabelled:
        print(f"{YELLOW}  ! {len(unlabelled)} queries are partially labelled "
              f"and will be excluded.{RESET}")
        problems += 1

    spread = {round(r["rate"], 1) for r in rows if r["n"]}
    if len(spread) <= 2 and len(rows) > 5:
        print(f"{YELLOW}  ! every query has nearly the same relevance rate. "
              f"The queries are not\n    probing different retrieval "
              f"behaviour.{RESET}")
        problems += 1

    print()
    if problems:
        print(f"{YELLOW}{problems} concern(s). The table will still run — read "
              f"it with these in mind,\nand say so in the writeup rather than "
              f"reporting the numbers bare.{RESET}\n")
    else:
        print(f"{GREEN}Distribution looks healthy. Run `evaluate`.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())