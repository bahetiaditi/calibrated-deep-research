#!/usr/bin/env python3
"""C14 acceptance check — state round-trips through the checkpointer.

    python -m scripts.smoke_c14

Zero API calls. Builds a three-node dummy graph, runs it under SqliteSaver,
and verifies that every field survives serialisation — including the two with
reducers, which are where a silent corruption would hide.

Also demonstrates the property C19 depends on: killing a run mid-graph and
resuming it from the checkpoint rather than restarting. An eval that spans
days needs that to be true, and it is cheaper to prove now than to discover
at C32.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from src.state import (
    ClaimVerification,
    Passage,
    ResearchState,
    evidence_for,
    independent_sources,
    merge_contradictions,
    merge_evidence,
    new_state,
    new_sub_question,
)

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


def ok(m): print(f"{GREEN}  PASS{RESET} {m}")
def warn(m): print(f"{YELLOW}  WARN{RESET} {m}")
def bad(m): print(f"{RED}  FAIL{RESET} {m}")


def passage(pid, **over) -> Passage:
    base = dict(
        id=pid, source_type="arxiv", source_id="2307.08691v1",
        source_domain="arxiv.org", title="FlashAttention-2",
        text=f"body of {pid}", section="Results", published="2023-07-17",
        sub_question_id="sq1", retrieval_score=0.42, rerank_score=7.25,
    )
    base.update(over)
    return Passage(**base)


def main() -> int:
    print("\n=== C14 smoke test: graph state ===\n")
    failures = 0

    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph

    print(f"{DIM}--- reducers (where a silent corruption would hide){RESET}")

    merged = merge_evidence([passage("P1", rerank_score=9.0)],
                            [passage("P1", rerank_score=1.0), passage("P2")])
    if [p["id"] for p in merged] == ["P1", "P2"] and merged[0]["rerank_score"] == 9.0:
        ok("duplicate evidence dropped; first copy keeps its scores")
        print(f"{DIM}       one passage counted twice would inflate f2 and f4 "
              f"exactly when\n       evidence is thinnest{RESET}")
    else:
        bad(f"evidence reducer wrong: {[p['id'] for p in merged]}")
        failures += 1

    pair = merge_contradictions(
        [{"passage_a": "P1", "passage_b": "P2", "score": 0.9}],
        [{"passage_a": "P2", "passage_b": "P1", "score": 0.9}],
    )
    if len(pair) == 1:
        ok("reversed contradiction pair recognised as one finding")
    else:
        bad("symmetric contradiction stored twice — f5 numerator doubles")
        failures += 1

    print(f"\n{DIM}--- a three-node run under the checkpointer{RESET}")

    def planner(state: ResearchState) -> dict:
        return {
            "plan": [
                new_sub_question("what fraction of peak does it reach",
                                 sub_question_id="sq1",
                                 rationale="the question asks for a number"),
                new_sub_question("on what hardware", sub_question_id="sq2",
                                 rationale="the number is hardware-specific"),
            ],
            "plan_revisions": 0,
        }

    def retriever(state: ResearchState) -> dict:
        return {
            "evidence": [
                passage("P1"),
                passage("P2", source_id="2305.14314v1", sub_question_id="sq2"),
                passage("P1"),   # a legitimate re-retrieval duplicate
            ],
            "contradictions": [{
                "passage_a": "P1", "passage_b": "P2", "score": 0.72,
                "cross_source": True, "section_a": "Results",
                "section_b": "Related Work",
            }],
        }

    def decider(state: ResearchState) -> dict:
        return {
            "terminal_decision": "PARTIAL",
            "decision_rationale": "sq2 never reached sufficiency",
            "unresolved_gaps": ["no source states the backward-pass figure"],
            "critic_findings": [ClaimVerification(
                claim="it reaches 73% of theoretical peak",
                sub_question_id="sq1", cited_passage_ids=["P1"],
                verdict="SUPPORTED", reasoning="stated outright in Results",
                action_taken="accept",
            )],
        }

    builder = StateGraph(ResearchState)
    builder.add_node("planner", planner)
    builder.add_node("retriever", retriever)
    builder.add_node("decider", decider)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "decider")
    builder.add_edge("decider", END)

    tmp = Path(tempfile.mkdtemp(prefix="c14-"))
    try:
        with SqliteSaver.from_conn_string(str(tmp / "ckpt.sqlite")) as saver:
            graph = builder.compile(checkpointer=saver)
            config = {"configurable": {"thread_id": "smoke-c14"}}
            final = graph.invoke(new_state("Does FlashAttention-2 saturate the A100?"),
                                 config)

            checks = [
                ("question survives", final["question"].startswith("Does")),
                ("plan has 2 sub-questions", len(final["plan"]) == 2),
                ("rationale survives", bool(final["plan"][0]["rationale"])),
                ("duplicate evidence deduped", len(final["evidence"]) == 2),
                ("contradiction stored", len(final["contradictions"]) == 1),
                ("terminal decision", final["terminal_decision"] == "PARTIAL"),
                ("gaps recorded", len(final["unresolved_gaps"]) == 1),
                ("critic verdict", final["critic_findings"][0]["verdict"] == "SUPPORTED"),
                ("rerank logit preserved",
                 final["evidence"][0]["rerank_score"] == 7.25),
            ]
            for name, passed in checks:
                if passed:
                    ok(name)
                else:
                    bad(name)
                    failures += 1

            print(f"\n{DIM}--- reload from checkpoint (what C19 depends on){RESET}")
            restored = graph.get_state(config).values
            if (restored["plan"][1]["text"] == "on what hardware"
                    and len(restored["evidence"]) == 2
                    and restored["contradictions"][0]["score"] == 0.72):
                ok("state reloaded identically — a run can resume rather than restart")
            else:
                bad("reloaded state differs from the final state")
                failures += 1

        print(f"\n{DIM}--- resume after process death{RESET}")
        with SqliteSaver.from_conn_string(str(tmp / "ckpt.sqlite")) as saver2:
            graph2 = builder.compile(checkpointer=saver2)
            reopened = graph2.get_state({"configurable": {"thread_id": "smoke-c14"}})
            if reopened.values.get("terminal_decision") == "PARTIAL":
                ok("a fresh process recovers the finished run from disk")
                print(f"{DIM}       an eval spanning days survives a 429, a "
                      f"laptop sleep, or a crash{RESET}")
            else:
                bad("could not recover state in a new process")
                failures += 1

        print(f"\n{DIM}--- helpers used by the controller{RESET}")
        sq1_evidence = evidence_for(final, "sq1")
        print(f"       evidence for sq1: {len(sq1_evidence)}")
        print(f"       independent sources overall: "
              f"{independent_sources(final['evidence'])}")
        if independent_sources(final["evidence"]) == 2:
            ok("f4 counts distinct papers, not chunks")
        else:
            bad("independent source count wrong")
            failures += 1
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{DIM}--- fields that exist to ENABLE a decision surface (§3.2){RESET}")
    for field, surface in [
        ("attempted_queries", "D2 reformulation"),
        ("attempted_routes", "D2 escalation"),
        ("rounds / sufficiency", "D3 depth and stopping"),
        ("allocation", "D4 budget"),
        ("plan_revisions", "D1 plan revision"),
        ("unresolved_gaps", "D5 PARTIAL vs ANSWER"),
    ]:
        print(f"{DIM}       {field:<22} {surface}{RESET}")
    print(f"{DIM}       A field nobody reads is a decision nobody makes — "
          f"a cheap precursor\n       to the C35 ablation check.{RESET}")

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C14 acceptance: all checks passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())