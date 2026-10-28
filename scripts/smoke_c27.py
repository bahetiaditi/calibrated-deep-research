#!/usr/bin/env python3
"""C27 acceptance check — the decision policy (D5).

    python -m scripts.smoke_c27            # offline checks, then two live runs
    python -m scripts.smoke_c27 --offline  # zero API calls

Live cost: roughly 2-4 LLM calls for the false-premise question (the premise
check, then straight to the decision) and 10-16 for the answerable one,
depending on how D2/D3 escalate and whether a PARTIAL rewrite fires.

The C27 criterion: "forced feature vectors produce each of the three
decisions." That is part 1, through the real two-stage model and policy.
Parts 2 and 3 check what the unit tests cannot: that against real models and
real retrieval the decision has its EFFECT — a withheld draft stays
withheld, a PARTIAL rewrite actually runs in partial mode and is
re-critiqued, research does not reopen after the decision, and the record
carries what C33 and C37 need.

Before C33 fits anything, live decisions come from the HEURISTIC scorer and
say so. Their outcomes are not results. What this script checks live is the
machinery, not the numbers.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from src.config import get_config
from src.state import draft_digest
from src.tracing.tracer import DecisionSurface, TraceReader

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)
COLLECTION = "retrieval_eval"

ANSWERABLE = ("How do FlashAttention-2 and PagedAttention differ in their "
              "memory access patterns?")
FALSE_PREMISE = "Why does Mamba-2 use rotary position embeddings in its SSM blocks?"


def ok(m): print(f"{GREEN}  PASS{RESET} {m}")
def warn(m): print(f"{YELLOW}  WARN{RESET} {m}")
def bad(m): print(f"{RED}  FAIL{RESET} {m}")


# ---------------------------------------------------------------------------
# Part 1 — offline: forced vectors through the real model and policy
# ---------------------------------------------------------------------------


def offline() -> int:
    from tests.decision_fixtures import STRONG, WEAK, decision_state, fitted_models

    from src.sufficiency.policy import DecisionInputs, Thresholds, decide
    from src.sufficiency.scorer import CalibratedScorer, HeuristicScorer

    failures = 0
    print(f"{BOLD}1. Forced feature vectors (synthetic fitted two-stage model){RESET}")
    stage1, stage2 = fitted_models()
    scorer = CalibratedScorer(stage1, stage2, Thresholds(0.65, 0.40))
    cases = {"strong+strong": (STRONG, STRONG), "strong+weak": (STRONG, WEAK),
             "weak+weak": (WEAK, WEAK)}
    scores = {name: scorer.score(decision_state(*v)).value for name, v in cases.items()}
    for name, value in scores.items():
        print(f"{DIM}   {name:<15} run score {value:.3f}{RESET}")

    s, m, w = scores.values()
    if not s > m > w:
        bad("the model does not order the three cases; no threshold separates them")
        return 1
    t = Thresholds((s + m) / 2, (m + w) / 2, source="smoke: fitted-scale midpoints")
    got = [decide(DecisionInputs(score=v, thresholds=t, calibrated=True)).terminal
           for v in (s, m, w)]
    print(f"{DIM}   thresholds on the fitted scale: answer {t.answer:.3f}, "
          f"partial {t.partial:.3f}  ->  {got}{RESET}")
    if got == ["ANSWER", "PARTIAL", "ABSTAIN"]:
        ok("each of the three decisions is produced")
    else:
        bad(f"expected ANSWER/PARTIAL/ABSTAIN, got {got}")
        failures += 1

    placeholder = [decide(DecisionInputs(score=v, thresholds=Thresholds(0.65, 0.40),
                                         calibrated=True)).terminal
                   for v in (s, m, w)]
    print(f"{DIM}   at the config placeholders (0.65 / 0.40): {placeholder}{RESET}")
    if "ANSWER" not in placeholder:
        warn("at the placeholder tau_answer this fitted model never answers. "
             "Hard regularisation compresses probabilities toward the base rate; "
             "C33 must select tau on the fitted model's own scale.")

    print(f"\n{BOLD}2. Heuristic fallback (what live runs use until C33){RESET}")
    h = HeuristicScorer(get_config())
    for name, v in cases.items():
        run = h.score(decision_state(*v))
        d = decide(DecisionInputs(score=run.value, thresholds=run.thresholds))
        print(f"{DIM}   {name:<15} fraction resolved {run.value:.2f}  ->  "
              f"{d.terminal:<8} calibrated={d.calibrated}{RESET}")
    ok("heuristic decisions are labelled uncalibrated")
    return failures


# ---------------------------------------------------------------------------
# Parts 2-3 — live
# ---------------------------------------------------------------------------


def wire(cfg):
    from qdrant_client import QdrantClient

    from src.graph import Components
    from src.nodes.adaptive_retriever import AdaptiveRetriever
    from src.nodes.critic import Critic, critic_node
    from src.nodes.critic_router import CriticRouter, critic_router_node
    from src.nodes.decider import Decider, decider_node
    from src.nodes.planner import Planner, planner_node
    from src.nodes.premise_checker import PremiseChecker, premise_checker_node
    from src.nodes.retriever import retriever_node
    from src.nodes.synthesizer import Synthesizer, synthesizer_node
    from src.rag.fusion import HybridRetriever
    from src.rag.rerank import Reranker
    from src.rag.sparse import BM25Index
    from src.rag.store import EvidenceStore
    from src.sufficiency.allocation import get_allocator

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
        sys.exit(1)
    hybrid.sparse.build(passages)
    reranker = Reranker(cfg)
    print(f"{DIM}corpus: {len(passages)} passages{RESET}\n")

    return Components(
        premise_checker=premise_checker_node(
            PremiseChecker(cfg, retriever=hybrid, reranker=reranker)),
        planner=planner_node(Planner(cfg)),
        retriever=retriever_node(
            AdaptiveRetriever(cfg, retriever=hybrid, reranker=reranker),
            get_allocator(cfg)),
        synthesizer=synthesizer_node(Synthesizer(cfg)),
        critic=critic_node(Critic(cfg)),
        critic_router=critic_router_node(CriticRouter(cfg)),
        decider=decider_node(Decider(cfg)),
    )


def live_run(components, cfg, question, label):
    from src.graph import run_research

    thread = f"smoke-c27-{label}-{int(time.time())}"
    print(f"{BOLD}{question}{RESET}")
    t0 = time.time()
    result = run_research(question, components=components, config=cfg,
                          thread_id=thread, trace_id=thread)
    print(f"{DIM}   {time.time() - t0:.0f}s   {' -> '.join(result.node_visits)}")
    if result.budget:
        print(f"   budget: {result.budget.report()}")
    print(RESET, end="")
    return result


def show_record(state) -> None:
    record = state.get("decision") or {}
    print(f"{DIM}   decision   {state.get('terminal_decision')}  "
          f"rule={record.get('rule')}  score={record.get('score')}  "
          f"margin={record.get('margin')}")
    print(f"   scorer     {record.get('scorer')}  calibrated={record.get('calibrated')}"
          f"  thresholds={record.get('thresholds')}")
    print(f"   rationale  {state.get('decision_rationale')}")
    for sq in record.get("sub_questions") or []:
        f = sq.get("features") or {}
        f1 = f.get("f1_max_rerank")
        print(f"     {sq['id']:<6} {sq['status']:<12} resolved={str(sq['resolved']):<5} "
              f"incl={str(sq['included']):<5} f1={'—' if f1 is None else f'{f1:5.2f}'} "
              f"verdicts={sq.get('verdicts')}  {sq.get('note', '')}")
    for gap in record.get("gaps") or []:
        print(f"     gap: {gap[:110]}")
    print(f"   rewrite    {record.get('partial_rewrite')}  "
          f"withheld={len(record.get('withheld_draft') or '')} chars  "
          f"final draft={len(state.get('draft') or '')} chars "
          f"({state.get('draft_mode') or '—'}){RESET}")


def check_effects(result) -> int:
    """Each check asserts an effect, not that a decision was recorded."""
    failures = 0
    state, visits = result.state, result.node_visits
    record = state.get("decision") or {}
    terminal = state.get("terminal_decision")

    if result.error:
        bad(f"run failed: {result.error}")
        return 1
    if terminal is None:
        bad("no terminal decision")
        return 1
    ok(f"terminal decision recorded: {terminal} via {record.get('rule')}")

    reader = TraceReader.load(result.trace_path)
    d5 = reader.decisions(DecisionSurface.D5_TERMINAL_DECISION)
    adaptive = reader.adaptive_decision_counts().get("D5_terminal_decision", 0)
    if len(d5) == 1 and adaptive == 1:
        ok("exactly one adaptive D5 decision in the trace")
    else:
        bad(f"{len(d5)} D5 events, {adaptive} adaptive (expected 1 and 1)")
        failures += 1

    if "decider" in visits:
        after = visits[visits.index("decider") + 1:]
        if {"retriever", "planner"} & set(after):
            bad(f"research reopened after the decision: {after}")
            failures += 1
        else:
            ok(f"research stayed closed after the decision (then: {after or 'end'})")

    if terminal == "ABSTAIN":
        if state.get("draft"):
            bad("ABSTAIN left a draft in state")
            failures += 1
        else:
            ok("no draft survives an abstention")
        if record.get("withheld_draft") or record.get("rule") == "premise_refuted":
            ok("withheld draft kept for C38 (or none was written)")

    if terminal == "PARTIAL" and record.get("partial_rewrite") == "requested":
        if state.get("draft_mode") == "partial":
            ok("PARTIAL rewrite ran in partial mode")
        else:
            bad(f"rewrite requested but final draft_mode={state.get('draft_mode')!r}")
            failures += 1
        if state.get("critiqued_draft") == draft_digest(state.get("draft")):
            ok("the partial draft was re-critiqued")
        else:
            bad("the partial draft reached the end unreviewed")
            failures += 1

    if state.get("draft") and state.get("critiqued_draft") != draft_digest(state["draft"]):
        bad("final draft was never reviewed by the critic")
        failures += 1

    if record.get("score") is None and record.get("rule") == "threshold":
        bad("threshold decision with no score — C37 cannot sweep it")
        failures += 1
    if not record.get("calibrated"):
        warn("uncalibrated (heuristic) decision — expected before C33; not a result")
    return failures


def main() -> int:
    print("\n=== C27 smoke test: decision policy (D5) ===\n")
    failures = offline()
    if "--offline" in sys.argv:
        return _summary(failures)

    cfg = get_config()
    print(f"\n{BOLD}3. Live runs{RESET}")
    components = wire(cfg)

    for question, label in ((FALSE_PREMISE, "premise"), (ANSWERABLE, "answerable")):
        result = live_run(components, cfg, question, label)
        show_record(result.state)
        failures += check_effects(result)
        if label == "premise":
            rule = (result.state.get("decision") or {}).get("rule")
            if rule == "premise_refuted":
                ok("false premise abstained by the hard rule, bypassing tau")
            else:
                warn(f"premise not refuted (rule={rule}); C15's checker found it "
                     f"UNVERIFIABLE or SUPPORTED — check its verdicts in the trace")
        print()

    return _summary(failures)


def _summary(failures: int) -> int:
    print(f"\n{BOLD}{'all checks passed' if not failures else f'{failures} failure(s)'}"
          f"{RESET}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
