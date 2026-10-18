"""Tests for the static retriever.

This file is the A2/A3 baseline, not a placeholder. A deliberately crippled
baseline would make C21's adaptive version look good for the wrong reason and
leave the agency claim unfalsifiable — which is exactly what §2.1 exists to
prevent. So the tests check that the static policy is a *fair* opponent and
that its bookkeeping matches what an adaptive run records.
"""
import pytest

from src.config import Config
from src.nodes.retriever import StaticRetriever, retriever_node
from src.state import (
    Passage,
    evidence_for,
    find_sub_question,
    merge_evidence,
    new_state,
    new_sub_question,
)

CFG = Config({
    "agency": {"static": {"route_rule": "arxiv_meta", "fixed_top_k": 3}},
    "retrieval": {"fusion": {"candidates_per_retriever": 20}},
})


def passage(pid, score=7.0, sub_question_id="ingest-time"):
    return Passage(
        id=pid, source_type="arxiv", source_id=f"paper-{pid}",
        source_domain="arxiv.org", title="t", text=f"text {pid}",
        section="Results", published="2023-01-01",
        sub_question_id=sub_question_id, retrieval_score=0.1,
        rerank_score=score,
    )


class FakeResult:
    def __init__(self, p):
        self.passage = p


class FakeRetriever:
    def __init__(self, results=None, error=None):
        self.results = results if results is not None else [
            passage("P1", 9.0), passage("P2", 6.0), passage("P3", 2.0)
        ]
        self.error = error
        self.queries = []

    def retrieve(self, query, *, top_k, **kw):
        self.queries.append({"query": query, "top_k": top_k, **kw})
        if self.error:
            raise self.error
        return [FakeResult(p) for p in self.results]


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, results, *, top_k=None):
        self.calls.append({"query": query, "top_k": top_k})
        return list(results)[: top_k or 10]


def build(cfg=CFG, retriever=None, **kw):
    return StaticRetriever(
        cfg, retriever=retriever or FakeRetriever(),
        reranker=FakeReranker(), **kw,
    )


def state_with(*texts):
    state = new_state("root question")
    state["plan"] = [
        new_sub_question(t, sub_question_id=f"sq{i + 1}")
        for i, t in enumerate(texts)
    ]
    return state


# --- the static policy -----------------------------------------------------


def test_query_is_the_sub_question_verbatim():
    """D2 off: no reformulation."""
    retriever = FakeRetriever()
    state = state_with("How does PagedAttention manage the KV cache?")
    build(retriever=retriever).retrieve_one(state, "sq1")
    assert retriever.queries[0]["query"] == "How does PagedAttention manage the KV cache?"


def test_route_is_fixed_regardless_of_the_sub_question():
    retriever = build()
    state = state_with("a question about papers", "a question about blogs")
    for sqid in ("sq1", "sq2"):
        update = retriever.retrieve_one(state, sqid)
        state["plan"] = update["plan"]
    routes = [sq["attempted_routes"] for sq in state["plan"]]
    assert routes == [["arxiv_meta"], ["arxiv_meta"]]


def test_route_ignores_history():
    """D2 off: a second round would pick the same route. The adaptive
    version at C21 is what reads attempted_routes."""
    state = state_with("q")
    state["plan"][0]["attempted_routes"] = ["arxiv_meta", "web"]
    assert build().choose_route(state["plan"][0]) == "arxiv_meta"


def test_planner_route_honoured_when_configured():
    cfg = Config({
        "agency": {"static": {"route_rule": "planner", "fixed_top_k": 3}},
        "retrieval": {"fusion": {"candidates_per_retriever": 20}},
    })
    sq = new_sub_question("q", sub_question_id="sq1")
    sq["suggested_route"] = "arxiv_fulltext"
    assert build(cfg).choose_route(sq) == "arxiv_fulltext"


def test_invalid_configured_route_falls_back():
    cfg = Config({
        "agency": {"static": {"route_rule": "telepathy", "fixed_top_k": 3}},
        "retrieval": {"fusion": {"candidates_per_retriever": 20}},
    })
    assert build(cfg).static_route == "arxiv_meta"


def test_fixed_top_k_from_config():
    reranker = FakeReranker()
    r = StaticRetriever(CFG, retriever=FakeRetriever(), reranker=reranker)
    r.retrieve_one(state_with("q"), "sq1")
    assert reranker.calls[0]["top_k"] == 3


def test_always_stops_after_one_round():
    """D3 off."""
    state = state_with("q")
    update = build().retrieve_one(state, "sq1")
    assert update["plan"][0]["status"] == "sufficient"
    assert update["plan"][0]["rounds"] == 1


def test_no_results_marks_insufficient_not_sufficient():
    """Deterministic bookkeeping, not a judgment — but marking everything
    sufficient would hide from D5 that a sub-question found nothing."""
    state = state_with("q")
    update = build(retriever=FakeRetriever(results=[])).retrieve_one(state, "sq1")
    assert update["plan"][0]["status"] == "insufficient"


# --- bookkeeping (must match what an adaptive run records) -----------------


def test_attempted_route_and_query_are_recorded():
    """An A2/A3 trace has to be comparable to a full run, or C35 cannot tell
    a frozen decision from a missing one."""
    state = state_with("my sub-question")
    sq = build().retrieve_one(state, "sq1")["plan"][0]
    assert sq["attempted_routes"] == ["arxiv_meta"]
    assert sq["attempted_queries"] == ["my sub-question"]


def test_rounds_increment():
    retriever = build()
    state = state_with("q")
    update = retriever.retrieve_one(state, "sq1")
    state["plan"] = update["plan"]
    state["plan"][0]["status"] = "pending"      # simulate a second round
    assert retriever.retrieve_one(state, "sq1")["plan"][0]["rounds"] == 2


def test_evidence_ids_recorded_in_rank_order():
    state = state_with("q")
    sq = build().retrieve_one(state, "sq1")["plan"][0]
    assert sq["evidence_ids"] == ["P1", "P2", "P3"]


def test_evidence_ids_do_not_duplicate_across_rounds():
    retriever = build()
    state = state_with("q")
    update = retriever.retrieve_one(state, "sq1")
    state["plan"] = update["plan"]
    state["plan"][0]["status"] = "pending"
    sq = retriever.retrieve_one(state, "sq1")["plan"][0]
    assert sq["evidence_ids"] == ["P1", "P2", "P3"]


def test_passages_are_stamped_with_the_retrieving_sub_question():
    state = state_with("q")
    update = build().retrieve_one(state, "sq1")
    assert all(p["sub_question_id"] == "sq1" for p in update["evidence"])


def test_shared_evidence_stays_visible_to_both_sub_questions():
    """The C17 schema fix. merge_evidence dedupes by id, so without
    evidence_ids the second sub-question would appear to have retrieved
    nothing — and sibling sub-questions share evidence constantly."""
    retriever = build()
    state = state_with("first", "second")

    update = retriever.retrieve_one(state, "sq1")
    state["plan"] = update["plan"]
    state["evidence"] = merge_evidence(state["evidence"], update["evidence"])

    update = retriever.retrieve_one(state, "sq2")
    state["plan"] = update["plan"]
    state["evidence"] = merge_evidence(state["evidence"], update["evidence"])

    assert len(state["evidence"]) == 3          # deduped
    assert len(evidence_for(state, "sq1")) == 3
    assert len(evidence_for(state, "sq2")) == 3   # not hidden


def test_evidence_for_preserves_rank_order():
    retriever = build(retriever=FakeRetriever(results=[
        passage("Pc", 9.0), passage("Pa", 5.0), passage("Pb", 1.0)
    ]))
    state = state_with("q")
    update = retriever.retrieve_one(state, "sq1")
    state["plan"] = update["plan"]
    state["evidence"] = merge_evidence(state["evidence"], update["evidence"])
    assert [p["id"] for p in evidence_for(state, "sq1")] == ["Pc", "Pa", "Pb"]


def test_other_sub_questions_are_untouched():
    state = state_with("a", "b")
    update = build().retrieve_one(state, "sq1")
    assert update["plan"][1]["status"] == "pending"
    assert update["plan"][1]["rounds"] == 0


def test_unknown_sub_question_is_a_no_op():
    assert build().retrieve_one(state_with("q"), "nope") == {}


# --- degradation -----------------------------------------------------------


def test_retrieval_failure_does_not_raise():
    """Same contract as C6/C7: a failure is information the agent acts on."""
    state = state_with("q")
    update = build(retriever=FakeRetriever(error=RuntimeError("down"))).retrieve_one(
        state, "sq1"
    )
    assert update["evidence"] == []
    assert update["plan"][0]["status"] == "insufficient"
    assert update["plan"][0]["rounds"] == 1       # the attempt is still recorded


def test_failed_round_still_records_the_attempt():
    state = state_with("q")
    sq = build(retriever=FakeRetriever(error=RuntimeError("x"))).retrieve_one(
        state, "sq1"
    )["plan"][0]
    assert sq["attempted_routes"] == ["arxiv_meta"]
    assert sq["attempted_queries"] == ["q"]


# --- budget ----------------------------------------------------------------


def test_budget_charged_per_round():
    from src.budget import Budget

    budget = Budget(llm_calls_max=25, retrieval_calls_max=20,
                    wall_clock_max_s=900)
    retriever = build(budget=budget)
    state = state_with("a", "b")
    for sqid in ("sq1", "sq2"):
        update = retriever.retrieve_one(state, sqid)
        state["plan"] = update["plan"]
    assert budget.retrieval_calls_used == 2


def test_failed_retrieval_is_not_charged():
    from src.budget import Budget

    budget = Budget(llm_calls_max=25, retrieval_calls_max=20,
                    wall_clock_max_s=900)
    build(retriever=FakeRetriever(error=RuntimeError("x")),
          budget=budget).retrieve_one(state_with("q"), "sq1")
    assert budget.retrieval_calls_used == 0


def test_retrieve_all_stops_when_budget_runs_out():
    from src.budget import Budget

    budget = Budget(llm_calls_max=25, retrieval_calls_max=2,
                    wall_clock_max_s=900)
    final = build(budget=budget).retrieve_all(state_with("a", "b", "c", "d"))
    assert budget.retrieval_calls_used == 2
    assert sum(1 for sq in final["plan"] if sq["status"] == "pending") == 2


# --- retrieve_all ----------------------------------------------------------


def test_retrieve_all_covers_every_sub_question():
    """The C17 acceptance criterion."""
    final = build().retrieve_all(state_with("a", "b", "c"))
    assert all(sq["status"] != "pending" for sq in final["plan"])
    assert all(sq["rounds"] == 1 for sq in final["plan"])


def test_retrieve_all_accumulates_deduplicated_evidence():
    final = build().retrieve_all(state_with("a", "b", "c"))
    assert len(final["evidence"]) == 3      # same 3 passages, deduped


def test_retrieve_all_on_empty_plan():
    assert build().retrieve_all(new_state("q"))["plan"] == []


# --- tracing ---------------------------------------------------------------


def test_decisions_are_traced_as_non_adaptive():
    """C35 counts adaptive decisions per surface. An A2/A3 run must report
    zero for D2 and D3 — but the decision points still have to appear, or a
    frozen decision is indistinguishable from a missing one."""
    from src.tracing.tracer import DecisionSurface

    events = []

    class FakeTracer:
        def decision(self, surface, **kw):
            events.append((surface, kw))

        def retrieval(self, **kw):
            pass

    build(tracer=FakeTracer()).retrieve_one(state_with("q"), "sq1")
    surfaces = {s for s, _ in events}
    assert surfaces == {
        DecisionSurface.D2_ROUTE_AND_QUERY,
        DecisionSurface.D3_DEPTH_AND_STOPPING,
    }
    assert all(kw["was_adaptive"] is False for _, kw in events)


def test_retrieval_event_is_traced():
    events = []

    class FakeTracer:
        def decision(self, surface, **kw):
            pass

        def retrieval(self, **kw):
            events.append(kw)

    build(tracer=FakeTracer()).retrieve_one(state_with("q"), "sq1")
    assert events[0]["route"] == "arxiv_meta"
    assert events[0]["n_results"] == 3
    assert events[0]["sub_question_id"] == "sq1"


# --- node ------------------------------------------------------------------


def test_node_handles_one_sub_question_per_call():
    """The controller re-enters this node and chooses which sub-question to
    work on — that choice is D4's, and doing the whole plan here would take
    it away."""
    node = retriever_node(build())
    state = state_with("a", "b")
    update = node(state)
    statuses = [sq["status"] for sq in update["plan"]]
    assert statuses == ["sufficient", "pending"]


def test_node_is_a_no_op_when_nothing_is_pending():
    state = state_with("a")
    state["plan"][0]["status"] = "sufficient"
    assert retriever_node(build())(state) == {}