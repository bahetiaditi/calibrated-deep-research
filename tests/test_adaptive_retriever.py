"""Tests for the adaptive retriever (D2 + D3 live).

The point of these is that the decisions are genuinely CONDITIONED — on what
has been tried and on what came back. A retriever that escalates on a fixed
schedule would pass a naive "did it do two rounds?" check while being exactly
the retry loop §2.4 warns about.
"""
import pytest

from src.config import Config
from src.nodes.adaptive_retriever import (
    ABANDONED,
    IN_PROGRESS,
    INSUFFICIENT,
    SUFFICIENT,
    AdaptiveRetriever,
    outcome_features,
)
from src.nodes.retriever import RetrievalOutcome
from src.state import Passage, new_state, new_sub_question

CFG = Config({
    "agency": {
        "d2_adaptive_route_and_query": True,
        "d3_adaptive_depth_and_stopping": True,
        "static": {"route_rule": "arxiv_meta", "fixed_top_k": 3},
    },
    "budget": {"max_rounds_per_subquestion": 3},
    "retrieval": {
        "fusion": {"candidates_per_retriever": 20},
        "adaptive": {
            "sufficient_min_rerank": 4.0,
            "min_independent_sources": 2,
            "hopeless_max_rerank": 0.0,
            "reformulate_with_llm": True,
        },
    },
})


def passage(pid, score=7.0, source=None):
    return Passage(
        id=pid, source_type="arxiv", source_id=source or f"paper-{pid}",
        source_domain="arxiv.org", title="t", text=f"text {pid}",
        section="Results", published="2024-01-01", sub_question_id="sq1",
        retrieval_score=0.1, rerank_score=score,
    )


class FakeResult:
    def __init__(self, p):
        self.passage = p


class FakeRetriever:
    def __init__(self, by_query=None, default=None):
        self.by_query = by_query or {}
        self.default = default if default is not None else [
            passage("P1", 9.0, "A"), passage("P2", 6.0, "B")
        ]
        self.queries = []

    def retrieve(self, query, *, top_k, **kw):
        self.queries.append(query)
        return [FakeResult(p) for p in self.by_query.get(query, self.default)]


class FakeReranker:
    def rerank(self, query, results, *, top_k=None):
        return list(results)[: top_k or 10]


class FakeProvider:
    def __init__(self, decisions=None, error=None):
        self.decisions = list(decisions or [])
        self.error = error
        self.calls = []

    def complete_json(self, system, user, **kw):
        self.calls.append({"system": system, "user": user})
        if self.error:
            raise self.error
        return (self.decisions.pop(0) if self.decisions else
                {"action": "wider", "route": "web", "query": "rewritten",
                 "reasoning": "r"}), None


def build(cfg=CFG, retriever=None, provider=None, **kw):
    return AdaptiveRetriever(
        cfg, retriever=retriever or FakeRetriever(), reranker=FakeReranker(),
        provider=provider or FakeProvider(), **kw,
    )


def state_with(*texts, route="arxiv_meta"):
    state = new_state("parent question")
    state["plan"] = []
    for i, text in enumerate(texts):
        sq = new_sub_question(text, sub_question_id=f"sq{i + 1}")
        sq["suggested_route"] = route
        state["plan"].append(sq)
    return state


def outcome(scores, sources=None):
    passages = [
        passage(f"P{i}", s, (sources or [f"src{i}" for i in range(len(scores))])[i])
        for i, s in enumerate(scores)
    ]
    return RetrievalOutcome("sq1", "arxiv_meta", "q", passages, list(scores), 0.1)


# --- D3: stopping is conditioned on the evidence ---------------------------


def test_strong_evidence_stops_immediately():
    sq = new_sub_question("q", sub_question_id="sq1")
    assert build().decide_next(sq, outcome([8.6, 7.0], ["A", "B"])) == SUFFICIENT


def test_strong_evidence_from_one_source_does_not_stop():
    """A single source clearing the bar is not corroboration."""
    sq = new_sub_question("q", sub_question_id="sq1")
    assert build().decide_next(sq, outcome([8.6, 7.0], ["A", "A"])) != SUFFICIENT


def test_weak_evidence_escalates():
    sq = new_sub_question("q", sub_question_id="sq1")
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["q"]
    assert build().decide_next(sq, outcome([2.0, 1.0], ["A", "B"])) == IN_PROGRESS


def test_nothing_retrieved_escalates():
    sq = new_sub_question("q", sub_question_id="sq1")
    assert build().decide_next(sq, outcome([])) == IN_PROGRESS


def test_hopeless_evidence_is_abandoned_once_it_stops_improving():
    """An admitted gap beats a confident guess. But the stop is triggered by
    the lack of improvement, not by the level alone — a sub-question still
    climbing from deep below the bar gets its remaining rounds."""
    retriever = build()
    sq = new_sub_question("q", sub_question_id="sq1")
    retriever.decide_next(sq, outcome([-2.0], ["A"]))      # round 1
    sq["rounds"] = 1
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["q"]
    # Round 2 barely moves: no further work is coming, and the level says
    # this is a gap rather than a thin answer.
    assert retriever.decide_next(sq, outcome([-1.9, -2.5], ["A", "B"])) == ABANDONED


def test_hopeless_on_the_first_round_still_gets_a_second_chance():
    sq = new_sub_question("q", sub_question_id="sq1")
    assert build().decide_next(sq, outcome([-1.9], ["A"])) == IN_PROGRESS


def test_round_limit_reports_the_gap_rather_than_claiming_success():
    sq = new_sub_question("q", sub_question_id="sq1")
    sq["rounds"] = 2       # this will be the third
    assert build().decide_next(sq, outcome([2.0, 1.5], ["A", "B"])) == INSUFFICIENT


def test_round_limit_accepts_genuinely_good_evidence():
    sq = new_sub_question("q", sub_question_id="sq1")
    sq["rounds"] = 2
    assert build().decide_next(sq, outcome([8.0], ["A"])) == SUFFICIENT


def test_stopping_decision_depends_on_the_scores_not_the_round():
    """The distinction that separates adaptivity from a schedule."""
    retriever = build()
    strong = new_sub_question("q", sub_question_id="sq1")
    weak = new_sub_question("q", sub_question_id="sq2")
    weak["attempted_routes"] = ["arxiv_meta"]
    weak["attempted_queries"] = ["q"]
    assert retriever.decide_next(strong, outcome([9.0, 8.0], ["A", "B"])) == SUFFICIENT
    assert retriever.decide_next(weak, outcome([1.0, 0.5], ["A", "B"])) != SUFFICIENT


# --- D2: route and query are conditioned on history ------------------------


def test_first_attempt_uses_the_planners_route():
    """Honoured for cheap routes; the expensive one is clamped — see
    test_the_first_attempt_never_starts_at_full_text."""
    sq = new_sub_question("q", sub_question_id="sq1")
    sq["suggested_route"] = "web"
    assert build().choose_route(sq) == "web"


def test_first_attempt_uses_the_sub_question_verbatim():
    sq = new_sub_question("what is RRF", sub_question_id="sq1")
    assert build().formulate_query(sq) == "what is RRF"


def test_second_attempt_uses_the_reformulated_query():
    provider = FakeProvider([{
        "action": "wider", "route": "web",
        "query": "reciprocal rank fusion ranking algorithm", "reasoning": "r",
    }])
    retriever = build(provider=provider)
    sq = new_sub_question("what is RRF", sub_question_id="sq1")
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["what is RRF"]

    retriever.decide_next(sq, outcome([1.0], ["A"]))
    assert retriever.formulate_query(sq) == "reciprocal rank fusion ranking algorithm"
    assert retriever.choose_route(sq) == "web"


def test_a_repeated_query_is_rejected():
    """A rephrasing that returns the same passages is a retry, not a
    reformulation — §2.4's second anti-pattern."""
    provider = FakeProvider([{
        "action": "wider", "route": "web", "query": "what is RRF",
        "reasoning": "r",
    }])
    retriever = build(provider=provider)
    sq = new_sub_question("what is RRF", sub_question_id="sq1")
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["what is RRF"]
    retriever.decide_next(sq, outcome([1.0], ["A"]))
    assert retriever.formulate_query(sq) == "what is RRF"   # falls back, no loop


def test_route_escalates_rather_than_repeating_a_failure():
    retriever = build(provider=FakeProvider(error=RuntimeError("down")))
    sq = new_sub_question("q", sub_question_id="sq1")
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["q"]
    retriever.decide_next(sq, outcome([1.0], ["A"]))
    assert retriever.choose_route(sq) != "arxiv_meta"


def test_the_prompt_sees_what_was_already_tried():
    provider = FakeProvider()
    retriever = build(provider=provider)
    sq = new_sub_question("q", sub_question_id="sq1")
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["the original phrasing"]
    retriever.decide_next(sq, outcome([1.0], ["A"]))
    assert "the original phrasing" in provider.calls[0]["system"]


def test_the_prompt_sees_what_came_back():
    provider = FakeProvider()
    retriever = build(provider=provider)
    sq = new_sub_question("q", sub_question_id="sq1")
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["q"]
    retriever.decide_next(sq, outcome([1.0], ["A"]))
    assert "Results" in provider.calls[0]["system"]


def test_give_up_from_the_model_is_honoured():
    provider = FakeProvider([{"action": "give_up", "reasoning": "no source has this"}])
    sq = new_sub_question("q", sub_question_id="sq1")
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["q"]
    assert build(provider=provider).decide_next(sq, outcome([1.0], ["A"])) == ABANDONED


# --- cost ------------------------------------------------------------------


def test_a_first_round_success_costs_no_llm_call():
    """Round one has no history to condition on, so the common case is
    free."""
    provider = FakeProvider()
    sq = new_sub_question("q", sub_question_id="sq1")
    build(provider=provider).decide_next(sq, outcome([9.0, 8.0], ["A", "B"]))
    assert provider.calls == []


def test_escalation_costs_exactly_one_call():
    provider = FakeProvider()
    sq = new_sub_question("q", sub_question_id="sq1")
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["q"]
    build(provider=provider).decide_next(sq, outcome([1.0], ["A"]))
    assert len(provider.calls) == 1


def test_budget_is_charged_for_reformulation():
    from src.budget import Budget

    budget = Budget(llm_calls_max=25, retrieval_calls_max=20,
                    wall_clock_max_s=900)
    retriever = build(budget=budget)
    sq = new_sub_question("q", sub_question_id="sq1")
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["q"]
    retriever.decide_next(sq, outcome([1.0], ["A"]))
    assert budget.llm_calls_used == 1


def test_no_llm_budget_falls_back_to_route_escalation():
    from src.budget import Budget

    budget = Budget(llm_calls_max=1, retrieval_calls_max=20,
                    wall_clock_max_s=900)
    budget.spend("llm", 1)
    provider = FakeProvider()
    retriever = build(provider=provider, budget=budget)
    sq = new_sub_question("q", sub_question_id="sq1")
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["q"]
    assert retriever.decide_next(sq, outcome([1.0], ["A"])) == IN_PROGRESS
    assert provider.calls == []
    assert retriever.choose_route(sq) == "web"


def test_reformulation_failure_degrades_to_escalation():
    provider = FakeProvider(error=RuntimeError("provider down"))
    sq = new_sub_question("q", sub_question_id="sq1")
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["q"]
    assert build(provider=provider).decide_next(sq, outcome([1.0], ["A"])) == IN_PROGRESS


# --- the ablation switches -------------------------------------------------


def disabled(**flags):
    cfg = Config({
        "agency": {
            "d2_adaptive_route_and_query": flags.get("d2", True),
            "d3_adaptive_depth_and_stopping": flags.get("d3", True),
            "static": {"route_rule": "arxiv_meta", "fixed_top_k": 3},
        },
        "budget": {"max_rounds_per_subquestion": 3},
        "retrieval": {
            "fusion": {"candidates_per_retriever": 20},
            "adaptive": {"sufficient_min_rerank": 4.0,
                         "min_independent_sources": 2,
                         "hopeless_max_rerank": 0.0,
                         "reformulate_with_llm": True},
        },
    })
    return build(cfg)


def test_a3_restores_always_stop():
    """With D3 frozen the retriever must behave exactly like C17."""
    sq = new_sub_question("q", sub_question_id="sq1")
    assert disabled(d3=False).decide_next(sq, outcome([1.0], ["A"])) == SUFFICIENT


def test_a2_restores_the_fixed_route_and_verbatim_query():
    retriever = disabled(d2=False)
    sq = new_sub_question("what is RRF", sub_question_id="sq1")
    sq["suggested_route"] = "arxiv_fulltext"
    sq["attempted_routes"] = ["arxiv_meta"]
    assert retriever.choose_route(sq) == "arxiv_meta"      # config, not planner
    assert retriever.formulate_query(sq) == "what is RRF"


def test_decisions_are_traced_as_adaptive_when_live():
    from src.tracing.tracer import DecisionSurface

    events = []

    class FakeTracer:
        def decision(self, surface, **kw):
            events.append((surface, kw))

        def retrieval(self, **kw):
            pass

        def note(self, *a, **k):
            pass

    retriever = build(tracer=FakeTracer())
    sq = new_sub_question("q", sub_question_id="sq1")
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["q"]
    retriever.decide_next(sq, outcome([1.0], ["A"]))

    surfaces = {s for s, _ in events}
    assert DecisionSurface.D3_DEPTH_AND_STOPPING in surfaces
    assert DecisionSurface.D2_ROUTE_AND_QUERY in surfaces
    assert all(kw["was_adaptive"] for _, kw in events)


def test_frozen_decisions_are_traced_as_non_adaptive():
    """C35 counts adaptive decisions; an A3 run must report zero for D3
    while still recording that the decision point existed."""
    from src.tracing.tracer import DecisionSurface

    events = []

    class FakeTracer:
        def decision(self, surface, **kw):
            events.append((surface, kw))

        def retrieval(self, **kw):
            pass

        def note(self, *a, **k):
            pass

    cfg = Config({
        "agency": {"d2_adaptive_route_and_query": True,
                   "d3_adaptive_depth_and_stopping": False,
                   "static": {"route_rule": "arxiv_meta", "fixed_top_k": 3}},
        "budget": {"max_rounds_per_subquestion": 3},
        "retrieval": {"fusion": {"candidates_per_retriever": 20},
                      "adaptive": {"sufficient_min_rerank": 4.0,
                                   "min_independent_sources": 2,
                                   "hopeless_max_rerank": 0.0,
                                   "reformulate_with_llm": True}},
    })
    retriever = build(cfg, tracer=FakeTracer())
    state = state_with("q")
    retriever.retrieve_one(state, "sq1")
    d3 = [kw for s, kw in events if s is DecisionSurface.D3_DEPTH_AND_STOPPING]
    assert d3 and all(kw["was_adaptive"] is False for kw in d3)


# --- end to end ------------------------------------------------------------


def test_a_hard_sub_question_escalates_route_and_rewrites_the_query():
    """The C21 acceptance criterion."""
    retriever_impl = FakeRetriever(
        by_query={
            "obscure phrasing": [],                       # finds nothing
            "better phrasing": [passage("P1", 9.0, "A"),
                                passage("P2", 7.0, "B")],  # succeeds
        },
        default=[],
    )
    provider = FakeProvider([{
        "action": "wider", "route": "web", "query": "better phrasing",
        "reasoning": "original vocabulary did not match any source",
    }])
    retriever = build(retriever=retriever_impl, provider=provider)
    state = state_with("obscure phrasing")

    update = retriever.retrieve_one(state, "sq1")
    state["plan"] = update["plan"]
    # in_progress, NOT insufficient: the latter is terminal, and returning
    # it to mean "try again" stopped the loop before any second round.
    assert state["plan"][0]["status"] == IN_PROGRESS

    update = retriever.retrieve_one(state, "sq1")
    sq = update["plan"][0]
    assert sq["status"] == SUFFICIENT
    assert sq["attempted_routes"] == ["arxiv_meta", "web"]       # escalated
    assert sq["attempted_queries"] == ["obscure phrasing", "better phrasing"]
    assert sq["rounds"] == 2


def test_f8_and_f9_become_variable():
    """They are constants under the static policy by construction, so they
    are the direct measurement of whether D2 and D3 do anything."""
    retriever_impl = FakeRetriever(
        by_query={"easy": [passage("P1", 9.0, "A"), passage("P2", 8.0, "B")],
                  "hard": [],
                  "rewritten": [passage("P3", 9.0, "C"), passage("P4", 8.0, "D")]},
        default=[],
    )
    retriever = build(retriever=retriever_impl)
    state = state_with("easy", "hard")

    for _ in range(4):
        pending = [sq for sq in state["plan"]
                   if sq["status"] in ("pending", IN_PROGRESS)]
        if not pending:
            break
        state["plan"] = retriever.retrieve_one(state, pending[0]["id"])["plan"]

    rounds = {sq["rounds"] for sq in state["plan"]}
    routes = {tuple(sq["attempted_routes"]) for sq in state["plan"]}
    assert len(rounds) > 1, "f8 did not vary"
    assert len(routes) > 1, "f9 did not vary"


def test_outcome_features_shape():
    f = outcome_features(outcome([5.0, 3.0], ["A", "A"]))
    assert f == {"f1": 5.0, "sources": 1, "n_results": 2}


# --- the status machine (regression: C21 live run) -------------------------


def test_in_progress_is_not_terminal():
    """`insufficient` is in TERMINAL_STATUSES, so using it to mean 'try
    again' made retrieve_all stop after one round — the adaptive retriever
    behaved exactly like the static one and the smoke test measured a zero
    delta across every metric."""
    from src.state import TERMINAL_STATUSES

    assert IN_PROGRESS not in TERMINAL_STATUSES
    assert INSUFFICIENT in TERMINAL_STATUSES


def test_retrieve_all_actually_runs_a_second_round():
    retriever_impl = FakeRetriever(
        by_query={"hard": [], "rewritten": [passage("P1", 9.0, "A"),
                                            passage("P2", 8.0, "B")]},
        default=[],
    )
    provider = FakeProvider([{
        "action": "wider", "route": "web", "query": "rewritten", "reasoning": "r",
    }])
    final = build(retriever=retriever_impl,
                  provider=provider).retrieve_all(state_with("hard"))
    assert final["plan"][0]["rounds"] == 2
    assert final["plan"][0]["status"] == SUFFICIENT


# --- routes must change what is retrieved ----------------------------------


def test_each_route_maps_to_different_filters():
    """Until this was added the route was recorded and then ignored:
    `_execute` searched the whole corpus regardless, so escalating from
    arxiv_meta to arxiv_fulltext returned byte-identical passages. The trace
    showed an escalation that changed nothing."""
    r = build()
    assert r.route_filters("arxiv_meta") == {
        "source_type": "arxiv", "section": "Abstract"}
    assert r.route_filters("arxiv_fulltext") == {
        "source_type": "arxiv", "exclude_sections": ["Abstract"]}
    assert r.route_filters("web") == {"source_type": "web"}
    assert r.route_filters("evidence_store") == {}


def test_filters_reach_the_retriever():
    class RecordingRetriever(FakeRetriever):
        def __init__(self):
            super().__init__()
            self.kwargs = []

        def retrieve(self, query, *, top_k, **kw):
            self.kwargs.append(kw)
            return super().retrieve(query, top_k=top_k)

    impl = RecordingRetriever()
    state = state_with("q", route="arxiv_meta")
    build(retriever=impl).retrieve_one(state, "sq1")
    assert impl.kwargs[0]["source_type"] == "arxiv"
    assert impl.kwargs[0]["section"] == "Abstract"

    impl2 = RecordingRetriever()
    state2 = state_with("q", route="web")
    build(retriever=impl2).retrieve_one(state2, "sq1")
    assert impl2.kwargs[0]["source_type"] == "web"


def test_an_empty_route_slice_falls_back_unfiltered():
    """A route whose slice is empty is a fact about the corpus, not the
    query — a thin corpus must not look like a failed query."""
    calls = []

    class EmptyThenFull(FakeRetriever):
        def retrieve(self, query, *, top_k, **kw):
            calls.append(kw)
            if kw:
                return []
            return [FakeResult(passage("P1", 9.0, "A")),
                    FakeResult(passage("P2", 8.0, "B"))]

    state = state_with("q")
    update = build(retriever=EmptyThenFull()).retrieve_one(state, "sq1")
    assert len(update["evidence"]) == 2
    assert calls[0] and not calls[1]


# --- depth must be earned (regression: C21 live run) -----------------------


def test_the_first_attempt_never_starts_at_full_text():
    """3.3: the reader is 'invoked only when the retriever decides depth is
    needed (D3), which makes depth an EARNED cost rather than a default'.
    A planner suggesting arxiv_fulltext up front bypasses that.

    Measured: the planner suggested fulltext for all three sub-questions and
    independent sources fell 12 -> 2 against the static baseline, because
    full-text chunks concentrate in the few ingested papers while abstracts
    span every paper the search returns."""
    sq = new_sub_question("q", sub_question_id="sq1")
    sq["suggested_route"] = "arxiv_fulltext"
    assert build().choose_route(sq) == "arxiv_meta"


def test_cheap_suggestions_are_honoured():
    for route in ("arxiv_meta", "web", "evidence_store"):
        sq = new_sub_question("q", sub_question_id="sq1")
        sq["suggested_route"] = route
        assert build().choose_route(sq) == route


def test_full_text_is_still_reachable_by_escalation():
    """Clamping the first attempt must not make depth unreachable — it must
    make it earned."""
    provider = FakeProvider([{
        "action": "deeper", "route": "arxiv_fulltext",
        "query": "a more specific query", "reasoning": "abstract lacked it",
    }])
    retriever = build(provider=provider)
    sq = new_sub_question("q", sub_question_id="sq1")
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["q"]
    retriever.decide_next(sq, outcome([1.0], ["A"]))
    assert retriever.choose_route(sq) == "arxiv_fulltext"


# --- marginal improvement (regression: C21 live run) -----------------------


def test_a_round_that_does_not_help_stops_the_search():
    """A live run spent 9 retrieval calls, 6 LLM calls and 71 seconds to
    reach exactly the f1 the static retriever reached in 3 calls and 8
    seconds — no round after the first improved anything and nothing
    noticed. A fixed number of rounds with a threshold attached is a
    schedule, not a decision."""
    retriever = build()
    sq = new_sub_question("q", sub_question_id="sq1")

    assert retriever.decide_next(sq, outcome([2.5, 1.0], ["A", "B"])) == IN_PROGRESS
    sq["rounds"] = 1
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["q"]

    # Round two returns essentially the same quality.
    assert retriever.decide_next(sq, outcome([2.6, 1.0], ["A", "B"])) == INSUFFICIENT


def test_a_round_that_helps_keeps_going():
    retriever = build()
    sq = new_sub_question("q", sub_question_id="sq1")
    retriever.decide_next(sq, outcome([1.0], ["A"]))
    sq["rounds"] = 1
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["q"]
    # A clear improvement, still short of the bar: keep trying.
    assert retriever.decide_next(sq, outcome([3.0], ["A"])) == IN_PROGRESS


def test_improvement_past_the_bar_stops_as_sufficient():
    retriever = build()
    sq = new_sub_question("q", sub_question_id="sq1")
    retriever.decide_next(sq, outcome([1.0], ["A"]))
    sq["rounds"] = 1
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["q"]
    assert retriever.decide_next(sq, outcome([8.0, 7.0], ["A", "B"])) == SUFFICIENT


def test_the_number_of_rounds_depends_on_the_evidence():
    """The property that separates a decision from a schedule."""
    easy = FakeRetriever(default=[passage("P1", 9.0, "A"),
                                  passage("P2", 8.0, "B")])
    stuck = FakeRetriever(default=[passage("P1", 2.0, "A"),
                                   passage("P2", 1.9, "B")])
    easy_final = build(retriever=easy).retrieve_all(state_with("q"))
    stuck_final = build(retriever=stuck).retrieve_all(state_with("q"))

    assert easy_final["plan"][0]["rounds"] == 1
    assert stuck_final["plan"][0]["rounds"] == 2      # stops once it plateaus
    assert stuck_final["plan"][0]["rounds"] < 3


# --- convergence overrides the hopeless rule (regression: C22 live run) ----


def test_a_converging_sub_question_is_not_abandoned():
    """A live run abandoned a sub-question that had gone from f1 -4.21 to
    -0.29 — an improvement of 3.9, eight times min_improvement — on the
    strength of still being below zero. At that rate the next round would
    have cleared the bar. The absolute level says where the evidence IS; the
    improvement says where it is GOING."""
    retriever = build()
    sq = new_sub_question("q", sub_question_id="sq1")
    retriever.decide_next(sq, outcome([-4.21], ["A"]))
    sq["rounds"] = 1
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["q"]
    assert retriever.decide_next(sq, outcome([-0.29], ["A"])) == IN_PROGRESS


def test_a_stagnant_low_sub_question_is_still_abandoned():
    retriever = build()
    sq = new_sub_question("q", sub_question_id="sq1")
    retriever.decide_next(sq, outcome([-1.9], ["A"]))
    sq["rounds"] = 1
    sq["attempted_routes"] = ["arxiv_meta"]
    sq["attempted_queries"] = ["q"]
    assert retriever.decide_next(sq, outcome([-1.8], ["A"])) == ABANDONED