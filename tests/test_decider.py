"""Tests for the decider node (D5).

Following the lesson recorded at C21: these assert that a decision has an
EFFECT — the draft withheld, the run sent back for a partial rewrite, the
controller's next move — not merely that a decision was recorded.
"""
import pytest

from src.budget import Budget
from src.config import Config
from src.controller import (
    CRITIC,
    CRITIC_ROUTER,
    DECIDER,
    FINALIZER,
    FINISH,
    PLANNER,
    RETRIEVER,
    SYNTHESIZER,
    Controller,
)
from src.nodes.decider import (
    REWRITE_NOT_NEEDED,
    REWRITE_REQUESTED,
    REWRITE_UNAFFORDABLE,
    Decider,
    decider_node,
)
from src.state import merge_gaps
from src.sufficiency.policy import (
    ABSTAIN,
    ANSWER,
    PARTIAL,
    RULE_PREMISE,
    RULE_STATIC,
    RULE_SYSTEM_FAILURE,
    Thresholds,
)
from src.sufficiency.scorer import CalibratedScorer, HeuristicScorer
from tests.decision_fixtures import STRONG, WEAK, decision_state, fitted_models

CFG = Config({
    "agency": {"d5_terminal_decision": True, "d1_plan_revision": True,
               "skip_research_on_failed_premise": True},
    "sufficiency": {"hard_rules": {"abstain_on_failed_premise": True},
                    "partial_rewrite": {"enabled": True, "llm_calls": 3},
                    "heuristic": {"min_entailment": 0.5}},
    "budget": {"max_rounds_per_subquestion": 3},
})
STATIC = CFG.with_overrides({"agency.d5_terminal_decision": False})


def decider(cfg=CFG, scorer=None, **kw):
    return Decider(cfg, scorer=scorer or HeuristicScorer(cfg), **kw)


def budget(llm_max=25, used=0):
    b = Budget(llm_calls_max=llm_max, retrieval_calls_max=20, wall_clock_max_s=900)
    b.llm_calls_used = used
    return b


class FakeTracer:
    def __init__(self):
        self.decisions = []

    def decision(self, surface, **kw):
        self.decisions.append((surface, kw))


# --- the three outcomes and their effects ----------------------------------


def test_answer_keeps_the_draft():
    result = decider().decide(decision_state(STRONG, STRONG))
    assert result.decision.terminal == ANSWER
    assert "draft" not in result.update


def test_abstain_withholds_the_draft_but_keeps_it_for_analysis():
    state = decision_state(WEAK, WEAK, draft="A confident wrong report.")
    result = decider().decide(state)
    assert result.decision.terminal == ABSTAIN
    assert result.update["draft"] == ""
    assert result.update["decision"]["withheld_draft"] == "A confident wrong report."
    assert result.update["decision"]["withheld_draft_mode"] == "answer"


def test_partial_sends_an_answer_mode_draft_back_for_rewriting():
    state = decision_state(STRONG, WEAK)
    result = decider(budget=budget()).decide(state)
    assert result.decision.terminal == PARTIAL
    assert result.update["draft"] == ""
    assert result.update["decision"]["partial_rewrite"] == REWRITE_REQUESTED

    # The effect that matters: the controller now routes to the synthesizer,
    # which writes in partial mode because terminal_decision is PARTIAL.
    state.update(result.update)
    full = Controller(CFG, available={PLANNER, RETRIEVER, SYNTHESIZER, CRITIC,
                                      CRITIC_ROUTER, DECIDER, FINALIZER})
    assert full.route(state) == SYNTHESIZER


def test_partial_keeps_the_draft_when_the_rewrite_is_unaffordable():
    result = decider(budget=budget(llm_max=25, used=23)).decide(
        decision_state(STRONG, WEAK))
    assert result.decision.terminal == PARTIAL
    assert "draft" not in result.update
    assert result.update["decision"]["partial_rewrite"] == REWRITE_UNAFFORDABLE


def test_partial_does_not_rewrite_a_partial_draft():
    result = decider().decide(decision_state(STRONG, WEAK, draft_mode="partial"))
    assert "draft" not in result.update
    assert result.update["decision"]["partial_rewrite"] == REWRITE_NOT_NEEDED


# --- gaps ------------------------------------------------------------------


def test_unresolved_sub_questions_become_gaps_alongside_recorded_ones():
    state = decision_state(STRONG, WEAK)
    state["unresolved_gaps"] = ["could not establish: a specific claim"]
    result = decider().decide(state)
    gaps = result.update["decision"]["gaps"]
    assert gaps[0] == "could not establish: a specific claim"
    assert "could not establish: sub-question 2" in gaps
    assert "could not establish: sub-question 1" not in gaps


def test_gaps_reducer_appends_rather_than_replaces():
    """The planner returned only its own gaps and erased the router's."""
    assert merge_gaps(["router gap"], ["planner gap"]) == ["router gap", "planner gap"]
    assert merge_gaps(["a", "b"], ["a", "b", "c"]) == ["a", "b", "c"]


def test_gaps_survive_a_planner_revision_through_langgraph():
    from langgraph.graph import END, START, StateGraph

    from src.state import ResearchState, new_state

    builder = StateGraph(ResearchState)
    builder.add_node("router", lambda s: {"unresolved_gaps": ["from the router"]})
    builder.add_node("planner", lambda s: {"unresolved_gaps": ["from the planner"]})
    builder.add_edge(START, "router")
    builder.add_edge("router", "planner")
    builder.add_edge("planner", END)
    final = builder.compile().invoke(new_state("q"))
    assert final["unresolved_gaps"] == ["from the router", "from the planner"]


# --- hard rules and failure ------------------------------------------------


def test_refuted_premise_abstains_with_the_correction():
    state = decision_state(STRONG, STRONG, premise={
        "ran": True, "failed": True, "grounding_score": 0.0,
        "rationale": "false premise: Mamba-2 uses RoPE",
        "correction": "Mamba-2 uses no positional encoding"})
    result = decider().decide(state)
    assert result.decision.rule == RULE_PREMISE
    assert "no positional encoding" in result.update["decision_rationale"]


def test_a_crash_before_drafting_is_recorded_as_system_failure():
    state = decision_state(STRONG, STRONG, draft="")
    state["error"] = "synthesizer returned no text"
    record = decider().decide(state).update["decision"]
    assert record["rule"] == RULE_SYSTEM_FAILURE
    assert record["epistemic"] is False


# --- the record C33 and C37 read -------------------------------------------


def test_record_carries_the_decision_time_snapshot():
    """C33 labels stage 1 from these. On a PARTIAL run the rewrite makes no
    claims about the unresolved sub-questions, so reading verdicts from the
    final state would drop exactly the negatives."""
    state = decision_state(STRONG, STRONG,
                           verdicts={"sq1": "SUPPORTED", "sq2": "UNSUPPORTED"})
    record = decider(budget=budget()).decide(state).update["decision"]
    subs = {s["id"]: s for s in record["sub_questions"]}
    assert subs["sq2"]["verdicts"] == ["UNSUPPORTED"]
    assert subs["sq1"]["features"]["f1_max_rerank"] == pytest.approx(STRONG)
    assert record["score"] is not None and record["thresholds"]


def test_calibrated_probabilities_are_written_to_the_plan():
    stage1, stage2 = fitted_models()
    scorer = CalibratedScorer(stage1, stage2, Thresholds(0.65, 0.40))
    update = decider(scorer=scorer).decide(decision_state(STRONG, WEAK)).update
    probabilities = [sq["sufficiency"] for sq in update["plan"]]
    assert probabilities[0] > probabilities[1]
    assert update["decision"]["calibrated"] is True


def test_heuristic_never_writes_into_the_probability_field():
    update = decider().decide(decision_state(STRONG, WEAK)).update
    assert "plan" not in update
    assert update["decision"]["calibrated"] is False


# --- the static policy (B2) and the trace ----------------------------------


def test_static_policy_answers_keeps_the_draft_and_records_what_it_overrode():
    result = decider(STATIC).decide(decision_state(WEAK, WEAK))
    assert result.decision.terminal == ANSWER
    assert result.decision.rule == RULE_STATIC
    assert "draft" not in result.update
    assert result.update["decision"]["counterfactual"]["terminal"] == ABSTAIN


def test_decisions_are_traced_as_adaptive_when_live():
    from src.tracing.tracer import DecisionSurface

    tracer = FakeTracer()
    decider(tracer=tracer).decide(decision_state(STRONG, WEAK))
    surface, kw = tracer.decisions[0]
    assert surface == DecisionSurface.D5_TERMINAL_DECISION
    assert kw["was_adaptive"] is True
    assert kw["chosen"] == PARTIAL
    assert kw["alternatives"] == [ANSWER, PARTIAL, ABSTAIN]
    assert kw["inputs"]["calibrated"] is False


def test_b2_trace_shows_zero_adaptive_d5_decisions(tmp_path):
    """C35 validates the B2 baseline from the trace alone."""
    from src.tracing.tracer import TraceReader, Tracer

    tracer = Tracer(tmp_path, trace_id="b2", question="q")
    decider(STATIC, tracer=tracer).decide(decision_state(WEAK))
    tracer.close()
    counts = TraceReader.load(tracer.path).adaptive_decision_counts()
    assert counts.get("D5_terminal_decision", 0) == 0


# --- whole-graph flows with fake upstream nodes ----------------------------


def _graph_run(decision_strengths, draft_texts):
    """premise -> plan -> retrieve -> synth -> critic -> router -> decider.

    The synthesizer records each mode it was asked for, which is what shows
    the PARTIAL rewrite actually happened rather than being recorded.
    """
    from src.graph import Components, build_graph
    from src.state import draft_digest, new_state

    seen = {"modes": [], "critiqued": [], "visits": []}
    base = decision_state(*decision_strengths, draft="")

    def premise(s):
        seen["visits"].append("premise")
        return {"premise_check": base["premise_check"]}

    def planner(s):
        seen["visits"].append("planner")
        return {"plan": [{**sq, "status": "pending", "rounds": 0}
                         for sq in base["plan"]]}

    def retriever(s):
        # One call settles every sub-question to the fixture's final status.
        seen["visits"].append("retriever")
        return {"plan": base["plan"], "evidence": base["evidence"]}

    def synthesizer(s):
        mode = "partial" if s.get("terminal_decision") == PARTIAL else "answer"
        seen["modes"].append(mode)
        seen["visits"].append(f"synth:{mode}")
        return {"draft": draft_texts[mode], "draft_mode": mode}

    def critic(s):
        seen["visits"].append("critic")
        seen["critiqued"].append(s["draft"])
        return {"critic_findings": [{"claim": "c", "sub_question_id": "sq1",
                                     "cited_passage_ids": [], "verdict": "SUPPORTED",
                                     "reasoning": "", "action_taken": ""}],
                "critiqued_draft": draft_digest(s["draft"])}

    def router(s):
        seen["visits"].append("router")
        return {"critic_findings": [{**f, "action_taken": "ACCEPT"}
                                    for f in s["critic_findings"]]}

    # D1 off: these tests are about what happens after research, and the
    # fake planner has no revision mode.
    cfg = CFG.with_overrides({"agency.d1_plan_revision": False})
    d = decider(cfg, budget=None)
    graph = build_graph(Components(
        premise_checker=premise, planner=planner, retriever=retriever,
        synthesizer=synthesizer, critic=critic, critic_router=router,
        decider=decider_node(d),
    ), config=cfg)
    return graph.invoke(new_state("q"), {"recursion_limit": 30}), seen


def test_partial_run_is_rewritten_in_partial_mode_and_reverified():
    final, seen = _graph_run((STRONG, WEAK), {"answer": "ANSWER DRAFT",
                                              "partial": "PARTIAL DRAFT"})
    assert final["terminal_decision"] == PARTIAL
    assert seen["modes"] == ["answer", "partial"]
    assert seen["critiqued"] == ["ANSWER DRAFT", "PARTIAL DRAFT"]
    assert final["draft"] == "PARTIAL DRAFT"
    assert final["decision"]["withheld_draft"] == "ANSWER DRAFT"
    # Research never reopened, and the decider ran once.
    assert seen["visits"].count("retriever") == 1


def test_answer_run_is_not_rewritten():
    final, seen = _graph_run((STRONG, STRONG), {"answer": "ANSWER DRAFT",
                                                "partial": "unused"})
    assert final["terminal_decision"] == ANSWER
    assert seen["modes"] == ["answer"]
    assert final["draft"] == "ANSWER DRAFT"


def test_abstain_run_ends_with_no_draft_and_no_second_synthesis():
    final, seen = _graph_run((WEAK, WEAK), {"answer": "ANSWER DRAFT",
                                            "partial": "unused"})
    assert final["terminal_decision"] == ABSTAIN
    assert final["draft"] == ""
    assert seen["modes"] == ["answer"]
    assert final["decision"]["withheld_draft"] == "ANSWER DRAFT"


# --- the router after a decision -------------------------------------------


def test_router_offers_no_costly_route_once_decided():
    """Re-retrieving after the terminal decision would reopen research the
    decision closed, and clearing the draft would loop."""
    from src.nodes.critic_router import RE_RETRIEVE, REVISE_PLAN, CriticRouter

    router = CriticRouter(CFG, provider=object())
    state = decision_state(STRONG)
    assert RE_RETRIEVE in router._available_actions(state)
    state["terminal_decision"] = PARTIAL
    available = router._available_actions(state)
    assert RE_RETRIEVE not in available and REVISE_PLAN not in available
