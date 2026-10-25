"""Tests for the controller.

The controller makes no LLM calls, so its policy can be tested exhaustively
without an API key — which is the point of keeping dispatch deterministic and
decisions agentic. A routing bug that only appears under sampling noise would
be close to undiagnosable.
"""
import pytest

from src.budget import Budget
from src.config import Config
from src.controller import (
    CRITIC,
    DECIDER,
    FINALIZER,
    FINISH,
    PLANNER,
    PREMISE_CHECKER,
    RETRIEVER,
    SUFFICIENCY_SCORER,
    SYNTHESIZER,
    Controller,
)
from src.state import new_state, new_sub_question

CFG = Config({
    "agency": {"d1_plan_revision": True, "skip_research_on_failed_premise": True},
    "budget": {"max_rounds_per_subquestion": 3},
})

C19_NODES = {PREMISE_CHECKER, PLANNER, RETRIEVER, SYNTHESIZER}


def controller(available=None, budget=None, cfg=CFG):
    return Controller(cfg, available=available or set(C19_NODES), budget=budget)


def state(**over):
    s = new_state("How do X and Y differ?")
    s.update(over)
    return s


def planned(n=2, status="pending", evidence=True):
    s = state(premise_check={"ran": True, "failed": False})
    s["plan"] = [
        new_sub_question(f"sub {i}", sub_question_id=f"sq{i + 1}")
        for i in range(n)
    ]
    for sq in s["plan"]:
        sq["status"] = status
    if evidence:
        s["evidence"] = [{"id": "P1", "text": "retrieved evidence"}]
    return s


# --- the happy path, in order ----------------------------------------------


def test_fresh_state_routes_to_premise_checker():
    assert controller().route(state()) == PREMISE_CHECKER


def test_after_premise_check_routes_to_planner():
    s = state(premise_check={"ran": True, "failed": False})
    assert controller().route(s) == PLANNER


def test_with_a_plan_routes_to_retriever():
    assert controller().route(planned()) == RETRIEVER


def test_retriever_repeats_while_sub_questions_are_pending():
    s = planned(3)
    s["plan"][0]["status"] = "sufficient"
    assert controller().route(s) == RETRIEVER


def test_all_terminal_routes_to_synthesizer():
    assert controller().route(planned(2, status="sufficient")) == SYNTHESIZER


def test_draft_written_routes_to_finish_when_nothing_follows():
    s = planned(2, status="sufficient")
    s["draft"] = "a report"
    assert controller().route(s) == FINISH


# --- premise failure -------------------------------------------------------


def test_refuted_premise_skips_research_entirely():
    """The corpus refutes the question's premise and the checker already
    retrieved the correction. Researching anyway spends the whole budget
    elaborating something known to be false, and the decider abstains
    regardless (§5.4)."""
    s = state(premise_check={"ran": True, "failed": True,
                             "correction": "actually Z"})
    assert controller().route(s) == FINISH


def test_refuted_premise_routes_to_decider_when_available():
    s = state(premise_check={"ran": True, "failed": True})
    assert controller({PREMISE_CHECKER, PLANNER, RETRIEVER, SYNTHESIZER,
                       DECIDER}).route(s) == DECIDER


def test_refuted_premise_does_not_synthesise():
    """There is nothing to write. C18's node also declines, but routing
    around it is clearer than relying on a node to no-op."""
    s = state(premise_check={"ran": True, "failed": True})
    assert controller().route(s) != SYNTHESIZER


def test_research_proceeds_when_the_flag_is_off():
    cfg = Config({
        "agency": {"d1_plan_revision": True,
                   "skip_research_on_failed_premise": False},
        "budget": {"max_rounds_per_subquestion": 3},
    })
    s = state(premise_check={"ran": True, "failed": True})
    assert controller(cfg=cfg).route(s) == PLANNER


def test_unverifiable_premise_does_not_skip_research():
    """UNVERIFIABLE is silence, not refutation (C15). Skipping research here
    would abstain on every question the corpus does not restate."""
    s = state(premise_check={"ran": True, "failed": False,
                             "grounding_score": 0.5})
    assert controller().route(s) == PLANNER


# --- budget ----------------------------------------------------------------


def test_exhausted_retrieval_budget_stops_research():
    budget = Budget(llm_calls_max=25, retrieval_calls_max=2,
                    wall_clock_max_s=900)
    budget.spend("retrieval", 2)
    assert controller(budget=budget).route(planned(3)) == SYNTHESIZER


def test_exhausted_llm_budget_stops_research():
    budget = Budget(llm_calls_max=1, retrieval_calls_max=20,
                    wall_clock_max_s=900)
    budget.spend("llm", 1)
    assert controller(budget=budget).route(planned(3)) == SYNTHESIZER


def test_wall_clock_stops_research():
    clock = {"t": 1000.0}
    budget = Budget(llm_calls_max=25, retrieval_calls_max=20,
                    wall_clock_max_s=10.0, _started=1000.0,
                    _now=lambda: clock["t"])
    clock["t"] = 1100.0
    assert controller(budget=budget).route(planned(3)) == SYNTHESIZER


def test_no_budget_object_means_no_budget_limit():
    assert controller().route(planned(3)) == RETRIEVER


# --- D1 revision -----------------------------------------------------------


def test_insufficient_sub_question_triggers_revision():
    s = planned(2, status="sufficient")
    s["plan"][0]["status"] = "insufficient"
    s["plan"][0]["rounds"] = 1
    assert controller().route(s) == PLANNER


def test_revision_respects_the_ceiling():
    """Without it, an unsatisfiable sub-question sends the controller back to
    the planner forever — §2.4's anti-pattern with the counter removed."""
    budget = Budget(llm_calls_max=25, retrieval_calls_max=20,
                    wall_clock_max_s=900, max_plan_revisions=1)
    budget.record_plan_revision()
    s = planned(2, status="sufficient")
    s["plan"][0]["status"] = "insufficient"
    s["plan"][0]["rounds"] = 1
    assert controller(budget=budget).route(s) == SYNTHESIZER


def test_revision_skipped_when_d1_is_disabled():
    """The A1 ablation freezes plan revision."""
    cfg = Config({
        "agency": {"d1_plan_revision": False,
                   "skip_research_on_failed_premise": True},
        "budget": {"max_rounds_per_subquestion": 3},
    })
    s = planned(2, status="sufficient")
    s["plan"][0]["status"] = "insufficient"
    s["plan"][0]["rounds"] = 1
    assert controller(cfg=cfg).route(s) == SYNTHESIZER


def test_no_revision_before_a_round_has_run():
    s = planned(2, status="sufficient")
    s["plan"][0]["status"] = "insufficient"
    s["plan"][0]["rounds"] = 0
    assert controller().route(s) == SYNTHESIZER


# --- node availability -----------------------------------------------------


def test_missing_premise_checker_is_skipped():
    assert controller({PLANNER, RETRIEVER, SYNTHESIZER}).route(state()) == PLANNER


def test_full_chain_routes_through_every_stage():
    """Registering a node must change routing without changing routing code."""
    full = Controller(CFG, available=set(
        [PREMISE_CHECKER, PLANNER, RETRIEVER, SUFFICIENCY_SCORER,
         SYNTHESIZER, CRITIC, DECIDER, FINALIZER]
    ))
    s = planned(1, status="sufficient")
    assert full.route(s) == SUFFICIENCY_SCORER

    s["plan"][0]["sufficiency"] = 0.8
    assert full.route(s) == SYNTHESIZER

    s["draft"] = "a report"
    assert full.route(s) == CRITIC

    s["critic_findings"] = [{"claim": "x", "verdict": "SUPPORTED"}]
    assert full.route(s) == DECIDER

    s["terminal_decision"] = "ANSWER"
    assert full.route(s) == FINALIZER

    s["final_report"] = "# Report"
    assert full.route(s) == FINISH


def test_sufficiency_scorer_waits_for_every_sub_question():
    full = Controller(CFG, available={PLANNER, RETRIEVER, SUFFICIENCY_SCORER,
                                      SYNTHESIZER})
    s = planned(2, status="sufficient")
    s["plan"][0]["sufficiency"] = 0.8
    assert full.route(s) == SUFFICIENCY_SCORER


# --- errors and termination ------------------------------------------------


def test_error_state_stops_research():
    s = planned(3)
    s["error"] = "provider down"
    assert controller().route(s) != RETRIEVER


def test_empty_plan_with_planner_absent_does_not_loop():
    assert controller({RETRIEVER, SYNTHESIZER}).route(state()) == FINISH


def test_synthesizer_is_not_entered_without_evidence():
    """Beyond being correct, this closes a live loop: the synthesizer
    returns an empty draft when it has nothing, `not draft` stays true, and
    the controller sends it straight back."""
    s = planned(1, status="sufficient", evidence=False)
    assert controller().route(s) == FINISH


def test_synthesizer_is_entered_once_evidence_exists():
    s = planned(1, status="sufficient")
    s["evidence"] = [{"id": "P1", "text": "something"}]
    assert controller().route(s) == SYNTHESIZER


def test_abstain_decision_skips_synthesis():
    s = planned(2, status="sufficient")
    s["terminal_decision"] = "ABSTAIN"
    assert controller().route(s) != SYNTHESIZER


def test_routing_is_deterministic():
    s = planned(2)
    c = controller()
    assert len({c.route(s) for _ in range(10)}) == 1


# --- the state summary (what C38 replays) ----------------------------------


def test_summary_covers_what_decide_reads():
    budget = Budget(llm_calls_max=25, retrieval_calls_max=20,
                    wall_clock_max_s=900)
    s = planned(2)
    summary = controller(budget=budget).summary(s)
    assert set(summary) >= {
        "premise_ran", "premise_failed", "plan_size", "pending", "statuses",
        "plan_revisions", "evidence", "has_draft", "critic_run",
        "terminal_decision", "all_terminal", "budget",
    }


def test_summary_does_not_embed_passage_text():
    """§3.3 asks for the state that produced a decision; dumping full state
    is not viable when evidence runs to kilobytes and a run makes tens of
    routing decisions."""
    s = planned(1)
    s["evidence"] = [{"id": "P1", "text": "x" * 5000}]
    assert "x" * 100 not in str(controller().summary(s))
    assert controller().summary(s)["evidence"] == 1


def test_route_is_traced_with_its_reason():
    events = []

    class FakeTracer:
        def route(self, to_node, *, reason, state_summary=None, **kw):
            events.append((to_node, reason, state_summary))

    Controller(CFG, available=set(C19_NODES), tracer=FakeTracer()).route(planned())
    to_node, reason, summary = events[0]
    assert to_node == RETRIEVER
    assert "pending" in reason
    assert summary["plan_size"] == 2


# --- critic routing (C24) --------------------------------------------------


def test_findings_without_actions_route_to_the_critic_router():
    from src.controller import CRITIC_ROUTER

    full = Controller(CFG, available={PLANNER, RETRIEVER, SYNTHESIZER,
                                      CRITIC, CRITIC_ROUTER})
    s = planned(1, status="sufficient")
    s["draft"] = "a report"
    s["critic_findings"] = [{"claim": "x", "verdict": "UNSUPPORTED",
                             "action_taken": ""}]
    assert full.route(s) == CRITIC_ROUTER


def test_routed_findings_move_on():
    from src.controller import CRITIC_ROUTER

    full = Controller(CFG, available={PLANNER, RETRIEVER, SYNTHESIZER,
                                      CRITIC, CRITIC_ROUTER})
    s = planned(1, status="sufficient")
    s["draft"] = "a report"
    s["critic_findings"] = [{"claim": "x", "verdict": "UNSUPPORTED",
                             "action_taken": "UNRESOLVED"}]
    assert full.route(s) == FINISH


def test_a_cleared_draft_sends_the_run_back_through_synthesis():
    """RE_RETRIEVE clears the draft so the decider cannot see a stale report
    as though it reflected the new evidence."""
    from src.controller import CRITIC_ROUTER

    full = Controller(CFG, available={PLANNER, RETRIEVER, SYNTHESIZER,
                                      CRITIC, CRITIC_ROUTER})
    s = planned(1, status="in_progress")
    s["draft"] = ""
    s["critic_findings"] = [{"claim": "x", "verdict": "UNSUPPORTED",
                             "action_taken": "RE_RETRIEVE"}]
    assert full.route(s) == RETRIEVER