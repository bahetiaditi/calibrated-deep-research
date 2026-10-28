"""Tests for five-way critic routing.

"Five-way routing on a judgment call is the difference between a critic and
an `if` statement." The tests that matter are the ones showing all five
outcomes can follow the SAME verdict, decided by why it failed — and that an
unavailable route is never silently substituted.
"""
import pytest

from src.budget import Budget
from src.config import Config
from src.nodes.critic import CONTRADICTED, PARTIAL, SUPPORTED, UNSUPPORTED
from src.nodes.critic_router import (
    ACCEPT,
    ACTIONS,
    DROP_CLAIM,
    HEDGE,
    RE_RETRIEVE,
    REVISE_PLAN,
    UNRESOLVED,
    CriticRouter,
    critic_router_node,
)
from src.state import ClaimVerification, new_state, new_sub_question

CFG = Config({
    "agency": {"critic_five_way_routing": True, "d1_plan_revision": True},
})


def finding(claim="A claim about something.", verdict=UNSUPPORTED,
            sub_question_id="sq1", reasoning="the passage is silent"):
    return ClaimVerification(
        claim=claim, sub_question_id=sub_question_id,
        cited_passage_ids=["P000000000001"], verdict=verdict,
        reasoning=reasoning, action_taken="",
    )


class FakeProvider:
    def __init__(self, routes=None, error=None):
        self.routes = routes
        self.error = error
        self.calls = []

    def complete_json(self, system, user, **kw):
        self.calls.append({"system": system, "user": user})
        if self.error:
            raise self.error
        return {"routes": self.routes or []}, None


def router(routes=None, cfg=CFG, **kw):
    return CriticRouter(cfg, provider=FakeProvider(routes), **kw)


def state_with(*findings, plan_ids=("sq1",), queries=None):
    state = new_state("q")
    state["plan"] = []
    for sqid in plan_ids:
        sq = new_sub_question(f"sub {sqid}", sub_question_id=sqid)
        sq["attempted_queries"] = list(queries or ["original phrasing"])
        sq["rounds"] = len(sq["attempted_queries"])
        state["plan"].append(sq)
    state["critic_findings"] = list(findings)
    return state


def budget(**over):
    kwargs = dict(llm_calls_max=25, retrieval_calls_max=20,
                  wall_clock_max_s=900, max_plan_revisions=2)
    kwargs.update(over)
    return Budget(**kwargs)


# --- all five outcomes from one verdict (the C24 criterion) ---------------


@pytest.mark.parametrize("action", ACTIONS)
def test_every_route_can_fire(action):
    """One input per route. All five follow the SAME UNSUPPORTED verdict —
    which is the whole point: the verdict does not determine the action, the
    reason for failure does."""
    result = router(
        [{"index": 0, "action": action, "query": "a different query",
          "reasoning": "r"}],
        budget=budget(),
    ).route(state_with(finding()))
    assert result.findings[0]["action_taken"] == action


def test_supported_claims_need_no_action():
    result = router([]).route(state_with(finding(verdict=SUPPORTED)))
    assert result.findings[0]["action_taken"] == ACCEPT


def test_supported_claims_are_not_sent_to_the_model():
    provider = FakeProvider([])
    CriticRouter(CFG, provider=provider).route(
        state_with(finding(verdict=SUPPORTED))
    )
    assert provider.calls == []


def test_only_failed_claims_are_routed():
    provider = FakeProvider([{"index": 0, "action": HEDGE, "reasoning": "r"}])
    result = CriticRouter(CFG, provider=provider, budget=budget()).route(
        state_with(finding(verdict=SUPPORTED), finding(verdict=PARTIAL))
    )
    assert result.findings[0]["action_taken"] == ACCEPT
    assert result.findings[1]["action_taken"] == HEDGE
    assert "[0]" in provider.calls[0]["system"]      # only one claim listed


# --- the decisions change control flow ------------------------------------


def test_re_retrieve_reopens_the_sub_question():
    """Reopening is what makes this a decision rather than a note: the
    controller sees a non-terminal sub-question and routes back."""
    state = state_with(finding())
    state["plan"][0]["status"] = "sufficient"
    result = router([{"index": 0, "action": RE_RETRIEVE,
                      "query": "a genuinely different query", "reasoning": "r"}],
                    budget=budget()).route(state)
    assert result.plan[0]["status"] == "in_progress"
    assert result.needs_more_work is True


def test_revise_plan_marks_the_sub_question_for_the_planner():
    state = state_with(finding())
    state["plan"][0]["status"] = "sufficient"
    result = router([{"index": 0, "action": REVISE_PLAN, "reasoning": "vague"}],
                    budget=budget()).route(state)
    assert result.plan[0]["status"] == "insufficient"
    assert result.needs_more_work is True


def test_unresolved_reaches_d5_through_status_not_through_the_gap_list():
    """C27.5. An unverified sentence is a property of the report; a gap is a
    part of the question left unanswered. Writing the claim into the gap
    list put the report's own sentences under 'not established'."""
    result = router([{"index": 0, "action": UNRESOLVED, "reasoning": "absent"}],
                    budget=budget()).route(
        state_with(finding(claim="FlashAttention-2 reaches 99% of peak."))
    )
    assert not any("FlashAttention-2" in g for g in result.unresolved_gaps)
    assert result.plan[0]["status"] == "insufficient"
    assert result.findings[0]["action_taken"] == UNRESOLVED
    assert result.needs_more_work is False


def test_gaps_are_not_duplicated():
    state = state_with(finding())
    state["unresolved_gaps"] = ["could not establish: A claim about something"]
    result = router([{"index": 0, "action": UNRESOLVED, "reasoning": "r"}],
                    budget=budget()).route(state)
    assert len(result.unresolved_gaps) == 1


def test_drop_and_hedge_change_the_report_not_the_research():
    """Applied by the finalizer at C28; the research is already done."""
    for action in (DROP_CLAIM, HEDGE):
        result = router([{"index": 0, "action": action, "reasoning": "r"}],
                        budget=budget()).route(state_with(finding()))
        assert result.needs_more_work is False
        assert result.plan[0]["status"] == "pending"


# --- gates: an unavailable route is never silently substituted ------------


def test_no_budget_removes_the_costly_routes():
    b = budget(retrieval_calls_max=1)
    b.spend("retrieval", 1)
    available = router(budget=b)._available_actions(state_with(finding()))
    assert RE_RETRIEVE not in available and REVISE_PLAN not in available
    assert {DROP_CLAIM, HEDGE, UNRESOLVED} <= available


def test_choosing_an_unavailable_route_falls_to_unresolved_and_says_so():
    """Falling back silently would make the trace a record of a decision
    that was never taken."""
    b = budget(retrieval_calls_max=1)
    b.spend("retrieval", 1)
    result = router([{"index": 0, "action": RE_RETRIEVE, "query": "x",
                      "reasoning": "worth another try"}],
                    budget=b).route(state_with(finding()))
    assert result.findings[0]["action_taken"] == UNRESOLVED
    assert "unavailable" in result.findings[0]["reasoning"]


def test_revision_ceiling_removes_revise_plan():
    b = budget(max_plan_revisions=1)
    b.record_plan_revision()
    assert REVISE_PLAN not in router(budget=b)._available_actions(
        state_with(finding())
    )


def test_disabling_d1_removes_revise_plan():
    cfg = Config({"agency": {"critic_five_way_routing": True,
                             "d1_plan_revision": False}})
    assert REVISE_PLAN not in router(cfg=cfg, budget=budget())._available_actions(
        state_with(finding())
    )


def test_cheap_routes_are_always_available():
    b = budget(retrieval_calls_max=1, llm_calls_max=1)
    b.spend("retrieval", 1)
    b.spend("llm", 1)
    available = router(budget=b)._available_actions(state_with(finding()))
    assert {DROP_CLAIM, HEDGE, UNRESOLVED} <= available


# --- what the model is shown ----------------------------------------------


def test_the_prompt_shows_the_attempted_queries():
    """RE_RETRIEVE vs UNRESOLVED turns on whether retrieval was genuinely
    tried, which is only visible in the history."""
    provider = FakeProvider([{"index": 0, "action": UNRESOLVED, "reasoning": "r"}])
    CriticRouter(CFG, provider=provider, budget=budget()).route(
        state_with(finding(), queries=["first phrasing", "second phrasing"])
    )
    system = provider.calls[0]["system"]
    assert "first phrasing" in system and "second phrasing" in system


def test_the_prompt_shows_why_the_claim_failed():
    provider = FakeProvider([{"index": 0, "action": HEDGE, "reasoning": "r"}])
    CriticRouter(CFG, provider=provider, budget=budget()).route(
        state_with(finding(reasoning="the figure is for FlashAttention-1"))
    )
    assert "FlashAttention-1" in provider.calls[0]["system"]


def test_the_prompt_lists_only_available_actions():
    b = budget(retrieval_calls_max=1)
    b.spend("retrieval", 1)
    provider = FakeProvider([{"index": 0, "action": UNRESOLVED, "reasoning": "r"}])
    CriticRouter(CFG, provider=provider, budget=b).route(state_with(finding()))
    listed = provider.calls[0]["system"].split("available actions:")[-1]
    assert RE_RETRIEVE not in listed


def test_the_whole_draft_costs_one_routing_call():
    provider = FakeProvider([
        {"index": i, "action": UNRESOLVED, "reasoning": "r"} for i in range(5)
    ])
    CriticRouter(CFG, provider=provider, budget=budget()).route(
        state_with(*[finding(claim=f"Claim {i}.") for i in range(5)])
    )
    assert len(provider.calls) == 1


# --- the A5 ablation -------------------------------------------------------


def test_binary_routing_collapses_the_five_options():
    """A5's static counterpart: the conditional edge §2.3 describes, not a
    decision."""
    cfg = Config({"agency": {"critic_five_way_routing": False,
                             "d1_plan_revision": True}})
    result = router(cfg=cfg, budget=budget()).route(
        state_with(finding(verdict=UNSUPPORTED),
                   finding(verdict=CONTRADICTED))
    )
    assert {f["action_taken"] for f in result.findings} == {RE_RETRIEVE}


def test_binary_routing_without_budget_marks_unresolved():
    cfg = Config({"agency": {"critic_five_way_routing": False,
                             "d1_plan_revision": True}})
    b = budget(retrieval_calls_max=1)
    b.spend("retrieval", 1)
    result = router(cfg=cfg, budget=b).route(state_with(finding()))
    assert result.findings[0]["action_taken"] == UNRESOLVED


def test_binary_routing_makes_no_llm_call():
    cfg = Config({"agency": {"critic_five_way_routing": False,
                             "d1_plan_revision": True}})
    provider = FakeProvider([])
    CriticRouter(cfg, provider=provider, budget=budget()).route(
        state_with(finding())
    )
    assert provider.calls == []


def test_five_way_is_traced_as_adaptive_and_binary_is_not():
    from src.tracing.tracer import DecisionSurface

    for five_way, expected in ((True, True), (False, False)):
        events = []

        class FakeTracer:
            def decision(self, surface, **kw):
                events.append((surface, kw))

        cfg = Config({"agency": {"critic_five_way_routing": five_way,
                                 "d1_plan_revision": True}})
        router([{"index": 0, "action": HEDGE, "reasoning": "r"}],
               cfg=cfg, budget=budget(), tracer=FakeTracer()).route(
            state_with(finding())
        )
        surface, payload = events[0]
        assert surface is DecisionSurface.CRITIC_ROUTING
        assert payload["was_adaptive"] is expected


# --- degradation -----------------------------------------------------------


def test_no_findings_is_an_error_not_a_crash():
    assert router([]).route(new_state("q")).error


def test_routing_failure_marks_everything_unresolved():
    """Routing failure must not discard the verification; unresolved keeps
    the claims visible to D5 rather than silently accepting them."""
    r = CriticRouter(CFG, provider=FakeProvider(error=RuntimeError("down")),
                     budget=budget())
    result = r.route(state_with(finding()))
    assert result.findings[0]["action_taken"] == UNRESOLVED
    assert result.error


def test_an_unknown_action_falls_to_unresolved():
    result = router([{"index": 0, "action": "IGNORE_IT", "reasoning": "r"}],
                    budget=budget()).route(state_with(finding()))
    assert result.findings[0]["action_taken"] == UNRESOLVED


def test_a_missing_decision_falls_to_unresolved():
    result = router([], budget=budget()).route(state_with(finding()))
    assert result.findings[0]["action_taken"] == UNRESOLVED


def test_budget_charged_once():
    b = budget()
    router([{"index": 0, "action": HEDGE, "reasoning": "r"}],
           budget=b).route(state_with(finding()))
    assert b.llm_calls_used == 1


# --- node ------------------------------------------------------------------


def test_node_clears_the_draft_when_more_work_is_needed():
    """A stale draft reaching the decider would look like it reflected the
    new evidence."""
    state = state_with(finding())
    state["draft"] = "an earlier report"
    update = critic_router_node(
        router([{"index": 0, "action": RE_RETRIEVE, "query": "new",
                 "reasoning": "r"}], budget=budget())
    )(state)
    assert update["draft"] == ""


def test_node_keeps_the_draft_when_no_further_work_is_needed():
    state = state_with(finding())
    state["draft"] = "an earlier report"
    update = critic_router_node(
        router([{"index": 0, "action": HEDGE, "reasoning": "r"}],
               budget=budget())
    )(state)
    assert "draft" not in update


def test_node_writes_actions_and_gaps():
    update = critic_router_node(
        router([{"index": 0, "action": UNRESOLVED, "reasoning": "r"}],
               budget=budget())
    )(state_with(finding()))
    assert update["critic_findings"][0]["action_taken"] == UNRESOLVED
    # Existing question-level gaps pass through; no claim text is added.
    assert update["unresolved_gaps"] == []
    assert update["plan"][0]["status"] == "insufficient"
