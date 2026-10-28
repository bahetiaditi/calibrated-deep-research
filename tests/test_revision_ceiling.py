"""D1's revision ceiling, enforced in a real wiring and across a resume.

Found from the C27 live run: both runs revised the plan and the budget line
read `revisions 0/2`. The ceiling had never bound in a real run. Every
existing test passed a budget straight into `planner_node(...)`; production
never does, so the tests exercised a wiring nothing else used.
"""
from src.budget import Budget, plan_revisions_remain
from src.config import Config
from src.controller import PLANNER, RETRIEVER, SYNTHESIZER, Controller
from src.graph import Components, _attach, build_graph
from src.nodes.critic_router import REVISE_PLAN, CriticRouter
from src.nodes.planner import REWRITE, Planner, planner_node
from src.state import new_state, new_sub_question

CFG = Config({
    "agency": {"d1_plan_revision": True, "skip_research_on_failed_premise": True,
               "critic_five_way_routing": True},
    "budget": {"max_rounds_per_subquestion": 3, "max_plan_revisions": 2},
    "planning": {"max_sub_questions": 7},
})


def budget():
    return Budget(llm_calls_max=50, retrieval_calls_max=50,
                  wall_clock_max_s=900, max_plan_revisions=2)


class ScriptedProvider:
    """Initial plan first, then a fresh rewrite on every revision call."""

    def __init__(self):
        self.calls = 0

    def complete_json(self, system, user, **kw):
        self.calls += 1
        if self.calls == 1:
            return {"sub_questions": [{
                "text": "What rank did the LoRA authors use for GPT-3?",
                "route": "arxiv_meta", "rationale": "the number is the answer"}]}, None
        return {"action": REWRITE,
                "text": f"Which LoRA rank was reported in experiment variant {self.calls}?",
                "reasoning": "rephrase"}, None


def insufficient_sq(revisions=0):
    sq = new_sub_question("What rank did they use?", sub_question_id="sq1",
                          rationale="pin down the number")
    sq.update({"status": "insufficient", "rounds": 2,
               "attempted_queries": ["rank used"], "attempted_routes": ["arxiv_meta"]})
    state = new_state("What rank does LoRA use?")
    state["plan"] = [sq]
    state["plan_revisions"] = revisions
    state["premise_check"] = {"ran": True}
    return state


# --- the live bug, reproduced ----------------------------------------------


def test_revision_is_counted_on_the_budget_attach_binds():
    """default_components builds `planner_node(Planner(cfg))` with no budget;
    `_attach` binds the run's budget onto the planner. Before the fix the
    node read its own closure variable, which stayed None."""
    components = Components(planner=planner_node(Planner(CFG, provider=ScriptedProvider())))
    b = budget()
    _attach(components, tracer=None, budget=b)

    state = insufficient_sq()
    state["plan"][0]["text"] = "What rank did they use?"
    components.planner(state | {"plan": state["plan"]})   # revision mode: plan exists
    assert b.plan_revisions == 1


def test_a_revising_run_stops_at_the_ceiling():
    """End to end: a planner that would rewrite forever, a retriever that
    never succeeds. The ceiling, not the recursion limit, must end it."""
    visits = {"planner": 0, "retriever": 0}
    planner = planner_node(Planner(CFG, provider=ScriptedProvider()))

    def counted_planner(state):
        visits["planner"] += 1
        return planner(state)

    def failing_retriever(state):
        visits["retriever"] += 1
        plan = [{**sq, "status": "insufficient", "rounds": sq["rounds"] + 1}
                if sq["status"] == "pending" else sq for sq in state["plan"]]
        return {"plan": plan}

    components = Components(planner=counted_planner, retriever=failing_retriever)
    b = budget()
    _attach(Components(planner=planner), tracer=None, budget=b)
    graph = build_graph(components, config=CFG, budget=b)
    final = graph.invoke(new_state("What rank does LoRA use?"), {"recursion_limit": 40})

    assert final["plan_revisions"] == 2
    assert visits["planner"] == 3          # initial plan + exactly two revisions
    assert b.plan_revisions == 2


# --- the ceiling survives a resume -----------------------------------------


def test_a_resumed_run_does_not_get_its_revisions_back():
    """The Budget object is not checkpointed. state['plan_revisions'] is."""
    resumed = Controller(CFG, available={PLANNER, RETRIEVER, SYNTHESIZER},
                         budget=budget())          # fresh: 0 recorded
    assert resumed.route(insufficient_sq(revisions=2)) != PLANNER


def test_the_ceiling_holds_with_no_budget_object():
    controller = Controller(CFG, available={PLANNER, RETRIEVER, SYNTHESIZER})
    assert controller.route(insufficient_sq(revisions=2)) != PLANNER
    assert controller.route(insufficient_sq(revisions=1)) == PLANNER


def test_router_withholds_revise_plan_at_the_ceiling_after_resume():
    router = CriticRouter(CFG, provider=object(), budget=budget())
    assert REVISE_PLAN in router._available_actions(insufficient_sq(revisions=1))
    assert REVISE_PLAN not in router._available_actions(insufficient_sq(revisions=2))


def test_the_larger_count_wins():
    b = budget()
    b.plan_revisions = 2
    assert not plan_revisions_remain({"plan_revisions": 0}, b, 2)
    assert not plan_revisions_remain({"plan_revisions": 2}, budget(), 2)
    assert plan_revisions_remain({"plan_revisions": 1}, budget(), 2)
