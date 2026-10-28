"""C27.2: state that must belong to ONE question, and decisions that must
have an effect.

All three defects were found from the C27 live traces, and none was caught
by an existing test, because every existing test built a fresh retriever per
question and attributed claims through a field the fixtures set correctly:

- The second live question's sq1 was retrieved with the FIRST question's
  reformulated query. AdaptiveRetriever kept D2/D3 memory in instance dicts
  keyed by sub-question id, and every plan uses sq1, sq2, sq3.
- Eight RE_RETRIEVE decisions reopened nothing: the critic attributed claims
  from a stale passage payload, not this run's plan.
- A RE_RETRIEVE that did reopen replayed the exact query that had failed.
"""
from src.config import Config
from src.nodes.critic_router import (
    RE_RETRIEVE,
    UNRESOLVED,
    CriticRouter,
)
from src.nodes.planner import REWRITE, Planner
from src.state import new_sub_question
from tests.test_adaptive_retriever import (
    IN_PROGRESS,
    FakeProvider,
    FakeRetriever,
    build,
    outcome,
    passage,
    state_with,
)
from tests.test_critic_router import FakeProvider as RouterProvider
from tests.test_critic_router import finding
from tests.test_critic_router import state_with as router_state

ROUTER_CFG = Config({"agency": {"critic_five_way_routing": True,
                                "d1_plan_revision": True},
                     "budget": {"max_plan_revisions": 2}})


# --- the cross-question leak, reproduced -----------------------------------


def test_two_questions_through_one_retriever_do_not_share_queries():
    """The C27 live failure: one retriever instance, two questions, both with
    a sub-question called sq1. Before C27.2 the second question's first
    query was the first question's reformulation."""
    impl = FakeRetriever(
        by_query={"hard A": [], "rewritten A": [passage("P1", 9.0, "A"),
                                                passage("P2", 8.0, "B")]},
        default=[passage("P3", 9.0, "C"), passage("P4", 8.0, "D")],
    )
    retriever = build(retriever=impl, provider=FakeProvider([{
        "action": "wider", "route": "web", "query": "rewritten A", "reasoning": "r"}]))

    first = retriever.retrieve_all(state_with("hard A"))
    assert first["plan"][0]["rounds"] == 2        # it did escalate

    impl.queries.clear()
    retriever.retrieve_all(state_with("question B, entirely unrelated"))
    assert impl.queries[0] == "question B, entirely unrelated"
    assert "rewritten A" not in impl.queries


def test_round_one_is_never_judged_against_another_questions_f1():
    """Live: 'round 1 improved f1 by -0.34' — a comparison with a previous
    run. A first round has nothing to improve on, so weak evidence must
    escalate rather than be stopped by the marginal test."""
    retriever = build()
    sq_a = new_sub_question("A", sub_question_id="sq1")
    retriever.decide_next(sq_a, outcome([9.0, 8.5], ["A", "B"]))
    assert sq_a["best_f1"] == 9.0

    sq_b = new_sub_question("B", sub_question_id="sq1")
    assert retriever.decide_next(sq_b, outcome([2.0, 1.0], ["A", "B"])) == IN_PROGRESS
    assert sq_b["best_f1"] == 2.0


def test_the_retriever_instance_holds_no_per_question_state():
    retriever = build()
    retriever.decide_next(new_sub_question("A", sub_question_id="sq1"),
                          outcome([1.0], ["A"]))
    per_question = {k: v for k, v in vars(retriever).items()
                    if isinstance(v, dict) and v and k.startswith("_")}
    assert per_question == {}


def test_a_round_does_not_mutate_the_state_it_was_given():
    """The checkpointed state object must not change under a node; the
    carry fields arrive in the returned update instead."""
    state = state_with("hard A")
    retriever = build(retriever=FakeRetriever(default=[]), provider=FakeProvider([{
        "action": "wider", "route": "web", "query": "rewritten A", "reasoning": "r"}]))
    update = retriever.retrieve_one(state, "sq1")
    assert state["plan"][0]["next_query"] is None
    assert update["plan"][0]["next_query"] == "rewritten A"
    assert update["plan"][0]["next_route"] == "web"


def test_a_consumed_reformulation_is_not_replayed():
    retriever = build()
    sq = new_sub_question("A", sub_question_id="sq1")
    sq["next_query"] = "used last round"
    retriever.decide_next(sq, outcome([9.0, 8.0], ["A", "B"]))   # sufficient
    assert sq["next_query"] is None


def test_a_rewrite_resets_the_round_memory():
    """Live: a rewritten sub-question was abandoned after one round for not
    improving on the OLD question's best f1."""
    target = new_sub_question("What rank did they use?", sub_question_id="sq1",
                              rationale="pin down the number")
    target.update({"status": "insufficient", "rounds": 2, "best_f1": 3.1,
                   "next_query": "stale", "next_route": "web",
                   "attempted_queries": ["rank used"]})

    class Provider:
        def complete_json(self, system, user, **kw):
            return {"action": REWRITE, "reasoning": "vague",
                    "text": "Which LoRA rank was used in the GPT-3 experiments?"}, None

    update = Planner(Config({}), provider=Provider()).revise(
        {"question": "q", "plan": [target]}, "sq1")
    revised = update["plan"][0]
    assert (revised["best_f1"], revised["next_query"], revised["next_route"]) == (None, None, None)
    assert revised["attempted_queries"] == ["rank used"]   # still never reissued


# --- routing decisions must have an effect ---------------------------------


def router(routes, five_way=True):
    cfg = ROUTER_CFG.with_overrides({"agency.critic_five_way_routing": five_way})
    return CriticRouter(cfg, provider=RouterProvider(routes))


def test_re_retrieve_on_an_unowned_claim_becomes_unresolved():
    unowned = finding(sub_question_id="")
    result = router([{"index": 0, "action": RE_RETRIEVE, "query": "new phrasing",
                      "reasoning": "r"}]).route(router_state(unowned))
    assert result.findings[0]["action_taken"] == UNRESOLVED
    assert "no sub-question" in result.findings[0]["reasoning"]
    assert not result.needs_more_work


def test_re_retrieve_without_a_new_query_becomes_unresolved():
    claim = finding(sub_question_id="sq1")
    result = router([{"index": 0, "action": RE_RETRIEVE, "query": "Original Phrasing",
                      "reasoning": "r"}]).route(router_state(claim))
    assert result.findings[0]["action_taken"] == UNRESOLVED
    assert "already tried" in result.findings[0]["reasoning"]


def test_re_retrieve_with_a_new_query_reopens_and_hands_over_the_query():
    claim = finding(sub_question_id="sq1")
    result = router([{"index": 0, "action": RE_RETRIEVE, "query": "a fresh angle",
                      "reasoning": "r"}]).route(router_state(claim))
    assert result.findings[0]["action_taken"] == RE_RETRIEVE
    assert result.needs_more_work
    reopened = result.plan[0]
    assert reopened["status"] == "in_progress"
    assert reopened["next_query"] == "a fresh angle"


def test_the_handed_over_query_is_what_the_retriever_issues_next():
    """The effect, end to end: the router's query reaches the retrieval."""
    claim = finding(sub_question_id="sq1")
    routed = router([{"index": 0, "action": RE_RETRIEVE, "query": "a fresh angle",
                      "reasoning": "r"}]).route(router_state(claim))
    sq = routed.plan[0]
    assert build().formulate_query(sq) == "a fresh angle"


def test_a5_binary_routing_does_not_reopen_unowned_claims():
    result = router([], five_way=False).route(
        router_state(finding(sub_question_id=""), finding(sub_question_id="sq1")))
    actions = [f["action_taken"] for f in result.findings]
    assert actions == [UNRESOLVED, RE_RETRIEVE]
