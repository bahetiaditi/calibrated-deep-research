"""C28.2: from the C28.1 live run's answerable question, which abstained on
0 of 3 core sub-questions.

- sq2 (PagedAttention, best f1 5.08, sufficient) was reopened by the critic
  router for one comparison claim; the claim's query scored lower, D3's
  marginal test stopped it, and it ended ABANDONED.
- sq3 (the comparison) was `insufficient` on retrieval (f1 3.93 < 4.0)
  while the critic had verified its claims 6 SUPPORTED + 2 PARTIAL +
  1 UNSUPPORTED — and retrieval status vetoed the verification.
"""
from src.config import Config
from src.nodes.critic_router import RE_RETRIEVE, UNRESOLVED
from src.nodes.decider import Decider
from src.state import new_state, new_sub_question
from src.sufficiency.policy import ABSTAIN, PARTIAL
from src.sufficiency.scorer import HeuristicScorer, sub_question_scores
from tests.decision_fixtures import STRONG, WEAK, decision_state
from tests.test_adaptive_retriever import FakeProvider, FakeRetriever, build
from tests.test_adaptive_retriever import passage as rpassage
from tests.test_critic_router import budget, finding, router, state_with

CFG = Config({"sufficiency": {"heuristic": {"min_entailment": 0.5,
                                            "min_supported_claims": 2}},
              "agency": {"d5_terminal_decision": True}})

LIVE_SQ3 = ["SUPPORTED"] * 6 + ["PARTIAL", "UNSUPPORTED", "PARTIAL"]


# --- 1. a claim-driven round cannot revoke a sub-question ------------------


def test_the_router_records_what_it_reopened():
    s = state_with(finding(sub_question_id="sq1"))
    s["plan"][0]["status"] = "sufficient"
    result = router([{"index": 0, "action": RE_RETRIEVE, "query": "a fresh angle",
                      "reasoning": "r"}], budget=budget()).route(s)
    sq = result.plan[0]
    assert sq["status"] == "in_progress"
    assert sq["reopened_from"] == "sufficient"


def test_a_second_claim_does_not_overwrite_the_recorded_status():
    s = state_with(finding(sub_question_id="sq1"), finding(sub_question_id="sq1"))
    s["plan"][0]["status"] = "sufficient"
    result = router([
        {"index": 0, "action": RE_RETRIEVE, "query": "angle one", "reasoning": "r"},
        {"index": 1, "action": RE_RETRIEVE, "query": "angle two", "reasoning": "r"},
    ], budget=budget()).route(s)
    assert result.plan[0]["reopened_from"] == "sufficient"


def reopened(status_before, best_f1=5.08):
    sq = new_sub_question("What memory access pattern does PagedAttention use?",
                          sub_question_id="sq2")
    sq.update({"status": "in_progress", "reopened_from": status_before,
               "rounds": 2, "best_f1": best_f1, "suggested_route": "arxiv_meta",
               "attempted_routes": ["arxiv_meta"],
               "attempted_queries": ["PagedAttention memory access pattern"],
               "next_query": "FlashAttention-2 dense tiled reuse compared to "
                             "PagedAttention independent block reads",
               "evidence_ids": ["P_old1", "P_old2"]})
    state = new_state("q")
    state["plan"] = [sq]
    return state


def run_reopened(status_before, passages):
    retriever = build(retriever=FakeRetriever(default=passages),
                      provider=FakeProvider([{"action": "give_up", "reasoning": "r"}]))
    return retriever.retrieve_all(reopened(status_before))["plan"][0]


def test_the_live_case_a_weak_claim_round_leaves_sufficient_sufficient():
    """Before C28.2 this ended ABANDONED."""
    sq = run_reopened("sufficient", [rpassage("P_new", 1.2, "X")])
    assert sq["status"] == "sufficient"
    assert sq["reopened_from"] is None


def test_the_claim_round_still_adds_its_passages():
    sq = run_reopened("sufficient", [rpassage("P_new", 1.2, "X")])
    assert "P_new" in sq["evidence_ids"]
    assert {"P_old1", "P_old2"} <= set(sq["evidence_ids"])


def test_an_insufficient_sub_question_is_not_made_worse():
    sq = run_reopened("insufficient", [rpassage("P_new", -3.0, "X")])
    assert sq["status"] == "insufficient"


def test_a_claim_round_may_still_improve_a_sub_question():
    """The floor only stops a worse ending; a better one stands."""
    sq = run_reopened("insufficient", [rpassage("P_a", 9.0, "X"), rpassage("P_b", 8.5, "Y")],)
    assert sq["status"] == "sufficient"


def test_ordinary_rounds_are_unaffected():
    """No reopened_from, no floor: D3 decides as it always has."""
    retriever = build(retriever=FakeRetriever(default=[rpassage("P1", -3.0, "X")]),
                      provider=FakeProvider([{"action": "give_up", "reasoning": "r"}]))
    state = new_state("q")
    state["plan"] = [new_sub_question("hard", sub_question_id="sq1")]
    state["plan"][0]["suggested_route"] = "arxiv_meta"
    assert retriever.retrieve_all(state)["plan"][0]["status"] in ("abandoned", "insufficient")


# --- 2. verification can resolve -------------------------------------------


def with_verdicts(state, sqid, verdicts, action="ACCEPT"):
    state["critic_findings"] = state.get("critic_findings", []) + [
        {"claim": f"{sqid} claim {i}", "sub_question_id": sqid,
         "cited_passage_ids": [], "verdict": v, "reasoning": "", "action_taken": action}
        for i, v in enumerate(verdicts)
    ]
    return state


def test_the_live_sq3_resolves_on_verification():
    s = with_verdicts(decision_state(WEAK, statuses={"sq1": "insufficient"}),
                      "sq1", LIVE_SQ3)
    sub = sub_question_scores(s)[0]
    assert sub.resolved
    assert "resolved by verification: 6 supported" in sub.note


def test_one_supported_claim_is_not_enough():
    s = with_verdicts(decision_state(WEAK, statuses={"sq1": "insufficient"}),
                      "sq1", ["SUPPORTED"])
    assert not sub_question_scores(s)[0].resolved


def test_weak_verification_does_not_resolve():
    s = with_verdicts(decision_state(WEAK, statuses={"sq1": "insufficient"}),
                      "sq1", ["SUPPORTED", "SUPPORTED", "UNSUPPORTED", "UNSUPPORTED",
                              "UNSUPPORTED"])
    assert not sub_question_scores(s)[0].resolved      # f6 = 0.4


def test_retrieval_sufficiency_still_resolves_without_claims():
    s = decision_state(STRONG)
    assert sub_question_scores(s)[0].resolved


def test_the_threshold_is_configurable():
    s = with_verdicts(decision_state(WEAK, statuses={"sq1": "insufficient"}),
                      "sq1", ["SUPPORTED"])
    assert sub_question_scores(s, min_supported=1)[0].resolved


# --- 3. the live run, end to end at the decider ----------------------------


def test_the_live_answerable_run_now_reaches_partial():
    """sq1 abandoned; sq2 sufficient (no longer revoked); sq3 insufficient
    on retrieval but verified. Before C28.2: 0 of 3, ABSTAIN."""
    s = decision_state(WEAK, STRONG, WEAK,
                       statuses={"sq1": "abandoned", "sq3": "insufficient"})
    s = with_verdicts(s, "sq3", LIVE_SQ3)
    result = Decider(CFG, scorer=HeuristicScorer(CFG)).decide(s)
    assert result.decision.terminal == PARTIAL
    assert "2 of 3 core" in result.decision.rationale


def test_before_c28_2_the_same_evidence_abstained():
    """Revoked sq2 and status-vetoed sq3, as the live run had them."""
    s = decision_state(WEAK, STRONG, WEAK, statuses={
        "sq1": "abandoned", "sq2": "abandoned", "sq3": "insufficient"})
    s = with_verdicts(s, "sq3", ["SUPPORTED"])          # too little to verify
    assert Decider(CFG, scorer=HeuristicScorer(CFG)).decide(s).decision.terminal == ABSTAIN
