"""C28.1: fixes from the C28 live run.

The answerable question abstained because two PARTIAL claims, routed
UNRESOLVED, demoted a sub-question whose claims were 3 SUPPORTED + 2
PARTIAL. The false-premise question was not refuted because the verifier
read "without attention" as silence about attention.
"""
from src.config import Config
from src.nodes.critic_router import UNRESOLVED, CriticRouter
from src.nodes.decider import Decider
from src.sufficiency.policy import PARTIAL, gap_for
from src.sufficiency.scorer import HeuristicScorer
from tests.decision_fixtures import STRONG, WEAK, decision_state

CFG = Config({"sufficiency": {"heuristic": {"min_entailment": 0.5}},
              "agency": {"d5_terminal_decision": True,
                         "critic_five_way_routing": True}})


class Routes:
    def __init__(self, routes):
        self.routes = routes

    def complete_json(self, system, user, **kw):
        return {"routes": self.routes}, None


def live_answerable_state():
    """sq1 abandoned, sq2 sufficient with the live run's verdicts, sq3
    insufficient. Findings not yet routed."""
    state = decision_state(WEAK, STRONG, WEAK,
                           statuses={"sq1": "abandoned", "sq3": "insufficient"})
    verdicts = ["SUPPORTED", "PARTIAL", "SUPPORTED", "SUPPORTED", "PARTIAL"]
    state["critic_findings"] = [
        {"claim": f"claim {i}", "sub_question_id": "sq2", "cited_passage_ids": [],
         "verdict": v, "reasoning": "", "action_taken": ""}
        for i, v in enumerate(verdicts)
    ]
    return state


def test_the_live_answerable_run_now_reaches_partial():
    """Route the two PARTIAL claims UNRESOLVED, as the live router did, then
    decide. Before C28.1 sq2 was demoted and the run abstained."""
    state = live_answerable_state()
    failed_routes = [{"index": i, "action": UNRESOLVED, "reasoning": "r"} for i in range(2)]
    routed = CriticRouter(CFG, provider=Routes(failed_routes)).route(state)
    state["critic_findings"] = routed.findings
    state["plan"] = routed.plan
    assert state["plan"][1]["status"] == "sufficient"

    result = Decider(CFG, scorer=HeuristicScorer(CFG)).decide(state)
    assert result.decision.terminal == PARTIAL
    sq2 = [s for s in result.run.sub_questions if s.id == "sq2"][0]
    assert sq2.resolved and sq2.features["f6_entailment_strength"] == 0.8


def test_a_sub_question_whose_only_claim_fails_still_does_not_resolve():
    """The proportionate rule cuts both ways: f6 = 0 is below the bar."""
    state = decision_state(STRONG)
    state["critic_findings"] = [{"claim": "c", "sub_question_id": "sq1",
                                 "cited_passage_ids": [], "verdict": "UNSUPPORTED",
                                 "reasoning": "", "action_taken": UNRESOLVED}]
    run = HeuristicScorer(CFG).score(state)
    assert not run.sub_questions[0].resolved


def test_gaps_are_cut_at_a_word_boundary():
    long = ("According to the FlashAttention-2 paper, what specific blockwise "
            "memory access pattern (including read/write order and tiling "
            "strategy) is used during its attention computation on modern GPUs "
            "with large shared memory?")
    gap = gap_for(long)
    assert gap.endswith("…")
    assert not gap.endswith("atten…")
    assert gap_for("short question?") == "could not establish: short question?"


def test_planner_and_decider_write_identical_gap_lines():
    from src.nodes.planner import _gap
    from src.state import new_sub_question

    text = "x " * 150
    assert _gap(new_sub_question(text)) == gap_for(text)


def test_an_abstention_lists_its_gaps_once():
    from src.nodes.finalizer import Finalizer
    from tests.test_finalizer import state

    s = state("ABSTAIN", draft="", gaps=["could not establish: what pattern?"])
    s["decision_rationale"] = ("Abstained: score 0.00 < tau_partial 0.33 (…). "
                               "Not established: what pattern?.")
    text = Finalizer(Config({})).finalize(s)
    assert text.count("what pattern?") == 1


def test_the_verify_prompt_treats_explicit_absence_as_contradiction():
    from src.prompts import load

    prompt = load("premise_verify")
    assert "EXPLICIT STATEMENT OF ABSENCE is not silence" in prompt
    assert "without attention" in prompt
    assert "CONTRADICTED" in prompt.split("EXPLICIT STATEMENT OF ABSENCE")[1][:600]
