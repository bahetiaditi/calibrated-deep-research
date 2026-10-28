"""C27.8: D5 decides on core sub-questions; background ones are not evidence
that the question is answerable.

The live case: "Why does Mamba-2 use rotary position embeddings in its SSM
blocks?" was decomposed into what-is-RoPE (background), how-Mamba-2-uses-RoPE
and why-the-authors-chose-it (both resting on the false premise). Only the
background one resolved, and 1 of 3 met the heuristic's PARTIAL bar — so the
report explained RoPE beside a premise it never questioned.
"""
from src.config import Config
from src.nodes.decider import Decider
from src.nodes.planner import Planner
from src.state import new_sub_question
from src.sufficiency.policy import ABSTAIN, ANSWER, PARTIAL, Thresholds
from src.sufficiency.scorer import CalibratedScorer, HeuristicScorer
from tests.decision_fixtures import STRONG, WEAK, decision_state, fitted_models

CFG = Config({"sufficiency": {"heuristic": {"min_entailment": 0.5}},
              "agency": {"d5_terminal_decision": True}})


def with_kinds(state, *kinds):
    for sq, kind in zip(state["plan"], kinds):
        sq["kind"] = kind
    return state


def decide(state):
    return Decider(CFG, scorer=HeuristicScorer(CFG)).decide(state)


# --- the live case ---------------------------------------------------------


def test_a_resolved_background_sub_question_alone_does_not_make_a_partial():
    state = with_kinds(decision_state(STRONG, WEAK, WEAK),
                       "background", "core", "core")
    result = decide(state)
    assert result.decision.terminal == ABSTAIN
    assert "0 of 2 core" in result.decision.rationale
    assert "1 background not scored" in result.decision.rationale


def test_before_c27_8_the_same_run_was_partial():
    """The same evidence with every sub-question core — the old behaviour."""
    state = with_kinds(decision_state(STRONG, WEAK, WEAK), "core", "core", "core")
    assert decide(state).decision.terminal == PARTIAL


# --- the rule in both directions -------------------------------------------


def test_an_unresolved_background_sub_question_does_not_block_an_answer():
    """Missing context is not a missing answer."""
    state = with_kinds(decision_state(WEAK, STRONG, STRONG),
                       "background", "core", "core")
    assert decide(state).decision.terminal == ANSWER


def test_background_gaps_are_not_listed_as_gaps_in_the_answer():
    state = with_kinds(decision_state(WEAK, STRONG, WEAK),
                       "background", "core", "core")
    gaps = decide(state).update["decision"]["gaps"]
    assert "could not establish: sub-question 3" in gaps
    assert "could not establish: sub-question 1" not in gaps


def test_the_record_keeps_every_sub_question_with_its_kind():
    """C33 trains stage 1 on background sub-questions too; only stage 2's
    aggregation is core-only."""
    state = with_kinds(decision_state(STRONG, WEAK), "background", "core")
    subs = decide(state).update["decision"]["sub_questions"]
    assert [(s["id"], s["kind"]) for s in subs] == [("sq1", "background"), ("sq2", "core")]


def test_calibrated_stage2_aggregates_core_only():
    stage1, stage2 = fitted_models()
    scorer = CalibratedScorer(stage1, stage2, Thresholds(0.65, 0.40))
    state = with_kinds(decision_state(STRONG, WEAK), "background", "core")
    run = scorer.score(state)
    assert run.run_features["fraction_sufficient"] == 0.0     # the core one failed
    assert all(s.p is not None for s in run.sub_questions)    # both still scored
    core_p = [s.p for s in run.sub_questions if s.kind == "core"]
    assert run.run_features["mean_sub_question_p"] == core_p[0]


def test_old_plans_without_a_kind_are_core():
    state = decision_state(STRONG, WEAK)
    for sq in state["plan"]:
        sq.pop("kind", None)
    assert decide(state).decision.terminal == PARTIAL


# --- the planner -----------------------------------------------------------


class Provider:
    def __init__(self, response):
        self.response = response

    def complete_json(self, system, user, **kw):
        return self.response, None


def plan(items):
    return Planner(Config({}), provider=Provider({"sub_questions": items})).plan(
        "Why does Mamba-2 use rotary position embeddings?")


def test_the_planner_records_kind():
    sqs = plan([
        {"text": "What are rotary position embeddings?", "kind": "background",
         "route": "arxiv_meta", "rationale": "context"},
        {"text": "How does Mamba-2 use rotary position embeddings?", "kind": "core",
         "route": "arxiv_meta", "rationale": "the question"},
    ])
    assert [sq["kind"] for sq in sqs] == ["background", "core"]


def test_a_missing_or_unknown_kind_is_core():
    """Mis-tagging core as background would remove it from the score; the
    reverse only restores the old behaviour. So only an explicit
    'background' counts."""
    sqs = plan([
        {"text": "What are rotary position embeddings?", "route": "web"},
        {"text": "How does Mamba-2 use rotary position embeddings?", "kind": "context"},
    ])
    assert [sq["kind"] for sq in sqs] == ["core", "core"]


def test_a_plan_with_nothing_core_is_treated_as_all_core():
    sqs = plan([
        {"text": "What are rotary position embeddings?", "kind": "background"},
        {"text": "What is a state space model in deep learning?", "kind": "background"},
    ])
    assert [sq["kind"] for sq in sqs] == ["core", "core"]


def test_split_parts_inherit_the_parents_kind():
    from src.nodes.planner import SPLIT

    target = new_sub_question("How and why does Mamba-2 use RoPE?",
                              sub_question_id="sq2", kind="core")
    target.update({"status": "insufficient", "rounds": 2})
    update = Planner(Config({}), provider=Provider({
        "action": SPLIT, "reasoning": "two questions",
        "parts": ["How does Mamba-2 use rotary position embeddings?",
                  "Why did the Mamba-2 authors choose rotary embeddings?"],
    })).revise({"question": "q", "plan": [target]}, "sq2")
    assert [sq["kind"] for sq in update["plan"]] == ["core", "core"]


def test_the_prompt_defines_kind():
    from src.prompts import load

    prompt = load("planner_initial")
    assert '"kind": "core"' in prompt
    assert "At least one sub-question must be core" in prompt
