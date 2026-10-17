"""Tests for the planner (initial decomposition).

The LLM is faked. What is tested is the contract around it: that the plan
carries enough for C25 to revise it, that malformed output degrades instead
of crashing, and that initial decomposition is not miscounted as adaptive —
which would make every A1 ablation run fail C35's check for the wrong reason.
"""
import pytest

from src.config import Config
from src.nodes.planner import (
    DEFAULT_ROUTE,
    MIN_TEXT_CHARS,
    VALID_ROUTES,
    Planner,
    planner_node,
)

CFG = Config({"planning": {"max_sub_questions": 7}})

QUESTION = "How do FlashAttention-2 and PagedAttention differ in memory access?"


class FakeProvider:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def complete_json(self, system, user, *, role=None, required_keys=None, **kw):
        self.calls.append({"system": system, "user": user, "role": role})
        if self.error:
            raise self.error
        return self.response, None


def planner(response=None, **kw):
    return Planner(CFG, provider=FakeProvider(response or THREE), **kw)


THREE = {"sub_questions": [
    {"text": "How does FlashAttention-2 structure its memory access?",
     "route": "arxiv_fulltext", "rationale": "need the kernel-level detail"},
    {"text": "How does PagedAttention structure its memory access?",
     "route": "arxiv_fulltext", "rationale": "the other half of the comparison"},
    {"text": "What is the key-value cache in transformer inference?",
     "route": "arxiv_meta", "rationale": "shared background both rely on"},
]}


# --- decomposition ---------------------------------------------------------


def test_returns_sub_questions():
    plan = planner().plan(QUESTION)
    assert len(plan) == 3
    assert plan[0]["text"].startswith("How does FlashAttention-2")


def test_ids_are_sequential_and_unique():
    plan = planner().plan(QUESTION)
    assert [sq["id"] for sq in plan] == ["sq1", "sq2", "sq3"]


def test_sub_questions_start_pending():
    for sq in planner().plan(QUESTION):
        assert sq["status"] == "pending"
        assert sq["rounds"] == 0
        assert sq["sufficiency"] is None
        assert sq["attempted_routes"] == [] and sq["attempted_queries"] == []


def test_rationale_is_captured():
    """C25 must know WHY a sub-question exists to choose between rewriting,
    splitting, merging and abandoning it. Without this, every retrieval
    failure looks the same and D1 collapses into a retry loop."""
    plan = planner().plan(QUESTION)
    assert all(sq["rationale"] for sq in plan)
    assert "comparison" in plan[1]["rationale"]


def test_suggested_route_recorded():
    plan = planner().plan(QUESTION)
    assert plan[0]["suggested_route"] == "arxiv_fulltext"
    assert plan[2]["suggested_route"] == "arxiv_meta"


def test_suggested_route_is_not_an_attempted_route():
    """Recording the planner's guess as already-attempted would tell D2 the
    route had been tried and failed — exactly wrong, and it would push the
    retriever to escalate on its first move."""
    plan = planner().plan(QUESTION)
    assert plan[0]["attempted_routes"] == []


def test_premise_check_is_passed_to_the_prompt():
    provider = FakeProvider(THREE)
    Planner(CFG, provider=provider).plan(QUESTION, premise_check={
        "ran": True, "failed": False,
        "presuppositions": [
            {"text": "FlashAttention-2 exists", "verdict": "SUPPORTED"},
        ],
    })
    assert "ALREADY ESTABLISHED" in provider.calls[0]["system"]
    assert "FlashAttention-2 exists" in provider.calls[0]["system"]


def test_false_premise_warns_the_planner():
    """A plan that elaborates a false premise wastes the whole budget on
    fiction."""
    provider = FakeProvider(THREE)
    Planner(CFG, provider=provider).plan("Why does Mamba-2 use RoPE?",
        premise_check={
            "ran": True, "failed": True,
            "correction": "Mamba-2 uses a selective state space mechanism.",
            "presuppositions": [],
        })
    system = provider.calls[0]["system"]
    assert "false premise" in system
    assert "selective state space" in system


def test_no_premise_check_adds_nothing():
    provider = FakeProvider(THREE)
    Planner(CFG, provider=provider).plan(QUESTION)
    assert "ALREADY ESTABLISHED" not in provider.calls[0]["system"]


# --- defensive parsing -----------------------------------------------------


def test_ceiling_is_enforced():
    """Each sub-question costs retrieval budget from a pool of about 20;
    splitting one lookup into ten starves the ones that needed it."""
    many = {"sub_questions": [
        {"text": f"A sufficiently long sub-question number {i}?",
         "route": "web", "rationale": "r"} for i in range(12)
    ]}
    assert len(planner(many).plan(QUESTION)) == 7


def test_configurable_ceiling():
    cfg = Config({"planning": {"max_sub_questions": 2}})
    many = {"sub_questions": [
        {"text": f"A sufficiently long sub-question number {i}?",
         "route": "web", "rationale": "r"} for i in range(6)
    ]}
    p = Planner(cfg, provider=FakeProvider(many))
    assert len(p.plan(QUESTION)) == 2


def test_duplicate_sub_questions_collapse():
    dupes = {"sub_questions": [
        {"text": "What memory does FlashAttention-2 use?", "route": "web",
         "rationale": "a"},
        {"text": "what memory does flashattention-2 use",  "route": "web",
         "rationale": "b"},
    ]}
    assert len(planner(dupes).plan(QUESTION)) == 1


def test_fragments_are_dropped():
    short = {"sub_questions": [
        {"text": "the rank?", "route": "web", "rationale": "r"},
        {"text": "What LoRA rank did the authors use in their experiments?",
         "route": "arxiv_fulltext", "rationale": "r"},
    ]}
    plan = planner(short).plan(QUESTION)
    assert len(plan) == 1
    assert len(plan[0]["text"]) >= MIN_TEXT_CHARS


def test_plain_strings_are_accepted():
    """The model controls this shape; a bare list of strings is a common
    deviation and is perfectly usable."""
    plan = planner({"sub_questions": [
        "How does FlashAttention-2 handle memory?",
        "How does PagedAttention handle memory?",
    ]}).plan(QUESTION)
    assert len(plan) == 2
    assert plan[0]["suggested_route"] == DEFAULT_ROUTE


def test_alternate_field_names_accepted():
    plan = planner({"sub_questions": [
        {"question": "How does FlashAttention-2 handle memory?",
         "reason": "needed for the comparison"},
    ]}).plan(QUESTION)
    assert plan[0]["rationale"] == "needed for the comparison"


@pytest.mark.parametrize("route", VALID_ROUTES)
def test_all_valid_routes_pass_through(route):
    plan = planner({"sub_questions": [
        {"text": "A sufficiently long sub-question here?", "route": route,
         "rationale": "r"},
    ]}).plan(QUESTION)
    assert plan[0]["suggested_route"] == route


def test_invalid_route_falls_back():
    plan = planner({"sub_questions": [
        {"text": "A sufficiently long sub-question here?",
         "route": "telepathy", "rationale": "r"},
    ]}).plan(QUESTION)
    assert plan[0]["suggested_route"] == DEFAULT_ROUTE


def test_empty_decomposition_falls_back_to_the_question():
    """A question the planner could not decompose is still a question.
    Researching it whole beats abandoning it before any retrieval."""
    plan = planner({"sub_questions": []}).plan(QUESTION)
    assert len(plan) == 1
    assert plan[0]["text"] == QUESTION
    assert "no usable decomposition" in plan[0]["rationale"]


def test_all_fragments_falls_back_to_the_question():
    plan = planner({"sub_questions": [{"text": "x?"}, {"text": "y?"}]}).plan(QUESTION)
    assert plan[0]["text"] == QUESTION


def test_empty_question_returns_empty_plan():
    assert planner().plan("   ") == []


def test_provider_failure_propagates():
    """Unlike the premise checker, a planner failure IS fatal to the run:
    there is nothing to research without a plan, and silently inventing one
    would hide the failure from the trace."""
    p = Planner(CFG, provider=FakeProvider(error=RuntimeError("down")))
    with pytest.raises(RuntimeError):
        p.plan(QUESTION)


# --- budget and tracing ----------------------------------------------------


def test_budget_charged_once():
    from src.budget import Budget

    budget = Budget(llm_calls_max=25, retrieval_calls_max=20,
                    wall_clock_max_s=900)
    Planner(CFG, provider=FakeProvider(THREE), budget=budget).plan(QUESTION)
    assert budget.llm_calls_used == 1


def test_initial_decomposition_is_not_counted_as_adaptive():
    """Initial decomposition happens in EVERY configuration, including the
    A1 ablation that freezes revision. Counting it as adaptive would make an
    A1 run report non-zero D1 activity and fail C35's check for the wrong
    reason."""
    from src.tracing.tracer import DecisionSurface

    events = []

    class FakeTracer:
        def decision(self, surface, **kw):
            events.append((surface, kw))

    Planner(CFG, provider=FakeProvider(THREE), tracer=FakeTracer()).plan(QUESTION)
    surface, payload = events[0]
    assert surface is DecisionSurface.D1_PLAN_REVISION
    assert payload["was_adaptive"] is False
    assert payload["inputs"]["n_sub_questions"] == 3


def test_trace_records_the_routes_chosen():
    events = []

    class FakeTracer:
        def decision(self, surface, **kw):
            events.append(kw)

    Planner(CFG, provider=FakeProvider(THREE), tracer=FakeTracer()).plan(QUESTION)
    assert events[0]["inputs"]["routes"] == [
        "arxiv_fulltext", "arxiv_fulltext", "arxiv_meta"
    ]


# --- node ------------------------------------------------------------------


def test_node_writes_plan_into_state():
    update = planner_node(planner())({"question": QUESTION})
    assert len(update["plan"]) == 3
    assert update["plan"][0]["id"] == "sq1"


def test_node_passes_premise_check_through():
    provider = FakeProvider(THREE)
    node = planner_node(Planner(CFG, provider=provider))
    node({"question": QUESTION,
          "premise_check": {"ran": True, "failed": False,
                            "presuppositions": [
                                {"text": "X exists", "verdict": "SUPPORTED"}]}})
    assert "X exists" in provider.calls[0]["system"]