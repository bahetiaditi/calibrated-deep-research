"""Tests for budget allocation (D4).

The property that matters is that value is NOT monotone in how badly a
sub-question is doing. An allocator that simply funds the weakest one is
throwing budget at the corpus gaps; one that funds the strongest is spending
on questions already answered. Value peaks in the middle, and these tests pin
that shape.
"""
import pytest

from src.budget import Budget, UniformAllocator
from src.config import Config
from src.state import new_sub_question
from src.sufficiency.allocation import (
    MIN_VALUE,
    MarginalValueAllocator,
    get_allocator,
)

CFG = Config({
    "agency": {"d4_budget_allocation": True},
    "retrieval": {"adaptive": {"sufficient_min_rerank": 4.0,
                               "hopeless_max_rerank": 0.0}},
})


def sq(sqid, *, status="pending", rounds=0, f1=None):
    s = new_sub_question(f"sub {sqid}", sub_question_id=sqid)
    s["status"] = status
    s["rounds"] = rounds
    s["sufficiency_features"] = {"f1_max_rerank": f1}
    return s


def allocator(cfg=CFG, **kw):
    return MarginalValueAllocator(cfg, **kw)


def budget(**over):
    kwargs = dict(llm_calls_max=25, retrieval_calls_max=20,
                  wall_clock_max_s=900, reserve_fraction=0.2,
                  min_calls_per_subquestion=1)
    kwargs.update(over)
    return Budget(**kwargs)


# --- the shape of value ----------------------------------------------------


def test_untried_sub_questions_are_funded_first():
    """No evidence exists because nobody looked, so any score-based estimate
    would be a guess dressed as a measurement."""
    assert allocator().value(sq("sq1")) == 1.0


def test_sufficient_sub_questions_have_no_value():
    assert allocator().value(sq("sq1", status="sufficient", rounds=1, f1=8.0)) == 0.0


def test_abandoned_sub_questions_have_no_value():
    assert allocator().value(sq("sq1", status="abandoned", rounds=2, f1=-2.0)) == 0.0


def test_close_to_the_bar_beats_far_below_it():
    """The core of D4: spend where one more round plausibly closes it."""
    a = allocator()
    close = a.value(sq("sq1", status="in_progress", rounds=1, f1=3.5))
    far = a.value(sq("sq2", status="in_progress", rounds=1, f1=-2.0))
    assert close > far


def test_value_is_not_monotone_in_how_badly_it_is_doing():
    """An allocator that funds the weakest throws budget at corpus gaps; one
    that funds the strongest spends on questions already answered."""
    a = allocator()
    answered = a.value(sq("a", status="sufficient", rounds=1, f1=8.0))
    close = a.value(sq("b", status="in_progress", rounds=1, f1=3.5))
    hopeless = a.value(sq("c", status="in_progress", rounds=1, f1=-2.0))
    assert close > answered
    assert close > hopeless


def test_rounds_already_spent_lower_the_estimate():
    """f8's rationale used as a decision: a sub-question on its third round
    at f1 1.0 is a worse bet than one on its first at f1 1.0, though their
    evidence is identical."""
    a = allocator()
    fresh = a.value(sq("sq1", status="in_progress", rounds=1, f1=1.0))
    stale = a.value(sq("sq2", status="in_progress", rounds=3, f1=1.0))
    assert fresh > stale


def test_retrieving_nothing_is_worse_than_a_low_score():
    """A low score at least means the route found the right neighbourhood."""
    a = allocator()
    nothing = a.value(sq("sq1", status="in_progress", rounds=1, f1=None))
    low = a.value(sq("sq2", status="in_progress", rounds=1, f1=2.0))
    assert nothing < low


def test_nothing_is_ever_certainly_hopeless():
    v = allocator().value(sq("sq1", status="in_progress", rounds=5, f1=-10.0))
    assert v >= MIN_VALUE


# --- allocation ------------------------------------------------------------


def test_every_untried_sub_question_gets_a_first_attempt():
    """A sub-question starved before it has been tried is pre-judged, not
    allocated — and it is exactly the one whose difficulty the planner
    failed to anticipate."""
    b = budget()
    allocation = allocator().allocate(b, [sq(f"sq{i}") for i in range(5)])
    assert all(v >= 1 for v in allocation.values())


def test_allocation_sums_to_the_pool():
    for n in range(1, 7):
        b = budget()
        allocation = allocator().allocate(
            b, [sq(f"sq{i}", status="in_progress", rounds=1, f1=2.0)
                for i in range(n)]
        )
        assert sum(allocation.values()) == b.allocatable(), f"n={n}"


def test_budget_moves_away_from_a_hopeless_sub_question():
    """The C22 acceptance criterion."""
    b = budget()
    allocation = allocator().allocate(b, [
        sq("promising", status="in_progress", rounds=1, f1=3.5),
        sq("hopeless", status="in_progress", rounds=3, f1=-2.0),
    ])
    assert allocation["promising"] > allocation["hopeless"]


def test_a_sufficient_sub_question_is_not_funded():
    b = budget()
    allocation = allocator().allocate(b, [
        sq("done", status="sufficient", rounds=1, f1=8.0),
        sq("open", status="in_progress", rounds=1, f1=3.0),
    ])
    assert allocation["done"] == 0
    assert allocation["open"] == b.allocatable()


def test_all_terminal_allocates_nothing():
    b = budget()
    allocation = allocator().allocate(b, [
        sq("a", status="sufficient", rounds=1, f1=8.0),
        sq("b", status="abandoned", rounds=2, f1=-1.0),
    ])
    assert sum(allocation.values()) == 0


def test_exhausted_budget_allocates_zero():
    b = budget()
    b.spend("llm", b.llm_calls_max)
    allocation = allocator().allocate(b, [sq("sq1"), sq("sq2")])
    assert allocation == {"sq1": 0, "sq2": 0}


def test_tiny_budget_still_funds_first_attempts():
    b = budget(llm_calls_max=5, reserve_fraction=0.2)   # allocatable 4
    allocation = allocator().allocate(b, [sq(f"sq{i}") for i in range(6)])
    assert sum(allocation.values()) == b.allocatable()


def test_no_sub_questions():
    assert allocator().allocate(budget(), []) == {}


def test_allocation_is_written_back_to_the_budget():
    b = budget()
    allocator().allocate(b, [sq("sq1"), sq("sq2")])
    assert set(b.allocation) == {"sq1", "sq2"}


# --- choosing what to work on next ----------------------------------------


def test_next_picks_the_highest_value_open_sub_question():
    """§3.3's routing table says 'pick highest-value SQ per D4'. Without
    this the retriever takes the first in plan order and the allocation is a
    number nobody reads."""
    chosen = allocator().next_sub_question([
        sq("hopeless", status="in_progress", rounds=3, f1=-2.0),
        sq("promising", status="in_progress", rounds=1, f1=3.5),
    ])
    assert chosen == "promising"


def test_next_prefers_an_untried_sub_question():
    chosen = allocator().next_sub_question([
        sq("tried", status="in_progress", rounds=1, f1=3.5),
        sq("untried"),
    ])
    assert chosen == "untried"


def test_next_ignores_terminal_sub_questions():
    chosen = allocator().next_sub_question([
        sq("done", status="sufficient", rounds=1, f1=8.0),
        sq("open", status="in_progress", rounds=1, f1=1.0),
    ])
    assert chosen == "open"


def test_next_returns_none_when_everything_is_terminal():
    assert allocator().next_sub_question([
        sq("a", status="sufficient", rounds=1, f1=8.0),
    ]) is None


def test_next_is_deterministic_on_ties():
    items = [sq("sq2"), sq("sq1")]
    assert len({allocator().next_sub_question(items) for _ in range(5)}) == 1


# --- the A4 ablation -------------------------------------------------------


def test_disabling_d4_restores_the_uniform_split():
    cfg = Config({
        "agency": {"d4_budget_allocation": False},
        "retrieval": {"adaptive": {"sufficient_min_rerank": 4.0,
                                   "hopeless_max_rerank": 0.0}},
    })
    assert isinstance(get_allocator(cfg), UniformAllocator)


def test_enabled_d4_returns_the_marginal_allocator():
    assert isinstance(get_allocator(CFG), MarginalValueAllocator)


def test_uniform_ignores_everything_the_marginal_allocator_reads():
    b = budget()
    allocation = UniformAllocator().allocate(b, [
        sq("promising", status="in_progress", rounds=1, f1=3.5),
        sq("hopeless", status="in_progress", rounds=3, f1=-2.0),
    ])
    assert allocation["promising"] == allocation["hopeless"]


def test_decisions_are_traced_as_adaptive():
    from src.tracing.tracer import DecisionSurface

    events = []

    class FakeTracer:
        def decision(self, surface, **kw):
            events.append((surface, kw))

    allocator(tracer=FakeTracer()).allocate(budget(), [
        sq("a", status="in_progress", rounds=1, f1=3.5),
        sq("b", status="in_progress", rounds=3, f1=-1.0),
    ])
    surface, payload = events[0]
    assert surface is DecisionSurface.D4_BUDGET_ALLOCATION
    assert payload["was_adaptive"] is True
    assert "values" in payload["inputs"]


# --- the node honours the choice ------------------------------------------


def test_retriever_node_asks_the_allocator_which_to_run():
    from src.nodes.retriever import retriever_node

    asked = {}

    class FakeRetrieverImpl:
        def retrieve_one(self, state, sub_question_id):
            asked["id"] = sub_question_id
            return {}

    state = {"plan": [
        sq("hopeless", status="in_progress", rounds=3, f1=-2.0),
        sq("promising", status="in_progress", rounds=1, f1=3.5),
    ]}
    retriever_node(FakeRetrieverImpl(), allocator())(state)
    assert asked["id"] == "promising"


def test_retriever_node_without_an_allocator_takes_plan_order():
    """Which is the static policy, and what A4 restores."""
    from src.nodes.retriever import retriever_node

    asked = {}

    class FakeRetrieverImpl:
        def retrieve_one(self, state, sub_question_id):
            asked["id"] = sub_question_id
            return {}

    state = {"plan": [
        sq("first", status="in_progress", rounds=3, f1=-2.0),
        sq("second", status="in_progress", rounds=1, f1=3.5),
    ]}
    retriever_node(FakeRetrieverImpl())(state)
    assert asked["id"] == "first"


# --- tie-breaking (regression: C22 live run) -------------------------------


def test_ties_break_toward_plan_order_not_id_order():
    """Every untried sub-question scores exactly 1.0, so the FIRST choice of
    a run is always a tie. Sorting on id meant a live run opened by
    retrieving for the hopeless sub-question, purely because 'sq2' sorts
    after 'sq1'. Plan order carries the planner's reasoning about sequence;
    id order carries nothing."""
    assert allocator().next_sub_question([sq("sq1"), sq("sq2")]) == "sq1"


def test_plan_order_wins_even_with_unhelpful_ids():
    assert allocator().next_sub_question([sq("zeta"), sq("alpha")]) == "zeta"


def test_value_still_beats_plan_order():
    chosen = allocator().next_sub_question([
        sq("first", status="in_progress", rounds=3, f1=-2.0),
        sq("second"),
    ])
    assert chosen == "second"