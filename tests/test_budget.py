"""Tests for the per-run budget.

The budget exists so that D4 is a real decision surface: an agent that never
faces scarcity is not allocating, it is just spending. So the properties that
matter are that it can actually run out, that the reserve is untouchable by
allocation, and that allocations sum to exactly the pool.
"""
import pytest

from src.budget import (
    Budget,
    BudgetExhausted,
    UniformAllocator,
    _distribute,
    get_allocator,
)
from src.config import Config

CLOCK = {"t": 1000.0}


def clock():
    return CLOCK["t"]


@pytest.fixture(autouse=True)
def reset_clock():
    CLOCK["t"] = 1000.0


def make(**over):
    kwargs = dict(
        llm_calls_max=25, retrieval_calls_max=20, wall_clock_max_s=900.0,
        reserve_fraction=0.20, min_calls_per_subquestion=1,
        max_rounds_per_subquestion=3, max_plan_revisions=2,
        _started=1000.0, _now=clock,
    )
    kwargs.update(over)
    return Budget(**kwargs)


# --- accounting ------------------------------------------------------------


def test_spend_and_remaining():
    b = make()
    b.spend("llm", 3)
    assert b.llm_calls_used == 3 and b.remaining("llm") == 22


def test_llm_and_retrieval_are_separate_pools():
    b = make()
    b.spend("llm", 25)
    assert not b.can_spend("llm")
    assert b.can_spend("retrieval")


def test_spend_past_cap_raises_cleanly():
    """The C5 acceptance check. This should never fire in correct operation:
    the controller consults can_spend() first. If it fires, a node bypassed
    the check, and that is a bug worth hearing loudly."""
    b = make()
    b.spend("llm", 25)
    with pytest.raises(BudgetExhausted) as exc:
        b.spend("llm", 1)
    assert exc.value.used == 25 and exc.value.cap == 25


def test_failed_spend_does_not_mutate():
    b = make()
    b.spend("llm", 24)
    with pytest.raises(BudgetExhausted):
        b.spend("llm", 5)
    assert b.llm_calls_used == 24


def test_can_spend_is_non_raising():
    b = make()
    b.spend("llm", 25)
    assert b.can_spend("llm") is False      # controller's graceful path


def test_per_sub_question_spend_tracked():
    b = make()
    b.spend("llm", 2, sub_question_id="sq1")
    b.spend("llm", 1, sub_question_id="sq2")
    assert b.spend_by_sub_question == {"sq1": 2, "sq2": 1}


# --- time ------------------------------------------------------------------


def test_wall_clock_stops_spending():
    b = make()
    CLOCK["t"] = 1000.0 + 901
    assert b.out_of_time()
    assert b.can_spend("llm") is False      # even with calls left


# --- the reserve -----------------------------------------------------------


def test_reserve_is_withheld_from_allocation():
    """The critic and decider run after retrieval. A system that spends
    everything researching and cannot afford to decide has failed in the
    most embarrassing way available."""
    b = make()
    assert b.reserved() == 5                # 20% of 25
    assert b.allocatable() == 20


def test_allocatable_shrinks_as_spend_happens():
    b = make()
    b.spend("llm", 8)
    assert b.allocatable() == 12


def test_in_reserve_when_only_the_reserve_remains():
    b = make()
    b.spend("llm", 20)
    assert b.in_reserve()
    assert b.can_spend("llm")               # reserve is still spendable


def test_allocatable_never_negative():
    b = make()
    b.spend("llm", 25)
    assert b.allocatable() == 0


# --- allocation ------------------------------------------------------------


def test_allocation_sums_exactly_to_the_pool():
    """The other half of the C5 check. Integer division would silently lose
    up to n-1 calls, which at a 25-call budget is material."""
    for n in range(1, 8):
        b = make()
        alloc = UniformAllocator().allocate(b, [f"sq{i}" for i in range(n)])
        assert sum(alloc.values()) == b.allocatable(), f"n={n}"


def test_uniform_split_is_even_with_remainder_distributed():
    b = make()                              # allocatable 20
    alloc = UniformAllocator().allocate(b, ["a", "b", "c"])
    assert sorted(alloc.values()) == [6, 7, 7]


def test_allocation_written_back_to_budget():
    b = make()
    UniformAllocator().allocate(b, ["a", "b"])
    assert b.allocation == {"a": 10, "b": 10}


def test_scarce_pool_funds_some_rather_than_starving_all():
    b = make(llm_calls_max=5, reserve_fraction=0.2)   # allocatable 4
    alloc = _distribute(4, ["a", "b", "c", "d", "e", "f"], minimum=2)
    assert sum(alloc.values()) == 4
    assert sorted(alloc.values(), reverse=True)[:2] == [2, 2]


def test_zero_pool_allocates_zero_to_each():
    b = make()
    b.spend("llm", 25)
    alloc = UniformAllocator().allocate(b, ["a", "b"])
    assert alloc == {"a": 0, "b": 0}


def test_no_sub_questions_allocates_nothing():
    assert UniformAllocator().allocate(make(), []) == {}


def test_accepts_dicts_and_objects_as_sub_questions():
    b = make()
    alloc = UniformAllocator().allocate(b, [{"id": "sq1"}, {"id": "sq2"}])
    assert set(alloc) == {"sq1", "sq2"}


# --- D4 wiring -------------------------------------------------------------


def test_uniform_allocation_is_logged_as_non_adaptive():
    """A4 freezes D4. The trace must show zero ADAPTIVE D4 decisions, or
    C35 cannot verify the ablation took effect."""
    from src.tracing.tracer import DecisionSurface

    events = []

    class FakeTracer:
        def decision(self, surface, **kw):
            events.append((surface, kw))

        def note(self, *a, **k):
            pass

    b = make(tracer=FakeTracer())
    UniformAllocator().allocate(b, ["a", "b"])
    surface, kw = events[0]
    assert surface is DecisionSurface.D4_BUDGET_ALLOCATION
    assert kw["was_adaptive"] is False


def test_get_allocator_returns_uniform_when_d4_disabled():
    cfg = Config({"agency": {"d4_budget_allocation": False}})
    assert isinstance(get_allocator(cfg), UniformAllocator)


# --- plan revisions --------------------------------------------------------


def test_plan_revision_ceiling():
    b = make()
    assert b.can_revise_plan()
    b.record_plan_revision()
    b.record_plan_revision()
    assert not b.can_revise_plan()          # D1 has a hard stop


# --- serialisation ---------------------------------------------------------


def test_to_state_matches_budget_state_schema():
    b = make()
    b.spend("llm", 2)
    b.spend("retrieval", 3)
    UniformAllocator().allocate(b, ["a"])
    state = b.to_state()
    assert set(state) == {
        "llm_calls_used", "llm_calls_max",
        "retrieval_calls_used", "retrieval_calls_max", "allocation",
    }
    assert state["llm_calls_used"] == 2 and state["retrieval_calls_used"] == 3


def test_fraction_consumed_feeds_run_level_features():
    b = make()
    b.spend("llm", 5)
    b.spend("retrieval", 4)
    assert b.fraction_consumed() == pytest.approx(9 / 45)


def test_from_real_config():
    from src.config import get_config

    b = Budget.from_config(get_config(reload=True))
    assert b.llm_calls_max > 0 and 0 <= b.reserve_fraction < 1
    assert b.allocatable() < b.llm_calls_max      # reserve is real


def test_report_renders():
    assert "llm" in make().report()