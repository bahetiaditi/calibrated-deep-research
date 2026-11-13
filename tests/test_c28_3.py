"""C28.3: a critic that cannot run is an outage, not a verdict.

Live (C28.2 smoke): the critic's single batched call returned an EMPTY
generation under JSON mode on all three judgment models — the output cap
spent on reasoning before any JSON. Every claim became a placeholder
UNSUPPORTED, the heuristic scored 0 of 3, and the run was recorded as an
epistemic ABSTAIN.
"""
import re

from src.config import Config
from src.nodes.critic import MAX_SPLIT_DEPTH, Critic, critic_node
from src.nodes.critic_router import UNRESOLVED, CriticRouter
from src.nodes.decider import Decider
from src.sufficiency.policy import ABSTAIN, RULE_SYSTEM_FAILURE
from src.sufficiency.scorer import HeuristicScorer
from tests.decision_fixtures import STRONG, decision_state
from tests.test_critic import PID, passage, state_with

CFG = Config({})


class CappedProvider:
    """Fails any batch larger than `cap` sentences, as an exhausted output
    budget does; judges smaller batches SUPPORTED."""

    def __init__(self, cap):
        self.cap = cap
        self.batches = []

    def complete_json(self, system, user, **kw):
        indices = [int(i) for i in re.findall(r"^\[(\d+)\] ", system, flags=re.M)]
        self.batches.append(len(indices))
        if len(indices) > self.cap:
            raise RuntimeError("json_validate_failed: failed_generation ''")
        return {"verdicts": [{"index": i, "verdict": "SUPPORTED", "reasoning": "ok"}
                             for i in indices]}, None


def long_draft(n=8):
    return " ".join(f"Claim number {i} about attention kernels is checkable. [{PID[1]}]"
                    for i in range(n))


def review(provider, n=8, **kw):
    state = state_with(long_draft(n), [passage(PID[1])])
    return Critic(CFG, provider=provider, **kw).review(state)


def test_a_failed_batch_is_split_and_every_claim_still_judged():
    provider = CappedProvider(cap=4)
    report = review(provider)
    assert report.error is None
    assert len(report.findings) == 8
    assert all(f["verdict"] == "SUPPORTED" for f in report.findings)
    assert provider.batches == [8, 4, 4]


def test_a_persistent_failure_gives_up_fast():
    """When nothing fits — the model is down, not merely over budget — the
    first leaf to fail at the depth limit ends the review: one call per
    level, not the whole tree."""
    provider = CappedProvider(cap=0)
    report = review(provider)
    assert report.error
    assert provider.batches == [8, 4, 2]
    assert MAX_SPLIT_DEPTH == 2


def test_no_extra_calls_when_the_batch_fits():
    provider = CappedProvider(cap=100)
    review(provider)
    assert provider.batches == [8]


def test_the_split_is_traced():
    notes = []

    class Tracer:
        def note(self, name, **kw):
            notes.append((name, kw))

    review(CappedProvider(cap=4), tracer=Tracer())
    assert any(name == "critic_split" and kw["sentences"] == 8 for name, kw in notes)


def test_the_node_writes_critic_failed_on_every_review():
    failed = critic_node(Critic(CFG, provider=CappedProvider(cap=0)))(
        state_with(long_draft(2), [passage(PID[1])]))
    ok = critic_node(Critic(CFG, provider=CappedProvider(cap=100)))(
        state_with(long_draft(2), [passage(PID[1])]))
    assert failed["critic_failed"] is True and failed["error"]
    assert ok["critic_failed"] is False


def test_the_router_does_not_route_placeholders():
    class NoCalls:
        def complete_json(self, *a, **k):
            raise AssertionError("routing placeholders must not spend a call")

    state = decision_state(STRONG)
    state["critic_failed"] = True
    state["critic_findings"] = [{"claim": "c", "sub_question_id": "sq1",
                                 "cited_passage_ids": [], "verdict": "UNSUPPORTED",
                                 "reasoning": "critic unavailable (RuntimeError)",
                                 "action_taken": ""}]
    result = CriticRouter(CFG, provider=NoCalls()).route(state)
    assert result.findings[0]["action_taken"] == UNRESOLVED
    assert not result.needs_more_work
    assert result.plan[0]["status"] == "sufficient"


def test_the_live_run_is_now_a_system_failure_not_an_abstention():
    """A draft exists, the critic failed, every claim is a placeholder."""
    state = decision_state(STRONG, STRONG, STRONG)
    state["error"] = "role 'judgment': every model in the chain failed"
    state["critic_failed"] = True
    state["critic_findings"] = [{"claim": f"c{i}", "sub_question_id": f"sq{i}",
                                 "cited_passage_ids": [], "verdict": "UNSUPPORTED",
                                 "reasoning": "critic unavailable", "action_taken": UNRESOLVED}
                                for i in (1, 2, 3)]
    result = Decider(CFG, scorer=HeuristicScorer(CFG)).decide(state)
    assert result.decision.terminal == ABSTAIN
    assert result.decision.rule == RULE_SYSTEM_FAILURE
    assert result.update["decision"]["epistemic"] is False


def test_the_report_says_it_could_not_be_verified():
    from src.nodes.finalizer import Finalizer
    from tests.test_finalizer import state as fstate

    s = fstate("ABSTAIN", rule="system_failure", draft="")
    s["decision_rationale"] = ("Abstained because the report could not be verified: "
                               "the critic failed (x). This is a system failure, "
                               "not a judgment about the evidence.")
    text = Finalizer(CFG).finalize(s)
    assert "not a judgment about the evidence" in text
    assert "could not be verified" in text
