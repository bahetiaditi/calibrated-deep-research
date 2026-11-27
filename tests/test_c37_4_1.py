"""C37.4.1: a run whose LLM chains gave up is not a measurement.

Live (A09, calibration a2, run after the day's quota was spent): the
critic's C37.4 re-ask and the router both hit the day quota. Each degraded
gracefully, as designed — so nothing reached the run, and the record was
saved as a normal ANSWER with five claims "no verdict returned". The tests
follow the failure end to end with the real provider: exhaustion in a node
→ trace note → RunResult → eval record → failure, quota stop, re-run.
"""
import json
from types import SimpleNamespace

import pytest

from src.config import Config
from src.eval import metrics as M
from src.eval.runner import EvalRunner
from src.graph import Components, run_research
from src.llm.provider import CHAIN_EXHAUSTED
from src.llm.rate_limit import QuotaExhausted
from src.state import new_sub_question
from tests.test_provider import FakeStatusError, build

DAY_429 = ("Rate limit reached for model on tokens per day (TPD): Limit 200000, "
           "Used 199000, Requested 6917.")


class Critic:
    """A node helper with a provider, as the real nodes have; `_attach`
    binds the run's tracer onto both."""
    tracer = None
    budget = None

    def __init__(self, provider):
        self.provider = provider

    def review(self):
        try:
            self.provider.complete("s", "u")
        except QuotaExhausted:
            return "degraded"           # what the real critic and router do
        return "ok"


def components(critic):
    def premise(state):
        return {"premise_check": {"ran": True, "failed": False}}

    def planner(state):
        return {"plan": [new_sub_question("what?", sub_question_id="sq1")]}

    def retriever(state):
        sq = dict(state["plan"][0]); sq["status"] = "sufficient"; sq["evidence_ids"] = ["P1"]
        p = {"id": "P1", "source_type": "arxiv", "source_id": "x", "source_domain": "x",
             "title": "t", "text": "e", "section": None, "published": None,
             "sub_question_id": "sq1", "retrieval_score": 0.5, "rerank_score": 5.0}
        return {"plan": [sq], "evidence": [p]}

    def synthesizer(state):
        critic.review()
        return {"draft": "a draft [P1]", "draft_mode": "answer"}

    return Components(premise_checker=premise, planner=planner,
                      retriever=retriever, synthesizer=synthesizer)


def cfg(tmp_path):
    return Config({
        "run": {"checkpoint_db": str(tmp_path / "ckpt.sqlite"),
                "trace_dir": str(tmp_path / "traces")},
        "budget": {"llm_calls_max": 25, "retrieval_calls_max": 20,
                   "wall_clock_max_s": 900, "max_rounds_per_subquestion": 3},
        "agency": {"d1_plan_revision": False},
    })


def run_with(tmp_path, script):
    provider, backend, _, _ = build(tmp_path)
    backend.script = list(script)
    critic = Critic(provider)
    return run_research("q", components=components(critic), config=cfg(tmp_path),
                        thread_id="t1")


# --- provider → trace → RunResult --------------------------------------------


def test_an_exhausted_chain_inside_a_node_reaches_the_run_result(tmp_path):
    result = run_with(tmp_path, [FakeStatusError(429, DAY_429)] * 2)
    assert result.error is None                   # the node degraded, as designed
    (d,) = result.degraded
    assert d["node"] == "synthesizer" and d["role"] == "judgment"
    assert d["day_quota"] is True


def test_a_non_quota_exhaustion_is_degraded_but_not_a_day_quota(tmp_path):
    result = run_with(tmp_path, [FakeStatusError(500)] * 20)
    (d,) = result.degraded
    assert d["day_quota"] is False


def test_a_clean_run_reports_nothing(tmp_path):
    assert run_with(tmp_path, []).degraded == []


def test_the_provider_notes_before_raising(tmp_path):
    notes = []
    provider, backend, _, _ = build(tmp_path)
    provider.tracer = SimpleNamespace(note=lambda name, **kw: notes.append((name, kw)),
                                      llm_call=lambda **kw: None)
    backend.script = [FakeStatusError(429, DAY_429)] * 2
    with pytest.raises(QuotaExhausted):
        provider.complete("s", "u")
    assert [n for n, _ in notes] == [CHAIN_EXHAUSTED]
    assert notes[0][1]["models"] == ["primary", "secondary"]


# --- RunResult → record → runner and metrics -----------------------------------


BASE = Config({"llm": {"cache": {"dir": "data/cache/llm"}},
               "run": {"checkpoint_db": "data/checkpoints.sqlite", "trace_dir": "traces"}})


class FakeRun:
    def __init__(self, degraded):
        self.degraded, self.calls = degraded, []

    def __call__(self, question, *, config, thread_id, trace_id, resume):
        self.calls.append(thread_id)
        return SimpleNamespace(
            state={"terminal_decision": "ANSWER", "final_report": "r",
                   "decision": {"rule": "threshold", "score": 1.0, "calibrated": False}},
            budget=None, resumed="", node_visits=[], trace_path="", error=None,
            degraded=self.degraded.pop(0) if self.degraded else [])


def make(tmp_path, run):
    return EvalRunner("agent", "calibration", tmp_path, base_config=BASE, run_fn=run)


DAY = [{"node": "critic", "role": "judgment", "day_quota": True, "error": "x"}]
OTHER = [{"node": "critic_router", "role": "judgment", "day_quota": False, "error": "500"}]


def test_a_quota_degraded_run_is_a_failure_and_stops_the_batch(tmp_path):
    """The A09 a2 shape: decision ANSWER, rule threshold, error None."""
    run = FakeRun([DAY])
    s = make(tmp_path, run).run(limit=3)
    assert len(run.calls) == 1                     # stopped, as on any day-quota hit
    assert s.failed == [s.stopped_on_quota] and s.done == []
    record = json.loads((tmp_path / "agent" / "calibration" / f"{s.failed[0]}.json").read_text())
    assert record["infra_failures"] == DAY


def test_the_next_invocation_re_runs_it_on_a_new_thread(tmp_path):
    run = FakeRun([DAY, []])
    qid = make(tmp_path, run).run(limit=1).failed[0]
    s = make(tmp_path, run).run(limit=1)
    assert s.rerun == [qid] and s.done == [qid]
    assert run.calls == [f"agent:calibration:{qid}:a0", f"agent:calibration:{qid}:a1"]


def test_a_non_quota_degradation_fails_the_question_but_the_batch_goes_on(tmp_path):
    run = FakeRun([OTHER, [], []])
    s = make(tmp_path, run).run(limit=3)
    assert len(s.failed) == 1 and len(s.done) == 2 and not s.stopped_on_quota


def test_a_degraded_record_is_graded_as_a_failure_not_an_answer():
    record = {"id": "A09", "tier": "A", "terminal_decision": "ANSWER", "rule": "threshold",
              "score": 1.0, "infra_failures": DAY, "budget": {}}
    o = M.grade(record, {"coverage_delivered": ["PRESENT"], "coverage_answer": ["PRESENT"]})
    assert o.kind == "failure" and not o.correct
    s = M.summarise([o])
    assert s["failures"] == 1 and s["ac_accuracy_answered"] is None and s["aurc"] is None
