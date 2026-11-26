"""C32: the eval runner applies every rule the open table accumulated.

A fake run function stands in for the graph, so each rule is tested on its
own: skip, resume, re-run, stop on quota, one configuration per directory,
the held-out guard, pinning and isolation.
"""
import json
from types import SimpleNamespace

import pytest

from src.config import Config
from src.eval.configs import CONFIGS, EVAL_PINS
from src.eval.runner import EvalRefused, EvalRunner

BASE = Config({"llm": {"cache": {"dir": "data/cache/llm"}},
               "run": {"checkpoint_db": "data/checkpoints.sqlite", "trace_dir": "traces"}})


class FakeRun:
    """Records each call; returns ANSWER unless told otherwise."""

    def __init__(self, outcomes=None, interrupt_on=None):
        self.calls = []
        self.outcomes = outcomes or {}
        self.interrupt_on = interrupt_on

    def __call__(self, question, *, config, thread_id, trace_id, resume):
        self.calls.append({"question": question, "thread": thread_id,
                           "trace": trace_id, "resume": resume, "config": config})
        if self.interrupt_on and self.interrupt_on in thread_id:
            self.interrupt_on = None
            raise KeyboardInterrupt
        decision, rule, error = self.outcomes.get(question, ("ANSWER", "threshold", None))
        return SimpleNamespace(
            state={"terminal_decision": decision, "final_report": "report",
                   "decision": {"rule": rule, "score": 0.9, "calibrated": False,
                                "rationale": error or ""}},
            budget=None, resumed="resumed" if len(self.calls) > 1 else "",
            node_visits=["decider", "finalizer"], trace_path="t.jsonl", error=error)


def runner(tmp_path, run, name="agent", split="calibration", **kw):
    return EvalRunner(name, split, tmp_path, base_config=BASE, run_fn=run, **kw)


def first_questions(tmp_path, n):
    r = runner(tmp_path, FakeRun())
    return [q["question"] for q in r.questions[:n]]


# --- skip, resume, re-run ----------------------------------------------------


def test_completed_questions_are_skipped_on_the_next_invocation(tmp_path):
    run = FakeRun()
    s1 = runner(tmp_path, run).run(limit=2)
    assert len(s1.done) == 2 and len(run.calls) == 2
    s2 = runner(tmp_path, run).run(limit=2)
    assert len(s2.skipped) == 2
    assert len(s2.done) == 2 and len(run.calls) == 4   # the NEXT two, not the first two again


def test_an_interrupted_question_is_resumed_on_the_same_thread(tmp_path):
    """No result file was written, so the next invocation passes the same
    thread with resume=True — run_research then continues from the
    checkpoint instead of starting over."""
    first_id = runner(tmp_path, FakeRun()).questions[0]["id"]
    run = FakeRun(interrupt_on=f":{first_id}:")
    with pytest.raises(KeyboardInterrupt):
        runner(tmp_path, run).run(limit=1)
    runner(tmp_path, run).run(limit=1)
    assert run.calls[0]["thread"] == run.calls[1]["thread"]
    assert run.calls[1]["resume"] is True


def test_a_system_failure_is_re_run_on_a_new_thread_not_resumed(tmp_path):
    """Its graph reached FINISH; resuming it would replay the failure."""
    q = first_questions(tmp_path, 1)[0]
    run = FakeRun(outcomes={q: ("ABSTAIN", "system_failure", "critic failed: 500")})
    s1 = runner(tmp_path, run).run(limit=1)
    assert len(s1.failed) == 1
    run.outcomes = {}
    s2 = runner(tmp_path, run).run(limit=1)
    assert len(s2.rerun) == 1
    assert run.calls[0]["thread"].endswith(":a0") and run.calls[1]["thread"].endswith(":a1")


def test_each_invocation_gets_its_own_trace_id(tmp_path):
    """The tracer truncates a file whose id it is given; a resume must not
    erase the crashed attempt's trace."""
    first_id = runner(tmp_path, FakeRun()).questions[0]["id"]
    clock = iter([1000.0, 2000.0])
    run = FakeRun(interrupt_on=f":{first_id}:")
    with pytest.raises(KeyboardInterrupt):
        EvalRunner("agent", "calibration", tmp_path, base_config=BASE, run_fn=run,
                   clock=lambda: next(clock)).run(limit=1)
    EvalRunner("agent", "calibration", tmp_path, base_config=BASE, run_fn=run,
               clock=lambda: next(clock)).run(limit=1)
    assert run.calls[0]["trace"] != run.calls[1]["trace"]


# --- stopping and refusing ---------------------------------------------------


def test_a_day_quota_failure_stops_the_batch(tmp_path):
    q = first_questions(tmp_path, 1)[0]
    run = FakeRun(outcomes={q: ("ABSTAIN", "system_failure",
                                "role 'judgment': every model in the chain failed or is out of quota")})
    s = runner(tmp_path, run).run(limit=5)
    assert s.stopped_on_quota and len(run.calls) == 1


def test_an_ordinary_failure_does_not_stop_the_batch(tmp_path):
    q = first_questions(tmp_path, 1)[0]
    run = FakeRun(outcomes={q: ("ABSTAIN", "system_failure", "critic failed: 500 INTERNAL")})
    s = runner(tmp_path, run).run(limit=3)
    assert not s.stopped_on_quota and len(run.calls) == 3


def test_a_directory_started_under_other_settings_is_refused(tmp_path):
    runner(tmp_path, FakeRun()).run(limit=1)
    manifest = tmp_path / "agent" / "calibration" / "manifest.json"
    data = json.loads(manifest.read_text())
    data["pins"] = {"llm.pin.judgment": "some-other-model"}
    manifest.write_text(json.dumps(data))
    with pytest.raises(EvalRefused, match="different settings"):
        runner(tmp_path, FakeRun()).run(limit=1)


def test_the_test_split_needs_explicit_confirmation(tmp_path):
    with pytest.raises(EvalRefused, match="held-out"):
        runner(tmp_path, FakeRun(), split="test")


def test_the_test_split_forces_the_calibrated_scorer(tmp_path):
    r = runner(tmp_path, FakeRun(), split="test", held_out=True)
    assert r.cfg.get("sufficiency.scorer") == "calibrated"
    assert runner(tmp_path, FakeRun()).cfg.get("sufficiency.scorer") == "auto"


def test_unbuilt_baselines_are_refused_by_name(tmp_path):
    for name in ("B0", "B3"):
        with pytest.raises(EvalRefused, match="C34"):
            runner(tmp_path, FakeRun(), name=name)


def test_an_unfrozen_split_is_refused(tmp_path):
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps({"questions": []}))
    with pytest.raises(EvalRefused, match="not frozen"):
        EvalRunner("agent", "calibration", tmp_path, base_config=BASE,
                   run_fn=FakeRun(), questions_path=questions)


# --- what each run is configured with ----------------------------------------


def test_every_role_is_pinned_and_working_state_is_isolated(tmp_path):
    cfg = runner(tmp_path, FakeRun()).cfg
    for key, model in EVAL_PINS.items():
        assert cfg.get(key) == model
    assert cfg.get("llm.cache.dir") == str(tmp_path / "cache")
    assert cfg.get("llm.cache.dir") != BASE.get("llm.cache.dir")   # never the dev cache
    assert cfg.get("run.checkpoint_db") == str(tmp_path / "checkpoints.sqlite")


def test_configurations_differ_only_in_their_switches(tmp_path):
    a = runner(tmp_path, FakeRun()).cfg
    b2 = runner(tmp_path, FakeRun(), name="B2").cfg
    assert a.get("agency.d5_terminal_decision", True) is True
    assert b2.get("agency.d5_terminal_decision") is False
    assert b2.get("agency.skip_research_on_failed_premise") is False
    assert b2.get("llm.pin.judgment") == a.get("llm.pin.judgment")


def test_b1_freezes_research_agency_but_keeps_d5():
    b1 = CONFIGS["B1"].overrides
    assert all(b1[k] is False for k in b1)
    assert "agency.d5_terminal_decision" not in b1


def test_each_ablation_freezes_exactly_one_surface():
    for name in ("A1", "A2", "A3", "A4", "A5"):
        assert len(CONFIGS[name].overrides) == 1


def test_the_result_file_records_what_the_metrics_need(tmp_path):
    runner(tmp_path, FakeRun()).run(limit=1)
    files = [p for p in (tmp_path / "agent" / "calibration").glob("*.json")
             if p.name != "manifest.json"]
    record = json.loads(files[0].read_text())
    for key in ("terminal_decision", "rule", "score", "calibrated", "decision",
                "final_report", "tier", "split", "attempt", "thread_id"):
        assert key in record
