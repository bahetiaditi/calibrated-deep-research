"""C34: baselines B0 (single shot), B2 and B3 (derived from the agent's runs)."""
import json
from types import SimpleNamespace

from src.config import Config
from src.eval import baselines as bl
from benchmark.derive_baselines import derive

ANSWER_DRAFT = "FlashAttention-2 partitions work across thread blocks. [P1]"


def rec(qid, decision, rule, *, draft=ANSWER_DRAFT, draft_mode="answer",
        withheld="", withheld_mode="", tier="A", subtype=None):
    return {"id": qid, "tier": tier, "subtype": subtype, "split": "calibration",
            "terminal_decision": decision, "rule": rule, "thread_id": f"agent:x:{qid}:a0",
            "draft": draft, "draft_mode": draft_mode,
            "decision": {"rule": rule, "withheld_draft": withheld,
                         "withheld_draft_mode": withheld_mode},
            "question": f"question {qid}", "premise_check": {}}


# --- which draft B2/B3 rest on ---------------------------------------------


def test_an_answered_question_uses_its_final_draft():
    assert bl.answer_draft(rec("A", "ANSWER", "threshold")) == ANSWER_DRAFT


def test_an_abstention_uses_the_withheld_answer_draft():
    r = rec("A", "ABSTAIN", "threshold", draft="", draft_mode="answer",
            withheld=ANSWER_DRAFT, withheld_mode="answer")
    assert bl.answer_draft(r) == ANSWER_DRAFT


def test_a_partial_rewrite_uses_the_answer_draft_not_the_rewrite():
    """B2 always answers: its output is what the agent wrote BEFORE deciding."""
    r = rec("A", "PARTIAL", "threshold", draft="partial rewrite", draft_mode="partial",
            withheld=ANSWER_DRAFT, withheld_mode="answer")
    assert bl.answer_draft(r) == ANSWER_DRAFT


def test_a_refuted_premise_has_no_draft_and_needs_a_live_b2():
    r = rec("D", "ABSTAIN", "premise_refuted", draft="", draft_mode="")
    assert bl.answer_draft(r) == ""
    assert bl.needs_live_b2(r)


def test_records_from_before_c34_fall_back_to_the_checkpoint_state():
    r = rec("A", "ANSWER", "threshold"); del r["draft"]; del r["draft_mode"]
    assert bl.answer_draft(r, {"draft": ANSWER_DRAFT, "draft_mode": "answer"}) == ANSWER_DRAFT


# --- B3's record ------------------------------------------------------------


def test_b3_keeps_the_agents_hard_rules():
    """B3 swaps the scorer, not the hard rules (§6.4)."""
    r = bl.b3_record(rec("D", "ABSTAIN", "premise_refuted", draft=""), "", None, "")
    assert (r["terminal_decision"], r["rule"]) == ("ABSTAIN", "premise_refuted")


def test_b3_leaves_threshold_decisions_to_scoring_time():
    """Its tau is chosen on the calibration split, like the calibrated model's."""
    r = bl.b3_record(rec("A", "ANSWER", "threshold"), ANSWER_DRAFT, 0.8, "sure")
    assert r["terminal_decision"] is None and r["rule"] == "threshold_pending"
    assert r["score"] == 0.8


class Confident:
    def __init__(self, value=80):
        self.value, self.calls = value, 0

    def complete_json(self, system, user, **kw):
        self.calls += 1
        return {"confidence": self.value, "reason": "looks right"}, None


def test_verbalized_confidence_is_scaled_to_a_probability():
    assert bl.verbalized_confidence("q", "a", Confident(73))[0] == 0.73
    assert bl.verbalized_confidence("q", "a", Confident("lots"))[0] is None


# --- the derivation ---------------------------------------------------------


def setup_agent(tmp_path):
    d = tmp_path / "agent" / "calibration"; d.mkdir(parents=True)
    records = [
        rec("A01", "ANSWER", "threshold"),
        rec("A02", "ABSTAIN", "threshold", draft="", withheld=ANSWER_DRAFT, withheld_mode="answer"),
        rec("D101", "ABSTAIN", "premise_refuted", draft="", draft_mode="", tier="D", subtype="D1"),
        rec("B01", "ABSTAIN", "system_failure", draft=""),
    ]
    for r in records:
        (d / f"{r['id']}.json").write_text(json.dumps(r))
    return records


def test_derivation_writes_paired_b2_and_b3_records(tmp_path):
    setup_agent(tmp_path)
    provider, live = Confident(), []
    s = derive("calibration", tmp_path, provider=provider, live_runner=live.extend,
               base_config=Config({}))
    b2 = json.loads((tmp_path / "B2" / "calibration" / "A02.json").read_text())
    # Paired: B2's output IS the agent's own draft, byte for byte.
    assert b2["draft"] == ANSWER_DRAFT and b2["terminal_decision"] == "ANSWER"
    assert s["b2"] == 2
    assert live == ["D101"]                       # no draft: run live
    assert s["b3"] == 3 and provider.calls == 2   # the refuted premise needs no call
    assert not (tmp_path / "B3" / "calibration" / "B01.json").exists()  # failures excluded


def test_derivation_is_idempotent(tmp_path):
    setup_agent(tmp_path)
    provider = Confident()
    derive("calibration", tmp_path, provider=provider, base_config=Config({}))
    calls = provider.calls
    s = derive("calibration", tmp_path, provider=provider, base_config=Config({}))
    assert provider.calls == calls and s["b3"] == 0


# --- B0 ---------------------------------------------------------------------


class Hybrid:
    def retrieve(self, q, top_k):
        return [SimpleNamespace(passage={"id": f"P{i}", "text": f"evidence {i}", "source_id": f"s{i}",
                                         "source_type": "arxiv", "source_domain": "x", "title": "t",
                                         "section": None, "published": None, "retrieval_score": 0.5,
                                         "rerank_score": 5.0}) for i in range(3)]


class Reranker:
    def rerank(self, q, results, top_k):
        return results[:top_k]


class Synth:
    def synthesize(self, state):
        assert len(state["evidence"]) == 3
        return SimpleNamespace(text="An answer. [P0]", error=None)


def test_b0_is_one_retrieval_and_one_call():
    result = bl.single_shot_run("what?", config=Config({}), thread_id="t", trace_id="t",
                                retrieval=(Hybrid(), Reranker()), synthesizer=Synth())
    assert result.state["terminal_decision"] == "ANSWER"
    assert "An answer." in result.state["final_report"]
    assert result.node_visits == ["retrieve", "synthesize"]
