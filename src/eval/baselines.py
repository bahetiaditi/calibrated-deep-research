"""Baselines B0, B2, B3 (C34). B1 is a plain configuration of the graph.

  B0  single-shot: one retrieval over the frozen corpus, one LLM call that
      writes the answer with citations. No planner, no critic, no decision —
      "the value of the entire system" (§6.4).

  B2  always answer — DERIVED from the agent's runs: its output is the
      agent's critiqued answer-mode draft, which the decision record keeps
      (withheld when the agent abstained or rewrote it as PARTIAL).
  B3  verbalized confidence — DERIVED: the same draft, with D5's score
      replaced by the judgment model's stated confidence in it. Its threshold
      is chosen on the calibration split exactly as the calibrated model's is.

Why derive rather than re-run. B2 and B3 run the agent's pipeline unchanged
up to D5; only the decision differs. Re-running them would re-sample every
upstream LLM call — and the same question has gone PARTIAL in one run and
ABSTAIN in the next (C27.7) — so "agent vs B2" would mix run-to-run noise
into the comparison it exists to make. Derived, the comparison is PAIRED:
same research, same evidence, same draft; only the decision rule differs,
which is exactly what B2 and B3 isolate. It also saves ~10 quota-days.

The exception is where the agent's research never happened: a refuted
premise skips research (skip_research_on_failed_premise), which B2 turns off.
Those few questions are run live under the B2 configuration.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.config import Config

log = logging.getLogger(__name__)

B0_TOP_K = 10


# --------------------------------------------------------------------------
# B0: single shot
# --------------------------------------------------------------------------


def single_shot_run(question: str, *, config: Config, thread_id: str, trace_id: str,
                    resume: bool = False, retrieval=None, synthesizer=None) -> Any:
    """One retrieval, one synthesis call. Same result shape as run_research."""
    from src.nodes.finalizer import Finalizer
    from src.state import new_state, new_sub_question

    hybrid, reranker = retrieval or _frozen_retrieval(config)
    results = reranker.rerank(question, hybrid.retrieve(question, top_k=B0_TOP_K * 3),
                              top_k=B0_TOP_K)
    passages = [dict(r.passage, sub_question_id="sq1") for r in results]

    state = new_state(question)
    sq = new_sub_question(question, sub_question_id="sq1")
    sq.update({"status": "sufficient", "evidence_ids": [p["id"] for p in passages]})
    state["plan"] = [sq]
    state["evidence"] = passages

    if synthesizer is None:
        from src.nodes.synthesizer import Synthesizer
        synthesizer = Synthesizer(config)
    draft = synthesizer.synthesize(state)
    from src.nodes.synthesizer import synthesis_entry
    state.update({"draft": draft.text, "draft_mode": "answer",
                  "synthesis_log": [synthesis_entry(draft)],
                  "terminal_decision": "ANSWER" if draft.text else "ABSTAIN"})
    rule = "static_always_answer" if draft.text else "system_failure"
    state["decision"] = {"rule": rule, "score": None, "calibrated": False,
                         "rationale": draft.error or "single-shot baseline: always answers"}
    state["final_report"] = Finalizer(config).finalize(state) if draft.text else ""
    return SimpleNamespace(state=state, budget=None, error=draft.error,
                           node_visits=["retrieve", "synthesize"], trace_path="",
                           resumed="")


def _frozen_retrieval(cfg: Config):
    from src.eval.components import frozen_retrieval
    return frozen_retrieval(cfg)


# --------------------------------------------------------------------------
# B2 / B3: derived from the agent's runs
# --------------------------------------------------------------------------


def answer_draft(record: dict, state: dict | None = None) -> str:
    """The agent's critiqued ANSWER-MODE draft for one question, or "".

    ABSTAIN or a PARTIAL rewrite: the decision record's withheld draft.
    ANSWER (or a PARTIAL whose rewrite was unaffordable): the final draft.
    `state` is the checkpointed final state, used for records written before
    C34 started storing the draft.
    """
    decision = record.get("decision") or {}
    if decision.get("withheld_draft") and decision.get("withheld_draft_mode") in ("answer", ""):
        return decision["withheld_draft"]
    source = record if record.get("draft") is not None else (state or {})
    if source.get("draft_mode") == "answer" and source.get("draft"):
        return source["draft"]
    return ""


def needs_live_b2(record: dict) -> bool:
    """No research happened, so there is no draft to derive B2 from."""
    return (record.get("rule") == "premise_refuted"
            and not (record.get("decision") or {}).get("withheld_draft"))


def b2_record(record: dict, draft: str) -> dict:
    return {**_base(record, "B2"), "terminal_decision": "ANSWER",
            "rule": "static_always_answer", "draft": draft, "draft_mode": "answer",
            "derived_from": record.get("thread_id")}


def b3_record(record: dict, draft: str, confidence: float | None, reason: str) -> dict:
    rule = record.get("rule")
    hard = rule in ("premise_refuted", "system_failure", "no_evidence")
    return {**_base(record, "B3"), "draft": draft, "draft_mode": "answer",
            # B3 swaps the SCORER, not the hard rules (§6.4): a refuted
            # premise or a crash decides exactly as it did for the agent.
            "terminal_decision": record.get("terminal_decision") if hard else None,
            "rule": rule if hard else "threshold_pending",
            "score": confidence, "confidence_reason": reason,
            "derived_from": record.get("thread_id")}


def verbalized_confidence(question: str, draft: str, provider) -> tuple[float | None, str]:
    from src.llm.provider import Role
    from src.prompts import render

    parsed, _ = provider.complete_json(
        system="You judge your own confidence. Output only JSON.",
        user=render("verbalized_confidence", question=question, answer=draft),
        role=Role.JUDGMENT, required_keys=["confidence"])
    try:
        value = float(parsed.get("confidence"))
    except (TypeError, ValueError):
        return None, "unparseable confidence"
    return max(0.0, min(1.0, value / 100.0)), str(parsed.get("reason") or "")[:300]


def _base(record: dict, config: str) -> dict:
    keep = ("id", "tier", "subtype", "split", "premise_check")
    return {**{k: record.get(k) for k in keep}, "config": config, "attempt": 0,
            "thread_id": f"{config}:derived:{record.get('id')}", "calibrated": False,
            "decision": {}, "final_report": "", "error": None}


def load_checkpoint_state(db: Path, thread_id: str) -> dict:
    """Final state of a finished agent run, from the eval's checkpoints."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    if not db.exists():
        return {}
    with SqliteSaver.from_conn_string(str(db)) as saver:
        snap = saver.get_tuple({"configurable": {"thread_id": thread_id}})
    return dict(snap.checkpoint.get("channel_values") or {}) if snap else {}
