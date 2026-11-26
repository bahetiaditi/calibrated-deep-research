"""The evaluation's configurations, as config overrides (§6.4, §6.5).

Every configuration is the same graph with switches changed, so "the full
agent versus X" differs in exactly the switches listed here — nothing else.

Two decisions recorded here rather than buried:

**B1 keeps D5.** §6.4 describes B1 as "every adaptive decision frozen". D5
is the decision layer the whole project studies; freezing it too would make
B1 a second always-answer baseline and confound "value of agency" with
"value of abstention", which B2 already isolates. So B1 freezes D1-D4 and
the five-way critic routing, and keeps D5 live.

**B2 also turns off skip-research-on-refuted-premise** (C27 note). With it
on, B2 "answers" a refuted premise with no draft at all — not an answer, a
hole. Always-answer means the system researches and answers.

B0 (single-shot) and B3 (verbalized confidence) need components that do not
exist yet; they are declared so the runner can name them, and refused until
C34 builds them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalConfig:
    name: str
    isolates: str
    overrides: dict[str, Any] = field(default_factory=dict)
    available: bool = True
    note: str = ""


_D1_D4_AND_ROUTING_OFF = {
    "agency.d1_plan_revision": False,
    "agency.d2_adaptive_route_and_query": False,
    "agency.d3_adaptive_depth_and_stopping": False,
    "agency.d4_budget_allocation": False,
    "agency.critic_five_way_routing": False,
}

CONFIGS: dict[str, EvalConfig] = {c.name: c for c in [
    EvalConfig("agent", "the full system"),
    EvalConfig("B0", "value of the entire system (single-shot LLM + retrieval)",
               available=False, note="built at C34"),
    EvalConfig("B1", "value of agency (D1-D4 and five-way routing frozen; D5 live)",
               dict(_D1_D4_AND_ROUTING_OFF)),
    EvalConfig("B2", "value of the abstention layer (always answer)",
               {"agency.d5_terminal_decision": False,
                "agency.skip_research_on_failed_premise": False}),
    EvalConfig("B3", "value of calibration (verbalized confidence gates D5)",
               available=False, note="built at C34"),
    EvalConfig("A1", "D1 plan revision", {"agency.d1_plan_revision": False}),
    EvalConfig("A2", "D2 route and query", {"agency.d2_adaptive_route_and_query": False}),
    EvalConfig("A3", "D3 depth and stopping", {"agency.d3_adaptive_depth_and_stopping": False}),
    EvalConfig("A4", "D4 budget allocation", {"agency.d4_budget_allocation": False}),
    EvalConfig("A5", "five-way critic routing", {"agency.critic_five_way_routing": False}),
]}

BASELINES = ["B0", "B1", "B2", "B3"]
ABLATIONS = ["A1", "A2", "A3", "A4", "A5"]

# Eval mode pins every role (C27.6, C27.7): one model per role for every
# configuration, so configurations differ only in their switches.
EVAL_PINS = {
    "llm.pin.judgment": "openai/gpt-oss-120b",
    "llm.pin.mechanical": "openai/gpt-oss-20b",
}
