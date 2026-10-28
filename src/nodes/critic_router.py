"""Five-way critic routing (C24) — §2.3.

> "Five-way routing on a judgment call is the difference between a critic
> and an `if` statement."

C23 decides what is *true*. This decides what to *do*, and the five outcomes
are not interchangeable:

  RE_RETRIEVE  the claim is probably right and the passage was missed
  REVISE_PLAN  the sub-question is malformed; no query will fix it (D1)
  DROP_CLAIM   the evidence says the claim is wrong
  HEDGE        sources genuinely disagree; keep it and say so
  UNRESOLVED   nothing further will help; report the gap (feeds D5)

Why this is not an `if` statement
---------------------------------
The naive version routes on the verdict alone: `UNSUPPORTED → re-retrieve`,
everything else → finalize. But four of the five outcomes above can follow
an UNSUPPORTED verdict, and which one is right depends on *why* it failed —
whether the queries tried were phrased the way sources phrase it, whether
the sub-question is answerable as asked, whether the evidence is silent or
contradictory. None of that is in the verdict.

Deterministic gates, then judgment
----------------------------------
Two things are decided without the model, because they are facts rather than
judgments: a route requiring budget that does not exist is unavailable, and
a sub-question already at its revision ceiling cannot be revised again.
Offering the model options it cannot have invites it to pick one, and then
the code silently substitutes something else — which makes the trace a
record of a decision that was never taken.

The remaining choice is a genuine judgment and goes to the model, once, for
all failed claims together.

Two biases, both toward honesty
-------------------------------
When RE_RETRIEVE and UNRESOLVED are both arguable, the prompt prefers
UNRESOLVED: an admitted gap is a usable result, a third failed retrieval is
spent budget and the same gap. When DROP and HEDGE are both arguable, it
prefers HEDGE: removing a claim removes the reader's chance to judge it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.budget import plan_revisions_remain
from src.config import Config, get_config
from src.llm.provider import LLMProvider, Role
from src.nodes.critic import CONTRADICTED, PARTIAL, SUPPORTED
from src.prompts import render
from src.state import ClaimVerification, ResearchState, find_sub_question

log = logging.getLogger(__name__)

RE_RETRIEVE = "RE_RETRIEVE"
REVISE_PLAN = "REVISE_PLAN"
DROP_CLAIM = "DROP_CLAIM"
HEDGE = "HEDGE"
UNRESOLVED = "UNRESOLVED"
ACCEPT = "ACCEPT"          # SUPPORTED claims need no action

ACTIONS = (RE_RETRIEVE, REVISE_PLAN, DROP_CLAIM, HEDGE, UNRESOLVED)

# Routes that cost further work, and so require budget to be available.
COSTLY = (RE_RETRIEVE, REVISE_PLAN)

MAX_CLAIM_CHARS = 300


@dataclass
class RoutingResult:
    findings: list[ClaimVerification] = field(default_factory=list)
    plan: list[dict[str, Any]] = field(default_factory=list)
    unresolved_gaps: list[str] = field(default_factory=list)
    needs_more_work: bool = False     # re-retrieval or revision was chosen
    error: str | None = None

    def count(self, action: str) -> int:
        return sum(1 for f in self.findings if f["action_taken"] == action)


class CriticRouter:
    def __init__(
        self,
        config: Config | None = None,
        *,
        provider: LLMProvider | None = None,
        tracer: Any = None,
        budget: Any = None,
    ) -> None:
        self.cfg = config or get_config()
        self.provider = provider or LLMProvider(self.cfg, tracer=tracer)
        self.tracer = tracer
        self.budget = budget
        self.five_way = bool(
            self.cfg.get("agency.critic_five_way_routing", True)
        )

    # -- main entry point ---------------------------------------------------

    def route(self, state: ResearchState) -> RoutingResult:
        findings = list(state.get("critic_findings") or [])
        if not findings:
            return RoutingResult(error="no critic findings to route")

        failed = [f for f in findings if f["verdict"] != SUPPORTED]
        routed = {id(f): ACCEPT for f in findings}

        if not failed:
            return self._finish(state, findings, routed)

        if not self.five_way:
            # A5's static counterpart: binary routing. Everything that is
            # not supported gets re-retrieved if there is budget, and is
            # otherwise left alone — which is the conditional edge §2.3
            # describes, not a decision.
            available = self._available_actions(state)
            fallback = RE_RETRIEVE if RE_RETRIEVE in available else UNRESOLVED
            for finding in failed:
                # The ownership gate applies here too: A5 is binary, not
                # blind, and a re-retrieval that reopens nothing is not the
                # static policy's decision, it is no decision (C27.2). The
                # repeated-query gate deliberately does NOT apply — replaying
                # the sub-question is exactly what the static policy does.
                owned = find_sub_question(
                    state, finding.get("sub_question_id") or ""
                ) is not None
                routed[id(finding)] = fallback if owned else UNRESOLVED
            return self._finish(state, findings, routed, adaptive=False)

        try:
            decisions = self._judge(state, failed)
        except Exception as exc:  # noqa: BLE001
            # Routing failure must not discard the verification. Marking
            # everything unresolved keeps the claims visible to D5 rather
            # than silently accepting them.
            log.warning("critic routing failed: %s", exc)
            for finding in failed:
                routed[id(finding)] = UNRESOLVED
            result = self._finish(state, findings, routed)
            result.error = str(exc)
            return result

        available = self._available_actions(state)
        retry: dict[int, str] = {}
        for index, finding in enumerate(failed):
            action, query, reason = decisions.get(
                index, (UNRESOLVED, "", "no routing decision returned")
            )
            if action not in available:
                # The gate already excluded it; falling back silently would
                # make the trace record a decision never taken.
                action, reason = UNRESOLVED, (
                    f"{action} unavailable ({reason})" if reason
                    else f"{action} unavailable"
                )
            blocked = self._claim_gate(state, finding, action, query)
            if blocked:
                action, reason = UNRESOLVED, (
                    f"{blocked} ({reason})" if reason else blocked
                )
            routed[id(finding)] = action
            finding["reasoning"] = (
                f"{finding.get('reasoning', '')} | {reason}".strip(" |")[:400]
            )
            if action == RE_RETRIEVE:
                retry[id(finding)] = query.strip()
                finding["reasoning"] = (
                    f"{finding['reasoning']} | retry: {query}"[:400]
                )

        return self._finish(state, findings, routed, retry=retry)

    def _claim_gate(
        self,
        state: ResearchState,
        finding: ClaimVerification,
        action: str,
        query: str,
    ) -> str:
        """Per-claim facts that make a costly route impossible (C27.2).

        Two, both found in the C27 live traces:

        - A costly route needs a sub-question to act on. A claim the critic
          could not attribute to one cannot be re-retrieved or revised: in
          the live run eight RE_RETRIEVE decisions were recorded and none had
          any effect — the record of a decision never taken, which is the
          defect C24 was written to prevent.
        - RE_RETRIEVE needs a query that has not already been tried. Without
          one the retriever falls back to the sub-question's verbatim text —
          live, the exact query that had just failed, returning the same
          passages and a byte-identical cached draft. That is the retry §2.4
          names as an anti-pattern, at the price of a retrieval call.

        Both resolve to UNRESOLVED, which C24's prompt already prefers when
        RE_RETRIEVE and UNRESOLVED are both arguable.
        """
        if action not in COSTLY:
            return ""
        sub_question = find_sub_question(
            state, finding.get("sub_question_id") or ""
        )
        if sub_question is None:
            return f"{action} unavailable: claim belongs to no sub-question in the plan"
        if action == RE_RETRIEVE:
            tried = {
                q.strip().lower()
                for q in sub_question.get("attempted_queries") or []
            }
            if not query.strip() or query.strip().lower() in tried:
                return "RE_RETRIEVE unavailable: no query not already tried"
        return ""

    # -- gates --------------------------------------------------------------

    def _available_actions(self, state: ResearchState) -> set[str]:
        """Which routes are actually possible right now.

        Facts, not judgments. Offering the model an option it cannot have
        invites it to pick one and have the code substitute something else,
        which makes the trace a record of a decision that was never taken.
        """
        available = {DROP_CLAIM, HEDGE, UNRESOLVED}

        # Once D5 has decided, research is closed (C27). The router still
        # runs on a PARTIAL rewrite — its claims need HEDGE/DROP/UNRESOLVED
        # like any other — but reopening a sub-question would send the run
        # back into retrieval after the decision that ended it, and clearing
        # the draft would loop synthesis -> critic -> router indefinitely.
        if state.get("terminal_decision") is not None:
            return available

        has_budget = self.budget is None or (
            self.budget.can_spend("retrieval") and self.budget.can_spend("llm")
        )
        if has_budget:
            available.add(RE_RETRIEVE)
            can_revise = plan_revisions_remain(
                state, self.budget,
                int(self.cfg.get("budget.max_plan_revisions", 2)),
            )
            if can_revise and self.cfg.get("agency.d1_plan_revision", True):
                available.add(REVISE_PLAN)
        return available

    # -- the single batched call --------------------------------------------

    def _judge(
        self, state: ResearchState, failed: Sequence[ClaimVerification]
    ) -> dict[int, tuple[str, str, str]]:
        claims = "\n\n".join(
            f"[{i}] {f['claim'][:MAX_CLAIM_CHARS]}\n"
            f"      verdict: {f['verdict']}\n"
            f"      why it failed: {f.get('reasoning', '')[:200]}\n"
            f"      sub-question: {f.get('sub_question_id') or '(none)'}"
            for i, f in enumerate(failed)
        )

        history_lines = []
        for sub_question in state.get("plan") or []:
            queries = sub_question.get("attempted_queries") or []
            history_lines.append(
                f"{sub_question['id']}  status={sub_question.get('status')}  "
                f"rounds={sub_question.get('rounds', 0)}\n"
                f"   text: {sub_question['text'][:140]}\n"
                f"   tried: "
                + ("; ".join(f"{q[:70]!r}" for q in queries) or "(nothing)")
            )
        history = "\n".join(history_lines) or "(no plan recorded)"

        available = self._available_actions(state)
        budget_text = (
            self.budget.report() if self.budget is not None else "unlimited"
        )
        budget_text += f"\navailable actions: {', '.join(sorted(available))}"

        parsed, _ = self.provider.complete_json(
            system=render(
                "critic_router",
                claims=claims, history=history, budget=budget_text,
            ),
            user="Decide what to do about each failed claim.",
            role=Role.JUDGMENT,
            required_keys=["routes"],
        )
        self._charge()

        out: dict[int, tuple[str, str, str]] = {}
        for raw in parsed.get("routes") or []:
            if not isinstance(raw, dict):
                continue
            try:
                index = int(raw.get("index"))
            except (TypeError, ValueError):
                continue
            action = str(raw.get("action") or "").upper().strip()
            if action not in ACTIONS:
                action = UNRESOLVED
            out[index] = (
                action,
                str(raw.get("query") or "").strip(),
                str(raw.get("reasoning") or "")[:300],
            )
        return out

    # -- applying the decisions ---------------------------------------------

    def _finish(
        self,
        state: ResearchState,
        findings: list[ClaimVerification],
        routed: dict[int, str],
        *,
        adaptive: bool = True,
        retry: dict[int, str] | None = None,
    ) -> RoutingResult:
        plan = [dict(sq) for sq in (state.get("plan") or [])]
        by_id = {sq["id"]: sq for sq in plan}
        gaps: list[str] = list(state.get("unresolved_gaps") or [])
        needs_work = False

        for finding in findings:
            action = routed.get(id(finding), ACCEPT)
            finding["action_taken"] = action
            sub_question = by_id.get(finding.get("sub_question_id") or "")

            if action == RE_RETRIEVE and sub_question is not None:
                # Reopening the sub-question is what makes this a decision
                # rather than a note: the controller sees a non-terminal
                # sub-question and routes back to the retriever.
                sub_question["status"] = "in_progress"
                # The router's query becomes the next round's query, so the
                # re-retrieval asks something new rather than replaying the
                # sub-question's text (C27.2).
                query = (retry or {}).get(id(finding))
                if query:
                    sub_question["next_query"] = query
                needs_work = True
            elif action == REVISE_PLAN and sub_question is not None:
                sub_question["status"] = "insufficient"
                needs_work = True
            elif action == UNRESOLVED:
                gap = _gap_text(finding)
                if gap not in gaps:
                    gaps.append(gap)
                if sub_question is not None:
                    sub_question["status"] = "insufficient"
            # DROP_CLAIM and HEDGE change the report, not the research, and
            # are applied by the finalizer at C28.

        result = RoutingResult(
            findings=findings, plan=plan, unresolved_gaps=gaps,
            needs_more_work=needs_work,
        )
        self._trace(result, adaptive)
        return result

    # -- plumbing -----------------------------------------------------------

    def _charge(self) -> None:
        if self.budget is not None and self.budget.can_spend("llm"):
            self.budget.spend("llm", 1)

    def _trace(self, result: RoutingResult, adaptive: bool) -> None:
        if self.tracer is None:
            return
        from src.tracing.tracer import DecisionSurface

        counts = {action: result.count(action) for action in ACTIONS}
        self.tracer.decision(
            DecisionSurface.CRITIC_ROUTING,
            chosen=counts,
            alternatives=list(ACTIONS),
            rationale=(
                "five-way routing on claim failure"
                if adaptive else "binary routing (A5: five-way disabled)"
            ),
            inputs={
                "claims": len(result.findings),
                "accepted": result.count(ACCEPT),
                "needs_more_work": result.needs_more_work,
            },
            was_adaptive=adaptive,
        )


def critic_router_node(router: CriticRouter):
    """LangGraph node: critic findings -> actions, plan and gaps."""

    def node(state: dict) -> dict:
        result = router.route(state)  # type: ignore[arg-type]
        if result.error and not result.findings:
            return {}
        update: dict[str, Any] = {
            "critic_findings": result.findings,
            "unresolved_gaps": result.unresolved_gaps,
        }
        if result.plan:
            update["plan"] = result.plan
        if result.needs_more_work:
            # Clearing the draft sends the controller back through
            # retrieval and synthesis; leaving it would let a stale draft
            # reach the decider as though it reflected the new evidence.
            update["draft"] = ""
        return update

    return node


def _gap_text(finding: ClaimVerification) -> str:
    claim = finding["claim"][:160].rstrip(" .")
    return f"could not establish: {claim}"