"""Planner — initial decomposition and revision (D1).

Initial mode produces the first plan. Revision mode (C25) is what makes D1 a
*decision surface* rather than a preprocessing step: the plan can be
rewritten when it is failing, and the four ways of rewriting it are a
judgment rather than a retry.

Why `rationale` is not decoration
---------------------------------
When a sub-question fails to retrieve anything, revision must choose between
rewriting it, splitting it, merging it with another, or abandoning it as
unanswerable. That choice depends on *what the sub-question was for*. A
sub-question that exists to pin down a number can be rewritten with
different phrasing; one that exists to cover a whole topic should probably be
split; one whose contribution is already covered by a sibling should be
merged. Without the rationale, the reviser can only see that retrieval
failed, and every failure looks the same — which collapses four options into
"try again", and D1 becomes a retry loop.

What a rewrite resets, and what it does not
-------------------------------------------
A rewritten sub-question gets a **fresh round counter**: it is a different
question in substance, and leaving it at its round limit would make revision
useless exactly when it is needed. But it **keeps `attempted_queries`**, so
D2 cannot reissue a query that already failed, and it **spends a plan
revision**, which has its own ceiling. The budget that bounds revision is
`max_plan_revisions`, not the round counter — otherwise the two ceilings
interact and a late rewrite silently does nothing.

Why the suggested route is a suggestion
---------------------------------------
The planner proposes a route; D2 (C21) decides. Keeping the planner's guess
in `attempted_routes`-adjacent form rather than binding it is what leaves
room for escalation: the retriever can start where the planner pointed, find
nothing, and move to full text. If the planner's route were binding, D2 would
have nothing to decide.

Over-decomposition
------------------
§6.7 lists planner over-decomposition as an expected failure mode, and it is
not merely untidy: each sub-question costs retrieval budget from a pool of
about 20 (§7.1), so splitting one lookup into five starves the sub-questions
that needed the budget. The prompt says so, and `max_sub_questions` enforces
a ceiling — but the honest check is the trace, where a one-hop factual
question producing six sub-questions is visible.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

from src.config import Config, get_config
from src.llm.provider import LLMProvider, Role
from src.prompts import render
from src.state import Route, SubQuestion, new_sub_question

log = logging.getLogger(__name__)

REWRITE = "REWRITE"
SPLIT = "SPLIT"
MERGE = "MERGE"
ABANDON = "ABANDON"
REVISION_ACTIONS = (REWRITE, SPLIT, MERGE, ABANDON)

MAX_SPLIT_PARTS = 3

VALID_ROUTES: tuple[Route, ...] = (
    "arxiv_meta", "arxiv_fulltext", "web", "evidence_store",
)
DEFAULT_ROUTE: Route = "arxiv_meta"

MIN_SUB_QUESTIONS = 1
MAX_SUB_QUESTIONS = 7

# A sub-question shorter than this is almost always a fragment ("the rank?")
# rather than a self-contained query, and it will retrieve badly.
MIN_TEXT_CHARS = 15


class Planner:
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
        self.max_sub_questions = int(
            self.cfg.get("planning.max_sub_questions", MAX_SUB_QUESTIONS)
        )

    # -- initial decomposition ---------------------------------------------

    def plan(
        self, question: str, *, premise_check: dict[str, Any] | None = None
    ) -> list[SubQuestion]:
        """Decompose a question. Returns `[]` only if the LLM call fails."""
        question = (question or "").strip()
        if not question:
            return []

        parsed, _ = self.provider.complete_json(
            system=render(
                "planner_initial",
                question=question,
                premise_note=_premise_note(premise_check),
            ),
            user=f"Decompose this question:\n{question}",
            role=Role.JUDGMENT,
            required_keys=["sub_questions"],
        )
        self._charge()

        sub_questions = self._parse(parsed, question)
        self._trace(question, sub_questions)
        return sub_questions

    # -- revision (D1) ------------------------------------------------------

    def revise(
        self, state: dict[str, Any], sub_question_id: str
    ) -> dict[str, Any]:
        """Rewrite, split, merge or abandon a failing sub-question.

        Returns a state update containing the new plan. The caller is
        responsible for charging the revision against the budget ceiling.
        """
        plan = list(state.get("plan") or [])
        target = next((sq for sq in plan if sq["id"] == sub_question_id), None)
        if target is None:
            return {}

        siblings = [sq for sq in plan if sq["id"] != sub_question_id]
        try:
            parsed, _ = self.provider.complete_json(
                system=render(
                    "planner_revise",
                    question=state.get("question", ""),
                    sub_question=_describe(target),
                    siblings=_describe_siblings(siblings),
                ),
                user="Decide what to do with this sub-question.",
                role=Role.JUDGMENT,
                required_keys=["action"],
            )
            self._charge()
        except Exception as exc:  # noqa: BLE001
            # A failed revision must not strand the sub-question in a
            # non-terminal state, or the controller will route back here
            # forever. Abandoning is the honest fallback: we tried, we
            # could not decide how to fix it, and the gap gets reported.
            log.warning("plan revision failed for %s: %s", sub_question_id, exc)
            return self._apply(
                plan, target, ABANDON, {},
                f"revision failed ({type(exc).__name__})",
            )

        action = str(parsed.get("action") or "").upper().strip()
        if action not in REVISION_ACTIONS:
            action = ABANDON
        return self._apply(
            plan, target, action, parsed,
            str(parsed.get("reasoning") or "")[:300],
        )

    def _apply(
        self,
        plan: list[SubQuestion],
        target: SubQuestion,
        action: str,
        parsed: dict[str, Any],
        reasoning: str,
    ) -> dict[str, Any]:
        new_plan: list[SubQuestion] = []
        gaps: list[str] = []

        for sub_question in plan:
            if sub_question["id"] != target["id"]:
                new_plan.append(sub_question)
                continue

            if action == REWRITE:
                text = _text_of(parsed)
                if len(text) < MIN_TEXT_CHARS or _normalise(text) == _normalise(
                    target["text"]
                ):
                    # A rewrite that is not a rewrite would consume a
                    # revision and change nothing — §2.4's "the loop counter
                    # is the only thing that changes between iterations".
                    new_plan.append(_abandoned(
                        target, "rewrite was not substantially different"
                    ))
                    gaps.append(_gap(target))
                    continue
                revised = dict(target)
                revised["text"] = text
                revised["status"] = "pending"
                # Fresh rounds (a different question in substance), but the
                # attempted queries survive so D2 cannot reissue a failure.
                revised["rounds"] = 0
                revised["rationale"] = (
                    f"{target.get('rationale', '')} | revised: {reasoning}"
                ).strip(" |")[:300]
                new_plan.append(revised)  # type: ignore[arg-type]

            elif action == SPLIT:
                parts = [
                    p for p in (parsed.get("parts") or [])
                    if isinstance(p, str) and len(p.strip()) >= MIN_TEXT_CHARS
                ][:MAX_SPLIT_PARTS]
                if len(parts) < 2:
                    new_plan.append(_abandoned(
                        target, "split produced fewer than two usable parts"
                    ))
                    gaps.append(_gap(target))
                    continue
                for index, part in enumerate(parts):
                    child = new_sub_question(
                        " ".join(part.split()),
                        sub_question_id=f"{target['id']}.{index + 1}",
                        rationale=f"split from {target['id']}: {reasoning}",
                        parent_id=target["id"],
                    )
                    child["suggested_route"] = target.get("suggested_route")
                    new_plan.append(child)

            elif action == MERGE:
                into = str(parsed.get("merge_into") or "").strip()
                covered = any(sq["id"] == into for sq in plan)
                new_plan.append(_abandoned(
                    target,
                    f"covered by {into}: {reasoning}" if covered
                    else f"merge target {into!r} not in the plan; abandoned",
                ))
                if not covered:
                    gaps.append(_gap(target))

            else:   # ABANDON
                new_plan.append(_abandoned(target, reasoning))
                gaps.append(_gap(target))

        self._trace_revision(target, action, reasoning, new_plan)
        update: dict[str, Any] = {"plan": new_plan}
        if gaps:
            update["unresolved_gaps"] = gaps
        return update

    def _trace_revision(
        self,
        target: SubQuestion,
        action: str,
        reasoning: str,
        new_plan: list[SubQuestion],
    ) -> None:
        if self.tracer is None:
            return
        from src.tracing.tracer import DecisionSurface

        self.tracer.decision(
            DecisionSurface.D1_PLAN_REVISION,
            chosen=action,
            alternatives=[a for a in REVISION_ACTIONS if a != action],
            rationale=reasoning,
            inputs={
                "sub_question_id": target["id"],
                "text": target["text"][:160],
                "rounds": target.get("rounds", 0),
                "attempted_queries": target.get("attempted_queries", []),
                "plan_size_after": len(new_plan),
            },
            # Revision IS the adaptive half of D1 — unlike initial
            # decomposition, which happens in every configuration including
            # the A1 ablation.
            was_adaptive=True,
        )

    # -- parsing ------------------------------------------------------------

    def _parse(self, parsed: dict[str, Any], question: str) -> list[SubQuestion]:
        raw_items = parsed.get("sub_questions") or []
        out: list[SubQuestion] = []
        seen: set[str] = set()

        for index, raw in enumerate(raw_items):
            text = _text_of(raw)
            if len(text) < MIN_TEXT_CHARS:
                continue
            # Two sub-questions with identical text would retrieve the same
            # passages twice and split the budget for one job.
            key = _normalise(text)
            if key in seen:
                continue
            seen.add(key)

            sub_question = new_sub_question(
                text,
                sub_question_id=f"sq{len(out) + 1}",
                rationale=_rationale_of(raw),
            )
            # The planner's route is a SUGGESTION. Recording it as the first
            # attempted route would tell D2 the route had already been tried
            # and failed, which is exactly wrong.
            sub_question["suggested_route"] = _route_of(raw)  # type: ignore[typeddict-unknown-key]
            out.append(sub_question)

            if len(out) >= self.max_sub_questions:
                break

        if not out:
            # A question the planner could not decompose is still a question.
            # Researching it whole beats abandoning it before retrieval.
            log.info("planner returned no usable sub-questions; using the "
                     "question verbatim")
            fallback = new_sub_question(
                question, sub_question_id="sq1",
                rationale="planner produced no usable decomposition; "
                          "researching the question as asked",
            )
            fallback["suggested_route"] = DEFAULT_ROUTE  # type: ignore[typeddict-unknown-key]
            out = [fallback]
        return out

    # -- plumbing -----------------------------------------------------------

    def _charge(self) -> None:
        if self.budget is not None and self.budget.can_spend("llm"):
            self.budget.spend("llm", 1)

    def _trace(self, question: str, sub_questions: Sequence[SubQuestion]) -> None:
        if self.tracer is None:
            return
        from src.tracing.tracer import DecisionSurface

        self.tracer.decision(
            DecisionSurface.D1_PLAN_REVISION,
            chosen=[sq["text"] for sq in sub_questions],
            alternatives=[],
            rationale="initial decomposition",
            inputs={
                "question": question[:200],
                "n_sub_questions": len(sub_questions),
                "routes": [sq.get("suggested_route") for sq in sub_questions],
            },
            # Initial decomposition happens in every configuration, including
            # the A1 ablation that freezes revision. Counting it as adaptive
            # would make an A1 run report non-zero D1 activity and fail C35's
            # check for the wrong reason.
            was_adaptive=False,
        )


def planner_node(planner: Planner, budget: Any = None):
    """LangGraph node: initial decomposition, or revision if a plan exists.

    One node, two modes. The controller routes here both when the plan is
    empty and when a sub-question is failing, and which is meant is visible
    from the state — a plan that exists cannot be decomposed again.
    """

    def node(state: dict) -> dict:
        if not state.get("plan"):
            return {"plan": planner.plan(
                state.get("question", ""),
                premise_check=state.get("premise_check"),
            )}

        failing = next(
            (sq for sq in state["plan"]
             if sq.get("status") == "insufficient"
             and int(sq.get("rounds", 0)) >= 1),
            None,
        )
        if failing is None:
            return {}

        update = planner.revise(state, failing["id"])
        if update:
            # The revision counter is what bounds D1, not the round counter
            # — a rewrite resets rounds deliberately, so without this the
            # controller would route back here indefinitely.
            update["plan_revisions"] = int(state.get("plan_revisions", 0)) + 1
            # The run's budget is bound onto the planner by graph._attach,
            # not passed to this factory — `default_components` never passes
            # one. Reading only the closure variable left the live counter at
            # zero in every real run (C27 live: "revisions 0/2" after a
            # revision). The ceiling itself is enforced from state as well;
            # see budget.plan_revisions_remain.
            active = budget if budget is not None else getattr(planner, "budget", None)
            if active is not None:
                active.record_plan_revision()
        return update

    return node


# ---------------------------------------------------------------------------
# Field extraction — defensive, because the model controls this shape
# ---------------------------------------------------------------------------


def _text_of(raw: Any) -> str:
    if isinstance(raw, str):
        return " ".join(raw.split())
    if isinstance(raw, dict):
        for key in ("text", "question", "sub_question"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
    return ""


def _rationale_of(raw: Any) -> str:
    if isinstance(raw, dict):
        value = raw.get("rationale") or raw.get("reason") or ""
        return " ".join(str(value).split())[:300]
    return ""


def _route_of(raw: Any) -> Route:
    if isinstance(raw, dict):
        value = str(raw.get("route") or "").strip().lower()
        if value in VALID_ROUTES:
            return value  # type: ignore[return-value]
    return DEFAULT_ROUTE


def _normalise(text: str) -> str:
    return " ".join(text.lower().split()).rstrip("?.")


def _premise_note(premise_check: dict[str, Any] | None) -> str:
    """Tell the planner what the premise checker established.

    Two reasons this is worth the tokens. A confirmed presupposition does not
    need its own sub-question — re-researching it wastes budget. And a
    *refuted* one means the question is built on something false, so the plan
    should investigate the correction rather than elaborate the fiction.
    """
    if not premise_check or not premise_check.get("ran"):
        return ""

    lines: list[str] = []
    confirmed = [
        p["text"] for p in premise_check.get("presuppositions", [])
        if p.get("verdict") == "SUPPORTED"
    ]
    if confirmed:
        lines.append(
            "ALREADY ESTABLISHED (do not create sub-questions for these):\n"
            + "\n".join(f"- {c}" for c in confirmed[:4])
        )
    if premise_check.get("failed"):
        correction = premise_check.get("correction") or ""
        lines.append(
            "WARNING: this question rests on a false premise."
            + (f" What is actually the case: {correction}" if correction else "")
            + "\nPlan to establish the correction, not to elaborate the "
              "false premise."
        )
    return ("\n\n" + "\n\n".join(lines)) if lines else ""


def _abandoned(target: SubQuestion, reasoning: str) -> SubQuestion:
    out = dict(target)
    out["status"] = "abandoned"
    out["rationale"] = f"{target.get('rationale', '')} | {reasoning}".strip(" |")[:300]
    return out  # type: ignore[return-value]


def _gap(target: SubQuestion) -> str:
    return f"could not establish: {target['text'][:160].rstrip(' .')}"


def _describe(sub_question: SubQuestion) -> str:
    """What the reviser sees. The purpose line is the one that decides."""
    features = sub_question.get("sufficiency_features") or {}
    f1 = features.get("f1_max_rerank")
    queries = sub_question.get("attempted_queries") or []
    return (
        f"id: {sub_question['id']}\n"
        f"text: {sub_question['text']}\n"
        f"why it exists: {sub_question.get('rationale') or '(not recorded)'}\n"
        f"rounds spent: {sub_question.get('rounds', 0)}\n"
        f"routes tried: {', '.join(sub_question.get('attempted_routes') or []) or '(none)'}\n"
        f"queries tried:\n"
        + ("\n".join(f"   - {q[:100]!r}" for q in queries) or "   (none)")
        + f"\nbest relevance score: "
        + ("(nothing retrieved)" if f1 is None else f"{f1:.2f}")
        + "   (above ~4 means a passage answered the query)"
    )


def _describe_siblings(siblings: list[SubQuestion]) -> str:
    if not siblings:
        return "(this is the only sub-question)"
    return "\n".join(
        f"{sq['id']}  status={sq.get('status')}  {sq['text'][:110]}"
        for sq in siblings
    )