"""Planner — initial decomposition (D1, first half).

Revision mode lands at C25. This commit produces the first plan; what makes
D1 a *decision surface* rather than a preprocessing step is that the plan can
later be rewritten, and that only works if this node records enough for the
reviser to act on.

Why `rationale` is not decoration
---------------------------------
When a sub-question fails to retrieve anything, C25 must choose between
rewriting it, splitting it, merging it with another, or abandoning it as
unanswerable. That choice depends on *what the sub-question was for*. A
sub-question that exists to pin down a number can be rewritten with
different phrasing; one that exists to cover a whole topic should probably be
split; one whose contribution is already covered by a sibling should be
merged. Without the rationale, the reviser can only see that retrieval
failed, and every failure looks the same — which collapses four options into
"try again", and D1 becomes a retry loop.

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


def planner_node(planner: Planner):
    """LangGraph node: question -> state['plan']."""

    def node(state: dict) -> dict:
        plan = planner.plan(
            state.get("question", ""),
            premise_check=state.get("premise_check"),
        )
        return {"plan": plan}

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