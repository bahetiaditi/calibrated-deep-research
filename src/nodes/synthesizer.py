"""Synthesizer — drafts the report with validated citations.

The label map, and why C6 deferred this problem here
----------------------------------------------------
Passage ids are content-addressed (`Pa3f9c2b1d004`). That was the right call
for storage — it makes ingestion idempotent across runs — but it is a bad
thing to ask a language model to transcribe. A model copying twelve hex
characters gets them wrong often enough to matter, and **a citation that does
not resolve is indistinguishable from a fabricated one**: both are a claim
the critic cannot verify, and the system would report a transcription slip as
a hallucination.

So the prompt uses short sequential labels — `P01`, `P02` — and this module
maps them back. The storage id stays stable and the model gets something it
can copy. C6 recorded that this was the plan; this is where it comes due.

Validation is not optional
--------------------------
Every citation in the draft is checked against the label map before the draft
leaves this node. Three outcomes, each handled differently:

  resolves        rewritten to the real passage id, ready for the critic
  unknown label   the model invented `[P47]` when only 19 passages exist.
                  **The marker is stripped**, leaving the claim uncited.
  no citation     left alone.

Stripping an unresolvable citation rather than keeping it is deliberate. An
uncited claim is something the critic will mark UNSUPPORTED, which is
correct and visible. A citation pointing at nothing would either crash the
critic's lookup or be silently skipped — and a claim that *looks* cited but
was never checked is the exact failure this project exists to measure.

The count is recorded either way: a model that invents labels is a signal
about prompt quality, and burying it would hide a real problem.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.config import Config, get_config
from src.llm.provider import LLMProvider, Role
from src.prompts import render
from src.state import Passage, ResearchState, evidence_for, find_sub_question

log = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[(P\d{2,}|P[0-9a-f]{12})\]")

# How much of each passage the synthesizer sees. The full text of 20 passages
# would dominate the context and push the model toward summarising evidence
# rather than answering the question.
MAX_PASSAGE_CHARS = 900
MAX_PASSAGES = 24


@dataclass
class Draft:
    text: str = ""
    label_map: dict[str, str] = field(default_factory=dict)   # P01 -> passage id
    cited_passage_ids: list[str] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)
    uncited_sentences: int = 0
    mode: str = "answer"            # answer | partial
    error: str | None = None

    @property
    def citation_count(self) -> int:
        return len(self.cited_passage_ids)


class Synthesizer:
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

    # -- main entry point ---------------------------------------------------

    def synthesize(
        self,
        state: ResearchState,
        *,
        mode: str = "answer",
        gaps: Sequence[str] | None = None,
    ) -> Draft:
        question = (state.get("question") or "").strip()
        passages = self._select(state)
        if not question or not passages:
            return Draft(mode=mode, error="no question or no evidence")

        label_map = {
            f"P{i + 1:02d}": passage["id"] for i, passage in enumerate(passages)
        }
        rendered = self._render_evidence(passages)
        gaps = list(gaps or state.get("unresolved_gaps") or [])

        prompt = "synthesizer_partial" if mode == "partial" else "synthesizer"
        gap_block = (
            "\n".join(f"- {g}" for g in gaps) if gaps
            else "(none)" if mode == "partial" else ""
        )
        if mode == "answer" and gap_block:
            gap_block = (
                "NOT ESTABLISHED — do not claim these:\n"
                + gap_block + "\n\n"
            )

        try:
            response = self.provider.complete(
                system=render(
                    prompt, question=question, gaps=gap_block, evidence=rendered
                ),
                user=f"Write the report answering:\n{question}",
                role=Role.JUDGMENT,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("synthesis failed: %s", exc)
            return Draft(mode=mode, label_map=label_map, error=str(exc))
        self._charge()

        draft = self._validate(response.text, label_map, mode)
        self._trace(draft, len(passages))
        return draft

    # -- evidence selection -------------------------------------------------

    def _select(self, state: ResearchState) -> list[Passage]:
        """Order evidence by sub-question, then by rank within it.

        Grouping by sub-question rather than pooling everything by score
        keeps each sub-question's best evidence visible. A global sort lets
        one well-covered sub-question occupy every slot and starve the
        others, which produces a report that answers part of the question
        confidently and ignores the rest.
        """
        ordered: list[Passage] = []
        seen: set[str] = set()

        def take(passages) -> None:
            for passage in passages:
                if passage["id"] in seen:
                    continue
                seen.add(passage["id"])
                ordered.append(passage)

        plan = state.get("plan") or []

        # Live sub-questions first.
        for sub_question in plan:
            if sub_question.get("status") != "abandoned":
                take(evidence_for(state, sub_question["id"]))

        # Then evidence with no owning sub-question — the premise checker's,
        # chiefly — and finally anything an abandoned sub-question retrieved.
        #
        # Demoted rather than excluded. An abandoned sub-question was given
        # up on because it was malformed or unanswerable, so its evidence was
        # retrieved for a bad query and is likelier to be off-topic. But it is
        # not *disqualified*: a passage can be a poor answer to one question
        # and a good one for another, and silently discarding evidence the
        # system paid to retrieve is a stronger claim than the status
        # supports. Ordering last means MAX_PASSAGES drops it first when the
        # budget binds, which is the effect we want without the assertion.
        take(state.get("evidence") or [])

        return ordered[:MAX_PASSAGES]

    def _render_evidence(self, passages: Sequence[Passage]) -> str:
        blocks = []
        for i, passage in enumerate(passages):
            section = passage.get("section")
            # The section is shown because a Related Work passage describes
            # someone else's contribution, and the prompt asks the model to
            # distinguish measured from cited results (§4.2).
            header = f"P{i + 1:02d}  {passage.get('source_id', '?')}"
            if section:
                header += f"  [{section}]"
            if passage.get("published"):
                header += f"  {passage['published']}"
            text = " ".join((passage.get("text") or "").split())
            blocks.append(f"{header}\n{text[:MAX_PASSAGE_CHARS]}")
        return "\n\n".join(blocks)

    # -- validation ---------------------------------------------------------

    def _validate(self, text: str, label_map: dict[str, str], mode: str) -> Draft:
        invalid: list[str] = []
        cited: list[str] = []

        def replace(match: re.Match) -> str:
            label = match.group(1)
            passage_id = label_map.get(label)
            if passage_id is None:
                # Invented label. Strip it: an uncited claim is caught by the
                # critic, a dangling citation is not.
                invalid.append(label)
                return ""
            if passage_id not in cited:
                cited.append(passage_id)
            return f"[{passage_id}]"

        rewritten = CITATION_RE.sub(replace, text or "")
        rewritten = re.sub(r" +([.,;])", r"\1", rewritten)   # tidy after strips
        rewritten = re.sub(r"[ \t]{2,}", " ", rewritten).strip()

        return Draft(
            text=rewritten,
            label_map=label_map,
            cited_passage_ids=cited,
            invalid_labels=invalid,
            uncited_sentences=_count_uncited(rewritten),
            mode=mode,
        )

    # -- plumbing -----------------------------------------------------------

    def _charge(self) -> None:
        if self.budget is not None and self.budget.can_spend("llm"):
            self.budget.spend("llm", 1)

    def _trace(self, draft: Draft, n_passages: int) -> None:
        if self.tracer is None:
            return
        self.tracer.note(
            "synthesis",
            mode=draft.mode,
            passages_offered=n_passages,
            passages_cited=draft.citation_count,
            invalid_labels=draft.invalid_labels,
            uncited_sentences=draft.uncited_sentences,
            chars=len(draft.text),
        )


def synthesizer_node(synthesizer: Synthesizer):
    """LangGraph node: evidence -> state['draft'].

    On the ABSTAIN path the synthesizer does not run at all (§3.3) — there is
    no report to write, and generating one would invite the finalizer to show
    it.
    """

    def node(state: dict) -> dict:
        decision = state.get("terminal_decision")
        if decision == "ABSTAIN":
            return {"draft": ""}
        mode = "partial" if decision == "PARTIAL" else "answer"
        draft = synthesizer.synthesize(state, mode=mode)  # type: ignore[arg-type]
        update: dict[str, Any] = {"draft": draft.text}
        if draft.error:
            update["error"] = draft.error
        return update

    return node


def _count_uncited(text: str) -> int:
    """Sentences making a claim with no citation.

    A rough signal, not a verdict — the critic decides. Short sentences are
    skipped because transitions ("That said.") are not claims.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text or "")
    return sum(
        1 for s in sentences
        if len(s.strip()) > 40 and not CITATION_RE.search(s)
    )