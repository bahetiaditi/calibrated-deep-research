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
from src.llm.provider import LLMProvider, Role, estimate_tokens
from src.prompts import render
from src.state import draft_digest, Passage, ResearchState, evidence_for, find_sub_question

log = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[(P\d{2,}|P[0-9a-f]{12})\]")

# Citation forms models actually produce besides the instructed [P01]
# (C27.4). One citation group in brackets, full-width brackets or
# parentheses, holding one or more short labels separated by commas,
# semicolons, "and", "&" or whitespace, or a range "P01-P03". Unpadded
# labels ("P1") are accepted. A content-addressed id ("P1a2b3c4d5e6f")
# cannot match: a short label must be followed by a separator or the
# closing bracket, never a hex letter.
_LABEL = r"P\d{1,3}"
_GROUP_RE = re.compile(
    r"[\[【(]\s*("
    + _LABEL + r"(?:\s*(?:,|;|&|\band\b|[-–—]|\s)\s*" + _LABEL + r")*"
    + r")\s*[\]】)]"
)
_RANGE_RE = re.compile(r"(P\d{1,3})\s*[-–—]\s*(P\d{1,3})")
# Anything still label-shaped after normalisation was not understood. The
# synthesis trace reports how many, so a draft whose citations were all
# dropped is visible rather than silently uncited.
_LEFTOVER_RE = re.compile(r"(?<![A-Za-z0-9])P\d{1,3}(?![0-9a-f])")
MAX_RANGE = 10

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
    # C27.4: citations rewritten from a non-standard form, and label-shaped
    # tokens that could not be understood at all.
    normalised_citations: int = 0
    unparsed_citations: int = 0
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
        user = f"Write the report answering:\n{question}"

        passages, dropped = self._fit(state, passages, prompt, question, gap_block, user)
        if not passages:
            return Draft(mode=mode, error=(
                "prompt budget too small for even one passage — the prompt "
                "template and gaps alone exceed what the judgment chain accepts"
            ))

        label_map = {
            f"P{i + 1:02d}": passage["id"] for i, passage in enumerate(passages)
        }
        rendered = self._render_evidence(passages)

        try:
            response = self.provider.complete(
                system=render(
                    prompt, question=question, gaps=gap_block, evidence=rendered
                ),
                user=user,
                role=Role.JUDGMENT,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("synthesis failed: %s", exc)
            return Draft(mode=mode, label_map=label_map, error=str(exc))
        self._charge()

        draft = self._validate(response.text, label_map, mode)
        self._trace(draft, len(passages), dropped=dropped)
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

    def _fit(
        self,
        state: ResearchState,
        passages: list[Passage],
        prompt: str,
        question: str,
        gap_block: str,
        user: str,
    ) -> tuple[list[Passage], int]:
        """Trim evidence until the prompt fits the judgment chain (C27.3).

        Without this the synthesizer could not be sent to ANY judgment
        model: 24 passages came to ~6.1K estimated tokens, the ledger adds
        the 2,048-token output reservation, and every Groq model's TPM is
        8K. The first C27.2 smoke failed both questions this way.

        Trimming is round-robin across sub-questions, lowest-ranked first:
        cutting the tail of the grouped list would drop the LAST
        sub-question's evidence entirely, which is the starvation C18's
        grouping exists to prevent. The kept passages are then shown in the
        usual grouped order. Returns (kept, number dropped).
        """
        budget = self.provider.prompt_token_budget(Role.JUDGMENT) \
            if hasattr(self.provider, "prompt_token_budget") else None
        if budget is None:
            return passages, 0

        def fits(kept: Sequence[Passage]) -> bool:
            system = render(prompt, question=question, gaps=gap_block,
                            evidence=self._render_evidence(kept))
            return estimate_tokens(system, user, 0) <= budget

        if fits(passages):
            return passages, 0

        priority = _round_robin(state, passages)
        keep_n = len(priority)
        while keep_n > 0 and not fits(_in_order(passages, priority[:keep_n])):
            keep_n -= 1
        kept = _in_order(passages, priority[:keep_n])
        return kept, len(passages) - len(kept)

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

        normalised, n_normalised = normalize_citations(text or "")
        rewritten = CITATION_RE.sub(replace, normalised)
        # Count what is still label-shaped once valid citations are resolved
        # to content-addressed ids (which _LEFTOVER_RE cannot match).
        unparsed = len(_LEFTOVER_RE.findall(rewritten))
        rewritten = re.sub(r" +([.,;])", r"\1", rewritten)   # tidy after strips
        rewritten = re.sub(r"[ \t]{2,}", " ", rewritten).strip()

        return Draft(
            text=rewritten,
            label_map=label_map,
            cited_passage_ids=cited,
            invalid_labels=invalid,
            uncited_sentences=_count_uncited(rewritten),
            normalised_citations=n_normalised,
            unparsed_citations=unparsed,
            mode=mode,
        )

    # -- plumbing -----------------------------------------------------------

    def _charge(self) -> None:
        if self.budget is not None and self.budget.can_spend("llm"):
            self.budget.spend("llm", 1)

    def _trace(self, draft: Draft, n_passages: int, *, dropped: int = 0) -> None:
        if self.tracer is None:
            return
        self.tracer.note(
            "synthesis",
            mode=draft.mode,
            passages_offered=n_passages,
            passages_dropped_for_budget=dropped,
            normalised_citations=draft.normalised_citations,
            unparsed_citations=draft.unparsed_citations,
            passages_cited=draft.citation_count,
            invalid_labels=draft.invalid_labels,
            uncited_sentences=draft.uncited_sentences,
            chars=len(draft.text),
            # Lets a trace show whether a re-synthesis produced the SAME text
            # (a cache hit on unchanged evidence), which is when the
            # controller correctly skips a second critique (C27).
            digest=draft_digest(draft.text),
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
        # The mode is recorded because D5 needs to know which prompt wrote
        # the draft it is looking at: an answer-mode draft under a PARTIAL
        # decision still has to be rewritten, a partial-mode one does not.
        update: dict[str, Any] = {"draft": draft.text, "draft_mode": mode}
        if draft.error:
            update["error"] = draft.error
        elif not draft.text.strip():
            # An empty draft with no exception is still a failure, and it
            # must be reported as one. The controller routes to the
            # synthesizer while `draft` is falsy, so a silently empty result
            # sends it straight back — an infinite loop that costs an LLM
            # call per iteration until the recursion limit fires.
            update["error"] = "synthesizer returned no text"
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


def _round_robin(state: ResearchState, passages: Sequence[Passage]) -> list[Passage]:
    """Priority for trimming: each live sub-question's best passage, then
    each one's second-best, and so on; passages owned by no live
    sub-question last. Within a sub-question, `passages` is already in
    rank order (see `_select`)."""
    by_owner: dict[str, list[Passage]] = {}
    for sub_question in state.get("plan") or []:
        if sub_question.get("status") == "abandoned":
            continue
        ids = set(sub_question.get("evidence_ids") or [])
        by_owner[sub_question["id"]] = [p for p in passages if p["id"] in ids]

    order: list[Passage] = []
    seen: set[str] = set()
    depth = max((len(v) for v in by_owner.values()), default=0)
    for i in range(depth):
        for group in by_owner.values():
            if i < len(group) and group[i]["id"] not in seen:
                seen.add(group[i]["id"])
                order.append(group[i])
    order.extend(p for p in passages if p["id"] not in seen)
    return order


def _in_order(passages: Sequence[Passage], keep: Sequence[Passage]) -> list[Passage]:
    ids = {p["id"] for p in keep}
    return [p for p in passages if p["id"] in ids]



def normalize_citations(text: str) -> tuple[str, int]:
    """Rewrite citation variants to the instructed `[P01][P03]` form.

    Added at C27.4. The C27.3 live run produced an answer-mode draft with
    0 of 22 passages cited: CITATION_RE accepted only `[P01]`, so any other
    form the model used was left as plain text, the critic saw no citations,
    every claim came back UNSUPPORTED and unowned, the router turned each one
    into a "could not establish" gap — and those gaps were then handed to the
    partial rewrite as topics it must not answer. One formatting habit
    cascaded into a PARTIAL decision and a report forbidden from stating
    claims its evidence supported.

    Returns the rewritten text and how many groups were rewritten. Exact
    `[P01]` groups are left alone and not counted.
    """
    count = 0

    def labels_of(body: str) -> list[str] | None:
        out: list[str] = []
        for start, end in _RANGE_RE.findall(body):
            a, b = int(start[1:]), int(end[1:])
            if a > b or b - a > MAX_RANGE:
                return None
            out.extend(f"P{i:02d}" for i in range(a, b + 1))
        body = _RANGE_RE.sub(" ", body)
        out.extend(f"P{int(m[1:]):02d}" for m in re.findall(_LABEL, body))
        seen: list[str] = []
        for label in out:
            if label not in seen:
                seen.append(label)
        return seen

    def rewrite(match: re.Match) -> str:
        nonlocal count
        original = match.group(0)
        labels = labels_of(match.group(1))
        if not labels:
            return original
        result = "".join(f"[{label}]" for label in labels)
        if result != original:
            count += 1
        return result

    return _GROUP_RE.sub(rewrite, text), count
