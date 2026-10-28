"""The graph's state.

Completed at C14; `Passage` and the shared literals landed early at C6 because
the retrieval tools needed them before the graph existed.

What the fields are for
-----------------------
§3.2 makes a point worth repeating here, because it is the difference between
a graph and a pipeline: several fields exist purely to *enable* a decision
surface, and if one is never read, the corresponding surface is dead.

    attempted_queries   D2 cannot reformulate without knowing what failed
    attempted_routes    D2 cannot escalate without knowing what it tried
    rounds, sufficiency D3 cannot decide depth without them
    allocation          D4 is otherwise a uniform split by default
    plan_revisions      D1 is otherwise one-shot decomposition
    unresolved_gaps     D5's PARTIAL path is otherwise indistinguishable
                        from ANSWER

So the C35 ablation check has a cheap precursor: a field nobody reads is a
decision nobody makes.

Fields added at C27
-------------------
    critiqued_draft     digest of the draft the critic last reviewed. The
                        controller routes to the critic whenever the current
                        draft differs from it — see `draft_digest`.
    draft_mode          "answer" or "partial": which synthesizer prompt wrote
                        the current draft. D5 reads it to know whether a
                        PARTIAL decision still needs a partial-mode rewrite.
    decision            D5's full record: score, thresholds, which rule
                        fired, per-sub-question features and verdicts AT
                        DECISION TIME. `terminal_decision` is the label;
                        this is what C33 fits on and C37's risk-coverage
                        curve sweeps over. A label alone cannot be
                        re-thresholded.

On reducers
-----------
`evidence` and `contradictions` accumulate across retrieval rounds, so they
take reducers. Everything else is replaced wholesale by the node that owns
it — the planner owns `plan`, the critic owns `critic_findings`, and a stale
merge there would be a correctness bug rather than a convenience.

`unresolved_gaps` accumulates too (C27). It had no reducer, so the planner's
revision — which returns only the gaps *it* found — replaced every gap the
critic router had recorded before it. A gap is the content of a PARTIAL
answer; losing one means the report claims less was missing than was.

`merge_evidence` deduplicates by passage id rather than using
`operator.add`. Ids are content-addressed (C6), so a repeat id means
byte-identical text. Appending it twice would let ONE piece of evidence count
twice toward f2 (mean of top-3 rerank scores) and f4 (independent source
count) — inflating sufficiency exactly when evidence is thinnest, which is
the precise failure the abstention layer exists to prevent. The same bug was
caught at ingestion in C8; the reducer is where it would reappear, because
re-retrieval legitimately returns passages already in state.
"""
from __future__ import annotations

import hashlib
import uuid
from typing import Annotated, Any, Literal, Sequence, TypedDict

Route = Literal["arxiv_meta", "arxiv_fulltext", "web", "evidence_store"]
Status = Literal["pending", "in_progress", "sufficient", "insufficient", "abandoned"]
Verdict = Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"]
Terminal = Literal["ANSWER", "PARTIAL", "ABSTAIN"]
SourceType = Literal["arxiv", "web"]

# A sub-question in one of these states will not be worked on again.
TERMINAL_STATUSES: frozenset[str] = frozenset({"sufficient", "insufficient", "abandoned"})


# ---------------------------------------------------------------------------
# Passage (C6)
# ---------------------------------------------------------------------------


class Passage(TypedDict):
    """One retrievable unit of evidence.

    On `id`: §3.2 sketched sequential ids ("P0001 …"). They are
    content-addressed instead — `make_passage_id` hashes source, section and
    text. The evidence store accumulates across runs (§4.1), so a per-run
    counter would collide between runs and re-ingesting a paper would
    duplicate it. A stable id also means a cached passage is recognisably the
    same passage.

    The cost is that ids are unfriendly for an LLM to transcribe, which
    matters because the synthesizer must reproduce them exactly (C18). That is
    solved there with a per-prompt label map (P01…P20 → id), not by making the
    storage id fragile.
    """

    id: str
    source_type: SourceType
    source_id: str          # arXiv id (e.g. "2312.00752v2") or canonical URL
    source_domain: str      # for counting independent sources (feature f4)
    title: str
    text: str
    section: str | None     # section-aware chunking for papers (§4.2)
    published: str | None   # ISO date if known — staleness signal (f10)
    sub_question_id: str
    retrieval_score: float  # post-fusion
    rerank_score: float | None   # RAW LOGIT, not a probability (see C11)


def make_passage_id(source_id: str, text: str, section: str | None = None) -> str:
    """Deterministic, collision-resistant passage id."""
    digest = hashlib.sha256(
        "\x00".join([source_id, section or "", text]).encode("utf-8")
    ).hexdigest()
    return f"P{digest[:12]}"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


class SubQuestion(TypedDict):
    id: str
    text: str
    status: Status
    rationale: str                 # why the planner created it
    attempted_routes: list[Route]  # so D2 can escalate
    attempted_queries: list[str]   # so D2 does not repeat a failure
    rounds: int
    sufficiency: float | None      # from §5
    sufficiency_features: dict[str, Any]   # raw feature vector, logged for analysis
    parent_id: str | None          # set when the planner splits one (D1)
    # Passage ids retrieved FOR this sub-question, in rerank order.
    #
    # Added at C17 to close a gap. `evidence` is deduplicated by passage id
    # (merge_evidence), so when two sub-questions retrieve the same passage
    # only the first copy survives and it carries the first sub-question's
    # `sub_question_id`. Filtering state evidence by that field would then
    # silently hide shared evidence from the second sub-question — and shared
    # evidence is common, since sibling sub-questions are about one topic.
    # This list is the authoritative mapping; Passage.sub_question_id records
    # only which sub-question found a passage first.
    evidence_ids: list[str]
    suggested_route: Route | None  # the planner's guess (C16); D2 may override


class ClaimVerification(TypedDict):
    claim: str
    sub_question_id: str
    cited_passage_ids: list[str]
    verdict: Verdict
    reasoning: str
    action_taken: str              # which of the five routes in §2.3


class BudgetState(TypedDict):
    llm_calls_used: int
    llm_calls_max: int
    retrieval_calls_used: int
    retrieval_calls_max: int
    allocation: dict[str, int]     # sub_question_id -> calls granted (D4)


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------


def merge_evidence(
    existing: Sequence[Passage] | None, incoming: Sequence[Passage] | None
) -> list[Passage]:
    """Append new passages, dropping ids already present.

    Not `operator.add`: re-retrieval legitimately returns passages already in
    state, and content-addressed ids mean a repeat id is byte-identical text.
    Counting it twice inflates f2 and f4 — see the module docstring.

    A later copy of an existing id is discarded rather than replacing the
    earlier one, so a passage keeps the scores from the round that first
    found it. Overwriting would mean a sub-question's rerank score could be
    silently replaced by another sub-question's, and feature extraction
    reads those scores.
    """
    merged: list[Passage] = list(existing or [])
    seen = {p["id"] for p in merged}
    for passage in incoming or []:
        pid = passage.get("id")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        merged.append(passage)
    return merged


def merge_contradictions(
    existing: Sequence[dict[str, Any]] | None,
    incoming: Sequence[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Accumulate contradiction records, deduplicated by unordered pair.

    Contradiction is symmetric, so (A, B) and (B, A) are one finding. Storing
    both would double the numerator of feature f5 while the denominator
    counts each pair once.
    """
    merged: list[dict[str, Any]] = list(existing or [])
    seen = {
        frozenset((r.get("passage_a"), r.get("passage_b"))) for r in merged
    }
    for record in incoming or []:
        key = frozenset((record.get("passage_a"), record.get("passage_b")))
        if key in seen:
            continue
        seen.add(key)
        merged.append(record)
    return merged


def merge_gaps(
    existing: Sequence[str] | None, incoming: Sequence[str] | None
) -> list[str]:
    """Append gaps, keeping order and dropping exact repeats.

    Order-preserving rather than a set, because the order gaps were found in
    is the order a reader should meet them in. Idempotent on the full list,
    so a node that returns existing-plus-new (the critic router) and one
    that returns only new (the planner) are both correct.
    """
    merged: list[str] = list(existing or [])
    seen = set(merged)
    for gap in incoming or []:
        if gap and gap not in seen:
            seen.add(gap)
            merged.append(gap)
    return merged


def draft_digest(draft: str | None) -> str:
    """Identity of a draft, for "has the critic seen THIS draft?".

    Introduced at C27 to replace `not state["critic_findings"]` as the
    controller's test for whether to run the critic. That test was wrong in
    two directions. After RE_RETRIEVE the draft is rewritten but the old
    findings remain, so the new draft went to the decider unverified, judged
    by verdicts on text that no longer existed. And a draft the critic found
    to contain no claims leaves findings empty, so the controller sent it
    back to the critic until the recursion limit fired.
    """
    return hashlib.sha256((draft or "").strip().encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# ResearchState
# ---------------------------------------------------------------------------


class ResearchState(TypedDict, total=False):
    """The graph's state.

    `total=False` so nodes may return partial updates — LangGraph merges the
    returned keys and leaves the rest untouched. `new_state()` populates every
    field, so a node reading state never meets a missing key in practice.
    """

    question: str
    premise_check: dict[str, Any]       # false-premise detection result (C15)

    plan: list[SubQuestion]
    plan_revisions: int                 # D1 ceiling lives in budget.py

    evidence: Annotated[list[Passage], merge_evidence]
    contradictions: Annotated[list[dict[str, Any]], merge_contradictions]

    draft: str
    draft_mode: str                      # "answer" | "partial" | "" (C27)
    critic_findings: list[ClaimVerification]
    critiqued_draft: str                 # draft_digest of what the critic saw

    budget: BudgetState

    terminal_decision: Terminal | None   # D5
    decision_rationale: str
    decision: dict[str, Any]             # D5's full record (C27)
    unresolved_gaps: Annotated[list[str], merge_gaps]   # what we could not establish

    final_report: str
    trace_id: str
    error: str | None


# ---------------------------------------------------------------------------
# Constructors and helpers
# ---------------------------------------------------------------------------


def new_sub_question(
    text: str,
    *,
    sub_question_id: str | None = None,
    rationale: str = "",
    parent_id: str | None = None,
) -> SubQuestion:
    return SubQuestion(
        id=sub_question_id or f"sq{uuid.uuid4().hex[:8]}",
        text=text,
        status="pending",
        rationale=rationale,
        attempted_routes=[],
        attempted_queries=[],
        rounds=0,
        sufficiency=None,
        sufficiency_features={},
        parent_id=parent_id,
        evidence_ids=[],
        suggested_route=None,
    )


def new_state(
    question: str,
    *,
    trace_id: str | None = None,
    budget: BudgetState | None = None,
) -> ResearchState:
    """A fully populated initial state."""
    return ResearchState(
        question=question,
        premise_check={},
        plan=[],
        plan_revisions=0,
        evidence=[],
        contradictions=[],
        draft="",
        draft_mode="",
        critic_findings=[],
        critiqued_draft="",
        budget=budget or BudgetState(
            llm_calls_used=0, llm_calls_max=0,
            retrieval_calls_used=0, retrieval_calls_max=0,
            allocation={},
        ),
        terminal_decision=None,
        decision_rationale="",
        decision={},
        unresolved_gaps=[],
        final_report="",
        trace_id=trace_id or uuid.uuid4().hex[:16],
        error=None,
    )


def find_sub_question(state: ResearchState, sub_question_id: str) -> SubQuestion | None:
    for sub_question in state.get("plan") or []:
        if sub_question["id"] == sub_question_id:
            return sub_question
    return None


def pending_sub_questions(state: ResearchState) -> list[SubQuestion]:
    """Sub-questions still open for work — what the controller routes on."""
    return [
        sq for sq in (state.get("plan") or [])
        if sq["status"] not in TERMINAL_STATUSES
    ]


def all_terminal(state: ResearchState) -> bool:
    plan = state.get("plan") or []
    return bool(plan) and not pending_sub_questions(state)


def evidence_for(state: ResearchState, sub_question_id: str) -> list[Passage]:
    """Passages retrieved for a sub-question, in the order they were ranked.

    Reads `SubQuestion.evidence_ids` rather than filtering on
    `Passage.sub_question_id` — see the note on that field. Falls back to the
    old filter for sub-questions written before C17.
    """
    evidence = {p["id"]: p for p in (state.get("evidence") or [])}
    sub_question = find_sub_question(state, sub_question_id)
    ids = (sub_question or {}).get("evidence_ids") or []
    if ids:
        return [evidence[pid] for pid in ids if pid in evidence]
    return [
        p for p in evidence.values()
        if p.get("sub_question_id") == sub_question_id
    ]


def independent_sources(passages: Sequence[Passage]) -> int:
    """Feature f4: distinct origins among these passages.

    Keyed on `source_id` rather than `source_domain` so two different arXiv
    papers count as two sources, while two chunks of one paper count as one.
    """
    return len({p.get("source_id") for p in passages if p.get("source_id")})