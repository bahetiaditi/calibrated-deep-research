"""Shared state types.

Partial by design. C6 needs `Passage` to exist before C14 defines the full
`ResearchState`, so the passage schema and the shared literals land here now;
`SubQuestion`, `ClaimVerification`, `BudgetState` and `ResearchState` follow
at C14. Splitting it this way avoids a circular dependency between the tools
and the graph.
"""
from __future__ import annotations

import hashlib
from typing import Literal, TypedDict

Route = Literal["arxiv_meta", "arxiv_fulltext", "web", "evidence_store"]
Status = Literal["pending", "in_progress", "sufficient", "insufficient", "abandoned"]
Verdict = Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"]
Terminal = Literal["ANSWER", "PARTIAL", "ABSTAIN"]
SourceType = Literal["arxiv", "web"]


class Passage(TypedDict):
    """One retrievable unit of evidence.

    On `id`: REFERENCE.md §3.2 sketched sequential ids ("P0001 …"). They are
    content-addressed instead — `make_passage_id` hashes source, section and
    text. Two reasons. The evidence store accumulates across runs (§4.1), so
    a per-run counter would collide between runs and re-ingesting the same
    paper would duplicate it. And a stable id means a cached passage is
    recognisably the same passage.

    The cost is that ids are not friendly for an LLM to transcribe, which
    matters because the synthesizer must reproduce them exactly (C18). That
    is solved there with a per-prompt label map (P01…P20 → id), not by making
    the storage id fragile.
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
    rerank_score: float | None


def make_passage_id(source_id: str, text: str, section: str | None = None) -> str:
    """Deterministic, collision-resistant passage id.

    Same source + same section + same text always yields the same id, so
    re-ingesting a paper is idempotent rather than duplicative.
    """
    digest = hashlib.sha256(
        "\x00".join([source_id, section or "", text]).encode("utf-8")
    ).hexdigest()
    return f"P{digest[:12]}"