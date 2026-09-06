"""Reciprocal Rank Fusion and the unified retrieval entry point.

Why RRF rather than score blending
----------------------------------
Dense cosine similarity lives in [-1, 1] and clusters tightly around 0.5-0.9;
BM25 is unbounded and corpus-dependent, routinely producing scores from 2 to
30. Combining them by weighted sum requires normalising two distributions
whose shapes differ per query — you end up tuning a normalisation scheme that
silently re-weights every query differently.

RRF (Cormack et al., 2009) sidesteps this entirely by using **rank, not
score**: `score = Σ 1/(k + rank_i)` with k=60. It is parameter-free in
practice, robust to either retriever producing garbage scores, and it is what
the retrieval literature converged on. The cost is that it discards magnitude
— a document ranked first by a huge margin scores the same as one ranked
first by a hair. That is a real trade and C13's R1-R4 table is what tests
whether it was the right one on this corpus.

The asymmetry worth knowing
---------------------------
The dense side filters *during* traversal (Qdrant payload conditions inside
the HNSW search). The sparse side has no such mechanism and post-filters.
So a low-selectivity filter degrades the sparse arm specifically — which is
Project 1's regime appearing in a live system, and the honest bridge between
the two projects.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from src.config import Config, get_config
from src.state import Passage

log = logging.getLogger(__name__)


@dataclass
class FusedResult:
    passage: Passage
    rrf_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None

    @property
    def found_by_both(self) -> bool:
        return self.dense_rank is not None and self.sparse_rank is not None


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Passage]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[Passage, float]]:
    """Fuse ranked lists by reciprocal rank.

    `k` damps the influence of top ranks: with k=60 the gap between rank 1 and
    rank 2 is small, so a single retriever cannot dominate the fusion on the
    strength of one confident hit.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must align with ranked_lists")

    scores: dict[str, float] = {}
    passages: dict[str, Passage] = {}

    for ranked, weight in zip(ranked_lists, weights):
        for rank, passage in enumerate(ranked, start=1):
            pid = passage["id"]
            scores[pid] = scores.get(pid, 0.0) + weight * (1.0 / (k + rank))
            passages.setdefault(pid, passage)

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(passages[pid], score) for pid, score in ordered]


class HybridRetriever:
    """Dense + sparse retrieval fused by RRF.

    This is the unified `retrieve(query, filters, top_k)` entry point the rest
    of the system uses; nothing above this layer should know that two
    retrievers exist.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        store: Any = None,
        sparse: Any = None,
        tracer: Any = None,
    ) -> None:
        self.cfg = config or get_config()
        fusion = self.cfg.section("retrieval.fusion")
        self.k = int(fusion.get("k", 60))
        self.candidates = int(fusion.get("candidates_per_retriever", 50))
        self.tracer = tracer
        self._store = store
        self._sparse = sparse

    @property
    def store(self) -> Any:
        if self._store is None:
            from src.rag.store import EvidenceStore

            self._store = EvidenceStore(self.cfg)
        return self._store

    @property
    def sparse(self) -> Any:
        if self._sparse is None:
            from src.rag.sparse import BM25Index

            self._sparse = BM25Index(self.cfg)
        return self._sparse

    # -- indexing -----------------------------------------------------------

    def index(self, passages: Sequence[Passage]) -> int:
        """Write passages to both retrievers.

        Kept as one call deliberately: two indexes that can drift out of sync
        produce a retriever that finds a passage densely but cannot score it
        sparsely, and the resulting fusion is quietly wrong.
        """
        written = self.store.upsert(passages)
        self.sparse.add(passages)
        return written

    # -- retrieval ----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        section: str | Sequence[str] | None = None,
        source_type: str | None = None,
        source_domain: str | Sequence[str] | None = None,
        published_after: str | None = None,
        exclude_sections: Sequence[str] | None = None,
        dense_only: bool = False,
        sparse_only: bool = False,
    ) -> list[FusedResult]:
        """Hybrid retrieval with optional payload filters.

        `dense_only` and `sparse_only` exist for C13's R1/R2 configurations —
        the retrieval evaluation needs to run each arm in isolation, and
        building that in now avoids a parallel code path later.
        """
        query = (query or "").strip()
        if not query:
            return []

        dense: list[Passage] = []
        sparse: list[tuple[Passage, float]] = []

        if not sparse_only:
            dense = self.store.search(
                query,
                top_k=self.candidates,
                section=section,
                source_type=source_type,
                source_domain=source_domain,
                published_after=published_after,
                exclude_sections=exclude_sections,
            )

        if not dense_only:
            predicate = _make_predicate(
                section=section,
                source_type=source_type,
                source_domain=source_domain,
                published_after=published_after,
                exclude_sections=exclude_sections,
            )
            sparse = self.sparse.search(
                query, top_k=self.candidates, predicate=predicate
            )

        results = self._fuse(dense, sparse, top_k=top_k)
        if self.tracer is not None:
            self.tracer.note(
                "hybrid_retrieve",
                query=query[:200],
                dense_hits=len(dense),
                sparse_hits=len(sparse),
                fused=len(results),
                both=sum(1 for r in results if r.found_by_both),
                filtered=bool(section or source_type or source_domain
                              or published_after or exclude_sections),
            )
        return results

    def _fuse(
        self,
        dense: Sequence[Passage],
        sparse: Sequence[tuple[Passage, float]],
        *,
        top_k: int,
    ) -> list[FusedResult]:
        dense_rank = {p["id"]: i + 1 for i, p in enumerate(dense)}
        dense_score = {p["id"]: p.get("retrieval_score") for p in dense}
        sparse_rank = {p["id"]: i + 1 for i, (p, _) in enumerate(sparse)}
        sparse_score = {p["id"]: s for p, s in sparse}

        fused = reciprocal_rank_fusion(
            [list(dense), [p for p, _ in sparse]], k=self.k
        )
        out: list[FusedResult] = []
        for passage, score in fused[:top_k]:
            pid = passage["id"]
            merged = dict(passage)
            merged["retrieval_score"] = score
            out.append(
                FusedResult(
                    passage=merged,  # type: ignore[arg-type]
                    rrf_score=score,
                    dense_rank=dense_rank.get(pid),
                    sparse_rank=sparse_rank.get(pid),
                    dense_score=dense_score.get(pid),
                    sparse_score=sparse_score.get(pid),
                )
            )
        return out


def _make_predicate(
    *,
    section: str | Sequence[str] | None,
    source_type: str | None,
    source_domain: str | Sequence[str] | None,
    published_after: str | None,
    exclude_sections: Sequence[str] | None,
) -> Callable[[Passage], bool] | None:
    """Mirror the Qdrant payload filter for the sparse arm.

    The two must agree exactly. A mismatch would mean the arms search
    different corpora and the fusion silently favours whichever arm has the
    looser filter.
    """
    if not any([section, source_type, source_domain, published_after,
                exclude_sections]):
        return None

    def as_set(value: str | Sequence[str] | None) -> set[str] | None:
        if value is None:
            return None
        return {value} if isinstance(value, str) else set(value)

    sections = as_set(section)
    domains = as_set(source_domain)
    excluded = as_set(exclude_sections) or set()

    def predicate(passage: Passage) -> bool:
        if sections is not None and passage.get("section") not in sections:
            return False
        if source_type is not None and passage.get("source_type") != source_type:
            return False
        if domains is not None and passage.get("source_domain") not in domains:
            return False
        if passage.get("section") in excluded:
            return False
        if published_after is not None:
            published = passage.get("published")
            # A passage with no date cannot be shown to satisfy a date
            # constraint, so it is excluded — matching Qdrant, which cannot
            # match a missing field either.
            if not published or published < published_after:
                return False
        return True

    return predicate