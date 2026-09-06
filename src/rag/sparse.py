"""Sparse retrieval (BM25).

Why sparse matters here specifically
------------------------------------
Research questions are dense with exact-match tokens that embeddings blur:
model names (Mamba-2, Qwen3), method names (PagedAttention, FlashAttention-2),
metric names (nDCG, relaxed accuracy), library names (bm25s). A dense encoder
maps "PagedAttention" and "attention paging" to nearby vectors, which is
usually helpful and occasionally catastrophic — when a sub-question asks what
*PagedAttention* does, a passage about attention in general is a wrong answer
that looks right.

BM25 has the opposite failure mode, so fusing the two (C10, `fusion.py`)
recovers most of both.

Two decisions this file makes
-----------------------------
**Digits are kept.** bm25s's default token pattern requires two or more word
characters, which silently discards the "2" in "FlashAttention-2" and the "3"
in "Llama 3". Version numbers are precisely the tokens a research question
turns on, so the pattern is widened to keep them.

**Filtering is post-hoc, and that is a real limitation.** Qdrant applies
payload filters *during* graph traversal; BM25 has no equivalent, so a
filtered sparse search oversamples and then drops non-matching results. This
is exactly the post-filter regime Project 1 characterises, and at low
selectivity it degrades — which is worth saying out loud rather than
pretending the two retrievers are symmetric. Oversampling bounds the damage;
it does not remove it.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

from src.config import Config, get_config
from src.state import Passage

log = logging.getLogger(__name__)

# Keep single characters and digits: "Llama 3", "GPT-4", "FlashAttention-2".
# bm25s defaults to r"(?u)\b\w\w+\b", which drops them.
TOKEN_PATTERN = r"(?u)\b\w+\b"

# How much to oversample before applying a payload filter. Post-filtering a
# top-k list throws away matches; asking for more up front bounds the loss.
FILTER_OVERSAMPLE = 8


class BM25Index:
    """In-memory BM25 index over `Passage` objects."""

    def __init__(self, config: Config | None = None, *, stemmer: Any = None) -> None:
        cfg = config or get_config()
        self.stemmer_name = str(cfg.get("retrieval.sparse.stemmer", "english"))
        self._stemmer = stemmer
        self._index: Any = None
        self._passages: list[Passage] = []
        self._by_id: dict[str, int] = {}

    # -- tokenisation -------------------------------------------------------

    @property
    def stemmer(self) -> Any:
        if self._stemmer is None and self.stemmer_name:
            try:
                import Stemmer

                self._stemmer = Stemmer.Stemmer(self.stemmer_name)
            except ImportError:
                # PyStemmer is optional; BM25 works unstemmed, just less well
                # at matching plurals and inflections.
                log.info("PyStemmer unavailable — running BM25 without stemming")
                self._stemmer = False
        return self._stemmer or None

    def _tokenize(self, texts: Sequence[str], *, for_query: bool) -> Any:
        import bm25s

        return bm25s.tokenize(
            list(texts),
            lower=True,
            token_pattern=TOKEN_PATTERN,
            stopwords="english",
            stemmer=self.stemmer,
            show_progress=False,
            # Queries must map onto the index's vocabulary, so ids are only
            # meaningful at build time.
            return_ids=not for_query,
        )

    # -- building -----------------------------------------------------------

    def build(self, passages: Sequence[Passage]) -> int:
        """Build (or rebuild) the index. Later passages win on duplicate ids."""
        import bm25s

        deduped: dict[str, Passage] = {}
        for passage in passages:
            if passage.get("text"):
                deduped[passage["id"]] = passage

        self._passages = list(deduped.values())
        self._by_id = {p["id"]: i for i, p in enumerate(self._passages)}

        if not self._passages:
            self._index = None
            return 0

        corpus_tokens = self._tokenize(
            [_indexable_text(p) for p in self._passages], for_query=False
        )
        self._index = bm25s.BM25()
        self._index.index(corpus_tokens, show_progress=False)
        return len(self._passages)

    def add(self, passages: Sequence[Passage]) -> int:
        """Add passages and rebuild.

        bm25s has no incremental insert. A full rebuild is O(corpus) but fast
        (thousands of passages in well under a second), and correctness beats
        cleverness on a corpus this size.
        """
        return self.build(list(self._passages) + list(passages))

    def __len__(self) -> int:
        return len(self._passages)

    # -- searching ----------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        predicate: Callable[[Passage], bool] | None = None,
    ) -> list[tuple[Passage, float]]:
        """Return (passage, score) pairs, best first.

        `predicate` is applied AFTER retrieval — see the module docstring on
        why that is a genuine limitation rather than an implementation detail.
        """
        if self._index is None or not query.strip():
            return []

        want = top_k * FILTER_OVERSAMPLE if predicate else top_k
        want = min(want, len(self._passages))
        if want <= 0:
            return []

        query_tokens = self._tokenize([query], for_query=True)
        try:
            indices, scores = self._index.retrieve(
                query_tokens, k=want, show_progress=False
            )
        except Exception as exc:  # noqa: BLE001
            # A query whose every token is out of vocabulary is a legitimate
            # miss, not a crash.
            log.debug("bm25 retrieve failed for %r: %s", query[:60], exc)
            return []

        out: list[tuple[Passage, float]] = []
        for position, score in zip(indices[0], scores[0]):
            if score <= 0:
                continue
            passage = self._passages[int(position)]
            if predicate is not None and not predicate(passage):
                continue
            out.append((passage, float(score)))
            if len(out) >= top_k:
                break
        return out

    def get(self, passage_id: str) -> Passage | None:
        position = self._by_id.get(passage_id)
        return self._passages[position] if position is not None else None


def _indexable_text(passage: Passage) -> str:
    """Title plus body.

    The title carries the paper's own name — the single highest-signal exact
    match available when a sub-question names a method or model.
    """
    title = passage.get("title") or ""
    text = passage.get("text") or ""
    return f"{title}. {text}" if title else text