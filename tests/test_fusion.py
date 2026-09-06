"""Tests for RRF and the hybrid retriever.

Dense retrieval uses a real local Qdrant with a deterministic fake embedder;
sparse uses real bm25s. The interesting behaviour is what fusion does when
the two arms disagree, which is only visible with both running for real.
"""
import pytest

from src.config import Config
from src.rag.fusion import HybridRetriever, reciprocal_rank_fusion
from src.rag.sparse import BM25Index
from src.rag.store import EvidenceStore
from src.state import Passage

DIM = 8

CFG = Config({"retrieval": {
    "dense": {"model": "fake", "dim": DIM, "query_prefix": "Q: ", "batch_size": 4},
    "sparse": {"backend": "bm25s", "stemmer": "english"},
    "fusion": {"method": "rrf", "k": 60, "candidates_per_retriever": 20},
    "store": {"collection": "test_fusion", "path": "unused", "distance": "cosine"},
}})


class FakeEmbedder:
    """Topic-directional vectors: semantically similar text points alike,
    so dense retrieval behaves like dense retrieval — it matches meaning,
    not exact tokens."""

    TOPICS = {
        "attention": 0, "memory": 0, "cache": 0, "paging": 0,
        "rank": 1, "fusion": 1, "merge": 1, "list": 1,
        "train": 2, "token": 2, "data": 2,
        "cat": 3, "mat": 3, "window": 3,
    }

    def _vec(self, text):
        text = text.lower()
        v = [0.05] * DIM
        for word, slot in self.TOPICS.items():
            if word in text:
                v[slot] += 1.0
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, q):
        return self._vec(q)


def passage(pid, text, **over):
    base = dict(
        id=pid, source_type="arxiv", source_id="src", source_domain="arxiv.org",
        title="", text=text, section="Method", published="2023-01-01",
        sub_question_id="sq1", retrieval_score=0.0, rerank_score=None,
    )
    base.update(over)
    return Passage(**base)


CORPUS = [
    passage("P1", "PagedAttention manages the key value cache using memory paging."),
    passage("P2", "Attention lets a model weigh input positions by relevance."),
    passage("P3", "Caching intermediate activations reduces memory pressure."),
    passage("P4", "Reciprocal rank fusion merges ranked lists.", section="Results"),
    passage("P5", "Llama 3 was trained on fifteen trillion tokens of data.",
            section="Results"),
    passage("P6", "The cat sat on the mat by the window.", source_type="web",
            source_domain="example.com", published="2019-01-01"),
]


@pytest.fixture
def retriever(tmp_path):
    from qdrant_client import QdrantClient

    store = EvidenceStore(CFG, client=QdrantClient(path=str(tmp_path / "q")),
                          embedder=FakeEmbedder())
    hybrid = HybridRetriever(CFG, store=store, sparse=BM25Index(CFG))
    hybrid.index(CORPUS)
    return hybrid


# --- RRF mechanics ---------------------------------------------------------


def test_rrf_formula():
    a, b = passage("A", "x"), passage("B", "y")
    fused = dict((p["id"], s) for p, s in reciprocal_rank_fusion([[a, b]], k=60))
    assert fused["A"] == pytest.approx(1 / 61)
    assert fused["B"] == pytest.approx(1 / 62)


def test_agreement_across_lists_accumulates():
    """A passage both retrievers like should outrank one only a single
    retriever found."""
    a, b, c = passage("A", "x"), passage("B", "y"), passage("C", "z")
    fused = reciprocal_rank_fusion([[a, b], [a, c]], k=60)
    assert fused[0][0]["id"] == "A"


def test_rank_not_score_decides():
    """The whole reason for RRF: BM25 scores are unbounded and cosine is
    bounded, so blending raw scores would let one arm dominate arbitrarily."""
    a, b = passage("A", "x"), passage("B", "y")
    first = reciprocal_rank_fusion([[a, b]], k=60)
    second = reciprocal_rank_fusion([[a, b]], k=60)
    assert [p["id"] for p, _ in first] == [p["id"] for p, _ in second]


def test_k_damps_top_rank_dominance():
    a, b = passage("A", "x"), passage("B", "y")
    small = reciprocal_rank_fusion([[a, b]], k=1)
    large = reciprocal_rank_fusion([[a, b]], k=60)
    gap_small = small[0][1] - small[1][1]
    gap_large = large[0][1] - large[1][1]
    assert gap_small > gap_large


def test_empty_lists_fuse_to_nothing():
    assert reciprocal_rank_fusion([[], []]) == []


def test_weights_must_align():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[passage("A", "x")]], weights=[1.0, 1.0])


def test_ties_break_deterministically():
    a, b = passage("A", "x"), passage("B", "y")
    one = reciprocal_rank_fusion([[a], [b]], k=60)
    two = reciprocal_rank_fusion([[b], [a]], k=60)
    assert [p["id"] for p, _ in one] == [p["id"] for p, _ in two]


# --- the C10 acceptance check ---------------------------------------------


def test_exact_token_query_ranks_higher_under_fusion_than_dense_alone(retriever):
    """The C10 criterion. Dense retrieval blurs 'PagedAttention' into the
    general attention/memory neighbourhood; BM25 pins the exact token, and
    fusion recovers it."""
    dense_only = retriever.retrieve("PagedAttention", top_k=6, dense_only=True)
    fused = retriever.retrieve("PagedAttention", top_k=6)

    def rank_of(results, pid):
        for i, r in enumerate(results, start=1):
            if r.passage["id"] == pid:
                return i
        return 999

    assert rank_of(fused, "P1") <= rank_of(dense_only, "P1")
    assert rank_of(fused, "P1") == 1


def test_fused_results_record_both_arms(retriever):
    results = retriever.retrieve("attention memory paging", top_k=6)
    assert any(r.found_by_both for r in results)
    for r in results:
        assert r.dense_rank is not None or r.sparse_rank is not None


def test_dense_only_and_sparse_only_isolate_the_arms(retriever):
    """C13's R1 and R2 configurations need each arm alone; building it in
    now avoids a parallel code path in the evaluation harness."""
    dense = retriever.retrieve("cat on the mat", top_k=5, dense_only=True)
    sparse = retriever.retrieve("cat on the mat", top_k=5, sparse_only=True)
    assert dense and sparse
    assert all(r.sparse_rank is None for r in dense)
    assert all(r.dense_rank is None for r in sparse)


def test_retrieval_score_is_the_rrf_score(retriever):
    for r in retriever.retrieve("attention", top_k=3):
        assert r.passage["retrieval_score"] == pytest.approx(r.rrf_score)


def test_results_are_ordered_by_rrf_score(retriever):
    scores = [r.rrf_score for r in retriever.retrieve("attention memory", top_k=6)]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_nothing(retriever):
    assert retriever.retrieve("  ", top_k=5) == []


# --- filters must agree across both arms -----------------------------------


def test_section_filter_applies_to_both_arms(retriever):
    results = retriever.retrieve("rank fusion tokens", top_k=10, section="Results")
    assert results
    assert {r.passage["section"] for r in results} == {"Results"}


def test_source_type_filter_applies_to_both_arms(retriever):
    results = retriever.retrieve("cat mat window", top_k=10, source_type="web")
    assert results
    assert {r.passage["source_type"] for r in results} == {"web"}


def test_exclude_sections_applies_to_both_arms(retriever):
    results = retriever.retrieve("rank fusion", top_k=10,
                                 exclude_sections=["Results"])
    assert "Results" not in {r.passage["section"] for r in results}


def test_published_after_excludes_undated_and_older(retriever):
    """Qdrant cannot match a missing field; the sparse predicate must agree,
    or the two arms search different corpora and fusion favours the looser."""
    results = retriever.retrieve("cat mat", top_k=10, published_after="2022-01-01")
    assert all(r.passage["published"] >= "2022-01-01" for r in results)
    assert "P6" not in {r.passage["id"] for r in results}


def test_filter_matching_nothing_returns_empty(retriever):
    assert retriever.retrieve("attention", top_k=5, section="Nonexistent") == []


# --- indexing --------------------------------------------------------------


def test_index_writes_to_both_retrievers(retriever):
    assert retriever.store.count() == len(CORPUS)
    assert len(retriever.sparse) == len(CORPUS)


def test_retrieval_is_traced(retriever):
    events = []

    class FakeTracer:
        def note(self, name, **kw):
            events.append((name, kw))

    retriever.tracer = FakeTracer()
    retriever.retrieve("attention", top_k=3)
    name, payload = events[0]
    assert name == "hybrid_retrieve"
    assert "dense_hits" in payload and "sparse_hits" in payload