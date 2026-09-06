"""Tests for the evidence store.

Uses a REAL Qdrant client in local mode with a fake embedder. Mocking Qdrant
would test our belief about how payload filtering works rather than how it
actually works, and filtered retrieval is the whole reason Qdrant was chosen
over Chroma (§4.3).

The fake embedder maps text to a deterministic vector so similarity is
predictable without loading a 130MB model into the test suite.
"""
import pytest

from src.config import Config
from src.rag.store import EvidenceStore, point_id
from src.state import Passage

DIM = 8

CFG = Config({"retrieval": {
    "dense": {"model": "fake", "dim": DIM, "query_prefix": "Q: ", "batch_size": 4},
    "store": {"collection": "test_evidence", "path": "unused", "distance": "cosine"},
}})


class FakeEmbedder:
    """Deterministic, direction-carrying vectors.

    Texts sharing a keyword point the same way, so 'closest to X' is
    predictable without a real model.
    """

    KEYWORDS = ["bleu", "attention", "fusion", "rank", "memory", "quantis", "chunk"]

    def _vec(self, text: str) -> list[float]:
        text = text.lower()
        vector = [0.05] * DIM
        for i, kw in enumerate(self.KEYWORDS):
            if kw in text:
                vector[i % DIM] += 1.0
        norm = sum(v * v for v in vector) ** 0.5
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, query):
        return self._vec(query)


def passage(pid, text, **over):
    base = dict(
        id=pid, source_type="arxiv", source_id="1706.03762v5",
        source_domain="arxiv.org", title="Attention Is All You Need",
        text=text, section="Results", published="2017-06-12",
        sub_question_id="sq1", retrieval_score=0.0, rerank_score=None,
    )
    base.update(over)
    return Passage(**base)


@pytest.fixture
def store(tmp_path):
    from qdrant_client import QdrantClient

    return EvidenceStore(
        CFG,
        client=QdrantClient(path=str(tmp_path / "qdrant")),
        embedder=FakeEmbedder(),
    )


@pytest.fixture
def populated(store):
    """50 passages, as the C9 acceptance check specifies."""
    passages = []
    for i in range(50):
        section = ["Results", "Related Work", "Method", "Abstract", "Introduction"][i % 5]
        topic = ["bleu score", "attention head", "fusion rank", "memory use",
                 "quantisation"][i % 5]
        passages.append(passage(
            f"P{i:012d}",
            f"Passage {i} about {topic} in detail.",
            section=section,
            source_type="arxiv" if i % 3 else "web",
            source_domain="arxiv.org" if i % 3 else "example.com",
            published=f"20{15 + (i % 8)}-01-01",
        ))
    store.upsert(passages)
    return store


# --- write and count -------------------------------------------------------


def test_store_fifty_passages(populated):
    assert populated.count() == 50


def test_upsert_is_idempotent(store):
    """Re-ingesting a paper must overwrite, not duplicate — the idempotency
    established at C8 has to survive into storage."""
    p = passage("Pabc", "attention is all you need")
    store.upsert([p])
    store.upsert([p])
    assert store.count() == 1


def test_point_ids_are_deterministic():
    assert point_id("Pabc") == point_id("Pabc")
    assert point_id("Pabc") != point_id("Pdef")


def test_empty_upsert_is_a_no_op(store):
    assert store.upsert([]) == 0


def test_passages_without_text_are_skipped(store):
    assert store.upsert([passage("Pempty", "")]) == 0


# --- similarity search -----------------------------------------------------


def test_similarity_search_returns_relevant_passages(populated):
    results = populated.search("bleu score", top_k=5)
    assert results
    assert any("bleu" in r["text"].lower() for r in results)


def test_search_respects_top_k(populated):
    assert len(populated.search("attention", top_k=3)) == 3


def test_search_populates_retrieval_score(populated):
    for r in populated.search("fusion", top_k=3):
        assert isinstance(r["retrieval_score"], float)


def test_results_preserve_the_passage_schema(populated):
    result = populated.search("attention", top_k=1)[0]
    assert set(Passage.__annotations__) <= set(result)


# --- payload filtering (the reason Qdrant was chosen) ----------------------


def test_filter_by_section(populated):
    """The C9 acceptance check: section == 'Results'."""
    results = populated.search("attention", top_k=20, section="Results")
    assert results
    assert {r["section"] for r in results} == {"Results"}


def test_filter_by_several_sections(populated):
    results = populated.search(
        "attention", top_k=30, section=["Results", "Method"]
    )
    assert {r["section"] for r in results} <= {"Results", "Method"}


def test_filter_by_source_type(populated):
    results = populated.search("memory", top_k=20, source_type="web")
    assert results
    assert {r["source_type"] for r in results} == {"web"}


def test_exclude_sections(populated):
    """A claim from Related Work describes someone else's contribution;
    excluding it is a real query the retriever will make."""
    results = populated.search(
        "attention", top_k=30, exclude_sections=["Related Work"]
    )
    assert "Related Work" not in {r["section"] for r in results}


def test_filter_by_published_after(populated):
    results = populated.search("attention", top_k=30, published_after="2020-01-01")
    assert results
    assert all(r["published"] >= "2020-01-01" for r in results)


def test_combined_filters(populated):
    results = populated.search(
        "attention", top_k=30, section="Results", source_type="arxiv"
    )
    for r in results:
        assert r["section"] == "Results" and r["source_type"] == "arxiv"


def test_low_selectivity_filter_still_returns_results(populated):
    """Project 1's regime, live: filtering during traversal rather than after
    is why a narrow filter does not collapse recall here."""
    results = populated.search(
        "quantisation", top_k=10, section="Abstract", source_type="arxiv"
    )
    assert results


def test_no_filter_arguments_yields_no_filter():
    assert EvidenceStore.build_filter() is None


def test_filter_matching_nothing_returns_empty(populated):
    assert populated.search("attention", top_k=10, section="Nonexistent") == []


# --- lookup by id ----------------------------------------------------------


def test_get_by_id(populated):
    """The C9 acceptance check: retrieve by ID — what the critic uses to
    fetch the passage a claim cites."""
    got = populated.get("P000000000007")
    assert got is not None and got["id"] == "P000000000007"


def test_get_many(populated):
    ids = ["P000000000001", "P000000000002", "P000000000003"]
    assert {p["id"] for p in populated.get_many(ids)} == set(ids)


def test_get_missing_id_returns_none(populated):
    assert populated.get("Pdoesnotexist") is None


def test_get_many_empty(populated):
    assert populated.get_many([]) == []


# --- collection setup ------------------------------------------------------


def test_ensure_collection_is_idempotent(store):
    store.ensure_collection()
    store.ensure_collection()
    assert store.count() == 0


def test_collection_created_lazily_on_first_search(store):
    assert store.search("anything", top_k=3) == []