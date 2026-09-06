"""Tests for BM25 sparse retrieval.

Real bm25s throughout — the point of the sparse arm is exact-token matching,
and a mock would test our assumption about tokenisation rather than what the
tokeniser actually does.
"""
import pytest

from src.config import Config
from src.rag.sparse import TOKEN_PATTERN, BM25Index
from src.state import Passage

CFG = Config({"retrieval": {"sparse": {"backend": "bm25s", "stemmer": "english"}}})


def passage(pid, text, **over):
    base = dict(
        id=pid, source_type="arxiv", source_id="src", source_domain="arxiv.org",
        title="", text=text, section="Method", published="2023-01-01",
        sub_question_id="sq1", retrieval_score=0.0, rerank_score=None,
    )
    base.update(over)
    return Passage(**base)


CORPUS = [
    passage("P1", "PagedAttention manages the key value cache with virtual memory paging."),
    passage("P2", "Attention mechanisms let models weigh different input positions."),
    passage("P3", "FlashAttention-2 improves work partitioning across thread blocks."),
    passage("P4", "Llama 3 was trained on fifteen trillion tokens of public data."),
    passage("P5", "Reciprocal rank fusion merges ranked lists without normalisation.",
            section="Results"),
    passage("P6", "The cat sat quietly on a warm mat by the window.", source_type="web",
            source_domain="example.com"),
]


@pytest.fixture
def index():
    idx = BM25Index(CFG)
    idx.build(CORPUS)
    return idx


# --- exact-token matching, the reason sparse exists ------------------------


def test_exact_method_name_ranks_first(index):
    """A dense encoder maps 'PagedAttention' near 'attention paging'. When a
    sub-question asks what PagedAttention does, that is a wrong answer that
    looks right. BM25 does not make that mistake."""
    results = index.search("PagedAttention", top_k=3)
    assert results and results[0][0]["id"] == "P1"


def test_version_numbers_survive_tokenisation():
    """bm25s defaults to requiring two word characters, silently dropping
    the '2' in FlashAttention-2 and the '3' in Llama 3 — exactly the tokens
    a research question turns on."""
    assert TOKEN_PATTERN == r"(?u)\b\w+\b"
    idx = BM25Index(CFG)
    idx.build(CORPUS)
    results = idx.search("Llama 3", top_k=3)
    assert results and results[0][0]["id"] == "P4"


def test_title_is_indexed(index):
    """A paper's own name is the highest-signal exact match available."""
    idx = BM25Index(CFG)
    idx.build([passage("PT", "Some body text about nothing in particular.",
                       title="Mamba Linear-Time Sequence Modeling")])
    assert idx.search("Mamba", top_k=1)


def test_irrelevant_query_scores_nothing(index):
    assert index.search("zzzz quuxbar nonexistentterm", top_k=5) == []


def test_empty_query_returns_empty(index):
    assert index.search("   ", top_k=5) == []


def test_top_k_is_respected(index):
    assert len(index.search("attention", top_k=2)) <= 2


def test_scores_are_descending(index):
    scores = [s for _, s in index.search("attention memory", top_k=5)]
    assert scores == sorted(scores, reverse=True)


# --- building --------------------------------------------------------------


def test_build_reports_corpus_size(index):
    assert len(index) == len(CORPUS)


def test_duplicate_ids_collapse():
    idx = BM25Index(CFG)
    assert idx.build([passage("P1", "one"), passage("P1", "two")]) == 1


def test_passages_without_text_are_skipped():
    idx = BM25Index(CFG)
    assert idx.build([passage("P1", ""), passage("P2", "real content here")]) == 1


def test_empty_corpus_searches_safely():
    idx = BM25Index(CFG)
    idx.build([])
    assert idx.search("anything", top_k=5) == []


def test_add_extends_the_corpus(index):
    index.add([passage("P7", "Qdrant supports payload filtering natively.")])
    assert len(index) == len(CORPUS) + 1
    assert index.search("Qdrant", top_k=1)[0][0]["id"] == "P7"


def test_get_by_id(index):
    assert index.get("P3")["id"] == "P3"
    assert index.get("nope") is None


# --- filtering (post-hoc, and that matters) --------------------------------


def test_predicate_filters_results(index):
    results = index.search(
        "attention", top_k=5, predicate=lambda p: p["section"] == "Results"
    )
    assert all(p["section"] == "Results" for p, _ in results)


def test_predicate_matching_nothing_returns_empty(index):
    assert index.search("attention", top_k=5,
                        predicate=lambda p: p["section"] == "Nonexistent") == []


def test_oversampling_recovers_matches_beyond_top_k(index):
    """Post-filtering a top-k list throws away matches. Oversampling bounds
    the loss — this is Project 1's post-filter regime, live."""
    results = index.search(
        "attention memory data fusion cat", top_k=1,
        predicate=lambda p: p["source_type"] == "web",
    )
    assert results and results[0][0]["id"] == "P6"