"""Tests for the embedder.

The point of these is the prefix. bge-small is asymmetric: queries take an
instruction prefix, documents do not. Getting it wrong raises nothing — recall
just drops, and C13's retrieval evaluation would faithfully measure a crippled
system and report the number as a finding.
"""
import pytest

from src.config import Config
from src.rag.embed import Embedder

PREFIX = "Represent this sentence for searching relevant passages: "

CFG = Config({"retrieval": {"dense": {
    "model": "BAAI/bge-small-en-v1.5", "dim": 384,
    "query_prefix": PREFIX, "batch_size": 8,
}}})


class FakeModel:
    """Records exactly what it was asked to encode."""

    def __init__(self, dim=384):
        self.dim = dim
        self.seen: list[list[str]] = []
        self.kwargs: list[dict] = []

    def encode(self, texts, **kw):
        self.seen.append(list(texts))
        self.kwargs.append(kw)
        return [[float(len(t) % 7) / 7.0] * self.dim for t in texts]


def embedder(model=None):
    return Embedder(CFG, model=model or FakeModel())


# --- the prefix ------------------------------------------------------------


def test_query_gets_the_prefix():
    model = FakeModel()
    embedder(model).embed_query("what is RRF")
    assert model.seen[0] == [PREFIX + "what is RRF"]


def test_documents_do_not_get_the_prefix():
    """Prefixing documents is as wrong as omitting it on queries, and just
    as silent."""
    model = FakeModel()
    embedder(model).embed_documents(["a passage about fusion"])
    assert model.seen[0] == ["a passage about fusion"]
    assert PREFIX not in model.seen[0][0]


def test_batch_queries_all_get_the_prefix():
    model = FakeModel()
    embedder(model).embed_queries(["one", "two"])
    assert model.seen[0] == [PREFIX + "one", PREFIX + "two"]


def test_empty_prefix_is_rejected_at_construction():
    """A blank prefix is a config edit gone wrong; failing loudly beats
    halving retrieval quality silently."""
    cfg = Config({"retrieval": {"dense": {
        "model": "m", "dim": 384, "query_prefix": "   ",
    }}})
    with pytest.raises(ValueError, match="query_prefix"):
        Embedder(cfg, model=FakeModel())


# --- encoding contract -----------------------------------------------------


def test_vectors_are_normalised_for_cosine():
    model = FakeModel()
    embedder(model).embed_documents(["x"])
    assert model.kwargs[0]["normalize_embeddings"] is True


def test_batch_size_comes_from_config():
    model = FakeModel()
    embedder(model).embed_documents(["a", "b"])
    assert model.kwargs[0]["batch_size"] == 8


def test_dimension_matches_config():
    assert len(embedder().embed_query("q")) == 384


def test_empty_inputs_short_circuit():
    model = FakeModel()
    e = embedder(model)
    assert e.embed_documents([]) == []
    assert e.embed_queries([]) == []
    assert model.seen == []


def test_returns_plain_floats():
    vector = embedder().embed_documents(["x"])[0]
    assert all(isinstance(v, float) for v in vector)