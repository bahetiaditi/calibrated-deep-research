"""Tests for PDF ingestion.

No network. The downloader and extractor are injected so caching, failure
handling and passage construction are exercised without hitting arXiv — which
would be both slow (3s etiquette delay) and rude.
"""
import json

import pytest

from src.config import Config
from src.rag.ingest import MIN_USABLE_CHARS, PDFCache, PDFIngestor
from src.state import Passage

PDF_BYTES = b"%PDF-1.4\n" + b"x" * 500

PAPER_TEXT = """Some Paper Title
Authors Here

Abstract
""" + ("We present a method. " * 30) + """

1 Introduction
""" + ("Background context sentence. " * 40) + """

2 Related Work
""" + ("Someone else did this. " * 40) + """

3 Results
""" + ("We measure 42.1 BLEU. " * 40) + """

References
[1] Author. Title. Venue, 2020.
"""


def cfg(**over):
    data = {
        "sources": {"arxiv": {"pdf_cache_dir": over.pop("cache_dir", "data/cache/pdf")}},
        "chunking": {
            "paper": {"size_chars": 400, "overlap_chars": 80},
            "web": {"size_chars": 300, "overlap_chars": 60},
        },
    }
    return Config(data)


def ingestor(tmp_path, *, downloader=None, extractor=None, **kw):
    calls = {"downloads": 0}

    def default_downloader(url, timeout):
        calls["downloads"] += 1
        return PDF_BYTES

    ing = PDFIngestor(
        cfg(),
        cache=PDFCache(tmp_path / "pdf"),
        downloader=downloader or default_downloader,
        extractor=extractor or (lambda path: PAPER_TEXT),
        **kw,
    )
    return ing, calls


# --- caching ---------------------------------------------------------------


def test_download_then_cache_hit(tmp_path):
    """§4.1: a paper downloaded during question 7 must be free during
    question 23 — this is what makes the evidence store a store."""
    ing, calls = ingestor(tmp_path)
    first = ing.fetch("https://arxiv.org/pdf/1706.03762")
    second = ing.fetch("https://arxiv.org/pdf/1706.03762")
    assert first.from_cache is False and second.from_cache is True
    assert calls["downloads"] == 1


def test_cache_records_content_hash(tmp_path):
    ing, _ = ingestor(tmp_path)
    result = ing.fetch("https://arxiv.org/pdf/x")
    meta = json.loads(next((tmp_path / "pdf").glob("*.json")).read_text())
    assert meta["content_sha256"] == result.content_sha256
    assert meta["bytes"] == len(PDF_BYTES)


def test_cache_survives_a_new_instance(tmp_path):
    ing_a, calls_a = ingestor(tmp_path)
    ing_a.fetch("https://arxiv.org/pdf/y")
    ing_b, calls_b = ingestor(tmp_path)
    assert ing_b.fetch("https://arxiv.org/pdf/y").from_cache is True
    assert calls_b["downloads"] == 0


def test_different_urls_cache_separately(tmp_path):
    ing, calls = ingestor(tmp_path)
    ing.fetch("https://arxiv.org/pdf/a")
    ing.fetch("https://arxiv.org/pdf/b")
    assert calls["downloads"] == 2


# --- failure handling ------------------------------------------------------


def test_download_failure_returns_none(tmp_path):
    def boom(url, timeout):
        raise ConnectionError("reset")

    ing, _ = ingestor(tmp_path, downloader=boom)
    assert ing.fetch("https://arxiv.org/pdf/x") is None


def test_html_error_page_is_rejected(tmp_path):
    """An HTML error page saved as .pdf fails confusingly much later."""
    ing, _ = ingestor(tmp_path, downloader=lambda u, t: b"<html>404</html>")
    assert ing.fetch("https://arxiv.org/pdf/x") is None


def test_failed_download_is_not_cached(tmp_path):
    ing, _ = ingestor(tmp_path, downloader=lambda u, t: b"<html>")
    ing.fetch("https://arxiv.org/pdf/x")
    assert list((tmp_path / "pdf").glob("*.pdf")) == []


def test_extraction_failure_returns_empty(tmp_path):
    def boom(path):
        raise ValueError("corrupt xref table")

    ing, _ = ingestor(tmp_path, extractor=boom)
    assert ing.ingest_paper("https://x/p", source_id="1") == []


def test_scanned_pdf_with_no_text_layer_yields_nothing(tmp_path):
    """OCR is out of scope. Reporting nothing beats feeding the reranker
    a page of extraction noise."""
    ing, _ = ingestor(tmp_path, extractor=lambda p: "x" * (MIN_USABLE_CHARS - 1))
    assert ing.ingest_paper("https://x/p", source_id="1") == []


# --- passage construction --------------------------------------------------


def test_ingest_produces_section_tagged_passages(tmp_path):
    ing, _ = ingestor(tmp_path)
    passages = ing.ingest_paper(
        "https://arxiv.org/pdf/1706.03762",
        source_id="1706.03762v5",
        sub_question_id="sq1",
        published="2017-06-12",
        title="Attention Is All You Need",
    )
    assert passages
    sections = {p["section"] for p in passages}
    assert {"Abstract", "Introduction", "Related Work", "Results"} <= sections
    assert "References" not in sections


def test_passages_match_the_schema(tmp_path):
    ing, _ = ingestor(tmp_path)
    p = ing.ingest_paper("https://x/p", source_id="1", sub_question_id="sq1")[0]
    assert set(p) == set(Passage.__annotations__)
    assert p["source_type"] == "arxiv"
    assert p["sub_question_id"] == "sq1"


def test_related_work_and_results_never_share_a_chunk(tmp_path):
    """The misattribution failure this whole design exists to prevent."""
    ing, _ = ingestor(tmp_path)
    passages = ing.ingest_paper("https://x/p", source_id="1")
    for p in passages:
        if p["section"] == "Related Work":
            assert "BLEU" not in p["text"]
        if p["section"] == "Results":
            assert "Someone else" not in p["text"]


def test_passage_ids_are_stable_across_ingests(tmp_path):
    ing, _ = ingestor(tmp_path)
    a = ing.ingest_paper("https://x/p", source_id="1")
    b = ing.ingest_paper("https://x/p", source_id="1")
    assert [p["id"] for p in a] == [p["id"] for p in b]


def test_same_text_in_different_sections_gets_different_ids(tmp_path):
    from src.state import make_passage_id
    assert (make_passage_id("1", "identical text", "Results")
            != make_passage_id("1", "identical text", "Related Work"))


# --- web passage splitting -------------------------------------------------


def test_long_web_passage_is_split(tmp_path):
    ing, _ = ingestor(tmp_path)
    varied = " ".join(f"Fact number {i} is recorded here." for i in range(200))
    passage = Passage(
        id="Pold", source_type="web", source_id="https://a.com/x",
        source_domain="a.com", title="T", text=varied,
        section=None, published=None, sub_question_id="sq1",
        retrieval_score=0.5, rerank_score=None,
    )
    parts = ing.chunk_web_passage(passage)
    assert len(parts) > 1
    assert all(p["source_id"] == "https://a.com/x" for p in parts)
    assert len({p["id"] for p in parts}) == len(parts)


def test_byte_identical_chunks_are_deduplicated(tmp_path):
    """Content-addressed ids mean repeated text is the SAME evidence.
    Returning it twice would let one passage count twice toward f2 and f4,
    inflating sufficiency exactly when evidence is thin."""
    ing, _ = ingestor(tmp_path)
    passage = Passage(
        id="Pold", source_type="web", source_id="https://a.com/x",
        source_domain="a.com", title="T", text="Sentence here. " * 200,
        section=None, published=None, sub_question_id="sq1",
        retrieval_score=0.5, rerank_score=None,
    )
    parts = ing.chunk_web_passage(passage)
    assert len({p["id"] for p in parts}) == len(parts)


def test_short_web_passage_passes_through_unchanged(tmp_path):
    ing, _ = ingestor(tmp_path)
    passage = Passage(
        id="Pold", source_type="web", source_id="u", source_domain="d",
        title="T", text="short text", section=None, published=None,
        sub_question_id="sq1", retrieval_score=0.5, rerank_score=None,
    )
    assert ing.chunk_web_passage(passage) == [passage]


# --- tracing ---------------------------------------------------------------


def test_ingest_is_traced_as_the_expensive_route(tmp_path):
    """arxiv_fulltext is the route D3 must choose to escalate to, so it must
    be distinguishable from arxiv_meta in the trace."""
    events = []

    class FakeTracer:
        def retrieval(self, **kw):
            events.append(kw)

    ing, _ = ingestor(tmp_path, tracer=FakeTracer())
    ing.ingest_paper("https://x/p", source_id="1")
    assert events[0]["route"] == "arxiv_fulltext"
    assert events[0]["n_results"] > 0
    assert events[0]["from_cache"] is False


def test_cached_ingest_is_marked_in_the_trace(tmp_path):
    events = []

    class FakeTracer:
        def retrieval(self, **kw):
            events.append(kw)

    ing, _ = ingestor(tmp_path, tracer=FakeTracer())
    ing.ingest_paper("https://x/p", source_id="1")
    ing.ingest_paper("https://x/p", source_id="1")
    assert events[1]["from_cache"] is True