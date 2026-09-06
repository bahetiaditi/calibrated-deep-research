"""Tests for the arXiv tool.

Uses real `arxiv.Result` objects with a fake client, so normalisation is
tested against the actual 4.x object shape rather than a guess at it. No
network: arXiv asks for 3s between requests and a test suite must not spend
that, nor depend on a flaky third-party API to pass.
"""
from datetime import datetime, timezone

import arxiv
import pytest

from src.config import Config
from src.state import make_passage_id
from src.tools.arxiv_tool import ArxivTool, _short_id, pdf_url_of

CFG = Config({"sources": {"arxiv": {
    "delay_seconds": 3.0, "num_retries": 3, "max_results_default": 8,
}}})


def result(
    entry_id="http://arxiv.org/abs/2312.00752v2",
    title="Mamba: Linear-Time Sequence Modeling",
    summary="Foundation models are\nalmost universally based on Transformers.",
    published=datetime(2023, 12, 1, tzinfo=timezone.utc),
    with_pdf=True,
):
    links = []
    if with_pdf:
        links = [arxiv.Result.Link(
            href="http://arxiv.org/pdf/2312.00752v2", title="pdf", rel="related"
        )]
    return arxiv.Result(
        entry_id=entry_id, title=title, summary=summary,
        published=published, links=links,
    )


class FakeClient:
    def __init__(self, results=None, error=None):
        self._results = results or []
        self._error = error
        self.searches = []

    def results(self, search, offset=0):
        self.searches.append(search)
        if self._error:
            raise self._error
        return iter(self._results)


def tool(**kw):
    return ArxivTool(CFG, **kw)


# --- normalisation ---------------------------------------------------------


def test_search_returns_passages():
    t = tool(client=FakeClient([result()]))
    passages = t.search("mamba", sub_question_id="sq1")
    assert len(passages) == 1
    p = passages[0]
    assert p["source_type"] == "arxiv"
    assert p["source_id"] == "2312.00752v2"
    assert p["source_domain"] == "arxiv.org"
    assert p["sub_question_id"] == "sq1"
    assert p["published"] == "2023-12-01"


def test_abstract_is_labelled_as_a_section():
    """A claim from an abstract is a summary claim, not a measured result.
    The critic needs to be able to tell (§4.2)."""
    p = tool(client=FakeClient([result()])).search("q")[0]
    assert p["section"] == "Abstract"


def test_hard_wrapped_abstract_is_joined():
    p = tool(client=FakeClient([result()])).search("q")[0]
    assert "\n" not in p["text"]
    assert "are almost universally" in p["text"]


def test_passage_id_is_deterministic_and_content_addressed():
    """Re-ingesting the same paper must not duplicate it — the evidence
    store accumulates across runs (§4.1)."""
    a = tool(client=FakeClient([result()])).search("q")[0]
    b = tool(client=FakeClient([result()])).search("different query")[0]
    assert a["id"] == b["id"]
    assert a["id"].startswith("P")


def test_different_text_yields_different_id():
    a = tool(client=FakeClient([result()])).search("q")[0]
    b = tool(client=FakeClient([result(summary="Completely different.")])).search("q")[0]
    assert a["id"] != b["id"]


def test_results_keep_provider_ordering():
    results = [result(entry_id=f"http://arxiv.org/abs/230{i}.0001v1",
                      summary=f"abstract {i}") for i in range(3)]
    passages = tool(client=FakeClient(results)).search("q")
    scores = [p["retrieval_score"] for p in passages]
    assert scores == sorted(scores, reverse=True)


def test_short_id_falls_back_to_entry_id_tail():
    class Bare:
        entry_id = "http://arxiv.org/abs/1706.03762v5"
    assert _short_id(Bare()) == "1706.03762v5"


def test_missing_published_date_is_none():
    p = tool(client=FakeClient([result(published=None)])).search("q")[0]
    assert p["published"] is None


def test_pdf_url_extracted_from_links():
    assert pdf_url_of(result()) == "http://arxiv.org/pdf/2312.00752v2"


def test_pdf_url_none_when_absent():
    assert pdf_url_of(result(with_pdf=False)) is None


# --- graceful degradation --------------------------------------------------


def test_api_error_returns_empty_not_raises():
    """A retrieval failure is information the agent acts on — D2 escalates
    to another route. An exception would crash the run instead."""
    t = tool(client=FakeClient(error=arxiv.UnexpectedEmptyPageError("url", 1, None)))
    assert t.search("anything") == []


def test_arbitrary_exception_also_degrades():
    t = tool(client=FakeClient(error=RuntimeError("connection reset")))
    assert t.search("anything") == []


def test_empty_query_short_circuits_without_calling_api():
    client = FakeClient([result()])
    assert tool(client=client).search("   ") == []
    assert client.searches == []


def test_malformed_entry_is_skipped_not_fatal():
    """One bad entry must not discard the whole page."""
    class Broken:
        @property
        def summary(self):
            raise ValueError("corrupt feed entry")

    passages = tool(client=FakeClient([Broken(), result()])).search("q")
    assert len(passages) == 1


def test_entry_with_no_text_is_dropped():
    passages = tool(client=FakeClient([result(title="", summary="")])).search("q")
    assert passages == []


def test_no_results_returns_empty():
    assert tool(client=FakeClient([])).search("obscure query") == []


# --- search parameters -----------------------------------------------------


def test_max_results_defaults_from_config():
    client = FakeClient([])
    tool(client=client).search("q")
    assert client.searches[0].max_results == 8


def test_max_results_override():
    client = FakeClient([])
    tool(client=client).search("q", max_results=3)
    assert client.searches[0].max_results == 3


@pytest.mark.parametrize("key,expected", [
    ("relevance", arxiv.SortCriterion.Relevance),
    ("submitted", arxiv.SortCriterion.SubmittedDate),
    ("updated", arxiv.SortCriterion.LastUpdatedDate),
    ("nonsense", arxiv.SortCriterion.Relevance),   # unknown falls back safely
])
def test_sort_criteria(key, expected):
    client = FakeClient([])
    tool(client=client).search("q", sort_by=key)
    assert client.searches[0].sort_by == expected


def test_fetch_by_id():
    client = FakeClient([result()])
    passages = tool(client=client).fetch_by_id(["2312.00752"])
    assert len(passages) == 1
    assert client.searches[0].id_list == ["2312.00752"]


def test_fetch_by_id_ignores_blanks():
    client = FakeClient([])
    assert tool(client=client).fetch_by_id(["", "   "]) == []
    assert client.searches == []


# --- etiquette and tracing -------------------------------------------------


def test_client_configured_with_arxiv_delay():
    """arXiv asks for 3s between requests. Lowering this to speed up a
    benchmark risks the project's IP, which costs far more than it saves."""
    t = tool()
    assert t.delay_seconds == 3.0
    assert t.client.delay_seconds == 3.0
    assert t.client.num_retries == 3


def test_search_is_traced():
    events = []

    class FakeTracer:
        def retrieval(self, **kw):
            events.append(kw)

    tool(client=FakeClient([result()]), tracer=FakeTracer()).search(
        "mamba", sub_question_id="sq1"
    )
    assert events[0]["route"] == "arxiv_meta"
    assert events[0]["n_results"] == 1


def test_failure_is_traced_with_the_error():
    """A zero-result round must be distinguishable from a failed one when
    annotating against MAST at C38."""
    events = []

    class FakeTracer:
        def retrieval(self, **kw):
            events.append(kw)

    tool(client=FakeClient(error=RuntimeError("boom")), tracer=FakeTracer()).search("q")
    assert events[0]["n_results"] == 0
    assert "boom" in events[0]["error"]