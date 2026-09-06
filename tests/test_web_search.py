"""Tests for web search.

No network. Both SDKs are faked at the client boundary so the normalisation,
fallback and credit-accounting logic is exercised against the real response
shapes each library returns.
"""
import pytest

from src.config import Config
from src.llm.rate_limit import ModelLimits, QuotaLedger
from src.tools.web_search import (
    MIN_CONTENT_CHARS,
    DuckDuckGoSearchProvider,
    SearchProvider,
    SearchQuotaExceeded,
    TavilySearchProvider,
    WebSearchTool,
    _domain,
)

T0 = 1_757_000_000.0

CFG = Config({"sources": {"web": {
    "provider": "tavily", "max_results_default": 6,
    "tavily_search_depth": "basic",
}}})

LONG = "Reciprocal rank fusion combines ranked lists without score normalisation. " * 3


class FakeTavily:
    """Mirrors tavily-python's response shape."""

    def __init__(self, results=None, error=None):
        self.results = results if results is not None else [{
            "title": "RRF explained", "content": LONG,
            "url": "https://example.com/rrf", "score": 0.91,
            "published_date": "2024-03-01",
        }]
        self.error = error
        self.calls = []

    def search(self, query, max_results=None, search_depth=None, **kw):
        self.calls.append({"query": query, "max_results": max_results,
                           "search_depth": search_depth})
        if self.error:
            raise self.error
        return {"results": self.results}


class FakeDDGS:
    """Mirrors ddgs's list-of-dicts shape."""

    def __init__(self, results=None, error=None):
        self.results = results if results is not None else [{
            "title": "RRF", "body": LONG, "href": "https://www.other.org/a",
        }]
        self.error = error
        self.calls = []

    def text(self, query, **kw):
        self.calls.append({"query": query, **kw})
        if self.error:
            raise self.error
        return self.results


def ledger(tmp_path, daily=33):
    return QuotaLedger(tmp_path / "credits.json",
                       {"tavily": ModelLimits(rpd=daily, reset_timezone="UTC")})


# --- normalisation ---------------------------------------------------------


def test_tavily_returns_passages():
    p = TavilySearchProvider(CFG, client=FakeTavily()).search(
        "rrf", max_results=3, sub_question_id="sq1")[0]
    assert p["source_type"] == "web"
    assert p["source_id"] == "https://example.com/rrf"
    assert p["source_domain"] == "example.com"
    assert p["sub_question_id"] == "sq1"
    assert p["published"] == "2024-03-01"
    assert p["retrieval_score"] == 0.91


def test_ddgs_returns_passages():
    p = DuckDuckGoSearchProvider(CFG, client=FakeDDGS()).search(
        "rrf", max_results=3)[0]
    assert p["source_domain"] == "other.org"      # www. stripped
    assert p["section"] is None                   # web pages have no sections


def test_raw_content_preferred_over_snippet():
    """Tavily's raw_content is the full page; content is its extraction.
    A critic cannot verify a claim against a teaser."""
    fake = FakeTavily(results=[{
        "title": "t", "content": "short extraction " * 6,
        "raw_content": "FULL PAGE " + LONG,
        "url": "https://example.com/x",
    }])
    p = TavilySearchProvider(CFG, client=fake).search("q", max_results=1)[0]
    assert p["text"].startswith("FULL PAGE")


def test_short_results_are_dropped():
    fake = FakeTavily(results=[
        {"title": "frag", "content": "too short", "url": "https://a.com/1"},
        {"title": "ok", "content": LONG, "url": "https://a.com/2"},
    ])
    passages = TavilySearchProvider(CFG, client=fake).search("q", max_results=5)
    assert len(passages) == 1
    assert len(passages[0]["text"]) >= MIN_CONTENT_CHARS


def test_result_without_url_is_dropped():
    fake = FakeTavily(results=[{"title": "t", "content": LONG, "url": ""}])
    assert TavilySearchProvider(CFG, client=fake).search("q", max_results=1) == []


def test_ids_are_content_addressed_and_stable():
    a = TavilySearchProvider(CFG, client=FakeTavily()).search("q1", max_results=1)[0]
    b = TavilySearchProvider(CFG, client=FakeTavily()).search("q2", max_results=1)[0]
    assert a["id"] == b["id"]


@pytest.mark.parametrize("url,expected", [
    ("https://www.arxiv.org/abs/1", "arxiv.org"),
    ("http://blog.example.co.uk/p", "blog.example.co.uk"),
    ("not a url", "unknown"),
])
def test_domain_extraction(url, expected):
    assert _domain(url) == expected


def test_same_domain_results_share_a_domain_label():
    """Feature f4 counts independent sources. Two pages from one site are
    one source; counting them as two would inflate sufficiency exactly when
    the evidence is weakest."""
    fake = FakeTavily(results=[
        {"title": "a", "content": LONG, "url": "https://site.com/1"},
        {"title": "b", "content": LONG, "url": "https://www.site.com/2"},
    ])
    passages = TavilySearchProvider(CFG, client=fake).search("q", max_results=5)
    assert {p["source_domain"] for p in passages} == {"site.com"}


# --- fallback chain --------------------------------------------------------


def test_falls_back_when_primary_rate_limits():
    """The C7 acceptance check."""
    from ddgs.exceptions import RatelimitException

    ddg = DuckDuckGoSearchProvider(CFG, client=FakeDDGS(error=RatelimitException("429")))
    tav = TavilySearchProvider(CFG, client=FakeTavily())
    passages = WebSearchTool([ddg, tav], config=CFG).search("q")
    assert len(passages) == 1
    assert passages[0]["source_domain"] == "example.com"


def test_falls_back_when_primary_returns_nothing():
    tav = TavilySearchProvider(CFG, client=FakeTavily(results=[]))
    ddg = DuckDuckGoSearchProvider(CFG, client=FakeDDGS())
    passages = WebSearchTool([tav, ddg], config=CFG).search("q")
    assert passages[0]["source_domain"] == "other.org"


def test_whole_chain_failing_returns_empty_not_raises():
    """'Nothing found' is information D2 and D3 act on. An exception would
    crash the run instead."""
    tav = TavilySearchProvider(CFG, client=FakeTavily(error=RuntimeError("down")))
    ddg = DuckDuckGoSearchProvider(CFG, client=FakeDDGS(error=RuntimeError("down")))
    assert WebSearchTool([tav, ddg], config=CFG).search("q") == []


def test_unavailable_provider_is_skipped_not_an_error(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    tav = TavilySearchProvider(CFG)                 # no client, no key
    ddg = DuckDuckGoSearchProvider(CFG, client=FakeDDGS())
    assert WebSearchTool([tav, ddg], config=CFG).search("q")


def test_empty_query_short_circuits():
    fake = FakeTavily()
    tool = WebSearchTool([TavilySearchProvider(CFG, client=fake)], config=CFG)
    assert tool.search("  ") == []
    assert fake.calls == []


def test_requires_at_least_one_provider():
    with pytest.raises(ValueError):
        WebSearchTool([], config=CFG)


# --- credit accounting -----------------------------------------------------


def test_tavily_stops_at_the_daily_credit_cap(tmp_path):
    led = ledger(tmp_path, daily=2)
    provider = TavilySearchProvider(CFG, client=FakeTavily(), ledger=led,
                                    now=lambda: T0)
    provider.search("q1", max_results=1)
    provider.search("q2", max_results=1)
    with pytest.raises(SearchQuotaExceeded):
        provider.search("q3", max_results=1)


def test_credit_exhaustion_falls_back_to_ddgs(tmp_path):
    """Overrun must degrade to free search, not to a dead key two weeks
    before the eval run."""
    led = ledger(tmp_path, daily=1)
    tav = TavilySearchProvider(CFG, client=FakeTavily(), ledger=led, now=lambda: T0)
    ddg = DuckDuckGoSearchProvider(CFG, client=FakeDDGS())
    tool = WebSearchTool([tav, ddg], config=CFG)

    assert tool.search("q1")[0]["source_domain"] == "example.com"
    assert tool.search("q2")[0]["source_domain"] == "other.org"


def test_credits_persist_across_instances(tmp_path):
    led = ledger(tmp_path, daily=1)
    TavilySearchProvider(CFG, client=FakeTavily(), ledger=led,
                         now=lambda: T0).search("q", max_results=1)

    fresh = ledger(tmp_path, daily=1)
    with pytest.raises(SearchQuotaExceeded):
        TavilySearchProvider(CFG, client=FakeTavily(), ledger=fresh,
                             now=lambda: T0).search("q2", max_results=1)


# --- search parameters and tracing -----------------------------------------


def test_search_depth_comes_from_config():
    """'advanced' costs two credits. Defaulting to it would halve the
    monthly allowance silently."""
    fake = FakeTavily()
    TavilySearchProvider(CFG, client=fake).search("q", max_results=3)
    assert fake.calls[0]["search_depth"] == "basic"


def test_max_results_defaults_from_config():
    fake = FakeTavily()
    WebSearchTool([TavilySearchProvider(CFG, client=fake)], config=CFG).search("q")
    assert fake.calls[0]["max_results"] == 6


def test_trace_records_provider_and_fallback():
    events = []

    class FakeTracer:
        def retrieval(self, **kw):
            events.append(kw)

    tav = TavilySearchProvider(CFG, client=FakeTavily(error=RuntimeError("x")))
    ddg = DuckDuckGoSearchProvider(CFG, client=FakeDDGS())
    WebSearchTool([tav, ddg], config=CFG, tracer=FakeTracer()).search("q")
    assert events[0]["provider"] == "ddgs"
    assert events[0]["fell_back"] is True


def test_trace_records_why_the_chain_failed():
    """C38 must be able to tell a zero-result round from a broken one."""
    events = []

    class FakeTracer:
        def retrieval(self, **kw):
            events.append(kw)

    tav = TavilySearchProvider(CFG, client=FakeTavily(error=RuntimeError("boom")))
    WebSearchTool([tav], config=CFG, tracer=FakeTracer()).search("q")
    assert "boom" in events[0]["error"]


def test_provider_interface_is_implemented_by_both():
    assert issubclass(TavilySearchProvider, SearchProvider)
    assert issubclass(DuckDuckGoSearchProvider, SearchProvider)
    assert TavilySearchProvider.returns_full_content is True
    assert DuckDuckGoSearchProvider.returns_full_content is False