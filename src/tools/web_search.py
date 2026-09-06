"""Web search behind a provider interface.

Two providers, different roles
------------------------------
**Tavily** returns pre-extracted page content rather than snippets, which is
what makes a passage citable — a critic cannot verify a claim against forty
words of search-result teaser. It costs credits: 1000/month free, one per
basic search. Reserved for eval runs.

**DuckDuckGo** (`ddgs`) is free and unlimited but returns snippets only, and
rate-limits aggressively at roughly 30 requests/minute from one IP. Fine for
development and debugging, wrong for producing evidence the critic will judge.

The `SearchProvider` interface exists so the retriever never knows which one
answered, and so C13's retrieval evaluation can hold the provider fixed while
varying the retrieval strategy.

Credit accounting
-----------------
Tavily's monthly budget is a real constraint: the full eval needs roughly 250
credits and careless development could burn 1000 in an afternoon. Rather than
inventing a second accounting mechanism, Tavily is registered in a
`QuotaLedger` (the same one that polices LLM quota) with a daily cap derived
from the monthly allowance. Overrunning then costs a graceful fallback to
DuckDuckGo instead of a dead API key two weeks before the eval.

Failure contract
----------------
Same as the arXiv tool: `search()` returns `[]` rather than raising, because
"this route found nothing" is information D2 and D3 act on. A rate-limited or
broken provider falls through to the next one in the chain; only exhausting
the whole chain yields an empty result, and the trace records why.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse

from src.config import Config, get_config, has_env, require_env
from src.llm.rate_limit import ModelLimits, QuotaLedger
from src.state import Passage, make_passage_id

log = logging.getLogger(__name__)

# Minimum characters for a result to be worth storing. Below this it is a
# fragment, not evidence, and it will only dilute the reranker's candidates.
MIN_CONTENT_CHARS = 80


class SearchProvider(ABC):
    """Contract every web source implements."""

    name: str = "abstract"
    returns_full_content: bool = False

    @abstractmethod
    def search(
        self, query: str, *, max_results: int, sub_question_id: str = ""
    ) -> list[Passage]:
        """Return passages, or `[]`. Must not raise."""

    def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------------


class TavilySearchProvider(SearchProvider):
    name = "tavily"
    returns_full_content = True

    def __init__(
        self,
        config: Config | None = None,
        *,
        client: Any = None,
        ledger: QuotaLedger | None = None,
        now: Any = time.time,
    ) -> None:
        self.cfg = config or get_config()
        self.search_depth = str(
            self.cfg.get("sources.web.tavily_search_depth", "basic")
        )
        self._client = client
        self._now = now
        self.ledger = ledger

    def is_available(self) -> bool:
        return self._client is not None or has_env("TAVILY_API_KEY")

    @property
    def client(self) -> Any:
        if self._client is None:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=require_env("TAVILY_API_KEY"))
        return self._client

    def search(
        self, query: str, *, max_results: int, sub_question_id: str = ""
    ) -> list[Passage]:
        if self.ledger is not None:
            # One credit per basic search. Tokens are meaningless here, so
            # the ledger is used purely for its request counting.
            decision = self.ledger.check(self.name, 0, now=self._now())
            if not decision.allowed:
                log.warning("tavily credit budget reached: %s", decision.detail)
                raise SearchQuotaExceeded(decision.detail)

        response = self.client.search(
            query=query,
            max_results=max_results,
            search_depth=self.search_depth,
        )

        if self.ledger is not None:
            self.ledger.record(self.name, 0, now=self._now())

        return _to_passages(
            [
                {
                    "title": r.get("title", ""),
                    # raw_content when present is the full page; content is
                    # Tavily's extraction. Either beats a snippet.
                    "text": r.get("raw_content") or r.get("content", ""),
                    "url": r.get("url", ""),
                    "published": r.get("published_date"),
                    "score": r.get("score"),
                }
                for r in (response or {}).get("results", [])
            ],
            sub_question_id,
        )


# ---------------------------------------------------------------------------
# DuckDuckGo
# ---------------------------------------------------------------------------


class DuckDuckGoSearchProvider(SearchProvider):
    name = "ddgs"
    returns_full_content = False   # snippets only — dev use, not eval

    def __init__(
        self,
        config: Config | None = None,
        *,
        client: Any = None,
        region: str = "us-en",
    ) -> None:
        self.cfg = config or get_config()
        self.region = region
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from ddgs import DDGS

            self._client = DDGS()
        return self._client

    def search(
        self, query: str, *, max_results: int, sub_question_id: str = ""
    ) -> list[Passage]:
        results = self.client.text(
            query, region=self.region, safesearch="moderate", max_results=max_results
        )
        return _to_passages(
            [
                {
                    "title": r.get("title", ""),
                    "text": r.get("body", ""),
                    "url": r.get("href", "") or r.get("url", ""),
                    "published": None,
                    "score": None,
                }
                for r in (results or [])
            ],
            sub_question_id,
        )


class SearchQuotaExceeded(RuntimeError):
    """Provider budget spent — fall through to the next provider."""


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _domain(url: str) -> str:
    """Registrable-ish host, used to count independent sources (feature f4).

    Two results from the same site are one source, not two, and treating
    them as two would inflate the sufficiency score exactly when the
    evidence is weakest.
    """
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return "unknown"
    return host[4:] if host.startswith("www.") else (host or "unknown")


def _to_passages(raw: list[dict[str, Any]], sub_question_id: str) -> list[Passage]:
    passages: list[Passage] = []
    for rank, item in enumerate(raw):
        text = " ".join((item.get("text") or "").split())
        url = item.get("url") or ""
        if len(text) < MIN_CONTENT_CHARS or not url:
            continue  # a fragment is not evidence
        passages.append(
            Passage(
                id=make_passage_id(url, text, None),
                source_type="web",
                source_id=url,
                source_domain=_domain(url),
                title=" ".join((item.get("title") or "").split()),
                text=text,
                section=None,          # web pages have no section structure
                published=item.get("published"),
                sub_question_id=sub_question_id,
                retrieval_score=(
                    float(item["score"]) if item.get("score") is not None
                    else 1.0 / (rank + 1)
                ),
                rerank_score=None,
            )
        )
    return passages


# ---------------------------------------------------------------------------
# Chained tool
# ---------------------------------------------------------------------------


class WebSearchTool:
    """Ordered provider chain with graceful fallback."""

    def __init__(
        self,
        providers: list[SearchProvider],
        *,
        config: Config | None = None,
        tracer: Any = None,
        now: Any = time.time,
    ) -> None:
        if not providers:
            raise ValueError("WebSearchTool needs at least one provider")
        self.providers = providers
        self.cfg = config or get_config()
        self.max_results_default = int(
            self.cfg.get("sources.web.max_results_default", 6)
        )
        self.tracer = tracer
        self._now = now

    def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        sub_question_id: str = "",
    ) -> list[Passage]:
        max_results = max_results or self.max_results_default
        query = (query or "").strip()
        started = self._now()

        if not query:
            self._trace("none", query, 0, started, error="empty query")
            return []

        errors: list[str] = []
        for provider in self.providers:
            if not provider.is_available():
                errors.append(f"{provider.name}: unavailable (no credentials)")
                continue
            try:
                passages = provider.search(
                    query, max_results=max_results, sub_question_id=sub_question_id
                )
            except Exception as exc:  # noqa: BLE001 — see module docstring
                # Rate limits, credit exhaustion, transport errors: all mean
                # "try the next provider", none mean "crash the run".
                errors.append(f"{provider.name}: {type(exc).__name__}: {exc}")
                log.warning("%s search failed for %r: %s", provider.name, query[:60], exc)
                continue

            if passages:
                self._trace(provider.name, query, len(passages), started,
                            fell_back=provider is not self.providers[0])
                return passages
            errors.append(f"{provider.name}: no results")

        self._trace("none", query, 0, started, error="; ".join(errors))
        return []

    def _trace(
        self,
        provider: str,
        query: str,
        n_results: int,
        started: float,
        error: str | None = None,
        fell_back: bool = False,
    ) -> None:
        if self.tracer is None:
            return
        self.tracer.retrieval(
            route="web",
            query=query,
            n_results=n_results,
            latency_s=self._now() - started,
            provider=provider,
            fell_back=fell_back,
            error=error,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_search_ledger(config: Config | None = None) -> QuotaLedger:
    """Ledger guarding Tavily's monthly credit allowance.

    The monthly figure is converted to a daily cap so a careless development
    session cannot spend the eval's budget. Overrun degrades to DuckDuckGo
    rather than to a dead key.
    """
    cfg = config or get_config()
    monthly = int(cfg.get("sources.web.tavily_credits_per_month", 1000))
    daily = max(1, int(cfg.get("sources.web.tavily_credits_per_day", monthly // 30)))
    return QuotaLedger(
        cfg.path("sources.web.credit_ledger_path"),
        {"tavily": ModelLimits(rpd=daily, reset_timezone="UTC")},
    )


def get_search_provider(
    config: Config | None = None,
    *,
    tracer: Any = None,
    ledger: QuotaLedger | None = None,
) -> WebSearchTool:
    """Build the provider chain named in config, skipping unavailable ones.

    `sources.web.provider` names the preferred provider; the remainder follow
    as fallbacks. Configuring Tavily first without a key is not an error — it
    is simply skipped, so a contributor without a Tavily account can still
    run the system on DuckDuckGo.
    """
    cfg = config or get_config()
    preferred = str(cfg.get("sources.web.provider", "tavily"))
    ledger = ledger if ledger is not None else make_search_ledger(cfg)

    built: dict[str, SearchProvider] = {
        "tavily": TavilySearchProvider(cfg, ledger=ledger),
        "ddgs": DuckDuckGoSearchProvider(cfg),
    }
    order = [preferred] + [n for n in ("tavily", "ddgs") if n != preferred]
    chain = [built[n] for n in order if n in built]
    return WebSearchTool(chain, config=cfg, tracer=tracer)