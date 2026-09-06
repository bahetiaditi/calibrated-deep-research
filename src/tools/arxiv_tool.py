"""arXiv metadata search.

This is the cheap, broad retrieval route (§4.1): title, abstract, authors,
date, categories. Full-text reading is a separate, expensive route the
retriever must *choose* to escalate to (D3, C8) — keeping them apart is what
makes depth an earned cost rather than a default.

Two design commitments
----------------------
**Failure is a result, not an exception.** The arXiv API is genuinely flaky:
timeouts, empty pages, malformed feeds. A retrieval failure must reach the
retriever as "this route returned nothing", because that is information the
agent acts on — D2 escalates to another route, D3 decides the sub-question is
unresolvable. An exception propagating into the graph turns a normal,
informative outcome into a crashed run. So `search()` returns `[]` and
records the failure in the trace.

**Etiquette is non-negotiable.** arXiv asks for 3 seconds between requests
and the `Client` enforces it internally. Do not lower `delay_seconds` to make
a benchmark run faster; getting the project's IP blocked mid-eval would cost
far more than the time saved.

API note: `arxiv` is at 4.x here, not the 3.x the original plan assumed.
`Result.download_pdf()` no longer exists — `pdf_url` is still present and
C8 downloads it directly, which is what content-hash caching wanted anyway.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Iterable

from src.config import Config, get_config
from src.state import Passage, make_passage_id

log = logging.getLogger(__name__)

ARXIV_DOMAIN = "arxiv.org"

SORT_CRITERIA = {
    "relevance": "Relevance",
    "submitted": "SubmittedDate",
    "updated": "LastUpdatedDate",
}


class ArxivTool:
    """Searches arXiv and normalises results into `Passage` objects."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        client: Any = None,
        tracer: Any = None,
        now: Any = time.time,
    ) -> None:
        cfg = config or get_config()
        section = cfg.section("sources.arxiv")
        self.delay_seconds = float(section.get("delay_seconds", 3.0))
        self.num_retries = int(section.get("num_retries", 3))
        self.max_results_default = int(section.get("max_results_default", 8))
        self.tracer = tracer
        self._now = now
        self._client = client  # injected in tests; built lazily otherwise

    # -- client -------------------------------------------------------------

    @property
    def client(self) -> Any:
        if self._client is None:
            import arxiv

            self._client = arxiv.Client(
                page_size=100,
                delay_seconds=self.delay_seconds,
                num_retries=self.num_retries,
            )
        return self._client

    # -- search -------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        sub_question_id: str = "",
        sort_by: str = "relevance",
    ) -> list[Passage]:
        """Search arXiv metadata. Returns `[]` on any failure — never raises.

        An empty list is a legitimate, informative outcome: it is what tells
        D2 to try another route and D3 that this line of inquiry is not
        working.
        """
        import arxiv

        max_results = max_results or self.max_results_default
        query = (query or "").strip()
        started = self._now()

        if not query:
            self._trace(query, 0, started, error="empty query")
            return []

        try:
            criterion = getattr(
                arxiv.SortCriterion, SORT_CRITERIA.get(sort_by, "Relevance")
            )
            search = arxiv.Search(
                query=query, max_results=max_results, sort_by=criterion
            )
            results = list(self.client.results(search))
        except Exception as exc:  # noqa: BLE001 — see module docstring
            log.warning("arxiv search failed for %r: %s", query[:80], exc)
            self._trace(query, 0, started, error=f"{type(exc).__name__}: {exc}")
            return []

        passages = self._to_passages(results, sub_question_id)
        self._trace(query, len(passages), started)
        return passages

    def fetch_by_id(
        self, arxiv_ids: Iterable[str], *, sub_question_id: str = ""
    ) -> list[Passage]:
        """Fetch specific papers by id — used when a citation names one."""
        import arxiv

        ids = [i.strip() for i in arxiv_ids if i and i.strip()]
        started = self._now()
        if not ids:
            return []
        try:
            results = list(self.client.results(arxiv.Search(id_list=ids)))
        except Exception as exc:  # noqa: BLE001
            log.warning("arxiv id fetch failed for %s: %s", ids, exc)
            self._trace(",".join(ids), 0, started, error=str(exc))
            return []
        passages = self._to_passages(results, sub_question_id)
        self._trace(",".join(ids), len(passages), started)
        return passages

    # -- normalisation ------------------------------------------------------

    def _to_passages(self, results: list[Any], sub_question_id: str) -> list[Passage]:
        passages: list[Passage] = []
        for rank, result in enumerate(results):
            passage = self._to_passage(result, sub_question_id, rank)
            if passage is not None:
                passages.append(passage)
        return passages

    def _to_passage(
        self, result: Any, sub_question_id: str, rank: int
    ) -> Passage | None:
        try:
            summary = _clean(getattr(result, "summary", "") or "")
            title = _clean(getattr(result, "title", "") or "")
            if not summary and not title:
                return None  # nothing citable

            source_id = _short_id(result)
            published = getattr(result, "published", None)
            published_iso = (
                published.date().isoformat()
                if published is not None and hasattr(published, "date")
                else None
            )

            return Passage(
                id=make_passage_id(source_id, summary, "Abstract"),
                source_type="arxiv",
                source_id=source_id,
                source_domain=ARXIV_DOMAIN,
                title=title,
                text=summary,
                # Abstracts are labelled as such so the critic can tell an
                # abstract's summary claim from a Results-section measurement
                # (§4.2). Full-text sections arrive via the reader at C8.
                section="Abstract",
                published=published_iso,
                sub_question_id=sub_question_id,
                # Rank-derived placeholder. Real scores come from fusion at
                # C10; this only preserves the provider's ordering until then.
                retrieval_score=1.0 / (rank + 1),
                rerank_score=None,
            )
        except Exception as exc:  # noqa: BLE001
            # One malformed entry must not discard the whole page of results.
            log.warning("skipping malformed arxiv result: %s", exc)
            return None

    # -- tracing ------------------------------------------------------------

    def _trace(
        self, query: str, n_results: int, started: float, error: str | None = None
    ) -> None:
        if self.tracer is None:
            return
        self.tracer.retrieval(
            route="arxiv_meta",
            query=query,
            n_results=n_results,
            latency_s=self._now() - started,
            error=error,
        )


def _clean(text: str) -> str:
    """arXiv abstracts arrive hard-wrapped; joining lines aids chunking."""
    return " ".join(text.split())


def _short_id(result: Any) -> str:
    """Stable arXiv identifier, e.g. '2312.00752v2'."""
    getter = getattr(result, "get_short_id", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:  # noqa: BLE001
            pass
    entry_id = str(getattr(result, "entry_id", "") or "")
    return entry_id.rsplit("/", 1)[-1] if entry_id else "unknown"


def pdf_url_of(result: Any) -> str | None:
    """PDF URL for a result, or None.

    `Result.download_pdf()` was removed in arxiv 4.x; `pdf_url` remains and
    C8 fetches it directly so the download can be content-hash cached.
    """
    return getattr(result, "pdf_url", None)