"""PDF ingestion — download, extract, chunk, normalise into Passages.

This is the expensive retrieval route (§4.1). The retriever must *choose* to
escalate here via D3, which is what makes depth an earned cost rather than a
default. Reading every paper found would be both slow and, more importantly,
would remove a decision surface the project exists to measure.

Caching is cross-run and idempotent (§4.1): "a paper downloaded during
question 7 is available for free during question 23." That is not only a
speed optimisation — it is what turns the evidence store into a genuine
*store* and makes the `evidence_store` retrieval route in D2 meaningful.

`arxiv.Result.download_pdf()` was removed in arxiv 4.x, so the fetch happens
here. That is the better place for it anyway: the download is content-hash
cached and the arXiv library has no notion of our cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.config import Config, get_config
from src.rag.chunking import chunk_paper, chunk_web
from src.state import Passage, make_passage_id

log = logging.getLogger(__name__)

# Below this, extraction has effectively failed — a scanned PDF with no text
# layer, or a download that returned an error page. Better to report nothing
# than to feed the reranker garbage.
MIN_USABLE_CHARS = 500


def _dedupe(passages: list[Passage]) -> list[Passage]:
    """Drop passages whose id already appeared, keeping the first.

    Ids are content-addressed, so a repeat id means byte-identical text —
    boilerplate, a repeated abstract, an overlap artefact. Returning it twice
    would let ONE piece of evidence count twice toward the sufficiency
    features: f2 (mean of top-3 rerank scores) and f4 (independent source
    count) would both read higher than the evidence warrants. Inflating
    sufficiency on thin evidence is precisely the failure the abstention
    layer exists to prevent, so the dedup belongs here, at ingestion.
    """
    seen: set[str] = set()
    out: list[Passage] = []
    for passage in passages:
        if passage["id"] in seen:
            continue
        seen.add(passage["id"])
        out.append(passage)
    return out


@dataclass
class FetchResult:
    path: Path
    url: str
    content_sha256: str
    bytes: int
    from_cache: bool


class PDFCache:
    """Content-addressed PDF store on disk.

    Filenames key on the URL because that is known before the download; the
    content hash is recorded in a sidecar so a corrupted or changed file is
    detectable. Nothing is ever evicted — a re-downloaded paper costs
    bandwidth and arXiv's patience, both of which are worth more than disk.
    """

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, url: str) -> tuple[Path, Path]:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
        return self.dir / f"{key}.pdf", self.dir / f"{key}.json"

    def get(self, url: str) -> FetchResult | None:
        pdf_path, meta_path = self._paths(url)
        if not (pdf_path.is_file() and meta_path.is_file()):
            return None
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        return FetchResult(
            path=pdf_path, url=url,
            content_sha256=meta.get("content_sha256", ""),
            bytes=pdf_path.stat().st_size, from_cache=True,
        )

    def put(self, url: str, content: bytes) -> FetchResult:
        pdf_path, meta_path = self._paths(url)
        digest = hashlib.sha256(content).hexdigest()

        fd, tmp = tempfile.mkstemp(dir=str(self.dir), suffix=".tmp")
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        os.replace(tmp, pdf_path)   # atomic: no half-written PDFs

        meta_path.write_text(json.dumps({
            "url": url,
            "content_sha256": digest,
            "bytes": len(content),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2))
        return FetchResult(pdf_path, url, digest, len(content), from_cache=False)


def _default_downloader(url: str, timeout: float) -> bytes:
    import urllib.request

    request = urllib.request.Request(
        url,
        headers={
            # arXiv asks that automated clients identify themselves.
            "User-Agent": "calibrated-deep-research/0.1 (research project)"
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


class PDFIngestor:
    """Fetches a PDF, extracts its text, and emits section-tagged passages."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        cache: PDFCache | None = None,
        downloader: Callable[[str, float], bytes] | None = None,
        extractor: Callable[[Path], str] | None = None,
        tracer: Any = None,
        timeout: float = 60.0,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.cfg = config or get_config()
        self.cache = cache or PDFCache(self.cfg.path("sources.arxiv.pdf_cache_dir"))
        self._download = downloader or _default_downloader
        self._extract = extractor or extract_text
        self.tracer = tracer
        self.timeout = timeout
        self._now = now

        paper = self.cfg.section("chunking.paper")
        self.paper_size = int(paper.get("size_chars", 1000))
        self.paper_overlap = int(paper.get("overlap_chars", 150))
        web = self.cfg.section("chunking.web")
        self.web_size = int(web.get("size_chars", 800))
        self.web_overlap = int(web.get("overlap_chars", 150))

    # -- fetch --------------------------------------------------------------

    def fetch(self, url: str) -> FetchResult | None:
        """Download with caching. Returns None on failure — never raises."""
        cached = self.cache.get(url)
        if cached is not None:
            return cached
        try:
            content = self._download(url, self.timeout)
        except Exception as exc:  # noqa: BLE001 — same contract as C6/C7
            log.warning("pdf download failed for %s: %s", url, exc)
            return None
        if not content or not content.startswith(b"%PDF"):
            # An HTML error page saved as .pdf would fail confusingly later.
            log.warning("download from %s is not a PDF (%d bytes)", url, len(content))
            return None
        return self.cache.put(url, content)

    # -- ingest -------------------------------------------------------------

    def ingest_paper(
        self,
        url: str,
        *,
        source_id: str,
        sub_question_id: str = "",
        published: str | None = None,
        title: str = "",
        source_domain: str = "arxiv.org",
    ) -> list[Passage]:
        """Full-text ingestion of one paper into section-tagged passages."""
        started = self._now()
        fetched = self.fetch(url)
        if fetched is None:
            self._trace(url, 0, started, error="download failed")
            return []

        try:
            text = self._extract(fetched.path)
        except Exception as exc:  # noqa: BLE001
            log.warning("pdf extraction failed for %s: %s", url, exc)
            self._trace(url, 0, started, error=f"extraction: {exc}")
            return []

        if len(text) < MIN_USABLE_CHARS:
            # Almost always a scanned PDF with no text layer. OCR is out of
            # scope; reporting nothing beats feeding the reranker noise.
            self._trace(url, 0, started, error=f"only {len(text)} chars extracted")
            return []

        chunks = chunk_paper(
            text, size_chars=self.paper_size, overlap_chars=self.paper_overlap
        )
        passages = _dedupe([
            Passage(
                id=make_passage_id(source_id, chunk.text, chunk.section),
                source_type="arxiv",
                source_id=source_id,
                source_domain=source_domain,
                title=title,
                text=chunk.text,
                section=chunk.section,
                published=published,
                sub_question_id=sub_question_id,
                retrieval_score=0.0,   # set by fusion at C10
                rerank_score=None,
            )
            for chunk in chunks
        ])
        self._trace(url, len(passages), started, from_cache=fetched.from_cache)
        return passages

    def chunk_web_passage(self, passage: Passage) -> list[Passage]:
        """Split an over-long web passage, preserving its provenance.

        Tavily returns whole pages; a 20k-character page is one bad retrieval
        unit but many good ones.
        """
        if len(passage["text"]) <= self.web_size:
            return [passage]
        return _dedupe([
            {
                **passage,
                "id": make_passage_id(passage["source_id"], chunk.text, None),
                "text": chunk.text,
            }
            for chunk in chunk_web(
                passage["text"],
                size_chars=self.web_size,
                overlap_chars=self.web_overlap,
            )
        ])

    # -- tracing ------------------------------------------------------------

    def _trace(
        self,
        url: str,
        n_results: int,
        started: float,
        error: str | None = None,
        from_cache: bool = False,
    ) -> None:
        if self.tracer is None:
            return
        self.tracer.retrieval(
            route="arxiv_fulltext",
            query=url,
            n_results=n_results,
            latency_s=self._now() - started,
            from_cache=from_cache,
            error=error,
        )


def extract_text(path: Path) -> str:
    """Extract text from a PDF with pypdf, page by page.

    A single unparseable page is skipped rather than failing the document:
    papers routinely have one figure-heavy page that breaks extraction, and
    losing that page is much better than losing the paper.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for number, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            log.debug("skipping unparseable page %d of %s: %s", number, path.name, exc)
    return "\n".join(pages)