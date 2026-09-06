"""Qdrant evidence store.

Why Qdrant rather than Chroma (§4.3): payload filtering is first-class, and
filtered retrieval is not a nice-to-have here. The retriever restricts to
Results sections when a sub-question asks for a number, excludes pre-2023
sources when it asks about recent work, and prefers papers over web pages on
technical sub-questions (§4.4). That is also the live instance of Project 1's
pre-filter / post-filter / predicate-aware problem, which is the honest
bridge between the two portfolio pieces.

Local mode (`QdrantClient(path=...)`) — no server, no Docker, one directory
on disk that persists across runs. That persistence is what makes the
`evidence_store` route in D2 real: a paper ingested for question 7 is
searchable during question 23 without re-downloading anything.

Ids
---
Qdrant point ids must be unsigned integers or UUIDs; our passage ids are
strings like `Pa3f9c2b1d004`. Each is mapped through `uuid5`, which is
deterministic — so re-ingesting a passage *overwrites* rather than
duplicating, preserving the idempotency established at C8. The original
string id is kept in the payload and is what the rest of the system cites.

API note: qdrant-client 1.19 removed `search()`. `query_points()` is the
current entry point and returns a response object with `.points`.
"""
from __future__ import annotations

import logging
import uuid
import warnings
from typing import Any, Iterable, Sequence

from src.config import Config, get_config
from src.state import Passage

log = logging.getLogger(__name__)

# Fixed namespace so passage-id -> point-id is stable across machines and runs.
_NAMESPACE = uuid.UUID("6f1d6b4e-6a1e-4f5a-9f2a-2d6f2a5a7c31")

# Payload fields that get an index. Only these are worth indexing: an index
# on a field nothing filters by costs build time and memory for nothing.
INDEXED_FIELDS = {
    "section": "keyword",
    "source_type": "keyword",
    "source_domain": "keyword",
    "source_id": "keyword",
    "sub_question_id": "keyword",
    "published": "keyword",
}


def point_id(passage_id: str) -> str:
    """Deterministic UUID for a passage id."""
    return str(uuid.uuid5(_NAMESPACE, passage_id))


class EvidenceStore:
    """Vector store over `Passage` objects with payload filtering."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        client: Any = None,
        embedder: Any = None,
        path: str | None = None,
        collection: str | None = None,
    ) -> None:
        self.cfg = config or get_config()
        store = self.cfg.section("retrieval.store")
        self.collection = collection or str(store.get("collection", "evidence"))
        self.dim = int(self.cfg.get("retrieval.dense.dim"))
        self._path = path if path is not None else str(self.cfg.path("retrieval.store.path"))
        self._client = client
        self._embedder = embedder
        self._ready = False

    # -- lazy resources -----------------------------------------------------

    @property
    def client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(path=self._path)
        return self._client

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from src.rag.embed import Embedder

            self._embedder = Embedder(self.cfg)
        return self._embedder

    def ensure_collection(self) -> None:
        """Create the collection and payload indexes if absent. Idempotent."""
        if self._ready:
            return
        from qdrant_client import models

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dim, distance=models.Distance.COSINE
                ),
            )
            # Local Qdrant ignores payload indexes and says so on every call.
            # Filtering itself works correctly without them (the store tests
            # verify this against a real local client) — an index only makes
            # it faster. We still declare them so that moving to a server
            # deployment needs no code change, and we silence the notice
            # rather than printing it on every ingestion.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for field, schema in INDEXED_FIELDS.items():
                    try:
                        self.client.create_payload_index(
                            collection_name=self.collection,
                            field_name=field,
                            field_schema=schema,
                        )
                    except Exception as exc:  # noqa: BLE001
                        # Never fail ingestion over an optimisation.
                        log.debug("payload index on %s unavailable: %s", field, exc)
        self._ready = True

    # -- writing ------------------------------------------------------------

    def upsert(self, passages: Sequence[Passage], *, batch_size: int = 64) -> int:
        """Store passages. Re-storing the same passage overwrites it."""
        from qdrant_client import models

        passages = [p for p in passages if p.get("text")]
        if not passages:
            return 0
        self.ensure_collection()

        written = 0
        for start in range(0, len(passages), batch_size):
            batch = passages[start : start + batch_size]
            vectors = self.embedder.embed_documents([p["text"] for p in batch])
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=point_id(p["id"]),
                        vector=vector,
                        payload=dict(p),
                    )
                    for p, vector in zip(batch, vectors)
                ],
            )
            written += len(batch)
        return written

    # -- reading ------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        section: str | Sequence[str] | None = None,
        source_type: str | None = None,
        source_domain: str | Sequence[str] | None = None,
        published_after: str | None = None,
        exclude_sections: Sequence[str] | None = None,
        score_threshold: float | None = None,
    ) -> list[Passage]:
        """Similarity search with optional payload filters.

        Filtering happens *during* traversal, not after — Qdrant applies the
        payload condition inside the HNSW search rather than post-filtering
        the result set. That is the predicate-aware regime from Project 1, and
        it is why low-selectivity filters (a single section, a narrow date
        range) do not collapse recall here.
        """
        self.ensure_collection()
        vector = self.embedder.embed_query(query)
        query_filter = self.build_filter(
            section=section,
            source_type=source_type,
            source_domain=source_domain,
            published_after=published_after,
            exclude_sections=exclude_sections,
        )
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            score_threshold=score_threshold,
        )
        return [_to_passage(p) for p in response.points]

    def get(self, passage_id: str) -> Passage | None:
        return next(iter(self.get_many([passage_id])), None)

    def get_many(self, passage_ids: Iterable[str]) -> list[Passage]:
        """Exact lookup by passage id — what the critic uses to fetch the
        passage a claim cites."""
        ids = [point_id(pid) for pid in passage_ids]
        if not ids:
            return []
        self.ensure_collection()
        records = self.client.retrieve(
            collection_name=self.collection, ids=ids, with_payload=True
        )
        return [_to_passage(r) for r in records]

    def count(self) -> int:
        self.ensure_collection()
        return int(self.client.count(collection_name=self.collection).count)

    # -- filters ------------------------------------------------------------

    @staticmethod
    def build_filter(
        *,
        section: str | Sequence[str] | None = None,
        source_type: str | None = None,
        source_domain: str | Sequence[str] | None = None,
        published_after: str | None = None,
        exclude_sections: Sequence[str] | None = None,
    ) -> Any | None:
        """Compose a Qdrant filter, or None when nothing is constrained."""
        from qdrant_client import models

        must: list[Any] = []
        must_not: list[Any] = []

        def match(field: str, value: str | Sequence[str]) -> Any:
            if isinstance(value, str):
                return models.FieldCondition(
                    key=field, match=models.MatchValue(value=value)
                )
            return models.FieldCondition(
                key=field, match=models.MatchAny(any=list(value))
            )

        if section:
            must.append(match("section", section))
        if source_type:
            must.append(match("source_type", source_type))
        if source_domain:
            must.append(match("source_domain", source_domain))
        if published_after:
            # ISO dates compare correctly as strings, so a lexical range is
            # sufficient and avoids a datetime index.
            must.append(
                models.FieldCondition(
                    key="published", range=models.DatetimeRange(gte=published_after)
                )
            )
        if exclude_sections:
            must_not.append(match("section", list(exclude_sections)))

        if not must and not must_not:
            return None
        return models.Filter(must=must or None, must_not=must_not or None)


def _to_passage(record: Any) -> Passage:
    payload = dict(getattr(record, "payload", None) or {})
    score = getattr(record, "score", None)
    if score is not None:
        payload["retrieval_score"] = float(score)
    return payload  # type: ignore[return-value]