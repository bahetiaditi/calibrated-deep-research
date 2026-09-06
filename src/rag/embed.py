"""Dense embeddings.

The one silent failure in this layer
------------------------------------
`bge-small-en-v1.5` is an *asymmetric* retrieval model: queries and documents
are encoded differently. Queries must be prefixed with

    "Represent this sentence for searching relevant passages: "

and documents must **not** be. Get this wrong in either direction and nothing
raises — no error, no warning, no crash. Recall just drops, and you attribute
it to the reranker, or the chunking, or the model being weak. It is the most
expensive kind of bug this project could carry, because C13's retrieval
evaluation would faithfully measure a crippled system and report the number
as a finding.

So the prefix is not a parameter callers pass. `embed_query` applies it,
`embed_documents` does not, and the two are separate methods precisely so
that the choice cannot be made by accident at a call site.

Vectors are L2-normalised, which makes cosine similarity equal to a dot
product and lets Qdrant's COSINE distance behave predictably.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

from src.config import Config, get_config

log = logging.getLogger(__name__)


class Embedder:
    """Wraps a sentence-transformers bi-encoder with correct prefix handling."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        model: Any = None,
    ) -> None:
        cfg = config or get_config()
        dense = cfg.section("retrieval.dense")
        self.model_name = str(dense["model"])
        self.dim = int(dense["dim"])
        self.query_prefix = str(dense["query_prefix"])
        self.batch_size = int(dense.get("batch_size", 32))
        self._model = model   # injected in tests; loaded lazily otherwise

        if not self.query_prefix.strip():
            # A blank prefix is almost certainly a config edit gone wrong, and
            # it would silently halve retrieval quality.
            raise ValueError(
                f"retrieval.dense.query_prefix is empty; {self.model_name} "
                f"requires an instruction prefix on queries"
            )

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # -- encoding -----------------------------------------------------------

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode passages. **No prefix** — see the module docstring."""
        if not texts:
            return []
        return self._encode(list(texts))

    def embed_query(self, query: str) -> list[float]:
        """Encode a search query. The instruction prefix is applied here."""
        return self._encode([self.query_prefix + query])[0]

    def embed_queries(self, queries: Sequence[str]) -> list[list[float]]:
        if not queries:
            return []
        return self._encode([self.query_prefix + q for q in queries])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # cosine == dot product
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [list(map(float, v)) for v in vectors]