"""Cross-encoder reranking.

What reranking buys
-------------------
Dense and sparse retrieval both score a query against a passage
*independently* — the query is encoded once, the passage once, and a similarity
is computed between two vectors that never saw each other. A cross-encoder
reads query and passage **together**, so it can judge whether this passage
answers this question rather than whether they occupy similar semantic
territory. It is far more accurate and far too slow to run over a whole
corpus, which is why it runs over the top ~30 fused candidates and keeps 5-10.

Why raw logits are stored, not probabilities
--------------------------------------------
`bge-reranker-base` emits unbounded logits (roughly -10 to +11). A sigmoid
would map them to [0,1], which reads more nicely — and would be the wrong
thing to persist, because sigmoid **saturates**.

This matters concretely. Feature f3 (§5.2) is the score gap between rank 1 and
rank 5: "sharp peak versus flat mush", where flat means the evidence is
ambiguous. In probability space, logits of 9.0 and 7.5 both map to ~0.999 and
the gap vanishes to 0.0005 — the signal is destroyed exactly where evidence is
strongest and the distinction matters most. In logit space that gap is 1.5 and
survives.

So `Passage.rerank_score` holds the raw logit. The sigmoid is derivable from
it; the logit is not recoverable from a saturated sigmoid. Never persist the
lossy form when the lossless one is free.
"""
from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Any, Sequence

from src.config import Config, get_config
from src.rag.fusion import FusedResult

log = logging.getLogger(__name__)


def sigmoid(logit: float) -> float:
    """Logit -> probability, for display and thresholds only.

    Do not store the result: see the module docstring on saturation.
    """
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp = math.exp(logit)          # avoids overflow on large negative logits
    return exp / (1.0 + exp)


class Reranker:
    """Cross-encoder reranker over fused candidates."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        model: Any = None,
        tracer: Any = None,
    ) -> None:
        cfg = config or get_config()
        section = cfg.section("retrieval.rerank")
        self.model_name = str(section["model"])
        self.input_candidates = int(section.get("input_candidates", 30))
        self.output_top_k = int(section.get("output_top_k", 8))
        self.batch_size = int(section.get("batch_size", 16))
        self.tracer = tracer
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("loading reranker %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    # -- reranking ----------------------------------------------------------

    def rerank(
        self,
        query: str,
        results: Sequence[FusedResult],
        *,
        top_k: int | None = None,
        input_candidates: int | None = None,
    ) -> list[FusedResult]:
        """Rerank fused candidates and return the best `top_k`.

        Results carry both their rerank score and their pre-rerank fusion
        rank, so C13 can measure what reranking actually changed rather than
        assuming it helped.
        """
        top_k = top_k or self.output_top_k
        limit = input_candidates or self.input_candidates

        if not results or not query.strip():
            return []

        candidates = list(results)[:limit]
        pairs = [(query, r.passage.get("text") or "") for r in candidates]

        try:
            scores = self.model.predict(pairs, batch_size=self.batch_size)
        except Exception as exc:  # noqa: BLE001
            # Degrade to fusion order rather than failing retrieval. A worse
            # ordering is recoverable; a crashed run is not.
            log.warning("reranking failed, falling back to fusion order: %s", exc)
            if self.tracer is not None:
                self.tracer.note("rerank_failed", error=str(exc))
            return list(results)[:top_k]

        reranked: list[FusedResult] = []
        for candidate, score in zip(candidates, scores):
            logit = float(score)
            passage = dict(candidate.passage)
            passage["rerank_score"] = logit       # raw logit — see docstring
            reranked.append(replace(candidate, passage=passage))  # type: ignore[arg-type]

        order = sorted(
            range(len(reranked)),
            key=lambda i: reranked[i].passage["rerank_score"],
            reverse=True,
        )
        out = [reranked[i] for i in order][:top_k]

        if self.tracer is not None:
            moved = sum(
                1 for new_pos, i in enumerate(order[:top_k]) if new_pos != i
            )
            self.tracer.note(
                "rerank",
                query=query[:200],
                candidates=len(candidates),
                kept=len(out),
                reordered=moved,
                top_logit=round(out[0].passage["rerank_score"], 4) if out else None,
            )
        return out


# ---------------------------------------------------------------------------
# Features derived from rerank scores (f1-f3, consumed at C20)
# ---------------------------------------------------------------------------


def rerank_features(results: Sequence[FusedResult]) -> dict[str, float | None]:
    """Compute f1-f3 from reranked results (§5.2).

    Defined here rather than at C20 because the semantics live with the
    scores: these are *logit-space* quantities, and computing them from
    sigmoids would silently flatten f3.

      f1  max rerank score      — peak evidence quality
      f2  mean of top-3         — depth of support, not one lucky hit
      f3  gap between 1 and 5   — sharp peak vs flat mush; flat = ambiguous

    Returns None for a feature the evidence cannot support (fewer than five
    results for f3). None is honest; zero would read as "no gap", which is a
    strong claim about ambiguity that the data does not license.
    """
    scores = [
        r.passage.get("rerank_score")
        for r in results
        if r.passage.get("rerank_score") is not None
    ]
    if not scores:
        return {"f1_max_rerank": None, "f2_mean_top3": None, "f3_score_gap": None}

    ordered = sorted(scores, reverse=True)
    top3 = ordered[:3]
    return {
        "f1_max_rerank": float(ordered[0]),
        "f2_mean_top3": float(sum(top3) / len(top3)),
        "f3_score_gap": float(ordered[0] - ordered[4]) if len(ordered) >= 5 else None,
    }