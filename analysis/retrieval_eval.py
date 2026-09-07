"""Retrieval evaluation — the R1-R4 study (§4.5).

Why this is a commit and not an afterthought
--------------------------------------------
Without it, retrieval quality is an untested assumption sitting underneath
every agent result. When the full system gets a question wrong at C37, the
question "was that a retrieval failure or a reasoning failure?" has no answer,
and every conclusion in the writeup inherits that ambiguity. This table is
what lets the project say which it was — and it is a self-contained study
worth presenting on its own.

It also tests, rather than assumes, the two design bets made in C10 and C11:
that fusion beats either arm alone, and that reranking beats fusion. §8's exit
criterion says "R4 ≥ R3 ≥ max(R1, R2), **or you have an explanation for why
not**". The second clause is the honest one and it is why this runs before any
agent evaluation.

Methodology
-----------
**Pooled judgments** (the TREC convention). Candidates are the union of the
top-N from every configuration, so no system is advantaged by having its
results labelled more thoroughly. Everything unjudged is treated as
irrelevant — standard, and sound only because the pool is the union of all
systems under comparison. It would be invalid to add a fifth system later and
score it against this pool without re-pooling.

**Graded relevance**: 0 irrelevant, 1 partially relevant, 2 relevant. nDCG
uses exponential gain `2^rel - 1`, so a fully relevant passage is worth three
partial ones — which matches how evidence actually works here: one passage
that answers the sub-question beats three that circle it.

**Recall is pool-relative.** The denominator is the judged-relevant set, not
all relevant passages in the corpus (unknowable without exhaustive labelling).
Report it as such; it compares systems fairly but is not an absolute.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

log = logging.getLogger(__name__)

RELEVANT = 2
PARTIAL = 1
IRRELEVANT = 0

# The four configurations under comparison (§4.5).
CONFIGS = {
    "R1": "Dense only",
    "R2": "BM25 only",
    "R3": "RRF fusion",
    "R4": "RRF + cross-encoder rerank",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def gain(relevance: int) -> float:
    """Exponential gain: a fully relevant passage is worth three partial ones."""
    return (2.0 ** relevance) - 1.0


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    """Discounted cumulative gain over the first k results."""
    return sum(
        gain(rel) / math.log2(rank + 1)
        for rank, rel in enumerate(relevances[:k], start=1)
    )


def ndcg_at_k(relevances: Sequence[int], all_relevances: Sequence[int], k: int) -> float:
    """nDCG@k against the best achievable ranking of the judged set.

    `all_relevances` is every judgment for this query, which defines the ideal
    ranking. Normalising against only the retrieved items would score a system
    that retrieved one relevant passage as perfectly as one that retrieved ten.
    """
    ideal = dcg_at_k(sorted(all_relevances, reverse=True), k)
    if ideal == 0.0:
        # No relevant passage exists for this query — the query cannot
        # discriminate between systems, so it is excluded upstream rather
        # than scored as 0 (which would punish every system equally and
        # dilute the comparison).
        return 0.0
    return dcg_at_k(relevances, k) / ideal


def recall_at_k(
    relevances: Sequence[int], all_relevances: Sequence[int], k: int,
    *, threshold: int = PARTIAL,
) -> float:
    """Fraction of the judged-relevant set retrieved in the top k.

    Pool-relative — see the module docstring.
    """
    total = sum(1 for r in all_relevances if r >= threshold)
    if total == 0:
        return 0.0
    found = sum(1 for r in relevances[:k] if r >= threshold)
    return found / total


def reciprocal_rank(relevances: Sequence[int], *, threshold: int = RELEVANT) -> float:
    """1/rank of the first fully relevant result.

    Reported alongside nDCG because the agent consumes the top few passages;
    a system that puts the answer at rank 1 is materially different from one
    that puts it at rank 8, and nDCG@10 partly hides that.
    """
    for rank, rel in enumerate(relevances, start=1):
        if rel >= threshold:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Judgments
# ---------------------------------------------------------------------------


@dataclass
class Judgments:
    """Relevance labels: query id -> passage id -> grade."""

    by_query: dict[str, dict[str, int]] = field(default_factory=dict)
    queries: dict[str, str] = field(default_factory=dict)   # id -> query text

    @classmethod
    def load(cls, path: str | Path) -> "Judgments":
        data = json.loads(Path(path).read_text())
        by_query: dict[str, dict[str, int]] = {}
        queries: dict[str, str] = {}
        for entry in data.get("queries", []):
            qid = str(entry["id"])
            queries[qid] = entry.get("query", "")
            by_query[qid] = {
                str(pid): int(grade)
                for pid, grade in (entry.get("judgments") or {}).items()
                if grade is not None
            }
        return cls(by_query=by_query, queries=queries)

    def grade(self, query_id: str, passage_id: str) -> int:
        """Unjudged means irrelevant — valid only because the pool is the
        union of all systems under comparison."""
        return self.by_query.get(query_id, {}).get(passage_id, IRRELEVANT)

    def all_grades(self, query_id: str) -> list[int]:
        return list(self.by_query.get(query_id, {}).values())

    def has_relevant(self, query_id: str, *, threshold: int = PARTIAL) -> bool:
        return any(g >= threshold for g in self.all_grades(query_id))

    def judged_query_ids(self) -> list[str]:
        return sorted(self.by_query)

    def coverage(self) -> dict[str, Any]:
        judged = sum(len(v) for v in self.by_query.values())
        relevant = sum(
            1 for v in self.by_query.values() for g in v.values() if g >= PARTIAL
        )
        with_relevant = sum(1 for q in self.by_query if self.has_relevant(q))
        return {
            "queries": len(self.by_query),
            "judgments": judged,
            "relevant": relevant,
            "queries_with_relevant": with_relevant,
            "mean_judgments_per_query": (
                judged / len(self.by_query) if self.by_query else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class ConfigScore:
    config: str
    label: str
    ndcg_at_10: float
    recall_at_20: float
    mrr: float
    queries: int

    def as_row(self) -> str:
        return (
            f"{self.config:<4} {self.label:<28} "
            f"{self.ndcg_at_10:>8.4f} {self.recall_at_20:>10.4f} {self.mrr:>7.4f}"
        )


def evaluate_ranking(
    query_id: str,
    ranked_passage_ids: Sequence[str],
    judgments: Judgments,
    *,
    ndcg_k: int = 10,
    recall_k: int = 20,
) -> dict[str, float]:
    grades = [judgments.grade(query_id, pid) for pid in ranked_passage_ids]
    all_grades = judgments.all_grades(query_id)
    return {
        "ndcg": ndcg_at_k(grades, all_grades, ndcg_k),
        "recall": recall_at_k(grades, all_grades, recall_k),
        "mrr": reciprocal_rank(grades),
    }


class RetrievalEvaluator:
    """Runs R1-R4 over a labelled query set and produces the comparison table."""

    def __init__(
        self,
        retriever: Any,
        judgments: Judgments,
        *,
        reranker: Any = None,
        ndcg_k: int = 10,
        recall_k: int = 20,
        depth: int = 20,
    ) -> None:
        self.retriever = retriever
        self.judgments = judgments
        self.reranker = reranker
        self.ndcg_k = ndcg_k
        self.recall_k = recall_k
        self.depth = depth

    def rank(self, config: str, query: str) -> list[str]:
        """Ranked passage ids for one configuration."""
        if config == "R1":
            results = self.retriever.retrieve(query, top_k=self.depth, dense_only=True)
        elif config == "R2":
            results = self.retriever.retrieve(query, top_k=self.depth, sparse_only=True)
        elif config in ("R3", "R4"):
            results = self.retriever.retrieve(query, top_k=self.depth)
            if config == "R4":
                if self.reranker is None:
                    raise ValueError("R4 requires a reranker")
                results = self.reranker.rerank(query, results, top_k=self.depth)
        else:
            raise ValueError(f"unknown configuration {config!r}")
        return [r.passage["id"] for r in results]

    def evaluate(self, configs: Iterable[str] = tuple(CONFIGS)) -> list[ConfigScore]:
        # Queries with no relevant passage in the pool cannot discriminate
        # between systems; including them would drag every score toward zero
        # equally and dilute the comparison without adding information.
        query_ids = [
            qid for qid in self.judgments.judged_query_ids()
            if self.judgments.has_relevant(qid)
        ]
        if not query_ids:
            raise ValueError("no queries have any relevant judgment")

        scores: list[ConfigScore] = []
        for config in configs:
            per_query = []
            for qid in query_ids:
                ranked = self.rank(config, self.judgments.queries[qid])
                per_query.append(
                    evaluate_ranking(
                        qid, ranked, self.judgments,
                        ndcg_k=self.ndcg_k, recall_k=self.recall_k,
                    )
                )
            scores.append(ConfigScore(
                config=config,
                label=CONFIGS.get(config, config),
                ndcg_at_10=_mean(m["ndcg"] for m in per_query),
                recall_at_20=_mean(m["recall"] for m in per_query),
                mrr=_mean(m["mrr"] for m in per_query),
                queries=len(query_ids),
            ))
        return scores


def render_table(scores: Sequence[ConfigScore], *, ndcg_k: int = 10,
                 recall_k: int = 20) -> str:
    header = (
        f"{'':<4} {'configuration':<28} "
        f"{'nDCG@' + str(ndcg_k):>8} {'recall@' + str(recall_k):>10} {'MRR':>7}"
    )
    lines = [header, "-" * len(header)]
    lines.extend(s.as_row() for s in scores)
    return "\n".join(lines)


def check_expected_ordering(scores: Sequence[ConfigScore]) -> list[str]:
    """Test §8's exit criterion: R4 >= R3 >= max(R1, R2).

    Returns the violations. An empty list means the design bets from C10 and
    C11 held on this corpus; a non-empty one is a finding to explain, not a
    failure to hide.
    """
    by_name = {s.config: s for s in scores}
    problems: list[str] = []

    def ndcg(name: str) -> float | None:
        return by_name[name].ndcg_at_10 if name in by_name else None

    r1, r2, r3, r4 = (ndcg(n) for n in ("R1", "R2", "R3", "R4"))
    if r3 is not None and r1 is not None and r2 is not None:
        best_single = max(r1, r2)
        if r3 < best_single:
            problems.append(
                f"fusion (R3 {r3:.4f}) did not beat the best single retriever "
                f"({best_single:.4f}) — RRF is not helping on this corpus"
            )
    if r4 is not None and r3 is not None and r4 < r3:
        problems.append(
            f"reranking (R4 {r4:.4f}) did not beat fusion (R3 {r3:.4f}) — the "
            f"cross-encoder is not earning its latency"
        )
    return problems


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0