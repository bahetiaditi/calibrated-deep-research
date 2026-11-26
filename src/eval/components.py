"""The graph wired to the FROZEN corpus (C29's decision).

The same nodes as `graph.default_components`, with retrieval served from the
982-passage C13 collection — what every smoke test since C19 ran on — and
the retriever chosen by the D2/D3 switches exactly as default_components
chooses it, so ablations A2/A3 get the static retriever they freeze to.
"""
from __future__ import annotations

from pathlib import Path

from src.config import Config

COLLECTION = "retrieval_eval"


def frozen_retrieval(cfg: Config):
    """Hybrid retrieval + reranker over the frozen collection (C29)."""
    from qdrant_client import QdrantClient

    from src.rag.fusion import HybridRetriever
    from src.rag.rerank import Reranker
    from src.rag.sparse import BM25Index
    from src.rag.store import EvidenceStore

    path = Path(cfg.get("retrieval.store.path")) / COLLECTION
    store = EvidenceStore(cfg, client=QdrantClient(path=str(path)), collection=COLLECTION)
    hybrid = HybridRetriever(cfg, store=store, sparse=BM25Index(cfg))
    records, _ = store.client.scroll(collection_name=COLLECTION, limit=10_000, with_payload=True)
    passages = [dict(r.payload) for r in records if r.payload]
    if not passages:
        raise RuntimeError(f"frozen corpus is empty at {path}; "
                           f"run `python -m scripts.retrieval_labels build`")
    hybrid.sparse.build(passages)
    return hybrid, Reranker(cfg)


def frozen_components(cfg: Config):
    from src.graph import Components
    from src.nodes.adaptive_retriever import AdaptiveRetriever
    from src.nodes.critic import Critic, critic_node
    from src.nodes.critic_router import CriticRouter, critic_router_node
    from src.nodes.decider import Decider, decider_node
    from src.nodes.finalizer import Finalizer, finalizer_node
    from src.nodes.planner import Planner, planner_node
    from src.nodes.premise_checker import PremiseChecker, premise_checker_node
    from src.nodes.retriever import StaticRetriever, retriever_node
    from src.nodes.synthesizer import Synthesizer, synthesizer_node
    from src.rag.contradiction import ContradictionDetector
    from src.sufficiency.allocation import get_allocator

    hybrid, reranker = frozen_retrieval(cfg)
    detector = ContradictionDetector(cfg)

    adaptive = bool(cfg.get("agency.d2_adaptive_route_and_query", True)
                    or cfg.get("agency.d3_adaptive_depth_and_stopping", True))
    retriever_cls = AdaptiveRetriever if adaptive else StaticRetriever
    retriever = retriever_cls(cfg, retriever=hybrid, reranker=reranker, detector=detector)

    return Components(
        premise_checker=premise_checker_node(
            PremiseChecker(cfg, retriever=hybrid, reranker=reranker)),
        planner=planner_node(Planner(cfg)),
        retriever=retriever_node(retriever, get_allocator(cfg)),
        synthesizer=synthesizer_node(Synthesizer(cfg)),
        critic=critic_node(Critic(cfg)),
        critic_router=critic_router_node(CriticRouter(cfg)),
        decider=decider_node(Decider(cfg)),
        finalizer=finalizer_node(Finalizer(cfg)),
    )
