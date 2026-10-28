"""Graph assembly.

Hub and spoke, deliberately
---------------------------
Every node returns to the controller, which decides what runs next. The
alternative — hardcoded edges between workers, with conditional edges where
the path branches — spreads routing across the graph definition and makes
"why did it stop researching?" a question about four files. Here it is a
question about one function.

In LangGraph terms every node gets `add_conditional_edges(node,
controller.route, path_map)`, including the entry point. The controller is
the path function, so the routing table in `controller.py` IS the graph's
control flow rather than a description of it.

Durable execution
-----------------
Compiled with `SqliteSaver`, so a run checkpoints after every node. §7.1
needs this: the eval spans days, Gemini and Groq both rate-limit, and a run
that cannot resume has to restart and re-spend the quota it already used. C32
makes the eval runner resumable on top of this.
"""
from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.budget import Budget
from src.config import Config, get_config
from src.controller import (
    ALL_NODES,
    FINISH,
    PLANNER,
    PREMISE_CHECKER,
    RETRIEVER,
    SYNTHESIZER,
    Controller,
)
from src.state import ResearchState, new_state
from src.tracing.tracer import Tracer

log = logging.getLogger(__name__)

# Guards against a routing bug looping forever. Generous: 7 sub-questions
# across up to 3 rounds each, plus revisions and the downstream chain.
DEFAULT_RECURSION_LIMIT = 60


@dataclass
class Components:
    """The node callables the graph registers.

    Each is optional. The controller skips any stage whose node is absent,
    so C19 assembles a working graph from four of them and C23/C27/C28 add
    the rest without touching routing.
    """

    premise_checker: Any = None
    planner: Any = None
    retriever: Any = None
    sufficiency_scorer: Any = None
    synthesizer: Any = None
    critic: Any = None
    critic_router: Any = None
    decider: Any = None
    finalizer: Any = None

    def registered(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in ALL_NODES
            if getattr(self, name, None) is not None
        }


@dataclass
class RunResult:
    state: ResearchState
    trace_path: Path | None = None
    budget: Budget | None = None
    error: str | None = None
    node_visits: list[str] = field(default_factory=list)


def build_graph(
    components: Components,
    *,
    config: Config | None = None,
    budget: Budget | None = None,
    tracer: Any = None,
    checkpointer: Any = None,
) -> Any:
    """Assemble and compile the StateGraph."""
    from langgraph.graph import START, StateGraph

    cfg = config or get_config()
    nodes = components.registered()
    if not nodes:
        raise ValueError("no nodes registered — nothing to run")

    controller = Controller(
        cfg, available=set(nodes), budget=budget, tracer=tracer
    )
    builder = StateGraph(ResearchState)
    for name, action in nodes.items():
        builder.add_node(name, _traced(name, action, tracer))

    # Every node, and the entry point, routes through the controller.
    path_map = {name: name for name in nodes}
    path_map[FINISH] = FINISH
    builder.add_conditional_edges(START, controller.route, path_map)
    for name in nodes:
        builder.add_conditional_edges(name, controller.route, path_map)

    return builder.compile(checkpointer=checkpointer)


def _traced(name: str, action: Any, tracer: Any) -> Any:
    """Wrap a node so its entry, exit and duration reach the trace.

    C4 built the span machinery; until now nothing used it, so traces
    recorded routing but not what ran between routes — no node durations, and
    every routing decision appeared to originate at START because the node
    stack was always empty. C38 replays failed runs from these traces, and
    "which node was slow" and "what ran before this decision" are among the
    first questions it asks.
    """
    if tracer is None:
        return action

    def wrapped(state: dict) -> dict:
        with tracer.node(name):
            return action(state)

    wrapped.__name__ = name
    return wrapped


def _checkpointer(cfg: Config, enabled: bool) -> Iterator[Any]:
    if not enabled:
        return nullcontext(None)  # type: ignore[return-value]
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(cfg.get("run.checkpoint_db", "data/checkpoints.sqlite"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver.from_conn_string(str(path))


def run_research(
    question: str,
    *,
    components: Components | None = None,
    config: Config | None = None,
    thread_id: str | None = None,
    trace_id: str | None = None,
    checkpoint: bool = True,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> RunResult:
    """Run one research question end to end.

    `thread_id` identifies the run to the checkpointer. Passing the same one
    again resumes rather than restarts — which is what makes an interrupted
    eval cheap to continue (C32).
    """
    cfg = config or get_config()
    components = components or default_components(cfg)

    tracer = Tracer(
        cfg.path("run.trace_dir"),
        trace_id=trace_id,
        question=question,
        metadata={"thread_id": thread_id or trace_id},
    )
    budget = Budget.from_config(cfg, tracer=tracer)
    _attach(components, tracer=tracer, budget=budget)

    state = new_state(question, trace_id=tracer.trace_id, budget=budget.to_state())
    result = RunResult(state=state, trace_path=tracer.path, budget=budget)

    try:
        with _checkpointer(cfg, checkpoint) as saver:
            graph = build_graph(
                components, config=cfg, budget=budget,
                tracer=tracer, checkpointer=saver,
            )
            run_config = {
                "configurable": {"thread_id": thread_id or tracer.trace_id},
                "recursion_limit": recursion_limit,
            }
            result.state = graph.invoke(state, run_config)
    except Exception as exc:  # noqa: BLE001
        # A crashed run still has a trace and a partial state, and both are
        # worth more than a stack trace alone when annotating at C38.
        log.exception("run failed for %r", question[:80])
        result.error = f"{type(exc).__name__}: {exc}"
        from src.tracing.tracer import EventType

        tracer.emit(EventType.ERROR, "run",
                    error_type=type(exc).__name__, error=str(exc))
    finally:
        tracer.close(
            ok=result.error is None,
            terminal_decision=result.state.get("terminal_decision"),
            budget=budget.to_state(),
        )

    from src.tracing.tracer import TraceReader

    try:
        result.node_visits = TraceReader.load(tracer.path).nodes_visited()
    except OSError:
        pass
    return result


def default_components(config: Config | None = None) -> Components:
    """Everything built so far, wired with real models.

    Imported lazily so that constructing a Components by hand — which the
    tests do — never loads a 500 MB cross-encoder.
    """
    cfg = config or get_config()

    from src.nodes.adaptive_retriever import AdaptiveRetriever
    from src.nodes.critic import Critic, critic_node
    from src.nodes.critic_router import CriticRouter, critic_router_node
    from src.nodes.decider import Decider, decider_node
    from src.nodes.planner import Planner, planner_node
    from src.nodes.premise_checker import PremiseChecker, premise_checker_node
    from src.nodes.retriever import StaticRetriever, retriever_node
    from src.nodes.synthesizer import Synthesizer, synthesizer_node

    # Which retriever runs is decided by the agency flags, so the A2 and A3
    # ablations are a config change rather than a code path. When BOTH are
    # frozen the static class is used directly: the adaptive class with
    # every decision disabled would behave identically, but running the
    # original makes "the ablation restores C17" a fact rather than a claim.
    d2 = bool(cfg.get("agency.d2_adaptive_route_and_query", True))
    d3 = bool(cfg.get("agency.d3_adaptive_depth_and_stopping", True))
    retriever = (
        StaticRetriever(cfg) if not (d2 or d3) else AdaptiveRetriever(cfg)
    )

    from src.sufficiency.allocation import get_allocator

    return Components(
        premise_checker=premise_checker_node(PremiseChecker(cfg)),
        planner=planner_node(Planner(cfg)),
        retriever=retriever_node(retriever, get_allocator(cfg)),
        synthesizer=synthesizer_node(Synthesizer(cfg)),
        critic=critic_node(Critic(cfg)),
        critic_router=critic_router_node(CriticRouter(cfg)),
        decider=decider_node(Decider(cfg)),
    )


def _attach(components: Components, *, tracer: Any, budget: Budget) -> None:
    """Give every node the run's tracer and budget.

    Nodes are built once and reused across questions in an eval, but the
    tracer and budget are per-run. Rebinding here rather than rebuilding the
    components keeps model loading out of the per-question path — the
    cross-encoder alone takes seconds.
    """
    for name in ALL_NODES:
        node = getattr(components, name, None)
        inner = getattr(node, "__closure__", None)
        if not inner:
            continue
        for cell in inner:
            target = cell.cell_contents
            if hasattr(target, "tracer"):
                target.tracer = tracer
            if hasattr(target, "budget"):
                target.budget = budget
            # Sub-components (provider, retriever, reranker) trace too.
            for attr in ("provider", "retriever", "reranker", "_retriever"):
                child = getattr(target, attr, None)
                if child is not None and hasattr(child, "tracer"):
                    child.tracer = tracer