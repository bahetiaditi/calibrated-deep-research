"""Structured execution traces.

Why this exists, in order of importance
--------------------------------------
1. **It is how the agency claim is verified.** REFERENCE.md §2.1 commits to a
   falsifiable test: freeze a decision surface and measure the degradation.
   C35's check is "verify by trace inspection that the disabled surface really
   is disabled — e.g. A1 shows zero plan revisions." That check is only
   possible if every exercise of D1–D5 is recorded as a first-class event with
   the alternatives that were considered. Hence `DecisionSurface` below: it is
   not decoration, it is the evidence.

2. **It is the substrate for MAST annotation (§6.6, C38).** Annotating a run
   against the 14 failure modes requires seeing what actually happened —
   which node ran, what it was given, what it chose, what came back.

3. **It makes failures debuggable.** An eval that dies at 3am leaves a trace
   up to the point of death, because every event is flushed on write.

Format
------
One JSONL file per run, named by `trace_id`. Append-only, one JSON object per
line, flushed immediately. A truncated final line (process killed mid-write)
costs one event, not the file — `TraceReader` skips unparseable lines rather
than failing.

On state snapshots
------------------
§3.3 requires routing decisions to be logged "with the state snapshot that
produced it". Dumping full state is not viable: evidence passages run to
kilobytes and a run makes tens of routing decisions. So routing events carry
a *summary* — the fields routing actually reads (plan status, sufficiency,
budget) — and all payloads pass through a truncator. A trace that is too
large to read is a trace nobody reads.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

TRACE_FORMAT_VERSION = 1

# Payload guards. Generous enough to keep decisions legible, tight enough
# that a 200-run benchmark's traces stay browsable.
MAX_STRING_CHARS = 2000
MAX_LIST_ITEMS = 25


class EventType(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    NODE_ENTER = "node_enter"
    NODE_EXIT = "node_exit"
    ROUTE = "route"            # controller chose the next node
    DECISION = "decision"      # an agency surface D1-D5 was exercised
    LLM_CALL = "llm_call"
    RETRIEVAL = "retrieval"
    ERROR = "error"
    NOTE = "note"


class DecisionSurface(str, Enum):
    """The five points where the system has agency (REFERENCE.md §2.2).

    Every exercise of one of these is recorded. Ablations freeze exactly one,
    and C35 confirms the freeze took effect by counting these events.
    """

    D1_PLAN_REVISION = "D1_plan_revision"
    D2_ROUTE_AND_QUERY = "D2_route_and_query"
    D3_DEPTH_AND_STOPPING = "D3_depth_and_stopping"
    D4_BUDGET_ALLOCATION = "D4_budget_allocation"
    D5_TERMINAL_DECISION = "D5_terminal_decision"
    CRITIC_ROUTING = "critic_five_way_routing"


def _truncate(value: Any, _depth: int = 0) -> Any:
    """Bound payload size while keeping structure recognisable."""
    if _depth > 6:
        return "<max depth>"
    if isinstance(value, str):
        if len(value) > MAX_STRING_CHARS:
            return value[:MAX_STRING_CHARS] + f"... <+{len(value) - MAX_STRING_CHARS} chars>"
        return value
    if isinstance(value, dict):
        return {str(k): _truncate(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_truncate(v, _depth + 1) for v in list(value)[:MAX_LIST_ITEMS]]
        if len(value) > MAX_LIST_ITEMS:
            items.append(f"<+{len(value) - MAX_LIST_ITEMS} more>")
        return items
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, Enum):
        return value.value
    return _truncate(str(value), _depth)


@dataclass
class TraceEvent:
    seq: int
    ts: str
    elapsed_ms: float
    type: str
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None

    def to_json(self) -> str:
        d = {
            "seq": self.seq,
            "ts": self.ts,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "type": self.type,
            "name": self.name,
        }
        if self.duration_ms is not None:
            d["duration_ms"] = round(self.duration_ms, 2)
        if self.payload:
            d["payload"] = self.payload
        return json.dumps(d, ensure_ascii=False, default=str)


class Tracer:
    """Writes one JSONL trace file for one run."""

    def __init__(
        self,
        trace_dir: str | Path,
        *,
        trace_id: str | None = None,
        question: str | None = None,
        metadata: dict[str, Any] | None = None,
        enabled: bool = True,
        now: Any = time.time,
    ) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.enabled = enabled
        self._now = now
        self._t0 = now()
        self._seq = 0
        self._lock = threading.RLock()
        self._closed = False
        self._node_stack: list[str] = []

        self.dir = Path(trace_dir)
        self.path = self.dir / f"{self.trace_id}.jsonl"
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")  # truncate any prior file for this id
            self.emit(
                EventType.RUN_START,
                "run",
                trace_format_version=TRACE_FORMAT_VERSION,
                trace_id=self.trace_id,
                question=question,
                **(metadata or {}),
            )

    # -- core ---------------------------------------------------------------

    def emit(
        self,
        event_type: EventType,
        name: str,
        *,
        duration_ms: float | None = None,
        **payload: Any,
    ) -> TraceEvent | None:
        if not self.enabled or self._closed:
            return None
        with self._lock:
            self._seq += 1
            if self._node_stack:
                payload.setdefault("node", self._node_stack[-1])
            event = TraceEvent(
                seq=self._seq,
                ts=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                elapsed_ms=(self._now() - self._t0) * 1000.0,
                type=event_type.value,
                name=name,
                payload=_truncate(payload),
                duration_ms=duration_ms,
            )
            # Flush per event: a run killed mid-eval must keep what it did.
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(event.to_json() + "\n")
                fh.flush()
            return event

    # -- spans --------------------------------------------------------------

    @contextmanager
    def node(self, name: str, **payload: Any) -> Iterator["Tracer"]:
        """Wrap a node's execution: enter, exit with duration, errors."""
        self.emit(EventType.NODE_ENTER, name, **payload)
        self._node_stack.append(name)
        start = self._now()
        try:
            yield self
        except Exception as exc:
            self._node_stack.pop()
            self.emit(
                EventType.ERROR,
                name,
                duration_ms=(self._now() - start) * 1000.0,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        else:
            self._node_stack.pop()
            self.emit(
                EventType.NODE_EXIT,
                name,
                duration_ms=(self._now() - start) * 1000.0,
            )

    # -- typed events -------------------------------------------------------

    def route(
        self,
        to_node: str,
        *,
        reason: str,
        state_summary: dict[str, Any] | None = None,
        from_node: str | None = None,
    ) -> None:
        """Record a controller routing decision.

        `state_summary` should be the fields routing actually read, not the
        whole state object — see the module docstring.
        """
        self.emit(
            EventType.ROUTE,
            to_node,
            from_node=from_node or (self._node_stack[-1] if self._node_stack else None),
            reason=reason,
            state=state_summary or {},
        )

    def decision(
        self,
        surface: DecisionSurface,
        *,
        chosen: Any,
        alternatives: list[Any] | None = None,
        rationale: str = "",
        inputs: dict[str, Any] | None = None,
        was_adaptive: bool = True,
    ) -> None:
        """Record an exercise of one of the five agency surfaces.

        `was_adaptive=False` marks a call made under an ablation, where the
        static policy chose. C35 counts adaptive decisions per surface to
        confirm a freeze actually took effect.
        """
        self.emit(
            EventType.DECISION,
            surface.value,
            surface=surface.value,
            chosen=chosen,
            alternatives=alternatives or [],
            rationale=rationale,
            inputs=inputs or {},
            was_adaptive=was_adaptive,
        )

    def llm_call(
        self,
        *,
        role: str,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency_s: float,
        from_cache: bool,
        fell_back: bool = False,
        attempts: int = 1,
    ) -> None:
        self.emit(
            EventType.LLM_CALL,
            model,
            duration_ms=latency_s * 1000.0,
            role=role,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            from_cache=from_cache,
            fell_back=fell_back,
            attempts=attempts,
        )

    def retrieval(
        self,
        *,
        route: str,
        query: str,
        n_results: int,
        latency_s: float,
        sub_question_id: str | None = None,
        **extra: Any,
    ) -> None:
        self.emit(
            EventType.RETRIEVAL,
            route,
            duration_ms=latency_s * 1000.0,
            query=query,
            n_results=n_results,
            sub_question_id=sub_question_id,
            **extra,
        )

    def note(self, name: str, **payload: Any) -> None:
        self.emit(EventType.NOTE, name, **payload)

    def close(self, **payload: Any) -> None:
        if not self.enabled or self._closed:
            return
        self.emit(
            EventType.RUN_END,
            "run",
            duration_ms=(self._now() - self._t0) * 1000.0,
            **payload,
        )
        self._closed = True

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.emit(EventType.ERROR, "run", error_type=exc_type.__name__, error=str(exc))
        self.close(ok=exc is None)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


class TraceReader:
    """Loads a trace and answers the questions C35 and C38 need to ask."""

    def __init__(self, events: list[dict[str, Any]], path: Path | None = None) -> None:
        self.events = events
        self.path = path

    @classmethod
    def load(cls, path: str | Path) -> "TraceReader":
        path = Path(path)
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # A process killed mid-write leaves one partial line. Losing
                # that event beats losing the whole trace.
                continue
        return cls(events, path)

    # -- queries ------------------------------------------------------------

    def of_type(self, event_type: EventType) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == event_type.value]

    @property
    def trace_id(self) -> str | None:
        starts = self.of_type(EventType.RUN_START)
        return starts[0].get("payload", {}).get("trace_id") if starts else None

    def nodes_visited(self) -> list[str]:
        return [e["name"] for e in self.of_type(EventType.NODE_ENTER)]

    def routes(self) -> list[tuple[str | None, str, str]]:
        return [
            (e.get("payload", {}).get("from_node"), e["name"],
             e.get("payload", {}).get("reason", ""))
            for e in self.of_type(EventType.ROUTE)
        ]

    def decisions(self, surface: DecisionSurface | None = None) -> list[dict[str, Any]]:
        out = self.of_type(EventType.DECISION)
        if surface is not None:
            out = [e for e in out if e.get("payload", {}).get("surface") == surface.value]
        return out

    def adaptive_decision_counts(self) -> dict[str, int]:
        """Adaptive exercises per surface — the C35 ablation check.

        An ablation that froze D1 must report zero for D1_plan_revision.
        A non-zero count means the freeze did not take effect and the
        ablation result is invalid.
        """
        counts = {s.value: 0 for s in DecisionSurface}
        for e in self.of_type(EventType.DECISION):
            p = e.get("payload", {})
            if p.get("was_adaptive"):
                counts[p.get("surface", "")] = counts.get(p.get("surface", ""), 0) + 1
        return counts

    def llm_totals(self) -> dict[str, Any]:
        calls = self.of_type(EventType.LLM_CALL)
        live = [c for c in calls if not c.get("payload", {}).get("from_cache")]
        return {
            "calls": len(calls),
            "live_calls": len(live),
            "cached_calls": len(calls) - len(live),
            "total_tokens": sum(
                c.get("payload", {}).get("total_tokens", 0) for c in calls
            ),
            "live_tokens": sum(
                c.get("payload", {}).get("total_tokens", 0) for c in live
            ),
            "by_model": {
                m: sum(1 for c in calls if c["name"] == m)
                for m in {c["name"] for c in calls}
            },
        }

    def errors(self) -> list[dict[str, Any]]:
        return self.of_type(EventType.ERROR)

    def wall_clock_ms(self) -> float:
        ends = self.of_type(EventType.RUN_END)
        if ends:
            return ends[-1].get("duration_ms", 0.0)
        return self.events[-1]["elapsed_ms"] if self.events else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "events": len(self.events),
            "nodes": self.nodes_visited(),
            "routes": len(self.of_type(EventType.ROUTE)),
            "decisions": self.adaptive_decision_counts(),
            "llm": self.llm_totals(),
            "retrievals": len(self.of_type(EventType.RETRIEVAL)),
            "errors": len(self.errors()),
            "wall_clock_ms": round(self.wall_clock_ms(), 1),
        }