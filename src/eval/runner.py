"""Run one configuration over one split, resumably (C32).

    EvalRunner(config_name, split, out_root).run()

One question at a time, sequentially (§2.5). Every rule the open table
accumulated for C32 is applied here, and each is tested:

  pinned roles          every role pinned (C27.6/7): configurations differ
                        only in their switches, not in which model answered
  isolated cache        a fresh LLM cache under the eval directory, shared
                        across configurations, never the development cache
                        (C27.6): identical prompts across configurations
                        then get identical responses, which is wanted
  skip completed        a question with a result is not re-run
  resume interrupted    a question with a checkpoint but no result is
                        RESUMED (invoke(None)), with its budget restored —
                        `run_research(resume=True)`, made real at C32
  re-run failures       a result whose rule is system_failure is RE-RUN on a
                        new thread, not resumed: its graph reached FINISH, so
                        a resume would replay the failure (C27.6)
  stop on quota         a pinned model out of day quota fails every later
                        question too; the batch stops instead of writing a
                        page of identical system failures
  one directory, one    a manifest records the configuration, pins and the
  configuration         split hash; continuing a directory under different
                        settings is refused
  held-out guard        the test split needs `held_out=True` and forces the
                        calibrated scorer — the reported numbers can never be
                        produced by the heuristic by accident (C27)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.config import Config, get_config
from src.eval.configs import CONFIGS, EVAL_PINS

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "benchmark" / "questions.json"
QUOTA_MARKERS = ("out of quota", "per day", "(tpd)", "(rpd)", "daily")


class EvalRefused(RuntimeError):
    """A run the harness will not start, with the reason."""


@dataclass
class EvalSummary:
    config: str
    split: str
    done: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    resumed: list[str] = field(default_factory=list)
    rerun: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    stopped_on_quota: str = ""


def load_questions(split: str, path: Path = QUESTIONS) -> tuple[list[dict], dict]:
    data = json.loads(path.read_text())
    frozen = data.get("split_frozen")
    if not frozen:
        raise EvalRefused("the split is not frozen (C31); run scripts.make_split first")
    qs = [q for q in data["questions"] if split == "all" or q.get("split") == split]
    return qs, frozen


def eval_config(base: Config, name: str, split: str, out_root: Path) -> Config:
    spec = CONFIGS.get(name)
    if spec is None:
        raise EvalRefused(f"unknown configuration {name!r}; known: {sorted(CONFIGS)}")
    if not spec.available:
        raise EvalRefused(f"{name} ({spec.isolates}) is not built yet — {spec.note}")
    overrides: dict[str, Any] = {
        **spec.overrides,
        **EVAL_PINS,
        "llm.cache.dir": str(out_root / "cache"),
        "run.checkpoint_db": str(out_root / "checkpoints.sqlite"),
        "run.trace_dir": str(out_root / "traces"),
        # The reported numbers come from the fitted model, never the
        # heuristic (C27): on the test split a missing model is an error.
        "sufficiency.scorer": "calibrated" if split == "test" else "auto",
    }
    return base.with_overrides(overrides)


class EvalRunner:
    def __init__(
        self,
        config_name: str,
        split: str,
        out_root: Path,
        *,
        held_out: bool = False,
        base_config: Config | None = None,
        components_factory: Callable[[Config], Any] | None = None,
        run_fn: Callable[..., Any] | None = None,
        questions_path: Path = QUESTIONS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if split == "test" and not held_out:
            raise EvalRefused(
                "the test split is the held-out half, run once at C36; pass "
                "held_out=True (CLI: --held-out) to confirm")
        if split not in ("calibration", "test", "all"):
            raise EvalRefused(f"unknown split {split!r}")
        self.name, self.split = config_name, split
        self.out_root = Path(out_root)
        self.dir = self.out_root / config_name / split
        self.cfg = eval_config(base_config or get_config(), config_name, split, self.out_root)
        self.questions, self.frozen = load_questions(split, questions_path)
        self._components_factory = components_factory
        self._components = None
        self._run_fn = run_fn
        self._clock = clock

    # -- public ---------------------------------------------------------------

    def run(self, limit: int | None = None, only: list[str] | None = None) -> EvalSummary:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._check_manifest()
        summary = EvalSummary(self.name, self.split)
        todo = [q for q in self.questions if not only or q["id"] in only]
        attempted = 0
        for q in todo:
            if limit is not None and attempted >= limit:
                break
            outcome = self._one(q, summary)
            if outcome == "skipped":
                continue
            attempted += 1
            if outcome == "quota":
                summary.stopped_on_quota = q["id"]
                log.warning("stopping: %s hit a model's day quota; continue after "
                            "the reset — completed questions are kept", q["id"])
                break
        return summary

    # -- one question ---------------------------------------------------------

    def _one(self, q: dict, summary: EvalSummary) -> str:
        qid = q["id"]
        path = self.dir / f"{qid}.json"
        previous = json.loads(path.read_text()) if path.exists() else None
        if previous and not _is_failure(previous):
            summary.skipped.append(qid)
            return "skipped"

        attempt = (previous or {}).get("attempt", 0) + (1 if previous else 0)
        thread = f"{self.name}:{self.split}:{qid}:a{attempt}"
        # A fresh trace id per invocation: the tracer truncates a file whose
        # id it is given, and a resume must not erase the crashed attempt's
        # trace — C38 annotates exactly those.
        trace_id = f"{self.name}-{self.split}-{qid}-a{attempt}-{int(self._clock())}"

        result = self._run(q["question"], thread_id=thread, trace_id=trace_id)
        record = self._record(q, result, attempt, thread)
        _write_atomic(path, record)

        if previous:
            summary.rerun.append(qid)
        elif result.resumed == "resumed":
            summary.resumed.append(qid)
        if _is_failure(record):
            summary.failed.append(qid)
            if _is_quota(record):
                return "quota"
            return "failed"
        summary.done.append(qid)
        return "done"

    def _run(self, question: str, *, thread_id: str, trace_id: str):
        if self._run_fn is not None:
            return self._run_fn(question, config=self.cfg, thread_id=thread_id,
                                trace_id=trace_id, resume=True)
        from src.graph import run_research

        if self._components is None:
            factory = self._components_factory
            if factory is None:
                from src.eval.components import frozen_components as factory
            self._components = factory(self.cfg)
        return run_research(question, components=self._components, config=self.cfg,
                            thread_id=thread_id, trace_id=trace_id, resume=True)

    def _record(self, q: dict, result: Any, attempt: int, thread: str) -> dict:
        state = getattr(result, "state", {}) or {}
        decision = state.get("decision") or {}
        budget = getattr(result, "budget", None)
        return {
            "id": q["id"], "tier": q["tier"], "subtype": q.get("subtype"),
            "split": q.get("split"), "config": self.name,
            "attempt": attempt, "thread_id": thread,
            "resumed": getattr(result, "resumed", ""),
            "terminal_decision": state.get("terminal_decision"),
            "rule": decision.get("rule"),
            "score": decision.get("score"),
            "calibrated": decision.get("calibrated"),
            "decision": decision,
            "final_report": state.get("final_report", ""),
            "critic_findings": state.get("critic_findings", []),
            "premise_check": state.get("premise_check", {}),
            "budget": budget.to_state() if budget is not None else state.get("budget"),
            "node_visits": list(getattr(result, "node_visits", []) or []),
            "trace_path": str(getattr(result, "trace_path", "") or ""),
            "error": getattr(result, "error", None) or state.get("error"),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    # -- manifest -------------------------------------------------------------

    def _check_manifest(self) -> None:
        spec = CONFIGS[self.name]
        manifest = {
            "config": self.name,
            "isolates": spec.isolates,
            "overrides": spec.overrides,
            "pins": EVAL_PINS,
            "split": self.split,
            "split_hash": self.frozen.get("hash"),
            "scorer": self.cfg.get("sufficiency.scorer"),
        }
        path = self.dir / "manifest.json"
        if path.exists():
            existing = json.loads(path.read_text())
            existing.pop("started_at", None)
            if existing != manifest:
                raise EvalRefused(
                    f"{self.dir} was started under different settings; results "
                    f"from two configurations must not share a directory.\n"
                    f"  existing: {existing}\n  now:      {manifest}")
            return
        _write_atomic(path, {**manifest, "started_at":
                             datetime.now(timezone.utc).isoformat(timespec="seconds")})


def _is_failure(record: dict) -> bool:
    return record.get("rule") == "system_failure" or (
        bool(record.get("error")) and not record.get("terminal_decision"))


def _is_quota(record: dict) -> bool:
    text = " ".join(str(record.get(k) or "") for k in ("error",)).lower()
    text += " " + str((record.get("decision") or {}).get("rationale") or "").lower()
    return any(marker in text for marker in QUOTA_MARKERS)


def _write_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    tmp.replace(path)
