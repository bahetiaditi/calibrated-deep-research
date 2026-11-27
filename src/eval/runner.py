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
  explicit re-run       `rerun={id: reason}` re-runs a COMPLETED question on a
                        new attempt and thread (C37.1), keeping what it
                        replaced under `supersedes`. Calibration only: on the
                        test split a re-run chosen after seeing a result is
                        cherry-picking, so it is refused there
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
# DAY-quota markers only (C32.1). "out of quota" was here, and it appears in
# every chain-exhausted error — including a MINUTE-window failure on a pinned
# model, which would have stopped the whole batch as if the day were gone.
# These name the day explicitly: Groq's "(TPD)"/"(RPD)", Gemini's
# "...PerDay...", and the local ledger's "tokens used today".
from src.llm.provider import DAY_QUOTA_MARKERS as QUOTA_MARKERS  # noqa: E402  (C37.4.1: one list)


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
    # C37.1: questions whose synthesis left citations unparsed, with the
    # count — shown after the run so a lost-citation draft is seen the day
    # it is produced, not at scoring.
    unparsed: dict[str, int] = field(default_factory=dict)


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
    if spec.pipeline == "derived":
        raise EvalRefused(f"{name} is not run; it is {spec.note}")
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

    def run(self, limit: int | None = None, only: list[str] | None = None,
            rerun: dict[str, str] | None = None) -> EvalSummary:
        rerun = dict(rerun or {})
        if rerun:
            self._check_rerun(rerun)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._check_manifest()
        summary = EvalSummary(self.name, self.split)
        todo = [q for q in self.questions if not only or q["id"] in only]
        if rerun:
            # A re-run names its questions; nothing else runs in that
            # invocation, so a forced re-run never quietly spends the day's
            # quota on the regular queue as well.
            todo = [q for q in self.questions if q["id"] in rerun]
        attempted = 0
        for q in todo:
            # Completed questions are counted before the limit is checked: a
            # limit counts ATTEMPTS. Until C32.1 the limit came first, so
            # `--limit 0` reported nothing at all, not the completed ones.
            if self._completed(q["id"]) and (q["id"] not in rerun
                                             or self._already_rerun(q["id"], rerun[q["id"]])):
                # An explicit re-run already done for the same reason is not
                # repeated: the day's quota can stop a batch of re-runs half
                # way, and re-issuing the same command must finish the batch,
                # not re-run its first half a second time (C37.3).
                summary.skipped.append(q["id"])
                rerun.pop(q["id"], None)
                continue
            if limit is not None and attempted >= limit:
                break
            outcome = self._one(q, summary, rerun_reason=rerun.pop(q["id"], None))
            attempted += 1
            if outcome == "quota":
                summary.stopped_on_quota = q["id"]
                log.warning("stopping: %s hit a model's day quota; continue after "
                            "the reset — completed questions are kept", q["id"])
                break
        return summary

    def remaining(self) -> list[str]:
        """Questions in this split without a successful result ON DISK.

        From the files, not from one invocation's tallies: C32's CLI printed
        "24 remaining" with 2 complete, because it subtracted only what that
        invocation had done or skipped.
        """
        return [q["id"] for q in self.questions if not self._completed(q["id"])]

    def _check_rerun(self, rerun: dict[str, str]) -> None:
        if self.split != "calibration":
            raise EvalRefused(
                "explicit re-runs are calibration-only: re-running a held-out "
                "question after seeing its result is choosing which result to "
                "report. Test-split system failures are re-run automatically.")
        known = {q["id"] for q in self.questions}
        unknown = sorted(set(rerun) - known)
        if unknown:
            raise EvalRefused(f"not in the {self.split} split: {', '.join(unknown)}")
        missing = sorted(i for i in rerun if not (self.dir / f"{i}.json").exists())
        if missing:
            raise EvalRefused(f"nothing to re-run (no result yet): {', '.join(missing)}")
        blank = sorted(i for i, why in rerun.items() if not (why or "").strip())
        if blank:
            raise EvalRefused(f"a re-run needs a reason, recorded in the result: {', '.join(blank)}")

    def _already_rerun(self, qid: str, reason: str) -> bool:
        try:
            record = json.loads((self.dir / f"{qid}.json").read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return (record.get("supersedes") or {}).get("reason") == reason

    def _completed(self, qid: str) -> bool:
        path = self.dir / f"{qid}.json"
        if not path.exists():
            return False
        try:
            return not _is_failure(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            return False

    # -- one question ---------------------------------------------------------

    def _one(self, q: dict, summary: EvalSummary, rerun_reason: str | None = None) -> str:
        qid = q["id"]
        path = self.dir / f"{qid}.json"
        previous = json.loads(path.read_text()) if path.exists() else None

        attempt = (previous or {}).get("attempt", 0) + (1 if previous else 0)
        thread = f"{self.name}:{self.split}:{qid}:a{attempt}"
        # A fresh trace id per invocation: the tracer truncates a file whose
        # id it is given, and a resume must not erase the crashed attempt's
        # trace — C38 annotates exactly those.
        trace_id = f"{self.name}-{self.split}-{qid}-a{attempt}-{int(self._clock())}"

        result = self._run(q["question"], thread_id=thread, trace_id=trace_id)
        record = self._record(q, result, attempt, thread)
        if rerun_reason and previous:
            # The replaced result stays visible in the record that replaced
            # it, not only in git history.
            record["supersedes"] = {
                "attempt": previous.get("attempt"),
                "thread_id": previous.get("thread_id"),
                "terminal_decision": previous.get("terminal_decision"),
                "rule": previous.get("rule"),
                "score": previous.get("score"),
                "reason": rerun_reason,
            }
        _write_atomic(path, record)

        lost = sum(int(e.get("unparsed_citations") or 0) for e in record["synthesis_log"])
        if lost:
            summary.unparsed[qid] = lost
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
        if CONFIGS[self.name].pipeline == "single_shot":
            from src.eval.baselines import single_shot_run
            return single_shot_run(question, config=self.cfg, thread_id=thread_id,
                                   trace_id=trace_id)
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
            # The raw draft and its mode (C34): B2/B3 are derived from the
            # agent's answer-mode draft, which final_report has already
            # rewritten with verification marks.
            "draft": state.get("draft", ""),
            "draft_mode": state.get("draft_mode", ""),
            "synthesis_log": list(state.get("synthesis_log") or []),
            "critic_findings": state.get("critic_findings", []),
            "premise_check": state.get("premise_check", {}),
            "budget": budget.to_state() if budget is not None else state.get("budget"),
            "node_visits": list(getattr(result, "node_visits", []) or []),
            "trace_path": str(getattr(result, "trace_path", "") or ""),
            "error": getattr(result, "error", None) or state.get("error"),
            # C37.4.1: LLM chains that gave up mid-run. Non-empty means the
            # nodes ran degraded (a critique half-judged, routing skipped):
            # the record is not a measurement of the system, whatever its
            # decision says, and is re-run like a system failure.
            "infra_failures": list(getattr(result, "degraded", []) or []),
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
    return record.get("rule") == "system_failure" or bool(record.get("infra_failures")) or (
        bool(record.get("error")) and not record.get("terminal_decision"))


def _is_quota(record: dict) -> bool:
    if any(f.get("day_quota") for f in record.get("infra_failures") or []):
        return True
    text = " ".join(str(record.get(k) or "") for k in ("error",)).lower()
    text += " " + str((record.get("decision") or {}).get("rationale") or "").lower()
    return any(marker in text for marker in QUOTA_MARKERS)


def _write_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    tmp.replace(path)
