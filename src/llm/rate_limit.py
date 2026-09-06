"""Local quota accounting — the thing that makes a free-tier eval survivable.

Why this exists
---------------
Both providers enforce FOUR limits simultaneously (requests/min, requests/day,
tokens/min, tokens/day) and return 429 on whichever is hit first. On this
project's free tier the binding constraint is *tokens per day*, not requests:

    openai/gpt-oss-120b   30 RPM  |  1K RPD  |   8K TPM  |  200K TPD
    openai/gpt-oss-20b    30 RPM  |  1K RPD  |   8K TPM  |  200K TPD
    qwen/qwen3.8-27b      30 RPM  |  1K RPD  |   8K TPM  |  200K TPD
    gemini-2.5-flash       5 RPM  |   20 RPD |  250K TPM |  (no published TPD)

At ~3k tokens per call, 200K TPD is ~66 calls/day — the 1K RPD ceiling is
never reached. Discovering that by collecting 429s wastes the quota you are
trying to conserve, so we predict spend locally and refuse in advance.

State is persisted to disk because a benchmark run spans days and gets
interrupted. A ledger that forgets on restart is worse than none: it would
happily re-spend a quota that is already gone.

Daily windows reset in the provider's stated timezone (Google resets at
midnight Pacific; we treat Groq as UTC), so the reset boundary is per model.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

Dimension = Literal["rpm", "rpd", "tpm", "tpd"]

# Events older than this cannot affect any sliding-minute window.
_MINUTE_WINDOW_S = 60.0
_EVENT_RETENTION_S = 130.0


@dataclass(frozen=True)
class ModelLimits:
    """Published limits for one model. None means "no published limit"."""

    rpm: int | None = None
    rpd: int | None = None
    tpm: int | None = None
    tpd: int | None = None
    reset_timezone: str = "UTC"

    @classmethod
    def from_config(cls, raw: dict) -> "ModelLimits":
        return cls(
            rpm=raw.get("rpm"),
            rpd=raw.get("rpd"),
            tpm=raw.get("tpm"),
            tpd=raw.get("tpd"),
            reset_timezone=raw.get("reset_timezone", "UTC"),
        )


@dataclass
class QuotaDecision:
    """Answer to 'may I spend est_tokens on this model right now?'"""

    allowed: bool
    dimension: Dimension | None = None
    wait_seconds: float = 0.0
    recoverable_today: bool = True
    detail: str = ""

    def __bool__(self) -> bool:
        return self.allowed


@dataclass
class _ModelState:
    """Rolling event log plus daily counters for one model."""

    events: list[tuple[float, int]] = field(default_factory=list)  # (monotonic-ish ts, tokens)
    day_key: str = ""
    day_requests: int = 0
    day_tokens: int = 0

    def prune(self, now: float) -> None:
        cutoff = now - _EVENT_RETENTION_S
        if self.events and self.events[0][0] < cutoff:
            self.events = [e for e in self.events if e[0] >= cutoff]

    def minute_window(self, now: float) -> tuple[int, int]:
        cutoff = now - _MINUTE_WINDOW_S
        reqs = 0
        toks = 0
        for ts, tk in self.events:
            if ts >= cutoff:
                reqs += 1
                toks += tk
        return reqs, toks

    def oldest_in_minute(self, now: float) -> float | None:
        cutoff = now - _MINUTE_WINDOW_S
        for ts, _ in self.events:
            if ts >= cutoff:
                return ts
        return None


class QuotaExhausted(RuntimeError):
    """Raised when no model in a role's chain has quota left today."""

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class QuotaLedger:
    """Thread-safe, disk-persisted quota accounting across all models.

    Uses wall-clock time (not monotonic) because state must survive process
    restarts, which monotonic clocks cannot express.
    """

    def __init__(self, path: str | Path, limits: dict[str, ModelLimits]) -> None:
        self.path = Path(path)
        self.limits = limits
        self._states: dict[str, _ModelState] = {}
        self._lock = threading.RLock()
        self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            # A corrupt ledger must not brick the run. Starting fresh risks
            # over-spending once; refusing to start guarantees zero progress.
            return
        for model, data in raw.get("models", {}).items():
            self._states[model] = _ModelState(
                events=[tuple(e) for e in data.get("events", [])],
                day_key=data.get("day_key", ""),
                day_requests=int(data.get("day_requests", 0)),
                day_tokens=int(data.get("day_tokens", 0)),
            )

    def _save(self) -> None:
        payload = {
            "version": 1,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "models": {
                model: {
                    "events": [list(e) for e in st.events],
                    "day_key": st.day_key,
                    "day_requests": st.day_requests,
                    "day_tokens": st.day_tokens,
                }
                for model, st in self._states.items()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace — an interrupted write must not corrupt the ledger.
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    # -- internals ----------------------------------------------------------

    def _limits_for(self, model: str) -> ModelLimits:
        if model not in self.limits:
            raise KeyError(f"no published limits configured for model {model!r}")
        return self.limits[model]

    def _day_key(self, model: str, now: float) -> str:
        tz = ZoneInfo(self._limits_for(model).reset_timezone)
        return datetime.fromtimestamp(now, tz).strftime("%Y-%m-%d")

    def _seconds_to_day_reset(self, model: str, now: float) -> float:
        tz = ZoneInfo(self._limits_for(model).reset_timezone)
        local = datetime.fromtimestamp(now, tz)
        tomorrow = local.replace(hour=0, minute=0, second=0, microsecond=0)
        return (tomorrow.timestamp() + 86400) - now

    def _state(self, model: str, now: float) -> _ModelState:
        st = self._states.setdefault(model, _ModelState())
        today = self._day_key(model, now)
        if st.day_key != today:
            st.day_key = today
            st.day_requests = 0
            st.day_tokens = 0
        st.prune(now)
        return st

    # -- public API ---------------------------------------------------------

    def check(self, model: str, est_tokens: int, *, now: float | None = None) -> QuotaDecision:
        """May we spend roughly `est_tokens` on `model` right now?

        Minute-window blocks are recoverable (wait and retry). Day-window
        blocks are not recoverable today — the caller should fall through to
        another model rather than sleep for hours.
        """
        now = time.time() if now is None else now
        lim = self._limits_for(model)
        with self._lock:
            st = self._state(model, now)
            reqs_min, toks_min = st.minute_window(now)

            if lim.rpd is not None and st.day_requests + 1 > lim.rpd:
                return QuotaDecision(
                    False, "rpd", self._seconds_to_day_reset(model, now), False,
                    f"{model}: {st.day_requests}/{lim.rpd} requests used today",
                )
            if lim.tpd is not None and st.day_tokens + est_tokens > lim.tpd:
                return QuotaDecision(
                    False, "tpd", self._seconds_to_day_reset(model, now), False,
                    f"{model}: {st.day_tokens}/{lim.tpd} tokens used today, "
                    f"need {est_tokens}",
                )
            if lim.rpm is not None and reqs_min + 1 > lim.rpm:
                oldest = st.oldest_in_minute(now)
                wait = max(0.0, (oldest + _MINUTE_WINDOW_S) - now) if oldest else 1.0
                return QuotaDecision(
                    False, "rpm", wait, True,
                    f"{model}: {reqs_min}/{lim.rpm} requests in the last minute",
                )
            if lim.tpm is not None and toks_min + est_tokens > lim.tpm:
                oldest = st.oldest_in_minute(now)
                wait = max(0.0, (oldest + _MINUTE_WINDOW_S) - now) if oldest else 1.0
                # A single call larger than the whole TPM budget can never
                # pass. Say so rather than looping forever.
                if est_tokens > lim.tpm:
                    return QuotaDecision(
                        False, "tpm", 0.0, False,
                        f"{model}: single call needs {est_tokens} tokens but "
                        f"TPM ceiling is {lim.tpm} — reduce prompt size",
                    )
                return QuotaDecision(
                    False, "tpm", wait, True,
                    f"{model}: {toks_min}/{lim.tpm} tokens in the last minute, "
                    f"need {est_tokens}",
                )
            return QuotaDecision(True)

    def record(self, model: str, tokens: int, *, now: float | None = None) -> None:
        """Record one completed call. Always called with ACTUAL token usage."""
        now = time.time() if now is None else now
        with self._lock:
            st = self._state(model, now)
            st.events.append((now, tokens))
            st.day_requests += 1
            st.day_tokens += tokens
            self._save()

    def snapshot(self, *, now: float | None = None) -> dict[str, dict]:
        """Current utilisation per model. For logging and the eval runner."""
        now = time.time() if now is None else now
        out: dict[str, dict] = {}
        with self._lock:
            for model, lim in self.limits.items():
                st = self._state(model, now)
                reqs_min, toks_min = st.minute_window(now)
                out[model] = {
                    "rpm": [reqs_min, lim.rpm],
                    "tpm": [toks_min, lim.tpm],
                    "rpd": [st.day_requests, lim.rpd],
                    "tpd": [st.day_tokens, lim.tpd],
                    "day_key": st.day_key,
                }
        return out

    def remaining_tokens_today(self, model: str, *, now: float | None = None) -> int | None:
        now = time.time() if now is None else now
        lim = self._limits_for(model)
        if lim.tpd is None:
            return None
        with self._lock:
            return max(0, lim.tpd - self._state(model, now).day_tokens)