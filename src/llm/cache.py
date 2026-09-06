"""Prompt-hash response cache.

Why this is not an optimization
-------------------------------
Development means running the same benchmark question over and over while
fixing the node above it. Without a cache, every debugging cycle re-spends
quota on calls whose inputs have not changed. With one, the second run of an
unchanged prompt costs nothing and takes microseconds.

There is a second, more important reason. Reported numbers must be
reproducible. When `analysis/score.py` regenerates a results table, it should
read the same model outputs that produced the original table — not re-roll
them. A cached response is a recorded observation; re-running the model is a
new experiment. Conflating the two makes results quietly unrepeatable.

Keying
------
sha256 over model, system, user, temperature, max_output_tokens and
json_mode. Every one of those changes the response, so every one is in the
key. Note the consequence: **temperature > 0 calls are frozen after their
first execution.** That is intended here — it makes runs reproducible — but
it does mean the cache must be bypassed when deliberately measuring
sampling variance, which is what `use_cache=False` is for.

Nothing is ever evicted. Disk is cheap, and a cache that forgets is a cache
that silently changes your results.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Bump when the stored payload shape changes. Old entries then read as misses
# rather than deserializing into something the code no longer understands.
CACHE_FORMAT_VERSION = 1


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    errors: int = 0
    tokens_saved: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def render(self) -> str:
        return (
            f"cache: {self.hits} hits / {self.lookups} lookups "
            f"({self.hit_rate:.0%}), {self.writes} writes, "
            f"{self.tokens_saved:,} tokens saved"
            + (f", {self.errors} errors" if self.errors else "")
        )


@dataclass
class CachedResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    created_at: str = ""
    version: int = CACHE_FORMAT_VERSION


class PromptCache:
    """Content-addressed disk cache for LLM responses."""

    def __init__(self, directory: str | Path, *, enabled: bool = True) -> None:
        self.dir = Path(directory)
        self.enabled = enabled
        self.stats = CacheStats()
        self._lock = threading.RLock()
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    # -- keying -------------------------------------------------------------

    @staticmethod
    def make_key(
        *,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_output_tokens: int,
        json_mode: bool,
    ) -> str:
        # Delimited with a byte that cannot appear in the fields, so that
        # ("ab", "c") and ("a", "bc") cannot collide.
        payload = "\x00".join(
            [
                str(CACHE_FORMAT_VERSION),
                model,
                system,
                user,
                f"{float(temperature):.4f}",
                str(int(max_output_tokens)),
                "1" if json_mode else "0",
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        # Shard on the first two hex chars: 256 subdirectories keeps any one
        # of them small enough that directory listing stays fast.
        return self.dir / key[:2] / f"{key}.json"

    # -- access -------------------------------------------------------------

    def get(self, key: str) -> CachedResponse | None:
        if not self.enabled:
            return None
        path = self._path(key)
        with self._lock:
            if not path.is_file():
                self.stats.misses += 1
                return None
            try:
                raw = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                # A damaged entry must degrade to a miss, never crash a run.
                log.warning("unreadable cache entry %s: %s", key[:12], exc)
                self.stats.errors += 1
                self.stats.misses += 1
                return None

            if raw.get("version") != CACHE_FORMAT_VERSION:
                self.stats.misses += 1
                return None

            self.stats.hits += 1
            self.stats.tokens_saved += int(raw.get("total_tokens", 0) or 0)
            return CachedResponse(**raw)

    def put(self, key: str, response: CachedResponse) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        with self._lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
                with os.fdopen(fd, "w") as fh:
                    json.dump(asdict(response), fh, indent=2)
                os.replace(tmp, path)   # atomic: no half-written entries
                self.stats.writes += 1
            except OSError as exc:
                # Failing to cache is not failing the call.
                log.warning("could not write cache entry %s: %s", key[:12], exc)
                self.stats.errors += 1

    # -- maintenance --------------------------------------------------------

    def count(self) -> int:
        if not self.dir.is_dir():
            return 0
        return sum(1 for _ in self.dir.glob("*/*.json"))

    def size_bytes(self) -> int:
        if not self.dir.is_dir():
            return 0
        return sum(p.stat().st_size for p in self.dir.glob("*/*.json"))

    def clear(self) -> int:
        """Delete every entry. Returns how many were removed.

        Rarely correct during a project: clearing invalidates the recorded
        observations behind any results already reported.
        """
        removed = 0
        with self._lock:
            for path in list(self.dir.glob("*/*.json")):
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    def report(self) -> str:
        return (
            f"{self.stats.render()} | {self.count()} entries, "
            f"{self.size_bytes() / 1_048_576:.1f} MB on disk"
        )