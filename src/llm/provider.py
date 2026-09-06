"""The LLM provider layer.

Design, and why it is not "primary + fallback"
----------------------------------------------
The original plan was Gemini primary, Groq fallback on 429. Measured free-tier
limits killed that: Gemini 2.5 Flash allows 20 requests/day on this account,
which is half of one benchmark question. So the layer is built around three
ideas instead:

1. ROLES, not a single model. Calls differ in how much judgment they need.
   Claim segmentation and query reformulation are mechanical; planning,
   criticism, and the terminal decision are not. Routing mechanical work to a
   smaller model preserves the scarce token budget on the big one. This was
   always an interview talking point ("swap models per role"); the free tier
   just made it mandatory.

2. CHAINS, not a pair. Each role has an ordered list of models. Exhausting one
   model's daily tokens falls through to the next rather than failing the run.

3. PREDICT, don't discover. Quota is checked locally before every call
   (see rate_limit.py). Learning your limits by collecting 429s spends the
   quota you are trying to conserve.

Gemini is reserved for the evaluation judge. That is not a consolation prize:
scoring a system's output with the same model that produced it invites
self-preference bias, so an independent judge is better methodology than the
original design had. 20 RPD is ample for a judge run once at the end.

Reasoning tokens
----------------
The gpt-oss models are reasoning models and Gemini 2.5 Flash thinks by
default. Those hidden tokens are billed against the same TPD ceiling as
visible output, so both are controlled explicitly per role rather than left
at provider defaults.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

from src.config import Config, get_config, has_env, require_env
from src.llm.cache import CachedResponse, PromptCache
from src.llm.rate_limit import (
    ModelLimits,
    QuotaExhausted,
    QuotaLedger,
)
from src.llm.structured import extract_json

log = logging.getLogger(__name__)

# Upper bound on how many times we will sleep waiting for a minute-window to
# clear before giving up on a model. Prevents an unbounded hang.
_MAX_QUOTA_WAITS = 4


class Role(str, Enum):
    """What kind of thinking a call needs. Maps to a model chain in config."""

    JUDGMENT = "judgment"      # planner, critic, decider — quality matters
    MECHANICAL = "mechanical"  # segmentation, reformulation, extraction
    JUDGE = "judge"            # evaluation only, must be provider-independent


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    role: Role
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    attempts: int
    fell_back: bool
    from_cache: bool = False

    def json(self) -> Any:
        return extract_json(self.text)


@dataclass
class ModelSpec:
    provider: str
    model: str
    limits: ModelLimits
    max_output_tokens: int = 2048
    reasoning_effort: str | None = None   # groq gpt-oss: none|low|medium|high
    thinking_budget: int | None = None    # gemini: 0 disables thinking.
    #   Omit entirely for models that reject it — Gemma returns 400
    #   "Thinking budget is not supported". None means "do not send".
    supports_json_mode: bool = True
    #   Gemma on the Gemini API does not accept a separate system role.
    #   When False the system prompt is prepended to the user content.
    supports_system_instruction: bool = True

    @classmethod
    def from_config(cls, raw: dict) -> "ModelSpec":
        return cls(
            provider=raw["provider"],
            model=raw["model"],
            limits=ModelLimits.from_config(raw.get("limits", {})),
            max_output_tokens=raw.get("max_output_tokens", 2048),
            reasoning_effort=raw.get("reasoning_effort"),
            thinking_budget=raw.get("thinking_budget"),
            supports_json_mode=raw.get("supports_json_mode", True),
            supports_system_instruction=raw.get("supports_system_instruction", True),
        )


class ProviderError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class Backend(Protocol):
    name: str

    def complete(
        self,
        spec: ModelSpec,
        system: str,
        user: str,
        temperature: float,
        max_output_tokens: int,
        json_mode: bool,
    ) -> tuple[str, int, int]:
        """Return (text, prompt_tokens, completion_tokens)."""
        ...


class GroqBackend:
    name = "groq"

    def __init__(self, api_key: str | None = None) -> None:
        from groq import Groq

        # max_retries=0: retry/backoff is our concern, not the SDK's, because
        # a silent SDK retry would spend quota the ledger never sees.
        self._client = Groq(api_key=api_key or require_env("GROQ_API_KEY"), max_retries=0)

    def complete(self, spec, system, user, temperature, max_output_tokens, json_mode):
        kwargs: dict[str, Any] = {
            "model": spec.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_completion_tokens": max_output_tokens,
        }
        if spec.reasoning_effort:
            kwargs["reasoning_effort"] = spec.reasoning_effort
            # Keep chain-of-thought out of the content field so downstream
            # JSON parsing sees only the answer.
            kwargs["reasoning_format"] = "hidden"
        if json_mode and spec.supports_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return text, int(usage.prompt_tokens), int(usage.completion_tokens)


class GeminiBackend:
    name = "gemini"

    def __init__(self, api_key: str | None = None) -> None:
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=api_key or require_env("GOOGLE_API_KEY"))

    def complete(self, spec, system, user, temperature, max_output_tokens, json_mode):
        from google.genai import types

        cfg: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            # We never pass tools; leaving AFC on emits a warning on every
            # call and would silently make extra remote calls that the quota
            # ledger cannot see.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }

        contents = user
        if spec.supports_system_instruction:
            cfg["system_instruction"] = system
        else:
            # Gemma rejects a separate system role — fold it into the prompt.
            contents = f"{system}\n\n---\n\n{user}"

        if json_mode and spec.supports_json_mode:
            cfg["response_mime_type"] = "application/json"

        # Sent only when explicitly configured. Gemma returns 400 on this key
        # ("Thinking budget is not supported"), and it cannot be disabled
        # there anyway — Gemma always thinks, which is why its calls carry a
        # large hidden-token overhead.
        if spec.thinking_budget is not None:
            cfg["thinking_config"] = types.ThinkingConfig(
                thinking_budget=spec.thinking_budget
            )

        resp = self._client.models.generate_content(
            model=spec.model,
            contents=contents,
            config=types.GenerateContentConfig(**cfg),
        )
        usage = resp.usage_metadata
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        completion = int(getattr(usage, "candidates_token_count", 0) or 0)
        # Thinking tokens bill against the same ceiling — count them or the
        # ledger under-reports and we hit a surprise 429.
        completion += int(getattr(usage, "thoughts_token_count", 0) or 0)
        return resp.text or "", prompt_tokens, completion


# ---------------------------------------------------------------------------
# Retry classification
# ---------------------------------------------------------------------------


def _status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None and isinstance(getattr(response, "status_code", None), int):
        return response.status_code
    return None


def classify(exc: Exception, retry_statuses: list[int]) -> str:
    """-> 'retry' (same model), 'switch' (next model), or 'fatal'."""
    status = _status_code(exc)
    if status == 429:
        # Our ledger thought there was room and the provider disagreed.
        # Trust the provider: stop using this model today.
        return "switch"
    if status in (401, 403):
        return "fatal"
    if status in retry_statuses:
        return "retry"
    name = type(exc).__name__
    if "Timeout" in name or "Connection" in name:
        return "retry"
    if status is not None and 400 <= status < 500:
        return "fatal"
    return "retry"


def estimate_tokens(system: str, user: str, max_output_tokens: int) -> int:
    """Conservative pre-call estimate: ~4 chars/token, plus full output budget.

    Over-estimating costs a little unused headroom. Under-estimating costs a
    429 and the tokens already spent, so we round up deliberately.
    """
    prompt_chars = len(system) + len(user)
    return int(prompt_chars / 3.5) + max_output_tokens


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class LLMProvider:
    def __init__(
        self,
        config: Config | None = None,
        *,
        backends: dict[str, Backend] | None = None,
        ledger: QuotaLedger | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
        cache: PromptCache | None = None,
    ) -> None:
        self.cfg = config or get_config()
        self._sleep = sleep
        self._now = now
        self.cache = cache if cache is not None else PromptCache(
            self.cfg.path("llm.cache.dir"),
            enabled=bool(self.cfg.get("llm.cache.enabled", True)),
        )

        self.chains: dict[Role, list[ModelSpec]] = {}
        all_specs: dict[str, ModelSpec] = {}
        for role in Role:
            specs = [
                ModelSpec.from_config(raw)
                for raw in self.cfg.get(f"llm.roles.{role.value}", [])
            ]
            self.chains[role] = specs
            for spec in specs:
                all_specs[spec.model] = spec
        if not any(self.chains.values()):
            raise ProviderError("no model chains configured under llm.roles")

        self.ledger = ledger or QuotaLedger(
            self.cfg.path("llm.quota_ledger_path"),
            {m: s.limits for m, s in all_specs.items()},
        )

        self._backends: dict[str, Backend] = backends if backends is not None else {}
        self._lazy_backends = backends is None

        retry = self.cfg.section("llm.retry")
        self.max_attempts = int(retry.get("max_attempts", 3))
        self.initial_backoff = float(retry.get("initial_backoff_s", 2.0))
        self.backoff_multiplier = float(retry.get("backoff_multiplier", 2.0))
        self.retry_statuses = list(retry.get("retry_on_status", [500, 502, 503, 504]))
        self.max_quota_wait_s = float(
            self.cfg.get("llm.retry.max_quota_wait_s", 90.0)
        )

    # -- backends -----------------------------------------------------------

    def _backend(self, name: str) -> Backend:
        if name not in self._backends:
            if not self._lazy_backends:
                raise ProviderError(f"backend {name!r} not injected")
            if name == "groq":
                self._backends[name] = GroqBackend()
            elif name == "gemini":
                self._backends[name] = GeminiBackend()
            else:
                raise ProviderError(f"unknown provider {name!r}")
        return self._backends[name]

    # -- main entry point ---------------------------------------------------

    def complete(
        self,
        system: str,
        user: str,
        *,
        role: Role = Role.JUDGMENT,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        json_mode: bool = False,
        use_cache: bool = True,
    ) -> LLMResponse:
        chain = self.chains.get(role) or []
        if not chain:
            raise ProviderError(f"no models configured for role {role.value!r}")

        temperature = (
            self.cfg.get("llm.default_temperature", 0.0)
            if temperature is None
            else temperature
        )

        errors: list[str] = []
        started = self._now()

        # Cache is checked across the ENTIRE chain before any live call.
        #
        # The naive alternative — check each model's cache just before calling
        # it — breaks re-runs. If the first run fell back to the secondary
        # model, a re-run would miss on the primary, spend real quota there,
        # and never reach the cached secondary entry. Checking the whole chain
        # first makes "re-running an unchanged question costs zero" true
        # regardless of which model answered originally.
        if use_cache and self.cache.enabled:
            for index, spec in enumerate(chain):
                budget = max_output_tokens or spec.max_output_tokens
                key = PromptCache.make_key(
                    model=spec.model,
                    system=system,
                    user=user,
                    temperature=temperature,
                    max_output_tokens=budget,
                    json_mode=json_mode and spec.supports_json_mode,
                )
                hit = self.cache.get(key)
                if hit is not None:
                    return LLMResponse(
                        text=hit.text,
                        model=hit.model,
                        provider=hit.provider,
                        role=role,
                        prompt_tokens=hit.prompt_tokens,
                        completion_tokens=hit.completion_tokens,
                        total_tokens=hit.total_tokens,
                        latency_s=self._now() - started,
                        attempts=0,
                        fell_back=index > 0,
                        from_cache=True,
                    )

        for index, spec in enumerate(chain):
            budget = max_output_tokens or spec.max_output_tokens
            est = estimate_tokens(system, user, budget)

            if not self._await_quota(spec, est, errors):
                continue

            backoff = self.initial_backoff
            for attempt in range(1, self.max_attempts + 1):
                try:
                    text, prompt_tok, completion_tok = self._backend(spec.provider).complete(
                        spec, system, user, temperature, budget, json_mode
                    )
                except Exception as exc:  # noqa: BLE001 - classified below
                    action = classify(exc, self.retry_statuses)
                    errors.append(f"{spec.model}: {type(exc).__name__}: {exc}")
                    if action == "fatal":
                        raise ProviderError(
                            f"unrecoverable error from {spec.model}: {exc}"
                        ) from exc
                    if action == "switch":
                        # Provider says the quota is gone. Burn the local
                        # day budget so we stop choosing this model.
                        self._exhaust_locally(spec)
                        log.warning("429 on %s — switching model", spec.model)
                        break
                    if attempt == self.max_attempts:
                        log.warning("%s failed %d attempts", spec.model, attempt)
                        break
                    log.info(
                        "%s attempt %d failed (%s); retrying in %.1fs",
                        spec.model, attempt, type(exc).__name__, backoff,
                    )
                    self._sleep(backoff)
                    backoff *= self.backoff_multiplier
                    continue

                total = prompt_tok + completion_tok
                # Quota is charged only for calls that actually happened —
                # a cache hit returns above and never reaches this line.
                self.ledger.record(spec.model, total, now=self._now())
                if not text.strip():
                    # Never cache an empty response: it would poison every
                    # future run of this prompt with a permanent failure.
                    errors.append(f"{spec.model}: empty response")
                    break
                if use_cache:
                    self.cache.put(
                        PromptCache.make_key(
                            model=spec.model,
                            system=system,
                            user=user,
                            temperature=temperature,
                            max_output_tokens=budget,
                            json_mode=json_mode and spec.supports_json_mode,
                        ),
                        CachedResponse(
                            text=text,
                            model=spec.model,
                            provider=spec.provider,
                            prompt_tokens=prompt_tok,
                            completion_tokens=completion_tok,
                            total_tokens=total,
                            created_at=datetime.now().isoformat(timespec="seconds"),
                        ),
                    )
                return LLMResponse(
                    text=text,
                    model=spec.model,
                    provider=spec.provider,
                    role=role,
                    prompt_tokens=prompt_tok,
                    completion_tokens=completion_tok,
                    total_tokens=total,
                    latency_s=self._now() - started,
                    attempts=attempt,
                    fell_back=index > 0,
                )

        raise QuotaExhausted(
            f"role {role.value!r}: every model in the chain failed or is out of "
            f"quota. Tried {[s.model for s in chain]}. Details: {errors}"
        )

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        role: Role = Role.JUDGMENT,
        required_keys: list[str] | None = None,
        repair_attempts: int = 1,
        **kwargs: Any,
    ) -> tuple[Any, LLMResponse]:
        """Structured call. Repairs locally first, re-prompts only if needed.

        A re-prompt costs real quota, so local repair (fences, trailing
        commas, truncation) is always tried first.
        """
        from src.llm.structured import StructuredOutputError, require_keys

        response = self.complete(system, user, role=role, json_mode=True, **kwargs)
        last_error: Exception | None = None

        for correction in range(repair_attempts + 1):
            try:
                parsed = extract_json(response.text)
                if required_keys:
                    require_keys(parsed, required_keys, context=f"role={role.value}")
                return parsed, response
            except StructuredOutputError as exc:
                last_error = exc
                if correction >= repair_attempts:
                    break
                log.info("structured output invalid (%s); re-prompting once", exc)
                response = self.complete(
                    system,
                    (
                        f"{user}\n\n---\nYour previous reply could not be parsed as "
                        f"JSON ({exc}). Reply with valid JSON only — no prose, no "
                        f"markdown fences."
                    ),
                    role=role,
                    json_mode=True,
                    **kwargs,
                )

        raise StructuredOutputError(
            f"structured output failed after {repair_attempts + 1} attempts: {last_error}",
            response.text,
        )

    # -- quota helpers ------------------------------------------------------

    def _await_quota(self, spec: ModelSpec, est: int, errors: list[str]) -> bool:
        """Wait out short minute-window blocks; skip the model on day blocks.

        The loop is bounded. Without a cap, a stalled clock or a wait that
        does not actually clear the window spins forever and the run hangs
        with no error — the worst possible failure for an overnight eval.
        Falling through to the next model is always the safer outcome.
        """
        for _ in range(_MAX_QUOTA_WAITS):
            decision = self.ledger.check(spec.model, est, now=self._now())
            if decision.allowed:
                return True
            if not decision.recoverable_today:
                errors.append(f"{spec.model}: {decision.detail}")
                log.info("skipping %s — %s", spec.model, decision.detail)
                return False
            if decision.wait_seconds > self.max_quota_wait_s:
                errors.append(
                    f"{spec.model}: would wait {decision.wait_seconds:.0f}s "
                    f"({decision.detail})"
                )
                return False
            log.info(
                "%s rate-limited on %s; waiting %.1fs",
                spec.model, decision.dimension, decision.wait_seconds,
            )
            self._sleep(decision.wait_seconds + 0.5)

        errors.append(
            f"{spec.model}: quota did not clear after {_MAX_QUOTA_WAITS} waits"
        )
        return False

    def _exhaust_locally(self, spec: ModelSpec) -> None:
        """Mark a model as spent for today after an unexpected 429."""
        remaining = self.ledger.remaining_tokens_today(spec.model, now=self._now())
        if remaining:
            self.ledger.record(spec.model, remaining, now=self._now())
        elif spec.limits.rpd is not None:
            self.ledger.record(spec.model, 0, now=self._now())

    def quota_report(self) -> str:
        lines = ["model                          rpm        tpm            rpd       tpd"]
        for model, s in self.ledger.snapshot(now=self._now()).items():
            def fmt(pair):
                used, cap = pair
                return f"{used}/{cap if cap is not None else '-'}"
            lines.append(
                f"{model:30s} {fmt(s['rpm']):10s} {fmt(s['tpm']):14s} "
                f"{fmt(s['rpd']):9s} {fmt(s['tpd'])}"
            )
        return "\n".join(lines)


def available_providers() -> dict[str, bool]:
    return {
        "groq": has_env("GROQ_API_KEY"),
        "gemini": has_env("GOOGLE_API_KEY"),
    }