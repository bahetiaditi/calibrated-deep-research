"""Tests for the provider layer.

Uses fake backends throughout — no network, no API keys, no quota spent.
Tests that would need real credentials are marked and skipped.
"""
import pytest

from src.config import Config
from src.llm.provider import (
    LLMProvider,
    ProviderError,
    Role,
    classify,
    estimate_tokens,
)
from src.llm.rate_limit import QuotaExhausted, QuotaLedger

T0 = 1_757_000_000.0


# --- fakes -----------------------------------------------------------------


class FakeStatusError(Exception):
    def __init__(self, status_code, message="boom"):
        super().__init__(message)
        self.status_code = status_code


class FakeBackend:
    """Scripted backend. `script` is a list of str (success) or Exception."""

    def __init__(self, name="groq", script=None, default="{\"ok\": true}"):
        self.name = name
        self.script = list(script or [])
        self.default = default
        self.calls = []

    def complete(self, spec, system, user, temperature, max_output_tokens, json_mode):
        self.calls.append(
            {"model": spec.model, "system": system, "user": user,
             "json_mode": json_mode, "max_output_tokens": max_output_tokens}
        )
        item = self.script.pop(0) if self.script else self.default
        if isinstance(item, Exception):
            raise item
        return item, 100, 50


def make_config(tmp_path, **over):
    data = {
        "llm": {
            "default_temperature": 0.0,
            "quota_ledger_path": "data/quota_ledger.json",
            "retry": {
                "max_attempts": 3,
                "initial_backoff_s": 0.0,
                "backoff_multiplier": 1.0,
                "retry_on_status": [500, 502, 503, 504],
                "max_quota_wait_s": 90.0,
            },
            "roles": {
                "judgment": [
                    {"provider": "groq", "model": "primary", "max_output_tokens": 512,
                     "limits": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000}},
                    {"provider": "groq", "model": "secondary", "max_output_tokens": 512,
                     "limits": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000}},
                ],
                "mechanical": [
                    {"provider": "groq", "model": "small", "max_output_tokens": 256,
                     "limits": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000}},
                ],
                "judge": [
                    {"provider": "gemini", "model": "judge-model", "max_output_tokens": 256,
                     "limits": {"rpm": 5, "rpd": 20, "tpm": 250000}},
                ],
            },
        }
    }
    data["llm"].update(over)
    return Config(data)


def build(tmp_path, backend=None, **over):
    cfg = make_config(tmp_path, **over)
    backend = backend or FakeBackend()
    specs = {}
    for chain in cfg.get("llm.roles").values():
        for raw in chain:
            from src.llm.rate_limit import ModelLimits
            specs[raw["model"]] = ModelLimits.from_config(raw["limits"])
    ledger = QuotaLedger(tmp_path / "q.json", specs)
    slept = []
    provider = LLMProvider(
        cfg,
        backends={"groq": backend, "gemini": backend},
        ledger=ledger,
        sleep=slept.append,
        now=lambda: T0,
    )
    return provider, backend, ledger, slept


# --- helpers ---------------------------------------------------------------


def test_estimate_tokens_is_conservative():
    est = estimate_tokens("s" * 350, "u" * 350, 500)
    assert est > 500 + 175          # never under-counts the prompt


@pytest.mark.parametrize("status,expected", [
    (429, "switch"), (401, "fatal"), (403, "fatal"), (400, "fatal"),
    (500, "retry"), (503, "retry"),
])
def test_classify_status_codes(status, expected):
    assert classify(FakeStatusError(status), [500, 502, 503, 504]) == expected


def test_classify_timeout_is_retryable():
    class APITimeoutError(Exception):
        pass
    assert classify(APITimeoutError(), []) == "retry"


# --- routing ---------------------------------------------------------------


def test_role_selects_correct_chain(tmp_path):
    provider, backend, _, _ = build(tmp_path)
    provider.complete("sys", "usr", role=Role.MECHANICAL)
    assert backend.calls[0]["model"] == "small"


def test_role_uses_per_model_output_cap(tmp_path):
    """TPM is only 8K — per-role output caps are load-bearing, not cosmetic."""
    provider, backend, _, _ = build(tmp_path)
    provider.complete("sys", "usr", role=Role.MECHANICAL)
    assert backend.calls[0]["max_output_tokens"] == 256


def test_unconfigured_role_raises(tmp_path):
    cfg = make_config(tmp_path)
    cfg = cfg.with_overrides({"llm.roles.mechanical": []})
    provider = LLMProvider(cfg, backends={"groq": FakeBackend()},
                           ledger=QuotaLedger(tmp_path / "q.json", {}),
                           now=lambda: T0)
    with pytest.raises(ProviderError, match="no models configured"):
        provider.complete("s", "u", role=Role.MECHANICAL)


# --- retry and fallback ----------------------------------------------------


def test_retries_on_500_then_succeeds(tmp_path):
    backend = FakeBackend(script=[FakeStatusError(500), '{"ok": true}'])
    provider, backend, _, slept = build(tmp_path, backend=backend)
    resp = provider.complete("s", "u")
    assert resp.attempts == 2
    assert resp.model == "primary"
    assert not resp.fell_back
    assert len(slept) == 1          # backed off once


def test_429_switches_model_without_retrying_same_one(tmp_path):
    backend = FakeBackend(script=[FakeStatusError(429), '{"ok": true}'])
    provider, backend, _, slept = build(tmp_path, backend=backend)
    resp = provider.complete("s", "u")
    assert resp.model == "secondary"
    assert resp.fell_back
    assert slept == []              # a 429 is not waited out, it switches


def test_429_marks_model_spent_for_the_day(tmp_path):
    backend = FakeBackend(script=[FakeStatusError(429), '{"ok": 1}', '{"ok": 2}'])
    provider, backend, ledger, _ = build(tmp_path, backend=backend)
    provider.complete("s", "u")
    provider.complete("s", "u")     # second call must skip 'primary' entirely
    assert [c["model"] for c in backend.calls] == ["primary", "secondary", "secondary"]


def test_401_is_fatal_and_does_not_fall_through(tmp_path):
    backend = FakeBackend(script=[FakeStatusError(401)])
    provider, backend, _, _ = build(tmp_path, backend=backend)
    with pytest.raises(ProviderError, match="unrecoverable"):
        provider.complete("s", "u")
    assert len(backend.calls) == 1  # auth failure must not burn the chain


def test_exhausts_attempts_then_falls_through(tmp_path):
    backend = FakeBackend(
        script=[FakeStatusError(500)] * 3 + ['{"ok": true}']
    )
    provider, backend, _, _ = build(tmp_path, backend=backend)
    resp = provider.complete("s", "u")
    assert resp.model == "secondary" and resp.fell_back


def test_whole_chain_failing_raises_quota_exhausted(tmp_path):
    backend = FakeBackend(script=[FakeStatusError(500)] * 20)
    provider, backend, _, _ = build(tmp_path, backend=backend)
    with pytest.raises(QuotaExhausted, match="every model in the chain"):
        provider.complete("s", "u")


# --- quota integration -----------------------------------------------------


def test_skips_model_with_no_daily_tokens_left(tmp_path):
    provider, backend, ledger, _ = build(tmp_path)
    ledger.record("primary", 200_000, now=T0)     # primary is spent
    resp = provider.complete("s", "u")
    assert resp.model == "secondary" and resp.fell_back


def test_records_actual_not_estimated_usage(tmp_path):
    provider, backend, ledger, _ = build(tmp_path)
    provider.complete("s", "u")
    assert ledger.snapshot(now=T0)["primary"]["tpd"][0] == 150   # 100 + 50


def test_waits_out_a_minute_window_block(tmp_path):
    provider, backend, ledger, slept = build(tmp_path)
    ledger.record("primary", 7900, now=T0)        # TPM nearly gone
    provider.complete("s", "u")
    assert slept, "should have waited out the TPM window"


def test_does_not_wait_longer_than_max_quota_wait(tmp_path):
    provider, backend, ledger, slept = build(
        tmp_path, retry={"max_attempts": 3, "initial_backoff_s": 0.0,
                         "backoff_multiplier": 1.0,
                         "retry_on_status": [500], "max_quota_wait_s": 0.5}
    )
    ledger.record("primary", 7900, now=T0)
    resp = provider.complete("s", "u")
    assert resp.model == "secondary"              # switched instead of sleeping
    assert slept == []


# --- structured output -----------------------------------------------------


def test_complete_json_parses(tmp_path):
    backend = FakeBackend(script=['```json\n{"plan": ["a"]}\n```'])
    provider, backend, _, _ = build(tmp_path, backend=backend)
    parsed, resp = provider.complete_json("s", "u")
    assert parsed == {"plan": ["a"]}
    assert backend.calls[0]["json_mode"] is True


def test_complete_json_repairs_locally_without_extra_api_call(tmp_path):
    """Local repair must be free — a re-prompt costs quota we do not have."""
    backend = FakeBackend(script=['{"plan": ["a"],}'])
    provider, backend, _, _ = build(tmp_path, backend=backend)
    parsed, _ = provider.complete_json("s", "u")
    assert parsed == {"plan": ["a"]}
    assert len(backend.calls) == 1


def test_complete_json_reprompts_when_unrepairable(tmp_path):
    backend = FakeBackend(script=["not json at all", '{"plan": []}'])
    provider, backend, _, _ = build(tmp_path, backend=backend)
    parsed, _ = provider.complete_json("s", "u")
    assert parsed == {"plan": []}
    assert len(backend.calls) == 2
    assert "could not be parsed" in backend.calls[1]["user"]


def test_complete_json_enforces_required_keys(tmp_path):
    from src.llm.structured import StructuredOutputError
    backend = FakeBackend(default='{"wrong": 1}')
    provider, backend, _, _ = build(tmp_path, backend=backend)
    with pytest.raises(StructuredOutputError, match="missing required keys"):
        provider.complete_json("s", "u", required_keys=["plan"], repair_attempts=0)


def test_empty_response_falls_through(tmp_path):
    backend = FakeBackend(script=["   ", '{"ok": true}'])
    provider, backend, _, _ = build(tmp_path, backend=backend)
    resp = provider.complete("s", "u")
    assert resp.model == "secondary"


def test_quota_report_renders(tmp_path):
    provider, _, _, _ = build(tmp_path)
    provider.complete("s", "u")
    report = provider.quota_report()
    assert "primary" in report and "judge-model" in report


# --- model capability flags ------------------------------------------------


def test_system_prompt_folded_into_content_when_unsupported(tmp_path):
    """Gemma has no system role. The prompt must survive, folded into the
    user content — silently dropping it would degrade quality with no error."""
    from src.llm.provider import ModelSpec
    from src.llm.rate_limit import ModelLimits

    captured = {}

    class RecordingGemini:
        name = "gemini"

        def complete(self, spec, system, user, temperature, max_out, json_mode):
            # Mirrors GeminiBackend's folding logic.
            captured["contents"] = (
                user if spec.supports_system_instruction
                else f"{system}\n\n---\n\n{user}"
            )
            captured["sends_system_kwarg"] = spec.supports_system_instruction
            return '{"ok": true}', 10, 5

    spec = ModelSpec(
        provider="gemini", model="gemma-4-31b-it",
        limits=ModelLimits(rpm=30, rpd=14400, tpm=16000),
        supports_system_instruction=False,
    )
    RecordingGemini().complete(spec, "SYSTEM RULES", "the question", 0.0, 512, False)
    assert "SYSTEM RULES" in captured["contents"]
    assert captured["sends_system_kwarg"] is False


def test_thinking_budget_defaults_to_not_sent():
    """None means 'omit the key'. Sending it to Gemma is a hard 400."""
    from src.llm.provider import ModelSpec
    from src.llm.rate_limit import ModelLimits

    spec = ModelSpec(provider="gemini", model="gemma-4-31b-it",
                     limits=ModelLimits(rpm=30))
    assert spec.thinking_budget is None
    assert spec.supports_system_instruction is True   # opt-out, not default