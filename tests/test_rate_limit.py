"""Tests for quota accounting.

All tests inject an explicit `now` so the sliding windows are deterministic
and no test sleeps.
"""
import json

import pytest

from src.llm.rate_limit import ModelLimits, QuotaLedger

GROQ = ModelLimits(rpm=30, rpd=1000, tpm=8000, tpd=200_000, reset_timezone="UTC")
TINY = ModelLimits(rpm=2, rpd=3, tpm=1000, tpd=5000, reset_timezone="UTC")

T0 = 1_757_000_000.0  # fixed epoch, mid-day UTC


@pytest.fixture
def ledger(tmp_path):
    return QuotaLedger(tmp_path / "quota.json", {"m": TINY, "big": GROQ})


def test_allows_when_fresh(ledger):
    assert ledger.check("m", 100, now=T0).allowed


def test_blocks_on_rpm_and_is_recoverable(ledger):
    ledger.record("m", 10, now=T0)
    ledger.record("m", 10, now=T0 + 1)
    d = ledger.check("m", 10, now=T0 + 2)
    assert not d.allowed
    assert d.dimension == "rpm"
    assert d.recoverable_today          # waiting fixes it
    assert 0 < d.wait_seconds <= 60


def test_rpm_window_slides(ledger):
    ledger.record("m", 10, now=T0)
    ledger.record("m", 10, now=T0 + 1)
    assert ledger.check("m", 10, now=T0 + 61).allowed


def test_blocks_on_tpm(ledger):
    ledger.record("m", 900, now=T0)
    d = ledger.check("m", 200, now=T0 + 1)
    assert not d.allowed and d.dimension == "tpm" and d.recoverable_today


def test_call_larger_than_tpm_ceiling_is_not_recoverable(ledger):
    """Waiting can never help — surface it instead of looping forever."""
    d = ledger.check("m", 5000, now=T0)
    assert not d.allowed
    assert d.dimension == "tpm"
    assert not d.recoverable_today
    assert "reduce prompt size" in d.detail


def test_blocks_on_rpd_not_recoverable_today(ledger):
    for i in range(3):
        ledger.record("m", 1, now=T0 + i * 61)
    d = ledger.check("m", 1, now=T0 + 300)
    assert not d.allowed
    assert d.dimension == "rpd"
    assert not d.recoverable_today      # must switch models, not wait


def test_blocks_on_tpd(ledger):
    ledger.record("m", 4900, now=T0)
    d = ledger.check("m", 200, now=T0 + 120)
    assert not d.allowed and d.dimension == "tpd" and not d.recoverable_today


def test_tpd_binds_before_rpd_on_real_limits(ledger):
    """The headline finding: at ~3k tokens/call, 200K TPD is ~66 calls,
    long before the 1000 RPD ceiling."""
    calls = 0
    now = T0
    while ledger.check("big", 3000, now=now).allowed:
        ledger.record("big", 3000, now=now)
        calls += 1
        now += 61          # step past the minute window each time
    assert 60 <= calls <= 70
    assert ledger.snapshot(now=now)["big"]["rpd"][0] < 1000


def test_daily_counters_reset_next_day(ledger):
    for i in range(3):
        ledger.record("m", 1, now=T0 + i * 61)
    assert not ledger.check("m", 1, now=T0 + 300).allowed
    assert ledger.check("m", 1, now=T0 + 86_400 + 300).allowed


def test_state_survives_restart(tmp_path):
    path = tmp_path / "quota.json"
    a = QuotaLedger(path, {"m": TINY})
    a.record("m", 4000, now=T0)

    b = QuotaLedger(path, {"m": TINY})       # simulates a resumed eval run
    d = b.check("m", 2000, now=T0 + 120)
    assert not d.allowed and d.dimension == "tpd"


def test_corrupt_ledger_does_not_crash(tmp_path):
    path = tmp_path / "quota.json"
    path.write_text("{ not json")
    led = QuotaLedger(path, {"m": TINY})
    assert led.check("m", 1, now=T0).allowed


def test_save_is_atomic_and_valid_json(tmp_path):
    path = tmp_path / "quota.json"
    led = QuotaLedger(path, {"m": TINY})
    led.record("m", 42, now=T0)
    data = json.loads(path.read_text())
    assert data["models"]["m"]["day_tokens"] == 42


def test_remaining_tokens_today(ledger):
    ledger.record("m", 1500, now=T0)
    assert ledger.remaining_tokens_today("m", now=T0 + 1) == 3500


def test_unknown_model_raises(ledger):
    with pytest.raises(KeyError):
        ledger.check("nope", 1, now=T0)


def test_snapshot_shape(ledger):
    ledger.record("m", 10, now=T0)
    snap = ledger.snapshot(now=T0 + 1)["m"]
    assert snap["rpm"] == [1, 2] and snap["tpd"] == [10, 5000]