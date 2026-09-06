"""Tests for the config loader."""
import pytest

from src.config import Config, ConfigError, get_config


def test_dotted_get():
    cfg = Config({"a": {"b": {"c": 1}}})
    assert cfg.get("a.b.c") == 1


def test_missing_key_raises():
    with pytest.raises(ConfigError, match="missing config key"):
        Config({}).get("a.b")


def test_missing_key_with_default():
    assert Config({}).get("a.b", 7) == 7


def test_overrides_do_not_mutate_original():
    """Ablation runners depend on this: A1 must not leak into A2."""
    base = Config({"agency": {"d1": True, "d2": True}})
    ablated = base.with_overrides({"agency.d1": False})
    assert ablated.get("agency.d1") is False
    assert base.get("agency.d1") is True


def test_override_creates_missing_path():
    cfg = Config({}).with_overrides({"x.y.z": 3})
    assert cfg.get("x.y.z") == 3


def test_section_rejects_scalar():
    with pytest.raises(ConfigError, match="not a section"):
        Config({"a": 1}).section("a")


def test_real_config_loads_and_has_agency_flags():
    cfg = get_config(reload=True)
    for flag in ("d1_plan_revision", "d2_adaptive_route_and_query",
                 "d3_adaptive_depth_and_stopping", "d4_budget_allocation",
                 "d5_terminal_decision"):
        assert cfg.get(f"agency.{flag}") is True