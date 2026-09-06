"""Configuration loading.

One rule (see config.yaml header): no magic numbers in src/. Everything
tunable comes from config.yaml, which is what makes the Phase 5 ablations
possible — run_ablations.py overrides a single key and re-runs.

Usage:
    from src.config import get_config
    cfg = get_config()
    k = cfg.get("retrieval.fusion.k")
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_MISSING = object()


class ConfigError(RuntimeError):
    """Raised when the configuration is absent, malformed, or incomplete."""


class Config:
    """Read-only view over config.yaml with dotted-path access.

    Deliberately not a dataclass tree. The config is large, evolves every
    phase, and is overridden wholesale by ablation runners; a dict with
    validated access at the point of use is less friction than keeping a
    parallel schema in sync.
    """

    def __init__(self, data: dict[str, Any], source: Path | None = None) -> None:
        self._data = data
        self.source = source

    # -- construction -------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")
        return cls(data, source=path)

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        """Return a copy with dotted-path overrides applied.

        This is how ablations work:
            cfg.with_overrides({"agency.d1_plan_revision": False})
        """
        import copy

        data = copy.deepcopy(self._data)
        for dotted, value in overrides.items():
            keys = dotted.split(".")
            node = data
            for key in keys[:-1]:
                node = node.setdefault(key, {})
                if not isinstance(node, dict):
                    raise ConfigError(f"cannot override {dotted}: {key} is not a mapping")
            node[keys[-1]] = value
        return Config(data, source=self.source)

    # -- access -------------------------------------------------------------

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self._data
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                if default is _MISSING:
                    raise ConfigError(f"missing config key: {dotted}")
                return default
            node = node[key]
        return node

    def section(self, dotted: str) -> dict[str, Any]:
        node = self.get(dotted)
        if not isinstance(node, dict):
            raise ConfigError(f"config key {dotted} is not a section")
        return node

    def path(self, dotted: str) -> Path:
        """Resolve a config value as a path relative to the project root."""
        return (PROJECT_ROOT / str(self.get(dotted))).resolve()

    def as_dict(self) -> dict[str, Any]:
        import copy

        return copy.deepcopy(self._data)

    def __contains__(self, dotted: str) -> bool:
        return self.get(dotted, None) is not None


_cached: Config | None = None


def get_config(path: str | Path | None = None, *, reload: bool = False) -> Config:
    """Process-wide config singleton. Pass reload=True in tests."""
    global _cached
    if _cached is None or reload or path is not None:
        _cached = Config.load(path)
    return _cached


def require_env(name: str) -> str:
    """Fetch a required environment variable, with a useful error.

    Loads .env on first call so scripts do not each need load_dotenv().
    """
    _load_dotenv_once()
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"environment variable {name} is not set. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


def has_env(name: str) -> bool:
    _load_dotenv_once()
    return bool(os.environ.get(name, "").strip())


_dotenv_loaded = False


def _load_dotenv_once() -> None:
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:  # pragma: no cover - dotenv is a hard dependency
        pass
    _dotenv_loaded = True