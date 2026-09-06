"""C1 scaffold tests: the repo is wired correctly before any logic exists.

These are deliberately trivial. Their job is to fail loudly if someone
renames a directory, breaks config.yaml, or commits a .env file.
"""
from pathlib import Path
import yaml
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_directory_tree_intact():
    expected = [
        "src/nodes", "src/rag", "src/sufficiency", "src/tools",
        "src/llm", "src/tracing", "src/prompts",
        "benchmark", "analysis", "results", "traces", "demo", "tests",
    ]
    missing = [d for d in expected if not (ROOT / d).is_dir()]
    assert not missing, f"missing directories: {missing}"


def test_config_parses():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert isinstance(cfg, dict)


@pytest.mark.parametrize("path", [
    ("llm", "primary", "model"),
    ("llm", "cache", "enabled"),
    ("budget", "llm_calls_max"),
    ("retrieval", "dense", "query_prefix"),
    ("retrieval", "fusion", "k"),
    ("sufficiency", "thresholds", "tau_answer"),
    ("agency", "d1_plan_revision"),
])
def test_required_config_keys(path):
    node = yaml.safe_load((ROOT / "config.yaml").read_text())
    for key in path:
        assert key in node, f"config.yaml missing {'.'.join(path)}"
        node = node[key]


def test_bge_query_prefix_present():
    """bge-small REQUIRES this prefix. Losing it silently degrades recall,
    which is the worst kind of bug: no error, just worse numbers."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    prefix = cfg["retrieval"]["dense"]["query_prefix"]
    assert prefix.startswith("Represent this sentence")
    assert prefix.endswith(" ")


def test_all_agency_flags_default_on():
    """Ablations flip these off one at a time. Default must be all-on,
    or a stray commit silently turns the headline run into an ablation."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    for flag, val in cfg["agency"].items():
        if flag == "static":
            continue
        assert val is True, f"agency.{flag} is not True by default"


SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def test_env_example_has_no_real_secrets():
    """Non-secret flags (e.g. LANGSMITH_TRACING=false) may carry a default.
    Anything credential-shaped must be blank."""
    for line in (ROOT / ".env.example").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if any(m in key.upper() for m in SECRET_MARKERS):
            assert value == "", f"{key} has a value committed in .env.example"


def test_dotenv_is_gitignored():
    assert ".env" in (ROOT / ".gitignore").read_text().splitlines()