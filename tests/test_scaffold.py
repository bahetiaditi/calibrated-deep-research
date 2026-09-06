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
    ("llm", "roles", "judgment"),
    ("llm", "roles", "mechanical"),
    ("llm", "roles", "judge"),
    ("llm", "quota_ledger_path"),
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


def test_every_model_declares_its_limits():
    """The quota ledger cannot police a model whose limits it does not know.
    A model added without limits would silently bypass all accounting."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    for role, chain in cfg["llm"]["roles"].items():
        assert chain, f"role {role} has an empty model chain"
        for spec in chain:
            assert "provider" in spec and "model" in spec, f"{role}: incomplete spec"
            limits = spec.get("limits")
            assert limits, f"{role}/{spec['model']} declares no limits"
            assert "rpm" in limits and "rpd" in limits, (
                f"{role}/{spec['model']} missing rpm/rpd"
            )


def test_retired_groq_model_is_not_referenced():
    """llama-3.3-70b-versatile was retired from Groq's catalog and is not
    callable on this account. Checks configured model VALUES, not raw text —
    the config comment explaining the retirement is deliberate and stays."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    configured = {
        spec["model"]
        for chain in cfg["llm"]["roles"].values()
        for spec in chain
    }
    assert "llama-3.3-70b-versatile" not in configured


def test_judge_is_independent_of_the_system_under_test():
    """Scoring output with the same model that produced it invites
    self-preference bias. The judge must not share a provider with the
    judgment chain."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    judge_providers = {m["provider"] for m in cfg["llm"]["roles"]["judge"]}
    judgment_providers = {m["provider"] for m in cfg["llm"]["roles"]["judgment"]}
    assert not (judge_providers & judgment_providers)


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


# Models verified by scripts/probe_gemini.py to REJECT the thinking_budget
# parameter with 400 "Thinking budget is not supported".
NO_THINKING_BUDGET = ("gemma-",)


def test_thinking_budget_not_sent_to_models_that_reject_it():
    """Gemma returns 400 if thinking_budget is present at all. This is a
    config-only failure with no local symptom, so assert it here."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    for role, chain in cfg["llm"]["roles"].items():
        for spec in chain:
            if any(spec["model"].startswith(p) for p in NO_THINKING_BUDGET):
                assert "thinking_budget" not in spec, (
                    f"{role}/{spec['model']} must omit thinking_budget entirely"
                )


def test_gemma_declares_no_system_instruction_support():
    """Gemma has no separate system role on the Gemini API. Left at the
    default the system prompt would be silently dropped — a quality
    regression with no error."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    for role, chain in cfg["llm"]["roles"].items():
        for spec in chain:
            if spec["model"].startswith("gemma-"):
                assert spec.get("supports_system_instruction") is False, (
                    f"{role}/{spec['model']} must set "
                    f"supports_system_instruction: false"
                )