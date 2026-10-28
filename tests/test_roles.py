"""C27.6: every LLM call names its role, and an eval can pin its writer.

The adaptive retriever's reformulation passed no role and inherited
complete_json's default, JUDGMENT — spending the scarce Groq chain on work
C2 allocated to MECHANICAL. Nothing failed; the cost showed up only as a
token count. The guard below makes the default unreachable from a node.
"""
import ast
from pathlib import Path

import pytest

from src.llm.provider import ProviderError, Role
from tests.test_provider import FakeBackend, FakeStatusError, build

NODES = Path(__file__).resolve().parents[1] / "src" / "nodes"


def _llm_calls_without_role(path: Path) -> list[int]:
    missing = []
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("complete", "complete_json")
                and not any(k.arg == "role" for k in node.keywords)):
            missing.append(node.lineno)
    return missing


@pytest.mark.parametrize("path", sorted(NODES.glob("*.py")), ids=lambda p: p.name)
def test_every_llm_call_in_a_node_names_its_role(path):
    """A default is a decision nobody made. Here it cost tokens; on another
    call it could put the judge on the writer's model family."""
    assert _llm_calls_without_role(path) == [], (
        f"{path.name}: LLM call(s) without role= at line(s) "
        f"{_llm_calls_without_role(path)}"
    )


def test_the_guard_would_have_caught_the_retriever():
    source = "self.provider.complete_json(system='s', user='u')\n"
    tmp = NODES.parent.parent / "data" / "_guard_probe.py"
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_text(source)
    try:
        assert _llm_calls_without_role(tmp) == [1]
    finally:
        tmp.unlink()


def test_reformulation_runs_on_the_mechanical_role():
    from tests.test_adaptive_retriever import build as build_retriever
    from tests.test_adaptive_retriever import outcome
    from src.state import new_sub_question

    roles = []

    class Provider:
        def complete_json(self, system, user, *, role=None, **kw):
            roles.append(role)
            return {"action": "wider", "route": "web", "query": "q2",
                    "reasoning": "r"}, None

    retriever = build_retriever(provider=Provider())
    retriever.decide_next(new_sub_question("hard", sub_question_id="sq1"),
                          outcome([1.0], ["A"]))
    assert roles == [Role.MECHANICAL]


# --- pinning ---------------------------------------------------------------


def test_pin_reduces_the_judgment_chain_to_one_model(tmp_path):
    provider, _, _, _ = build(tmp_path, pin={"judgment": "secondary"})
    assert [s.model for s in provider.chains[Role.JUDGMENT]] == ["secondary"]
    # Other roles untouched.
    assert [s.model for s in provider.chains[Role.MECHANICAL]] == ["small"]


def test_no_pin_keeps_the_chain(tmp_path):
    provider, _, _, _ = build(tmp_path, pin={"judgment": None})
    assert [s.model for s in provider.chains[Role.JUDGMENT]] == ["primary", "secondary"]


def test_an_unknown_pin_fails_at_construction(tmp_path):
    with pytest.raises(ProviderError, match="not in the judgment"):
        build(tmp_path, pin={"judgment": "gpt-imaginary"})


def test_a_pinned_model_out_of_day_fails_rather_than_falling_back(tmp_path):
    """The point of pinning: no report in the eval is written by another
    model because the pinned one ran out."""
    day = FakeStatusError(429, "Rate limit reached on tokens per day (TPD)")
    backend = FakeBackend(script=[day, '{"ok": true}'])
    provider, backend, _, _ = build(tmp_path, backend=backend,
                                    pin={"judgment": "primary"})
    with pytest.raises(Exception):
        provider.complete("s", "u", role=Role.JUDGMENT)
    assert [c["model"] for c in backend.calls] == ["primary"]


def test_a_pinned_model_still_waits_out_a_minute_window(tmp_path):
    minute = FakeStatusError(429, "tokens per minute (TPM) ... try again in 2s")
    backend = FakeBackend(script=[minute, '{"ok": true}'])
    provider, backend, _, slept = build(tmp_path, backend=backend,
                                        pin={"judgment": "primary"})
    provider._sleep = slept.append
    assert provider.complete("s", "u", role=Role.JUDGMENT).model == "primary"
    assert slept


def test_the_real_config_ships_unpinned():
    """Pinning is an eval decision, made deliberately at C32/C34."""
    from src.config import get_config

    cfg = get_config(reload=True)
    assert cfg.get("llm.pin.judgment") is None
    assert cfg.get("llm.pin.mechanical") is None


def test_mechanical_can_be_pinned_independently(tmp_path):
    provider, _, _, _ = build(tmp_path, pin={"mechanical": "small"})
    assert [s.model for s in provider.chains[Role.MECHANICAL]] == ["small"]
    assert [s.model for s in provider.chains[Role.JUDGMENT]] == ["primary", "secondary"]


def test_mechanical_leads_with_a_reliable_non_judge_model():
    """C27.7: gemma returned 500 on 2 of 3 probes; flash-lite is the judge.
    Reformulation must not depend on either as its first choice."""
    from src.config import get_config

    cfg = get_config(reload=True)
    mechanical = [s["model"] for s in cfg.get("llm.roles.mechanical")]
    judge = [s["model"] for s in cfg.get("llm.roles.judge")]
    assert mechanical[0] == "openai/gpt-oss-20b"
    assert mechanical[0] not in judge
