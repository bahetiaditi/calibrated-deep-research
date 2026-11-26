"""C28.5: the C12 contradiction detector, wired into the graph.

Until now nothing called it: f5 was None on every run, the critic's
contradiction flag never fired, and Tier D3 ("no consensus") had no signal.
Wiring it exposed two latent defects, both tested here: f5 divided by pairs
the detector never scores, and a failed scan read as a clean one.
"""
from src.config import Config
from src.rag.contradiction import ContradictionDetector
from src.state import new_state, new_sub_question
from src.sufficiency.features import features_for_sub_question
from tests.test_adaptive_retriever import FakeProvider, FakeRetriever, build
from tests.test_contradiction import CFG as NLI_CFG
from tests.test_contradiction import FakeNLI
from tests.test_contradiction import passage as cpassage

RANK_STRONG = 9.0


def sources(texts):
    """Passages from distinct sources, in rerank order."""
    out = []
    for i, text in enumerate(texts):
        p = cpassage(f"P{i:02d}", text, source=f"paper{i}")
        p["rerank_score"] = RANK_STRONG - i * 0.1
        out.append(p)
    return out


def detector(conflicts=(), error=None):
    return ContradictionDetector(NLI_CFG, model=FakeNLI(
        contradicting={frozenset(c) for c in conflicts}, error=error))


TEXTS = ["rank 8 is optimal", "rank 64 is optimal", "neutral passage two",
         "neutral passage three", "neutral passage four", "neutral passage five"]


# --- the detector reports what it did --------------------------------------


def test_scan_reports_pairs_scored_and_what_it_found():
    result = detector([("rank 8", "rank 64")]).scan(sources(TEXTS))
    assert result.pairs == 15                     # 6 sources -> 15 cross-source pairs
    assert len(result.records) == 1
    assert result.rate == 1 / 15


def test_a_failed_scan_is_failed_not_clean():
    """detect() returns [] on failure; scan() must say it failed."""
    result = detector(error=RuntimeError("model unavailable")).scan(sources(TEXTS))
    assert result.failed and result.rate is None
    assert detector(error=RuntimeError("x")).detect(sources(TEXTS)) == []


# --- f5 divides by what was scored ----------------------------------------


def scanned_state(scan, n_passages=20):
    state = new_state("q")
    sq = new_sub_question("what rank?", sub_question_id="sq1")
    sq["evidence_ids"] = [f"P{i:02d}" for i in range(n_passages)]
    sq["contradiction_scan"] = scan
    state["plan"] = [sq]
    state["evidence"] = sources([f"t{i}" for i in range(n_passages)])
    return state


def test_f5_is_found_over_pairs_scored():
    """3 contradictions in 15 scored pairs is 0.2. The old denominator was
    every pair among 20 passages (190), which would give 0.016."""
    state = scanned_state({"pairs": 15, "found": 3, "passages": []})
    assert features_for_sub_question(state, "sq1")["f5_contradiction_rate"] == 0.2


def test_unscanned_is_none_not_zero():
    assert features_for_sub_question(scanned_state(None), "sq1")["f5_contradiction_rate"] is None


def test_scanned_with_no_pairs_is_zero():
    state = scanned_state({"pairs": 0, "found": 0, "passages": []})
    assert features_for_sub_question(state, "sq1")["f5_contradiction_rate"] == 0.0


# --- the retriever scans, once, at a terminal status -----------------------


def retrieve(nli_detector, passages):
    retriever = build(retriever=FakeRetriever(default=passages),
                      provider=FakeProvider([{"action": "give_up", "reasoning": "r"}]),
                      detector=nli_detector)
    state = new_state("What is the optimal LoRA rank?")
    state["plan"] = [new_sub_question("optimal LoRA rank", sub_question_id="sq1")]
    state["plan"][0]["suggested_route"] = "arxiv_meta"
    return retriever.retrieve_all(state)


def test_the_retriever_scans_and_records_contradictions():
    """A Tier D3 shape: two sources disagree on the optimal rank."""
    final = retrieve(detector([("rank 8", "rank 64")]), sources(TEXTS))
    sq = final["plan"][0]
    scan = sq["contradiction_scan"]
    n = len(scan["passages"])                 # all distinct sources here
    assert scan["pairs"] == n * (n - 1) // 2
    assert scan["found"] == 1
    assert {"P00", "P01"} <= set(scan["passages"])
    assert len(final["contradictions"]) == 1
    assert features_for_sub_question(final, "sq1")["f5_contradiction_rate"] == 1 / scan["pairs"]


def test_a_failed_scan_leaves_the_sub_question_unscanned():
    final = retrieve(detector(error=RuntimeError("no model")), sources(TEXTS))
    assert final["plan"][0]["contradiction_scan"] is None
    assert features_for_sub_question(final, "sq1")["f5_contradiction_rate"] is None
    assert final.get("contradictions") == []


def test_no_detector_means_no_scan():
    final = retrieve(None, sources(TEXTS))
    assert final["plan"][0]["contradiction_scan"] is None


def test_a_scan_happens_only_at_a_terminal_status():
    """An in-progress round is not scanned: the scan is about the evidence
    the sub-question ENDS with."""
    nli = detector()
    retriever = build(retriever=FakeRetriever(default=sources(TEXTS)[:1]),
                      provider=FakeProvider([{"action": "wider", "route": "web",
                                              "query": "again", "reasoning": "r"}]),
                      detector=nli)
    state = new_state("q")
    state["plan"] = [new_sub_question("hard", sub_question_id="sq1")]
    state["plan"][0]["suggested_route"] = "arxiv_meta"
    update = retriever.retrieve_one(state, "sq1")
    assert update["plan"][0]["status"] == "in_progress"
    assert nli._model.calls == []


def test_the_critic_now_sees_the_contradiction():
    """The flag C23 built and nothing ever set."""
    from src.nodes.critic import contradicted_passages

    final = retrieve(detector([("rank 8", "rank 64")]), sources(TEXTS))
    assert contradicted_passages(final["contradictions"]) == {"P00", "P01"}


def test_the_default_graph_wires_a_detector():
    from src.config import get_config
    from src.graph import default_components

    comps = default_components(get_config(reload=True))
    retriever = comps.retriever.__closure__
    objects = [c.cell_contents for c in retriever if hasattr(c.cell_contents, "_detector")]
    assert objects and objects[0]._detector is not None
