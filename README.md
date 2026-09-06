# Calibrated Deep Research

**A multi-agent research system whose terminal decision is: answer, partially answer, or abstain.**

> **Status: in development.** The thesis below was written before the implementation, deliberately. No results are reported yet. Every number in this README will be filled in from a held-out test split that has not been touched at the time of writing. Sections marked *(pending)* contain no claims.

---

## The problem

Deep research agents — decompose a question, search, read, synthesize a cited report — are a solved architectural pattern and a crowded benchmark space. DeepResearch Bench, ResearcherBench, ReportBench, DeepScholar-Bench, LiveDRBench, FutureSearch's Deep Research Bench: the field has converged on the architecture and built a dozen ways to score it.

Every one of those benchmarks measures **how well a system answers**. Almost none measure **whether it should have answered at all**.

That gap is not incidental. AbstentionBench (Kirichenko et al., FAIR, 2025) evaluated 20 frontier LLMs across 20 datasets spanning unanswerable questions, underspecification, false premises, subjective interpretation, and stale information, and found:

- abstention is unsolved, and **model scale barely helps**;
- **reasoning fine-tuning degrades abstention by ~24% on average** — including in the math and science domains those models are explicitly trained on;
- a carefully engineered system prompt improves abstention in practice but **does not fix the underlying inability to reason about uncertainty**.

The last finding is the premise of this project. If prompting cannot fix it, it has to be fixed *structurally*.

## The approach

This system does not ask the language model whether it is confident. It computes an **evidence-sufficiency score from the retrieval process itself** — rerank score distributions, count of independent sources, cross-source contradiction rate, premise grounding, route escalation depth — and learns a calibrated mapping from those features to probability-of-correct-answer. A threshold chosen at a target risk level then gates a three-way terminal decision:

| Decision | When | Output |
|---|---|---|
| **ANSWER** | Evidence sufficient across the plan | Cited report with per-claim support labels |
| **PARTIAL** | Some sub-questions resolved, others not | Report of the resolved portions, plus an explicit statement of what could not be established and why |
| **ABSTAIN** | False premise, underspecified, or evidence does not support a confident answer | A statement of *why*, not a fluent report built on nothing |

LLM verbalized confidence is not the method here. It is a **baseline** the calibrated score is measured against.

## Claims this project will test

1. A retrieval-grounded evidence-sufficiency signal is **better-calibrated** than LLM verbalized confidence for deciding whether to answer. *(ECE, reliability diagram, risk–coverage curve, AURC.)*
2. Multi-agent structure buys the ability to **abstain correctly**, at a measurable cost in coverage. *(False-answer rate on unanswerable questions vs. single-shot baseline; coverage delta.)*
3. The agentic components are **load-bearing**, not decorative. *(Full ablation suite — see below.)*

All three are falsifiable. Claim 3 in particular can fail, and if it does, that result is reported.

## The agency contract

Multi-agent systems are easy to fake: several LLM-powered nodes, a graph, a loop — but the control flow is fixed by the engineer and the models only transform text. That is orchestration wearing a costume. This project commits in advance to a test that can catch it:

> A component is agentic **if and only if** replacing its decision with a fixed policy produces a measurable degradation in outcome. If a static policy matches it, that was orchestration with extra latency.

The system has agency at exactly five points, each with a named static counterpart and a corresponding ablation:

| | Decision surface | Static counterpart | Ablation |
|---|---|---|---|
| **D1** | Plan revision — rewrite / split / merge / abandon sub-questions mid-run | One-shot decomposition, never revisited | A1 |
| **D2** | Retrieval route and query formulation, conditioned on what previous attempts returned | Hardcoded route rule, query = sub-question verbatim | A2 |
| **D3** | Retrieval depth and stopping — sufficient / deeper / wider / unresolvable | Fixed top-k, one round, always stop | A3 |
| **D4** | Budget allocation across sub-questions by marginal expected value | Uniform split | A4 |
| **D5** | The terminal decision — answer / partial / abstain | Always answer | B2 |

Plus **B1**, the most important baseline in the project: the identical architecture with *every* adaptive decision frozen. If the full agent does not beat B1, the agency was decorative — and that is what gets reported.

Everything else is deliberately deterministic: chunking, embedding, fusion, citation renumbering, report formatting, and — importantly — the sufficiency scorer itself, which is a calibrated statistical model rather than an LLM judgment call.

## Architecture

```
                        ┌──────────────────┐
                        │    CONTROLLER    │  ← the only node that routes
                        └────────┬─────────┘
       ┌─────────────┬───────────┼───────────┬──────────────┐
       ▼             ▼           ▼           ▼              ▼
  ┌─────────┐  ┌──────────┐ ┌────────┐ ┌─────────┐  ┌─────────────┐
  │ PLANNER │  │ RETRIEVER│ │ READER │ │ CRITIC  │  │  DECIDER    │
  │  (D1)   │  │ (D2,D3)  │ │ (D3)   │ │ 5-way   │  │   (D5)      │
  └─────────┘  └────┬─────┘ └───┬────┘ └────┬────┘  └──────┬──────┘
                    └───────────┴─────┬─────┘              │
                                      ▼                    ▼
                            ┌──────────────────┐  ┌──────────────────┐
                            │  EVIDENCE STORE  │─▶│   SUFFICIENCY    │
                            │   (RAG layer)    │  │     SCORER       │
                            └──────────────────┘  └──────────────────┘
                                      └────────┬───────────┘
                                               ▼
                                  SYNTHESIZER → FINALIZER
```

A **premise checker** runs before planning: it extracts the question's presuppositions, retrieves against each, and scores whether the corpus supports them. A failed premise check is the cleanest abstention signal available and costs one retrieval round — most systems only discover a false premise after writing three paragraphs about it.

The **controller** is a single deterministic routing function. Keeping routing in one place is what makes the agency auditable: every routing decision is logged with the state that produced it.

**Not using Deep Agents.** LangChain ships a higher-level harness on top of LangGraph providing planning and subagents out of the box. This project deliberately uses the low-level `StateGraph` API, because the object of study *is* routing and decision quality — a harness that makes those decisions removes the thing being measured.

## The RAG layer

Retrieval is a first-class component here, evaluated on its own terms rather than assumed to work.

- **Hybrid retrieval:** BGE-small dense + BM25 sparse, combined with Reciprocal Rank Fusion (k=60), then cross-encoder reranking. Sparse matters because research questions are dense with exact-match tokens — model names, method names, metric names — that embeddings blur.
- **Section-aware chunking of papers.** Chunks never cross a section boundary and carry the section name as metadata. This matters more than it sounds: a claim sourced from a paper's *Related Work* describes **someone else's** contribution. A system that cannot distinguish Related Work from Results will confidently misattribute findings.
- **Payload-filtered retrieval** over Qdrant on section, source type, domain, and publication date.
- **Contradiction detection** via pairwise NLI over top passages. Contradiction density is both a sufficiency feature and a reason to hedge rather than silently pick a side.
- **Standalone retrieval evaluation:** nDCG@10 and recall@20 over hand-labeled relevance judgments, comparing dense-only / sparse-only / fusion / fusion+rerank. Without this, a wrong answer cannot be attributed to retrieval versus reasoning.

## Benchmark

48 questions in the ML-research domain, stratified 24 calibration / 24 test. **The test split is committed to git before any results exist and is touched once.**

| Tier | n | Description |
|---|---|---|
| A — Factual lookup | 10 | Single-hop, one source |
| B — Multi-source synthesis | 10 | Requires 2–3 sources |
| C — Open-ended comparison | 8 | Requires judgment across many sources |
| **D — Should abstain** | **20** | False premise (5), underspecified (5), no consensus (5), stale/unresolvable (5) |

Tier D is ~40% of the set by design: abstention is the object of study, and a small Tier D gives no statistical resolution on the headline metric. Its sub-types mirror AbstentionBench's scenario taxonomy, instantiated in this domain.

Gold annotations for Tiers A–C are 3–5 required factual claims. For Tier D they include **the reason abstention is correct** — because an abstention for the wrong reason is not a success, and scoring it as one would be exactly the self-flattering evaluation this project exists to avoid.

## Failure analysis

Every run emits a structured JSONL trace: node entries and exits, routing decisions with the state snapshot that produced them, LLM calls, retrievals. Incorrect outcomes are annotated against **MAST** (Cemri et al., UC Berkeley, 2025) — the first empirically grounded taxonomy of multi-agent LLM failures, 14 modes in 3 categories, derived from 1,600+ annotated traces across 7 frameworks at inter-annotator agreement κ = 0.88.

The result is a failure *distribution* cross-tabulated against question tier, rather than a list of anecdotes. MAST's own conclusion — that improvements in base model capability will be insufficient to address the full taxonomy — is the argument for treating this as an architecture problem.

## Results

*(pending — Phase 5)*

Nothing is reported until the calibration split has fit the sufficiency model, the four baselines and five ablations have run, and the test split has been executed once.

## Reproduction

*(pending — full instructions at C41)*

```bash
git clone https://github.com/bahetiaditi/calibrated-deep-research
cd calibrated-deep-research
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your free API keys
```

Requires Python 3.12+. Runs entirely on CPU — no GPU. All API tiers used are free.

## Repository layout

```
src/controller.py      routing logic — the agency core
src/nodes/             premise_checker, planner, retriever, reader,
                       critic, synthesizer, decider, finalizer
src/rag/               ingest, chunking, embed, sparse, fusion, rerank,
                       store, contradiction
src/sufficiency/       features, calibrated model, decision policy
src/llm/               provider (Gemini → Groq fallback), prompt-hash cache
src/tracing/           JSONL structured traces
benchmark/             questions, retrieval labels, eval runners
analysis/              scoring, calibration, plots, MAST annotation
REFERENCE.md           full design document — the source of truth
```

## References

- Kirichenko et al. (2025). *AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions.* arXiv:2506.09038
- Cemri et al. (2025). *Why Do Multi-Agent LLM Systems Fail?* arXiv:2503.13657
- Du et al. (2025). *DeepResearch Bench.* arXiv:2506.11763
- Cormack et al. (2009). *Reciprocal Rank Fusion.*

## License

MIT