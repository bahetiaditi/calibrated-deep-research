# Project 2 — Calibrated Deep Research

### A multi-agent research system that knows when not to answer

**Master reference document.** This is the single source of truth for the project. Every design decision, every commit, every metric, and every interview talking point lives here. Update it as reality diverges from the plan — a stale reference doc is worse than none.

---

## 0. How to use this document

Read Sections 1–3 once, carefully, before writing any code. They contain the thesis and the design constraints that everything else serves. Sections 4–7 are technical specifications you will return to repeatedly during implementation. Section 8 is the commit plan — work through it linearly, one commit at a time, with working tested code at every checkpoint. Sections 9–12 are for when you are shipping and interviewing.

There is no deadline. The phases are ordered by dependency, not by calendar. A phase is done when its exit criteria are met, not when a weekend ends.

**Repository name:** `calibrated-deep-research`

Descriptive over clever. It is searchable, it reads well in a resume link, and it states the thesis in three words. (Alternative if you want something shorter for the GitHub URL: `abstain`. I recommend the descriptive one — recruiters and interviewers skim repo names.)

---

## 1. Thesis

### 1.1 The one-sentence version

Deep research agents are now a solved architectural pattern and a crowded benchmark space; the unsolved problem is **knowing when the evidence does not support an answer**, so this project builds a multi-agent research system whose terminal decision is *answer / partial / abstain*, gates that decision on a **calibrated evidence-sufficiency score** rather than the model's verbalized confidence, and proves through ablation that the agentic components are load-bearing rather than decorative.

### 1.2 Why the obvious version of this project is not worth building

The planner → executor → synthesizer → critic pattern is the standard architecture as of 2026. The evaluation methodology that would normally differentiate a portfolio project — citation-grounding precision against a curated benchmark — is also standard. The field has DeepResearch Bench (100 expert-authored PhD-level tasks across 22 fields, ICLR 2026), ResearcherBench (expert-rubric insight quality plus citation-based factuality), DeepScholar-Bench and ReportBench (citation-grounded verifiability of survey reports), LiveDRBench (claim recovery in multi-step search), FutureSearch's Deep Research Bench (91 tasks with the web frozen offline for reproducibility), plus arena-style and agent-as-judge frameworks.

Against that landscape, "I built a deep research agent with a critic and measured citation grounding" is a competent implementation of a known pattern. It will not carry an interview.

### 1.3 What is actually unsolved

Every one of those benchmarks measures how well a system answers. Almost none measure whether it should have answered.

AbstentionBench (Kirichenko et al., FAIR, 2025) evaluated 20 frontier LLMs across 20 datasets covering six abstention scenarios — unknown answers, underspecification, false premises, subjective interpretation, and stale information. The findings that matter for us:

- Abstention is unsolved and **model scale barely helps**.
- **Reasoning fine-tuning degrades abstention by ~24% on average**, including in the math and science domains those models are explicitly trained on.
- A carefully engineered system prompt improves abstention *in practice* but does not fix the underlying inability to reason about uncertainty.

That last point is the entire justification for this project's architecture. If prompting cannot fix it, then it has to be fixed **structurally** — with retrieval-derived evidence signals and a calibrated decision boundary, not with a better instruction.

### 1.4 The three claims this project will make

1. **A retrieval-grounded evidence-sufficiency signal is better-calibrated than LLM verbalized confidence for deciding whether to answer.** (Measured: ECE, risk–coverage curve, AURC.)
2. **Multi-agent structure buys the ability to abstain correctly, at a measurable cost in coverage.** (Measured: false-answer rate on unanswerable questions vs. single-shot baseline; coverage delta.)
3. **The agentic components are load-bearing.** Replacing each adaptive decision with a static policy degrades performance measurably. (Measured: full ablation suite, Section 6.5.)

Claim 3 is the one that answers "is this real agency or just orchestration?" — and it is the one most portfolio projects cannot make, because they never test it.

---

## 2. The agency contract

This section exists because of a specific failure mode: systems that look agentic (multiple LLM-powered nodes, a graph, a loop) but are functionally a fixed pipeline with language models sprinkled at the nodes. The LLM is doing text transformation; the *control flow* is entirely predetermined by the engineer. That is orchestration.

### 2.1 The test

> **A component is agentic if and only if replacing its decision with a fixed policy produces a measurable degradation in outcome. If a static policy matches it, you did not build agency — you built orchestration with extra latency.**

This test is falsifiable, cheap to run, and becomes Section 6.5 of the evaluation. Committing to it in advance is the intellectually honest move: it means the project can *fail* to demonstrate agency, and we report that if it happens.

### 2.2 The five decision surfaces

The system has agency at exactly five points. Everything else is deterministic plumbing, and that is fine — the goal is not to make everything an LLM call, it is to make the genuinely uncertain decisions adaptive.

**D1 — Plan revision.** The planner does not decompose once and hand off. It is re-entered whenever the controller detects that the plan is failing: a sub-question that retrieval cannot satisfy, a discovered entity that opens a new line of inquiry, or evidence contradicting the plan's premise. The planner can add, drop, merge, or rewrite sub-questions mid-run. *Static counterpart: one-shot decomposition, never revisited.*

**D2 — Retrieval route and query formulation.** For each sub-question, the agent chooses (a) which retrieval route to use — arXiv metadata search, full-text paper reading, web search, or the accumulated evidence store — and (b) how to phrase the query, conditioned on what previous queries returned. A query that returned nothing informs the next one. *Static counterpart: hardcoded rule ("arxiv for academic-sounding sub-questions, web otherwise"), query = sub-question verbatim.*

**D3 — Retrieval depth and stopping.** After each retrieval round, the agent decides whether the evidence for this sub-question is sufficient, whether to go *deeper* (download and chunk the paper rather than using the abstract), whether to go *wider* (more sources), or whether to declare this sub-question unresolvable. *Static counterpart: fixed top-k, single round, always stop.*

**D4 — Budget allocation.** The run has a hard budget (LLM calls, retrieval calls, wall-clock). The agent decides how to spend it across sub-questions — spending more on the ones that are close to sufficiency and cutting losses on the ones that are not. *Static counterpart: uniform split across sub-questions.*

**D5 — The terminal decision.** Answer, partial answer with declared gaps, or abstain. This is the highest form of agency in the system: deciding not to act. *Static counterpart: always answer.*

### 2.3 The critic must have more than two options

In the naive design, the critic's verdict routes binarily: `UNSUPPORTED → re-retrieve`, everything else → finalize. That is a conditional edge, not a decision.

Here the critic's per-claim verdict routes into **five** outcomes, and *which* outcome is itself a judgment:

| Verdict + context | Route |
|---|---|
| Unsupported, but retrieval looks promising and budget remains | Targeted re-retrieval with a reformulated query |
| Unsupported, and the sub-question itself was malformed | Return to planner — revise the sub-question (D1) |
| Unsupported, and evidence suggests the claim is simply wrong | Drop the claim, flag as a corrected error in the trace |
| Partially supported, evidence is genuinely mixed | Keep with an explicit hedge and cite both sides |
| Unsupported, budget exhausted or retrieval repeatedly failed | Mark the sub-question unresolved; feeds D5 |

Five-way routing on a judgment call is the difference between a critic and an `if` statement.

### 2.4 Explicit anti-patterns — if you find yourself doing these, stop

- **The plan is written once and never read again.** If `state["plan"]` is consulted only by the executor's `for` loop, D1 does not exist.
- **The loop counter is the only thing that changes between iterations.** If iteration 2 issues the same queries as iteration 1, you have a retry, not a re-plan.
- **Every LLM call is a text transformation.** If no LLM output ever changes *which node runs next*, the graph is a straight line.
- **"Agentic" behaviour is asserted in the README but never measured.** This is what Section 6.5 exists to prevent.
- **The abstention decision is an LLM saying "I'm not sure."** That is verbalized confidence, and it is our *baseline*, not our method.

### 2.5 What is deliberately NOT agentic

Scope discipline. These stay deterministic, and saying so is itself a design signal:

- Chunking, embedding, indexing, and fusion — deterministic retrieval mechanics.
- Report formatting, citation renumbering, footnote generation.
- The sufficiency *scorer* — it is a calibrated model, not an LLM judgment call. That is the point.
- Parallel execution. Sequential throughout. Noted as future work.

---

## 3. System architecture

### 3.1 Component map

```
                        ┌──────────────────┐
                        │    CONTROLLER    │  ← the only node that routes
                        │  (routing logic) │
                        └────────┬─────────┘
       ┌─────────────┬───────────┼───────────┬──────────────┐
       ▼             ▼           ▼           ▼              ▼
  ┌─────────┐  ┌──────────┐ ┌────────┐ ┌─────────┐  ┌─────────────┐
  │ PLANNER │  │ RETRIEVER│ │ READER │ │ CRITIC  │  │  DECIDER    │
  │  (D1)   │  │ (D2,D3)  │ │ (D3)   │ │ (2.3)   │  │   (D5)      │
  └─────────┘  └────┬─────┘ └───┬────┘ └────┬────┘  └──────┬──────┘
                    │           │           │              │
                    └───────────┴─────┬─────┘              │
                                      ▼                    ▼
                            ┌──────────────────┐  ┌──────────────────┐
                            │  EVIDENCE STORE  │  │   SUFFICIENCY    │
                            │  (RAG layer, §4) │─▶│  SCORER (§5)     │
                            └──────────────────┘  └──────────────────┘
                                      │                    │
                                      └────────┬───────────┘
                                               ▼
                                       ┌───────────────┐
                                       │   SYNTHESIZER │
                                       └───────┬───────┘
                                               ▼
                                       ┌───────────────┐
                                       │   FINALIZER   │
                                       └───────────────┘
```

The **Controller** is the architectural centrepiece and the thing that makes this a graph rather than a pipeline. It is the only component that decides what runs next, and it decides based on state: current plan status, per-sub-question sufficiency scores, remaining budget, and critic findings. Everything else is a worker that does one job and returns.

### 3.2 State object

```python
from typing import TypedDict, Literal, Annotated
from operator import add

Route = Literal["arxiv_meta", "arxiv_fulltext", "web", "evidence_store"]
Status = Literal["pending", "in_progress", "sufficient", "insufficient", "abandoned"]
Verdict = Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"]
Terminal = Literal["ANSWER", "PARTIAL", "ABSTAIN"]

class SubQuestion(TypedDict):
    id: str
    text: str
    status: Status
    rationale: str              # why the planner created it
    attempted_routes: list[Route]
    attempted_queries: list[str]   # prevents D2 from repeating failures
    rounds: int
    sufficiency: float | None      # from §5
    sufficiency_features: dict     # the raw feature vector, logged for analysis
    parent_id: str | None          # set when planner splits a sub-question (D1)

class Passage(TypedDict):
    id: str                     # P0001 …
    source_type: Literal["arxiv", "web"]
    source_id: str              # arXiv ID or canonical URL
    source_domain: str          # for independence counting
    title: str
    text: str
    section: str | None         # section-aware chunking for papers
    published: str | None       # ISO date if known — staleness signal
    sub_question_id: str
    retrieval_score: float      # post-fusion
    rerank_score: float | None

class ClaimVerification(TypedDict):
    claim: str
    sub_question_id: str
    cited_passage_ids: list[str]
    verdict: Verdict
    reasoning: str
    action_taken: str           # which of the five routes in §2.3

class BudgetState(TypedDict):
    llm_calls_used: int
    llm_calls_max: int
    retrieval_calls_used: int
    retrieval_calls_max: int
    allocation: dict[str, int]  # sub_question_id -> calls granted (D4)

class ResearchState(TypedDict):
    question: str
    premise_check: dict                 # false-premise detection result
    plan: list[SubQuestion]
    plan_revisions: int
    evidence: Annotated[list[Passage], add]
    contradictions: list[dict]          # pairwise NLI findings
    draft: str
    critic_findings: list[ClaimVerification]
    budget: BudgetState
    terminal_decision: Terminal | None
    decision_rationale: str
    unresolved_gaps: list[str]          # what we could not establish, and why
    final_report: str
    trace_id: str
    error: str | None
```

Note the fields that exist purely to *enable* agency: `attempted_queries` (so D2 does not repeat itself), `attempted_routes` (so D2 can escalate), `rounds` and `sufficiency` (so D3 can decide), `allocation` (D4), `plan_revisions` (D1), `unresolved_gaps` (D5). If you find yourself never reading a field, the corresponding decision surface is dead.

### 3.3 Node specifications

**Controller.** Pure Python, no LLM. Reads state and returns the next node name. Routing table:

```
if premise_check not run                     → premise_checker
if plan is empty                             → planner
if any sub-question pending/in_progress
   and budget remains                        → retriever (pick highest-value SQ per D4)
if a sub-question is insufficient after N
   rounds and planner has not revised it     → planner (revision, D1)
if all sub-questions terminal                → sufficiency_scorer
if sufficiency computed and no draft         → synthesizer
if draft exists and critic not run           → critic
if critic findings require action            → route per §2.3 table
if all resolved or budget exhausted          → decider
after decider                                → finalizer
```

Keeping routing in one deterministic function is what makes the agency auditable — you can log every routing decision with the state that produced it, which is what MAST annotation (Section 6.6) needs.

**Premise checker.** Runs *before* planning. Extracts the presuppositions of the question ("Mamba-2 uses rotary position embeddings" is presupposed by "why does Mamba-2 use RoPE in its SSM blocks?"), issues a targeted retrieval for each presupposition, and scores whether the corpus supports it. A failed premise check is the cleanest possible abstention signal and it costs one retrieval round. This node is a genuine contribution — most systems only discover false premises after having written three paragraphs about them.

**Planner (D1).** Two modes. *Initial*: decompose into 3–7 sub-questions with a rationale and a suggested route for each. *Revision*: given the failing sub-question, its attempted queries and routes, and the evidence retrieved so far, decide whether to rewrite it, split it, merge it with another, or abandon it as unanswerable. Structured JSON output with defensive parsing.

**Retriever (D2, D3).** Given a sub-question and its history, chooses a route and formulates a query. Executes through the RAG layer (Section 4). After the round, computes sufficiency features and decides: sufficient / go deeper / go wider / give up. The choice among {deeper, wider} is real — deeper means reading full text of a paper already found, wider means new sources.

**Reader.** Full-text ingestion of a specific paper: download, parse, section-aware chunk, embed, index, retrieve top chunks against the sub-question. Invoked only when the retriever decides depth is needed (D3), which makes depth an *earned* cost rather than a default.

**Critic.** Claim segmentation, per-claim entailment judgment against cited passages, and the five-way routing decision from Section 2.3. Also consumes the contradiction findings from Section 4.6.

**Sufficiency scorer.** Not an LLM. A calibrated model over retrieval-derived features (Section 5).

**Decider (D5).** Consumes per-sub-question sufficiency, plan coverage, premise-check result, and contradiction density. Applies the calibrated threshold. Emits terminal decision plus a rationale and an explicit list of unresolved gaps.

**Synthesizer.** Drafts with mandatory `[Pxxxx]` citations. On the abstain path it does not run at all; on the partial path it is instructed to write only the resolved portions and to state the gaps.

**Finalizer.** Deterministic formatting: footnotes, verification summary, sources, decision banner, gap statement.

### 3.4 On not using Deep Agents

LangChain ships **Deep Agents**, a higher-level harness on top of LangGraph providing planning, subagents, and filesystem tools out of the box. We deliberately do not use it. The entire project is a study of *routing and decision quality*; a harness that makes those decisions for you removes the object of study. This is a talking point, not a limitation — state it in the README.

LangGraph itself is at 1.0+ (GA October 2025), with durable execution, checkpointing at every node, and built-in persistence. Note that `langgraph.prebuilt` is deprecated with functionality moved to `langchain.agents`; we use the low-level `StateGraph` API throughout, so this does not affect us.

---

## 4. The RAG layer

This is a real retrieval system, evaluated on its own terms, not a `Chroma.from_documents()` call. It is roughly a third of the project's engineering value and it is what makes this "RAG **and** multi-agent" rather than "multi-agent with a vector store attached."

### 4.1 Corpus and ingestion

Three source types, unified into the `Passage` schema:

- **arXiv metadata** — title, abstract, authors, date, categories. Cheap, broad, good for orientation.
- **arXiv full text** — downloaded PDF, parsed, section-aware chunked. Expensive, deep, invoked by D3.
- **Web pages** — search results with extracted content.

Ingestion is idempotent and cached on disk by content hash. A paper downloaded during question 7 is available for free during question 23. This cross-run corpus accumulation is what makes the evidence store a genuine *store* rather than a per-run scratchpad, and it is the mechanism behind the `evidence_store` retrieval route in D2.

### 4.2 Chunking

Web content: sliding window, 800 characters with 150 overlap, sentence-boundary aware.

Papers: **section-aware**. Parse section headers via regex over the extracted text (`^\d+(\.\d+)*\s+[A-Z]`, plus a list of canonical headers: Abstract, Introduction, Related Work, Method, Experiments, Results, Discussion, Conclusion, Limitations). Chunk within sections, never across a section boundary, and carry the section name into `Passage.section`.

Why this matters and is worth saying in an interview: a claim sourced from a paper's *Related Work* section is describing someone else's contribution, not the paper's. A retrieval system that cannot tell Related Work from Results will confidently attribute the wrong result to the wrong paper. Section metadata lets the critic down-weight or flag these. This is a small idea with a large correctness payoff and almost nobody implements it.

### 4.3 Hybrid retrieval

- **Dense:** `BAAI/bge-small-en-v1.5` (384-dim, CPU-fast, meaningfully stronger than MiniLM-L6 on retrieval benchmarks). Query prefix `"Represent this sentence for searching relevant passages: "` — required by the model, easy to forget, silently degrades everything if omitted.
- **Sparse:** BM25 via `bm25s` (faster than `rank_bm25`, same interface shape). Essential here because research questions are full of exact-match tokens — model names, method names, metric names — that dense retrieval blurs.
- **Fusion:** Reciprocal Rank Fusion, `score = Σ 1/(k + rank_i)` with `k=60`. Parameter-free, robust, no score normalization headaches.
- **Reranking:** cross-encoder `BAAI/bge-reranker-base` over the top 30 fused candidates, keep top 5–10. CPU-viable in batch.

**Vector store: Qdrant, local mode** (`QdrantClient(path=...)`, no server). Chosen over Chroma for two reasons: you already know it from Meghnad, and its payload filtering is first-class — which we need for metadata-filtered retrieval and which gives you a clean narrative bridge to Project 1.

### 4.4 Metadata filtering — and the bridge to Project 1

Every passage carries filterable payload: `source_type`, `source_domain`, `published` date, `section`, `sub_question_id`. The retriever uses these — restricting to papers when the sub-question is technical, excluding pre-2023 sources when the sub-question concerns recent work, targeting Results sections when the sub-question asks for a number.

This is exactly the pre-filter / post-filter / predicate-aware problem from Project 1, appearing in a live system. When an interviewer asks how the two projects connect: *"Project 1 characterizes when each filtering strategy wins as a function of selectivity. Project 2 is where I hit that regime in practice — filtering on section and date at low selectivity is precisely where post-filtering collapses, which is why the store does payload filtering during traversal rather than after."* That is a strong, specific, honest connection between two portfolio pieces.

### 4.5 Retrieval evaluation in its own right

Do not let retrieval quality be an untested assumption underneath the agent results.

Build a small labeled retrieval set: for ~30 sub-questions drawn from benchmark runs, manually label the top-20 pooled candidates as relevant / partially relevant / irrelevant. ~600 judgments; a focused afternoon. Then report **nDCG@10** and **recall@20** for four configurations:

| Config | Description |
|---|---|
| R1 | Dense only |
| R2 | BM25 only |
| R3 | RRF fusion |
| R4 | RRF + cross-encoder rerank |

This is a self-contained RAG study you can present independently. It also protects the main result: if the agent underperforms, you can say whether retrieval or reasoning was the bottleneck. Most projects cannot.

### 4.6 Contradiction detection

For each sub-question, run pairwise NLI over the top reranked passages using a small local model (`cross-encoder/nli-deberta-v3-small`). Record contradiction pairs into `state["contradictions"]`.

Two payoffs. First, contradiction density becomes a **sufficiency feature** — sources disagreeing is a legitimate reason to hedge or abstain, and it is the reason Tier D's "no-consensus" questions should trigger abstention rather than a confident pick. Second, on the answer path it lets the synthesizer present genuine disagreement as disagreement instead of silently choosing one side, which is a real quality improvement and a good demo moment.

---

## 5. Evidence sufficiency and calibrated abstention

The technical centre of the project. Everything above feeds this.

### 5.1 The core move

Do not ask the LLM whether it is confident. Compute a feature vector from the retrieval process and learn a calibrated mapping from features to probability-of-correct-answer. Then choose a threshold at a target risk level.

The LLM's verbalized confidence becomes **baseline B3**, and the expected result — that it is poorly calibrated relative to the feature-based score — is the paper-grounded hypothesis from Section 1.3.

### 5.2 Feature vector (per sub-question)

| # | Feature | Rationale |
|---|---|---|
| f1 | Max rerank score | Peak evidence quality |
| f2 | Mean of top-3 rerank scores | Depth of support, not just one lucky hit |
| f3 | Score gap (top-1 minus top-5) | Sharp peak vs. flat mush; flat = ambiguous |
| f4 | Count of independent sources | Distinct `source_domain` / arXiv ID among top passages |
| f5 | Contradiction rate | Fraction of top-passage pairs labeled contradiction |
| f6 | Entailment strength | Critic verdict distribution mapped to [0,1] |
| f7 | Premise-grounding score | From the premise checker — does the question's presupposition appear at all |
| f8 | Retrieval rounds consumed | Many rounds to reach thin evidence is a bad sign |
| f9 | Route escalation depth | Did it need full-text reading, or did an abstract suffice |
| f10 | Recency alignment | Gap between question's implied timeframe and source dates |

Run-level features aggregate these: mean and min sufficiency across sub-questions, fraction of sub-questions reaching `sufficient`, plan revision count, budget fraction consumed.

### 5.3 The model

**Logistic regression** on the ~10 sub-question features and ~5 run-level aggregates. Deliberately simple:

- Interpretable — you can report which signals actually predict answerability, which is itself an interesting finding.
- Trainable on ~24 calibration questions without overfitting, where a gradient-boosted model would not be.
- Naturally produces probabilities suitable for calibration analysis.

Fit on the calibration split. Apply **Platt scaling / isotonic regression** if the reliability diagram shows systematic miscalibration. Report **ECE** and a reliability diagram against the verbalized-confidence baseline.

### 5.4 Threshold selection

Choose the decision threshold τ to hit a target risk on the calibration split — e.g. *"error rate ≤ 10% among answered questions."* Then report the **achieved** risk on the held-out test split. If the achieved risk exceeds the target, say so; that is a real and interesting negative result about small-sample calibration, not a failure of the project.

Three-way decision:

```
if premise_check fails                          → ABSTAIN (hard rule, bypasses τ)
elif run_sufficiency ≥ τ_answer                 → ANSWER
elif run_sufficiency ≥ τ_partial                → PARTIAL (answer resolved parts, declare gaps)
else                                            → ABSTAIN
```

The hard premise rule is intentional: a false premise is a categorical failure, not a probabilistic one, and hard-coding it is the correct engineering call.

### 5.5 Metrics for the abstention layer

- **Risk–coverage curve** and **AURC** — the headline plot. Sweep τ from 0 to 1; at each point plot (fraction answered, error rate among answered).
- **Accuracy @ 80% coverage** — a single comparable number.
- **False-abstention rate** on Tiers A–C — refusing answerable questions.
- **False-answer rate** on Tier D — the headline number, and the one where the single-shot baseline should look bad.
- **ECE** and reliability diagram — calibrated score vs. verbalized confidence.

---

## 6. Evaluation protocol

### 6.1 Benchmark composition (48 questions)

| Tier | n | Description |
|---|---|---|
| **A — Factual lookup** | 10 | Single-hop, one source. *"What attention variant does Mistral 7B use?"* |
| **B — Multi-source synthesis** | 10 | Requires 2–3 sources. *"How do FlashAttention-2 and PagedAttention differ in memory access patterns?"* |
| **C — Open-ended comparison** | 8 | Requires judgment across many sources. *"What are the main critiques of the Chinchilla scaling laws?"* |
| **D — Should abstain** | 20 | Four sub-types, 5 each (below) |

**Tier D sub-types** — mirroring AbstentionBench's scenario taxonomy, instantiated in the ML-research domain:

- **D1 False premise (5).** *"Why does Mamba-2 use rotary position embeddings in its SSM blocks?"* — it does not. Correct behaviour: identify and correct the false presupposition.
- **D2 Underspecified (5).** *"Is the newer model better?"* / *"What's the best rank for fine-tuning?"* — better at what, for which model, under what budget. Correct behaviour: request specification.
- **D3 No consensus (5).** *"What is the optimal LoRA rank for a 7B model?"* — genuinely contested. Correct behaviour: present the disagreement, decline a single answer.
- **D4 Stale / unresolvable (5).** Questions whose answers move, or that require information not in the accessible corpus. Correct behaviour: state the limitation.

Balance matters: Tier D is ~40% of the set, which is high by design because abstention is the object of study and a small Tier D gives you no statistical resolution on the headline metric.

### 6.2 Gold annotations

For A–C: 3–5 gold claims each — factual statements a correct answer must contain.

For D: the *reason* abstention is correct, plus (for false-premise items) the correct correction. This matters — an abstention for the wrong reason is not a success, and grading it as one would be exactly the kind of self-flattering evaluation this project exists to avoid.

### 6.3 Splits

Stratified 50/50: **24 calibration** (fits the sufficiency model and τ) / **24 test** (reported numbers). Stratify across tiers *and* across Tier D sub-types. The test split is touched once, at the end. Write this rule in the README and honour it — a portfolio project with a genuinely held-out split is rarer than it should be.

### 6.4 Baselines

| ID | Baseline | Isolates |
|---|---|---|
| **B0** | Single-shot LLM + web search tool, one prompt | Value of the entire system |
| **B1** | **Static pipeline** — same nodes, fixed plan, fixed k, no revision, no adaptive stopping, no budget allocation | **Value of agency** |
| **B2** | Full agent, abstention disabled (always answers) | Value of the abstention layer |
| **B3** | Full agent, abstention gated on LLM verbalized confidence instead of the calibrated score | Value of calibration specifically |

**B1 is the most important baseline in the project** and the one that directly answers your concern. It is the same architecture with every adaptive decision frozen. If the full agent does not beat B1, you have measured that your agency was decorative — and you report that, honestly, which is a stronger interview position than an unfalsifiable claim.

### 6.5 Agency ablations

Each ablation disables exactly one decision surface, holding everything else fixed.

| ID | Disabled | Static replacement |
|---|---|---|
| **A1** | D1 plan revision | One-shot decomposition |
| **A2** | D2 route/query choice | Hardcoded routing rule, query = sub-question verbatim |
| **A3** | D3 adaptive stopping | Fixed top-k, one round always |
| **A4** | D4 budget allocation | Uniform split across sub-questions |
| **A5** | Five-way critic routing | Binary: re-retrieve or accept |

Report each against the headline metrics. Some will move the numbers meaningfully; some will not. **Report both.** An ablation table where two of five components turn out not to matter is a more credible artifact than one where everything conveniently helps, and "which parts of agentic design actually earn their cost" is a genuinely interesting question that most practitioners cannot answer empirically.

### 6.6 Failure analysis via MAST

MAST (Cemri et al., 2025, UC Berkeley) is the first empirically grounded taxonomy of multi-agent LLM failures: **14 failure modes in 3 categories** — system design issues, inter-agent misalignment, task verification — derived from 1,600+ annotated traces across 7 frameworks, with inter-annotator agreement κ = 0.88. The authors ship it as a pip-installable library (`agentdash`) and the annotated dataset is on HuggingFace as `mcemri/MAD`.

**Protocol:**

1. Every run emits a structured JSONL trace: every node entry/exit, every routing decision with the state that produced it, every LLM call, every retrieval.
2. For each run where the outcome was wrong (wrong answer, false abstention, false answer), annotate against the 14 MAST modes.
3. Report the failure distribution: *"of 24 test runs, 9 produced incorrect outcomes; 44% were task-verification failures (critic accepting weak support), 33% system-design (step repetition in the re-retrieval loop), 22% inter-agent misalignment (planner sub-questions the retriever could not operationalize)."*
4. Cross-tabulate failure mode against tier — do false-premise questions fail differently from no-consensus ones? Almost certainly yes, and that is a finding.

MAST's own conclusion is that improvements in base model capability will be insufficient to address the full taxonomy — which is the argument for architectural work like this, and a good line to have ready.

### 6.7 Cost and latency

Report honestly per question: LLM calls, tokens, retrieval calls, wall-clock. The multi-agent system will be slower and more expensive than B0. The interesting framing is **cost per correctly-answered-or-correctly-abstained question**, which changes the picture considerably once B0's confident wrong answers are counted as failures rather than free wins.

---

## 7. Stack, budget, and constraints

### 7.1 Verified as of August 2026

**Gemini free tier tightened significantly.** Since 1 April 2026, Pro models are paid-only; only **Flash and Flash-Lite** remain free. Limits run **5–15 RPM and up to ~1,000 requests/day**, and free-tier prompts may be used to improve Google's products. Limits are applied per project and vary by model and tier — check the live rate-limit view in AI Studio for your project before finalizing the budget rather than trusting any blog post, including this one.

**Budget arithmetic.** 48 questions × ~20 LLM calls ≈ 960 calls for one full multi-agent run. That is your entire daily quota. Therefore:

- **Prompt-hash response caching is mandatory from Commit 3**, not an optimization. Every LLM call keyed by `sha256(model + system + user + temperature)`, persisted to disk.
- Eval runs are **batched across days**. `run_eval.py` must be resumable and skip completed questions.
- Groq (Llama 3.3 70B) as automatic fallback on 429/500.
- The five ablations plus four baselines multiply this. Plan on eval taking one to two weeks of wall-clock. This is fine — there is no deadline.

**LangGraph 1.0+.** Durable execution and checkpointing are stable; use `SqliteSaver` so an interrupted eval run resumes from its last completed node rather than restarting. Note `langgraph.prebuilt` is deprecated (functionality moved to `langchain.agents`); we use low-level `StateGraph` regardless.

### 7.2 Dependencies

| Package | Purpose |
|---|---|
| `langgraph` (1.x) | Graph orchestration, checkpointing |
| `langchain-core` | Message/prompt abstractions only |
| `google-genai` | Gemini Flash (primary LLM). **Note:** `google-generativeai` is deprecated (last release 2025-12-16); Google unified its SDKs into `google-genai`. |
| `groq` | Llama 3.3 70B (fallback) |
| `qdrant-client` | Local vector store with payload filtering |
| `sentence-transformers` | bge-small-en-v1.5 embeddings, bge-reranker-base, NLI model |
| `bm25s` | Sparse retrieval |
| `arxiv` | arXiv API (3s inter-request delay, `num_retries` supported) |
| `tavily-python` | Web search, primary for eval (1000 credits/month) |
| `ddgs` | DuckDuckGo, dev/debug use (rate-limits aggressively) |
| `pypdf` | PDF text extraction |
| `scikit-learn` | Logistic regression, calibration, isotonic regression |
| `numpy`, `pandas`, `matplotlib` | Analysis and plots |
| `gradio` | Demo |
| `pytest` | Tests |
| `agentdash` | MAST annotation support (**optional, stale** — last release 2025-08-12; the 14-mode taxonomy can be applied by hand, do not let this block C38) |

API keys, all free: `GOOGLE_API_KEY` (aistudio.google.com), `TAVILY_API_KEY` (app.tavily.com), `GROQ_API_KEY` (console.groq.com).

**Python 3.12+ required** (numpy 2.5 and pandas 3.x set the floor). Everything runs on the MacBook Air. No GPU required. The embedding, reranking, and NLI models are all CPU-viable.

### 7.3 Repository structure

```
calibrated-deep-research/
├── README.md                      # thesis, architecture, headline results
├── REFERENCE.md                   # this document
├── requirements.txt
├── .env.example
├── config.yaml                    # budgets, thresholds, model names, k values
├── src/
│   ├── state.py                   # all TypedDicts
│   ├── graph.py                   # StateGraph assembly
│   ├── controller.py              # routing logic — the agency core
│   ├── budget.py                  # D4 budget manager
│   ├── nodes/
│   │   ├── premise_checker.py
│   │   ├── planner.py             # D1: initial + revision modes
│   │   ├── retriever.py           # D2, D3
│   │   ├── reader.py              # full-text ingestion
│   │   ├── critic.py              # five-way routing
│   │   ├── synthesizer.py
│   │   ├── decider.py             # D5
│   │   └── finalizer.py
│   ├── rag/
│   │   ├── ingest.py              # download, cache, parse
│   │   ├── chunking.py            # section-aware + sliding window
│   │   ├── embed.py               # bge-small wrapper
│   │   ├── sparse.py              # BM25
│   │   ├── fusion.py              # RRF
│   │   ├── rerank.py              # cross-encoder
│   │   ├── store.py               # Qdrant wrapper + payload filters
│   │   └── contradiction.py       # NLI pairwise
│   ├── sufficiency/
│   │   ├── features.py            # f1–f10 extraction
│   │   ├── model.py               # logistic regression + calibration
│   │   └── policy.py              # threshold selection, three-way decision
│   ├── tools/
│   │   ├── arxiv_tool.py
│   │   └── web_search.py          # SearchProvider interface
│   ├── llm/
│   │   ├── provider.py            # Gemini → Groq fallback, retries
│   │   └── cache.py               # prompt-hash disk cache
│   ├── tracing/
│   │   └── tracer.py              # JSONL structured traces
│   └── prompts/                   # all prompts as versioned .txt files
├── benchmark/
│   ├── questions.json             # 48 questions, gold claims, tier, split
│   ├── retrieval_labels.json      # ~600 relevance judgments
│   ├── run_agent.py
│   ├── run_baselines.py           # B0–B3
│   ├── run_ablations.py           # A1–A5
│   └── case_studies/
├── analysis/
│   ├── score.py                   # all metrics
│   ├── calibrate.py               # fit sufficiency model, pick τ
│   ├── plots.py                   # risk-coverage, reliability, ablation bars
│   └── mast_annotate.py           # trace → failure-mode annotation
├── results/                       # metrics, tables, figures
├── traces/                        # JSONL run traces
├── writeup/
│   └── writeup.md                 # the technical report
├── demo/
│   └── app.py
└── tests/
```

---

## 8. Commit plan

Forty commits across seven phases. Each is one logical unit with working, tested code at the end. Do not start the next until the current one passes its check.

### Phase 0 — Foundations (5 commits)

**C1 — Scaffold and thesis.** Repo init, directory tree, `requirements.txt` with pinned versions, `.env.example`, `config.yaml`, `.gitignore`. Write the README *thesis section first* — stating the claim before building forces the scope to stay honest. Copy this document in as `REFERENCE.md`.
*Check:* `pip install -r requirements.txt` succeeds in a clean venv.

**C2 — LLM provider with fallback.** `src/llm/provider.py`. Gemini Flash primary, Groq fallback on 429/500, exponential backoff, max 3 retries, structured-output helper with defensive JSON parsing (strip fences, repair truncation, retry-with-correction).
*Check:* both providers return; fallback triggers on a simulated 429.

**C3 — Prompt-hash cache.** `src/llm/cache.py`. SHA256 over (model, system, user, temperature) → disk. Toggleable. Hit/miss stats.
*Check:* identical call twice makes one API request; stats report one hit.

**C4 — Structured tracing.** `src/tracing/tracer.py`. Every node entry/exit, routing decision (with the state snapshot that produced it), LLM call, retrieval call → JSONL keyed by `trace_id`. This is the substrate for MAST annotation; building it now costs an hour, retrofitting it costs a weekend.
*Check:* a dummy three-node run produces a well-formed, parseable trace.

**C5 — Config and budget manager.** `config.yaml` + `src/budget.py`. Budget accounting, allocation interface (D4 uses it in C22), hard stops.
*Check:* budget exhaustion raises cleanly; allocations sum to the cap.

> **Phase 0 exit:** you can make a cached, traced, budgeted LLM call with automatic fallback.

### Phase 1 — RAG substrate (8 commits)

**C6 — arXiv tool.** Search, metadata → `Passage`, 3s delay, retries, graceful degradation on failure.
*Check:* returns valid passages for three sample queries; handles a deliberately malformed query.

**C7 — Web search with provider interface.** `SearchProvider` ABC; Tavily and DuckDuckGo implementations; factory selecting on available keys.
*Check:* both return valid passages; DDG rate-limit exception is caught and falls back.

**C8 — PDF ingestion and section-aware chunking.** Download with content-hash caching, `pypdf` extraction, section header detection, within-section chunking, sliding window for web.
*Check:* on three real papers, section names are correctly extracted and no chunk crosses a section boundary.

**C9 — Embeddings and Qdrant store.** bge-small-en-v1.5 with the required query prefix, local Qdrant, payload schema, upsert, similarity search, filtered search, get-by-id.
*Check:* store 50 passages, retrieve by similarity, retrieve with a `section == "Results"` filter, retrieve by ID.

**C10 — BM25 and RRF fusion.** `bm25s` index over the same corpus; RRF with k=60; unified `retrieve(query, filters, top_k)` returning fused results.
*Check:* an exact-token query ("PagedAttention") ranks higher under fusion than under dense alone.

**C11 — Cross-encoder reranking.** bge-reranker-base over top 30 → top 5–10, batched.
*Check:* reranking measurably reorders on a hand-checked example; latency is acceptable on CPU.

**C12 — Contradiction detection.** Pairwise NLI over top passages, contradiction records into state.
*Check:* on a hand-built contradicting pair, contradiction is detected; on an agreeing pair, it is not.

**C13 — Retrieval evaluation harness.** `benchmark/retrieval_labels.json` (~600 judgments over ~30 sub-questions), nDCG@10 and recall@20 for R1–R4.
*Check:* the R1–R4 table runs end to end and R4 ≥ R3 ≥ max(R1, R2), or you have an explanation for why not.

> **Phase 1 exit:** you have a measured retrieval system, with a table proving hybrid+rerank beats the alternatives on your corpus. This is publishable on its own.

### Phase 2 — Agentic core (9 commits)

**C14 — State definitions.** All TypedDicts from Section 3.2, with reducers.
*Check:* a state object round-trips through LangGraph's checkpointer.

**C15 — Premise checker.** Presupposition extraction, targeted retrieval per presupposition, grounding score, pass/fail.
*Check:* on three hand-written false-premise questions it fails; on three valid questions it passes.

**C16 — Planner, initial mode.** Decomposition to 3–7 sub-questions with rationale and suggested route.
*Check:* valid structured output on ten sample questions with no parse failures.

**C17 — Retriever, static version (D2/D3 disabled).** Fixed route, verbatim query, fixed top-k. This is deliberately the *static* version — it becomes ablation A2/A3 later and gives you a working pipeline sooner.
*Check:* retrieves for every sub-question in a plan.

**C18 — Synthesizer.** Draft with mandatory `[Pxxxx]` citations, post-processing validation that every citation resolves to a real passage.
*Check:* a full run produces a draft where 100% of citation IDs exist.

**C19 — Controller and minimal graph.** `src/controller.py` routing function, `StateGraph` assembly, SqliteSaver checkpointing, `run_research(question)`.
*Check:* end-to-end run on one question; kill the process mid-run and confirm it resumes from checkpoint.

> This is the point where you have a working system. Everything after this is making it *agentic* and *calibrated*.

**C20 — Sufficiency features (D3 substrate).** `src/sufficiency/features.py` computing f1–f10 per sub-question. Log them into state on every retrieval round.
*Check:* feature vectors are populated and dimensionally consistent across ten runs.

**C21 — Retriever, adaptive (D2 + D3 live).** Route selection conditioned on `attempted_routes`; query reformulation conditioned on `attempted_queries` and what came back; stop/deeper/wider decision from features + LLM judgment.
*Check:* on a question with a deliberately hard sub-question, the trace shows route escalation and a reformulated query that differs from the original.

**C22 — Budget allocation (D4).** Allocator distributing remaining budget across sub-questions by marginal expected value (distance to sufficiency × plausibility of closing it).
*Check:* on a run with one hopeless sub-question, the trace shows budget reallocated away from it.

> **Phase 2 exit:** the trace of a single run visibly shows the system changing its mind — different route on retry, reformulated query, uneven budget spend.

### Phase 3 — Verification, revision, decision (6 commits)

**C23 — Critic with entailment judgment.** Claim segmentation, per-claim verdict against cited passages, contradiction findings incorporated.
*Check:* on a hand-built draft with three good and three bad citations, all six are labeled correctly.

**C24 — Five-way critic routing.** Section 2.3 routing table; the choice among the five is itself an LLM judgment given claim, evidence, sub-question history, and budget.
*Check:* construct one input per route and verify each fires.

**C25 — Planner revision mode (D1).** Rewrite / split / merge / abandon given a failing sub-question and its history.
*Check:* on a deliberately vague sub-question, the planner rewrites it and the rewrite retrieves better.

**C26 — Sufficiency model.** Logistic regression, fitting interface, persistence, `predict_proba`. (Trained in Phase 5 — this commit is the machinery.)
*Check:* fits and predicts on synthetic data; coefficients are inspectable.

**C27 — Decision policy (D5).** Three-way policy with hard premise rule, threshold interface, gap enumeration, rationale generation.
*Check:* forced feature vectors produce each of the three decisions.

**C28 — Finalizer and full graph.** Deterministic formatting: decision banner, report or gap statement, footnotes, verification summary, sources.
*Check:* full graph runs to completion on an answerable question and on a false-premise question, producing appropriately different outputs.

> **Phase 3 exit:** the complete system runs, revises its own plan, and can refuse to answer.

### Phase 4 — Benchmark construction (4 commits)

**C29 — Tiers A–C (28 questions).** Questions plus 3–5 gold claims each.
*Check:* every question is answerable from accessible sources — verify by hand before committing.

**C30 — Tier D (20 questions).** Five per sub-type, each with the reason abstention is correct and, for false-premise items, the correction.
*Check:* each is genuinely unanswerable/ill-posed — adversarially self-review, since a Tier D question that turns out answerable poisons the headline metric.

**C31 — Splits and schema.** Stratified 24/24 across tier and sub-type, frozen in `questions.json`, split assignment committed to git so it is provably fixed before results exist.
*Check:* stratification verified; a schema validator passes.

**C32 — Eval runners.** `run_agent.py`, `run_baselines.py`, `run_ablations.py`. Resumable, per-question caching, full trace persistence, rate-limit-aware sleeping.
*Check:* interrupt a run mid-benchmark, restart, confirm it skips completed questions.

> **Phase 4 exit:** the benchmark is frozen, the test split is untouched, the runners are resumable.

### Phase 5 — Calibration, baselines, ablations (5 commits)

**C33 — Calibration split run and model fit.** Run the full agent on the 24 calibration questions, extract features, fit logistic regression, check the reliability diagram, apply isotonic scaling if needed, select τ at target risk.
*Check:* fitted model persisted; calibration-split ECE reported; coefficients interpreted in the writeup.

**C34 — Baselines B0–B3.** All four on the test split.
*Check:* four result files, same schema, all questions covered.

**C35 — Ablations A1–A5.** Each with one decision surface frozen.
*Check:* five result files; verify by trace inspection that the disabled surface really is disabled (e.g. A1 shows zero plan revisions).

**C36 — Test-split run.** The full system on the held-out 24. **Once.**
*Check:* completed with full traces retained.

**C37 — Metrics.** `analysis/score.py` computing every metric in Sections 5.5 and 6, emitting the comparison tables.
*Check:* all tables generate; spot-check 20% of LLM-judge decisions by hand and report the agreement rate.

> **Phase 5 exit:** you have numbers. Whatever they say.

### Phase 6 — Analysis and shipping (5 commits)

**C38 — MAST failure annotation.** Annotate every incorrect-outcome trace against the 14 modes; report the distribution and the tier cross-tabulation.
*Check:* every failed run has an annotation with a justification referencing specific trace events.

**C39 — Plots and case studies.** Risk–coverage curve, reliability diagram, ablation bars, retrieval R1–R4 table, MAST distribution. Four case studies: one Tier A, one Tier B/C where re-planning fired, one Tier D false premise, one honest failure.
*Check:* every figure is legible standalone with axis labels and a caption.

**C40 — Writeup.** `writeup/writeup.md`, 8–12 pages: motivation and lit positioning, agency contract, architecture, RAG layer with retrieval eval, sufficiency and calibration, results, ablations, MAST failure analysis, limitations, future work.
*Check:* someone unfamiliar with the project can read it and state the three claims from Section 1.4.

**C41 — README and demo.** README with thesis, architecture diagram, headline results table, reproduction instructions. Gradio demo showing the decision banner and expandable plan / evidence / critic / sufficiency panels.
*Check:* clean clone → install → demo runs on three questions including one that abstains.

**C42 — Test suite and cleanup.** Unit tests per node with mocked LLM, RAG component tests, one full integration test, `skipif` markers for API-key-dependent tests. Docstrings, type hints, no debug prints.
*Check:* `pytest` green; clean clone reproduces one benchmark question end to end.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| **Ablations show agency does not help** | Report it. Investigate whether the decision surfaces were too weak or the benchmark too easy. A negative result you predicted and measured is a strong interview position; an unfalsifiable claim is not. |
| **48 questions is too few for calibration** | Expected. Use simple models (logistic regression), report confidence intervals via bootstrap, be explicit about the limitation. Consider expanding Tier D if the headline metric is noisy. |
| **Gemini free tier tightens further** | Groq fallback is wired from C2. `LLMProvider` is a swappable interface. Cache everything. |
| **Tier D questions turn out answerable** | Adversarial self-review at C30, then re-review after the calibration run. Any question that gets confidently and correctly answered gets re-classified with a note. |
| **LLM-judge noise in scoring** | Manual spot-check 20%, report agreement rate, use structured outputs to reduce variance. |
| **Retrieval is the real bottleneck, not reasoning** | This is why C13 exists. The retrieval eval lets you say which it was. |
| **Scope creep** | Explicitly out: parallel execution, multi-hop query planning beyond one revision layer, fine-tuning anything, a web frontend beyond Gradio, WhatsApp (see below). |
| **The project stalls at 80%** | The commit plan is designed so every phase exit is a shippable checkpoint. Even stopping after Phase 3 leaves a working system; after Phase 1, a standalone retrieval study. |

### 9.1 On the optional messaging layer

If, after Phase 6, you still want a messaging surface, the framing is **durable execution**, not WhatsApp. LangGraph checkpoints at every node; a research run takes minutes and stalls on rate limits. So: submit a job, survive a 429 or a process death, resume from the last completed node, notify on completion. The channel is just the notification transport. Framed that way it demonstrates a production capability most portfolio projects never touch. Framed as "I added WhatsApp," it reads as a demo. Optional, last, cut without guilt.

---

## 10. Resume bullets

Draft now, fill numbers only when they exist. **Every bracketed placeholder is currently false and must not appear on a resume until it is measured.**

1. Built a multi-agent research system in LangGraph with adaptive planning, retrieval-route selection, and budget allocation, in which an evidence-sufficiency gate decides whether to answer, partially answer, or abstain — validated by ablations isolating each adaptive component.

2. Designed a hybrid RAG layer (BGE dense + BM25, reciprocal rank fusion, cross-encoder reranking, section-aware chunking of research papers, payload-filtered Qdrant retrieval) and evaluated it independently: [nDCG@10 of X for fusion+rerank vs Y for dense-only].

3. Replaced LLM verbalized confidence with a calibrated logistic sufficiency model over retrieval-derived features, [reducing expected calibration error from X to Y] and [cutting confident answers to unanswerable questions from X% to Y%] on a 48-question benchmark including 20 adversarial should-abstain items.

4. Annotated failure traces against the MAST multi-agent failure taxonomy, [attributing X% of errors to task-verification and Y% to system-design failures], and used the distribution to target architectural fixes.

---

## 11. Interview narrative

**"Walk me through the project."** Deep research agents are a solved architecture and a crowded benchmark space — every system is measured on how well it answers. Almost none are measured on whether it should have answered. AbstentionBench showed abstention is unsolved, that scale doesn't help, and that reasoning fine-tuning actively makes it worse. So I built a research system whose terminal decision is answer / partial / abstain, and I gated that decision on a calibrated score computed from retrieval evidence rather than on the model saying it's unsure.

**"How do you know it's actually agentic and not just a pipeline?"** I made that falsifiable. Five decision surfaces — plan revision, route and query selection, retrieval depth, budget allocation, and the terminal decision. For each one I built a static counterpart and ran the ablation. If freezing a decision doesn't change the numbers, that component wasn't doing anything, and I report that. [Here is what I found.]

**"Why not just prompt the model to say it doesn't know?"** That's baseline B3, and the AbstentionBench result predicts it fails: better prompting improves abstention in practice but doesn't fix the underlying inability to reason about uncertainty. My score comes from outside the model — rerank score distributions, independent source counts, cross-source contradiction rates, premise grounding. [Calibration comparison here.]

**"What did the retrieval work buy you?"** I evaluated it separately — dense only, sparse only, fusion, fusion plus rerank, on hand-labeled relevance judgments. That matters because without it I couldn't tell whether a wrong answer was a retrieval failure or a reasoning failure. Section-aware chunking was the highest-leverage small decision: a claim from a paper's Related Work section describes someone else's contribution, and a system that can't tell Related Work from Results will confidently misattribute results.

**"How does this connect to your other work?"** Project 1 characterizes when pre-filter, post-filter, and predicate-aware graph traversal each win as a function of selectivity. This project is where I hit that regime live — filtering evidence by section and publication date at low selectivity is exactly where post-filtering collapses. And Meghnad at Inxite Out uses the same decompose-execute-verify pattern for KPI extraction from call transcripts; this is that pattern in a domain where evaluation is tractable.

**"What failed?"** [MAST distribution.] The taxonomy's own finding is that better base models won't fix the full failure space — which is the argument for treating this as an architecture problem.

---

## 12. Reading list

**Read closely before Phase 2:**
- AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions — Kirichenko et al., 2025 (arXiv:2506.09038). The scenario taxonomy is the template for Tier D.
- Why Do Multi-Agent LLM Systems Fail? — Cemri et al., 2025 (arXiv:2503.13657). The 14 failure modes; read Appendix A for definitions before annotating.

**Read for positioning (skim, cite in related work):**
- DeepResearch Bench — Du et al., 2025 (arXiv:2506.11763). ICLR 2026.
- Deep Research Bench — FutureSearch. The frozen-web reproducibility idea.
- ResearcherBench, DeepScholar-Bench, ReportBench, LiveDRBench — citation-grounding and claim-recovery evaluations.
- BrowseComp-Plus — Chen et al., 2025 (arXiv:2508.06600).
- Search-time contamination in deep research agents — relevant to why your Tier D staleness questions are hard to evaluate fairly.

**Reference during implementation:**
- LangGraph docs — durable execution, checkpointers, conditional edges.
- BGE model cards — the query prefix requirement is documented and easy to miss.
- Reciprocal Rank Fusion — Cormack et al., 2009.

---

## 13. Progress tracker

| Phase | Commits | Status | Exit criterion |
|---|---|---|---|
| 0 — Foundations | C1–C5 | ☐ | Cached, traced, budgeted LLM call with fallback |
| 1 — RAG substrate | C6–C13 | ☐ | R1–R4 retrieval table generated |
| 2 — Agentic core | C14–C22 | ☐ | Trace shows route escalation + query reformulation |
| 3 — Verification & decision | C23–C28 | ☐ | System refuses a false-premise question correctly |
| 4 — Benchmark | C29–C32 | ☐ | 48 questions frozen, splits committed, runners resumable |
| 5 — Calibration & eval | C33–C37 | ☐ | All metrics computed on held-out test split |
| 6 — Analysis & shipping | C38–C42 | ☐ | Writeup, README, demo, tests green |

---

*Last updated: 2 August 2026. Amend this document whenever the design changes — including when it changes because something didn't work.*

---

## 14. Amendment log

Changes to this document after first commit. Every entry states what changed and why.

**2026-09-05 — C1.** Verified all dependency versions against PyPI before pinning.
Three corrections resulted:
1. `google-generativeai` is **deprecated** (final release 2025-12-16). Google unified
   its SDKs into `google-genai` (2.22.0). All Gemini calls use `from google import genai`.
   The migration guide is at ai.google.dev/gemini-api/docs/migrate.
2. **Python floor raised to 3.12** — numpy 2.5.2 requires >=3.12, pandas 3.0.5 and
   scikit-learn 1.9.0 require >=3.11.
3. `arxiv` is at **4.0.1**, a major version above the 3.x assumed in the original plan.
   Verify the client API surface at C6 rather than trusting 3.x examples.
   `agentdash` has not been released since 2025-08-12; treat as optional at C38.

Full resolved dependency set: 113 packages, no conflicts, on Python 3.12.

**2026-09-06 — C2. Provider architecture changed on measured evidence.**

Measured free-tier limits on this account (AI Studio + Groq console):

| Model | RPM | RPD | TPM | TPD |
|---|---|---|---|---|
| Gemini 2.5 Flash | 5 | 20 | 250K | — |
| Gemini 2.5 Flash Lite | 10 | 20 | 250K | — |
| Groq `openai/gpt-oss-120b` | 30 | 1K | 8K | 200K |
| Groq `openai/gpt-oss-20b` | 30 | 1K | 8K | 200K |
| Groq `qwen/qwen3.8-27b` | 30 | 1K | 8K | 200K |

Four consequences, all now implemented:

1. **§7.1's "Gemini primary, Groq fallback" is dead.** 20 RPD is half of one
   benchmark question. Groq is the workhorse; **Gemini is the evaluation judge
   only**. This is an improvement, not a concession: judging output with a
   different model family than produced it removes self-preference bias, which
   the original single-provider design had. Record it as a methodology choice
   in the writeup, not a constraint.

2. **`llama-3.3-70b-versatile` is retired** from Groq's catalog and not
   callable. §7.2's dependency table is superseded by `config.yaml`'s
   `llm.roles`. A scaffold test asserts it is never reintroduced.

3. **Roles replace a single model.** JUDGMENT (planner, critic, decider) →
   gpt-oss-120b → qwen3.8-27b → gpt-oss-20b. MECHANICAL (segmentation, query
   reformulation) → gpt-oss-20b → qwen3.8-27b. JUDGE → gemini-2.5-flash, no
   fallback (a judge that changes model mid-run makes scores incomparable).
   Limits are per model, so chains genuinely add capacity.

4. **TPD binds, not RPD.** At ~3K tokens/call, 200K TPD ≈ 66 calls/day/model,
   ~200/day across the chain — the 1000 RPD ceiling is never approached. A
   test asserts this empirically. Two further implications:
   - **TPM is only 8K.** One fat synthesis call can consume most of a minute's
     budget, hence per-role `max_output_tokens` caps.
   - **gpt-oss are reasoning models** and Gemini 2.5 Flash thinks by default.
     Hidden reasoning tokens bill against the same TPD ceiling, so
     `reasoning_effort` (low for mechanical, medium for judgment) and
     `thinking_budget: 0` (judge) are set explicitly. The Gemini backend adds
     `thoughts_token_count` to recorded usage — omitting it would make the
     ledger under-report and cause surprise 429s.

**Budget revision to §6 and §7.1.** The critic must batch all claims into ONE
structured call rather than one call per claim. That takes a question from
~20 LLM calls to ~9, which is what makes ~190 question-runs (calibration,
test, B1, A1–A5) feasible at ~200 calls/day: roughly 2–3 weeks of eval
wall-clock rather than four months. **This is now a hard design constraint on
C23, not an optimization.**

**New in C2, beyond the original plan:** `src/config.py` (dotted-path loader
with non-mutating overrides — the mechanism ablations use) and
`src/llm/rate_limit.py` (persistent four-dimensional quota ledger). The ledger
survives restarts deliberately: a resumed eval that forgets yesterday's spend
would burn quota it does not have.

**2026-09-06 — C2 addendum. Model availability probed; allocation revised.**

Two failure modes discovered that are worth carrying forward as method:

1. **Listed ≠ callable.** `gemini-2.5-flash` appears on AI Studio's
   rate-limit page with 5 RPM / 20 RPD *and* in `models.list()` with a 1M
   context window, yet `generateContent` returns 404 "no longer available to
   new users". Neither surface is authoritative. `scripts/probe_gemini.py`
   makes a real call and is the only check that counts.

2. **200 ≠ usable.** `gemini-3.7-flash` and `gemini-3.8-flash` returned HTTP
   200 with **empty text**, having spent the whole output budget on hidden
   thinking — they ignore `thinking_budget: 0`. The probe now treats an
   empty-text 200 as a failure. Any model added to `llm.roles` must be shown
   to return non-empty text, not merely to respond.

**Verified callable (2026-09-06):**

| Model | RPM | TPM | RPD | Status |
|---|---|---|---|---|
| `gemini-3.1-flash-lite` | 15 | 250K | **500** | text OK |
| `gemini-3.5-flash` | 5 | 250K | 20 | text OK |
| `gemini-3.7/3.8-flash` | 5 | 250K | 20 | empty text — unusable |
| `gemini-3.6-flash`, `3.5-flash-lite` | — | — | — | 400 INVALID_ARGUMENT |
| `gemini-2.5-flash`, `2.5-flash-lite` | — | — | — | 404 retired |

**Allocation now:** JUDGMENT = Groq only (gpt-oss-120b → qwen3.8-27b →
gpt-oss-20b). MECHANICAL = gpt-oss-20b → gemini-3.1-flash-lite. JUDGE =
gemini-3.1-flash-lite, pinned, no fallback.

Judge choice is driven by **volume, not quality**. Scoring the test split plus
B0–B3 plus A1–A5 is ~10 result files × 24 questions ≈ **240 judge calls**. At
20 RPD that is twelve days; at 500 RPD it is one. Flash-Lite is the weaker
model and that is a real limitation — the mitigation is the 20% manual
spot-check already specified in §6.4/C37, which becomes load-bearing rather
than a formality. State this trade-off explicitly in the writeup.

Judgment stays wholly on Groq so the model family that writes the report is
never the family that grades it. A scaffold test enforces the separation.

**Unexplored upside:** `gemma-4-31b-it` and `gemma-4-26b-a4b-it` show
30 RPM / 16K TPM / **14.4K RPD** — an order of magnitude more daily calls
than anything else on the account. Not yet probed. If they work, the eval
budget stops being a constraint at all. Gemma on the Gemini API has
historically ignored `system_instruction`, so verify output quality, not just
a 200 response.

**2026-09-06 — C2 final. Gemma probed; roles reallocated; budget crisis over.**

`gemma-4-31b-it` and `gemma-4-26b-a4b-it` are callable at **30 RPM / 16K TPM
/ 14.4K RPD** — roughly 30x the daily allowance of anything else on the
account. The earlier 400 was our own bug: they reject `thinking_budget`
outright ("Thinking budget is not supported"), so the probe now retries
without it.

Two Gemma properties that must be respected in config, both enforced by
scaffold tests:

- **Thinking cannot be disabled and is expensive.** A one-word reply cost 52
  tokens, 43 of them hidden reasoning — a ~5x overhead. `max_output_tokens`
  is raised to 2048 accordingly; a 1024 cap would leave little for the answer.
  With 16K TPM this means ~5 calls/minute sustained, which is the real
  ceiling for this model, not its 14.4K RPD.
- **No system role.** `supports_system_instruction: false`; the provider
  folds the system prompt into the user content. Left at the default the
  system prompt would be silently dropped — a quality regression with no
  error, which is the worst kind.

`supports_json_mode: false` for Gemma until proven otherwise; the structured
layer's local repair covers fenced and damaged output without it.

**Final allocation:**

| Role | Chain | Rationale |
|---|---|---|
| judgment | gpt-oss-120b → qwen3.8-27b → gpt-oss-20b (Groq) | Quality. Now the *only* consumer of Groq's 600K TPD. |
| mechanical | gemma-4-31b-it → gemini-3.1-flash-lite → gpt-oss-20b | Volume. 14.4K RPD absorbs the high-count, low-judgment work. |
| judge | gemini-3.1-flash-lite (pinned, no fallback) | Quality over capacity — see below. |

Judge stays Flash-Lite rather than moving to Gemma's larger allowance:
entailment judging is the one place in the eval where model quality directly
moves the reported numbers, and 500 RPD already covers the ~240 judge calls
in a single day. Capacity is not the binding constraint for that role.

**Effect on §7.1's budget arithmetic.** Moving mechanical work off Groq frees
the entire 600K TPD for judgment, and Gemma absorbs the volume. Eval is no
longer quota-bound in any meaningful sense — the earlier estimate of
2–3 weeks of wall-clock collapses to days. **The C23 constraint that the
critic batch all claims into one call still stands**: it is now good design
rather than a survival requirement, and batched judging is also less noisy
than per-claim calls.

**2026-09-06 — C3. Prompt-hash cache.**

`src/llm/cache.py`, wired into the provider. sha256 over model, system, user,
temperature, max_output_tokens and json_mode; sharded on the first two hex
chars; atomic writes; nothing ever evicted.

One design decision worth recording because the naive version is subtly
wrong: **the cache is checked across the entire role chain before any live
call**, not per model just before calling it. If run 1 fell back to the
secondary model, per-model checking would miss on the primary during a
re-run, spend real quota there, and only then reach the cached secondary
entry. Chain-wide lookup makes "re-running an unchanged question costs zero"
true regardless of which model originally answered.

Three invariants, each covered by a test:
- **A cache hit charges no quota.** It never reaches `ledger.record`. Charging
  for it would over-report spend and could falsely exhaust a model.
- **Empty responses are never cached.** Caching a failure would poison every
  future run of that prompt permanently.
- **Corrupt or stale-version entries degrade to a miss**, never to a crash.

`use_cache=False` bypasses read and write both. This is what C37 needs: the
scoring pass re-runs the judge blind, and a cached verdict would defeat the
point. `evaluation.judge.cache_enabled: false` in config.yaml already
anticipated this.

Note a consequence of keying on temperature: **calls with temperature > 0 are
frozen after their first execution.** That is intended — it makes reported
numbers reproducible — but any deliberate measurement of sampling variance
must pass `use_cache=False`.

Also fixed here: `.gitignore` now excludes all of `data/`.
`data/quota_ledger.json` was committed at C2, which is wrong — it records what
one machine spent today, so a fresh clone would start life believing it had
already consumed someone else's quota, and every pull would conflict.

**2026-09-06 — C4. Structured tracing.**

`src/tracing/tracer.py`: JSONL, one file per run named by `trace_id`,
append-only, flushed per event so a run killed mid-eval keeps what it did.
`TraceReader` skips unparseable lines, so a partial final line from a killed
process costs one event rather than the file.

**The design point beyond the original spec.** §8 framed C4 as MAST
substrate. It is also the *verification mechanism for the agency claim*.
C35's check reads "verify by trace inspection that the disabled surface
really is disabled — e.g. A1 shows zero plan revisions", and that is only
possible if every exercise of D1–D5 is a first-class, queryable event. So the
tracer has an explicit `DecisionSurface` enum and a `decision()` method
recording the chosen option, **the alternatives considered**, the rationale,
and the inputs.

Critically, `decision(..., was_adaptive=False)` marks a call made under a
static policy. `TraceReader.adaptive_decision_counts()` counts only adaptive
ones, so an A1 run must report `D1_plan_revision: 0`. **A non-zero count means
the ablation did not take effect and its result is invalid** — without this,
a broken ablation would silently produce a plausible-looking number.

Recording alternatives also matters for §2.4's anti-patterns: a decision with
an empty alternatives list every time is a decision in name only, and now
that is visible in the data rather than a matter of opinion.

Payloads pass through a truncator (2000 chars, 25 list items, depth 6).
§3.3 asks routing events to carry "the state snapshot that produced it";
dumping full state is not viable when evidence passages run to kilobytes and
a run makes tens of routing decisions, so `route()` takes a `state_summary` —
the fields routing actually reads. A trace too large to read is a trace
nobody reads.

Also wired: `LLMProvider` takes an optional `tracer` and records every call,
**including cache hits**. Omitting cached calls would make cache
effectiveness invisible in the per-run cost analysis §6.7 requires.

`scripts/smoke_c4.py` is the acceptance check and costs zero API calls — it
simulates a four-node run including a route escalation, a static (ablated)
decision, and a node failure, then asserts the reader recovers all of it.

**2026-09-06 — C5. Budget manager. Phase 0 complete.**

`src/budget.py`. Smaller than §8 planned, because `src/config.py` already
landed at C2; this commit is budget only.

**Budget is not quota, and the distinction is load-bearing.**
`llm/rate_limit.py` tracks what this machine may spend against the providers
today — an external constraint, persisted across runs. `budget.py` tracks
what a *single question* may consume before the system must stop researching
and decide — an internal constraint we impose on ourselves. The second exists
because **D4 is only a real decision surface if the budget can run out.** An
agent that never faces scarcity is not allocating, it is just spending, and
the A4 ablation would be measuring nothing.

**The reserve.** `reserve_fraction` (20%) of the LLM budget is withheld from
allocation. The critic and decider run *after* retrieval; a system that
spends everything researching and then cannot afford to verify or decide has
failed in the most embarrassing way available — it did all the work and
produced nothing. `allocatable()` excludes the reserve; `in_reserve()` is
what the controller reads to stop retrieving and start verifying.

**Two exhaustion paths, deliberately.** `can_spend()` is the graceful path
the controller consults to route to the decider. `spend()` raises
`BudgetExhausted` and should never fire in correct operation — if it does, a
node bypassed the controller's check, and that is a bug worth hearing loudly
rather than a run that silently overruns.

**Allocation sums exactly to the pool.** Plain integer division would discard
up to n-1 calls; at a 25-call budget with 5 sub-questions that is a material
fraction silently lost. `_distribute` hands out the remainder one call at a
time, and under scarcity funds as many sub-questions as it can at the floor
rather than starving all of them equally.

`UniformAllocator` is the static policy and **is literally ablation A4**. It
logs its decision with `was_adaptive=False`, so an A4 run's trace shows zero
adaptive D4 decisions and C35's check works from the trace alone. The
adaptive allocator arrives at C22 and must beat it, or D4 was decorative.
`get_allocator()` currently returns uniform in both branches — stated
plainly in a comment rather than pretending D4 is already live.

**Phase 0 exit criterion met:** `scripts/smoke_c5.py` makes a cached, traced,
budgeted LLM call with automatic fallback, and verifies each of those four
properties independently.

---

## Phase 1 — RAG substrate

**2026-09-06 — C6. arXiv tool.**

`src/tools/arxiv_tool.py`, plus `src/state.py` pulled forward from C14 (only
`Passage`, `make_passage_id`, and the shared Literals — C6 cannot build
passages without the schema, and defining it here avoids a circular
dependency between the tools and the graph).

**arxiv 4.x API differences from the 3.x the plan assumed.** Verified by
introspecting the installed package rather than trusting examples:
- `Result.download_pdf()` **no longer exists**. `Result.pdf_url` is still
  present (derived from the result's links), and C8 will fetch it directly —
  which is what content-hash caching wanted anyway.
- `Client(page_size, delay_seconds, num_retries)` and `Client.results(search)`
  are the current surface; `Search.results()` is gone.
- `SortCriterion` has exactly Relevance, LastUpdatedDate, SubmittedDate.

**Failure is a result, not an exception.** `search()` returns `[]` on any
error and records the failure in the trace. This is a deliberate contract:
a retrieval failure is *information the agent acts on* — it is what tells D2
to escalate to another route and D3 that a sub-question may be unresolvable.
An exception propagating into the graph would turn a normal, informative
outcome into a crashed run. One malformed feed entry is skipped rather than
discarding the whole page. The trace distinguishes "returned zero results"
from "failed with an error", which C38 needs to annotate retrieval failures
correctly against MAST.

**Passage ids are content-addressed, not sequential.** §3.2 sketched
"P0001 …"; `make_passage_id` hashes (source_id, section, text) instead. The
evidence store accumulates across runs (§4.1), so a per-run counter would
collide between runs and re-ingesting a paper would duplicate it. Section is
part of the hash because the same sentence in Results and in Related Work is
not the same evidence.

The cost: hex ids are error-prone for an LLM to transcribe, and C18 requires
the synthesizer to reproduce them exactly. **That is solved at C18 with a
per-prompt label map (P01…P20 → id), not by weakening the storage id.**

**Abstracts carry `section="Abstract"`.** A claim sourced from an abstract is
a summary claim, not a measured result; labelling it now means the critic can
weigh it differently without re-deriving where it came from.

`scripts/smoke_c6.py` is the acceptance check: three real queries, a
malformed query, an empty query, and an id-stability re-run. ~20s because
arXiv's 3-second inter-request delay is honoured — do not lower it to speed
up a benchmark, since a blocked IP costs far more than the time saved.

**2026-09-06 — C7. Web search with a provider interface.**

`src/tools/web_search.py`: `SearchProvider` ABC, Tavily and DuckDuckGo
implementations, a `WebSearchTool` chain, and a factory that skips providers
whose credentials are absent rather than treating that as an error.

**The two providers are not interchangeable, and the interface says so.**
Tavily returns pre-extracted page content; ddgs returns snippets. That
distinction is recorded as `returns_full_content` because it decides what the
provider is *for*: a critic cannot verify a claim against forty words of
search-result teaser, so ddgs is a development tool and Tavily produces the
evidence the eval judges. Where both are available, Tavily's `raw_content` is
preferred over its own extraction.

**Tavily credits are policed by the existing QuotaLedger.** 1000/month free,
one per basic search; the full eval needs ~250 and careless development could
spend the lot in an afternoon. Rather than invent a second accounting
mechanism, Tavily is registered as a pseudo-model with a daily cap
(`tavily_credits_per_day: 33`) in `data/search_credits.json`. Overrunning then
costs a graceful fallback to ddgs instead of a dead key two weeks before the
eval. `search_depth` stays `basic` in config because `advanced` costs two
credits and defaulting to it would halve the allowance silently.

**Same failure contract as C6.** A rate-limited, credit-exhausted or broken
provider falls through to the next; only exhausting the chain yields `[]`,
and the trace records the provider that answered, whether it fell back, and
the accumulated errors if none did. C38 needs to distinguish "found nothing"
from "everything was broken".

**Results shorter than 80 characters are dropped.** A fragment is not
evidence and would only dilute the reranker's candidate pool at C11.

**`source_domain` strips `www.`** so two pages from one site count as one
independent source in feature f4 (§5.2). Counting them as two would inflate
the sufficiency score exactly when the evidence is weakest — the failure mode
the whole abstention layer exists to prevent.

**2026-09-06 — C8. PDF ingestion and section-aware chunking.**

`src/rag/chunking.py` and `src/rag/ingest.py`.

**Section canonicalisation, not just detection.** §4.2 specified detecting
headers; that is not sufficient on its own. "2.1 RELATED WORK", "Prior Work"
and "Related work" are one section, and unless they map to a single canonical
label the retriever's section filter (§4.4) matches none of them reliably.
`SECTION_ALIASES` handles the variants seen in practice. An *unrecognised*
heading still splits the document but gets `section=None` — it is a real
boundary, and inventing a label would let the retriever filter on a category
that means nothing.

**Two additions not in the plan, both defensible:**

*Bibliographies are dropped.* A References chunk is never evidence, but it is
dense with exactly the high-IDF tokens (author names, paper titles, venues)
that make BM25 rank it highly. Retaining it would have quietly degraded
sparse retrieval at C10 in a way that is very hard to notice.

*Hyphenation is repaired first.* pypdf preserves the line breaks of a
justified two-column layout, so "atten-\ntion" arrives as two fragments.
Dense retrieval degrades quietly; **BM25 breaks outright**, because
"attention" stops being a token in a paper entirely about attention.

**Deduplication at ingestion.** Content-addressed ids mean byte-identical
chunks share an id. Returning them all would let ONE piece of evidence count
repeatedly toward f2 (mean of top-3 rerank scores) and f4 (independent source
count) — inflating sufficiency exactly when evidence is thinnest, which is
the precise failure the abstention layer exists to prevent. Found by a test;
the dedup now lives in `ingest`.

**Failure handling, consistent with C6/C7.** A download that is not a PDF (an
HTML error page) is rejected rather than cached; an unparseable page is
skipped rather than failing the document; extraction yielding under 500
characters returns nothing, since that means a scanned PDF with no text layer
and OCR is out of scope. Reporting nothing beats feeding the reranker noise.

**Caching is cross-run and idempotent** (§4.1): a paper downloaded for
question 7 is free for question 23. That is not only speed — it is what makes
the `evidence_store` route in D2 meaningful. Downloads are atomic (temp file
plus rename) with a content-hash sidecar.

Minor fix: `sliding_window` validated its parameters *after* an early return,
so an invalid overlap/size config passed silently until the first long
document arrived. Validation now runs first.

**2026-09-06 — C9. Embeddings and Qdrant store.**

`src/rag/embed.py` and `src/rag/store.py`.

**The prefix is not a parameter.** `bge-small-en-v1.5` is asymmetric: queries
take the instruction prefix, documents must not. Getting it wrong in *either*
direction raises nothing — recall simply drops, and the natural conclusion is
that the reranker or the chunking is at fault. This is the most expensive bug
this project could carry, because C13's retrieval evaluation would faithfully
measure a crippled system and report the number as a finding. So `embed_query`
applies the prefix and `embed_documents` does not, as separate methods, so the
choice cannot be made by accident at a call site. A blank prefix in config
raises at construction. `smoke_c9.py` measures the effect directly on the real
model, since no unit test can catch a silent quality loss.

**Point ids are uuid5 of the passage id.** Qdrant requires ints or UUIDs;
passage ids are strings like `Pa3f9c2b1d004`. The mapping is deterministic, so
re-ingesting a passage overwrites rather than duplicates — the idempotency
established at C8 survives into storage. The string id stays in the payload
and remains what the rest of the system cites.

**API note:** qdrant-client 1.19 has removed `search()`. `query_points()` is
the current entry point and returns a response object with `.points`.

**Local Qdrant ignores payload indexes** and warns on every call. Filtering
itself works correctly without them — the store tests verify this against a
real local client, not a mock — an index only makes it faster. The indexes are
still declared so that moving to a server deployment needs no code change, and
the notice is suppressed rather than printed on every ingestion.

**Store tests run against real Qdrant in local mode**, with a fake embedder
supplying deterministic vectors. Mocking Qdrant would have tested our belief
about how payload filtering behaves rather than how it behaves, and filtered
retrieval is the entire reason Qdrant was chosen over Chroma (§4.3).

Filters implemented: section (one or many), source_type, source_domain,
published_after, and exclude_sections. That last one is the misattribution
guard from C8 made queryable — a claim from Related Work describes someone
else's contribution, and the retriever can now say so.

**2026-09-06 — C10. BM25 and RRF fusion.**

`src/rag/sparse.py` and `src/rag/fusion.py`. `HybridRetriever.retrieve()` is
the unified entry point; nothing above this layer knows two retrievers exist.

**The tokeniser default would have cost us version numbers.** bm25s defaults
to `(?u)\b\w\w+\b`, requiring two or more word characters — which silently
discards the "2" in FlashAttention-2 and the "3" in Llama 3. Those are
precisely the tokens a research question turns on. Widened to `(?u)\b\w+\b`.
This is the same class of silent-degradation bug as the bge query prefix:
nothing errors, retrieval is just quietly worse.

**Titles are indexed with the body.** A paper's own name is the highest-signal
exact match available when a sub-question names a method or model.

**RRF over score blending.** Cosine lives in [-1,1] and clusters at 0.5-0.9;
BM25 is unbounded and corpus-dependent, routinely 2-30. Blending them by
weighted sum means normalising two distributions whose shapes differ per
query. RRF uses rank, not score: parameter-free, robust to either arm
producing garbage. The cost is that magnitude is discarded — a first place won
by a mile scores the same as one won by a hair — and **C13's R1-R4 table is
what tests whether that trade was right on this corpus.**

**The asymmetry worth stating in the writeup.** Qdrant applies payload filters
*during* HNSW traversal; BM25 has no equivalent and must post-filter with
oversampling. So a low-selectivity filter degrades the sparse arm
specifically. That is Project 1's regime appearing inside a live system, and
it is the honest version of the bridge between the two projects — not "I used
filtering in both", but "here is where post-filtering measurably hurts and
why the dense arm does not have that problem".

The two filter implementations must agree exactly: a mismatch would mean the
arms search different corpora and fusion silently favours whichever has the
looser filter. Both exclude passages with a missing `published` field under a
date constraint, because Qdrant cannot match a missing field either.

**`dense_only` / `sparse_only` are built in now**, not bolted on later — C13's
R1 and R2 configurations need each arm in isolation, and a parallel code path
in the evaluation harness would risk measuring something subtly different from
what the system actually runs.

**2026-09-06 — C11. Cross-encoder reranking.**

`src/rag/rerank.py`. `bge-reranker-base` over the top 30 fused candidates,
keeping 5-10.

**The decision that matters: raw logits are persisted, not probabilities.**
`CrossEncoder.predict()` returns unbounded logits (verified against the
library's own documented example: 8.6, 5.5, 6.35). A sigmoid would map them to
[0,1] and read more nicely. It would also be wrong to store, because sigmoid
**saturates**, and feature f3 (§5.2) is precisely a score *gap*.

Concretely: logits of 9.0 and 7.5 differ by 1.5, a clear difference in
evidence strength. In probability space both map to ~0.999 and the gap becomes
0.0005 — the signal is destroyed exactly where evidence is strongest and the
sharp-peak-versus-flat-mush distinction matters most. `Passage.rerank_score`
therefore holds the logit; the sigmoid is derivable from it, the logit is not
recoverable from a saturated sigmoid. A test asserts this property directly so
nobody "tidies" it later.

**f1-f3 are computed here, not at C20**, because their semantics live with the
scores: they are logit-space quantities and computing them from sigmoids would
silently flatten f3. `f3_score_gap` returns **None** with fewer than five
results rather than 0.0 — None is honest, whereas zero reads as "no gap",
which is a strong claim about ambiguity the data does not license. That
distinction matters when the sufficiency model is fit at C33.

**Rerank failure degrades to fusion order** rather than raising. A worse
ordering is recoverable; a crashed run mid-eval is not.

**Pre-rerank position is preserved** on every result (`dense_rank` /
`sparse_rank` survive `replace`), so C13 can measure what reranking actually
changed rather than assuming R4 ≥ R3.

The smoke test's hand-checked example is built adversarially: the correct
passage answers the question *without using its words*, while four distractors
repeat the query's wording without answering it. Lexical retrieval buries the
right answer; if the cross-encoder cannot rescue it, it is not earning the
latency it costs.

**2026-09-06 — C12. Contradiction detection.**

`src/rag/contradiction.py`. Pairwise NLI over the top reranked passages,
producing `ContradictionRecord`s for `state["contradictions"]` and feature f5.

**Label indices are resolved from the model, never hardcoded.** NLI heads
differ in label order between checkpoints. Assuming index 0 is "contradiction"
is a silent correctness bug of the worst kind: it produces plausible-looking
numbers that point the *wrong way*, with no error anywhere, and f5 would then
push the sufficiency model toward abstaining on agreement and answering on
disagreement. The detector reads `config.id2label` and only falls back to the
documented order if unavailable — and the smoke test prints what it resolved,
because no unit test can verify the real checkpoint's ordering.

**Only cross-source pairs count toward f5.** Two chunks of the *same* paper
appearing to contradict each other is usually an artefact — a hypothesis
stated before it is refuted, a limitation acknowledged after a result, a "one
might expect X" setup. That is not sources disagreeing, and counting it would
inflate f5 on a corpus that actually agrees. Same-source pairs are still
detected and recorded (`cross_source: False`) but excluded from the rate.

**Both directions are scored.** Contradiction is symmetric in principle but
NLI heads are asymmetric in practice; missing a real disagreement costs more
than the extra forward pass.

**Known limitation, stated rather than hidden:** NLI models are trained on
sentence pairs, not 1000-character chunks. Passages are truncated to a leading
400-character claim-bearing window before scoring. Sentence-level pairing
would be more faithful but multiplies cost by roughly the square of sentences
per chunk. **This belongs in the writeup's limitations section** — the method
is a reasonable proxy for cross-source disagreement, not a rigorous
entailment analysis.

Records carry both passages' sections, so downstream a conflict involving a
Related Work chunk can be weighted differently: that chunk describes someone
else's claim, so the "contradiction" may be a competing method rather than a
genuine dispute.

`contradiction_rate()` returns 0.0 when there are too few passages to form a
pair. That is the honest reading — no disagreement was *observed*, which is
different from evidence agreeing — and thin evidence is already captured by
f1, f2 and f4.

**2026-09-06 — C13. Retrieval evaluation harness. Phase 1 complete.**

`analysis/retrieval_eval.py` (metrics + R1-R4 evaluator),
`scripts/retrieval_labels.py` (build / label / stats / evaluate),
`benchmark/retrieval_questions.json` (30 sub-questions, 8 seed papers → ~600
pooled judgments).

**Three protocol decisions that determine whether the numbers mean anything:**

*Pooling.* Candidates are the union of the top-20 from all four
configurations. No system gets its results labelled more thoroughly than
another, which is what makes "unjudged = irrelevant" a fair assumption rather
than a bias toward whichever system was labelled first. Consequence to
remember: **adding a fifth configuration later and scoring it against this
pool would be invalid** without re-pooling.

*Blind.* The labelling view shows the passage and nothing else — no rank, no
system, no score. Rank is the strongest anchor available; shown "rank 1 of the
reranker", a labeller agrees with it.

*Shuffled with a fixed seed.* Pool order within a query is randomised so
labelling fatigue late in a session does not systematically penalise one
system.

**Metric choices.** nDCG uses exponential gain (`2^rel - 1`), so one fully
relevant passage is worth three partial ones — which matches how evidence
works here. The ideal ranking is computed over **all** judgments for a query,
not just the retrieved ones; normalising against what was retrieved would
score a system that found one relevant passage as perfectly as one that found
ten. Recall is **pool-relative** and must be reported as such: the denominator
is the judged-relevant set, not all relevant passages in the corpus. MRR is
reported alongside because the agent consumes the top few passages, and nDCG@10
partly hides the difference between rank 1 and rank 8.

**Queries with no relevant passage are excluded**, not scored as zero. They
cannot discriminate between systems, so including them drags every score
toward zero equally and dilutes the comparison without adding information.

**`check_expected_ordering` implements §8's exit criterion literally**,
including its second clause. R4 ≥ R3 ≥ max(R1, R2) is a *hypothesis under
test*, not a requirement — if fusion or reranking fails to help on this
corpus, the runner says so plainly and names the likely causes to check
(too few queries, the bge prefix, a corpus small enough that every system
finds everything). That is the honest form of the C10/C11 design bets.

**Phase 1 exit criterion met.** The retrieval system is measured rather than
assumed, and the R1-R4 table is a self-contained study. Its real payoff comes
at C37: when the full system gets a question wrong, this is what lets the
writeup say whether it was a retrieval failure or a reasoning failure.
Remaining work before the table has numbers is labelling, which is manual and
takes an afternoon.

---

## Phase 2 — agentic core

**2026-09-06 — C14. State definitions.**

`src/state.py` completed. `Passage` and the shared literals landed early at
C6 because the retrieval tools needed them before the graph existed; C14 adds
`SubQuestion`, `ClaimVerification`, `BudgetState`, `ResearchState`, the
reducers and the helpers.

**Custom reducers, not `operator.add`.** §3.2 specified
`Annotated[list[Passage], add]`. Plain concatenation is wrong here, and the
reason matters: re-retrieval legitimately returns passages already in state,
and content-addressed ids mean a repeat id is byte-identical text. Appending
it twice lets ONE piece of evidence count twice toward f2 (mean of top-3
rerank scores) and f4 (independent source count) — inflating sufficiency
exactly when evidence is thinnest, which is the precise failure the whole
abstention layer exists to prevent. The same bug was caught at ingestion in
C8; the reducer is where it would reappear.

`merge_evidence` also keeps the **first** copy rather than the last. A
passage retains the scores from the round that first found it; overwriting
would let one sub-question's rerank score silently replace another's, and
feature extraction reads those scores.

`merge_contradictions` deduplicates by *unordered* pair. Contradiction is
symmetric, so storing (A,B) and (B,A) would double f5's numerator while the
denominator counts each pair once.

**`total=False` on ResearchState** so nodes return only the keys they own.
`new_state()` populates every field, so a node reading state never meets a
missing key, but a node writing state never has to echo the rest back — which
would make every node a merge conflict.

**The fields that exist to enable a decision surface** (§3.2) are documented
in the module docstring with the surface each serves. That gives the C35
ablation check a cheap precursor: *a field nobody reads is a decision nobody
makes*, and that can be spotted by grep long before an ablation is run.

**Acceptance (C14's criterion) met and exceeded.** State round-trips through
`SqliteSaver` on LangGraph 1.2, verified against the real library rather than
a mock. `smoke_c14.py` additionally demonstrates recovery in a **fresh
process**, which is what C19 and C32 depend on: an eval spanning days has to
survive a 429, a laptop sleep or a crash by resuming from the last completed
node rather than restarting.

API note: LangGraph 1.2's `StateGraph(state_schema)` and
`SqliteSaver.from_conn_string()` (a context manager) are current;
`langgraph.prebuilt` remains deprecated and unused — we stay on the
low-level API throughout, per §3.4.

**2026-09-06 — C15. Premise checker.**

`src/nodes/premise_checker.py`, plus `src/prompts/` with a versioned prompt
loader and the two templates this node uses.

**Three verdicts, not two, and the distinction is the whole design.**
§5.4 says a failed premise check abstains, bypassing τ. That is only safe if
"failed" means the corpus *positively refutes* the presupposition:

| verdict | meaning | fails the check |
|---|---|---|
| SUPPORTED | the corpus confirms it | no |
| CONTRADICTED | the corpus says otherwise | **yes** |
| UNVERIFIABLE | the corpus is silent | no |

Collapsing UNVERIFIABLE into CONTRADICTED would refuse every question whose
presuppositions the corpus happens not to restate — which is most of them,
since a paper does not restate its own premises. The false-abstention rate on
Tiers A-C (§5.5) would be catastrophic, and the project would report an
excellent false-answer rate achieved by answering almost nothing. The
verification prompt states the distinction explicitly and sends ties to
UNVERIFIABLE; an unparseable verdict also degrades to UNVERIFIABLE, so a
parse error can never cause an abstention.

**f7 scores UNVERIFIABLE at 0.5, not 0.** Silence is genuinely uninformative.
Scoring it as failure would make f7 a proxy for corpus coverage rather than
for whether the question is well-posed.

**Core vs incidental.** Only core presuppositions can fail the check or
contribute to f7. "Mamba-2 was published in 2024" being wrong does not make
"why does Mamba-2 use RoPE?" unanswerable; "Mamba-2 uses RoPE" being wrong
does.

**Two LLM calls per question, regardless of presupposition count** —
one to extract, one to verify them all together. Per-presupposition
verification would be three to five calls against a budget of roughly nine
(§7.1), making the guard cost more than the research it guards. Same batching
requirement C23 has for the critic.

**Prompts are versioned .txt files** (§7.3). A prompt change becomes a
readable diff, the writeup can quote the exact text behind the reported
numbers, and the C3 prompt-hash cache key changes when the file does — which
is correct, since a different prompt is a different experiment.

**Caught by a test:** with every presupposition UNVERIFIABLE, grounding sits
at 0.5, above the 0.35 floor, and the node originally reported
"presuppositions grounded". Nothing had been verified; it merely had not been
refuted. The rationale is surfaced in the decision explanation and quoted in
case studies, so it now distinguishes confirmed from unrefuted explicitly.

**Failure degrades, never propagates.** Extraction, retrieval or verification
failing yields `ran=False` with no abstention: the hard rule is lost but the
calibrated threshold still governs the decision. A premise checker that
crashes a run would be worse than no premise checker.

`smoke_c15.py` tests both directions. The valid-question half matters more:
a checker that fails everything scores perfectly on false premises and
quietly ruins the headline metric.

**2026-09-06 — C15 live result. A constraint on Tier D1 (affects C30).**

The smoke test caught 2 of 3 false premises. The miss is the more useful
finding, and it is **correct behaviour** rather than a bug.

| question | verdict | f7 |
|---|---|---|
| Why does Mamba-2 use **rotary position embeddings** in its SSM blocks? | UNVERIFIABLE | 0.75 |
| How does QLoRA **avoid quantising** the base model weights? | CONTRADICTED | 0.67 |
| Why does FlashAttention-2 **increase** memory usage? | CONTRADICTED | 0.00 |

The QLoRA and FlashAttention papers positively state the opposite of their
false premise, so the corpus refutes them. The Mamba paper simply never
mentions rotary embeddings — it describes a selective state space mechanism
and says nothing about RoPE either way. Under the three-verdict rule that is
UNVERIFIABLE, and the hard abstention rule correctly does not fire.

**This is the distinction earning its place.** The alternative — treating
silence as refutation — would have "caught" all three and would equally have
refused any well-posed question whose premises the corpus does not restate.
The valid-question half of the smoke test passed with f7 = 1.00 on all three,
which is the evidence that the strictness is not costing coverage.

**Two consequences:**

1. **f7 still carries the signal.** 0.75 for the Mamba case against 1.00 for
   every valid question. The hard rule is a shortcut for the clear cases; the
   calibrated score is what must catch the rest. This is the architecture
   working as designed — §5.4's hard rule bypasses τ, it does not replace it.

2. **C30 must account for this when writing Tier D1.** A false-premise
   question whose falsehood the corpus only *fails to confirm* will not
   trigger the hard rule, and grading it as "should have abstained via
   premise check" would be grading the benchmark against a mechanism that
   cannot fire. When curating the five D1 items, record for each whether the
   corpus **refutes** the premise or is merely **silent** on it, and expect
   the refuted ones to abstain via the hard rule and the silent ones via the
   calibrated threshold. Both are correct abstentions; they are not the same
   measurement, and §6.2 already requires recording *the reason* abstention
   is correct.

**Possible future feature, deliberately not built now.** An UNVERIFIABLE
presupposition about a subject the corpus covers *thoroughly* is weak
evidence of absence — if twenty passages describe Mamba-2's architecture and
none mentions RoPE, silence becomes informative. That is a coverage-
conditioned absence signal and would be a genuine addition to §5.2. It is
scope creep for Phase 2; revisit only if C33 shows f7 underperforming.

**Cost measured:** exactly 2.0 LLM calls per question over 7 questions, and
2.6 retrieval calls. Batched verification holds. Wall-clock varied from 6.5s
to 57s per check, dominated by CPU reranking rather than the LLM — worth
watching at C32, since 48 questions is roughly 20-45 minutes of premise
checking alone.

**2026-09-06 — C16. Planner, initial mode.**

`src/nodes/planner.py`, `src/prompts/planner_initial.txt`, plus a
`planning.max_sub_questions` config key.

**`rationale` is load-bearing, not documentation.** When a sub-question fails
to retrieve, C25 must choose between rewriting it, splitting it, merging it
with a sibling, or abandoning it as unanswerable. That choice depends on what
the sub-question was *for*: one that exists to pin down a number can be
rephrased, one that covers a whole topic should be split, one already covered
by a sibling should be merged. Without the rationale the reviser sees only
that retrieval failed, every failure looks identical, four options collapse
into "try again", and **D1 becomes a retry loop** — §2.4's second
anti-pattern, arrived at by omission rather than design.

**The planner's route is a suggestion, stored as `suggested_route`, and
deliberately NOT written into `attempted_routes`.** Recording it as attempted
would tell D2 the route had already been tried and failed, pushing the
retriever to escalate on its first move. Leaving the planner's guess
non-binding is what gives D2 something to decide (§2.2).

**Initial decomposition is traced with `was_adaptive=False`.** It happens in
every configuration including the A1 ablation that freezes revision, so
counting it as adaptive would make an A1 run report non-zero D1 activity and
fail C35's check for the wrong reason. Only C25's revisions count as adaptive
D1.

**Premise-check results feed the prompt.** A confirmed presupposition does
not need its own sub-question — re-researching it wastes budget. A *refuted*
one means the plan should investigate the correction rather than elaborate
the fiction, which is the difference between a useful abstention and a
well-researched answer to a question that does not make sense.

**Failure is fatal here, unlike C15.** The premise checker degrades to "no
premise information" because the calibrated threshold still works without it.
A planner failure propagates: there is nothing to research without a plan,
and silently inventing one would hide the failure from the trace. An empty or
all-fragment decomposition does fall back to researching the question
verbatim — a question the planner could not split is still a question.

**Prompt tests now introspect.** `tests/test_prompts.py` discovers each
template's placeholders with `string.Formatter` rather than a hand-maintained
field list, after the C16 prompt broke a hard-coded one. Undoubled braces in
a JSON example otherwise raise at call time, mid-run, after retrieval has
already been paid for.

**2026-09-06 — C17. Retriever, static version. D2 and D3 deliberately off.**

`src/nodes/retriever.py`.

**This file is the A2/A3 baseline, not a placeholder.** §6.5 restores exactly
this policy for those ablations, so it has to be a *fair* opponent. A
deliberately weak baseline would make C21's adaptive retriever look good for
the wrong reason and leave the agency claim unfalsifiable — which is what
§2.1 exists to prevent. So the static policy is the sensible one a competent
engineer would write without an agent: fixed route from
`agency.static.route_rule`, the sub-question verbatim as the query, one round
at a fixed top-k, always stop.

What C21 adds is **conditioning**: route chosen from what previous attempts
returned, query reformulated after a miss, and a stop/deeper/wider judgment.
If those three changes do not beat this file, D2 and D3 were decorative.

**Decisions are traced even when frozen**, with `was_adaptive=False`. An
A2/A3 trace must be structurally comparable to a full run, or C35 cannot
distinguish a frozen decision from a missing one — and a missing decision
surface would silently pass the ablation check.

**Status still depends on what was found.** Marking everything `sufficient`
would be simpler and would hide from D5 that a sub-question retrieved
nothing. That is deterministic bookkeeping rather than a judgment, so it does
not smuggle D3 back in.

**One sub-question per node invocation.** The controller (C19) re-enters the
retriever and chooses which sub-question to work on next — that choice is
D4's, and processing the whole plan inside the node would take it away.
`retrieve_all()` exists for use before the graph is assembled.

**Schema gap closed: `SubQuestion.evidence_ids`.** `merge_evidence` dedupes
by passage id, so when two sub-questions retrieve the same passage only the
first copy survives, carrying the first sub-question's `sub_question_id`.
Filtering state evidence by that field would then **silently hide shared
evidence from the second sub-question** — and sibling sub-questions share
evidence constantly, since they are about one topic. `evidence_ids` is now
the authoritative mapping (in rerank order); `Passage.sub_question_id`
records only which sub-question found a passage first. `evidence_for()` reads
the list and falls back to the old filter for pre-C17 state.

This is the same class of bug as the C14 reducer and the C8 ingestion dedup:
deduplication is correct for storage and wrong for attribution, and the fix
is to keep the mapping separately rather than to stop deduplicating.

`smoke_c17.py` records the baseline numbers — passages found, independent
sources, f1/f2/f3 — that C21 has to improve on.

**2026-09-06 — C17 live result: the reranker was returning sigmoids. Fixed.**

The C17 smoke run reported `max_rerank` of 0.96, 0.89 and 0.97, and
**f3 = 0.080**. Those are probabilities, not logits — C11's decision to
persist raw logits had been silently reversed since C11 itself.

**Cause.** `sentence-transformers` defaults `activation_fn` to
`nn.Sigmoid()` whenever `num_labels == 1`, which `bge-reranker-base` is. The
documentation is explicit about it; we constructed `CrossEncoder(model_name)`
with no activation argument and got sigmoid output.

**Why nothing caught it.** Sigmoid is monotonic, so the *ranking* is
identical — every retrieval test still passed, the smoke tests still passed,
and the reranker still improved nDCG and MRR at C13. The only casualty is f3,
the rank-1-to-rank-5 gap, which exists precisely to distinguish "sharp peak"
from "flat mush". Above ~0.95 the sigmoid compresses everything into a few
thousandths: the measured 0.080 would be roughly 6-9 in logit space. **The
sufficiency model would have been fit at C33 on a feature that cannot
distinguish strong evidence from marginally stronger evidence**, and the
resulting coefficient would have read as "f3 does not predict answerability"
— a false negative about the project's own design.

**Fix.** `CrossEncoder(..., activation_fn=torch.nn.Identity())`, with a
comment stating it is mandatory rather than preferred. Plus
`_assert_logits()`, which inspects each batch and logs an error (once) if the
scores carry the saturation signature: all within [0,1] with several above
0.9. It **warns and does not transform** — guessing an inverse sigmoid would
replace a visible bug with an invisible one.

**This is the fifth silent-degradation trap in the retrieval layer**, after
the bge query prefix (C9), bm25s dropping version numbers (C10), NLI label
ordering (C12), and sigmoid-vs-logit persistence (C11, which this one
reversed). The common shape is worth stating in the writeup: *every one of
them left the system working, produced plausible numbers, and would have
been reported as a finding.* None was caught by a unit test; four were caught
by deliberately inspecting real output, and this one by reading a number in
a smoke report that looked wrong for its units.

**Action required before C33:** any rerank scores already persisted — the
C13 evaluation corpus included — were produced under sigmoid. Re-running
`scripts.retrieval_labels evaluate` is not necessary (ranking is unchanged,
so nDCG/recall/MRR are unaffected), but **no f1/f2/f3 values recorded before
this fix may be used**, and the C17 baseline numbers must be regenerated
before C21 compares against them.

**2026-09-06 — C18. Synthesizer.**

`src/nodes/synthesizer.py`, plus `synthesizer.txt` and
`synthesizer_partial.txt`.

**The label map, as promised at C6.** Passage ids are content-addressed
(`Pa3f9c2b1d004`), which is right for storage and wrong to ask a model to
transcribe. A slip of one hex character produces a citation that does not
resolve — and **a citation that does not resolve is indistinguishable from a
fabricated one**. Both are a claim the critic cannot verify, so the system
would report a transcription error as a hallucination and the headline metric
would absorb it. The prompt therefore uses `P01`…`P24` and this module maps
them back; the real id never reaches the model.

**Invented labels are stripped, not kept.** When the model cites `[P47]` with
19 passages available, the marker is removed and the claim survives uncited.
That is deliberate: an uncited claim is marked UNSUPPORTED by the critic,
which is correct and visible, whereas a dangling citation either crashes the
critic's lookup or is silently skipped — and **a claim that looks cited but
was never checked is the exact failure this project exists to measure.** The
count is recorded, because a model inventing labels is a signal about prompt
quality.

**Evidence is grouped by sub-question, not sorted globally.** A global sort
by score lets one well-covered sub-question occupy every slot and starve the
others, producing a report that answers part of the question confidently and
ignores the rest — which would look like a good report and be a bad one.

**Abandoned sub-questions' evidence is demoted, not excluded.** The first
draft of `_select` skipped it; a test showed the fallback pass re-admitted it
anyway, which forced the question. Demotion is the better answer: an
abandoned sub-question was given up on because it was malformed, so its
evidence was retrieved for a bad query and is likelier to be off-topic — but
a passage can be a poor answer to one question and a good one for another,
and discarding evidence the system paid to retrieve is a stronger claim than
the status supports. Ordering it last means `MAX_PASSAGES` drops it first
when the budget binds, which is the effect wanted without the assertion.

**Two prompts, because PARTIAL is a different task.** The partial prompt
forbids speculation explicitly and demands the form "the evidence does not
establish X" rather than "X may possibly be the case". Hedged prose that
*reads* as an answer while establishing nothing is precisely what the
abstention layer exists to prevent, and it is the easiest failure to produce
by accident.

**The synthesizer does not run on the ABSTAIN path at all** (§3.3). There is
no report to write, and generating one invites the finalizer to show it.

**Section labels are shown to the model**, because the prompt asks it to
distinguish a measured result from a cited one and it cannot do that without
knowing whether a passage came from Results or Related Work (§4.2).

**2026-09-06 — C18 live result: uncited summary sentences. Prompt fixed;
constraint recorded for C23.**

The first real draft cited 8 markers across 7 of 19 offered passages, all
resolving, with no invented labels. But **4 substantial sentences carried no
citation**, and inspecting them showed they were not hallucinations — they
were *synthesis* sentences aggregating claims cited immediately above, e.g.
"In sum, FlashAttention-2's pattern is a tightly-coupled block-tiling scheme
that keeps data movement local to the GPU's fast memory hierarchy."

**Why this matters more than it looks.** The critic marks an uncited claim
UNSUPPORTED, and §5.5 counts unsupported claims toward the hallucination
rate. A correct, well-grounded summary of three cited claims would therefore
be scored as a hallucination. The headline metric would be measuring the
synthesizer's citation habits rather than the system's grounding, and it
would do so in the direction that makes the system look worse than it is —
which is less dangerous than the reverse but still wrong.

**Fix applied here:** both synthesizer prompts now state explicitly that a
summary sentence is a factual claim and must carry the citations of
everything it rests on, and distinguish a summary from a transition ("The
second approach differs" asserts nothing; "Both reduce memory traffic, but
differently" is a claim).

**Constraint for C23.** The prompt fix reduces the problem but cannot
eliminate it, so the critic must handle the residue. When segmenting claims
it needs to distinguish three cases, not two:
- a cited factual claim — verify against the cited passage;
- an **uncited** factual claim — UNSUPPORTED, correctly;
- a transition or discourse sentence — not a claim at all, and excluded from
  the denominator rather than counted as unsupported.

Counting discourse sentences in the denominator would depress every
grounding metric uniformly across the agent and all four baselines. It would
not change which system wins, but it would make the absolute numbers
meaningless and uncomparable to published citation-grounding figures.

**Secondary observation worth tracking:** only 7 of 19 retrieved passages
were cited (37%). That is a usable proxy for retrieval precision — if C21's
adaptive retriever is genuinely better, the citation rate should rise, since
fewer retrieved passages will be irrelevant. Recorded alongside the C17
baseline.

**2026-09-06 — C19. Controller and minimal graph. The system runs end to end.**

`src/controller.py` and `src/graph.py`.

**Hub and spoke.** Every node routes back through the controller via
`add_conditional_edges(node, controller.route, path_map)`, including the entry
point. The routing table in `controller.py` *is* the control flow rather than
a description of it. The alternative — hardcoded edges with conditionals
where the path branches — makes "why did it stop researching?" a question
about four files; here it is a question about one function, which is exactly
what C38's MAST annotation needs.

**Deterministic and LLM-free.** The controller makes no model calls, so the
whole policy is testable exhaustively without an API key and a routing bug
cannot hide behind sampling noise. The *decisions* are agentic (D1-D5 live in
the nodes); the *dispatch* is not. Conflating them would make the agency
claim unmeasurable.

**The full table is written once.** Nodes arrive across Phases 2 and 3, and
the controller skips any stage whose node is not registered. C19 runs
premise_checker → planner → retriever → synthesizer; the critic at C23 and
decider at C27 change no routing code, only the registry. The usual outcome —
routing growing a branch per commit until nobody can state the whole policy —
is what this avoids.

**New config key: `agency.skip_research_on_failed_premise`.** A refuted core
premise skips research and goes straight to the decision. The correction is
already in state from the premise check, and the decider abstains regardless,
so researching anyway spends the whole budget elaborating something known to
be false. Only CONTRADICTED triggers it; UNVERIFIABLE is silence and research
proceeds (C15). It is a config key rather than a constant because §6.5 may
want to ablate it.

**Two loops caught by tests, both costing real money.**

The first: with the planner absent and an empty plan, routing fell through to
the synthesizer, which returns an empty draft, which leaves `not
state["draft"]` true, which routes to the synthesizer again. Guarded by
requiring evidence before synthesis — correct in its own right, since no
evidence means no report.

The second, which the first guard did not close: with evidence present and
the model returning an empty string, the same loop reappears. Fixed in two
places. `synthesizer_node` now reports an empty draft as an **error** — a
silently empty result is a failure and must be recorded as one — and the
controller refuses to re-enter the synthesizer once `state["error"]` is set.
**Each iteration of that loop was an LLM call**, so it would have burned a
day's quota before the recursion limit fired.

**Phase 2 milestone reached.** The system runs end to end, checkpoints after
every node, and resumes in a fresh process. Everything after this makes it
agentic (C20-C22) and calibrated (C26-C27) rather than building new
machinery.

**2026-09-06 — C19 live result: node spans were never wired. Fixed.**

The first end-to-end run succeeded — premise check, plan of 3, three
retrieval rounds, 22 passages, a 2898-character cited draft, correct routing
with reasons, and a clean resume from checkpoint. The Groq fallback also
fired live (`429 on openai/gpt-oss-120b — switching model`), which is the
first confirmation of C2's chain under real load.

Two smoke checks failed, and both were the script misreading the trace rather
than the system misbehaving — but they exposed a genuine gap.

**C4 built the `tracer.node()` span machinery and nothing ever used it.**
Nodes emitted no NODE_ENTER or NODE_EXIT events, so:
- `TraceReader.nodes_visited()` returned empty, and the trace recorded
  *routing* without recording what ran between routes — no node durations at
  all;
- every routing decision appeared to originate at START, because
  `route()` reads the node stack and the stack was always empty.

Both matter for C38. "Which node was slow" and "what ran before this
decision" are among the first questions a failure annotation asks, and
neither was answerable.

**Fix.** `build_graph` now wraps every registered node in `tracer.node(name)`,
so entries, exits and durations are recorded. And the tracer remembers the
last completed node: the controller runs on a *conditional edge*, i.e. after
the node has exited, so the stack is legitimately empty by then and the
origin has to come from somewhere else.

Worth noting the shape of this one. It is not a logic error — every unit test
passed, the system did the right thing, and the only symptom was a blank line
in a report. **The trace was quietly less useful than it appeared**, and the
cost would have landed at C38 with the test split already spent. That is the
sixth instance of the same pattern recorded in this document, and the first
outside the retrieval layer.

**2026-09-06 — C20. Sufficiency features.**

`src/sufficiency/features.py`, wired into the retriever so f1-f10 are
recomputed every round.

**Missing is not zero, and the distinction is the main design decision.**
At C20, f5 (contradiction rate) and f6 (entailment strength) are genuinely
unavailable — contradiction scanning is not wired into the retriever and the
critic lands at C23. They return `None`. `to_vector()` then emits a neutral
imputation **and an explicit `_missing` indicator** for every feature.

Imputing 0.0 for f5 without the indicator would tell the model "no
contradictions were found" when the truth is "nobody looked" — two very
different states collapsed into one, and the coefficient fit at C33 would be
meaningless. With the indicator the model can also learn that missingness
itself predicts something, which it plausibly does: a sub-question whose
critic never ran is one the system gave up on.

**Dimensional consistency is a hard requirement**, not a nicety.
`FEATURE_NAMES` fixes the order and every extraction returns every key. A
vector whose width depends on what happened cannot be stacked into a design
matrix, and that failure would surface at C33 as a shape error with the eval
already paid for.

**Units are deliberately left unscaled.** f1-f3 are logits (unbounded),
f4/f8 are counts, the rest are in [0,1]. Standardisation belongs in the model
pipeline at C26 — scaling at extraction time would make a stored feature
depend on the batch it was extracted with, so a feature logged during a run
could not be compared with one logged a week later.

**Two places where None beats a number**, both inherited from earlier
decisions: f3 is None below five results (a zero gap asserts "uniformly
flat", which four passages do not license — C11), and f4 is None with no
evidence at all ("no sources" and "zero independent among those found" are
the same number but not the same situation).

**r2_min_sufficiency is reported alongside the mean** because a report
resting on one unsupported sub-question is unsafe however well the others
went, and averaging hides precisely that.

**Documented deviation from §5.2 — f10.** The spec describes "gap between the
question's implied timeframe and source dates". Extracting an implied
timeframe costs an LLM call per question, and the premise checker already
costs two; the guard would start costing more than the research. f10 measures
**evidence recency** instead (median publication date against an 8-year
horizon), which is a proxy. It still carries the Tier D4 signal — a question
whose answer has moved retrieves old evidence — but it will **not** catch a
recent question answered with recent but superseded evidence. This belongs in
the writeup's limitations, stated plainly rather than implying the full
version was built.

**Features are recomputed every round, not only at the end.** D3 reads them
at C21 to decide whether to stop, so they must exist while a sub-question is
still in flight; and per-round logging means the trace shows whether a second
round actually improved the evidence, which is what separates a useful
re-retrieval from a retry.

**2026-09-06 — C20 live result: two bugs, one loud and one silent.**

The feature vectors came out well — f1 ranging 6.9-8.6, f3 2.6-6.0, f4 4-7 —
so the signal is there. But the run surfaced two defects.

**Loud: a 400 from Groq killed the run outright.** `json_validate_failed`
with an empty `failed_generation` — the model failed to produce parseable
JSON under `response_format: json_object`, and Groq validates that *server
side*, returning 400 instead of the text. C2's `classify()` treated every 400
as fatal, so one transient generation failure ended a ten-question run on
question six.

Worse than the crash: **the provider is stricter than our parser.** C2 built
a repair layer for fences, trailing commas and truncation, and server-side
enforcement discards exactly the responses it was built to fix, before we
ever see them. `classify()` now returns `retry_plain` for this case — same
model, enforcement off, repair layer in play — and escalates to the next
model only if that also fails. `context_length` and similar 400s now
`switch` rather than die; a genuinely malformed request is still fatal,
because retrying a code bug only wastes quota.

**Silent, and the more dangerous one: f5 read 0.00 on all ten questions.**
Contradiction scanning has never been wired into the graph. `ResearchState`
initialises `contradictions` to `[]`, so an unscanned run and a
scanned-and-clean run are **indistinguishable from the data alone** — and
`_contradiction_rate` read the empty list as "scanned, no disagreement
found".

The feature was asserting agreement that nothing had checked. At C33 the
model would have been fit on a constant mislabelled as an observation, and
the resulting coefficient would have read as "cross-source disagreement does
not predict answerability" — a false negative about one of the project's own
design claims, produced by a default value. Fixed with an explicit
`contradictions_scanned` flag, defaulting to False, which C21 flips where the
scan actually runs.

This is the seventh instance of the recorded pattern, and the sharpest
illustration of why C20 insists that **missing is not zero**: the rule was
already written, and the bug slipped in through a default that made missing
*look* observed.

`smoke_c20.py` now survives a failed question rather than discarding nine
good vectors with the tenth.

**2026-09-06 — C20 second run: a self-inflicted quota bug, and a useful
finding about f8/f9.**

**The bug was in the fix.** Folding `retry_plain` into the `switch` branch
meant a JSON validation failure reached `_exhaust_locally()`, which records
the model's *entire remaining daily token budget* as spent. So one
badly-formatted response burned a model for the day in the local ledger, the
chain fell through to the next, and three models were dead after five
questions with almost nothing actually consumed.

The failure mode is worth naming: **the ledger became wrong in the
conservative direction**, which looks like prudence and is not. Nothing
errored, no quota was really lost, and the symptom was "the eval mysteriously
stopped" — exactly the shape a quota ledger exists to prevent. A genuine 429
is now the *only* path that exhausts a model; JSON failures and
`context_length` move on with the model's quota untouched. Four regression
tests pin this, including one asserting a format failure on question one does
not starve question two.

**f8 and f9 are constant, and that is the most useful line in the table.**

| feature | C20 (static) | what makes it vary |
|---|---|---|
| f8 rounds consumed | 1.00, stdev 0 | D3 deciding to go deeper |
| f9 route escalation | 0.50, stdev 0 | D2 escalating to another route |

These two features **can only vary if the retriever is adaptive**. Under the
static policy every sub-question gets exactly one round on one route, so
both are constants by construction. That makes them a second, independent
check on the agency claim alongside C35's trace counts: if f8 and f9 are
still constant after C21, D2 and D3 changed nothing, and no ablation is
needed to know it. Worth stating in the writeup — a feature that only moves
when the agent is adaptive is a direct measurement of adaptivity.

The five features that do vary (f1 6.91-8.64, f2 3.94-7.77, f3 2.58-5.99,
f4 4-7, f10 0.33-0.64) carry real spread across only five questions, which
is the encouraging part.

**f7 was "never observed" for a script reason, not a feature reason** — the
smoke test ran the planner without the premise checker. Fixed; it now runs
the checker so f7 is exercised.

**2026-09-07 — C20 complete run. The first evidence that the feature vector
separates answerable from unanswerable questions.**

The ledger rolled over at midnight and cleared the poisoned state on its own;
the reset tool confirmed it was already at 0/200000 rather than genuinely
spent. All ten questions ran.

**f1 now ranges -1.93 to 8.64 (stdev 3.85).** Negative logits mean the
cross-encoder judges that nothing retrieved answers the query — a signal that
did not exist at all while the reranker was returning sigmoids (C17 fix),
where everything compressed into [0.89, 0.97].

Grouping the ten questions by what the C13 labelling already told us about
corpus coverage:

| | f1 mean | f1 range |
|---|---|---|
| Well covered (≥3 fully-relevant passages in the C13 pool) | **7.17** | 4.74 to 8.64 |
| Poorly covered (≤1) or false premise | **0.02** | -1.93 to 2.58 |

| feature | worst good | best bad | separation |
|---|---|---|---|
| f1 max rerank | 4.74 | 2.58 | **clean** |
| f2 mean top-3 | 3.94 | 1.47 | **clean** |
| f3 score gap | 1.27 | 5.05 | overlaps |
| f4 independent sources | 4.00 | 6.00 | overlaps |
| f7 premise grounding | 0.83 | 1.00 | overlaps |
| f10 recency | 0.33 | 0.93 | overlaps |

The three questions the C13 labelling flagged as corpus gaps — rq20
(perplexity, 1 grade-2), rq21 (limitations, 1), rq29 (RAG failure modes, 0) —
land at f1 of 1.16, 2.58 and -1.75. The false-premise question lands at
-1.93. **The features found the same questions the human labelling did**,
without being told.

**What this is and is not.** It is a signal check on ten hand-picked
questions with no labels, not a result: n=10, no held-out split, and the
grouping uses the C13 pool as a proxy for answerability. Claim 1 (§1.4) is
tested at C33 by fitting the calibrated model and at C36 on the held-out 24,
and nothing here substitutes for that. What it does establish is that the
vector is worth fitting a model to — which before this run was an assumption.

**f4 separates in the wrong direction**, and that is worth knowing early:
poorly-covered questions averaged 5.25 independent sources against 5.67 for
well-covered ones. Retrieval returns *something* from many sources regardless
of whether any of it answers the question, so source count measures breadth,
not sufficiency. On its own it may be near-useless; its value, if any, will
be as an interaction with f1 — many sources AND high rerank is different from
many sources and low rerank. Worth noting in the C33 coefficient discussion
rather than being surprised by it then.

**f8 and f9 remain at stdev 0**, exactly as predicted. They are constants
under the static policy by construction, and they are the two features that
can only move if D2 and D3 are live. They are the direct measurement of
whether C21 does anything.

**2026-09-07 — C21. Adaptive retriever. D2 and D3 live.**

`src/nodes/adaptive_retriever.py`, subclassing `StaticRetriever` so the
*only* difference between the two is the decision. A2 and A3 restore the
static policy while everything else stays fixed, and that comparison is only
clean if nothing else differs.

**Where the LLM is, and where it deliberately is not.**

D3 (stop / deeper / wider / give up) is decided from features, not by a
model. A sub-question whose best passage scores 8.6 is plainly answered; one
scoring -1.9 plainly is not. Asking a model costs ~2,500 tokens per
sub-question per round against a 200,000-token daily budget, and it makes the
decision *less* auditable — a threshold on f1 can be inspected, "I think
that's enough" cannot.

This is still adaptive in the sense §2.1 requires. The decision is
**conditioned on the evidence retrieved**, so freezing it changes behaviour,
which is exactly what A3 measures. Adaptive means the outcome depends on what
happened, not that a language model was consulted. Stating this now matters,
because "but it's only a threshold" is the obvious interview challenge and
the answer is the falsifiable test, not the mechanism.

D2's *query reformulation* does use the LLM, because it genuinely needs
language: no threshold can know that "what perplexity does the model reach"
should become "WikiText-103 perplexity results table", and that rewrite is
where recovery actually comes from.

**Cost: one LLM call per escalation, zero for a sub-question answered first
time.** Round one has no history to condition on, so conditioning there would
be vacuous — the common case is free, and only failure costs anything.

**Thresholds fitted to measured data, not guessed.**
`sufficient_min_rerank: 4.0` sits inside the gap the C20 run measured:
well-covered questions averaged f1 7.17 (worst 4.74), corpus gaps averaged
0.02 (best 2.58). `hopeless_max_rerank: 0.0` is the other side of it. Both
live in config so C33 can revisit them once the calibration split exists.

**Two guards against §2.4's anti-patterns.** A reformulated query identical
to one already attempted is rejected and the sub-question falls back rather
than looping — "if iteration 2 issues the same queries as iteration 1, you
have a retry, not a re-plan". And when the LLM is unavailable, the route
escalates deterministically rather than repeating a route that just failed.

**The graph now chooses its retriever from the agency flags**, so A2 and A3
are a config change rather than a code path. With both frozen it instantiates
`StaticRetriever` directly — the adaptive class with every decision disabled
would behave identically, but running the original makes "the ablation
restores C17" a fact rather than a claim.

**f8 and f9 are the acceptance test that matters.** They are constants under
the static policy by construction, so if they are still constant after this
commit, D2 and D3 changed nothing and no ablation is needed to know it.

**2026-09-07 — C21 live run: D2 was decorative by construction. Two fixes.**

The smoke test reported a **zero delta on every evidence metric** — f1, f2,
f3, sources and passages all identical between static and adaptive — while
costing 3 extra LLM calls and 4 seconds. The adaptive retriever was doing
work and changing nothing. Two causes, and the first is the more serious.

**1. The route was never used.** `_execute` called
`self.retriever.retrieve(query, top_k=...)` with no filters. The route was
written into `attempted_routes`, logged in the trace, shown in the smoke
output — and then discarded. Escalating from `arxiv_meta` to
`arxiv_fulltext` searched the identical corpus and returned byte-identical
passages.

This is **§2.4's first anti-pattern reached by omission**: "every LLM call is
a text transformation... if no LLM output ever changes which node runs next,
the graph is a straight line." Here the LLM output changed a field that
nothing read. The trace showed an escalation; the system performed none.

Worth sitting with, because it is the exact failure the project exists to
detect, and it survived a full commit with 29 passing tests. The tests
verified that the route was *recorded*, never that it *mattered* — and the
smoke test caught it only because it compared against the static baseline
rather than checking the adaptive path alone. **A test that the decision was
made is not a test that the decision had an effect.**

Routes now map to payload filters (C9's machinery): `arxiv_meta` to
abstracts, `arxiv_fulltext` to paper bodies with abstracts excluded, `web` to
web sources, `evidence_store` unfiltered. An empty slice falls back to an
unfiltered search, because a route finding nothing is a fact about the corpus
rather than about the query.

**2. `insufficient` is terminal, and the adaptive retriever used it to mean
"try again".** `TERMINAL_STATUSES` includes it, so `retrieve_all` stopped
after round one on every sub-question. The reformulated query was computed,
charged to the budget, stored — and never used. `in_progress`, already in the
`Status` literal and already excluded from `TERMINAL_STATUSES`, is now the
continuation state; `insufficient` keeps its original meaning of "finished,
without enough evidence".

Both fixes are covered by regression tests that assert the *effect* rather
than the record: that filters reach the retriever, and that `retrieve_all`
genuinely runs a second round.

**2026-09-07 — C21 working. First real evidence that D2 and D3 do something.**

With routes actually filtering and `in_progress` non-terminal, the adaptive
retriever recovered a question the static one could not:

```
round 1  [arxiv_meta]      "what proportion of the chip's arithmetic
                            capability does the improved attention kernel
                            actually manage to use"          -> f1 -2.04
round 2  [arxiv_fulltext]  "FlashAttention-2 chip arithmetic utilization
                            percentage of arithmetic capacity"
round 3  [arxiv_fulltext]  "FlashAttention-2 chip arithmetic utilization
                            72% model FLOPs"                 -> f1  6.01
```

Vernacular phrasing to technical vocabulary to the specific number, with the
route escalating from abstracts into paper bodies. **This is the case study
for §6.5 and the interview answer to "is it actually agentic?"** — a static
retriever issues the first query and stops at f1 -2.04.

**The comparison on the C17 question is mixed, and the mixture is
informative:**

| metric | static | adaptive | |
|---|---|---|---|
| f1 max rerank | 3.07 | **4.79** | better |
| f2 mean top-3 | 1.33 | **3.10** | better |
| passages | 14 | **27** | more |
| f3 score gap | 7.86 | 2.92 | worse |
| f4 independent sources | **12** | 2 | much worse |
| cost | 3 retrieval, 0 LLM, 10s | 8 retrieval, 5 LLM, 52s | 5x slower |

**f4 collapsing 12 → 2 was a real defect, now fixed.** The planner suggested
`arxiv_fulltext` for all three sub-questions, so every one started inside
paper bodies. Full-text chunks concentrate in the handful of ingested papers;
abstracts span every paper a search returns. Depth bought precision and cost
breadth — a bad trade to make *before the cheap route has even been tried*.

§3.3 already said so: the reader is "invoked only when the retriever decides
depth is needed (D3), which makes depth an **earned** cost rather than a
default". A planner suggesting full text up front bypasses D3 entirely. The
first attempt is now clamped to a cheap route; escalation still reaches full
text, but only after a shallow attempt has failed.

**f3 must be read jointly with f1, not alone.** Static scored f3 7.86 with f1
3.07 — a "sharp peak" where the peak itself is mediocre and ranks 2-5 are
terrible. Adaptive scored f3 2.92 with f1 4.79: several decent passages
rather than one acceptable one among junk. A high gap over a low maximum is
not evidence quality, and the C33 model must have both features available to
disentangle them. Worth watching whether the fitted coefficients reflect
that; if f3 gets a large positive weight on its own, the model has learned
something misleading.

**The cost is real and must be reported.** 5x wall clock, 5 LLM calls, 8
retrieval calls against 3. §6.7's framing applies: the comparison that
matters is cost per *correctly-answered-or-correctly-abstained* question, not
cost per question. sq3 ending `insufficient` after three honest rounds is
part of what the extra cost bought.

**2026-09-07 — C21 third run: f4 recovered, and a schedule masquerading as a
decision.**

Clamping the first attempt worked: independent sources back to 12, matching
static. But the run exposed something more important.

**Every sub-question ran exactly 3 rounds — the round limit — and reached
f1 3.07, which is exactly what the static retriever reached in one round.**
9 retrieval calls against 3, 6 LLM calls against 0, 71 seconds against 8, for
an identical headline number.

The cause: `sufficient_min_rerank: 4.0` was calibrated on the C20 run, where
retrieval was *unfiltered* and well-covered questions scored f1 7-8.6. With
route filters live the candidate pool is a slice of the corpus, so f1 is
systematically lower — abstracts alone cannot score what the whole corpus
could. **Thresholds calibrated before filtering do not transfer after it**,
which is worth remembering at C33 when τ is chosen.

But the threshold was the symptom. The defect was that the retriever had
**only an absolute test**: is the evidence good enough? With nothing able to
clear the bar it escalated until it ran out of rounds, every time, on every
sub-question. That is §2.4's anti-pattern exactly — "the loop counter is the
only thing that changes between iterations" — and it would have passed an
ablation naively, because A3 (one round, always stop) really would have
produced different numbers.

**Fix: a marginal test alongside the absolute one.** If a round improves f1
by less than `min_improvement` (0.5) over the best so far, stop — further
rounds are unlikely to help. Whether that stop is `sufficient` or
`insufficient` depends on the evidence held, not on the fact of stopping.

This is what makes the number of rounds **depend on the evidence**: an easy
sub-question stops at 1, a plateauing one at 2, and only a genuinely
improving one runs to 3. A test asserts exactly that property.

**The smoke check was also testing the wrong thing.** It compared f8/f9
variance *across sub-questions within one run*; three sub-questions of a
single question are often genuinely similar, so that measures the question's
shape rather than the retriever's behaviour. It now compares adaptive against
the static baseline — the comparison the ablations actually make — and
separately flags the schedule signature (every sub-question hitting the round
limit).

**2026-09-07 — C21 complete. All criteria met; the honest reading of the
numbers.**

```
STATIC    sq1 sufficient  r=1   sq2 sufficient  r=1   sq3 sufficient  r=1
ADAPTIVE  sq1 insufficient r=2  sq2 sufficient  r=3   sq3 insufficient r=2
```

Rounds now vary with the evidence (2, 3, 2 against a flat 1), routes escalate,
queries are reformulated, and cost fell from 71s to 18s once the marginal
test was added.

**The most important difference is not in the delta table.** Static marked
every sub-question `sufficient`; adaptive marked two `insufficient`. Static
never *checked* — one round, no evaluation, status assigned by whether
anything came back at all. Adaptive examined the evidence and reported that
two sub-questions did not reach the bar.

Those two labels are what D5 reads. The difference between a system that
answers everything and one that knows what it lacks is invisible to f1, f2
and f3, and it is the thing this project is actually about. Any comparison
table that reports only the evidence metrics understates the adaptive
retriever in exactly the dimension that matters — worth stating when the
ablation results are written up.

**On the evidence metrics themselves, the result is modest and should be
reported as modest:**

| | static | adaptive | |
|---|---|---|---|
| f1 max rerank | 3.07 | 3.07 | unchanged |
| f2 mean top-3 | 1.33 | **2.57** | better |
| passages | 14 | **35** | more |
| f3 | 7.86 | 1.54 | see below |
| sources | 12 | 12 | unchanged |
| cost | 3 retrieval, 0 LLM, 9s | 7 retrieval, 4 LLM, 18s | 2x |

f1 being *identical* is informative rather than disappointing: both
retrievers used `arxiv_meta` on round one and found the same best passage
immediately. Escalation then added depth — f2 up, passages up — but could
not improve on a peak the corpus had already yielded. **When the corpus has
a ceiling, more rounds buy breadth, not a better top passage.** That is a
real finding about when D3 helps and when it cannot.

f3 falling from 7.86 to 1.54 is the joint-reading problem again: static's
"sharp peak" was one acceptable passage among junk, adaptive's flat
distribution is several decent ones. Reported alone, f3 says static won.

**No further threshold tuning here.** Fitting `sufficient_min_rerank` to this
one question would be overfitting to a single example, and whether D2 and D3
earn their cost is C35's question across 24 benchmark questions, not C21's
across one. The thresholds stay in config and are revisited at C33 with the
calibration split.

**2026-09-07 — C22. Budget allocation (D4). Phase 2 complete.**

`src/sufficiency/allocation.py`. `MarginalValueAllocator` is the adaptive
counterpart to C5's `UniformAllocator`, which *is* ablation A4.

**Value is not monotone in how badly a sub-question is doing**, and that is
the whole idea. It peaks in the middle:

| state | value | why |
|---|---|---|
| already sufficient | 0.0 | nothing left to buy |
| just below the bar | high | one more round plausibly closes it |
| far below the bar | low | the corpus does not appear to have it |
| never attempted | 1.0 | unknown, and unknown must be tried |

An allocator that funds the weakest sub-question throws budget at corpus
gaps; one that funds the strongest spends on questions already answered.
Both are easy to write by accident, and a test pins the non-monotonicity so
neither can creep in.

**Untried sub-questions are guaranteed a first attempt before allocation
touches anything.** A sub-question nobody has retrieved for has no evidence
*because nobody looked*, so any score-based estimate of its prospects is a
guess dressed as a measurement. Starving it would mean D4 is pre-judging
rather than allocating — and it would systematically starve exactly the
sub-questions whose difficulty the planner failed to anticipate.

**Plausibility decays with rounds already spent.** This is f8's rationale
from §5.2 ("many rounds to reach thin evidence is a bad sign") used as a
*decision* rather than only as a feature: a sub-question on its third round
at f1 1.0 is a worse bet than one on its first round at f1 1.0, though their
evidence is identical.

**`next_sub_question` makes the allocation load-bearing.** §3.3's routing
table says the controller picks "highest-value SQ per D4". Without it the
retriever takes the first pending sub-question in plan order and the
allocation is a number nobody reads — the same defect as C21's unused route,
which is recent enough to be worth guarding against deliberately. The node
now asks the allocator; with no allocator it falls back to plan order, which
is precisely what A4 restores.

Allocation uses largest-remainder apportionment so shares sum exactly to the
pool rather than losing up to n-1 calls to truncation — the same correctness
point as C5's `_distribute`.

> **Phase 2 exit criterion met.** The trace of a single run visibly shows the
> system changing its mind: a different route on retry, a reformulated query,
> a round count that depends on the evidence, and uneven budget spend.

**2026-09-07 — C22 live run: D3 and D4 overlap, and that is a prediction
worth recording before C35.**

The run behaved correctly:

```
sq1  "What fraction of theoretical peak FLOPs does FlashAttention-2 reach?"
     sufficient,  1 round,  f1  7.01
sq2  "What makes retrieval augmentation fail on questions requiring
      synthesis?"  (C13: zero fully-relevant passages in the pool)
     abandoned,   2 rounds, f1 -0.29
```

The smoke test reported two WARNs because it sampled allocation *after*
`retrieve_all` finished, by which point both sub-questions were terminal,
every value was 0.0 and there was nothing left to allocate. The system was
right and the measurement arrived too late. The script now steps through
rounds and samples while sub-questions are still open.

**The substantive finding: D3 cut the loss before D4 could.** The hopeless
sub-question was abandoned after two rounds because round two did not
improve on round one, so the budget D4 would have withheld was never
requested.

This has a direct consequence for §6.5. **A4 (uniform split) with D3 still
live may show little or no effect**, because D3 already prevents the waste
that D4 exists to prevent. The two surfaces are not independent: D3 stops a
losing sub-question, D4 declines to fund one. Recording the prediction now
matters for two reasons — it is intellectually cheap to claim afterwards that
a null result was expected, and the honest reading of a small A4 effect is
"its job was already being done elsewhere", not "budget allocation does not
matter".

If A4 does come out flat, the informative follow-up is an A3+A4 pair: freeze
both and see whether the combined effect exceeds the sum of the individual
ones. That is outside §6.5's one-surface-at-a-time design, so it would be an
addition to report separately rather than a change to the ablation table.

**2026-09-07 — C22 per-round trace: two bugs the aggregate view hid.**

Sampling allocation during the run rather than after it made both visible
immediately.

**1. The first choice of every run was backwards.** Every untried
sub-question scores exactly 1.0, so the opening choice is always a tie, and
`max(scored, key=(value, id))` resolved it lexically — so the run opened by
retrieving for `sq2`, the hopeless one, purely because "sq2" sorts after
"sq1". Ties now break toward **plan order**, which carries the planner's
reasoning about sequence; id order carries nothing.

**2. A converging sub-question was abandoned.** sq2 went from f1 **-4.21 to
-0.29** between rounds — an improvement of 3.9, eight times
`min_improvement` — and was abandoned for still being below zero. At that
rate the next round would have cleared the bar.

The fix changed the shape of the stopping decision rather than patching a
threshold. `_judge` now asks two questions in order:

1. Is the evidence good enough *and* corroborated? → stop, `sufficient`.
2. Is any further work coming — has it plateaued, or are the rounds gone?
   If so, the **level** decides the label: above the bar `sufficient`, at or
   below `hopeless_max_rerank` `abandoned`, in between `insufficient`.
3. Otherwise keep going, **including from deep below the bar**, as long as
   it is still climbing.

The principle, worth stating in the writeup because it generalises: **the
absolute level says where the evidence is; the improvement says where it is
going, and only the second justifies stopping early.** Abandoning on level
alone discards sub-questions that are converging, which is the opposite of
cutting losses — it cuts the ones about to pay off and keeps funding the
ones that have flatlined at a mediocre score.

This also tidied a real inconsistency: the hopeless check and the marginal
check were asking the same question from two sides and could disagree, so a
stagnant sub-question below zero was being labelled `insufficient` by
whichever test happened to fire first. They are now one branch.

**2026-09-07 — C22 verified. Where D4's value actually shows.**

With both fixes in, the per-round trace reads correctly:

```
round 1  D4 chose sq1   sq1 sufficient f1 7.01 (value 0.000)
                        sq2 pending              (value 1.000)
round 2  D4 chose sq2   sq2 in_progress f1 -4.21 (value 0.050)
round 3  D4 chose sq2   sq2 in_progress f1 -0.29 (value 0.050)
```

sq1 is chosen first on the tie (plan order), finishes in one round, and
drops to value 0.000. sq2 climbs from -4.21 to -0.29 and is **not**
abandoned, because it is still improving.

**The D4 win is measured against the uniform baseline, not between
sub-questions.** The final allocation table makes it concrete:

| | uniform (A4) | marginal (D4) |
|---|---|---|
| sq1 (finished in round 1) | 10 | **0** |
| sq2 (still open) | 10 | **20** |

Uniform would commit half the remaining budget to a sub-question that was
already answered. That is where the saving lives, and it is invisible if the
check only asks whether the answerable sub-question got more than the
hopeless one — the comparison the smoke test originally made, and the reason
it reported a spurious warning on a correct run.

Worth carrying into §6.5's reporting: **A4's effect is a cost difference, not
an evidence-quality difference.** Freezing D4 does not make the evidence
worse, it makes the same evidence more expensive — so an ablation table
reporting only nDCG-style quality metrics would score A4 as having no effect
while it burns half the budget. §6.7's cost-per-correctly-resolved-question
framing is what catches it.

---

## Phase 3 — verification and decision

**2026-09-07 — C23. Critic with entailment judgment.**

`src/nodes/critic.py` and `src/prompts/critic.txt`. Five-way routing on what
to *do* about a verdict is C24; this node decides only what is true.

**Three categories, not two** — the C18 constraint, implemented. A draft
contains cited claims, uncited claims, and sentences that assert nothing.
`NOT_A_CLAIM` sentences are excluded from the denominator entirely. Counting
them would depress every grounding metric uniformly across the agent and all
four baselines: not changing which wins, but making the absolute numbers
meaningless and incomparable to published citation-grounding figures.

The classification is deliberately biased toward CLAIM when the model is
unsure. Judging a transition is harmless; excusing a claim is not.

**Two bugs found while testing, both about conflating distinct things.**

*Segmentation.* A citation written after the full stop — "…reaches 73%.
[P1a2b…]" — was split onto the *following* sentence, crediting the wrong
claim and leaving the real one uncited. Models produce both orders. Leading
citations are now re-attached to the sentence they actually support.

*Uncited sentences.* The first version skipped them entirely, on the
reasoning that a claim with no citation is UNSUPPORTED by definition and
asking the model costs tokens. That conflated **"no citation"** with **"not
a claim"**, and the gap was papered over with a length heuristic — which
cannot tell "The second approach differs" (28 characters, pure discourse)
from "FlashAttention-2 reaches 73% of peak" (also short, a claim). Both land
in the hallucination rate if guessed wrong, in opposite directions.

Now every substantial sentence is classified, cited or not. A cited sentence
needs two judgments — is it a claim, and does its citation support it. An
uncited one needs only the first; the **absence of a citation** then decides
the verdict, whatever the model says about support it could not have checked.

**One call for the whole draft** (§7.1). Per-claim calls would be 8-10 per
question against a budget of roughly 9, so the critic alone would exceed the
run. Batching also reduces variance: one judge seeing all claims together
applies a more consistent standard than ten independent calls.

**An unparseable verdict defaults to UNSUPPORTED.** Claiming support the
critic never gave is the one error the grounding metric cannot survive, so
every ambiguity resolves against the claim.

**Contradiction findings are incorporated** (§4.6): a passage in a
cross-source disagreement is flagged in the prompt, and a claim that picks
one side without acknowledging it is PARTIAL at best.

The prompt names three traps explicitly, because each produces a confident
wrong verdict: a claim can be **true and unsupported** (the model's own
knowledge is not the evidence under test); a Related Work passage describes
**someone else's** work, so attributing it to this paper is unsupported
however accurate the number; and numbers must match in the direction that
matters ("about 70%" from a passage saying 73% is fine, "73%" from "about
half" is not).

**2026-09-07 — C24. Five-way critic routing (§2.3).**

`src/nodes/critic_router.py` and `src/prompts/critic_router.txt`, registered
in the controller as a node after the critic.

**Why it is not an `if` statement.** The naive version routes on the verdict:
`UNSUPPORTED → re-retrieve`, everything else → finalize. But **four of the
five outcomes can follow a single UNSUPPORTED verdict**, and which is right
depends on *why* it failed — whether the queries tried were phrased the way
sources phrase it, whether the sub-question is answerable as asked, whether
the evidence is silent or contradictory. None of that is in the verdict. A
test fires all five from the same verdict, which is the clearest statement of
the point available.

**Gates are facts; the rest is judgment.** Two things are decided without the
model: a route needing budget that does not exist is unavailable, and a
sub-question at its revision ceiling cannot be revised. Offering the model an
option it cannot have invites it to pick one and have the code silently
substitute something else — which makes **the trace a record of a decision
that was never taken**, the exact defect C21's unused route turned out to be.
When the model picks an unavailable route the fallback is recorded as
"RE_RETRIEVE unavailable", not quietly rewritten.

**Two biases, both toward honesty**, stated in the prompt. RE_RETRIEVE vs
UNRESOLVED resolves to UNRESOLVED: an admitted gap is a usable result, a
third failed retrieval is spent budget and the same gap. DROP vs HEDGE
resolves to HEDGE: removing a claim removes the reader's chance to judge it.

**The decisions change control flow, not just records.** RE_RETRIEVE reopens
the sub-question (`in_progress`), so the controller routes back to the
retriever; REVISE_PLAN marks it `insufficient` for the planner; UNRESOLVED
writes a gap that feeds D5. DROP_CLAIM and HEDGE change the report rather
than the research and are applied by the finalizer at C28.

The node **clears the draft** when further work is chosen. Leaving it would
let a stale draft reach the decider as though it reflected evidence gathered
afterwards — a quiet way to report a verified conclusion that was verified
against different evidence.

**A5's static counterpart is binary routing**: everything unsupported is
re-retrieved if budget allows, otherwise left alone. That is precisely the
conditional edge §2.3 describes as "not a decision", and it costs **zero**
LLM calls — so A5 measures the value of the judgment, not of the extra call.