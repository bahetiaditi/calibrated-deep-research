# HANDOFF — calibrated-deep-research

**Snapshot: 26 November 2026, after commit `8f93afa` (C37).**
Written to start a new chat with full context. Read this first, then
`REFERENCE.md` sections 1–13 (the design) and its dated build log (every
commit's reasoning and live findings). REFERENCE.md is the source of truth;
this file is the index, the current state, and the plan.

---

## 0. How the next chat should start

1. The user uploads a zip of the repo (HEAD of `main`). Unzip it as the
   working copy; this file and REFERENCE.md are inside.
2. Read §2 (working agreement) before writing anything: it defines how
   deliverables, commands and commits are expected.
3. **First task: §8, item N1** — the hallucination number from the first
   scoring run (0.484) is probably inflated by the grader, not the system.
   Diagnose and fix before any scoring is trusted.
4. Then follow §9 (plan). Quota-free work happens in chat; the user runs
   the daily eval on their Mac.

---

## 1. The project in one paragraph

A multi-agent research system (LangGraph) whose terminal decision is
**ANSWER / PARTIAL / ABSTAIN**, gated on a **calibrated evidence-sufficiency
score** computed from retrieval-derived features — not on the model's
verbalized confidence. A portfolio project for Gen AI Scientist roles.
Repo: `github.com/bahetiaditi/calibrated-deep-research`, branch `main`.

**The three claims (§1.4):**
1. A retrieval-grounded sufficiency signal is better calibrated than LLM
   verbalized confidence (ECE, risk–coverage, AURC) — agent vs **B3**.
2. Multi-agent structure buys correct abstention at a measurable coverage
   cost (false-answer rate on Tier D) — agent vs **B0, B2**.
3. The agentic components are load-bearing (ablations **A1–A5**, static
   baseline **B1**); a component that a static policy matches is reported
   as not load-bearing.

---

## 2. Working agreement (standing instructions — follow exactly)

- **One commit at a time, in order.** Working, tested code at every
  checkpoint. Don't run ahead.
- **Deliverables: a zip of ONLY the changed files**, plus exact commands:
  `unzip -o ~/Downloads/<name>.zip -d .`, `python -m pytest -q`, any smoke
  or live command, then `git add <explicit files>`, `git commit -m … -m …`,
  `git push`. **No inline `#` comments in commands** (zsh at an interactive
  prompt errors on them). The user unzips into the repo themselves.
- Every commit that touches live behaviour gets a `scripts/smoke_cNN.py` or
  a probe that exercises real APIs and prints what happened.
- **Amend REFERENCE.md after every commit** with reasoning and live
  findings. Corrections are marked explicitly ("corrects C27"), never
  silently rewritten.
- Comments explain **why**, not what. Report negative results honestly.
- Tests **assert effects**, not that a decision was made. Do **mutation
  checks**: revert the fix (or monkeypatch the old behaviour) and confirm
  the test fails. Several first-draft tests in this project passed without
  the fix and had to be strengthened.
- **Read the output, not only the assertions.** Be suspicious of numbers
  that look right for the wrong reason (C28.6: a probe printed "OK" while
  every pair it counted was an artifact).
- The user often delegates decisions ("your decision"). Make the call,
  state it, record it in REFERENCE, and say how to overturn it.
- Explain findings plainly; the user wants to know *what broke and why*.

---

## 3. Environment

- Mac, Python 3.12, `.venv`, repo dir `calibrated-deep-research`.
- Keys in `.env`: `GROQ_API_KEY`, `GOOGLE_API_KEY` (Gemini),
  `TAVILY_API_KEY`. Development used live tools; **the eval uses only the
  frozen corpus** (§5).
- **Groq free tier** (confirmed from the user's console, 26 Nov):

  | model | RPM | RPD | TPM | TPD |
  |---|---|---|---|---|
  | openai/gpt-oss-120b | 30 | 1K | 8K | 200K |
  | openai/gpt-oss-20b | 30 | 1K | 8K | 200K |
  | qwen/qwen3.8-27b | 30 | 1K | 8K | 200K |

  Day reset 00:00 UTC = **05:30 IST**. **Developer (paid) tier: upgrades
  "temporarily unavailable due to high demand"** as of 26 Nov. If it opens:
  ~$0.15/$0.60 per M tokens on gpt-oss-120b, est. **$3–10** for the whole
  remaining eval, 250K TPM, no daily cap — but first do **C32.2** (§8, N3).
- Gemini judge (`gemini-3.1-flash-lite`): 15 RPM, 500 RPD, own quota.
- HF Hub "unauthenticated requests" warnings are harmless (models cached).
- Test suite: **1222 passed, 2 skipped** at C37. `ddgs` may be needed for
  `test_web_search`.

---

## 4. Architecture (as built)

**Graph** (`src/graph.py`): LangGraph `StateGraph`, hub-and-spoke. Every
node returns to `src/controller.py`, which routes deterministically (no LLM).
Nodes: premise_checker → planner → retriever (adaptive) → synthesizer →
critic → critic_router → decider → finalizer. Checkpointed with SqliteSaver.
`run_research(question, …, thread_id, trace_id, resume)`:
**since C32 it really resumes** (an interrupted thread continues via
`invoke(None)` with its budget restored; a finished thread returns without
running). Every node is wrapped (`_budgeted`) to write `budget.to_state()`
into state, so checkpoints carry budget spend and `elapsed_s`.

**State** (`src/state.py`): `ResearchState`, `SubQuestion` (per-question
retriever state: `next_route`, `next_query`, `best_f1`, `evidence_ids`,
`evidence_scores` (C27.9), `kind` core/background (C27.8), `reopened_from`
(C28.2), `contradiction_scan` (C28.5)), reducers `merge_evidence`,
`merge_contradictions` (unordered-pair dedup), `merge_gaps`.

**LLM** (`src/llm/provider.py`): roles **judgment** (gpt-oss-120b →
qwen3.8-27b → gpt-oss-20b), **mechanical** (gpt-oss-20b → gemma-4-31b-it →
gemini-3.1-flash-lite), **judge** (gemini-3.1-flash-lite). Per-role pins
`llm.pin.judgment` / `llm.pin.mechanical`. Quota ledger
`src/llm/rate_limit.py` + `scripts/quota.py` (`status`, `reset --model`,
`reset --all`). Prompt cache `src/llm/cache.py`. 429 handling: minute window
→ wait; day window → exhaust; **pinned one-model chain: minute waits don't
consume attempts, capped at `llm.retry.max_pinned_wait_s` = 600 s (C32.1)**.
`prompt_token_budget(role)` is derived from TPM (C27.3) — see N3.

**Retrieval** (`src/rag/`): bge-small dense + BM25 (`bm25s`) + RRF + bge
reranker; Qdrant local; section-aware chunking. NLI contradiction detector
(`contradiction.py`), wired at C28.5: scans each sub-question once per
terminal status over its top 8 passages; **both directions must exceed 0.60
and hub passages are excluded (C28.7)** — f5 is near-constant on this corpus
(negative result, §7).

**Premise checker** (`src/nodes/premise_checker.py`): extracts
presuppositions, retrieves per presupposition, verifies. **C30.1: evidence is
fitted to the prompt budget and trimmed only at sentence boundaries with a
`[…]` marker** (was a 700-char mid-sentence cut); attribution rule for
Related Work passages in `prompts/premise_verify.txt`.

**Critic** (`critic.py`): per-claim verdicts incl. GAP_STATEMENT (C27.5);
split-on-failure, 2 levels (C28.3); `critic_failed` → `system_failure`.

**Sufficiency** (`src/sufficiency/`): `features.py` (f1–f10; f5 from the
scan record, found ÷ pairs scored, C28.5), `model.py` (two-stage logistic;
stage 1 per sub-question excluding f6 for leakage; stage 2 on run-level
aggregates), `policy.py` (D5 rule order: premise_refuted → system_failure →
no_evidence → threshold), `scorer.py` (Calibrated / Heuristic; only core
sub-questions scored; heuristic resolves a sub-question by retrieval
sufficiency, or by verification: ≥2 SUPPORTED and entailment ≥ 0.5).

**Finalizer**: deterministic report, verification marks in text, footnotes,
per-decision shape; strips stray markers case-insensitively (C28.4).

**Eval** (`src/eval/`): `configs.py` (agent, B0–B3, A1–A5 as override sets;
`EVAL_PINS` judgment=gpt-oss-120b, mechanical=gpt-oss-20b), `components.py`
(frozen-corpus wiring, `frozen_retrieval`), `runner.py` (EvalRunner),
`baselines.py` (B0 single-shot; B2/B3 derivation), `judge.py`, `metrics.py`.

---

## 5. Decisions recorded (and why)

| decision | why | where |
|---|---|---|
| **Frozen corpus** for the eval (982 passages, collection `retrieval_eval`) | reproducible; D1 refutations checkable; no search quota | C29 |
| Every role **pinned** in eval | configs differ only in switches | C27.6/7, C32 |
| **B1 keeps D5 live** | else B1 duplicates B2, confounding agency with abstention | C32 |
| **B2 also disables skip-research-on-refuted-premise** | else B2 "answers" with no draft | C32 |
| **B2 and B3 derived** from the agent's runs (paired) | same research/evidence/draft; only the decision differs; removes run-to-run noise (a question flipped PARTIAL→ABSTAIN between runs); saves ~10 quota-days | C34 |
| B2 runs **live** only where the agent skipped research on a refuted premise | no draft to derive from | C34 |
| B3 keeps the agent's **hard rules**; its τ chosen on calibration like the agent's | swaps the scorer, not the policy; threshold-fair | C34 |
| **Independent judge** (Gemini flash-lite) | the system never grades itself | C37 |
| Grading rules fixed **before** test results | §7.1 | C37 |
| Contradiction: both directions + hub suppression; claim-level NLI is future work | data from `probe_contradiction` | C28.7 |
| Eval stays on **free tier** | Developer tier unavailable | 26 Nov |
| Commit per-question results + manifests; not cache/checkpoints/traces | evidence vs regeneratable state | C32 |

---

## 6. Benchmark (frozen)

`benchmark/questions.json` — **48 questions**: A 10 (single-source), B 10
(multi-source), C 8 (judgment across sources), D 20 (D1 false premise,
D2 underspecified, D3 no consensus, D4 unresolvable; 5 each). 101 gold
claims for A–C, **each citing the corpus passages that support it**; all
reviewed against full passage text (5 corrections, recorded in
`questions.json` → `review`). Tier D items carry `abstain_reason`,
`expected_behavior`, and per subtype: `correction` + `refutation_passage_ids`
(D1), `missing` (D2), `side_passage_ids` from ≥2 sources (D3),
`absence_patterns` verified by co-occurrence (D4).

**All five D1 false premises refute live** (`probe_premise --tier-d1`).

**Split frozen (C31)**: seed 31, stratified by tier and D subtype,
**hash `708266628ff9…`** (SHA-256 over test content + membership; the
checker fails on any post-freeze change to the test half; calibration items
remain editable). `make_split` refuses to run again without `--refreeze` —
**never refreeze now.**

Test split (24): A01 A02 A04 A07 A08 · B03 B05 B07 B08 B10 · C02 C03 C04
C05 · D1-01 D1-05 · D2-01 D2-02 D2-03 · D3-03 D3-04 · D4-02 D4-04 D4-05.

Exposure: `dev_exposed` B01 (near-copy of the smoke question) and D1-03
(drove the C30.1 fix) forced to calibration. D1-01 and D1-05 (test) were
*observed* in pre-freeze probes (passed; nothing changed) — recorded in
`pre_freeze_observations`. The Mamba-attention false premise is **excluded**
(it is the premise prompt's worked example).

Tools: `scripts/check_questions.py` (validator + `benchmark/review_questions.md`),
`tests/test_questions.py`.

---

## 7. Commit history

**Phases 0–2** (C1–C22): foundations, RAG substrate (R1–R4 retrieval table),
agentic core. See REFERENCE log.

**Phase 3** (complete):
| commit | hash | what |
|---|---|---|
| C27 | f0ecf29 | D5 decision policy, scorers, decider |
| C27.1 | db4ccf5 | revision ceiling; checkpointed plan_revisions; trace inspector |
| C27.2 | 7dbc97b | per-question retriever state on SubQuestion; critic ownership from plan |
| C27.3 | 781b574 | window-aware 429; prompt_token_budget; synthesizer fits prompt |
| C27.4 | fe599bc | citation variant normalisation; show_run.py |
| C27.5 | 0ee3ba5 | GAP_STATEMENT critic category |
| C27.6 | 43f509f | reformulation on mechanical role; judgment pin |
| C27.7 | 9bced86 | mechanical chain led by gpt-oss-20b; per-role pins |
| C27.8 | 7d67da3 | core vs background sub-questions |
| C27.9 | (in C28) | per-sub-question evidence_scores |
| C28 | 26c2a6d | finalizer; full graph; smoke_c28 |
| C28.1 | 368bb5e | premise verify: explicit absence = CONTRADICTED; probe_premise |
| C28.2 | 12af105 | reopened_from floor; verification resolves a sub-question |
| C28.3 | 2870ca8 | failed critique = system_failure; critic split-on-failure |
| C28.4 | 0266322 | **Phase 3 complete**; case-insensitive stray markers |

**Contradiction work:**
| C28.5 | 19121ed | NLI scan wired; f5 denominator fix; failed scan ≠ clean scan |
| C28.6 | 7cc8b27 | probe v2: f5 measured topic mismatch (hub artifact) |
| C28.7 | ad7d7bc | both directions + hub suppression; negative result stated |

**Phase 4** (complete):
| C29 draft/fix/final | ae6fae8, 9a2ed78, 88f6a23 | Tiers A–C, gold claims, reviewed |
| C30 | 7641d83 | Tier D (20) with evidence |
| C30.1 | 1b7048f | premise verifier saw truncated passages — fixed |
| C31 | 807922b | frozen split, hashed |
| C32 | 715ec3b | eval runner; run_research really resumes; budget survives resume |
| C32.1 | e9af977 | limit/remaining fixes; day-only quota stop; pinned minute waits |

**Phase 5** (in progress):
| C33 runs | (user commits daily) | agent calibration runs |
| C34 | d09b94d | B0 single-shot; B2/B3 derived |
| C37 code | 8f93afa | scoring: judge, rules, spot-check |

Not yet: **C33 fit**, C35 ablation runs, C36 test runs, C38–C42.

---

## 8. Open items (consolidated; most important first)

**N1 — TOP PRIORITY. Hallucination 0.484 in the first scoring run is
probably the grader.** `src/eval/judge.py::grounding` cuts each cited
passage at `PASSAGE_CHARS = 900` **mid-sentence** — 341 of 982 passages are
longer — the same defect class as C30.1 (a cut can remove the supporting
sentence). Second candidate: the judge's `max_output_tokens: 1024`; any
sentence the judge doesn't label defaults to **UNSUPPORTED**, so a truncated
judge output inflates the rate. Diagnose with `results/spot_check_calibration.md`
and `results/eval/agent/calibration/judged/*.json` (count labels returned vs
sentences). Fix: fit passages to the judge's budget and trim at sentence
boundaries (reuse `premise_checker._trim_at_sentence`); missing labels →
**UNJUDGED** (excluded, counted), not UNSUPPORTED; batch long drafts. Same
for `coverage()`: missing gold-claim labels currently default to ABSENT →
make them UNJUDGED. Then **delete the cached judged files** and re-score.
Record in REFERENCE as a grading correction *before any test result exists*
(legitimate under C37's rule).

**N2 — C33 prerequisite: records don't store the plan.** The sufficiency
model needs per-sub-question features (`plan[*].sufficiency_features`,
`evidence_scores`, `kind`) — currently only in the checkpoint DB
(`results/eval/checkpoints.sqlite`, gitignored). Add `plan` (compact: id,
kind, status, sufficiency_features) to runner records now (additive), and
recover existing records via `baselines.load_checkpoint_state`.
**Do not delete `results/eval/checkpoints.sqlite` or `results/eval/cache/`**
— they're needed for C33 features and for B2/B3 derivation of pre-C34
records.

**N3 — C32.2, only if a paid tier opens:** make the prompt budget a fixed
design constant (not derived from TPM) and update the ledger limits, so the
system's behaviour doesn't change with the billing tier mid-eval.

**N4 — Freeze before test runs:** after C33, tag (`git tag eval-freeze`);
no code changes to the system after that. Grading code may still be fixed
before test results exist, and any such change is recorded.

**N5 — Judge agreement rate:** the human marks `results/spot_check_<split>.md`;
report agreement beside the metrics (§6.2).

From the older open table (REFERENCE "Open before C33"):
- **#4 (decided)** pin gpt-oss-120b; measured **~4–5 questions/day**.
- **#5** claim ownership is by retrieval, not topic → writeup limitation.
- **#6** τ must be selected on the fitted model's scale (use
  `metrics.select_threshold`).
- **#7** comparison sub-questions that restate the parent drag core
  resolution (planner prompt) — not fixed; affects B-tier; writeup.
- **#9** critic/router batch sizing (split-on-failure handles it reactively).
- **#11** adaptive D2 executed traced as "(static)" — fix in C38 tooling.
- **#12** D4 allocates without regard to `kind` — decide with C33 data.
- **#13** nodes charge budget only `if can_spend`.
- **#14** stage 2 sees no critic evidence: add a run-level verification
  aggregate (not leaky at stage 2); decide which predictor it replaces —
  **a C33 decision**.
- **#1 remainder** heuristic scorer ignores f5 (calibrated stage 1 uses it);
  the synthesizer doesn't present disagreement as disagreement; f5 is
  near-constant (expect a ~0 coefficient — report as a measured null).

---

## 9. Plan and timeline (free tier, ~4–5 agent questions/day)

### The user's daily routine (after 05:30 IST)
```
python -m benchmark.run_eval --config agent --split calibration
git add results/eval/agent/calibration/
git commit -m "C33 (in progress): agent calibration runs"
git push
```
The runner skips completed questions, resumes interrupted ones, re-runs
`system_failure` ones on a new thread, and stops at the day quota.
`--limit 0` shows progress without LLM calls.

### Calendar (estimates)

| when | what | quota? |
|---|---|---|
| Nov 27 – ~Dec 1 | finish agent calibration runs (19 left + B01 re-run) | 120b, daily |
| in parallel, now | **N1 grading fix**, N2 records, C33 fitting code, C38 annotation helper, C39 plot scripts, writeup skeleton | no |
| ~Dec 2 | derive B2/B3 on calibration (`derive_baselines --split calibration`; ~24 confidence calls ≈ part of a day) | 120b |
| ~Dec 2–3 | **C33**: fit the sufficiency model on calibration, choose τ for the agent and B3 (`results/eval/B3/tau.json`), ECE/reliability on calibration; score calibration (`analysis.score --split calibration`) | judge only |
| ~Dec 3 | **freeze tag** | — |
| Dec 3 – ~Jan 6 | test-split runs with `--held-out`: agent (C36), B0, B1, B2-live (D1-01, D1-05), A1–A5 (C35); then `derive_baselines --split test --held-out` | 120b, daily, ~30–35 days |
| ~Jan 7–15 | C37 test scoring + spot-check; C38 MAST failure annotation; C39 plots and 4 case studies; C40 writeup (8–12 pp); C41 README + Gradio demo; C42 tests/cleanup | judge only |

**End: ~mid-January 2027** on the free tier; ~mid-December if the paid tier
opens (after N3). The bottleneck becomes chat work, not quota.

### C33 in detail (next big code task after N1/N2)
`analysis/calibrate.py`: load calibration agent records (+ plan/features via
N2), labels from C37 judgments (`correct_if_answered`), fit the two-stage
model (`src/sufficiency/model.py`), reliability diagram + ECE, select τ at
the target risk with `metrics.select_threshold` on the fitted scale (#6),
persist the model (config `sufficiency.scorer: calibrated` loads it; the
test split forces it), report coefficients (which signals predict
answerability is itself a finding). Decide #14 and #12 here. Do the same τ
selection for B3's verbalized confidence.

### Phase 6 notes
- C38: MAST (Cemri et al. 2025, 14 modes) on every wrong-outcome trace;
  `scripts/inspect_trace.py` exists; fix #11 first.
- C39: risk–coverage (agent vs B3 vs B0/B2 points), reliability diagram,
  ablation bars, R1–R4 retrieval table, MAST distribution; case studies:
  one Tier A, one where re-planning fired, one D1, one honest failure.
- C40 writeup must include: the negative contradiction result (C28.7), the
  premise-truncation finding (C30.1), run_research resume bug (C32), derived
  baselines rationale (C34), judge fairness (C37), claim-ownership limitation
  (#5), topic overlap between retrieval sub-questions and the benchmark, and
  "unit tests verify rules; only live runs exposed their interactions".

---

## 10. Results so far

Agent, calibration (pinned, heuristic scorer):

| id | decision | rule | score | LLM calls |
|---|---|---|---|---|
| A03 | ANSWER | threshold | 1.0 | 17 |
| A05 | PARTIAL | threshold | 0.67 | 17 |
| A06 | ABSTAIN | threshold | 0.0 | 14 — a false abstention |
| A09 | ANSWER | threshold | 1.0 | 13 |
| A10 | PARTIAL | threshold | 0.33 | 22 |
| B01 | ABSTAIN | system_failure | — | day quota hit; will re-run |

First scoring run (pipeline check only, n=5 decided):
A–C acc (answered) 1.000 · false abstention 0.200 · coverage 0.917 ·
**hallucination 0.484 (suspect, N1)** · AURC 0.040 · ECE 0.200 ·
15.3 LLM calls/question.

---

## 11. Command reference

| purpose | command |
|---|---|
| tests | `python -m pytest -q` |
| quota | `python -m scripts.quota status` / `reset --model M` / `reset --all` |
| eval run | `python -m benchmark.run_eval --config agent --split calibration [--limit N] [--only IDs]` |
| list configs | `python -m benchmark.run_eval --list` |
| test split (C36 only) | `python -m benchmark.run_eval --config agent --split test --held-out` |
| wrappers | `python -m benchmark.run_agent / run_baselines / run_ablations …` |
| derive B2/B3 | `python -m benchmark.derive_baselines --split calibration` (`--held-out` for test, `--no-live`) |
| score | `python -m analysis.score --split calibration [--configs …] [--no-judge] [--spot-check]` |
| validate benchmark | `python -m scripts.check_questions` |
| premise probe | `python -m scripts.probe_premise [--tier-d1]` |
| contradiction probe | `python -m scripts.probe_contradiction` (zero API calls) |
| trace / run state | `python -m scripts.inspect_trace …`, `python -m scripts.show_run …` |
| smokes | `python -m scripts.smoke_c28` (full graph, two questions) |

Outputs: `results/eval/<config>/<split>/<id>.json` + `manifest.json` +
`judged/` (committed); `results/metrics_<split>.md|json`;
`results/spot_check_<split>.md`; working state (gitignored):
`results/eval/cache/`, `results/eval/traces/`, `results/eval/checkpoints.sqlite`.

---

## 12. Lessons worth keeping (interview material)

1. Most defects were two individually-correct rules combining badly; unit
   tests verified rules, live runs exposed interactions.
2. "Unscanned" vs "clean" must be distinguishable (f5 None vs 0.0; failed
   NLI scan ≠ clean scan).
3. Numbers that look right for the wrong reason: f5's denominator capped it
   near 0.08; the contradiction probe passed on hub artifacts.
4. Truncation inverts meaning: the premise verifier saw "…In contrast, our
   analysis predicts that both quantities" and called a false premise
   SUPPORTED. Trim at sentence boundaries, mark trims.
5. A docstring is not a test: `run_research` claimed to resume and never did.
6. Paired derived baselines remove run-to-run noise from exactly the
   comparison being made.
7. An evaluation must not count its own failures as the system's (UNJUDGED;
   system_failure separate from abstention).
