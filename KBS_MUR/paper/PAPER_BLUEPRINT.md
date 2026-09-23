# KBS Paper Blueprint

## Working title

**Beyond Standalone Utility: Marginal-Utility Routing for Multi-Context Prediction**

## Abstract draft

Auxiliary contexts can improve prediction, but selecting them by relevance or
standalone value can waste a limited context budget. A context that helps on
its own may add little after similar information has already been selected. A
less similar context may add the missing information and improve the current
prediction. We formulate context selection as a decision problem based on the
loss reduction obtained by adding a candidate to the selected context set. We
introduce a router that learns this state-dependent value from observed
prediction outcomes and updates its choices after each selection. The router
also stops when the remaining candidates have no reliable positive value. This
formulation separates candidate relevance from incremental predictive value
and supports explicit control of redundancy, harmful transfer, and context
cost. We analyze the resulting decision error and evaluate the method in
controlled multi-context prediction and an electricity-load task.

This paragraph is the only abstract source until results are audited. It has
no numerical claims, internal run labels, version strings, or unintroduced
abbreviations.

## Publication terminology audit

| Internal or overly compressed term | First-use publication wording | Keep internal label in manuscript? |
|---|---|---:|
| `MUR` | “marginal-utility router” after defining the term | Yes, after first use |
| `MUR-light` | “embedding-based router” | No |
| `MUR-full` | “exact-counterfactual diagnostic” | No |
| `expert-train` / `router-label` | “expert-training split” / “router-label split” | No |
| `Gate A/B/C/D/N`, `R021` | omit; report the corresponding experiment by its scientific purpose | No |
| `NTR` | “negative-transfer rate” at first use | Table shorthand only |
| `MAE` | “mean absolute error” at first use | Table shorthand only |
| `LCB` / `UCB` | “lower / upper utility bound” | Table shorthand only |
| `MMR` | “maximum marginal relevance” with citation and definition | After first use |
| `roll-in` | “one validation-stage state-augmentation pass” | No |
| `seed` | “independent training run” | No in abstract; methods may state it |
| `K`, `B`, `alpha` | define candidate-pool size, context budget, and calibration level | Symbols after definition |

The manuscript will not mention GPU model names, cache paths, commit IDs,
software version strings, or unverified expansion datasets. Experiment logs
retain those details for auditability.

## One-sentence paper claim

The predictive value of auxiliary knowledge is a state-dependent marginal. We
separate the structural headroom created by selected-set conditioning from the
utility predictability available to a practical router.

## Main figures

### Figure 1 — Problem phenomenon and novelty stress

- **A** candidate pool: relevance and standalone utility before selection.
- **B** the same candidates after selecting `C1`; redundant `C2` collapses and complementary `C3` remains valuable.
- **C** relevance versus observed marginal utility, with useful/harmful quadrants.
- **Required evidence**: relation-positive candidates; Hashimoto Incremental,
  Static Utility, Set-Coverage/MMR, and MUR-light shown together.

### Figure 2 — Training and deployment architecture

- Top: offline frozen-expert label generation from `loss(f_A)-loss(f_{A+j})`.
- Bottom: MUR-light embedding-only inference with cached candidates, DeepSets
  state, calibrated interval, select/stop, and state update.
- Dashed side path: MUR-full exact-delta diagnostic.

### Figure 3 — Controlled mechanism

- **A** redundancy ratio → loss / utility recovery.
- **B** candidate count → utility recovery and duplicate selections.
- **C** budget → gain and actual contexts used.
- **D** gain–harm frontier under stopping policies.

### Figure 4 — Real-data gain–cost frontiers

- Main submission begins with Electricity panel.
- Traffic and Activity panels are added only after gates pass.
- Horizontal axis uses measured total cost or average selected contexts; both
  are reported when they lead to different conclusions.

### Figure 5 — Auditable routing trace

One held-out Electricity episode: predicted interval for each candidate at
each state, selected candidate, changed utilities, and final stop. Static
utility/relevance trace shown beside MUR.

## Main tables

### Table 1 — Dataset and protocol contract

Task, split unit, anchor, candidate types, K/B, expert, primary metric, label
cache size, and no-overlap audit.

### Table 2 — Main prediction and decision results

Base-only, Pool-all, Similarity, Static/Hashimoto, Coverage, MUR-light, and
Oracle. Columns: primary task metric, NTR, utility recovery, average contexts,
latency, and online expert calls.

### Table 3 — Mechanism ablations

No set state, relevance target, forced budget, MUR-full exact delta, ranking
loss, and optional single roll-in. High-redundancy and high-harm settings only.

### Table 4 — Calibration, robustness, and cost

K/B/lambda/alpha, interval coverage by state source, conservative/uncertain stop,
utility MAE, memory, offline label cost, and online latency.

## Section contract and page budget

| Section | Pages | Required argumentative move |
|---|---:|---|
| 1 Introduction | 2.0 | More context can hurt; relevance and `u(j|empty)` miss state dependence; define MUR and contributions |
| 2 Related Work | 2.5 | Negative transfer; routing/MoE; selective prediction; context selection; explicit closest-work contract |
| 3 Decision Problem | 3.0 | Define `F(A)`, `m(j|A)`, realized target, squared-loss identity, and non-goals |
| 4 MUR | 4.0 | Frozen expert, four partitions, MUR-light, calibration, sequential policy, cost |
| 5 Decision Analysis | 2.5 | One-step bound, per-state certificate, select/stop intervals; submodular result in appendix |
| 6 Experiments | 8.0 | Gates, controlled mechanism, Electricity, calibration/cost, ablations |
| 7 Discussion | 1.5 | Expert dependence, label cost, state shift, pure complementarity, no universal submodularity |
| 8 Conclusion | 0.5 | Auxiliary knowledge has state-dependent marginal value |

## Introduction paragraph jobs

1. External contexts are abundant; more context can be redundant or harmful.
2. Relevance and standalone value answer an intrinsic candidate question, not
   the incremental decision after other knowledge is selected.
3. Define state-conditioned marginal predictive utility using before/after
   end-task loss.
4. Introduce MUR-light, outcome-supervised training, sequential re-scoring,
   and calibrated stop without task IDs or relation priors.
5. State four contributions: formulation, method, decision analysis, evidence.

## Discussion boundary

Greedy MUR can miss pure complementarity where every singleton marginal is
nonpositive but a bundle is useful. This is a formal limitation, illustrated
by a constructed case. Pair look-ahead, bundle utility, and beam search are
future work, not emergency additions to the submitted method.

---

> **Superseded (2026-09-21).** The submitted manuscript follows the structure
> in `sections/6_experiments.tex` with the tables in `tables/` and the figures
> mapped in `figures/SOURCES.md`. This file is retained as a planning record.
