# TKDE-MUR (repositioned 2026-09-23)

Primary submission target. The manuscript's identity is now the **identifiability
mechanism**, not the routing method: *why predictor response determines marginal
utility, when that utility can be estimated at all, why a stronger predictor
makes routing fail, and how to select once identifiability is measured.*

## Live files

| file | role |
|---|---|
| `paper/main.tex` | title, abstract, inputs |
| `paper/sections/1_introduction.tex` | identifiability framing and contributions |
| `paper/sections/2_related_work.tex` | selection, retrieval, data selection, observability |
| `paper/sections/3_problem.tex` | `m(j|A)`, structural headroom (unchanged problem definition) |
| `paper/sections/4_theory.tex` | Propositions 1-4 + identifiability corollary + regret bound |
| `paper/sections/5_diagnostics.tex` | the three measurements (screen, predictability probes, novelty) |
| `paper/sections/6_adaptive.tex` | adaptive rule, response-aware routing, head design, stopping |
| `paper/sections/7_experiments.tex` | regimes A/B/C, competitor matrix, budget geometry, compute |
| `paper/sections/8_discussion.tex`, `9_conclusion.tex`, `10_appendix.tex` | boundary, proofs, validation |

## Frozen files (kept for provenance, no longer `\input`)

`sections/4_method.tex`, `sections/5_analysis.tex`, `sections/6_experiments.tex`,
`sections/7_discussion.tex`, `sections/8_conclusion.tex` are the previous
method-first draft. A full copy is in `paper_frozen_20260923/`.

## Evidence base

All numbers come from `PAMI_MUR/results/{raw,derived}` (regimes A/B/C, competitor
matrices, pool geometry, budget curve, diagnostics, head ablations) plus the
frozen six-dataset ridge evidence described in `paper_frozen_20260923/`.

## Separation from the companion manuscripts

* IPM submission (`Response-Aware Marginal-Utility Routing for Multi-Context
  Prediction`): method + ridge-traffic main line. Unchanged.
* PAMI candidate: method-centric with the PAMI-level generality additions.
* This manuscript: diagnosis-centric --- theory, measurement protocol, adaptive
  rule. Its claims are about *when* selection works, not about being the best
  ranker.
