# TKDE-MUR — active consolidated manuscript

This is the **primary active submission**. The earlier PAMI candidate has been
retired: every advantageous result developed there that is not part of the
submitted IPM companion is now material for this TKDE manuscript.

The manuscript is intentionally not a diagnosis-only paper. Its identity is:

> **response-aware context selection + general marginal-utility theory +
> identifiability mechanism + mechanism-derived adaptation.**

The paper should open with the strong positive regime, use theory to explain why
predictor response is the right signal, then show the regime transition under
stronger predictors and convert that mechanism into an adaptive deployment
procedure.

## Live structure

| file | role |
|---|---|
| `paper/main.tex` | title, abstract, inputs |
| `paper/sections/1_introduction.tex` | strong-method framing, theory/mechanism contributions, companion disclosure |
| `paper/sections/2_related_work.tex` | adaptive information/data selection, acquisition, routing, observability |
| `paper/sections/3_problem.tex` | state-conditioned marginal utility and structural opportunity |
| `paper/sections/4_theory.tex` | smooth-loss expansion, exact Bregman identity, CE bound, response sufficiency, regret |
| `paper/sections/5_diagnostics.tex` | identifiability diagnostics and measurable decision signals |
| `paper/sections/6_adaptive.tex` | R-MUR/theory-shaped head, novelty adaptation, stopping and deployment procedure |
| `paper/sections/7_experiments.tex` | theory validation, identifiable regime, strong-backbone transition, pool geometry, second task family, compute |
| `paper/sections/8_discussion.tex`, `9_conclusion.tex`, `10_appendix.tex` | interpretation, scope, proofs and detailed validation |

## Evidence to use

Use **all non-submitted advantages** from `../PAMI_MUR`:

- P1--P4 general response--utility theory;
- numerical validation of the Bregman identity and CE sandwich;
- theory-shaped bilinear and signed response heads;
- recent-selector matrices beyond the submitted companion;
- strong nonlinear subset-capable backbone and DeepSets confirmation;
- expert-strength/adaptation ladder;
- candidate-variance, response-predictability and chance-AUC diagnostics;
- pool-size 16→128 scaling and shortlist-removal test;
- pool-geometry/novelty mechanism and label-free adaptive rule;
- harmful-pool and budget-direction experiments;
- policy-gradient and response-geometry alternatives;
- CLIP demonstration-selection family and forced-budget analysis;
- latency, expert-row and memory measurements;
- pilot-power, time-shift and split-half deployment validation.

The old `PAMI_MUR/paper/` is not an independent paper anymore. It is a source
of evidence/prose to be folded into TKDE and then left frozen for provenance.

## What not to reuse as TKDE's main evidence

The submitted IPM companion is frozen. Do not make its submitted six-dataset
multi-target tables, shortlist/stopping figures or basic method narrative the
headline evidence of TKDE. Shared notation and the basic R-MUR primitive may be
summarised and disclosed, but TKDE's substantive claims must be supported by the
new theory, new regimes, new baselines, new mechanisms and new deployment
experiments above.

## Target narrative

```text
state-conditioned utility
        ↓
response–utility theory
        ↓
response-aware / theory-shaped selection
        ↓
strong identifiable-regime result
        ↓
predictor-strength identifiability collapse
        ↓
pool-geometry mechanism
        ↓
adaptive rule + deployment protocol
```

The intended headline is not “routing sometimes fails”. It is:

> **Response-aware selection is powerful when marginal utility is identifiable;
> the theory predicts what signal makes it identifiable, the experiments show
> how predictor strength and pool geometry change that signal, and the same
> mechanism yields an actionable selection rule.**


## PDF rebuild note

The LaTeX sources are the authoritative version after consolidation. The tracked
`paper/main.pdf` may lag the source until the next local/Overleaf compile; do
not use the binary snapshot for submission before rebuilding it from
`paper/main.tex`.
