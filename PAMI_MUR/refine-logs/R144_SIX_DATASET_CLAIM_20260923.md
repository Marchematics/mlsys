# R144 — Six-dataset headline corrected: R-MUR wins 4 of 6, ties 1, loses 1

**Date:** 2026-09-23
**Affects:** PAMI, TKDE and KBS manuscripts (all three carried the same claim).

## What was wrong

All three papers summarised the six-dataset ridge matrix as a uniform win:

* PAMI abstract: "learned marginal-utility routing beats every one of thirteen
  competitors".
* PAMI experiments: "R-MUR is the best method on every dataset and has the best
  average rank"; the `tab:deployable` behind it listed only five methods and so
  awarded an average rank of 1.00.
* TKDE abstract / conclusion / introduction: "beats thirteen competitors";
  "roughly doubling the strongest non-selection baseline".
* TKDE experiments: "puts the response-aware router first on five datasets and
  within rounding of first on the sixth".
* KBS abstract: "consistent improvements ... on every benchmark"; introduction:
  "performs best on every benchmark"; experiments: "R-MUR obtains the highest
  mean gain on every dataset".
* The KBS `tab:deployable` **omitted the pool-everything policy altogether**,
  which is the strongest competitor on every dataset.

The verified numbers were already in
`results/derived/R134_six_dataset_matrix/six_dataset_matrix.json`; the prose and
the KBS table did not reflect them.

## Verified result (paired, 96 cells per dataset, 32 targets x 3 seeds)

R-MUR against the best competitor. The best competitor is the trivial
pool-everything policy on **all six** datasets:

| dataset | R-MUR | pool all | paired Δ | 95% CI | verdict |
|---|---|---|---|---|---|
| METR-LA | +.0146 | +.0080 | +.0067 | [+.0044, +.0089] | ahead |
| PEMS-BAY | +.0557 | +.0254 | +.0303 | [+.0232, +.0378] | ahead |
| PEMS03 | +.0269 | +.0256 | +.0013 | [+.0002, +.0024] | ahead (marginal) |
| PEMS04 | +.0357 | +.0365 | −.0008 | [−.0031, +.0013] | tie |
| PEMS07 | +.0284 | +.0242 | +.0041 | [+.0025, +.0058] | ahead |
| PEMS08 | +.0333 | +.0359 | −.0026 | [−.0036, −.0016] | **below** |

**Significantly ahead on four of six, tied on one, significantly below on one.**

Secondary facts also verified:

* R-MUR beats every *specialised* policy (relevance, MMR, DPP, facility
  location, k-center, k-means representatives, random, mutual information) on
  all six datasets by point estimate.
* Relevance and MMR are **significantly harmful** (paired against using no
  context) on four of six datasets — METR-LA, PEMS-BAY, PEMS03, PEMS07 — and
  weakly positive on the two dense-flow ones. DPP is harmful on two.
* Pool-all is beneficial on all six, as is R-MUR.
* In the KBS sixteen-method table, R-MUR has the best average rank (1.83) and the
  lowest negative-transfer rate (.335), but is *not* top on every dataset: the
  one-shot response-aware variant exceeds it on PEMS04, PEMS07 and PEMS08, and
  pool-all exceeds it on PEMS04 and PEMS08.

## Fixes applied

* `PAMI_MUR/paper/main.tex` — abstract now states four ahead / one tied / one
  below, and that the strongest competitor is the trivial policy.
* `PAMI_MUR/paper/sections/6_experiments.tex` — `tab:deployable` rebuilt from the
  matrix JSON with **Pool all 16 added** (average ranks recomputed over six
  rows: R-MUR 1.33, pool 2.00, cached 3.00, static 3.67, MMR 5.33, relevance
  5.67), caption updated, paragraph rewritten with the paired intervals, and a
  provenance comment added above the table.
* `TKDE_MUR/paper/main.tex`, `sections/1_introduction.tex`,
  `sections/7_experiments.tex`, `sections/9_conclusion.tex` — same corrections.
* `KBS_MUR/scripts/build_kbs_result_tables.py` — `pool_all` added to
  `DEPLOYABLE`; tables regenerated, so `tab_deployable.tex` now contains the
  "Pool all 16" row and rank column computed over sixteen methods.
* `KBS_MUR/paper/main.tex`, `sections/1_introduction.tex`,
  `sections/6_experiments.tex` — superlatives replaced with the paired result,
  including an explicit statement of where the method loses.

## Why this matters for the paper's argument

The corrected claim is *weaker but more useful*: the thing to beat in this
problem is not a sophisticated selection heuristic but the trivial decision to
use every candidate, and the method does beat it — four times out of six, not
six. Stating the PEMS08 loss also gives the boundary an honest anchor in the
estimable regime, which pairs with the regime-B and pool-size findings rather
than contradicting them.

## Reproduction

Every number above is recomputed from
`KBS_MUR/results/raw/R080b_multitarget_rmur_s{1,202,303}_*` (learned policies)
merged with `PAMI_MUR/results/raw/R125_ridge_baselines_*` and
`PAMI_MUR/results/raw/R133_ridge_baselines_*` (competitors) on identical
(target, seed) cells; the matrix artifact is
`PAMI_MUR/results/derived/R134_six_dataset_matrix/`.
