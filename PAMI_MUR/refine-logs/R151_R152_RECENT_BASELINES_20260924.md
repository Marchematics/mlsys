# R151 / R152 — Two recent-method baselines, and a theory prediction that fails

**Date:** 2026-09-24

Two gaps in the comparison set were closed this round, and one of them turned
into a sharp negative result for the theory's own limit case.

## R151 — the degenerate-regime prediction is refuted

**The prediction.** The squared-loss identity gives
`E[m(j|A)|x,A] = ⟨g_A(x), d_j⟩ − ‖d_j‖² − λc_j` with `g_A = 2(μ − f_A)`. When
the anchor is strong and adapted, the identifiable coefficient `g_A` is
near zero because `μ − f_A` is dominated by irreducible noise. The score then
collapses to `−‖d_j‖²`, so the optimal action should be to select the
candidates that *perturb the predictor least*. This is a sharp, falsifiable
prediction for regime B, and it had never been tested: the survey listed the
response-geometry policies (`min_disturbance_static`, `min_disturbance_greedy`,
`max_disturbance_static`, `consensus_alignment`, `anti_consensus_alignment`,
`last_k`) as implemented, but no raw run existed for the neural expert.

**Design.** `run_response_baselines.py` on the R101 expert, strong-backbone
protocol (16 targets, seeds 101/202/303, 4,000/1,000/2,000 episodes, ten
adaptation epochs, `K=16`, budget 4), both datasets. The runner stores only the
geometry policies, so the analysis merges it with
`R103_strong_backbone_gate_.../<dataset>` — the *same* targets, seeds and
protocol — which supplies pooling, the learned routers, the classical rules and
the oracles.

**Result: the prediction fails on both datasets.**

| | METR-LA | PEMS-BAY |
|---|---|---|
| pool all | **+.0133** | **+.0224** |
| `min_disturbance_static` | +.0026 | +.0037 |
| paired Δ vs pooling | **−.0106 [−.0163, −.0051]** | **−.0188 [−.0227, −.0150]** |
| rank among the 18 non-oracle policies | 13th | 16th |

The least-perturbing policy is the best of the *six geometry policies* on
METR-LA, so the sign of the theory's limit is visible — but it is
**significantly below pooling on both datasets**, and on PEMS-BAY it is beaten
by fourteen of the eighteen policies.

**Interpretation, and why it matters.** A score that only avoids harm cannot
create value: `−‖d_j‖²` is largest exactly for the candidates that barely move
the predictor, which are also the candidates that contribute least. The
interaction term `⟨g_A, d_j⟩` is where the value lives, and the regime is
defined by that term being unidentifiable. So the theory's degenerate case is
*correct as a statement about the score* and *useless as a policy* — which
locates the regime-B boundary in the absence of the interaction term rather
than in the choice of scoring function. This closes the last "but what if you
score it differently" objection with a resolved negative result.

## R152 — a fourteenth competitor: DELIFT-style

**What it is.** DELIFT (ICLR 2025) combines a submodular coverage objective
with k-means representatives. The paper previously implemented the two
components separately (`facility_location`, `kmeans_representatives`); the
survey implied the composition was covered, which it was not.

**A degeneracy found and fixed.** The first implementation used
`n_clusters = budget`, in which case the medoids alone fill the budget and the
policy is *numerically identical* to `select_kmeans_representatives` — verified
by comparing per-cell gains, which matched to the last digit. The pool is now
over-segmented (`n_clusters = 2 × budget`), so the coverage stage actually
chooses among representatives. Post-fix the two policies differ (max per-cell
difference `.081` on METR-LA, `.210` on PEMS-BAY). The first run
(`R150_ridge_delfit`) is retained on disk as the record of the degenerate
version and must not be cited.

**Result (32 targets × 3 seeds = 96 cells per dataset).**

| dataset | DELIFT-style | pool all | paired Δ vs pooling |
|---|---|---|---|
| METR-LA | −.0117 | +.0080 | −.0196 [−.0241, −.0155] |
| PEMS-BAY | +.0042 | +.0254 | −.0212 [−.0297, −.0128] |
| PEMS03 | +.0071 | +.0256 | −.0185 [−.0202, −.0168] |
| PEMS04 | +.0135 | +.0365 | −.0230 [−.0258, −.0204] |
| PEMS07 | +.0062 | +.0242 | −.0181 [−.0207, −.0156] |
| PEMS08 | +.0150 | +.0359 | −.0209 [−.0234, −.0183] |

**Significantly below pooling on all six datasets**, every interval resolved.
The competitor count rises from thirteen to fourteen and no verdict changes:
the strongest competitor remains the trivial pool-everything policy on five of
six datasets, and greedy mutual information is the best non-trivial rule.

## Provenance

* `results/raw/R151_response_geometry_20260924_0910/{METRLA,PEMSBAY}`
* `results/raw/R152_ridge_delfit_v2_20260924_1000/<dataset>`
* `results/raw/R150_ridge_delfit_20260924_0900` — degenerate first version, do not cite
* `results/derived/R151_response_geometry/response_geometry.json`
* `results/derived/R152_delfit_competitor.json`
* `experiments/analyze_response_geometry.py`
