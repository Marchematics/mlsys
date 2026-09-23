# R123 — Theory-shaped bilinear response head in the second family

Run date: 2026-09-23.

* Raw run: `PAMI_MUR/results/raw/R123_bilinear_second_family_<stamp>/` (immutable, logged, manifest).
* Derived: `PAMI_MUR/results/derived/R123_bilinear_second_family_summary/`.
* New modules: `bilinear_response.py`, `run_r123_bilinear.py`, `diagnose_r123_heads.py`,
  `analyze_r123.py`; tests `tests/test_bilinear_response.py`.
* R110/R112/R115 code, runs and derived artefacts are not modified; R123 imports their evaluation
  stack, pools, predictor and diagnostics unchanged.

## 0. Pre-registered prediction (written before the R123 sweep was launched)

R115's query-comparative family has genuine but small state-conditioning headroom
(pooled `H_state` = +0.0126) and its frozen scalar response head beats Cached Utility everywhere
but beats Static Utility on only 3 of 5 benchmarks. R123 replaces only the response-aware scorer
with the theory-shaped bilinear head,

    score(state, j) = <g_theta(state, candidate), d_bar_j> - lambda * kappa_j ,

where `d_bar_j` is the query-batch mean class-logit change produced by adding demonstration `j`,
`kappa_j` the curvature term, and `lambda` the multiplier on it. The prediction, registered before
the sweep:

> With the bilinear head, **forced-budget R-MUR q=4 beats Static Utility with a paired 95 % CI
> lower bound above zero on at least 4 of 5 benchmarks** (the frozen head achieved 3/5), and the
> **pooled contrast excludes zero**.

Pass/fail is reported per benchmark and pooled, with paired and class-clustered intervals, and the
response-probe R² / ranking correlation of the new head versus the frozen head is reported so the
mechanism is visible rather than assumed.

## 1. What changed, and which curvature variant is used

Everything except the response-aware scorer is inherited: cached frozen-CLIP features, the R112
redundancy-structured pool (4 k-means modes × 3 near-duplicate in-class demonstrations + 4
distractors, K = 16, B = 4), the R115 query-comparative prototype predictor, the frozen R-MUR
pipeline (cached screen → top-q shortlist → response reranking), forced budget as the primary rule
with the thresholded rule reported alongside, q ∈ {2,4,8,16}, seeds 101/202/303 and 1 602 test
episodes. Both heads are trained on the *same* cells with identical seeds, and the frozen-head
policies are retained as `ranked_response_frozen_*` so the comparison is paired.

Response block (per state, candidate): `[7 scalar summaries | d_bar (C) | mean_i ||d_i||^2 |
||d_bar||^2 | realised curvature]`, so the head can be scored under either curvature convention:

* **bound (primary, pre-registered)**: `lambda = 0.25` multiplying `mean_i ||d_i||^2`. This is the
  proven worst-case constant of Proposition 3 (`H = diag(p) - p p^T` has spectral norm ≤ 1/2, so
  the second-order term is at most a quarter of the squared response norm), giving
  `<e_y - p_A, d> - 0.25 ||d||^2 ≤ m ≤ <e_y - p_A, d>`.
* **realised (secondary ablation)**: `lambda = 0.5` multiplying
  `mean_i d_i^T H(p_{A,i}) d_i`, the exact second-order term evaluated at the *base* point. Both
  `p_A` and `d` are observable at routing time, so this quantity is observable; it is the
  second-order-exact counterpart of the worst-case bound.

The realisation of Proposition 3 for this family is that `0.25 ||d||^2` is a *worst-case* penalty:
for confident predictions the realised curvature is near zero, so the bound can be far below the
true marginal. The secondary ablation measures how much of the R123 outcome is due to that
looseness.

## 2. Unit tests

`tests/test_bilinear_response.py` (6 tests): response-block layout and curvature identities
(`kappa >= ||d_bar||^2` by Jensen, realised curvature in `[0, ||d||^2]`), explicit
`<g, d_bar> - lambda kappa` scoring, learned multiplier confined to `[0, 0.25]`, packed marginals
equal to loss differences, per-candidate response independence, and finite scores after training.

<!-- RESULTS APPENDED AFTER THE SWEEP -->
## 3. Results (15 cells, 1 602 test episodes, seeds 101/202/303, zero failures)

Raw run: `results/raw/R123_bilinear_second_family_20260923_0420`. The secondary realised-curvature
ablation (CIFAR-10 and DTD) is `results/raw/R123_bilinear_realised_20260923_0515`.

### 3.1 Verdict on the pre-registered prediction

| clause | prediction | measured | verdict |
|---|---|---|---|
| per benchmark | bilinear forced q=4 beats Static with paired CI low > 0 on **≥4/5** | **3/5**: CIFAR-100 +0.0210 [+0.0077,+0.0343], SVHN +0.1136 [+0.0767,+0.1505], EuroSAT +0.1875 [+0.1168,+0.2582]; CIFAR-10 −0.0151 [−0.0206,−0.0096], DTD −0.0044 [−0.0280,+0.0191] | **FAIL** |
| pooled | contrast excludes zero | pooled **+0.0605** [−0.0036,+0.1343] → includes zero | **FAIL** |
| (reference) vs Cached | — | **5/5** benchmarks, pooled **+0.1630** [+0.1173,+0.2078] | pass |

The frozen scalar head in the same cells scores 3/5 as well (CIFAR-100, SVHN, EuroSAT), so the
bilinear head does **not** recover the missing benchmark. It does, however, move the *hard* cases:
the paired bilinear − frozen contrast is +0.0184 [+0.0118,+0.0250] on DTD (where the frozen head
was 0.0228 *below* Static) and +0.0253 [+0.0087,+0.0419] on SVHN, at the cost of −0.0112 on
CIFAR-10 where the frozen head was already the better of the two.

### 3.2 Gate table at q=4 (forced budget primary; paired CIs; hierarchical CIs cluster by class)

| benchmark | head | gain [95 % CI] | Δ Static [CI] | Δ Cached [CI] | cls>Static | pass paired / hier |
|---|---|---|---|---|---|---|
| CIFAR-10 | bilinear | 1.6962 [1.6893,1.7031] | −0.0151 [−0.0206,−0.0096] | +0.0892 [+0.0366,+0.1418] | 0.10 | no / no |
| CIFAR-10 | frozen | 1.7074 [1.7003,1.7145] | −0.0039 [−0.0096,+0.0017] | +0.1005 [+0.0463,+0.1546] | 0.50 | no / no |
| CIFAR-100 | bilinear | 2.5452 [2.5402,2.5501] | +0.0210 [+0.0077,+0.0343] | +0.1844 [+0.1310,+0.2378] | 0.65 | **yes / yes** |
| CIFAR-100 | frozen | 2.5388 [2.5336,2.5440] | +0.0146 [+0.0015,+0.0277] | +0.1697 [+0.1190,+0.2204] | 0.54 | yes / yes |
| SVHN | bilinear | 1.6258 [1.6044,1.6472] | +0.1136 [+0.0767,+0.1505] | +0.1985 [+0.1403,+0.2567] | 0.80 | **yes / yes** |
| SVHN | frozen | 1.6005 [1.5789,1.6220] | +0.0883 [+0.0548,+0.1218] | +0.1937 [+0.1356,+0.2518] | 0.80 | yes / yes |
| EuroSAT | bilinear | 1.6434 [1.6309,1.6559] | +0.1875 [+0.1168,+0.2582] | +0.2171 [+0.1442,+0.2900] | 0.70 | **yes / yes** |
| EuroSAT | frozen | 1.6440 [1.6312,1.6568] | +0.1881 [+0.1164,+0.2597] | +0.2152 [+0.1407,+0.2898] | 0.50 | yes / yes |
| DTD | bilinear | 2.0366 [2.0219,2.0513] | −0.0044 [−0.0280,+0.0191] | +0.1089 [+0.0466,+0.1712] | 0.38 | no / no |
| DTD | frozen | 2.0182 [2.0029,2.0336] | −0.0228 [−0.0463,+0.0007] | +0.0994 [+0.0388,+0.1600] | 0.28 | no / no |

Paired bilinear − frozen head (same cells, q=4 forced): CIFAR-10 −0.0112 [−0.0133,−0.0091],
CIFAR-100 +0.0064 [+0.0025,+0.0103], SVHN +0.0253 [+0.0087,+0.0419], EuroSAT −0.0006
[−0.0084,+0.0073], DTD +0.0184 [+0.0118,+0.0250]; all heads still select exactly 4 contexts, and
the bilinear head's selections cover more distinct visual modes on DTD (2.11 vs 1.94).

### 3.3 Mechanism: the bilinear head *does* estimate the interaction term better

`head_comparison.json` (both heads retrained on identical train data per cell, evaluated on 2×4
held-out test states per cell, mean over 3 seeds):

| benchmark | Spearman(score, m) frozen | bilinear | top-1 agreement frozen | bilinear | mean regret frozen | bilinear | corr(score, kappa) frozen | bilinear |
|---|---|---|---|---|---|---|---|---|
| CIFAR-10 | −0.001 | **−0.113** | 0.25 | 0.00 | 0.018 | 0.036 | **+0.257** | −0.095 |
| CIFAR-100 | +0.621 | **+0.833** | 0.08 | **0.67** | 0.001 | 0.021 | −0.590 | −0.475 |
| SVHN | +0.533 | **+0.756** | 0.08 | **0.33** | 0.054 | 0.072 | −0.360 | −0.151 |
| EuroSAT | −0.040 | **+0.196** | 0.08 | 0.00 | 0.135 | **0.051** | +0.174 | −0.135 |
| DTD | −0.331 | **+0.203** | 0.00 | 0.00 | 0.283 | **0.092** | +0.478 | +0.210 |
| **mean** | +0.156 | **+0.375** | 0.10 | 0.20 | 0.098 | 0.054 | −0.008 | −0.129 |

The bilinear head ranks the marginal better on **4 of 5** benchmarks (mean Spearman 0.375 vs 0.156;
top-1 agreement 0.67 vs 0.08 on CIFAR-100; regret 3× lower on DTD), exactly as the theory-shaped
parameterisation intends. The exception is CIFAR-10, and the last two columns explain it: there the
frozen head's score correlates **+0.257** with the response magnitude `kappa`, i.e. on CIFAR-10 a
larger response really does mean a larger marginal, whereas on CIFAR-100/SVHN the association is
**negative** (−0.59/−0.36). The bilinear form subtracts `lambda * kappa` with `lambda >= 0`, so it
can express the negative association (and wins on CIFAR-100/SVHN/EuroSAT/DTD) but structurally
cannot express the positive one; on CIFAR-10 the fixed penalty therefore removes signal the frozen
head uses, and its selections lose 0.011 nats. The learned-multiplier variant cannot repair this
either: its admissible range `[0, 0.25]` only shrinks the penalty towards zero, it cannot flip its
sign.

### 3.4 Secondary ablation: worst-case bound versus realised curvature

`results/raw/R123_bilinear_realised_20260923_0515` (CIFAR-10 and DTD, 3 seeds, `lambda = 0.5 ·
mean_i d_i^T H(p_{A,i}) d_i`):

| benchmark | bilinear, bound (λ=0.25, mean‖d‖²) | bilinear, realised curvature | frozen head | Static |
|---|---|---|---|---|
| CIFAR-10 (mean over 3 seeds) | 1.6962 | 1.6962 | 1.7074 | 1.7113 |
| DTD (mean over 3 seeds) | 2.0366 | 2.0452 | 2.0182 | 2.0410 |

The realised curvature changes CIFAR-10 by < 1e-4 (the selected sets are identical) and DTD by
+0.0086 on average, which does not reach a consistent win over Static (1/3 seeds). So the looseness
of the worst-case bound is **not** the cause of the R123 failure; the binding constraint is the
fixed *sign* of the curvature term identified in §3.3.

<!-- LIMITATIONS AND ARTIFACTS APPENDED BELOW -->

## 4. Negative results and limitations

1. **The pre-registered prediction fails.** The bilinear head beats Static Utility on 3/5
   benchmarks (not ≥4/5) and the pooled contrast includes zero (+0.0605 [−0.0036,+0.1343]). The
   frozen scalar head also achieves 3/5, so on this criterion the theory-shaped head is not an
   improvement in benchmark count — it is a *redistribution*: it wins clearly on SVHN (+0.025
   over the frozen head) and DTD (+0.018, turning a 0.023 loss against Static into a 0.004 loss),
   and loses on CIFAR-10 (−0.011).
2. **The mechanism is not "the head fails to learn the interaction".** It learns it better on 4/5
   benchmarks (Spearman 0.375 vs 0.156; top-1 agreement 0.67 vs 0.08 on CIFAR-100). The failure is
   structural: the identity's curvature term has a *fixed negative* sign, while the data show the
   association between response magnitude and marginal utility flipping sign across benchmarks
   (+0.257 on CIFAR-10, −0.59 on CIFAR-100). A `‑λ‖d‖²` term can only help where that association
   is negative; the frozen head, which is free to weight the response features with either sign,
   adapts per benchmark.
3. **The realised-curvature ablation rules out the obvious fix.** Replacing the worst-case 0.25
   bound by the observable realised curvature barely changes CIFAR-10 (< 1e-4) and moves DTD by
   +0.0086, so "the bound is too loose" is not the explanation.
4. **The 0.25 bound is valid but conservative in the way that matters for ranking.** Proposition 3
   bounds the Hessian at 1/2 *uniformly over probability vectors*; the realised curvature vanishes
   as predictions become confident, which is exactly the regime where the prototype predictor
   operates (4 relevant demonstrations give 0.998–1.000 accuracy in R115). The bound therefore
   under-states the marginal by an amount comparable to the first-order term, which is why the
   penalty cannot be the dominant term in a *ranking* score even though it is a valid *bound*.
5. **Utility is still dominated by the first demonstration.** The R115 structure is inherited
   unchanged: `H_state` is +0.0126 pooled and only CIFAR-10/EuroSAT have > 1 % headroom, so even a
   perfect reranker could only recover ~0.05 nats on CIFAR-10. The gate's margin against Static
   Utility (0.02–0.19 nats on the passing benchmarks) is of the same order as the total
   state-conditioned headroom, which puts a ceiling on what any scorer can achieve on the
   benchmarks where the oracles do not diverge.
6. **The R² calibration numbers in `head_comparison.json` are not informative** (both heads show
   large negative R²): the affine calibration is fitted on train-split states and applied to
   test-split states whose marginal scale differs, so only the *within-state ranking* statistics
   (Spearman, top-1, regret) should be read. The permutation-free design of that diagnostic is
   otherwise sound (identical data, identical seeds, held-out states).
7. **Scope of the secondary ablation.** The realised-curvature variant was run on CIFAR-10 and DTD
   only (6 cells) as a mechanism check, not as a full second sweep; the full 5-benchmark,
   3-seed version is not claimed.

## 5. Reproduction

```bash
cd /root/icl_ess_threshold/PAMI_MUR
python -m pytest tests/test_bilinear_response.py -q                    # 6 tests

python experiments/demo_selection/run_r123_bilinear.py \
    --benchmarks cifar10,cifar100,svhn,eurosat,dtd --seeds 101,202,303 \
    --out-root results/raw/R123_bilinear_second_family_<stamp>          # bound curvature (primary)
python experiments/demo_selection/run_r123_bilinear.py \
    --benchmarks cifar10,dtd --seeds 101,202,303 --curvature-mode realised \
    --out-root results/raw/R123_bilinear_realised_<stamp>               # secondary ablation

python experiments/demo_selection/diagnose_r123_heads.py \
    --run-root results/raw/R123_bilinear_second_family_<stamp> \
    --out results/derived/R123_bilinear_second_family_summary/head_comparison.json
python experiments/demo_selection/analyze_r123.py \
    --run-root results/raw/R123_bilinear_second_family_<stamp> \
    --out-root results/derived/R123_bilinear_second_family_summary
```

## 6. Artifacts

| path | content |
|---|---|
| `experiments/demo_selection/bilinear_response.py` | bilinear head (`<g, d_bar> − λ kappa`), packed response block, training |
| `experiments/demo_selection/run_r123_bilinear.py` | R123 driver (both heads trained per cell; frozen-head reference policies) |
| `experiments/demo_selection/diagnose_r123_heads.py` | paired head comparison (Spearman, top-1, regret, curvature correlation) |
| `experiments/demo_selection/analyze_r123.py` | gates, pooled contrasts, head-vs-head tables (reuses R112 aggregation) |
| `tests/test_bilinear_response.py` | 6 tests: layout/curvature identities, bilinear scoring, multiplier bounds, packed marginals, per-candidate independence, finite training |
| `results/raw/R123_bilinear_second_family_20260923_0420/` | 15 cells, `run.log`, manifest, zero failures |
| `results/raw/R123_bilinear_realised_20260923_0515/` | 6 cells, realised-curvature ablation |
| `results/raw/R123_smoke/`, `R123_smoke_learned/` | smoke runs (tiny CIFAR-10 cells, bound and learned multiplier) |
| `results/derived/R123_bilinear_second_family_summary/` | `summary.json`, `gate_table.csv`, `report_tables.md`, `head_comparison.json`, `provenance.json` |
