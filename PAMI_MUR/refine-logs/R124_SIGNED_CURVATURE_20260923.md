# R124 — Signed, unconstrained curvature coefficient (second family)

Run date: 2026-09-23.

* Raw run: `PAMI_MUR/results/raw/R124_signed_curvature_<stamp>/` (immutable, logged, manifest).
* Derived: `PAMI_MUR/results/derived/R124_signed_curvature_summary/`.
* New modules: `signed_curvature.py`, `run_r124_signed.py`, `diagnose_r124_heads.py`,
  `analyze_r124.py`; tests `tests/test_signed_curvature.py`.
* R110/R112/R115/R123 code, runs and derived artefacts are not modified; R124 imports their
  stacks unchanged and trains the frozen and fixed-sign bilinear heads as paired baselines on the
  *same* cells.

## 0. Pre-registered predictions (written before the R124 sweep was launched)

R123 diagnosed a *sign constraint*: the theory-shaped bilinear head scores
`<g, d_bar> - lambda * kappa` with `lambda >= 0` (fixed at 0.25, or learned inside `[0, 0.25]`),
while the R123 data show `corr(score, kappa)` = +0.257 on CIFAR-10 but −0.590 on CIFAR-100 and
−0.360 on SVHN. A fixed-sign penalty can express the negative association but not the positive one.
R124 adds a third head,

    score(state, j) = <g_theta(state, candidate), d_bar_j> + c_theta(state) * kappa_j ,

with `c_theta` **signed and unconstrained** (one extra output unit, initialised at −0.25 so training
starts from the fixed-sign solution). Registered predictions:

> **(i)** the signed head's mean within-state Spearman correlation with the true marginal is at
> least as high as the fixed-sign bilinear head's on **at least 4 of 5** benchmarks;
> **(ii)** forced-budget R-MUR q=4 with the signed head beats Static Utility with a paired 95 % CI
> lower bound above zero on **at least 4 of 5** benchmarks **and** on the pooled analysis.

Mechanism prediction: the learned `c` moves towards zero (or becomes positive) exactly on the
benchmarks where the fixed sign hurt (CIFAR-10). If `c` stays near −0.25 everywhere, the diagnosis
is wrong; if `c` moves but performance does not, the sign constraint was real but not binding.

Scope note: for squared loss the exact response–utility identity *fixes* the curvature coefficient
at `c = −1` (the second-order term is realised exactly, not bounded), so a learned signed `c` is
only meaningful for the cross-entropy family; the traffic experiments keep the exact form. `c` is
reported per benchmark with its mean, spread, fraction positive, and tests against −0.25 and 0.

## 1. What changed

Inherited unchanged: cached frozen-CLIP features, the R112 redundancy-structured pool (4 k-means
modes × 3 near-duplicate in-class demonstrations + 4 distractors, K = 16, B = 4), the R115
query-comparative prototype predictor, the frozen R-MUR pipeline (cached screen → top-q shortlist →
response reranking), forced budget as the primary rule with the thresholded rule alongside,
q ∈ {2,4,8,16}, seeds 101/202/303, 1 602 test episodes. The packed response block is R123's
(`[7 scalars | d_bar (C) | mean_i ||d_i||^2 | ||d_bar||^2 | realised curvature]`), so all three
heads consume identical features.

Three heads are trained per cell with identical seeds and data:

| head | score | policies |
|---|---|---|
| **signed (primary)** | `<g_theta, d_bar> + c_theta(state) * mean_i‖d_i‖²`, `c` unconstrained | `ranked_response_*` |
| fixed-sign bilinear | `<g_theta, d_bar> − 0.25 * mean_i‖d_i‖²` | `ranked_response_bilinear_*` |
| frozen scalar | 7-dim response MLP (R110/R112/R115 head) | `ranked_response_frozen_*` |

## 2. Unit tests

`tests/test_signed_curvature.py` (5 tests): exact `<g, d_bar> + c * kappa` identity;
`init_curvature = −0.25` reproduces the fixed-sign score; the coefficient is free to take values
outside `[−0.25, 0]` (a forced +2.0 is representable and changes the score by exactly `c * kappa`);
packed marginals consistent; training moves `c` away from its initialisation and produces finite
scores.

<!-- RESULTS APPENDED AFTER THE SWEEP -->
## 3. Results (15 cells, 1 602 test episodes, seeds 101/202/303, zero failures)

Raw run: `results/raw/R124_signed_curvature_20260923_0537`.

### 3.1 Verdict on the pre-registered predictions

| # | prediction | measured | verdict |
|---|---|---|---|
| (i) | signed head's mean within-state Spearman ≥ fixed-sign bilinear head's on **≥4/5** benchmarks | **2/5** (CIFAR-10 −0.112 vs −0.113, SVHN +0.820 vs +0.756); mean over benchmarks **+0.293 vs +0.375** | **FAIL** |
| (ii) | forced q=4 beats Static Utility (paired CI low > 0) on **≥4/5** benchmarks and pooled | **2/5** (EuroSAT +0.1739 [+0.1025,+0.2453], SVHN +0.1058 [+0.0665,+0.1451]); CIFAR-100 +0.0088 [−0.0045,+0.0220], CIFAR-10 −0.0152, DTD −0.0094. Pooled **+0.0528 [−0.0081,+0.1236]** | **FAIL** |
| (reference) | vs Cached Utility | **5/5** benchmarks, pooled **+0.1553** [+0.1128,+0.1957] | pass |

### 3.2 Three-way gate table (forced budget, q=4, paired CIs)

| benchmark | head | gain [95 % CI] | Δ Static [CI] | Δ Cached [CI] | cls>Static | pass paired / hier |
|---|---|---|---|---|---|---|
| CIFAR-10 | signed | 1.6961 [1.6892,1.7030] | −0.0152 [−0.0207,−0.0097] | +0.0888 [+0.0361,+0.1415] | 0.10 | no / no |
| CIFAR-10 | bilinear | 1.6962 [1.6893,1.7031] | −0.0151 [−0.0206,−0.0096] | +0.0892 [+0.0366,+0.1418] | 0.10 | no / no |
| CIFAR-10 | frozen | 1.7074 [1.7003,1.7145] | −0.0039 [−0.0096,+0.0017] | +0.1005 [+0.0463,+0.1546] | 0.50 | no / no |
| CIFAR-100 | signed | 2.5329 [2.5266,2.5392] | +0.0088 [−0.0045,+0.0220] | +0.1770 [+0.1244,+0.2296] | 0.47 | no / no |
| CIFAR-100 | bilinear | 2.5452 [2.5402,2.5501] | +0.0210 [+0.0077,+0.0343] | +0.1844 [+0.1310,+0.2378] | 0.65 | **yes / yes** |
| CIFAR-100 | frozen | 2.5388 [2.5336,2.5440] | +0.0146 [+0.0015,+0.0277] | +0.1697 [+0.1190,+0.2204] | 0.54 | yes / yes |
| SVHN | signed | 1.6180 [1.5966,1.6394] | +0.1058 [+0.0665,+0.1451] | +0.1911 [+0.1334,+0.2488] | 0.70 | **yes / yes** |
| SVHN | bilinear | 1.6258 [1.6044,1.6472] | +0.1136 [+0.0767,+0.1505] | +0.1985 [+0.1403,+0.2567] | 0.80 | yes / yes |
| SVHN | frozen | 1.6005 [1.5789,1.6220] | +0.0883 [+0.0548,+0.1218] | +0.1937 [+0.1356,+0.2518] | 0.80 | yes / yes |
| EuroSAT | signed | 1.6298 [1.6168,1.6428] | +0.1739 [+0.1025,+0.2453] | +0.2065 [+0.1335,+0.2795] | 0.40 | yes / no |
| EuroSAT | bilinear | 1.6434 [1.6309,1.6559] | +0.1875 [+0.1168,+0.2582] | +0.2171 [+0.1442,+0.2900] | 0.70 | yes / yes |
| EuroSAT | frozen | 1.6440 [1.6312,1.6568] | +0.1881 [+0.1164,+0.2597] | +0.2152 [+0.1407,+0.2898] | 0.50 | yes / no |
| DTD | signed | 2.0316 [2.0166,2.0466] | −0.0094 [−0.0328,+0.0139] | +0.1031 [+0.0410,+0.1652] | 0.34 | no / no |
| DTD | bilinear | 2.0366 [2.0219,2.0513] | −0.0044 [−0.0280,+0.0191] | +0.1089 [+0.0466,+0.1712] | 0.38 | no / no |
| DTD | frozen | 2.0182 [2.0029,2.0336] | −0.0228 [−0.0463,+0.0007] | +0.0994 [+0.0388,+0.1600] | 0.28 | no / no |

The signed head is **uniformly ≤ the fixed-sign bilinear head** on the gate criterion in every
benchmark (CIFAR-100: 0.021 → 0.009 and the pass is lost; EuroSAT: 0.188 → 0.174 and the
class-clustered pass is lost; SVHN 0.114 → 0.106; CIFAR-10 unchanged at −0.015; DTD −0.004 →
−0.009), and it improves on the frozen head only on SVHN and DTD-marginally.

### 3.3 Mechanism: the sign constraint was real, and releasing it did not help

The learned coefficient (held-out test states, mean over 3 seeds; `head_comparison.json`):

| benchmark | mean c | std c | fraction c > 0 | p(c = 0) | p(c = −0.25) | within-state std of first-order term | of curvature term | ratio |
|---|---|---|---|---|---|---|---|---|
| CIFAR-10 | **+0.1399** | 0.017 | 1.00 | 0 | 0 | 0.127 | 0.056 | 0.43 |
| CIFAR-100 | **+0.0848** | 0.134 | 0.72 | 0 | 0 | 0.133 | 0.187 | 1.46 |
| SVHN | −0.0458 | 0.091 | 0.32 | 6e-70 | 0 | 0.204 | 0.205 | 1.02 |
| EuroSAT | **+0.1271** | 0.038 | 1.00 | 0 | 0 | 0.129 | 0.074 | 0.56 |
| DTD | **+0.1126** | 0.117 | 0.85 | 0 | 0 | 0.113 | 0.138 | 1.19 |

Two facts settle the diagnosis:

1. **The mechanism prediction is confirmed.** The learned `c` is positive on 4 of 5 benchmarks
   (+0.085 … +0.140, positive for 72–100 % of states) and essentially zero on SVHN (−0.046), and it
   is significantly different from both 0 and −0.25 everywhere (p < 1e-69). The data genuinely
   reject the fixed negative sign — exactly as the R123 `corr(score, κ)` analysis predicted.
2. **Releasing the constraint did not transfer to selection.** The signed head ranks the marginal
   *worse* than the constrained head on 3 of 5 benchmarks (mean Spearman +0.293 vs +0.375) and
   selects worse on the gate (2/5 vs 3/5 passing). The term-scale columns show why the sign is a
   weak lever: the curvature term's within-state spread is comparable to the first-order term's
   (ratio 0.43–1.46), but it is fitted by the *same* ranking objective on the *same* data, so the
   unconstrained head spends its extra freedom re-fitting the ordering instead of improving it —
   top-1 agreement drops from 0.67 to 0.42 on CIFAR-100 and mean regret rises from 0.021 to 0.023,
   while on EuroSAT the Spearman falls from +0.196 to −0.037 and the regret rises from 0.051 to
   0.150.

So the R123 diagnosis was correct about the *coefficient*, and wrong about it being the binding
constraint on performance: the fixed-sign penalty is a modelling restriction the data violate, but
neither the constrained nor the unconstrained curvature term changes the selection materially,
because the learned first-order coefficient `g_theta` already absorbs whatever ordering signal the
curvature carries.

<!-- LIMITATIONS AND ARTIFACTS APPENDED BELOW -->

## 4. Negative results and limitations

1. **Both pre-registered predictions fail** — (i) 2/5 benchmarks with Spearman at least as high as
   the fixed-sign head (needed 4/5), (ii) 2/5 benchmarks passing the paired gate against Static
   Utility (needed 4/5) with the pooled contrast including zero (+0.0528 [−0.0081,+0.1236]).
   The signed head is the *weakest* of the three heads on the gate criterion (2/5 vs 3/5 for both
   the fixed-sign bilinear and the frozen scalar head).
2. **The mechanism prediction does hold, and that is the informative part.** `c` moves to +0.085 …
   +0.140 (72–100 % of states positive, p < 1e-69 against both 0 and −0.25) on CIFAR-10, CIFAR-100,
   EuroSAT and DTD, and to −0.046 on SVHN. The fixed negative sign is therefore a genuine
   misspecification — but releasing it costs ranking quality instead of gaining it.
3. **Extra freedom without extra data overfits the ranking objective.** The signed head has one more
   free output unit (state-dependent) fitted by the same Huber + gap-weighted pairwise loss on the
   same pairs; the measured ranking statistics degrade (mean Spearman +0.293 vs +0.375; top-1
   agreement 0.42 vs 0.67 on CIFAR-100; regret 0.150 vs 0.051 on EuroSAT). This is a capacity
   effect, not evidence that the sign constraint is useful in itself.
4. **The curvature term is not the lever.** Its within-state spread is 0.43–1.46× the first-order
   term's, so it is not negligible as a *value*; but both heads reach nearly the same selected sets
   (CIFAR-10 identical to 1e-4), which says the ordering signal the curvature carries is already
   absorbed by the learned first-order coefficient `g_theta`.
5. **Scope note.** For squared loss the exact identity fixes `c = −1`, so this variant is only
   meaningful for the cross-entropy family; the traffic experiments keep the exact form. The R124
   result should not be read as evidence about the traffic head.
6. **Inherited ceiling.** As in R115/R123, the family's pooled state-conditioning headroom is
   +0.0126 and only CIFAR-10/EuroSAT exceed 1 %, so all three heads are competing inside a narrow
   band: on CIFAR-10 the total available gain above the static oracle is 0.049 nats, and the
   differences between heads are 0.004–0.015 nats.
7. **Reporting caveat.** The per-benchmark `c` statistics come from 2–4 sampled test states per cell
   (all (episode, candidate) rows pooled); the p-values are therefore anti-conservative under
   within-episode correlation, and only the sign and rough magnitude should be used. The term-scale
   ratios share the same caveat.
8. **`head_comparison.json` R² columns remain uninformative** (affine calibration fitted on
   train-split states and applied to test-split states with a different marginal scale), exactly as
   noted in R123; the within-state ranking statistics are the ones to read.

## 5. Reproduction

```bash
cd /root/icl_ess_threshold/PAMI_MUR
python -m pytest tests/test_signed_curvature.py -q                     # 5 tests

python experiments/demo_selection/run_r124_signed.py \
    --benchmarks cifar10,cifar100,svhn,eurosat,dtd --seeds 101,202,303 \
    --out-root results/raw/R124_signed_curvature_<stamp>

python experiments/demo_selection/diagnose_r124_heads.py \
    --run-root results/raw/R124_signed_curvature_<stamp> \
    --out results/derived/R124_signed_curvature_summary/head_comparison.json
python experiments/demo_selection/analyze_r124.py \
    --run-root results/raw/R124_signed_curvature_<stamp> \
    --out-root results/derived/R124_signed_curvature_summary
```

## 6. Artifacts

| path | content |
|---|---|
| `experiments/demo_selection/signed_curvature.py` | signed head (`<g, d_bar> + c_theta(state)·kappa`), trainer, coefficient statistics |
| `experiments/demo_selection/run_r124_signed.py` | R124 driver: three heads per cell, signed primary, bilinear + frozen reference policies |
| `experiments/demo_selection/diagnose_r124_heads.py` | three-way head comparison + coefficient and term-scale diagnostics |
| `experiments/demo_selection/analyze_r124.py` | gates, pooled contrasts, three-way head tables (reuses R112/R123 aggregation) |
| `tests/test_signed_curvature.py` | 5 tests: scoring identity, fixed-sign initialisation, unconstrained coefficient, packed marginals, training |
| `results/raw/R124_signed_curvature_20260923_0537/` | 15 cells, `run.log`, manifest, zero failures |
| `results/raw/R124_smoke/` | tiny CIFAR-10 smoke cell |
| `results/derived/R124_signed_curvature_summary/` | `summary.json`, `gate_table.csv`, `report_tables.md`, `head_comparison.json`, `provenance.json` |
