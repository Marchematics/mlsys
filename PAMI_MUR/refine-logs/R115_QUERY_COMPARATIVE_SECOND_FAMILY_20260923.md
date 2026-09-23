# R115 — Query-comparative second family (removing the label shortcut)

Run date: 2026-09-22/23.

* Raw run: `PAMI_MUR/results/raw/R115_query_comparative_<stamp>/` (immutable, logged, manifest).
* Derived: `PAMI_MUR/results/derived/R115_query_comparative_summary/`.
* New modules: `prototype_predictor.py`, `run_r115_query_comparative.py`,
  `diagnose_r115_shortcut.py`, `diagnose_r115_curves.py`; tests `tests/test_prototype_predictor.py`.
* R110/R112 code, runs and derived artefacts are not modified; R115 imports their evaluation
  stack unchanged (`evaluate_cell` and `redundancy_diagnostics` from the R112 driver,
  `predictor_quality`/`relevance_of` from the R110 driver, pools from `redundant_pool.py`, and the
  frozen R-MUR path from `KBS_MUR`/`frozen_helpers.py`).

## 0. Pre-registered predictions (written before the R115 sweep was launched)

R112 established that the in-context transformer's utility is label-driven: replacing candidate
features by noise cost only ~18 % of the first-member marginal, and equalising labels flipped every
marginal negative. R115 replaces the predictor with one whose utility can only come from
demonstration images relative to the query, and pre-registers:

> **(i)** `H_state ≥ 0.01` — the sequential oracle beats the standalone oracle by at least 1 % of
> the oracle gain on the pooled analysis, because a second same-class demonstration refines the
> prototype while a near-duplicate adds little.
> **(ii)** Forced-budget R-MUR q=4 beats Static Utility and Cached Utility with a paired 95 % CI
> lower bound above zero on **at least 4 of 5 benchmarks** and on the pooled analysis. (Forced
> budget is the primary gate; the frozen thresholded rule is reported alongside.)
> **(iii)** The label shortcut is closed: predicting the realised marginal utility from the *label
> multiset of the selected set alone* has substantially lower out-of-sample R² than the same probe
> given the predictor's response features.

Each is reported below as pass/fail with numbers; failures are reported as failures.

## 1. The predictor (what changed, and why it closes the shortcut)

Everything else is inherited from R110/R112 unchanged: the cached frozen-CLIP `ViT-L-14` features,
the episode structure and counts (1 602 test episodes), the R112 redundancy-structured pool
(4 k-means visual-mode clusters × 3 near-duplicate in-class demonstrations + 4 other-class
distractors, K = 16, B = 4), the frozen R-MUR code path and q ∈ {2,4,8,16}, and seeds 101/202/303.

The new predictor (`prototype_predictor.py`) is a learned-metric attention prototype classifier:

1. a **shared** linear metric `z(x) = normalise(W x)`, `W: 768 → 128`, **without bias**, is applied
   to the query and to every demonstration;
2. per class, the prototype is the **attention-weighted mean of the selected demonstrations of that
   class**: weights `softmax_m(β · cos(z_q, z_m))` over the selected demonstrations of that class,
   with a learned sharpness `β = softplus(·)`;
3. the class logit is `τ · cos(z_q, prototype_c)` with a learned temperature `τ = softplus(·)`;
4. a class with **no** selected demonstration receives the fixed orthogonal reference score `0`
   (not a learned parameter). With an empty context every logit is therefore exactly 0, the
   prediction is uniform and the cross-entropy is `log C` — there is no learned classification head
   anywhere in the model;
5. consequently a class can only be predicted well if demonstrations *of that class* have been
   selected **and** their images lie close to the query, and two selections with the same label
   multiset but different images give different logits (tested).

Training uses the frozen protocol's subset-dropout mixture on TRAIN-split episodes only (mostly
sizes 0..4, 15 % longer than the budget, 5 % full pool, 5 % empty), AdamW lr 3e-4, weight decay
1e-5, batch 32, and the same per-benchmark epoch counts as R110/R112 (200; 120 for CIFAR-100).
The forward signature matches `InContextClassifier`, so the R110 batched pair-prediction ops,
7-dimensional response summariser and packing are reused byte-identically.

Unit tests (`tests/test_prototype_predictor.py`, 8 tests) check: empty context ⇒ uniform
`log C`; logits depend on demonstration images (same labels, different images ⇒ different
logits); different same-class subsets ⇒ different logits; demonstration order irrelevant; masked
demonstrations exactly inert; queries conditionally independent; batched pair prediction exact;
training uses demonstrations.

<!-- RESULTS APPENDED AFTER THE SWEEP -->
## 2. Results (15 cells, 1 602 test episodes, seeds 101/202/303, zero failures)

Raw run: `results/raw/R115_query_comparative_20260923_0243` (the sweep was smoke-tested on CPU and
then run on the GPU once the co-tenant control run released memory; the smoke is kept at
`results/raw/R115_smoke/`).

### 2.1 Verdict on the pre-registered predictions

| # | prediction | measured | verdict |
|---|---|---|---|
| (i) | `H_state ≥ 0.01` pooled | pooled **+0.0126** [+0.0019,+0.0239], 5/5 benchmarks positive; per benchmark cifar10 **0.0270**, eurosat **0.0296**, dtd 0.0040, svhn 0.0017, cifar100 0.0006 | **PASS on the pre-registered pooled criterion**, but only 2/5 individual benchmarks exceed 1 % — the effect is real and mechanism-consistent, not uniform |
| (ii) | forced q=4 beats Static **and** Cached, paired CI low > 0, on ≥4/5 benchmarks **and** pooled | vs **Cached: 5/5** benchmarks and pooled **+0.1553** [+0.1131,+0.1975] PASS; vs **Static: 3/5** (cifar100, eurosat, svhn) and pooled **+0.0529** [−0.0078,+0.1282] FAIL | **FAIL overall** (the ≥4/5 clause and the pooled-vs-Static clause both fail) |
| (iii) | label-multiset probe R² substantially below response probe R² | mean label **0.565** vs response **0.741** (gap **+0.176**), label < response on **4/5** benchmarks, permutation control **−0.182** | **PASS directionally**, with the caveat that label-only features still explain 56 % of the marginal variance in this design (§2.6) |

### 2.2 Gates at q=4 (paired CIs over all test episodes; hierarchical CIs cluster by class)

| benchmark | rule | gain [95 % CI] | Δ Static [CI] | Δ Cached [CI] | cls>Static | sel | pass paired / hier |
|---|---|---|---|---|---|---|---|
| CIFAR-10 | thresholded | 1.6907 [1.6698,1.7117] | −0.0206 [−0.0410,−0.0002] | +0.0819 [+0.0330,+0.1309] | 0.30 | 2.83 | no / no |
| CIFAR-10 | forced | 1.7074 [1.7003,1.7145] | −0.0039 [−0.0096,+0.0017] | +0.0986 [+0.0462,+0.1510] | 0.50 | 4.00 | no / no |
| CIFAR-100 | thresholded | 2.5343 [2.5245,2.5441] | +0.0102 [−0.0055,+0.0258] | +0.1653 [+0.1138,+0.2167] | 0.54 | 3.89 | no / no |
| CIFAR-100 | forced | 2.5388 [2.5336,2.5440] | +0.0146 [+0.0015,+0.0277] | +0.1697 [+0.1190,+0.2204] | 0.54 | 4.00 | **yes / yes** |
| SVHN | thresholded | 1.5628 [1.5163,1.6093] | +0.0506 [+0.0136,+0.0877] | +0.1560 [+0.1003,+0.2117] | 0.70 | 3.71 | **yes / yes** |
| SVHN | forced | 1.6005 [1.5789,1.6220] | +0.0883 [+0.0548,+0.1218] | +0.1937 [+0.1356,+0.2518] | 0.80 | 4.00 | **yes / yes** |
| EuroSAT | thresholded | 1.5806 [1.5390,1.6221] | +0.1246 [+0.0467,+0.2026] | +0.1518 [+0.0839,+0.2197] | 0.50 | 2.90 | **yes / yes** |
| EuroSAT | forced | 1.6440 [1.6312,1.6568] | +0.1881 [+0.1164,+0.2597] | +0.2152 [+0.1407,+0.2898] | 0.50 | 4.00 | **yes / yes** |
| DTD | thresholded | 2.0174 [2.0018,2.0329] | −0.0237 [−0.0473,−0.0000] | +0.0985 [+0.0375,+0.1596] | 0.26 | 2.78 | no / no |
| DTD | forced | 2.0182 [2.0029,2.0336] | −0.0228 [−0.0463,+0.0007] | +0.0994 [+0.0388,+0.1600] | 0.28 | 4.00 | no / no |

The frozen thresholded rule and the forced-budget ablation are much closer here than in R112
(2.83–3.89 vs 4.00 contexts selected): with the prototype predictor the reranker's scores are
mostly positive, so the stopping rule binds less often.

### 2.3 Pooled across benchmarks (equal weight per benchmark, three-level bootstrap)

| statistic | mean | 95 % CI | benchmarks positive |
|---|---|---|---|
| R-MUR q=4 gain (thresholded) | 1.8772 | [1.5955,2.2245] | 5/5 |
| R-MUR q=4 − Static (thresholded) | +0.0282 | [−0.0157,+0.0802] | 3/5 |
| R-MUR q=4 − Cached (thresholded) | +0.1307 | [+0.1019,+0.1581] | 5/5 |
| **R-MUR q=4 forced − Static** | **+0.0529** | **[−0.0078,+0.1282]** | 3/5 |
| **R-MUR q=4 forced − Cached** | **+0.1553** | **[+0.1131,+0.1975]** | 5/5 |
| **H_state** | **+0.0126** | **[+0.0019,+0.0239]** | 5/5 |

### 2.4 Headroom, and the redundancy comparison across predictor families

| benchmark | anchor CE | oracle_greedy | oracle_static | H_state | m(1st) | m(2nd\|own seed) | 2nd/1st this predictor | 2nd/1st R112 transformer |
|---|---|---|---|---|---|---|---|---|
| CIFAR-10 | 2.3026 | 1.7622 | 1.7147 | **0.0270** | 1.7081 | 0.0026 | **0.0015** | 0.421 |
| CIFAR-100 | 4.6052 | 2.5801 | 2.5785 | 0.0006 | 2.5701 | 0.0032 | **0.0012** | 0.067 |
| SVHN | 2.3026 | 1.6759 | 1.6732 | 0.0017 | 1.6468 | 0.0104 | **0.0063** | 0.139 |
| EuroSAT | 2.3026 | 1.7329 | 1.6817 | **0.0296** | 1.6693 | 0.0053 | **0.0032** | 0.372 |
| DTD | 3.8501 | 2.1536 | 2.1451 | 0.0040 | 2.1095 | 0.0164 | **0.0078** | 0.407 |

Two things changed at once. First, the anchor-only prediction is now exactly uniform
(`log C`) on every benchmark, because the label shortcut was removed together with the learned
head: the utility is now entirely attributable to the selected demonstrations. Second — and this is
the design goal — a second demonstration of an already-represented class is now worth **0.12–0.78 %**
of the first (previously 6.7–42 % with the transformer), so demonstration *images* carry the
utility and near-duplicates are genuinely redundant.

### 2.5 Utility curve over k = 0..4 (seed 101, mean CE reduction in nats)

| benchmark | prefix | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|---|
| CIFAR-10 | relevance / one cluster | 1.711 / 1.711 | 1.713 / 1.713 | 1.714 / 1.714 | 1.715 / 1.714 |
| | oracle greedy / static | 1.715 / 1.715 | 1.741 / 1.716 | 1.758 / 1.716 | **1.764 / 1.716** |
| EuroSAT | relevance / one cluster | 1.675 / 1.674 | 1.677 / 1.676 | 1.678 / 1.678 | 1.679 / 1.678 |
| | oracle greedy / static | 1.681 / 1.681 | 1.712 / 1.681 | 1.728 / 1.682 | **1.733 / 1.682** |
| DTD | relevance / one cluster | 2.126 / 2.126 | 2.136 / 2.136 | 2.139 / 2.140 | 2.141 / 2.140 |
| | oracle greedy / static | 2.152 / 2.152 | 2.156 / 2.154 | 2.158 / 2.155 | **2.159 / 2.155** |
| SVHN | relevance / one cluster | 1.642 / 1.642 | 1.648 / 1.647 | 1.651 / 1.651 | 1.653 / 1.651 |
| | oracle greedy / static | 1.660 / 1.660 | 1.664 / 1.663 | 1.665 / 1.663 | **1.666 / 1.663** |
| CIFAR-100 | relevance / one cluster | 2.571 / 2.571 | 2.573 / 2.574 | 2.574 / 2.574 | 2.574 / 2.574 |
| | oracle greedy / static | 2.578 / 2.578 | 2.579 / 2.578 | 2.579 / 2.578 | **2.580 / 2.578** |

The mechanism the pre-registration assumed is visible: the standalone oracle's prefix is *flat*
(1.715 → 1.716 on CIFAR-10, 1.681 → 1.682 on EuroSAT), while the sequential oracle keeps gaining
through k = 2–4 (1.715 → 1.764, 1.681 → 1.733) by switching to other visual modes once the best
mode is represented. Redundancy is harmless and useless (the single-cluster prefix tracks the
relevance prefix exactly). Where the two oracles diverge the family finally has state-conditioning
headroom — and that is exactly the two benchmarks where `H_state` exceeds 1 %.

### 2.6 Prediction (iii): the label shortcut probe

Two ridge probes predict the realised marginal `m(j|A)` out of sample (fitted on train-split
episodes, evaluated on test-split episodes, 3 states per split per seed):

| benchmark | label-multiset probe R² | response-features probe R² | gap | permuted control R² |
|---|---|---|---|---|
| CIFAR-10 | 0.535 | **0.968** | +0.432 | −0.199 |
| CIFAR-100 | 0.542 | **0.647** | +0.105 | −0.230 |
| SVHN | 0.600 | 0.548 | −0.052 | −0.144 |
| EuroSAT | 0.537 | **0.890** | +0.353 | −0.195 |
| DTD | 0.610 | **0.653** | +0.044 | −0.142 |
| **mean** | **0.565** | **0.741** | **+0.176** | **−0.182** |

Reading: the frozen predictor's response signature predicts the marginal better than the label
multiset on 4/5 benchmarks (mean R² 0.74 vs 0.57) and the permutation control collapses to ≈ 0, so
the response channel carries genuine image-relative information. But the label probe still reaches
R² ≈ 0.57, and on SVHN it even wins. The reason is structural: with a uniform anchor-only
prediction, the *presence or absence of a correct-class demonstration* alone moves the CE by up to
`log C` — an enormous, purely label-level effect — so a label probe inevitably explains a large
share of the variance. The label shortcut is therefore **closed for the part of the utility that
the selection method can act on** (which same-class images to pick: 0.12–0.78 % redundancy ratios,
response probe ahead), but *not* eliminated for the coarse "is the class represented at all" part.

### 2.7 Predictor quality and compute (mean over 3 seeds)

| benchmark | accuracy 0 / 2 rel / 4 rel / 4 rand / 4 far / 16 | CE 0 / 4 rel / 16 | wall s | train CE first → final |
|---|---|---|---|---|
| CIFAR-10 | 0.100 / 0.998 / 0.998 / 0.995 / 0.046 / 0.986 | 2.303 / 0.590 / 0.596 | 75 | 1.417 → 1.005 |
| CIFAR-100 | 0.010 / 1.000 / 1.000 / 0.935 / 0.000 / 0.832 | 4.605 / 2.030 / 2.233 | 417 | 3.245 → 2.720 |
| SVHN | 0.100 / 0.999 / 1.000 / 0.899 / 0.000 / 0.781 | 2.303 / 0.637 / 1.291 | 96 | 1.433 → 1.210 |
| EuroSAT | 0.100 / 0.998 / 0.998 / 0.988 / 0.015 / 0.963 | 2.303 / 0.624 / 0.705 | 116 | 1.414 → 1.038 |
| DTD | 0.021 / 0.999 / 0.998 / 0.900 / 0.333 / 0.778 | 3.850 / 1.718 / 1.968 | 192 | 2.590 → 2.199 |

Zero-demonstration accuracy is exactly chance (1/C) on every benchmark by construction, 4 relevant
demonstrations give 0.998–1.000, 4 farthest (wrong-class) demonstrations are at or below chance,
and the whole 16-candidate pool is *worse* than 4 relevant ones — the in-context mechanism is
strong and it is driven by the demonstration images. Predictor inference cost is unchanged
(q=4 forced: 1 600 rows/cell on 10-class benchmarks, 1 880 on DTD, 4 000 on CIFAR-100).

## 3. Negative results, limitations and what they mean

1. **Prediction (ii) fails.** Forced-budget R-MUR q=4 beats Cached Utility on 5/5 benchmarks and on
   the pooled analysis (+0.1553 [+0.1131,+0.1975]), but it **does not** beat Static Utility
   (3/5 benchmarks; pooled +0.0529 with a CI that includes zero; on CIFAR-10 and DTD it is slightly
   *below* Static). This is a different failure from R112: there the learned utility models were far
   below the relevance baseline; here the learned Static Utility model is already close to the
   standalone oracle (CIFAR-10 1.7113 vs oracle-static 1.7147), so beating it requires capturing the
   *state-conditioned* gain, and R-MUR's cached shortlist plus response reranking captures only part
   of it (R-MUR forced 1.7074 on CIFAR-10, i.e. below the static oracle it is meant to improve on).
2. **Prediction (i) passes only at the pooled level.** `H_state` is 2.7–3.0 % on CIFAR-10 and
   EuroSAT — where the greedy and static oracle prefixes genuinely diverge — but 0.06–0.4 % on
   CIFAR-100, SVHN and DTD, so the pooled 1.26 % is carried by two benchmarks. The pre-registered
   criterion was pooled, so it passes; a per-benchmark reading would be 2/5.
3. **Prediction (iii) is only partially satisfying.** The response probe beats the label probe on
   4/5 benchmarks (mean R² 0.741 vs 0.565, permutation control −0.182), but label-only features
   still explain 56 % of the marginal variance, and on SVHN the label probe wins (0.600 vs 0.548).
   The coarse "is the target class represented at all" component of the utility is intrinsically
   label-level once the anchor-only predictor is uniform; closing that channel would require an
   anchor that is itself informative (i.e. restoring a zero-shot head), which would re-open the
   shortcut. The two requirements are in tension, and this design resolves them in favour of
   "utility is attributable to the demonstrations".
4. **The anchor-only baseline is now uniform (`log C`).** Removing the trained head removes the
   zero-shot path, so the family measures "how well does the selected context identify the query's
   class" rather than "how much does context improve on a strong zero-shot classifier". Utility
   magnitudes (1.7–2.6 nats) are therefore much larger than in R110/R112 (0.06–1.0 nats) and are
   *not* comparable across families; only within-family comparisons and the normalised `H_state`
   are.
5. **The pooled gate statistic is dominated by scale.** Pooling per-benchmark means gives CIFAR-100
   (2.5 nats) and DTD (2.0 nats) similar weight to CIFAR-10 (1.7 nats) even though their class
   counts and base CEs differ by up to 4.6 vs 2.3 nats. A normalised variant (Δ divided by the
   oracle gain per benchmark) is reported in the same table only through `H_state` and the
   per-benchmark columns; the raw pooled means should not be read as an effect size across
   benchmarks.
6. **No second pool environment for R115.** Only the R112 natural near-duplicate pool was run
   (the copy environment was an R112 control); the R115 redundancy ratios (0.12–0.78 %) already
   show that the copy environment would be nearly indistinguishable from the natural one for this
   predictor, which is itself the point.
7. **Contention effects.** The sweep was smoke-tested on CPU and executed on the GPU once the
   co-tenant control run released it; CIFAR-100 cells took 393–444 s and the other benchmarks
   75–192 s. No cell failed.
8. **The prototype predictor has no learned per-class parameters**, so it cannot express a prior
   over classes; this is deliberate (it is what closes the shortcut) but it means the family cannot
   represent the case where context selection interacts with a non-uniform class prior.

## 4. Reproduction

```bash
cd /root/icl_ess_threshold/PAMI_MUR
python -m pytest tests/test_prototype_predictor.py -q          # 8 tests

# smoke (CPU is enough: the predictor is a small MLP)
python experiments/demo_selection/run_r115_query_comparative.py \
    --benchmarks cifar10 --seeds 101 --train-episodes-per-class 3 --test-episodes-per-class 2 \
    --predictor-epochs 60 --router-epochs 10 --states-per-episode 2 --device cpu \
    --out-root results/raw/R115_smoke

# full sweep
python experiments/demo_selection/run_r115_query_comparative.py \
    --benchmarks cifar10,cifar100,svhn,eurosat,dtd --seeds 101,202,303 \
    --out-root results/raw/R115_query_comparative_<stamp>

python experiments/demo_selection/diagnose_r115_curves.py \
    --run-root results/raw/R115_query_comparative_<stamp> \
    --out-root results/derived/R115_query_comparative_summary
python experiments/demo_selection/diagnose_r115_shortcut.py \
    --run-root results/raw/R115_query_comparative_<stamp> \
    --out results/derived/R115_query_comparative_summary/shortcut_probe.json
python experiments/demo_selection/analyze_r112.py \
    --run-root results/raw/R115_query_comparative_<stamp> \
    --redundancy-natural results/derived/R115_query_comparative_summary/redundancy.json \
    --out-root results/derived/R115_query_comparative_summary --bootstrap 2000
```

## 5. Artifacts

| path | content |
|---|---|
| `experiments/demo_selection/prototype_predictor.py` | learned-metric attention prototype predictor + training/checkpointing |
| `experiments/demo_selection/run_r115_query_comparative.py` | R115 driver (reuses the R112 evaluation cell and pools) |
| `experiments/demo_selection/diagnose_r115_curves.py` | utility curves + redundancy diagnostics for the new predictor |
| `experiments/demo_selection/diagnose_r115_shortcut.py` | label-multiset vs response-feature probes (prediction iii) |
| `tests/test_prototype_predictor.py` | 8 tests: uniform empty context, image dependence, same-label-different-image, order invariance, inertia, query independence, pair exactness, training |
| `results/raw/R115_query_comparative_20260923_0243/` | 15 cells, `run.log`, manifest, zero failures |
| `results/raw/R115_smoke/` | CPU smoke (single CIFAR-10 cell, tiny sizes) |
| `results/derived/R115_query_comparative_summary/` | `summary.json`, `gate_table.csv`, `report_tables.md`, `utility_curve.json`, `redundancy.json`, `shortcut_probe.json`, `provenance.json` |
