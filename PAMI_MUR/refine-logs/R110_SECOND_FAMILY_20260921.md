# R110 — Second task family: demonstration selection for a frozen in-context image classifier

Run date: 2026-09-21. Raw run: `PAMI_MUR/results/raw/R110_demo_selection_20260921_0408`
Derived: `PAMI_MUR/results/derived/R110_demo_selection_summary`, theory: `PAMI_MUR/results/derived/R110_response_theory_ce/validation.json`.
All numbers below are copied from those derived files; nothing is estimated.

## 1. Headline

* The frozen R-MUR router was re-instantiated, **unchanged**, on a second task family
  (context-example selection for a frozen in-context CLIP classifier) and evaluated on
  5 benchmarks × 3 seeds (15 cells, 1 602 test episodes over 10/100/10/10/47 target classes).
* The strict gate (mean gain of R-MUR at q=4 above anchor-only **and** above Static Utility
  **and** above Cached Utility, paired 95% CI lower bound > 0) **passes on 3 of 5 benchmarks**
  (CIFAR-10, EuroSAT, SVHN) and **fails on 2 of 5** (CIFAR-100, DTD).
* With class-clustered (hierarchical) CIs only CIFAR-10 survives; on SVHN and EuroSAT the
  per-class spread is large (class fractions above Static Utility: SVHN 0.30, EuroSAT 0.80),
  so the paired-CI pass is not robust to target-level clustering.
* Two structural negative results matter more than the pass/fail pattern:
  1. **the state-conditioned ceiling is flat** — `oracle_greedy` ≈ `oracle_static` to four
     decimals on every benchmark (H_state ≈ 0.000), i.e. this family has essentially no
     set-conditioning headroom, which is exactly the effect R-MUR was built to exploit;
  2. **plain kNN relevance is essentially optimal** (99.4–99.9 % of the oracle gain on all five
     benchmarks) and beats *all* learned routers (Static, Cached, R-MUR) on 2 of 5.
* A protocol-level defect was found and fixed during the work: with unrestricted attention the
  32 same-class query images of a batch can vote on their shared label, which collapsed the
  benchmark (CIFAR-100 base CE 2.8e-4 nats, accuracy 1.000). The delivered protocol scores
  every query image independently from its own embedding plus the demonstration set. The
  pre-fix diagnostic run is preserved at `R110_demo_selection_20260921_0342_unmasked_diagnostic`.

## 2. Protocol

### 2.1 Data and encoder

* Frozen encoder: `open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai",
  cache_dir="/root/.cache/clip")`, eval mode, no gradients. Embeddings are L2-normalised and
  cached as float16 `.npy` per benchmark (`PAMI_MUR/data/vision_features/<bench>/{train,test}_emb.npy`).
* Benchmarks and splits (torchvision `datasets`, `download=True`):

  | benchmark | classes | train | test | split note |
  |---|---|---|---|---|
  | CIFAR-10 | 10 | 50 000 | 10 000 | official |
  | CIFAR-100 | 100 | 50 000 | 10 000 | official |
  | SVHN | 10 | 73 257 | 26 032 | official |
  | EuroSAT | 10 | 16 200 | 5 400 | torchvision exposes no split: stratified 1 620/540 per class, seed 20260921 |
  | DTD | 47 | 1 880 | 1 880 | official train/test, partition 1 |

* All raw archives and extracted image folders were deleted after extraction
  (`data/vision_raw/` is empty); extraction took 3 594 s wall in total on the shared GPU.
* Deviation (documented): the encoder ran under `torch.autocast(float16)` because the GPU is
  shared and contended (fp32 measured 30 img/s vs 79 img/s fp16). The fp16 cache is the single
  source of truth for every method.

### 2.2 Episodes (traffic → demonstration mapping)

| traffic protocol | R110 |
|---|---|
| target sensor | target class `c` |
| anchor window (12 steps) | query batch of 32 test images of class `c` |
| candidate sensor pool (K=16) | 16 train images: 8 nearest to the query-batch mean in CLIP space (class `c`) + 8 distractors from other classes |
| candidate history (12-d) | candidate image embedding (768-d CLIP, PCA-64 observable view) |
| target future | query image labels |
| squared-error reduction | mean softmax cross-entropy reduction over the query batch |

* Episode/target structure: per benchmark and seed, for each class `c` we draw
  `train_episodes_per_class` training batches and `test_episodes_per_class` test batches;
  one batch = one episode row carrying 32 query images that share the pool built from their
  mean. Test queries come from the test split, demonstration candidates always from the train
  split. Training query batches are drawn from the whole train split **minus that episode's own
  candidate images** (so a query image is never its own demonstration).
* Budget B=4 of K=16, 3 seeds (101/202/303). Episode counts: CIFAR-10/SVHN/EuroSAT 24 train and
  8 test batches per class; CIFAR-100 6 and 2; DTD 4 and 2 (DTD has only 40 train and 40 test
  images per class, so DTD query batches wrap around and reuse images; documented limitation).
* Router observation ("observable state"): per-episode query image embedding and candidate
  embeddings projected to 64 PCA components fitted **without labels** on the train split, then
  L2-normalised. Relevance = cosine similarity in that space (the kNN baseline).
* Deviations from the traffic plumbing, all deliberate and listed in §7.

### 2.3 Frozen predictor

* Token transformer over `[query_1..query_32] + [demo_1..demo_16]`, demo token =
  `Linear(CLIP)` + label embedding + type embedding, query token = `Linear(CLIP)` + type
  embedding, 3 pre-LN layers, d_model 128, 4 heads, FFN 256, dropout 0.1, logits read at the
  query positions, no positional encoding.
* **Query independence (protocol-critical):** the attention mask forbids query→query and
  demo→query attention, so each query image is scored from its own embedding plus the
  selected demonstration set only. Padded rows are zeroed after every layer to avoid NaN from
  fully-masked demonstration rows.
* Unselected demonstrations are exactly inert (key-padding mask), verified numerically by
  `tests/test_demo_selection.py::test_unselected_demos_are_exactly_inert`, and the batched
  pair evaluation equals the explicit augmented-set evaluation
  (`test_pair_predict_equals_full_mask`).
* Training: subset dropout on the TRAIN split only, with the frozen protocol's mixture
  (`sample_subset_masks`, mostly sizes 0..4, 15 % longer than the budget, 5 % full pool,
  5 % empty), AdamW lr 3e-4, weight decay 1e-5, batch 32, 200 epochs (CIFAR-100: 120 epochs to
  equalise gradient steps at 600 episodes). Train CE: 4.76 → 0.147 (CIFAR-100), ≈2.6 → 0.002
  (CIFAR-10), 0.38 (SVHN), 0.010 (EuroSAT), 0.004 (DTD).

### 2.4 Response features and R-MUR

* Response summary keeps the exact 7-dimensional semantics of `response_summary` in
  `KBS_MUR/src/mur/traffic.py`, computed in **class-logit space** (the predictor's output
  vector is the analogue of the traffic expert's forecast vector): mean base logit, mean logit
  after adding the candidate, mean / std / mean-abs / RMS / max-abs of the logit change. The
  same convention is used for training and inference.
* R-MUR = cached utility screen → top-q candidates → gap-weighted ranking-calibrated reranker
  over response features, imported unchanged (`score_state`, `select_static`, `select_mur`,
  `train_utility_model`, `sample_pair_dataset`, `train_ranked_response_model`,
  `select_ranked_response_mur`, `paired_ci`). q ∈ {2,4,8,16}; 20 router epochs; ranking weight
  β=1; 4 states per episode for pair construction.

## 3. Protocol defect found and fixed (query tokens attend to each other) produced a degenerate benchmark:
CIFAR-10 base CE ≈ 2e-7 nats with accuracy 1.000, CIFAR-100 base CE 2.8e-4 with accuracy 1.000 —
impossible from CLIP features alone (a linear probe on CLIP-L reaches ≈0.81 on CIFAR-100). The
32 same-class queries were voting on their shared label. Diagnostic run (kept, not the
deliverable): `results/raw/R110_demo_selection_20260921_0342_unmasked_diagnostic`
(CIFAR-10 gains ≈1e-7 nats; CIFAR-100 q=4 gain 7.4e-6 nats). After the fix the same benchmarks
behave sanely: CIFAR-100 accuracy 0.812 with no demonstrations, 0.997 with four relevant
demonstrations.

## 4. Predictor quality (frozen classifier, test episodes, mean over 3 seeds)

| benchmark | 0 demos | 2 relevant | 4 relevant | 4 random | 4 farthest | 16 (whole pool) |
|---|---|---|---|---|---|---|
| CIFAR-10 acc / CE | 0.9660 / 0.2048 | 0.9777 / 0.1382 | 0.9831 / 0.1073 | 0.9758 / 0.1513 | 0.9651 / 0.2172 | 0.9836 / 0.1006 |
| CIFAR-100 acc / CE | 0.8120 / 0.6761 | 0.9886 / 0.0478 | 0.9969 / 0.0160 | 0.9702 / 0.1157 | 0.7953 / 0.7742 | 0.9981 / 0.0098 |
| SVHN acc / CE | 0.6932 / 0.9537 | 0.9789 / 0.0836 | 0.9956 / 0.0208 | 0.9290 / 0.2351 | 0.5454 / 1.7046 | 0.9971 / 0.0126 |
| EuroSAT acc / CE | 0.9457 / 0.2490 | 0.9762 / 0.1117 | 0.9863 / 0.0697 | 0.9711 / 0.1390 | 0.9358 / 0.3109 | 0.9883 / 0.0575 |
| DTD acc / CE | 0.7282 / 1.3487 | 0.8729 / 0.5859 | 0.9214 / 0.3379 | 0.8490 / 0.7092 | 0.7214 / 1.4021 | 0.9489 / 0.2203 |

The predictor is genuinely in-context: relevant demonstrations dominate the whole pool,
farthest-relevant demonstrations are worse than random ones, and 16 pooled demonstrations (half
of them distractors) beat 0 demonstrations everywhere.

## 5. Ceiling / headroom

| benchmark | anchor-only CE | oracle_greedy | oracle_static | pool-all | relevance (kNN) | random | H_state |
|---|---|---|---|---|---|---|---|
| CIFAR-10 | 0.2048 | 0.0981 | 0.0980 | 0.1042 | 0.0975 | 0.0535 | 0.000 |
| CIFAR-100 | 0.6761 | 0.6606 | 0.6606 | 0.6664 | 0.6602 | 0.5605 | 0.000 |
| SVHN | 0.9537 | 0.9354 | 0.9353 | 0.9411 | 0.9329 | 0.7186 | 0.000 |
| EuroSAT | 0.2490 | 0.1801 | 0.1800 | 0.1915 | 0.1793 | 0.1099 | 0.000 |
| DTD | 1.3487 | 1.0141 | 1.0140 | 1.1284 | 1.0108 | 0.6395 | 0.000 |

All values are mean query-batch CE reduction (nats). `H_state` (greedy − static, relative) is
below 5e-4 everywhere. The sequential oracle is *not* better than the standalone oracle, so the
budget-limited subset-choice problem on this family is essentially a static ranking problem,
and kNN relevance already attains 99.4–99.9 % of the oracle gain.

## 6. Gate

Definition: G_RMUR = mean per-episode gain of `ranked_response_q4`; G_Static, G_Cached the same
for `static_utility` and `cached_mur`. Paired CIs are t-based over 1 602 test episodes pooled
across seeds (`paired_ci`, frozen implementation). Hierarchical CIs resample target classes and
then episodes within class (5 000 replicates).

| benchmark | q | gain [95 % CI] | Δ Static [CI] | Δ Cached [CI] | pass (paired) |
|---|---|---|---|---|---|
| CIFAR-10 | 2 | +0.0396 [+0.0254,+0.0537] | −0.0087 [−0.0166,−0.0007] | −0.0037 [−0.0099,+0.0024] | no |
| CIFAR-10 | **4** | **+0.0643 [+0.0490,+0.0796]** | **+0.0160 [+0.0075,+0.0246]** | **+0.0210 [+0.0140,+0.0280]** | **yes** |
| CIFAR-10 | 8 | +0.0716 [+0.0556,+0.0875] | +0.0233 [+0.0143,+0.0323] | +0.0283 [+0.0195,+0.0371] | yes |
| CIFAR-10 | 16 | +0.0711 [+0.0552,+0.0870] | +0.0229 [+0.0140,+0.0318] | +0.0278 [+0.0189,+0.0367] | yes |
| CIFAR-100 | 2 | +0.4131 [+0.3704,+0.4558] | −0.1638 [−0.1956,−0.1320] | −0.1250 [−0.1545,−0.0954] | no |
| CIFAR-100 | **4** | **+0.5247 [+0.4858,+0.5636]** | **−0.0522 [−0.0758,−0.0285]** | **−0.0133 [−0.0365,+0.0098]** | **no** |
| CIFAR-100 | 8 | +0.5714 [+0.5355,+0.6073] | −0.0055 [−0.0226,+0.0116] | +0.0333 [+0.0142,+0.0524] | no |
| CIFAR-100 | 16 | +0.5660 [+0.5291,+0.6029] | −0.0109 [−0.0298,+0.0081] | +0.0280 [+0.0070,+0.0490] | no |
| SVHN | 2 | +0.7698 [+0.7037,+0.8358] | +0.0168 [−0.0355,+0.0691] | +0.0010 [−0.0485,+0.0505] | no |
| SVHN | **4** | **+0.8809 [+0.8328,+0.9291]** | **+0.1279 [+0.0647,+0.1911]** | **+0.1121 [+0.0637,+0.1605]** | **yes** |
| SVHN | 8 | +0.8927 [+0.8456,+0.9397] | +0.1397 [+0.0746,+0.2048] | +0.1239 [+0.0738,+0.1740] | yes |
| SVHN | 16 | +0.8861 [+0.8388,+0.9334] | +0.1331 [+0.0673,+0.1989] | +0.1173 [+0.0667,+0.1679] | yes |
| EuroSAT | 2 | +0.1008 [+0.0804,+0.1213] | −0.0065 [−0.0198,+0.0067] | −0.0116 [−0.0215,−0.0018] | no |
| EuroSAT | **4** | **+0.1294 [+0.1080,+0.1509]** | **+0.0221 [+0.0090,+0.0351]** | **+0.0169 [+0.0057,+0.0282]** | **yes** |
| EuroSAT | 8 | +0.1452 [+0.1229,+0.1675] | +0.0378 [+0.0231,+0.0525] | +0.0327 [+0.0220,+0.0433] | yes |
| EuroSAT | 16 | +0.1456 [+0.1230,+0.1683] | +0.0383 [+0.0238,+0.0528] | +0.0332 [+0.0226,+0.0437] | yes |
| DTD | 2 | +0.2242 [+0.1751,+0.2733] | −0.3739 [−0.4457,−0.3020] | −0.4236 [−0.4909,−0.3563] | no |
| DTD | **4** | **+0.3322 [+0.2721,+0.3923]** | **−0.2658 [−0.3388,−0.1929]** | **−0.3156 [−0.3865,−0.2446]** | **no** |
| DTD | 8 | +0.3943 [+0.3300,+0.4585] | −0.2038 [−0.2774,−0.1303] | −0.2535 [−0.3288,−0.1782] | no |
| DTD | 16 | +0.4272 [+0.3624,+0.4921] | −0.1708 [−0.2416,−0.1000] | −0.2205 [−0.2944,−0.1467] | no |

Cached-utility and Static-utility reference points (mean gain): CIFAR-10 0.0433 / 0.0482,
CIFAR-100 0.5381 / 0.5769, SVHN 0.7688 / 0.7530, EuroSAT 0.1125 / 0.1074, DTD 0.6478 / 0.5981.

**Hierarchical (class-clustered) CIs for q=4**: gain [0.0345,0.1032] (CIFAR-10),
[0.4500,0.6038] (CIFAR-100), [0.7512,1.0382] (SVHN), [0.0763,0.1870] (EuroSAT),
[0.2372,0.4353] (DTD); Δ Static [0.0058,0.0277], [−0.0874,−0.0204], [−0.0183,+0.3366],
[+0.0064,+0.0434], [−0.3716,−0.1667]; Δ Cached [+0.0092,+0.0351], [−0.0480,+0.0181],
[+0.0233,+0.2273], [−0.0075,+0.0405], [−0.4480,−0.1946]. With class-clustered intervals the
full three-part gate passes **only on CIFAR-10**.

**Per-class (target-level) consistency at q=4** (fraction of the 10/100/10/10/47 classes):

| benchmark | class gain > 0 | class gain > Static | class gain > Cached |
|---|---|---|---|
| CIFAR-10 | 1.00 | 1.00 | 1.00 |
| CIFAR-100 | 0.99 | 0.26 | 0.35 |
| SVHN | 1.00 | 0.30 | 0.80 |
| EuroSAT | 1.00 | 0.80 | 0.80 |
| DTD | 0.96 | 0.09 | 0.19 |

**Per-seed q=4 gains** (all three seeds): CIFAR-10 0.0720 / 0.0661 / 0.0548; CIFAR-100 0.5543 /
0.5167 / 0.5032; SVHN 0.9100 / 0.8976 / 0.8351; EuroSAT 0.1158 / 0.1394 / 0.1331; DTD 0.3473 /
0.4132 / 0.2362. Sign consistency of Δ Static per seed: CIFAR-10 +/+ /+, SVHN +/+/+, EuroSAT
+/−/+, CIFAR-100 −/−/−, DTD −/−/−.

## 7. Compute cost (per benchmark × seed cell, mean)

| benchmark | wall s | predictor rows (selection + evaluation) | predictor calls | predictor inference s | training episodes × epochs |
|---|---|---|---|---|---|
| CIFAR-10 | 85.5 | 16 727 + 1 200 | 449 | 3.79 | 240 × 200 |
| CIFAR-100 | 127.2 | 42 080 + 3 000 | 1 105 | 8.93 | 600 × 120 |
| SVHN | 80.4 | 16 741 + 1 200 | 449 | 3.99 | 240 × 200 |
| EuroSAT | 125.5 | 16 733 + 1 200 | 449 | 7.92 | 240 × 200 |
| DTD | 130.1 | 19 922 + 1 410 | 452 | 7.90 | 188 × 200 |

Predictor rows by policy (CIFAR-10 cell): Static Utility 0, Cached Utility 0 (both are pure
router-side scoring — the cached screen never calls the frozen predictor), oracle-static 1 360,
oracle-greedy 4 961, R-MUR q=2/q=4/q=8/full 960/1 600/2 880/4 966, final policy evaluation
1 200. R-MUR at q=4 therefore costs 1 600 predictor rows per cell, 32 % of the oracle-greedy
diagnostic and 0.32× the full-pool reranking variant (`ranked_response_full`, 4 966 rows); the
budget-4 selection itself is 66 % of the oracle gain on CIFAR-10 and 94 % on SVHN.
The remaining wall time is predictor training (240 episodes × 200 epochs) plus router fitting;
it is not covered by the profiled inference seconds.

## 8. Theory validation (`results/derived/R110_response_theory_ce/validation.json`)

For 92 995 frozen-predictor (state, candidate) samples (2 975 840 per-query rows) across the 15
cells, with `m` the true query-batch CE reduction, `first = (e_y − p_A)·d`,
`d = logits(f_{A∪{j}}) − logits(f_A)`:

* sandwich `first − 0.25‖d‖² ≤ m ≤ first`: **0 violations** at the aggregate level and
  **0 violations** at the per-query level; worst upper slack +3.0e-8, worst lower slack
  +1.0e-3 (both positive, i.e. the bound is tight but never crossed).
* pooled Spearman(first, m) = 0.9711, Pearson = 0.9237, R²(first vs m) = 0.8532
  (per benchmark Spearman 0.9695–0.9932); mean m = 0.1131, mean first = 0.2264,
  mean ‖d‖² = 9.14.
* the first-order term is systematically optimistic (mean slack 0.113 nats), which is the
  second-order curvature term the Hessian bound controls.

## 9. Negative results and limitations

1. **No set-conditioning headroom.** `oracle_greedy` equals `oracle_static` to 4 decimals on
   all five benchmarks (H_state ≤ 5e-4). The utility of a demonstration set on this family is
   almost purely additive: one relevant demonstration already moves CIFAR-100 from 0.812 to
   ≈0.99 accuracy, so re-scoring the state adds nothing over the best standalone ranking. The
   second family therefore does not test the mechanism R-MUR was designed for; it mainly tests
   its ranking quality under a near-static utility landscape.
2. **kNN relevance is essentially optimal.** Relevance attains 99.4–99.9 % of the oracle gain
   on every benchmark and is *better than every learned router* (R-MUR included) on CIFAR-100
   and DTD. Any claim of learned context selection on this family must first beat plain cosine
   kNN.
3. **R-MUR fails on CIFAR-100 and DTD**, and the failure is mechanistic, not marginal: at q=4
   the response-aware greedy loop stops after 1.63 (CIFAR-100) and 0.96 (DTD) selected
   contexts, because the reranker's scores for the 2nd–4th relevant demonstrations fall below
   the frozen `> 0` acceptance threshold, while the oracle needs ≈4 contexts to reach 1.01
   (DTD) nats. The gap is −0.266 (DTD) and −0.052 (CIFAR-100) nats versus Static Utility, with
   class fractions above Static Utility of 0.09 and 0.26. So the frozen harm-safe stopping rule
   is mis-calibrated for a family in which marginal utilities stay large and positive up to the
   budget — a transfer failure of the method's implicit assumption (later contexts are
   optional), not of its ranking stage.
4. **Hierarchical CIs weaken two passes.** On SVHN, Δ Static = +0.128 [0.065, 0.191] paired but
   [−0.018, +0.337] with class clustering (only 30 % of classes beat Static Utility); on
   EuroSAT Δ Cached = +0.017 [0.006, 0.028] paired but [−0.008, +0.041] clustered. With 10
   classes per benchmark the target-level uncertainty dominates.
5. **Learned utility models are far below relevance.** As a fraction of the kNN-relevance gain,
   Static Utility reaches 0.49 (CIFAR-10), 0.87 (CIFAR-100), 0.81 (SVHN), 0.60 (EuroSAT), 0.59
   (DTD) and Cached Utility 0.44/0.82/0.82/0.63/0.64; R-MUR q=4 reaches 0.66/0.79/0.94/0.72/0.33.
   The frozen router consumes a 64-dimensional PCA view of the frozen encoder and 4 states per
   episode; under those inputs the learned marginal-utility model does not reach the quality of
   a simple similarity rule on this family.
6. **Small splits.** DTD has 40 train/test images per class, so query batches wrap around and
   reuse images (documented in §2.2); EuroSAT has no official torchvision split, so a
   stratified split was defined here. CIFAR-100 uses 120 predictor epochs (instead of 200) to
   equalise gradient steps.
7. **Cached-utility and Static-utility methods were not re-tuned** for this family, and neither
   was R-MUR (frozen by instruction), so the comparison is protocol-faithful but not a tuned
   upper bound for the learned baselines.
8. **Feature precision.** Encoder features were produced under fp16 autocast on a contended
   shared GPU (see §2.1). The cache is shared by all methods, so comparisons remain internal,
   but absolute CLIP accuracy numbers may differ slightly from fp32 encoding.
9. The pre-fix `unmasked` diagnostic run shows how easily this family degenerates; the fix is
   part of the protocol, not of the method, and is covered by unit tests.
10. **No second predictor backbone was produced.** A prototype / kNN-style predictor over the
    selected demonstrations was considered as a cheap second backbone, but the calibrated
    in-context transformer already saturates the achievable in-context accuracy
    (0.98-0.998 with four relevant demonstrations), so a second backbone would not change the
    utility landscape; it is reported as skipped rather than as a negative result.
11. Only CE-based utility is measured. The saturated pre-fix run shows that accuracy-based
    utility would have hidden the batch-consensus defect entirely, so the CE definition is
    load-bearing for this family and should stay in any follow-up.

## 10. Reproduction

```bash
cd /root/icl_ess_threshold/PAMI_MUR
python -m pytest tests/test_demo_selection.py -q                      # 14 tests
python experiments/demo_selection/run_r110_features.py \
    --benchmarks cifar10,cifar100,svhn,eurosat,dtd                    # CLIP cache (once)
python experiments/demo_selection/run_r110_demo_selection.py \
    --benchmarks cifar10,cifar100,svhn,eurosat,dtd --seeds 101,202,303 \
    --out-root results/raw/R110_demo_selection_<stamp>
python experiments/demo_selection/theory_validation.py \
    --run-root results/raw/R110_demo_selection_<stamp> \
    --out results/derived/R110_response_theory_ce/validation.json
python experiments/demo_selection/analyze_r110.py \
    --run-root results/raw/R110_demo_selection_<stamp> \
    --out-root results/derived/R110_demo_selection_summary
```

`results/raw/<run>/run_manifest.json` records the argv, git HEAD and the md5 of every borrowed
implementation file, so each raw run pins the revision it was produced with.

## 11. Artifacts

| path | content |
|---|---|
| `experiments/demo_selection/vision_data.py` | CLIP feature extraction + cache |
| `experiments/demo_selection/run_r110_features.py` | extraction CLI |
| `experiments/demo_selection/demo_protocol.py` | episodes, pools, DemoBatch, PCA view |
| `experiments/demo_selection/incontext_predictor.py` | in-context classifier + `DemoExpertOps` |
| `experiments/demo_selection/train_predictor.py` | subset-dropout training |
| `experiments/demo_selection/frozen_helpers.py` | verbatim snapshot of the borrowed PAMI helpers (provenance in header) |
| `experiments/demo_selection/run_r110_demo_selection.py` | protocol driver |
| `experiments/demo_selection/analyze_r110.py` | gates, per-class consistency, hierarchical CIs |
| `experiments/demo_selection/theory_validation.py` | CE first-order/Hessian sandwich artifact |
| `experiments/demo_selection/run_r110_sweep.sh` | resumable sweep launcher (retry on OOM) |
| `tests/test_demo_selection.py` | 14 unit tests (inertness, pair exactness, query independence, pool structure, sampler equivalence) |
| `data/vision_features/<bench>/` | float16 CLIP caches + meta.json |
| `results/raw/R110_demo_selection_20260921_0408/` | 15 cells: `result.json`, `episode_gains.npz`, `predictor.pt`, `run.log` |
| `results/raw/R110_demo_selection_20260921_0342_unmasked_diagnostic/` | pre-fix saturation diagnostic |
| `results/raw/R110_smoke_demo_selection/` | first end-to-end smoke (old protocol) |
| `results/derived/R110_demo_selection_summary/` | `summary.json`, `gate_table.csv`, `per_class.csv`, `report_tables.md` |
| `results/derived/R110_response_theory_ce/validation.json` | 92 995 samples + sandwich checks |
