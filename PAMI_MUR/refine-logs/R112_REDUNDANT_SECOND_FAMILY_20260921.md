# R112 — Redundancy-structured second family (pre-registered variant of R110)

Run date: 2026-09-21.

* Raw runs: `PAMI_MUR/results/raw/R112_redundant_demo_<stamp>/` (natural near-duplicate pools) and
  `PAMI_MUR/results/raw/R112_redundant_copies_<stamp>/` (controlled near-copy pools).
* Derived: `PAMI_MUR/results/derived/R112_redundant_demo_summary/`.
* Smoke: `results/raw/R112_smoke/`, `results/raw/R112_smoke_copy/`.
* R110 code, runs and derived artefacts are untouched; R112 adds new modules
  (`redundant_pool.py`, `forced_budget.py`, `run_r112_redundant_demo.py`, `analyze_r112.py`) and new
  tests (`tests/test_redundant_pool.py`).

## 0. Pre-registered prediction (written before the R112 sweep was launched)

R110 measured `H_state = (oracle_greedy − oracle_static) / oracle_greedy ≤ 5e-4` on all five
benchmarks, i.e. the second family had no state-conditioning headroom. The R112 hypothesis,
registered before any R112 result existed:

> If candidate pools contain redundant near-duplicate demonstrations, then
> **(i)** the sequential oracle must beat the standalone oracle (`H_state > 0`), and
> **(ii)** R-MUR q=4 must beat Static Utility and Cached Utility with a paired-CI lower bound
> above zero, because redundancy is exactly what makes a candidate's value depend on what is
> already selected.

Operationalisation: `H_state > 0` means `oracle_greedy > oracle_static` in mean query-batch CE
reduction on the pooled test episodes; the gate is the R110 gate (paired 95 % CI lower bound > 0
for q=4 gain over anchor-only, over Static Utility and over Cached Utility). Both are reported per
benchmark and pooled; the forced-budget ablation separates ranking from stopping.

Failure of either clause is reported as a failure with the measured numbers.

## 1. What R112 changes relative to R110

Everything except the candidate pool is inherited unchanged: same frozen CLIP `ViT-L-14` feature
caches, same episode counts (10/100/10/10/47 classes; 24/6/24/24/4 train and 8/2/8/8/2 test
batches per class; 1 602 test episodes), same in-context predictor family and training recipe
(including the query-independence attention mask), same frozen R-MUR code path (`score_state`,
`train_utility_model`, `train_ranked_response_model`, `select_ranked_response_mur`), same q values
{2,4,8,16}, same seeds 101/202/303, same budget B=4 and pool size K=16.

New: (a) the redundancy-structured pool rule, (b) a controlled near-copy environment, (c) the
forced-budget R-MUR ablation, (d) redundancy diagnostics.

### 1.1 Pool rule (natural near-duplicate environment)

For every episode (target class `c`, query batch of 32 test images):

1. rank the class-`c` pool-half train images by cosine similarity to the query-batch summary in
   the R110 compact (PCA-64) view; take the `probe_size = 60` most relevant as the working set;
2. split the working set into `n_clusters = 4` visual modes with deterministic k-means
   (farthest-point initialisation, 10 Lloyd iterations, cosine objective);
3. each cluster keeps its `cluster_size = 3` most query-relevant members; the most relevant member
   is the cluster seed;
4. candidate order: cluster 0 (seed first, then members by decreasing relevance), cluster 1,
   cluster 2, cluster 3, then `16 − 4×3 = 4` distractors sampled uniformly from the other classes;
5. K = 16, B = 4, so the sequential oracle can take exactly one member per cluster, while a
   standalone-utility ranking can spend several slots inside one mode.

An earlier R112 variant used relevance-ordered "seed + its 3 nearest pool images" clusters
(3 × 4 + 4); it was replaced because the candidate clusters lost relevance monotonically, so the
low-relevance clusters contributed nothing and `H_state` stayed at ~0 (smoke diagnostics kept at
`results/raw/R112_smoke*`). The balanced k-means rule keeps every cluster inside the relevant head
of the class.

### 1.2 Controlled near-copy environment (symmetric redundancy)

Real near-duplicates in these benchmarks are only ≈0.80 cosine-similar in the compact view and
have *asymmetric* standalone utility: a seed's nearest neighbour is markedly less relevant than
the seed itself, so a standalone ranking never picks it and no trap exists. To test the mechanism
under redundancy whose standalone utilities are equal by construction, the second environment
replaces every non-seed cluster member by a perturbed copy of its seed
(`feature + 0.002 · N(0, I)` in the frozen encoder space, cosine ≈ 0.999). Both environments are
reported side by side; the natural one is the protocol-faithful variant, the copy one is the
controlled mechanism test.

### 1.3 Forced-budget ablation

`select_ranked_response_mur_forced` (new module `forced_budget.py`) is the frozen two-stage greedy
loop with the `> 0` acceptance test removed, so exactly B = 4 candidates are added. The scoring
path is unchanged; this isolates whether a benchmark's shortfall comes from ranking or from the
frozen stopping rule.

### 1.4 Redundancy diagnostics (measured, not assumed)

Per cell we record: mean pairwise cosine inside clusters, across clusters and for the R110-style
top-8 nearest reference pool; the true marginal utility of a cluster's first member at the empty
state, of a second member given its own cluster's seed, and of distractors at the empty state; and
the mean number of distinct clusters covered by each policy's selection.

<!-- RESULTS APPENDED BELOW AFTER THE SWEEP -->
## 2. Results (15 cells per environment, 1 602 test episodes each, seeds 101/202/303, no failures)

Raw runs: `results/raw/R112_redundant_demo_20260921_0526` (natural) and
`results/raw/R112_redundant_copies_20260921_0526` (controlled near-copies), each with
`run.log` and a manifest pinning git HEAD and the md5 of every imported module.

### 2.1 Verdict on the pre-registered prediction

| clause | prediction | natural near-duplicates | controlled near-copies | verdict |
|---|---|---|---|---|
| (i) | `H_state > 0` | +0.00095 / +0.00014 / +0.00012 / +0.00042 / +0.00007 (cifar10/100/dtd/eurosat/svhn), pooled +0.0003 [+0.0001,+0.0006] | +0.00032 / +0.00013 / +0.00008 / +0.00012 / +0.00004, pooled +0.0001 [+0.0001,+0.0002] | **FAILS in substance**: positive on 5/5 benchmarks but ≤ 0.1 % of the oracle gain; the sequential oracle does not meaningfully beat the standalone oracle |
| (ii) | R-MUR q=4 > Static and > Cached, paired CI > 0 | frozen thresholded rule: 1/5 benchmarks pass (SVHN only) | frozen thresholded rule: 2/5 (CIFAR-10, SVHN) | **FAILS for the frozen rule** |
| (ii-a) | same, forced-budget ablation | 5/5 pass, pooled ΔStatic +0.0451 [+0.0182,+0.0796], ΔCached +0.0703 [+0.0320,+0.1087] | 5/5 pass, pooled ΔStatic +0.0548 [+0.0202,+0.1013], ΔCached +0.1076 [+0.0509,+0.1606] | **PASSES on 5/5 in both environments, paired and class-clustered** |

The redundancy manipulation itself worked, and its *utility* signature was measured: a second
member of a cluster given its own cluster's seed retains 6.7–42 % of the first member's marginal
(mean over 3 seeds, `redundancy_{natural,copies}.json`), and the copy environment reaches
within-cluster similarity 0.998–0.999 versus 0.575–0.838 between clusters. Redundancy was
therefore present and strong; it simply did not create headroom for state conditioning, for the
mechanistic reason isolated in §2.7.

### 2.2 Gates at q=4 (paired CIs; hierarchical CIs are class-clustered)

Natural near-duplicate pools:

| benchmark | rule | gain [95 % CI] | Δ Static [CI] | Δ Cached [CI] | cls>Static | sel | pass paired / hier |
|---|---|---|---|---|---|---|---|
| CIFAR-10 | thresholded | 0.0741 [0.0604,0.0878] | −0.0043 [−0.0164,0.0078] | −0.0056 [−0.0156,0.0045] | 0.60 | 3.00 | no / no |
| CIFAR-10 | forced | 0.0973 [0.0814,0.1131] | +0.0188 [+0.0088,+0.0289] | +0.0176 [+0.0107,+0.0245] | 0.90 | 4.00 | **yes / yes** |
| CIFAR-100 | thresholded | 0.6383 [0.5963,0.6804] | −0.0379 [−0.0567,−0.0192] | +0.0782 [+0.0482,+0.1083] | 0.20 | 2.38 | no / no |
| CIFAR-100 | forced | 0.6955 [0.6548,0.7362] | +0.0192 [+0.0088,+0.0297] | +0.1354 [+0.1088,+0.1620] | 0.78 | 4.00 | **yes / yes** |
| DTD | thresholded | 0.2020 [0.1430,0.2609] | −0.8440 [−0.9337,−0.7542] | −0.8475 [−0.9429,−0.7521] | 0.02 | 0.63 | no / no |
| DTD | forced | 1.1121 [1.0194,1.2047] | +0.0661 [+0.0295,+0.1027] | +0.0626 [+0.0357,+0.0895] | 0.66 | 4.00 | **yes / yes** |
| EuroSAT | thresholded | 0.1456 [0.1234,0.1678] | −0.0057 [−0.0175,0.0061] | +0.0082 [−0.0066,0.0230] | 0.40 | 3.04 | no / no |
| EuroSAT | forced | 0.1685 [0.1447,0.1923] | +0.0172 [+0.0092,+0.0251] | +0.0310 [+0.0183,+0.0437] | 0.90 | 4.00 | **yes / yes** |
| SVHN | thresholded | 0.8582 [0.8045,0.9120] | +0.0816 [+0.0250,+0.1381] | +0.0823 [+0.0223,+0.1423] | 0.50 | 3.42 | yes / no |
| SVHN | forced | 0.8810 [0.8280,0.9339] | +0.1043 [+0.0434,+0.1652] | +0.1050 [+0.0420,+0.1681] | 0.70 | 4.00 | **yes / yes** |

Controlled near-copy pools:

| benchmark | rule | gain [95 % CI] | Δ Static [CI] | Δ Cached [CI] | cls>Static | sel | pass paired / hier |
|---|---|---|---|---|---|---|---|
| CIFAR-10 | thresholded | 0.0880 [0.0732,0.1028] | +0.0086 [+0.0008,+0.0164] | +0.0213 [+0.0110,+0.0316] | 0.80 | 3.34 | yes / no |
| CIFAR-10 | forced | 0.0943 [0.0794,0.1093] | +0.0149 [+0.0078,+0.0220] | +0.0276 [+0.0174,+0.0378] | 1.00 | 4.00 | **yes / yes** |
| CIFAR-100 | thresholded | 0.6308 [0.5896,0.6719] | −0.0423 [−0.0570,−0.0276] | +0.1193 [+0.0887,+0.1499] | 0.17 | 2.28 | no / no |
| CIFAR-100 | forced | 0.6953 [0.6547,0.7360] | +0.0222 [+0.0107,+0.0338] | +0.1839 [+0.1537,+0.2141] | 0.67 | 4.00 | **yes / yes** |
| DTD | thresholded | 0.1851 [0.1239,0.2463] | −0.8331 [−0.9255,−0.7406] | −0.7791 [−0.8803,−0.6779] | 0.00 | 0.53 | no / no |
| DTD | forced | 1.0881 [0.9950,1.1812] | +0.0700 [+0.0212,+0.1187] | +0.1239 [+0.0772,+0.1707] | 0.70 | 4.00 | **yes / yes** |
| EuroSAT | thresholded | 0.1365 [0.1146,0.1585] | +0.0083 [−0.0038,+0.0204] | +0.0195 [+0.0071,+0.0320] | 0.60 | 2.98 | no / no |
| EuroSAT | forced | 0.1548 [0.1324,0.1771] | +0.0266 [+0.0147,+0.0384] | +0.0378 [+0.0254,+0.0501] | 0.90 | 4.00 | **yes / yes** |
| SVHN | thresholded | 0.8277 [0.7668,0.8886] | +0.0847 [+0.0196,+0.1498] | +0.1090 [+0.0452,+0.1728] | 0.40 | 3.27 | yes / no |
| SVHN | forced | 0.8836 [0.8310,0.9361] | +0.1406 [+0.0683,+0.2128] | +0.1648 [+0.0918,+0.2379] | 0.80 | 4.00 | **yes / yes** |

### 2.3 Pooled across benchmarks (equal weight per benchmark, three-level bootstrap)

| statistic | natural | controlled near-copies |
|---|---|---|
| R-MUR q=4 gain (frozen thresholded) | +0.3836 [0.1283,0.6717], positive 5/5 | +0.3736 [0.1268,0.6501], positive 5/5 |
| R-MUR q=4 − Static (thresholded) | −0.1621 [−0.5148,+0.0405], above Static 1/5 | −0.1548 [−0.5066,+0.0441], above Static 3/5 |
| R-MUR q=4 − Cached (thresholded) | −0.1369 [−0.4939,+0.0650], above Cached 3/5 | −0.1020 [−0.4418,+0.0956], above Cached 4/5 |
| R-MUR q=4 **forced** − Static | **+0.0451 [+0.0182,+0.0796]**, above Static 5/5 | **+0.0548 [+0.0202,+0.1013]**, above Static 5/5 |
| R-MUR q=4 **forced** − Cached | **+0.0703 [+0.0320,+0.1087]**, above Cached 5/5 | **+0.1076 [+0.0509,+0.1606]**, above Cached 5/5 |
| `H_state` | +0.0003 [+0.0001,+0.0006], positive 5/5 | +0.0001 [+0.0001,+0.0002], positive 5/5 |

### 2.4 Headroom and redundancy diagnostics (means over 3 seeds)

| benchmark | anchor CE | oracle_greedy | oracle_static | H_state | within sim | between sim | R110 ref sim | m(1st) | m(2nd\|own seed) | 2nd/1st |
|---|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 (natural) | 0.1553 | 0.1128 | 0.1127 | +0.00095 | 0.806 | 0.767 | 0.795 | 0.0595 | 0.0251 | 0.421 |
| CIFAR-100 (natural) | 0.7075 | 0.7041 | 0.7040 | +0.00014 | 0.781 | 0.726 | 0.777 | 0.6471 | 0.0436 | 0.067 |
| DTD (natural) | 1.3606 | 1.2112 | 1.2111 | +0.00012 | 0.476 | 0.350 | 0.569 | 0.6990 | 0.2844 | 0.407 |
| EuroSAT (natural) | 0.2015 | 0.1778 | 0.1778 | +0.00042 | 0.830 | 0.793 | 0.823 | 0.1092 | 0.0407 | 0.372 |
| SVHN (natural) | 0.9097 | 0.9014 | 0.9013 | +0.00007 | 0.728 | 0.649 | 0.706 | 0.7607 | 0.1058 | 0.139 |
| CIFAR-10 (copies) | 0.1539 | 0.1084 | 0.1083 | +0.00032 | 0.999 | 0.815 | 0.795 | 0.0554 | 0.0259 | 0.468 |
| CIFAR-100 (copies) | 0.7074 | 0.7040 | 0.7039 | +0.00013 | 0.999 | 0.790 | 0.777 | 0.6468 | 0.0437 | 0.068 |
| DTD (copies) | 1.3604 | 1.2130 | 1.2129 | +0.00008 | 0.999 | 0.575 | 0.569 | 0.7013 | 0.2852 | 0.407 |
| EuroSAT (copies) | 0.1878 | 0.1653 | 0.1653 | +0.00012 | 0.998 | 0.838 | 0.823 | 0.0989 | 0.0353 | 0.357 |
| SVHN (copies) | 0.9095 | 0.9014 | 0.9014 | +0.00004 | 0.998 | 0.724 | 0.706 | 0.7598 | 0.1070 | 0.141 |

`m(1st)` is the true marginal of a cluster seed at the empty state; `m(2nd|own seed)` is the mean
marginal of that cluster's other members when *only that cluster's seed* is selected. These
numbers were re-measured with the corrected one-state-per-cluster definition by
`recompute_r112_redundancy.py` (the first driver version measured the member marginal with the
whole budget already spent); files `redundancy_natural.json` / `redundancy_copies.json`.
The mean distractor marginal (state = empty) is −0.017 (CIFAR-10), −0.156 (CIFAR-100), −0.452
(SVHN), −0.034 (EuroSAT), −0.047 (DTD) in the natural environment: wrong-class demonstrations are
actively harmful, which is why every method prefers the in-class part of the pool.

Distinct-cluster coverage (natural / copies): oracle_static 2.83/2.00 (CIFAR-10), 2.68/2.01
(CIFAR-100), 2.73/2.00 (DTD), 2.69/2.02 (EuroSAT), 2.60/2.00 (SVHN) out of 4 clusters; oracle_greedy
2.82/1.99, 2.62/1.96, 2.73/2.02, 2.70/2.00, 2.62/2.05. The copies environment does trap the
standalone oracle into fewer clusters (2.0 instead of 2.6–2.8), but the gain is unchanged because a
member of an already-used cluster is nearly as valuable as a fresh cluster's seed (see 2.5).

### 2.5 Why the prediction failed: the utility curve saturates

`results/derived/R112_redundant_demo_summary/utility_curve_{natural,copies}.json` (seed 101,
mean query-batch CE reduction in nats for k = 0..4 demonstrations):

| benchmark | selection prefix | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|---|
| CIFAR-10 (natural) | relevance | 0.0697 | 0.0989 | 0.1206 | 0.1356 |
| | **one cluster only** | 0.0697 | 0.0990 | 0.1206 | **0.1206** |
| | oracle greedy | 0.0710 | 0.1005 | 0.1218 | 0.1362 |
| | oracle static | 0.0710 | 0.1002 | 0.1218 | 0.1364 |
| CIFAR-100 (natural) | relevance / one cluster | 0.6702 | 0.7216 | 0.7328 | 0.7356 / **0.7327** |
| | oracle greedy / static | 0.6726 | 0.7225 | 0.7330 | 0.7357 / 0.7356 |
| SVHN (natural) | relevance / one cluster | 0.7696 | 0.8798 | 0.9089 | 0.9196 / **0.9089** |
| | oracle greedy / static | 0.7872 | 0.8870 | 0.9117 | 0.9209 / 0.9208 |
| EuroSAT (natural) | relevance / one cluster | 0.1189 | 0.1612 | 0.1778 | 0.1848 / **0.1778** |
| | oracle greedy / static | 0.1202 | 0.1624 | 0.1784 | 0.1851 / 0.1851 |
| DTD (natural) | relevance / one cluster | 0.6839 | 0.9783 | 1.1330 | 1.2220 / **1.1332** |
| | oracle greedy / static | 0.6904 | 0.9844 | 1.1374 | 1.2249 / 1.2247 |
| CIFAR-10 (copies) | relevance / one cluster | 0.0631 | 0.0880 | 0.1034 | 0.1148 / **0.1034** |
| DTD (copies) | relevance / one cluster | 0.6888 | 0.9843 | 1.1386 | 1.2268 / **1.1386** |

Two facts explain the null result quantitatively:

1. **Saturation.** The first demonstration already captures 51 % (CIFAR-10), 91 % (CIFAR-100),
   84 % (SVHN), 64 % (EuroSAT), 56 % (DTD) of the 4-demonstration gain; the second captures
   73–98 %. The marginal value of the third and fourth demonstration is 0.002–0.09 nats.
2. **Diversity is worth little.** Spending the whole budget inside a single redundant cluster
   reaches within 0.4–11 % of the diverse 4-demonstration gain (CIFAR-100 0.4 %, SVHN 1.2 %,
   EuroSAT 3.8 %, DTD 7.3 %, CIFAR-10 11 %). A standalone ranking therefore has almost nothing to
   lose, and the sequential oracle has almost nothing to gain — hence `H_state ≈ 1e-4`.

The prescriptive reading: redundancy creates state-conditioning headroom only if the marginal
value of *additional distinct* context is still large when the budget is exhausted, and only if
demonstration *content* (not just its label) carries the utility. In this family neither holds:
the predictor saturates after two relevant demonstrations, and §2.7 shows that the remaining
utility is a label-vote effect that near-duplicates reproduce exactly. No pool construction can
restore the headroom; it requires a different predictor (one that must use the demonstration
images rather than their labels) or a task whose query labels are not members of the candidate
label set.

### 2.7 Mechanism: demonstration utility in this family is label-driven, not information-driven

Ablation on CIFAR-10 (seed 101 predictor of the copy environment, 80 test episodes, same pools;
`redundancy_diagnostics` recomputed inline):

| variant | within-cluster sim | m(1st) | m(2nd \| own seed) | m(distractor) |
|---|---|---|---|---|
| natural near-duplicates | 0.800 | +0.0812 | +0.0312 | −0.0235 |
| ε-copies of the seed (0.999 sim) | 0.999 | +0.0812 | +0.0313 | −0.0235 |
| **candidate features replaced by noise, labels kept** | 0.999 | +0.0666 | +0.0338 | −0.0194 |
| **all candidate labels set equal** | — | −0.0159 | −0.0209 | −0.0174 |

Reading:

* Replacing real near-duplicates by photographic copies changes the marginals by < 0.5 %
  (0.0312 vs 0.0313). A second demonstration that carries the *same label* remains worth about
  40 % of the first even when it is an exact copy: the predictor multiplies evidence for a class
  rather than combining independent information.
* Destroying the demonstration *features* (pure noise, labels intact) changes the first-member
  marginal by only −18 % (0.0812 → 0.0666) and leaves the second-member and distractor marginals
  essentially unchanged; destroying the *labels* (all equal, wrong class) turns every marginal
  negative. The utility is therefore dominated by the class label a demonstration votes for, not
  by how similar its image is to the query.
* Consequence for context selection: m(j | A) is almost a function of the label multiset of A, so
  swapping one same-class near-duplicate for another barely moves the utility. That is precisely
  why redundancy cannot create `H_state` here, and why kNN relevance, the standalone oracle and
  the sequential oracle all land on the same value.

### 2.6 Compute (per benchmark × seed cell, mean)

| benchmark | wall s (natural / copies) | predictor rows: q=4 / q=4 forced / oracle-greedy | training episodes × epochs |
|---|---|---|---|
| CIFAR-10 | 164 / 155 | 1 600 / 1 600 / 4 960 | 240 × 200 |
| CIFAR-100 | 456 / 441 | 4 000 / 4 000 / 12 400 | 600 × 120 |
| SVHN | 157 / 112 | 1 600 / 1 600 / 4 960 | 240 × 200 |
| EuroSAT | 126 / 147 | 1 600 / 1 600 / 4 960 | 240 × 200 |
| DTD | 142 / 130 | 1 880 / 1 880 / 5 828 | 188 × 200 |

The forced-budget variant costs exactly the same number of predictor rows as the frozen rule
(both evaluate the same top-q shortlist at every step); it only stops later. The redundancy
diagnostic adds 4 marginal passes per cell (~1 280–3 200 rows).

## 3. Negative results and limitations

1. **The pre-registered prediction (i) fails.** Redundancy was injected and measured (within-cluster
   similarity 0.999 in the copy environment; second-member marginal 6.7–42 % of the first), yet
   `H_state` is +0.00004…+0.00095, i.e. ≤ 0.1 % of the oracle gain on every benchmark in both
   environments. The sequential oracle and the standalone oracle converge.
2. **The frozen thresholded R-MUR gate fails on both environments** (pooled ΔStatic −0.162 natural,
   −0.155 copies; 1/5 and 2/5 benchmarks pass). DTD is the extreme case: the rule selects 0.53–0.63
   of 4 contexts and loses 0.83–0.84 nats against Static Utility.
3. **Only the forced-budget ablation satisfies clause (ii)** — and it does so cleanly (5/5
   benchmarks, paired *and* class-clustered CIs in both environments). The R110 diagnosis therefore
   transfers: on this family the frozen acceptance rule (`> 0`), not the ranking stage, is what
   destroys the gate.
4. **Demonstration utility is label-driven (§2.7).** Replacing candidate features by noise costs
   only ~18 % of the first-member marginal, while equalising labels flips every marginal negative.
   Consequently the state-conditioned marginal m(j|A) is nearly independent of A beyond its label
   composition, and no near-duplicate pool can produce set-conditioning headroom. This is a
   property of the *predictor* (a trained classification head plus label voting), not of the pool.
5. **The controlled copy environment does not behave as a "worst case" for diversity.** Copies of a
   seed are worth ~the same as real near-duplicates and still add ~40 % of a fresh cluster's value,
   because duplicated demonstrations multiply evidence for their class. A mechanism argument that
   assumes duplicates are worthless does not hold for cross-entropy here.
6. **Pool geometry is a design lever, not a neutral choice.** The first R112 rule (relevance-ordered
   seed + 3 nearest, 3 × 4 + 4 distractors) lost cluster relevance monotonically and produced
   `H_state ≈ 0` for a *different* reason (the low-relevance clusters were worthless); the balanced
   k-means rule (4 × 3 + 4) keeps every cluster useful but still yields `H_state ≈ 0`. Both variants'
   smoke runs are kept for audit; neither supports the prediction.
7. **No second predictor backbone** (unchanged from R110): both families use the same in-context
   transformer family, so the label-voting behaviour documented in §2.7 is a property of that
   predictor and has not been shown to disappear with a prototype/kNN predictor over demonstrations.
   That is the most promising follow-up: a predictor that must compare demonstration *images* to the
   query (no trained head, no label-only shortcut) is the natural way to test whether
   state-conditioning headroom can exist in an image family at all.
8. **Compute asymmetry.** R112 cells cost 112–456 s (vs 80–127 s for R110) because the redundancy
   diagnostic adds marginal passes and the copy environment perturbs candidate features; the copy
   environment itself is not more expensive to *select* in (identical predictor-row counts).
9. **Detectability caveat.** The pooled `H_state` CIs exclude zero ([+0.0001,+0.0006] natural,
   [+0.0001,+0.0002] copies), so "the sequential oracle beats the standalone oracle" is
   statistically detectable while being practically nil. Anyone citing prediction (i) as satisfied
   would be technically correct and scientifically wrong; the magnitude, not the sign, is the result.

## 4. Reproduction

```bash
cd /root/icl_ess_threshold/PAMI_MUR
python -m pytest tests/test_redundant_pool.py tests/test_demo_selection.py -q     # 20 tests

# natural near-duplicate environment
python experiments/demo_selection/run_r112_redundant_demo.py \
    --benchmarks cifar10,cifar100,svhn,eurosat,dtd --seeds 101,202,303 \
    --out-root results/raw/R112_redundant_demo_<stamp>
# controlled near-copy environment
python experiments/demo_selection/run_r112_redundant_demo.py \
    --benchmarks cifar10,cifar100,svhn,eurosat,dtd --seeds 101,202,303 --copy-jitter 0.002 \
    --out-root results/raw/R112_redundant_copies_<stamp>

python experiments/demo_selection/recompute_r112_redundancy.py \
    --run-root results/raw/R112_redundant_demo_<stamp> \
    --out results/derived/R112_redundant_demo_summary/redundancy_natural.json
python experiments/demo_selection/recompute_r112_redundancy.py \
    --run-root results/raw/R112_redundant_copies_<stamp> --copy-jitter 0.002 \
    --out results/derived/R112_redundant_demo_summary/redundancy_copies.json
python experiments/demo_selection/diagnose_r112_utility_curve.py \
    --run-root results/raw/R112_redundant_demo_<stamp> \
    --out results/derived/R112_redundant_demo_summary/utility_curve_natural.json
python experiments/demo_selection/analyze_r112.py \
    --run-root results/raw/R112_redundant_demo_<stamp> \
    --copies-root results/raw/R112_redundant_copies_<stamp> \
    --out-root results/derived/R112_redundant_demo_summary
```

## 5. Artifacts

| path | content |
|---|---|
| `experiments/demo_selection/redundant_pool.py` | k-means near-duplicate pool rule, copy perturbation, redundancy bookkeeping |
| `experiments/demo_selection/forced_budget.py` | forced-budget R-MUR ablation |
| `experiments/demo_selection/run_r112_redundant_demo.py` | R112 driver (pool variants, ablation, diagnostics) |
| `experiments/demo_selection/analyze_r112.py` | gates, hierarchical and pooled CIs, redundancy tables |
| `experiments/demo_selection/diagnose_r112_utility_curve.py` | utility curves behind `H_state` |
| `experiments/demo_selection/recompute_r112_redundancy.py` | corrected redundancy diagnostics from frozen predictors |
| `tests/test_redundant_pool.py` | 6 new tests (pool structure, query exclusion, forced-budget equivalence, forced-vs-threshold behaviour, diagnostics) |
| `results/raw/R112_redundant_demo_20260921_0526/` | natural environment: 15 cells, `run.log`, manifest, zero failures |
| `results/raw/R112_redundant_copies_20260921_0526/` | controlled near-copy environment: 15 cells, `run.log`, manifest, zero failures |
| `results/raw/R112_smoke/`, `results/raw/R112_smoke_copy/` | smoke runs (first R112 pool rule, 3 × 4 + 4) |
| `results/derived/R112_redundant_demo_summary/` | `summary.json`, `gate_table.csv`, `report_tables.md`, `utility_curve_{natural,copies}.json`, `redundancy_{natural,copies}.json`, `provenance.json` |
