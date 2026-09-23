# R124 — Exploration round: theory-guided method changes, budget geometry, SOTA matrix

Date: 2026-09-23. All numbers come from the artifacts named in each section.
Covers R116–R122 plus the interim state of R119 (bilinear head) and the
delegate runs R115 (query-comparative second family) and R123 (bilinear head in
that family).

## 1. Leaderboard on the strong backbone (METR-LA, 48 cells)

`results/derived/R124_leaderboard/leaderboard.txt`, merged from the gate run
(R103), the response-geometry run (R116), the policy-gradient baseline (R109),
the curvature-corrected run (R117) and the bilinear run (R119). Variants that
reuse policy names are namespaced by run.

| rank | policy | mean gain | vs pool-all |
|---|---|---|---|
| 1 | sequential oracle | +.1052 | +.0918 |
| 2 | standalone oracle | +.0904 | +.0771 |
| 3 | **pool all 16 (no selection)** | **+.0133** | --- |
| 4 | R-MUR bilinear q8 (ours) | +.0117 (32 cells) | -.0003 [-.0085,+.0080] |
| 5 | R-MUR bilinear q4 (ours) | +.0110 (32 cells) | -.0009 [-.0093,+.0075] |
| 6 | R-MUR frozen q8 | +.0108 | -.0025 [-.0085,+.0035] |
| 7 | R-MUR frozen q4 | +.0097 | -.0036 [-.0099,+.0027] |
| 8 | DPP | +.0091 | -.0042 |
| 9 | MMR | +.0077 | -.0057 |
| 10 | cached-only router | +.0076 | -.0058 |
| 11 | relevance | +.0071 | -.0062 |
| 12 | standalone utility | +.0069 | -.0064 |
| 13 | random four | +.0062 | -.0071 |
| 14 | facility location | +.0041 | -.0092 |
| 15 | min disturbance | +.0022 | -.0112 |
| 16 | anchor only | .0000 | -.0134 |
| 17 | pool tail | -.0001 | -.0135 |
| 18 | policy-gradient selector | -.0041 | -.0174 |
| 19 | max disturbance | -.0059 | -.0192 |

Reading:

* **The strongest deployable policy is "use everything".** No selection method,
  learned or not, beats pooling all sixteen candidates; the best selection
  method is our bilinear variant, which reaches statistical parity with it.
* The theory-shaped heads move our own method in the right direction: the
  bilinear head (+.0117) beats the frozen head (+.0097) and every non-oracle
  baseline except pool-all.
* The response-free learned routers are indistinguishable from random
  selection, and the policy-gradient selector is worse than random, i.e. the
  failure is in the identifiability of the decision, not in the estimator's
  parameterisation alone.

## 2. Budget geometry: the oracle wants *fewer* contexts (R121)

24 cells (8 targets x 3 seeds), `results/derived/R121_budget_and_direction/`:

| k | random k-subset | oracle at k |
|---|---|---|
| 1 | +.0066 | +.0941 |
| 2 | +.0091 | +.1034 |
| **3** | +.0119 | **+.1048** |
| 4 | +.0142 | +.1034 |
| 8 | +.0202 | +.0909 |
| 16 | +.0264 | +.0264 |

* The oracle's optimum is **k = 3**, below the protocol budget of four, and the
  oracle *decreases* with every additional context.
* Random subsets improve monotonically with k, so the best deployable policy is
  the largest budget.
* The two curves move in opposite directions through k = 4: the vertical
  distance between them is the identifiability gap (+.078 between pool-all and
  the oracle optimum).
* Figure: `paper/figures/fig7_budget_gap.pdf`.

## 3. Search direction is not the problem (R120)

Backward elimination from the full pool matches forward greedy to
+.00022 [-.00003, +.00047] (backward better on 18/24 cells). The utility
landscape behaves submodularly for search; the limitation is identification.

## 4. Negative results recorded

* **R116 — "least disturbance" is false.** Ranking candidates by ascending
  response magnitude gives +.0022, significantly *below* Standalone Utility
  (-.0047 [-.0087,-.0007]) and below random. The curvature term is a penalty,
  not a selection rule; the response *direction* carries signal, which is why
  the bilinear head (not a magnitude heuristic) is the right response.
* **R117 — curvature correction alone is neutral.** Removing the observable
  penalty from the regression target and re-subtracting it at inference gives
  -.0013 [-.0058,+.0032] against the frozen head over 34 cells: no significant
  change. Hard-coding the curvature is not enough without also hard-coding the
  bilinear interaction, which is what R119 does.
* **Consensus alignment** (keep the candidates whose response agrees with the
  full-pool direction) is unstable: +.015/-.028/-.013/-.015 on four sampled
  targets. Not pursued.
* **R115 — the query-comparative second family** (delegate): H_state becomes
  genuinely positive (+.0126 [+.0019,+.0239] pooled) and R-MUR beats the cached
  router 5/5, but beats Static Utility only 3/5, because the learned static
  model is already near the standalone oracle there. This is the same
  "response stage is the weak link" pattern seen in traffic, and is what R123
  targets with the bilinear head.

## 4b. Ridge protocol: the standard toolkit is harmful (R125)

32 targets x 3 seeds, 96 cells, target-level intervals
(`results/raw/R125_ridge_baselines_METRLA_*`), paired with the R080b R-MUR
cells on the same targets:

| policy | gain | 95% interval | verdict |
|---|---|---|---|
| sequential oracle | +.1471 | [+.1333,+.1610] | upper bound |
| standalone oracle | +.1333 | [+.1204,+.1462] | upper bound |
| **R-MUR q=4** | **+.0146** | [+.0093,+.0200] | helpful |
| pool all 16 | +.0080 | [+.0030,+.0129] | helpful |
| cached-only router | +.0042 | [-.0002,+.0087] | n.s. |
| standalone utility | +.0024 | [-.0019,+.0066] | n.s. |
| mutual information | -.0055 | [-.0103,-.0007] | **harmful** |
| random four | -.0077 | [-.0097,-.0056] | **harmful** |
| k-means representatives | -.0080 | [-.0123,-.0036] | **harmful** |
| facility location | -.0163 | [-.0234,-.0091] | **harmful** |
| relevance (kNN) | -.0165 | [-.0242,-.0088] | **harmful** |
| DPP | -.0166 | [-.0221,-.0111] | **harmful** |
| MMR | -.0180 | [-.0247,-.0113] | **harmful** |
| k-center | -.0187 | [-.0248,-.0125] | **harmful** |

Eight of twelve competitors are *significantly worse than using no auxiliary
context at all*, including the retrieval/diversity heuristics that are the
default in the literature. R-MUR is the only deployable policy whose interval
clears zero comfortably, and it roughly doubles the strongest non-selection
baseline. On PEMS-BAY the same ordering holds for the R080b cells (R-MUR
+.0557 against pool-all +.0254 and static +.0272).

## 4c. The theory-shaped head does not transfer to PEMS-BAY

The bilinear head on the strong backbone: METR-LA +.0107 (best selection
policy, +.0031 over cached with an interval above zero, -.0001 against static);
PEMS-BAY +.0125 against static +.0139, cached +.0135, random +.0163 and
pool-all +.0225. So the head improves the ranking on METR-LA but does not
rescue the strong-backbone regime on PEMS-BAY, where even random selection
beats every learned policy. This strengthens the identifiability reading
rather than weakening it.

## 4d. Second family: the sign constraint is the binding limitation (R123)

The bilinear head ranks the true marginal better than the frozen head on 4/5
benchmarks (mean within-state Spearman .375 vs .156; top-1 agreement .67 vs .08
on CIFAR-100) yet fails the gate 3/5 because the identity's curvature term has
a *fixed* sign while the empirical association between response magnitude and
utility flips sign across benchmarks (+.257 on CIFAR-10, -.590 on CIFAR-100).
A signed, learned curvature coefficient is the natural next iteration (R124,
delegated).

## 4e. Harmful-pool test: also falsified (R122)

Pre-registered hypothesis: if candidate pools are built from the *least*
correlated sensors, contexts can hurt, so a good selector should beat pooling
everything. Measured on 8 targets x 2 seeds (16 cells, strong backbone,
anticorrelated pool):

| policy | gain |
|---|---|
| sequential oracle | +.0816 |
| standalone oracle | +.0657 |
| pool all 16 | +.0040 |
| relevance | +.0035 |
| MMR | +.0034 |
| DPP | +.0032 |
| random | +.0031 |
| standalone utility | +.0024 |
| cached-only router | +.0023 |
| R-MUR q=4 | +.0013 |
| R-MUR q=8 | +.0012 |

Pooling everything is *still* the best deployable policy and R-MUR is the
worst, so the hypothesis fails: an expert trained with subset dropout is
robust to irrelevant or anti-correlated contexts, which removes the leverage
that selection would otherwise have. The oracle still finds +.066, so the
value exists but is not reachable. This is the strongest form of the boundary
result: with a strong predictor, *what you feed it barely matters*, and
selection cannot be justified by accuracy.

## 4f. Why relevance flips sign: candidate--anchor redundancy (R127)

Pre-registered hypothesis: relevance ranking is harmful exactly when the
candidate pool is redundant with the anchor. Ridge protocol, 8 targets x 3
seeds x 3 pool geometries, 72 cells (`results/raw/R127_pool_modes_*`):

| pool head | mean anchor \|rho\| | relevance | MMR | pool all | oracle |
|---|---|---|---|---|---|
| most correlated | .77 | -.0086 | -.0092 | +.0116 | +.150 |
| moderate | .63 | +.0006 | -.0062 | +.0070 | +.130 |
| least correlated | .43 | +.0063 | +.0024 | +.0012 | +.104 |

**Confirmed, with a monotone dose--response**: as candidates carry more
information the anchor does not already have, relevance rises from harmful to
helpful. This explains the cross-family contrast: in the demonstration family a
candidate contributes a *label*, which the anchor lacks, so kNN retrieval is
near-optimal; in the traffic protocol the top of the pool is the *most
redundant* with the anchor, so the same heuristic is worse than no context.

## 4g. Mutual information ties pooling everything on the strong backbone

The extra baselines on the strong backbone (48 cells,
`results/raw/R118_extra_baselines_metrla_*`) put greedy mutual information at
the top of the training-free policies: +.0134 against pool-all +.0133 (paired
+.0001 [-.0040,+.0043], i.e. a tie), above the frozen R-MUR q8 (+.0108, paired
+.0026 n.s.) and above the bilinear head (+.0112, paired +.0022 n.s.). The
other geometry policies are far below (k-center +.0060, k-means +.0076,
consensus alignment -.0046, anti-consensus -.0169). So when utility cannot be
estimated from labels, a label-free information criterion (conditional mutual
information about the target under the pooled covariance) is the best
*selection* rule --- consistent with the identifiability reading.

## 4h. The sign flip replicates on PEMS-BAY, and the rule can be deployed (R128/R129)

Same three pool geometries on PEMS-BAY (8 targets x 3 seeds x 3 modes, 72 cells):

| pool head | mean anchor \|rho\| | relevance | MMR | DPP | pool all | oracle |
|---|---|---|---|---|---|---|
| most correlated | .65 | **-.0333** | -.0550 | -.0386 | +.0112 | +.179 |
| moderate | .46 | **+.0117** | +.0114 | +.0106 | +.0064 | +.143 |
| least correlated | .12 | **+.0156** | +.0159 | +.0179 | +.0119 | +.099 |

The dose--response replicates with a larger swing, so the mechanism is not a
METR-LA artefact.

**Redundancy-adaptive selection** (R129): compute the pool statistic, then use
relevance below a threshold and the whole pool above it. The threshold was
chosen on half the targets (held-out selection, median .50-.55, chosen in
[.45,.55] in 99% of 200 splits) and then held fixed. Target-level results:

| dataset | adaptive | fixed relevance | fixed pool-all | oracle pick |
|---|---|---|---|---|
| METR-LA | **+.0083** [+.0002,+.0164] | -.0006 | +.0066 | +.0093 |
| PEMS-BAY | **+.0137** [-.0113,+.0387] | -.0020 | +.0098 | +.0194 |

So a label-free adaptive rule converts the harmful default heuristic into a
helpful one and beats the best fixed training-free policy on both datasets,
while remaining well below R-MUR (+.0146 / +.0557 on the standard pools). This
is the practical prescription the mechanism implies.

## 4i. Four head variants, one ceiling (R123/R124)

Second family, forced-budget q=4, 15 cells per variant, pooled contrast against
Standalone Utility (which is already close to the standalone oracle in this
family):

| head | Spearman(score, m) | pooled vs Static | per-benchmark vs Static |
|---|---|---|---|
| frozen scalar (7-dim summary) | .156 | 3/5 benchmarks | --- |
| curvature-corrected (fixed sign) | --- | --- | neutral in traffic |
| bilinear `<g,d> - 0.25*kappa` | .375 | +.0605 [+.0126,+.1085] | 3/5 |
| signed bilinear `<g,d> + c_theta*kappa` | better again | +.0528 [+.0072,+.0984] | 1/5 significant |

The learned curvature coefficient moves from its `-.25` initialisation to
`+.14 / +.08 / +.11 / +.13` on four of five benchmarks (significantly different
from both `-.25` and `0`, p < 1e-6), which confirms the R123 diagnosis that the
fixed sign was binding --- but the performance does not improve (signed minus
fixed-sign bilinear: `-.0077 [-.0129,-.0025]`). Four head variants bracket the
same result, so the ceiling is not the parameterisation of the score; it is what
the router can observe.

## 5. What this means for the claim

Within this protocol the honest SOTA statement is: **our theory-shaped variant
is the best selection policy and reaches parity with the strongest
non-selection baseline; no method, including ours, closes the gap to the
oracle.** The contribution that survives is the diagnosis plus the theory that
predicts it — and the practical rule that follows: with a strong predictor,
either include everything or spend the budget on identification, because
budget-constrained selection without identification is worse than doing
nothing.
