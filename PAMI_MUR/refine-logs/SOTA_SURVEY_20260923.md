# SOTA survey and baseline matrix (2026-09-23)

Purpose: fix the comparison set for "beat the strongest recent baselines" in
each dimension the paper touches. Sources were located by web search; the
implemented column states what actually runs in this repository.

## Dimension 1 — context / demonstration selection for in-context learning

* kNN retrieval over a frozen encoder ("KATE"-style) and dense retrieval
  ("EPR"-style) remain the standard strong baselines; the [Survey on
  In-context Learning](https://ar5iv.labs.arxiv.org/html/2301.00234v1) and the
  [AAAI 2026 comparative
  analysis](https://ojs.aaai.org/index.php/AAAI/article/view/35299) both report
  that similarity retrieval is hard to beat.
* Diversity-aware selection: DPP and MMR are the classical competitors;
  [Auto-regressive In-context Demonstration
  Selection](https://icml.cc/virtual/2026/poster/65920) (ICML 2026) and
  [Unifying and Optimizing Data Values for Selection via Sequential
  Decision-Making](https://icml.cc/virtual/2026/poster/65157) (ICML 2026) are
  recent sequential/decision-theoretic formulations — the closest neighbours to
  our setup.
* Submodular data selection: [DELIFT](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f9d446812a6fdc05453f4093e54831e8-Abstract-Conference.html)
  (ICLR 2025) combines facility location with k-means representatives. Both
  components were implemented separately from the start; the *composition* was
  added on 2026-09-24 (`select_delfit_style`, R152) and is **significantly
  below pooling on all six datasets** (paired deltas `-.018` to `-.023`, every
  interval resolved). Note the first version degenerated to
  `kmeans_representatives` because it used one cluster per budget slot; the
  pool is now over-segmented so the coverage stage actually chooses. The
  degenerate run `R150_ridge_delfit_20260924_0900` is retained but must not be
  cited.

## Dimension 2 — sensor / feature selection for prediction

* Greedy conditional mutual information under a Gaussian model is the classical
  sensor-placement criterion ([Krause, thesis](https://las.inf.ethz.ch/files/krause08thesis.pdf));
  we implement it as a label-free baseline on the training covariance.
* Recent work continues to add metrics and Bayesian variants
  ([spike-and-slab sensor
  selection](https://ieeexplore.ieee.org/document/10942441),
  [value of
  sensing](https://ieeexplore.ieee.org/document/11173441)); these are
  prognostics-specific and are cited as related rather than reproduced.

## Dimension 3 — retrieval/context for time-series foundation models

* [Align-RAG](https://export.arxiv.org/pdf/2608.05571) argues that alignment
  between retrieved context and the query is what matters for TSFM in-context
  learning; our response-alignment feature `<f_A, d>` is the analogue inside a
  fixed predictor, and we test the ordering it implies.

## Dimension 4 — forecasting backbones (why this paper does not compete on them)

The objective of this work is not forecasting accuracy but the *selection*
decision made on top of a fixed forecaster, so the forecasting dimension enters
as a controlled variable rather than as a leaderboard. The predictor is a
subset-capable spatio-temporal transformer trained on the dataset's training
split and then held fixed (or adapted by a controlled number of epochs); the
expert ladder of R138 varies its capacity across three transformer depths, a
masked-mean MLP and a per-target ridge expert, which is what makes the
identifiability result a statement about predictor strength rather than about
one architecture. Comparing against 2024--2026 forecasting SOTA
(iTransformer/PatchTST/STAEformer-class models) would confound the selection
question with backbone quality, and the ladder shows why that confound matters:
a 4% improvement in anchor error removes three quarters of the value routing
can realise. The relevant forecasting-side claim is therefore the negative one
already established --- stronger backbones reduce, and can eliminate, the value
of context selection --- and it is evidenced by the ladder, the adaptation
sweep (R141) and the pool-size sweep (R142).

## Implemented baseline matrix (13 + 2 oracles, one protocol)

| # | Baseline | Family | Implemented in |
|---|---|---|---|
| 1 | Random subset | control | `run_strong_backbone.py` |
| 2 | Relevance (correlation/kNN) | retrieval | `run_strong_backbone.py` |
| 3 | MMR | diversity | `run_strong_backbone.py` |
| 4 | DPP greedy MAP | diversity | `run_strong_backbone.py` |
| 5 | Facility location (submodular coverage) | submodular | `run_strong_backbone.py` |
| 6 | Greedy mutual information | sensor selection | `run_response_baselines.py` |
| 7 | Greedy k-center | representative/diversity | `run_response_baselines.py` |
| 8 | k-means representatives | DELIFT-style | `run_response_baselines.py` |
| 9 | Pool all candidates | no selection | `run_strong_backbone.py` |
| 10 | Standalone Utility (learned, response-free) | learned utility | `run_strong_backbone.py` |
| 11 | Cached MUR (learned, state-conditioned) | learned utility | `run_strong_backbone.py` |
| 12 | Policy-gradient subset selector | RL | `run_learned_policy_baseline.py` |
| 13 | Response-geometry policies (min/max disturbance, pool tail) | response-only | `run_response_baselines.py` — **measured only on 2026-09-24** (R151); before that the row was listed as implemented but had no raw run on the neural expert |
| 14 | R-MUR (frozen) | ours, marginal-utility | `run_strong_backbone.py` |
| 15 | R-MUR + curvature-corrected head | ours, theory-shaped | `run_strong_backbone.py --curvature-corrected` |
| 16 | R-MUR + bilinear head | ours, theory-shaped | `run_strong_backbone.py --router-kind bilinear --response-identity` |
| — | DELIFT-style (facility location over k-means medoids) | submodular + representative | `select_delfit_style` (R152) |
| — | Standalone / sequential oracle | upper bounds | `run_strong_backbone.py` |

## Theory predictions tested against this matrix

Two predictions were derived from the squared-loss identity and tested directly;
both are reported whatever the outcome.

1. **The degenerate-regime prediction is refuted (R151).** As the anchor absorbs
   its residual, `g_A -> 0` and the score collapses to `-||d_j||^2`, so the
   least-perturbing policy should win. It does not: it gains `+.0026` (METR-LA)
   and `+.0037` (PEMS-BAY) against pooling's `+.0133` and `+.0224`, is
   significantly below pooling on both, and ranks 13th and 16th of the eighteen
   non-oracle policies. The sign of the limit is visible (it is the best of the
   six geometry rules on METR-LA) but a score that only avoids harm cannot
   create value.
2. **The identifiability prediction holds (R138/R141/R142/R145).** Stronger
   anchors leave less realizable value at a flat structural opportunity;
   adaptation is the operation that destroys it; a larger library raises the
   opportunity without raising the realized share; and removing the screen
   entirely changes the result by `+.0012` `[-.0023,+.0044]`, so the boundary is
   in what the response reveals rather than in the efficiency compromise.

## Positioning

The gap this paper targets is not "another selection heuristic" but the fact
that **no feature-based ranker, learned or not, recovers the oracle value once
the fixed predictor is strong** (measured: screen-truth correlation ~ 0,
within-episode rank correlation of every observable-feature probe |rho| <= .12,
policy-gradient selection below random). The theory-shaped heads (14, 15) are
the response to that measurement: they hard-code the only two terms the
identity guarantees — an estimated interaction with the response and the exact
curvature penalty — instead of learning a scalar score over concatenated
features.
