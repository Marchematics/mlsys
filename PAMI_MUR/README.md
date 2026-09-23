# PAMI-MUR

Local PAMI-level manuscript candidate. This is not a concurrent submission.

## Status of the four PAMI-level additions

| # | Addition | Status | Evidence |
|---|---|---|---|
| 1 | Strong nonlinear backbone (subset-capable) | done, negative gate | `refine-logs/R103_STRONG_BACKBONE_20260921.md` |
| 2 | Second task family (demonstration selection) | done, 3/5 paired pass | `refine-logs/R110_SECOND_FAMILY_20260921.md` |
| 3 | General smooth-loss response-utility theory | done | `refine-logs/THEORY_GENERAL_LOSS.md`, paper Sec. 5 + appendix |
| 4 | Real compute / latency / memory | in progress | `experiments/measure_compute.py`, R103 expert-row counts |

## What the additions changed

* The theory (addition 3) is a genuine strengthening: the squared-loss identity
  is now a special case of an exact Bregman identity, with a two-sided bound
  for general smooth losses and explicit constants for softmax cross-entropy.
  It is validated numerically (exact to 3.3e-7; 0 sandwich violations in 92,995
  samples, Spearman .971 between the first-order term and the true marginal).
* Additions 1 and 2 are **boundary conditions, not confirmations**. With a
  strong subset-capable transformer on METR-LA the structural headroom is
  preserved (H_state .075-.232) but the learned response-free routers become
  indistinguishable from random selection and R-MUR's advantage over them is
  not statistically resolved (Delta Static +.0028 [-.0012, +.0083], 16 targets
  x 3 seeds). The second family has no redundant candidates, so its
  state-conditioning headroom is ~0 and a nearest-neighbour relevance rule is
  within 0.1-0.6% of the oracle; R-MUR still beats both response-free learned
  routers on 3 of 5 benchmarks.
* The paper now states these limits explicitly (abstract, introduction,
  discussion, conclusion) instead of implying uniform superiority.

## Exploration round (2026-09-23): method changes, budget geometry, SOTA matrix

The frozen-method constraint was lifted. Findings, all from artifacts in
`refine-logs/R124_EXPLORATION_20260923.md`:

* **The strongest deployable baseline on the strong backbone is "use all 16
  contexts"** (+.0133). Our theory-shaped bilinear head is the best *selection*
  policy (+.0117) and reaches statistical parity with it; the frozen head is
  below it (+.0097); every response-free learned router is at random level and
  a policy-gradient selector is below random.
* **The oracle wants fewer contexts**: with true marginals the optimal budget is
  k = 3 (+.1048), and adding contexts reduces the gain, while random subsets
  improve monotonically. The two curves cross the protocol budget of four in
  opposite directions; the vertical distance between them is the
  identifiability gap (+.078).
* **Search direction is not the bottleneck**: backward elimination matches
  forward greedy to +.0002.
* **Two theory-guided fixes were tested**: curvature correction alone is neutral
  (-.0013 [-.0058,+.0032]); the bilinear head (explicit `<g, d> - ||d||^2`) is
  what moves the method. The "least-disturbance" heuristic derived from the
  curvature term is falsified (-.0047 below Static), showing that response
  *direction* carries the signal.
* Second family: the query-comparative predictor (delegate) gives genuine
  state-conditioning (H_state +.0126 pooled) and R-MUR beats Cached 5/5 but
  Static only 3/5; the bilinear-head port is running (R123).

## Method-iteration round (2026-09-23, rounds 4-5)

Four scoring heads were built and tested end to end: the frozen 7-dimensional
scalar head, a curvature-corrected head, a theory-shaped bilinear head
`<g_theta(state,cand), d> - 0.25*kappa`, and a signed variant
`<g_theta, d> + c_theta(state)*kappa`.

* In the ridge regime the method beats **all 13 baselines on both datasets**
  (METR-LA +.0146, PEMS-BAY +.0557; 1.8x and 2.2x the strongest non-selection
  baseline), and **8 of 12 training-free baselines are significantly harmful**
  (worse than using no context at all), including relevance, MMR and DPP.
* In the strong-backbone regime nothing beats pooling every candidate; greedy
  mutual information ties it (+.0134 vs +.0133) and every learned router is
  0.002-0.003 below.
* The sign of relevance ranking is governed by candidate-anchor redundancy; the
  effect replicates on PEMS-BAY and yields a deployable, label-free
  redundancy-adaptive rule (+.0083 METR-LA, +.0137 PEMS-BAY, held-out threshold).
* The second-family head ablation brackets the same ceiling with four different
  parameterisations: the learned curvature coefficient moves from -.25 to about
  +.1 (confirming the sign diagnosis) without changing performance, so the
  bottleneck is observability rather than the scoring function.

## Open items

* PEMS-BAY half of the strong-backbone sweep (running).
* Learned policy-gradient baseline and DeepSets second-backbone confirmation
  (queued).
* Redundancy-structured second-family variant, which is the pre-registered
  mechanism test for the second family (delegated, R112).
* Decide, with the user, whether a negative strong-backbone gate is acceptable
  for the target venue or whether the framing should move to the
  observability/limit-of-routing paper that the evidence supports.

## Layout

* `paper/` manuscript (`main.tex`, `sections/`, `figures/`).
* `experiments/` expert, router runner, diagnostics, baselines, figure code.
* `results/raw/` immutable runs; `results/derived/` summaries and reports.
* `refine-logs/` plans and run reports; `tests/` unit tests.
