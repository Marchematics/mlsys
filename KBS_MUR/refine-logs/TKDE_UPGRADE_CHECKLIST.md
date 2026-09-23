# TKDE first-submission checklist

## One-sentence claim
Auxiliary-context value is conditional on what has already been selected.

## Core objects
- `m(j | A) = F(A union {j}) - F(A)`
- structural headroom `H_state`
- one-step routing regret

## One method
Response-Aware Marginal-Utility Routing (R-MUR):
cached shortlist -> compact predictor-response summaries -> gap-weighted
ranking-calibrated reranking.

## Main evidence
- Controlled redundancy ladder.
- Controlled candidate-scaling margin analysis.
- Response-feature comparison.
- Six standard traffic datasets.
- 32 fixed targets per dataset.
- Multi-target oracle confirmation.
- Multi-target learned confirmation, three seeds.
- Hierarchical target -> seed -> episode CIs.
- Baseline matrix with 8-12 meaningful selection methods.

## Baseline matrix
Anchor-only, pool-all, relevance, MMR/diversity, Standalone Utility,
cached-only router, R-MUR, standalone oracle, sequential oracle; add a small
number of learned subset-selection or utility-routing baselines if directly
relevant.

## Writing constraints
- Target 12-14 IEEE double-column pages.
- No development run identifiers.
- No X1/X2/X3, MUR-Conservative, or oracle-development naming.
- Abstract one paragraph, no numerical results.
- Intro six paragraphs; Related Work four paragraphs.
- Figures: problem/R-MUR concept; structural headroom; multi-target benchmark;
  mechanism analysis.

## Stretch experiment
Two datasets x eight targets x three seeds with a subset-capable strong
predictor, to test whether the conclusions persist beyond the ridge
subset-predictor protocol. This is optional for competitiveness and not a
blocker for the first submission.

## Stop rules
Do not add datasets, methods, backbones, losses, or robustness suites unless
the multi-target three-seed analysis reveals genuine instability.
