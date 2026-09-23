# PAMI-MUR — archived evidence source

**No PAMI submission is planned.** This directory is retained for provenance,
experiments, immutable artifacts, tests and development history. All
advantageous material developed here that is not part of the already submitted
IPM companion is now assigned to the active TKDE manuscript in
`../TKDE_MUR/`.

Do not continue editing `paper/` as a submission candidate. New manuscript
work belongs in `../TKDE_MUR/paper/`.

## Assets transferred to TKDE

The active TKDE manuscript may use all of the following evidence:

- general smooth-loss response expansion and two-sided bound;
- exact Bregman response--utility identity;
- explicit softmax-cross-entropy Hessian bound and numerical sandwich check;
- squared-loss response sufficiency;
- theory-shaped bilinear and signed-curvature heads;
- strong subset-capable spatio-temporal transformer and DeepSets backbones;
- predictor/adaptation strength ladder;
- response observability and within-episode rankability probes;
- pool-geometry and novelty dose-response experiments;
- label-free redundancy-adaptive selection;
- recent-baseline and DELIFT-style selector audits;
- mutual information, clustering, k-center, DPP, policy-gradient and
  response-geometry alternatives;
- budget, backward-elimination, harmful-pool and candidate-library scaling;
- shortlist-removal and gate-ceiling experiments;
- CLIP demonstration-selection family, query-comparative predictor and
  forced-budget analysis;
- latency, expert-row and memory measurements;
- pilot-power, time-shift and split-half deployment validation;
- claim audits and artifact checkers.

## Evidence layout

- `experiments/`: experiment runners, analyses and checkers.
- `results/raw/`: immutable runs.
- `results/derived/`: sourced summaries used by papers.
- `refine-logs/`: one record per experiment, theory note, SOTA survey and
  claim audit.
- `tests/`: unit and numerical consistency tests.
- `paper/`: **frozen historical draft only**.

## Important provenance records

The final TKDE assembly should continue to rely on the existing immutable
records rather than copying numbers by hand, especially:

- `THEORY_GENERAL_LOSS.md`
- `R103_STRONG_BACKBONE_20260921.md`
- `R124_EXPLORATION_20260923.md`
- `R140_PROBE_LADDER_20260923.md`
- `R142_POOL_SIZE_20260923.md`
- `R143_ADAPTIVE_RULE_PROVENANCE_20260923.md`
- `R145_SCREEN_NOT_THE_BOTTLENECK_20260923.md`
- `R147_CLAIM_AUDIT_20260923.md`
- `R151_R152_RECENT_BASELINES_20260924.md`
- `R153_HEADROOM_PREDICTIVENESS_20260924.md`
- `R154_PILOT_POWER_20260924.md`
- `R155_PILOT_SHIFT_20260925.md`
- `R156_PILOT_CONSISTENCY_20260926.md`
- `R158_GATE_CEILING_20260927.md`
- `R159_GATE_NEGATIVE_20260928.md`
- `OBJECTIVE_COMPLETION_20260928.md`

This directory is now the evidence warehouse for TKDE, not a third manuscript.
