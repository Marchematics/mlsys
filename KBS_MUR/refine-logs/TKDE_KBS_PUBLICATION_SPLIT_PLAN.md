# TKDE / KBS publication split plan

## Rule
Do not submit two manuscripts concurrently if they share essentially the same
core research, method, main experiments, or conclusions. A renamed or lightly
modified version of R-MUR is not a separate KBS paper.

## Primary submission: TKDE
- Core question: how does context value change with the selected set, and can
  that value be learned from deployable information?
- Core object: state-conditioned marginal utility `m(j | A)`.
- Method: R-MUR, cached shortlist + compact predictor-response summaries +
  gap-weighted ranking.
- Main evidence:
  - controlled redundancy and structural headroom;
  - candidate-scaling margin compression;
  - response observability;
  - six standard traffic datasets;
  - 32 fixed targets per dataset;
  - multi-target learned confirmation;
  - multi-target oracle confirmation.
- All R070-R080 evidence belongs to this line.

## Independent KBS direction
- Core question: when is an uncertain marginal-utility estimate reliable
  enough to act on without causing negative transfer?
- Decision object: select / abstain / stop under risk.
- Method direction: conditional risk bound
  `L(j | A) = mhat(j | A) - q_alpha(z_{A,j})`,
  where the uncertainty term depends on state, candidate, response
  uncertainty, and decision margin.
- Main experiments must not be a re-formatted copy of the TKDE six-dataset
  table. The central setting should be harmful, unreliable, or shifted
  contexts, with negative-transfer rate, coverage, selective risk, and
  calibration as primary quantities.
- R072 gain-harm analysis can be a starting observation, but it is not yet a
  complete independent paper.

## Order of operations
1. Finish the R080b three-seed confirmation and final diagnostics.
2. Freeze the TKDE evidence package.
3. Adapt the current manuscript to the TKDE target while preserving the
   current R-MUR claims.
4. Only after that, develop the KBS risk-calibrated selective-routing method
   and its harm-focused benchmark.
5. If both eventually exist and still overlap, disclose the related manuscript
   in the cover letter rather than attempting to hide the relationship.
