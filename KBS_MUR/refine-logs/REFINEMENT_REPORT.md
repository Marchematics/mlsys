# Refinement Report

## Initial direction

The source idea correctly moved KBS away from pairwise finite-library relation
inference toward multi-candidate subset selection by set-conditioned marginal
utility. The initial plan already had a strong problem anchor and useful
synthetic/real-data gates.

## Corrections made during refinement

- Narrowed novelty from generic “marginal utility” to nonempty-state
  counterfactual end-task marginal learning with calibrated sequential stop.
- Added explicit boundaries against incremental utility, set coverage, MMR,
  and contextual submodular prediction.
- Split the method into deployable MUR-light and diagnostic MUR-full so exact
  candidate counterfactuals are not treated as free at inference.
- Replaced ambiguous data reuse with one frozen-expert, four-partition
  protocol and label-cache provenance.
- Corrected the multi-step theory: unconditional claims stop at per-visited-
  state approximate greedy quality; final-set guarantees require submodularity.
- Separated harm-safe selection, completeness-safe stopping, and uncertain
  stopping using calibrated utility intervals.
- Added a quantitative state-shift trigger for a single train-only roll-in
  augmentation followed by recalibration.
- Demoted ranking loss, MUR-full, Traffic, and Activity from the core method.

## Reuse boundary

Reusable engineering includes data loaders, encoders, experts, optimizers,
logging, seed control, bootstrap, and plotting. The KBS paper cannot reuse the
ICLR title, abstract, core equations, finite-library prior, BF-Gate, supervision
law, main figures, or main result tables.

## Final status

The proposal is ready for the five pre-experiment gates. Publication claims
remain blocked until those gates produce audited evidence.

