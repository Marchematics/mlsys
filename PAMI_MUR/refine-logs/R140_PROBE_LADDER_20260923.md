# R140 — Does the identifiability probe order expert rungs? (negative)

**Date:** 2026-09-23
**Status:** negative result, reported as negative and acted on in both manuscripts.

## Question

The theory (`prop:suff` in PAMI, Corollary 1 in TKDE) says the identifiable term
is the anchor's own residual, so a stronger anchor should leave less realizable
value. A cheap consequence would be a *deployment rule*: fit a probe to the
marginal, and if the probe is weak, do not select — fall back to the whole pool.
Round 10 tested whether the probe supports such a rule.

## Design

Two extremes of the R138 expert ladder (identical 8-target cells, seed 101):

| rung | anchor MSE | realized share |
|---|---|---|
| transformer, 1 layer `d=32` ("small") | .3721 | .268 |
| transformer, 3 layer `d=128` ("large") | .3562 | .063 |

Probe: `experiments/diagnose_observability.py --ridge-only`, METR-LA, seed 101,
three targets (0, 58, 147), **2,000 train / 1,200 test episodes** each. An
earlier underpowered run (800/800, one target, four rungs) is retained at
`results/derived/R139_probe_{small,medium,large,deepsets}/`.

## Result

| statistic | small (weak anchor) | large (strong anchor) |
|---|---|---|
| probe out-of-sample R², state only | −.004 ± .019 | −.015 ± .028 |
| probe R², state + response | −.027 ± .020 | +.022 ± .057 |
| probe value excess over random (share of oracle) | +.048 ± .037 | +.067 ± .072 |
| ladder realized share on the same rungs | **.268** | **.063** |

1. **Probe R² is zero for both extremes.** Adding response features does not fix
   it; at this sample size the added coefficients are noise.
2. **The value statistic must be read against a per-cell random baseline**, which
   ranges from .115 to .294 of the oracle across the six cells. Excess over
   random is only .048–.067 of the oracle, with a per-cell spread of .09.
3. **The probe's excess is a quarter of the effect it is meant to explain**
   (.205 realized-share gap on the same two rungs) and is noisier than it is
   large. Any threshold low enough to admit the strong anchor also admits the
   weak one, so a probe-gated fall-back rule degenerates to always falling back.
   **The rule is not shipped.**
4. The underpowered four-rung run agreed: it could not order the rungs either,
   and the R² ordering it produced was the reverse of the realized-share
   ordering.

## What survives, and what is now claimed

- **Positive, within family.** Across the four *neural* rungs anchor-only MSE
  orders realized share perfectly (Spearman +1.0), fitted slope 10.5
  realized-share points per unit MSE — a 1% anchor improvement corresponds to
  about +.038 share. This is a genuine, separately-measured confirmation of the
  corollary (plain prediction error vs. captured fraction of the oracle).
- **Negative, across families.** Including the ridge rung the rank correlation
  falls and the slope halves to 4.9: ridge has the *worst* anchor MSE (.3782)
  and a share of .198, below the weak neural rung's .268. Residual magnitude is
  a sufficient statistic only *within* a fixed hypothesis class; across classes
  the inductive bias changes which candidates are jointly improvable, so
  accuracy and residual geometry decouple.
- **Negative, methodological.** Predictability (R²) and value capture are
  different quantities, and neither is a usable per-cell statistic at attainable
  sample sizes. The identifiable quantity must be measured by running the
  routing protocol itself — screening, budget and objective included.

## Manuscript actions taken

- `PAMI_MUR/paper/sections/6_experiments.tex`: added the probe-gate negative to
  the observability discussion; replaced "which the measurement protocol
  estimates directly" with "has to be measured, not inferred from a
  leaderboard"; added the Spearman/slope quantification.
- `PAMI_MUR/paper/sections/7_discussion.tex`: the practical diagnostic list now
  excludes a fitted probe and says why.
- `TKDE_MUR/paper/sections/7_experiments.tex`: same negative in Measurement 2
  and in the ladder paragraph, with the explicit statement that the diagnostic
  protocol ranks *geometries*, not cells.
- Also fixed a pre-existing undefined reference in PAMI
  (`cor:identifiability` → `prop:suff`, TKDE-only label). PAMI now compiles with
  zero undefined references.

## Provenance

- `results/derived/R140_probe_{small,large}_t{0,58,147}/observability.json`
- `results/derived/R140_probe_deep.log`
- `results/derived/R140_probe_ladder_summary/summary.json` (full per-cell table)
- `results/derived/R138_cross_arch_ladder/cross_architecture_ladder.json`
