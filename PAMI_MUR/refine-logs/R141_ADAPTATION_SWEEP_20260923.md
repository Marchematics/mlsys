# R141 — Test-time adaptation sweep: what destroys realizable value (METR-LA)

**Date:** 2026-09-23
**Run:** `results/raw/R141_adapt_sweep_20260923_1500` (80 cells, 5 levels × 8 targets × 2 seeds)
**Analysis:** `results/derived/R141_adaptation_sweep/`

## Question

The expert ladder showed that a stronger anchor leaves less realizable value, but
the rungs differed in *architecture*, so strength and inductive bias moved
together. Per-target fine-tuning is the operation that absorbs the anchor's
residual, so sweeping the number of adaptation epochs gives a controlled
one-parameter version of the same test: same architecture, same pools, same
episodes, only the anchor's adaptation varies.

Protocol: METR-LA, R101 spatio-temporal transformer, `K=16`, budget 4,
`q ∈ {4,8}`, 4,000 train / 1,000 calibration / 2,000 test episodes per cell,
targets `0,27,54,82,109,137,164,192`, seeds `101,202`. Only
`--finetune-epochs` changes between levels.

## Result (16 matched cells in every level)

| finetune epochs | anchor val MSE | realized gain | pool all | random | standalone oracle | realized share | Δ(router − pool) |
|---|---|---|---|---|---|---|---|
| 0 | .4364 | **+.0169** | +.0124 | +.0048 | +.0858 | **.164** [+.082, +.240] | +.0045 [−.0036, +.0130] |
| 2 | .4056 | +.0035 | +.0068 | +.0016 | +.0831 | .024 [−.080, +.117] | −.0033 [−.0165, +.0139] |
| 5 | .4023 | +.0070 | +.0046 | −.0002 | +.0842 | .051 [−.010, +.122] | +.0024 [−.0092, +.0188] |
| 10 | .4003 | +.0076 | +.0057 | +.0005 | +.0851 | .057 [−.004, +.126] | +.0019 [−.0097, +.0183] |
| 30 | .4006 | +.0078 | +.0059 | +.0006 | +.0853 | .058 [−.004, +.128] | +.0019 [−.0097, +.0183] |

Anchor MSE is the `empty`-mask held-out MSE from
`results/derived/R141_anchor_mse_ft*`; the zero-shot column there is identical
(.4364) at every level, which is the expected control.

1. **The structural opportunity is invariant to adaptation.** The standalone
   oracle moves only from +.0858 to +.0831 across the whole range. Selection
   matters just as much after 30 epochs of fine-tuning as before any.
2. **Adaptation destroys realizable value, and the loss is resolved.** Paired
   across the 16 cells, the realized share falls $-.140$ $[-.197,-.085]$ and the
   learned gain falls $-.0134$ $[-.0198,-.0075]$ between 0 and 2 epochs. The
   endpoint contrast is also resolved ($-.106$ $[-.170,-.030]$ share, $-.0091$
   $[-.0160,-.0010]$ gain). Two epochs of fine-tuning — a $7\%$ MSE improvement —
   remove about $85\%$ of the value routing could realize.
3. **The apparent partial recovery is NOT significant, and we do not claim it.**
   The share reads .024, .051, .057, .058 at 2, 5, 10 and 30 epochs, but the
   paired increments are $+.027$ $[-.036,+.119]$, $+.006$ $[-.002,+.016]$ and
   $+.0015$ $[+.0000,+.0046]$, and the 2→30 contrast is $+.035$
   $[-.019,+.122]$. Only the last of these marginally excludes zero and its
   effect size is negligible. The defensible statement is a step down followed
   by a flat plateau, not a U-shape; the earlier reading of a recovery was an
   artefact of not pairing the cells.
4. **At no level does the router significantly beat pooling.** Every paired
   interval spans zero. The 0-epoch level gives the largest point estimate
   (+.0045). Note that the 0-epoch share CI excludes zero, so the router does
   capture a real share of the oracle there; it simply does not separate from
   the trivial pool-everything policy.
5. **Adaptation buys almost nothing in accuracy after the first step.**
   Anchor MSE goes .4364 → .4056 → .4006 from 0 to 2 to 30 epochs: the first two
   epochs deliver $7.1\%$, the next twenty-eight deliver $1.2\%$. The
   identifiability step therefore coincides with the *first* adaptation step,
   where the residual changes from mostly systematic to mostly noise, and not
   with any later accuracy gain.

## A correction forced by this run

The published regime-B claim ("nothing beats pooling every candidate") is **not
supported by paired tests**, and the sweep exposed it. Recomputing from the raw
R103 runs:

| comparison (METR-LA, 48 cells) | paired Δ vs pool-all | 95% CI |
|---|---|---|
| response-aware, frozen `q=4` | −.0036 | [−.0091, +.0031] |
| response-aware, frozen `q=8` | −.0025 | [−.0077, +.0040] |
| cached-only router | −.0057 | [−.0104, +.0001] |
| relevance | −.0062 | [−.0088, −.0034] |
| MMR | −.0057 | [−.0081, −.0031] |
| stand-alone utility | −.0064 | [−.0106, −.0020] |

and on the 16-cell sweep subsample the same router–pool difference is
**+.0036** [−.0078, +.0193]: the point estimate changes sign with the target
sample.

On PEMS-BAY the comparison *is* resolved, against routing: pool-all is
significantly above the frozen response-aware head (−.0094 [−.0135, −.0053]),
the cached router (−.0091 [−.0126, −.0057]) and random (−.0063 [−.0079,
−.0048]).

**Correct claim for regime B.** Pooling significantly beats every
response-free rule on both datasets; it significantly beats the response-aware
family on PEMS-BAY and *ties* it on METR-LA. The response signal is the only
thing that keeps a learned router in contact with pooling once the anchor is
strong. All policies remain an order of magnitude below the oracle, so the
failure is one of realisability, not of ranking.

## Reproduction check

R141 ft10 reproduces R103 ft10 on the 16 shared cells: `pool_all` +.0057 vs
+.0053, `ranked_response_q4` +.0076 vs +.0089, `oracle_static` +.0851 vs
+.0855. Residual differences ≤.0013 on the means (≤.0049 cell-wise), so the
sweep is on the same protocol as the published numbers.

## Manuscript actions

- `TKDE_MUR/paper/sections/7_experiments.tex`: regime B now reports the paired
  intervals and the METR-LA-tie / PEMS-BAY-loss split instead of "nothing beats
  pooling".
- `PAMI_MUR/paper/sections/6_experiments.tex`: same correction, with the
  per-policy intervals.
- The adaptation sweep itself is a new, cleanly controlled confirmation of the
  identifiability reading and is reported as such.

## A second correction: the harmful-pool ordering

While recomputing for the table above we also tested the regime-B claim about
the anticorrelated pool ("pooling everything is still the best deployable policy
and R-MUR the worst"). It is a point-estimate ordering with no resolved
differences. From `results/raw/R122_harmful_pool_metrla_20260923_0305` (16
cells), paired against pooling:

| policy | mean | paired Δ vs pool | 95% CI |
|---|---|---|---|
| relevance | +.0035 | −.0005 | [−.0033, +.0023] |
| MMR | +.0034 | −.0006 | [−.0035, +.0027] |
| DPP | +.0032 | −.0008 | [−.0038, +.0024] |
| random | +.0031 | −.0009 | [−.0029, +.0013] |
| Standalone Utility | +.0024 | −.0016 | [−.0038, +.0011] |
| cached-only router | +.0023 | −.0017 | [−.0049, +.0021] |
| R-MUR `q=4` | +.0013 | −.0027 | [−.0072, +.0020] |
| facility location | +.0009 | −.0031 | [−.0055, −.0009] |

Only facility location is significantly below pooling. Worse for the claim's
stability, on an independently built anticorrelated pool
(`results/raw/R127_pool_modes_20260923_0546`, 24 cells, three seeds) relevance
is significantly **above** pooling: +.0051 [+.0025, +.0077]. So the regime-B
ordering is not even stable in sign across pool constructions. Corrected claim:
pooling has the highest point estimate in the harmful-pool variant, nothing
separates from it there, and on another anticorrelated construction a
response-free rule beats it. What is stable is the gap to the oracle
(+.066 to +.082).

Both corrections are applied in `TKDE_MUR/paper/sections/7_experiments.tex`,
`TKDE_MUR/paper/sections/9_conclusion.tex`, `TKDE_MUR/paper/main.tex` and
`PAMI_MUR/paper/sections/6_experiments.tex`.
