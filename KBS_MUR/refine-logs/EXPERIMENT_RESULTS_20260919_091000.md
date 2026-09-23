# Experiment Results

## R002 — Synthetic oracle-headroom smoke

**Status**：PASS  
**Raw artifact**：`results/raw/R002_seed101_20260919_033100/result.json`

Configuration: 2,000 episodes, `K=8`, `B=2`, redundancy=.5,
harmful fraction=.25, seed=101.

| Metric | Value |
|---|---:|
| Base loss | 1.008675 |
| Pool-all loss | 0.801342 |
| Oracle-greedy loss | 0.610996 |
| Oracle gain | 0.397679 |
| Pool-all gain | 0.207333 |
| Mixed-sign state fraction | 1.000 |
| Positive standalone marginal fraction | 0.749 |
| Median duplicate marginal ratio after first copy | 0.000 |

### Gate interpretation

- Oracle headroom is material.
- Every tested initial state contains both positive and nonpositive candidates.
- Exact duplicate value collapses after the first copy, while its standalone
  value is positive in the relation-positive construction.
- Pool-all leaves substantial oracle headroom, so selection is meaningful.

This run validates the controlled environment contract only. It does not
support any claim about a learned router. Gates A/B/C/N remain untested.

## Preliminary Gate B/N pilot — learned MUR-light

These runs use 4,000 training episodes, 3,000 test episodes, 30 epochs, and
seeds 101/202/303. They are pilot evidence, not the final claim: no paired
bootstrap artifact or low-redundancy control has been generated yet.

| Setting | Method | Gain mean (SD) | Utility recovery | Duplicate selections |
|---|---|---:|---:|---:|
| K=8, r=.5 | Relevance | .2251 (.0031) | .523 | .201 |
| K=8, r=.5 | Coverage/MMR | .2497 (.0043) | .582 | .000 |
| K=8, r=.5 | Static Utility | .3471 (.0041) | .800 | .677 |
| K=8, r=.5 | **MUR-light** | **.4287 (.0054)** | **.986** | **.044** |
| K=16, r=.75 | Relevance | .1086 (.0021) | .514 | .240 |
| K=16, r=.75 | Coverage/MMR | .1233 (.0018) | .585 | .000 |
| K=16, r=.75 | Static Utility | .1564 (.0019) | .713 | .964 |
| K=16, r=.75 | **MUR-light** | **.2071 (.0012)** | **.947** | **.149** |

The pilot shows the intended direction in both settings: MUR-light improves on
Static Utility and Coverage while sharply reducing duplicate selections. The
original K=8, r=.75 smoke was marked invalid for the high-redundancy gate
because it leaves only two unique groups for a budget of two, making coverage
selection accidentally oracle-like. The formal high-redundancy setting is now
K=16, r=.75.

Raw artifacts:

- `results/raw/R021_pilot_seed101_r050_20260919_034100/result.json`
- `results/raw/R021_pilot_seed202_r050_20260919_034300/result.json`
- `results/raw/R021_pilot_seed303_r050_20260919_034500/result.json`
- `results/raw/R022_pilot_seed101_k16_r075_20260919_034800/result.json`
- `results/raw/R022_pilot_seed202_k16_r075_20260919_035000/result.json`
- `results/raw/R022_pilot_seed303_k16_r075_20260919_040000/result.json`

## Candidate-pool scaling — redundancy stress

A three-seed scaling check now covers the low-redundancy endpoint
($K=4$, no duplicates) and a larger high-redundancy pool ($K=32$). The mean
utility recovery and duplicate-selection counts are:

| Setting | Static utility recovery | MUR recovery | Static duplicates | MUR duplicates |
|---|---:|---:|---:|---:|
| $K=4$, no duplicates | .998 | .996 | .000 | .000 |
| $K=8$, medium redundancy | .800 | .986 | .677 | .044 |
| $K=16$, high redundancy | .713 | .947 | .964 | .149 |
| $K=32$, high redundancy | .543 | .776 | .951 | .202 |

The gap is small when every candidate is distinct and widens as redundancy
and pool size increase. The scaling check supports the mechanism claim in the
controlled environment. It is not an external-data result.

Raw scaling artifacts:

- `results/raw/R020_scale_k04_r000_s101_20260919_080000/result.json`
- `results/raw/R020_scale_k04_r000_s202_20260919_083000/result.json`
- `results/raw/R020_scale_k04_r000_s303_20260919_084000/result.json`
- `results/raw/R020_scale_k32_r075_s101_20260919_081000/result.json`
- `results/raw/R020_scale_k32_r075_s202_20260919_085000/result.json`
- `results/raw/R020_scale_k32_r075_s303_20260919_090000/result.json`

## Preliminary Gate A/C pilot — harmful candidates and calibrated stopping

The same three-seed protocol was rerun with a harmful-candidate fraction of
0.25 and a calibration split. The point router improves gain slightly over
Static Utility but does not yet improve its harm rate. The calibrated interval
router trades some gain for a near-zero negative-transfer rate and uses fewer
contexts.

| Method | Mean gain (SD) | Mean negative-transfer rate | Mean selected contexts |
|---|---:|---:|---:|
| Relevance | .0491 (.0059) | .2474 | 2.000 |
| Coverage/MMR | .0445 (.0059) | .2988 | 2.000 |
| Static Utility | .2808 (.0038) | .0558 | 2.000 |
| MUR-light point | .2999 (.0059) | .0657 | 1.968 |
| **MUR-light calibrated interval** | **.1898 (.0114)** | **.0006** | **.489** |
| Oracle | .3978 (.0067) | 0.0000 | 2.000 |

This is a useful frontier result, not a final Gate C claim. The calibrated
policy is conservative; threshold and coverage audits remain to be frozen
before test claims. It also shows that point MUR and harm-safe MUR must be
reported as separate policies.

Raw artifacts:

- `results/raw/R010_calibrated_pilot_seed101_k8_r050_h025_20260919_042500/result.json`
- `results/raw/R010_calibrated_pilot_seed202_k8_r050_h025_20260919_043000/result.json`
- `results/raw/R010_calibrated_pilot_seed303_k8_r050_h025_20260919_043500/result.json`

## Electricity smoke — code and split audit

The real-data smoke completed with the corrected historical-window sampler and
zero split-audit violations. The oracle has positive headroom, but the learned
router is not yet strong enough for Gate D: the point MUR gain is small and the
negative-transfer rate remains high, while the calibrated interval policy stops
all candidates in this smoke setting. These numbers are diagnostic only.

Raw artifacts:

- `results/raw/R060_electricity_cpu_smoke_20260919_053000/result.json`
- `results/raw/R061_electricity_gpu_smoke_20260919_054000/result.json`

The Electricity run remains **INCONCLUSIVE**. It does not support the planned
directional claim `MUR-light > Static Utility > Similarity`.

An exact prediction-delta diagnostic was also run on a smaller GPU smoke. It
slightly reduced negative transfer relative to MUR-light, but it still did not
approach the oracle and is not a deployable result. This does not justify
promoting the high-cost diagnostic into the main method.

Raw artifact:

- `results/raw/R061_electricity_full_gpu_smoke_20260919_060000/result.json`

A larger smoke with more training episodes improved MUR-light slightly over
Static Utility, but the gain remained small and negative-transfer remained
substantial. Increasing the same-client candidate fraction produced the same
pattern. These diagnostics do not justify a confirmatory Electricity table.

Raw artifacts:

- `results/raw/R061_electricity_large_smoke_20260919_061500/result.json`
- `results/raw/R061_electricity_same75_smoke_20260919_063000/result.json`

A same-client-only candidate pool also failed to create a decisive real-data
router margin. It confirms that the current issue is not only the proportion
of unrelated candidates; the frozen Fourier-ridge expert and observable
context features do not yield a reliable utility signal for this selection
task.

Raw artifact:

- `results/raw/R061_electricity_same100_smoke_20260919_064500/result.json`
