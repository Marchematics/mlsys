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
