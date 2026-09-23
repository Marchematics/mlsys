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

The fixed-$K$ redundancy sweep provides the orthogonal axis. At $K=16$, mean
utility recovery for Static Utility and MUR is:

| Duplicate ratio | Static utility | MUR | MUR-Conservative | Static duplicate rate | MUR duplicate rate |
|---:|---:|---:|---:|---:|---:|
| 0 | .957 | .942 | .939 | .000 | .000 |
| .25 | .840 | .878 | .852 | .290 | .137 |
| .5 | .722 | .846 | .792 | .654 | .226 |
| .75 | .713 | .947 | -- | .964 | .149 |

The last row uses the earlier high-redundancy run with a larger training
budget; its conservative interval was not stored with the same protocol and is
left blank. The fixed-$K$ sweep shows the main pattern: MUR's advantage grows
with redundancy, while its own recovery falls when the candidate ranking task
becomes harder.

Raw fixed-$K$ artifacts are stored under
`results/raw/R020_grid_k16_r*_s*_20260919_10/`.

## Candidate-ranking diagnosis at $K=32$

The diagnostic run separates pointwise utility regression from candidate
ranking. Utility MAE is similar across random, oracle-prefix, and MUR-rollout
states, while top-1 accuracy falls and one-step regret rises on MUR-rollout
states. This pattern is consistent with a max-over-candidates ranking effect,
with a smaller contribution from rollout-state shift.

| State source | Utility MAE | Top-1 accuracy | One-step regret |
|---|---:|---:|---:|
| Random | .0242 | .155 | .0220 |
| Oracle prefix | .0229 | .140 | .0214 |
| MUR rollout | .0245 | .122 | .0300 |

This is one three-seed diagnostic run and is not a final confidence interval.

## Oracle ladder at fixed $K=16$, $B=4$

A five-seed oracle ladder now separates formulation headroom from learned
utility estimation. Oracle-Static uses the true standalone utility
$m(j\mid\emptyset)$ once and keeps that order fixed. Oracle-Greedy re-evaluates
the true marginal utility after every selection. MMR's redundancy coefficient
is selected on the calibration split; no test labels are used for this tuning.

| Duplicate ratio | Oracle-Static gain | MUR gain | Oracle-Greedy gain | $H_{\mathrm{state}}$ | $H_{\mathrm{est}}$ |
|---:|---:|---:|---:|---:|---:|
| 0 | .6880 $\pm$ .0077 | .6636 $\pm$ .0101 | .6880 $\pm$ .0077 | .0000 | .0355 |
| .25 | .5161 $\pm$ .0054 | .5027 $\pm$ .0105 | .5813 $\pm$ .0064 | .1121 | .1352 |
| .5 | .3512 $\pm$ .0028 | .3652 $\pm$ .0030 | .4471 $\pm$ .0041 | .2145 | .1830 |
| .75 | .1771 $\pm$ .0019 | .2085 $\pm$ .0049 | .2481 $\pm$ .0035 | .2859 | .1598 |

The fixed standalone ranking has no headroom at zero redundancy. Its gap to
Oracle-Greedy grows from .112 at duplication ratio .25 to .286 at .75. MUR
exceeds Oracle-Static at the two higher redundancy levels, recovering part of
the state-conditioning headroom. At zero redundancy MUR remains below the
oracle because utility estimation and ranking are imperfect even when a fixed
order is sufficient. The artifact contains the full policy ladder, duplicate
selection rate, and state diagnostics for every seed:
`results/raw/R070_oracle_ladder_k16_b4_rgrid_s5_20260919_1540/`.

## Candidate-count scaling at fixed redundancy

The candidate-count diagnostic fixes $r_{\mathrm{dup}}=.5$ and $B=4$ and adds
the $K=64$ endpoint. Three seeds were used for each count with a shorter
screening protocol. Utility MAE remains small as $K$ grows, while top-1
accuracy falls and MUR-rollout one-step regret rises:

| $K$ | MUR recovery | Oracle-Static recovery | MUR rollout MAE | MUR rollout top-1 | MUR rollout regret |
|---:|---:|---:|---:|---:|---:|
| 4 | .963 | 1.000 | .1040 | .5660 | .0147 |
| 8 | .802 | .887 | .0779 | .2306 | .0631 |
| 16 | .698 | .779 | .0466 | .1811 | .0666 |
| 32 | .598 | .723 | .0268 | .1407 | .0605 |
| 64 | .478 | .694 | .0144 | .0943 | .0491 |

The pointwise error does not grow with the pool. The ranking statistics worsen,
which supports the max-over-candidates interpretation of the large-pool
degradation. These runs are a diagnostic screen rather than the final
confidence-interval table. Raw artifacts are under
`results/raw/R071_candidate_size_r050_b4_s3_20260919_1630/`.
Raw artifact: `results/raw/R023_diag_k32_20260919_110000/result.json`.

## MUR-Conservative calibration sweep

At fixed $K=16$, duplication ratio .25, harmful fraction .25, and budget 4,
the one-seed alpha sweep gives the expected gain--harm frontier:

| $\alpha$ | Gain | Negative-transfer rate | Mean selected contexts |
|---:|---:|---:|---:|
| .05 | .090 | .0004 | .306 |
| .10 | .203 | .0004 | 1.036 |
| .20 | .314 | .0048 | 2.317 |
| .30 | .349 | .0144 | 3.032 |
| .50 | .369 | .0176 | 3.620 |

MUR point estimate at the same setting has gain .376 and negative-transfer
rate .0188. This is a one-seed development sweep; the policy shape is useful
for the main experiment, but it is not a final statistical claim.

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

## MUR-Conservative confirmatory gain--harm frontier

R072 fixes $K=16$, $B=4$, $r_{\mathrm{dup}}=.5$ and evaluates five seeds at
two harmful-candidate fractions. MUR-Conservative uses the calibrated
residual quantile as a stopping threshold. The free-threshold control chooses
$\tau$ on calibration data to match the conservative policy's expected context
count. Results below are means over five seeds.

| Harmful fraction | $\alpha$ | MUR gain / harm | MUR-Conservative gain / harm / contexts | Free-$\tau$ gain / harm / contexts |
|---:|---:|---:|---:|---:|
| .25 | .05 | .273 / .0318 | .072 / .0001 / .308 | .078 / .0001 / .345 |
| .25 | .10 | .273 / .0318 | .152 / .0006 / .983 | .144 / .0006 / .887 |
| .25 | .20 | .273 / .0318 | .228 / .0092 / 2.203 | .227 / .0094 / 2.191 |
| .25 | .30 | .273 / .0318 | .252 / .0186 / 2.908 | .250 / .0170 / 2.831 |
| .25 | .50 | .273 / .0318 | .267 / .0281 / 3.569 | .268 / .0279 / 3.588 |
| .50 | .05 | .222 / .0854 | .046 / .0006 / .185 | .029 / .0002 / .101 |
| .50 | .10 | .222 / .0854 | .108 / .0021 / .627 | .116 / .0027 / .737 |
| .50 | .20 | .222 / .0854 | .177 / .0160 / 1.612 | .179 / .0155 / 1.632 |
| .50 | .30 | .222 / .0854 | .202 / .0389 / 2.321 | .203 / .0402 / 2.359 |
| .50 | .50 | .222 / .0854 | .216 / .0682 / 3.149 | .216 / .0685 / 3.152 |

The conservative policy moves along a continuous gain--harm frontier and
raises positive-selection precision at lower context counts. The free-$\tau$
control is close to the calibrated policy across both harmful settings. This
supports the restrained interpretation that calibration supplies a data-derived
stopping threshold; it does not define a new candidate ranking. Raw artifacts
and per-seed values are under
`results/raw/R072_mur_conservative_h025_h050_a005_a050_s5_20260920_0000/`.

## Electricity smoke — code and split audit

### Oracle audit

The three-seed oracle audit gives a small but repeatable Electricity
state-conditioning headroom. Mean gains are .0787 for Oracle-Static and .0849
for Oracle-Greedy, with
$H_{\mathrm{state}}=.0727\pm.0044$. The exact-delta MUR diagnostic reaches only
.0086 gain on average and has substantial negative transfer. This rules out a
simple representation fix based on exposing the frozen expert's exact
prediction change. Electricity is retained as a boundary case, not as positive
external evidence for the main routing claim.

Raw audit artifacts:

- `results/raw/R073_electricity_oracle_audit_s101_20260920_0030/result.json`
- `results/raw/R073_electricity_oracle_audit_s202_20260920_0040/result.json`
- `results/raw/R073_electricity_oracle_audit_s303_20260920_0040/result.json`

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

## Traffic oracle gate and router follow-up

The METR-LA gate uses a chronological split, a 12-step history, a 12-step
forecast horizon, a fixed $K=16$ candidate sensor pool, and budget $B=4$.
The frozen expert is trained on random context subsets and supports arbitrary
selected subsets at evaluation. Five seeds give:

| Policy | Mean gain | SD | Negative-transfer rate |
|---|---:|---:|---:|
| Oracle-Static | .1506 | .0054 | .1483 |
| Oracle-Greedy | .1692 | .0086 | .0000 |

The state-conditioning headroom is .1095 $\pm$ .0227. The gate passes: external
contexts have positive utility and the static oracle leaves a nonzero
set-conditioned gap.

The subsequent five-seed learned-router comparison does not reproduce the
oracle gap. MUR gain is -.0026 on average, compared with .0001 for matched
Static Utility. The pair-label diagnostic shows low test correlation between
observable context features and realized marginal utility. We therefore stop
before adding a new Traffic representation or training a second real task.
Traffic is a valid oracle gate and a clear learned-routing boundary under the
current frozen-expert protocol. In the saved diagnostic, test label
correlation is .074 for the static model and .027 for the set-conditioned
model, with test MAE .0355 and .0393 respectively.

Raw artifacts:

- `results/raw/R074_traffic_oracle_gate_metrla_k16_b4_s5_20260920_0100/`
- `results/raw/R075_traffic_full_router_metrla_k16_b4_s5_20260920_0115/`
- `results/raw/R075_traffic_label_diagnostic_s101_20260920_0200/`

## Traffic utility observability audit

R076 keeps the Traffic split, frozen expert, and candidate pool fixed and
raises probe capacity only for diagnosis. The five-seed replication confirms
that the cached context representation omits useful prediction-response
information:

| Probe inputs | MAE | Spearman | Positive AUC | Top-1 accuracy | One-step regret |
|---|---:|---:|---:|---:|---:|
| $\mathcal X_1$ cached context interactions | .0398 $\pm$ .0034 | .1637 $\pm$ .0118 | .6492 $\pm$ .0057 | .1480 $\pm$ .0199 | .0736 $\pm$ .0076 |
| $\mathcal X_2$ + expert response summaries | .0357 $\pm$ .0030 | .3905 $\pm$ .0128 | .8342 $\pm$ .0091 | .2595 $\pm$ .0451 | .0569 $\pm$ .0061 |
| $\mathcal X_3$ + full forecast trajectories | .0358 $\pm$ .0031 | .3725 $\pm$ .0252 | .8411 $\pm$ .0084 | .2446 $\pm$ .0159 | .0575 $\pm$ .0063 |

The exact-response summaries substantially improve utility predictability. Full
forecast trajectories add little. This is a diagnostic result, not a new
routing method. It identifies the current MUR limitation as lost
prediction-response information in cached representations rather than absence
of Traffic state-conditioning headroom.

Raw artifact:

- `results/raw/R076_traffic_utility_observability_s101_20260920_0215/`
- `results/raw/R076_traffic_utility_observability_s5_20260920_0300/`
