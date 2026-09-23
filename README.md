# mlsys — identifiable marginal utility for budgeted context selection

Code, experimental records and manuscripts for a study of **set-conditioned
marginal utility**: how much does adding one candidate context reduce a fixed
predictor's loss, and when can that quantity be estimated at all?

The short answer the evidence supports: context selection pays exactly when the
predictor leaves a *systematic residual* a selector can observe. Stronger
predictors leave less; adapting a predictor to its deployment is the operation
that destroys the value; enlarging the candidate library raises the opportunity
while lowering the share of it that can be realised.

## Contents

| path | what |
|---|---|
| `PAMI_MUR/paper/main.pdf` | method, theory, generalisation evidence, validated deployment procedure (20 pp) |
| `TKDE_MUR/paper/main.pdf` | identifiability: why response decides, when utility is estimable (17 pp) |
| `KBS_MUR/paper/main.pdf` | companion method paper (46 pp) |
| `PAMI_MUR/experiments/` | every experiment, analysis and checker |
| `PAMI_MUR/results/derived/` | the derived artifact behind every headline number |
| `PAMI_MUR/results/raw/` | immutable raw runs: metrics, per-episode gains, diagnostic traces |
| `PAMI_MUR/refine-logs/` | one record per experiment, the SOTA survey, the audits |
| `KBS_MUR/results/raw/` | the six-dataset learned runs and every run the artifacts cite |
| `DATA.md` | what is included, what is excluded, and how to regenerate it |

## Verify the claims rather than trusting them

Two scripts re-parse the manuscripts and compare every headline number against
the artifact it must have come from — including checks that the stated
*limitations* are themselves true of the data.

```bash
cd PAMI_MUR
python experiments/check_table_artifacts.py   # 13 suites, ~124 assertions
python experiments/check_cross_paper.py       # shared numbers must not drift between papers
python -m pytest tests/ -q                    # 78 tests
cd ../KBS_MUR && python -m pytest tests/ -q   # 17 tests
```

All of the above pass on this snapshot.

## Main findings

**Where selection works.** With a linear subset expert on six traffic
benchmarks, learned marginal-utility routing is significantly ahead of the best
of fourteen competitors on three benchmarks, unresolved on two, and below simple
pooling on one. The strongest competitor is always the trivial policy of using
every candidate, never one of the specialised heuristics.

**Where it stops.** With a subset-capable nonlinear predictor nothing separates
from pooling, and five theory-guided expectations were tested and failed, each
with a measured reason:

| finding | record |
|---|---|
| a predictability probe cannot gate deployment | `R140` |
| a larger library raises the opportunity 40% and drives the router to chance | `R142` |
| the cached screen is a symptom, not the cause | `R145` |
| the theory's own least-perturbation limit case loses to pooling | `R151` |
| structural headroom does not predict routing benefit in either regime | `R153` |
| the gate headroom is 3–7× the router's advantage and every decision-time signal is at chance, AUC ∈ [.481,.521] | `R159` |

**What survives is a procedure.** The measurement protocol is validated as a
deployment decision: a pilot of 24 targets reproduces the full-deployment
verdict 96% of the time, splitting the pilot in half certifies a verdict at 99%
agreement, and the direction of the effect survives a time shift (r = +.596)
while the significance call partly does not.

## Reproducing

Datasets and cached encoder features are not committed — see `DATA.md`. With the
STAEformer-format data in place, each record in `PAMI_MUR/refine-logs/` names the
command that produced it and the artifact it wrote.
