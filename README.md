# mlsys — response-aware context selection and identifiable marginal utility

This repository contains one submitted companion method manuscript and one
active journal manuscript.

The active line is now **TKDE_MUR**. All advantageous material that is not part
of the submitted IPM companion has been consolidated into the TKDE manuscript:
general response--utility theory, theory-shaped scoring heads, nonlinear
backbone regimes, pool-size and adaptation sweeps, observability diagnostics,
the second task family, mechanism-derived adaptation, deployment pilots, and
compute evidence. **PAMI_MUR is no longer a submission candidate**; it is kept
only as the provenance/evidence source from which those additions were
developed.

## Submission status

| path | status | role |
|---|---|---|
| `KBS_MUR/paper/` | submitted companion (IPM); frozen | basic R-MUR method and the submitted multi-target traffic evidence |
| `TKDE_MUR/paper/` | **active primary manuscript** | method + general theory + identifiability mechanism + adaptive/deployment evidence |
| `PAMI_MUR/` | **archived; no PAMI submission** | raw/derived artifacts, experiment logs, proofs and material now consolidated into TKDE |

The submitted companion is treated as frozen. The TKDE manuscript must not
depend on copying its main numerical tables. Shared notation and the basic
routing primitive are disclosed as common background; the TKDE contribution is
the new theory, new estimator structure, new regimes, new diagnostics, new
adaptation rule and new deployment evidence.

## Evidence base

The authoritative experimental record for the active TKDE line lives under
`PAMI_MUR/results/{raw,derived}` and `PAMI_MUR/refine-logs/`. Important
records include:

- general smooth-loss, Bregman and cross-entropy response--utility theory;
- strong nonlinear subset-capable backbones and the expert-strength ladder;
- recent-baseline matrices and theory-shaped response heads;
- pool-geometry, harmful-pool, budget and library-size sweeps;
- response/observability probes and the gate-ceiling audit;
- the CLIP demonstration-selection task family;
- novelty-based label-free adaptation;
- measured latency, expert-row and memory costs;
- pilot-power, time-shift and split-half deployment validation.

The old `PAMI_MUR/paper/` manuscript is provenance only. New prose and new
claims belong in `TKDE_MUR/paper/`.

## Verify claims rather than trusting them

The project keeps mechanical claim checks for the shared evidence base:

```bash
cd PAMI_MUR
python experiments/check_table_artifacts.py
python experiments/check_cross_paper.py
python -m pytest tests/ -q
cd ../KBS_MUR && python -m pytest tests/ -q
```

The TKDE paper should use only numbers that can be traced to the immutable
artifacts or to the frozen submitted companion where the overlap is explicitly
disclosed.

## TKDE scientific story

The active manuscript is organised around one positive-to-mechanistic chain:

1. **State-conditioned marginal utility** is the correct decision object.
2. **Predictor response** is the observable that carries that utility signal.
3. **General-loss theory** gives the response expansion, an exact Bregman
   identity, explicit softmax-cross-entropy constants and a response-sufficiency
   result.
4. **When the signal is identifiable**, response-aware selection strongly beats
   a broad selector toolbox.
5. **When a stronger predictor absorbs the systematic residual**, structural
   oracle opportunity can remain while observable ranking signal collapses.
6. **Pool geometry and novelty** explain when retrieval-style relevance helps or
   hurts and yield a label-free adaptation rule.
7. **A second task family, scaling sweeps and deployment pilots** test whether
   the mechanism transfers and how to act on it in practice.

The intended TKDE identity is therefore **strong method + strong theory + strong
mechanism + actionable adaptation**, not a negative-results or boundary-only
paper.
