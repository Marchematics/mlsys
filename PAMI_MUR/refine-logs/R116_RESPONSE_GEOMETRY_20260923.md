# R116 — Response-geometry policies: the "least disturbance" hypothesis is false

Run date: 2026-09-21/23. Raw run: `results/raw/R116_response_geometry_metrla_<stamp>/`.
Cells: the 48 METR-LA gate cells (16 targets x 3 seeds, 2000 test episodes each),
expert adapted exactly as in the gate run. Paired comparisons use the stored
episode gains of `R103_strong_backbone_gate_20260921_0350`.

## Hypothesis (theory-derived, stated before the run)

The exact identity gives `m(j|A) = <g_A(x), d_{A,j}> - ||d_{A,j}||^2 - lambda c_j`.
If the first-order term were not identifiable from observable features (which
the observability probe measures), the only systematic term left is the
curvature penalty, so the predicted-best contexts would be those that disturb
the frozen predictor *least*: rank by `||d||^2` ascending. This is
counter-intuitive (it ignores both relevance and predicted utility) and would
have been a strong practical prescription.

## Result: falsified

| policy | mean gain | paired difference vs Standalone Utility |
|---|---|---|
| Standalone Utility (frozen reference) | +.0068 | --- |
| **min disturbance, empty state** | **+.0022** | **-.0047 [-.0087, -.0007]** |
| min disturbance, greedy | +.0019 | -.0049 [-.0091, -.0008] |
| pool tail (least correlated four) | -.0001 | -.0069 [-.0116, -.0023] |
| max disturbance | -.0059 | -.0127 [-.0196, -.0058] |
| R-MUR q=4 (frozen) | +.0097 | +.0028 [-.0012, +.0083] |
| standalone oracle | +.0904 | |
| random four | +.0061 | |

Least-disturbance selection is significantly *worse* than a learned standalone
utility model, and its greedy variant is no better. The maximum-disturbance
control is worst, as the curvature term predicts, but the ordering between them
is not the practical rule.

## What this tells us

The falsification is informative: it means the first-order term `<g_A, d>` is
*partially identifiable*, so discarding it loses real signal. This matches the
direct measurement that the observable interaction `<f_A, d>` has a
within-episode rank correlation of up to .5 with the true marginal on
METR-LA (section 4 of the strong-backbone report). The design conclusion is the
opposite of the hypothesis and stronger than it: response *direction* matters,
so the head should estimate the interaction term explicitly and treat the
curvature exactly --- which is what the bilinear router does
(`experiments/bilinear_router.py`), not what a rank-by-magnitude heuristic does.

Recorded as a negative result; the hypothesis is not used in the manuscript
beyond a sentence in the analysis (the theory's curvature term is a penalty,
not a selection rule).
