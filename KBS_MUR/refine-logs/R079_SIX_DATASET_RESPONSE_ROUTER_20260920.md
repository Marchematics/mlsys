# R079: six-dataset R-MUR benchmark

## Protocol
- Datasets: METRLA, PEMSBAY, PEMS03, PEMS04, PEMS07, PEMS08.
- Standard STAEformer data split files (`data.npz`, `index.npz`).
- Frozen expert: subset-capable ridge expert, identical across datasets.
  - target sensor index 0;
  - 16-candidate pool from training-segment correlation;
  - history length 12, horizon 12;
  - 4000 training windows, 1000 validation windows, 2000 test windows;
  - three expert repeats, ridge penalty 10.
- Router: cached MUR screening Top-4, ranking-calibrated response-aware
  reranking.
- Seeds: 101, 202, 303, 404, 505.

## Results

| Dataset | H_state | Static Utility | Cached MUR | R-MUR q=4 | R-MUR q=4 95% CI | R_oracle | R_deploy |
|---|---:|---:|---:|---:|---:|---:|---:|
| METRLA | .112 | -.0005 | .0178 | .0362 | [.0236, .0489] | .201 | .204 |
| PEMSBAY | .154 | .0321 | .0320 | .0550 | [.0388, .0712] | .334 | .173 |
| PEMS03 | .265 | .0203 | .0191 | .0316 | [.0261, .0370] | .491 | .256 |
| PEMS04 | .041 | .0397 | .0384 | .0530 | [.0501, .0558] | .607 | .279 |
| PEMS07 | .112 | .0147 | .0175 | .0405 | [.0344, .0466] | .394 | .293 |
| PEMS08 | .114 | .0284 | .0273 | .0371 | [.0349, .0394] | .587 | .251 |

Definitions:
- `H_state = (G_OG - G_OS) / G_OG`.
- `R_oracle = G_RMUR_q4 / G_OG`.
- `R_deploy = (G_RMUR_q4 - G_Static) / (G_OG - G_Static)`.

## Consistency
- R-MUR q=4 gain positive on 6/6 datasets; seed-level 95% CI above zero on
  6/6.
- R-MUR q=4 exceeds learned Static Utility on 6/6 datasets; paired
  seed-level 95% CI above zero on 6/6.
- All six datasets show nonzero `H_state` from .041 to .265.
- `R_oracle` is .201 to .607.
- q=4 uses 25% of candidate response evaluations.

## Boundary
This run uses a unified subset ridge expert, not STAEformer. It establishes
breadth and consistency for the routing method under a fixed subset-capable
frozen predictor. The STAEformer-based replication requires a subset-capable
interface for a global traffic predictor and remains the next backbone step.
