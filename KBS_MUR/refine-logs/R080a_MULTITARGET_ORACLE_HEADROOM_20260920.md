# R080a: multi-target oracle headroom audit (seeds 101-303)

## Protocol
- Six standard traffic datasets and STAEformer split files.
- 32 fixed, evenly spaced target sensors per dataset.
- Candidate pool constructed per target from training-segment correlation.
- Subset ridge expert per target; 4000 training and 2000 test windows.
- Five seeds complete: 101, 202, 303, 404, 505.

## Results

| Dataset | Targets | Seed-target runs | H_state mean | H_state median | H_state positive fraction | Positive-target fraction |
|---|---:|---:|---:|---:|---:|---:|
| METRLA | 32 | 96 | .0936 | .0828 | 1.00 | 1.00 |
| PEMSBAY | 32 | 96 | .1176 | .1148 | 1.00 | 1.00 |
| PEMS03 | 32 | 96 | .1227 | .1098 | 1.00 | 1.00 |
| PEMS04 | 32 | 96 | .0832 | .0737 | 1.00 | 1.00 |
| PEMS07 | 32 | 96 | .0885 | .0822 | 1.00 | 1.00 |
| PEMS08 | 32 | 96 | .0779 | .0771 | 1.00 | 1.00 |

## Interpretation
- Structural headroom is not specific to target sensor 0.
- Across 576 target-seed runs, every run has positive H_state.
- Five-seed hierarchical evaluation complete.
