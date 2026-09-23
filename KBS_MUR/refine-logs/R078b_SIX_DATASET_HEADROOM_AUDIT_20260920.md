# R078b: six-dataset oracle headroom audit

## Protocol
- Datasets: METRLA, PEMSBAY, PEMS03, PEMS04, PEMS07, PEMS08.
- Source: standard STAEformer data splits (`data.npz`, `index.npz`).
- Target sensor: index 0.
- Candidate pool: 16 sensors constructed from training-segment correlation
  (8 high, 5 middle, 3 fixed distractors).
- Frozen expert: subset-capable ridge expert fitted on 4000 training windows,
  three repeats, ridge penalty 10.
- Test episodes: 2000 per seed.
- Seeds: 101, 202, 303, 404, 505.
- Metrics: `G_OS`, `G_OG`, and `H_state = (G_OG - G_OS) / G_OG`.

## Results

| Dataset | Oracle-Static gain | Oracle-Greedy gain | H_state |
|---|---:|---:|---:|
| METRLA | .1606 | .1810 | .1128 |
| PEMSBAY | .1429 | .1690 | .1565 |
| PEMS03 | .0452 | .0626 | .2774 |
| PEMS04 | .0835 | .0872 | .0419 |
| PEMS07 | .0947 | .1059 | .1056 |
| PEMS08 | .0549 | .0619 | .1135 |

All six datasets show nonzero structural headroom; PEMS03 has the largest
relative state-conditioning gap, while PEMS04 has the smallest.

## Interpretation
- The controlled redundancy phenomenon extends to a standard six-traffic
  benchmark family.
- The audit is an oracle diagnostic; it does not constitute a learned-router
  result.
- The subset ridge expert provides a unified, cheap protocol for this audit.
  Replacing it with a full STAEformer-based subset interface is the next
  backbone-based replication step.
