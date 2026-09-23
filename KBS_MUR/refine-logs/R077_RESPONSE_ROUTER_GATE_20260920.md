# R077 response-aware routing gate (METR-LA)

## Protocol
- Data: METR-LA, fixed chronological split from R074/R075.
- Frozen expert: same subset ridge expert as R075.
- Candidate pool: fixed 16-sensor pool.
- Budget: 4.
- Seeds: 101, 202, 303, 404, 505.
- Response model: MUR architecture with four compact response features:
  mean base forecast, mean forecast after adding the candidate, mean response
  delta, mean absolute response delta.
- Inference: cached MUR screens top-q candidates; the response-aware model
  reranks only those candidates.
- q values: 2, 4, 8, full candidate set.

## Aggregate gain over the anchor-only predictor

| Router | Gain | 95% paired CI over zero | Gain over learned Static Utility | Headroom recovery |
|---|---:|---|---:|---:|
| Static Utility | .0001 | -- | -- | -- |
| Cached MUR | -.0026 | -- | -- | -- |
| R-MUR q=2 | .0177 | [.0145, .0210] | .0176 | -7.52 |
| R-MUR q=4 | .0171 | [.0111, .0232] | .0170 | -7.55 |
| R-MUR q=8 | .0145 | [-.0033, .0323] | .0144 | -7.63 |
| R-MUR full | -.0008 | [-.0402, .0385] | -.0010 | -8.43 |
| Oracle-Static | .1506 | -- | -- | -- |
| Oracle-Greedy | .1692 | -- | -- | -- |

## Gate decision
- Positive gain over zero: PASS for q=2 and q=4.
- Gain over learned Static Utility: PASS for q=2, q=4, and q=8.
- Recovery of 20-30% of Oracle-Static to Oracle-Greedy headroom: FAIL.
  R-MUR improves learned routing substantially, but its gain remains far
  below the test-time Oracle-Static gain, so the relative headroom recovery is
  negative under the planned formula.

## Interpretation
- Compact response summaries convert to real routing gain relative to cached
  MUR and learned Static Utility.
- Top-q screening is necessary: response-aware reranking over the full
  candidate pool is unstable, consistent with the candidate-scaling margin
  compression in R071.
- The remaining gap is not explained by missing response information alone.
  Utility ranking and calibration relative to the oracle standalone scale
  remain the limiting factor.
- Following the planned gate order, the six-dataset benchmark expansion
  should not start from this version.

## Next method-level question
Before benchmark scaling, the required upgrade is a ranking-calibrated
response-aware utility objective, not more datasets. Candidate direction:
full-state responses and a utility-gap-weighted pairwise ranking loss, with
the frozen predictor unchanged.
