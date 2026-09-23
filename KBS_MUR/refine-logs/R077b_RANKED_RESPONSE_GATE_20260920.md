# R077b ranking-calibrated response-aware gate (METR-LA)

## Method
- R-MUR as in R077: cached MUR screens top-q candidates, response-aware model
  reranks the screened subset.
- Response-aware training uses full states with Huber regression plus a
  utility-gap-weighted pairwise ranking loss:
  `L = L_Huber + beta * sum_{i,j: m_i > m_j} |m_i - m_j| softplus(-(mhat_i - mhat_j))`.
- Frozen expert, split, candidate pool, budget, and seeds match R074/R075/R077.
- Response summaries: 4 compact expert-response features. A 7-feature variant
  was tested and performed similarly.

## Five-seed aggregate gain over anchor-only

| Router | Gain | 95% paired CI over zero | Gain over learned Static Utility | Ratio to Oracle-Greedy gain |
|---|---:|---:|---:|---:|
| Static Utility | .0001 | -- | -- | .001 |
| Cached MUR | -.0026 | -- | -- | -.016 |
| Ranked R-MUR q=2 | .0196 | [.0123, .0270] | .0195 | .116 |
| Ranked R-MUR q=4 | .0301 | [.0195, .0406] | .0299 | .178 |
| Ranked R-MUR q=8 | .0258 | [.0107, .0409] | .0256 | .152 |
| Ranked R-MUR full | .0194 | [.0078, .0310] | .0192 | .115 |
| Oracle-Static | .1506 | -- | -- | .890 |
| Oracle-Greedy | .1692 | -- | -- | 1.000 |

The 4-feature variant gave q=4 gain .0313 and ratio .185; the 7-feature variant
was not better. Ranking loss makes the full-pool response reranker stable
(gain .0194, CI positive), while q=4 remains the best deployment point.

## Gate decision
- Positive gain with paired CI away from zero: PASS for all q values.
- Gain over learned Static Utility: PASS for all q values.
- Gain over Cached MUR: PASS.
- Recovery of 20-30% of Oracle-Static to Oracle-Greedy headroom under the
  strict formula `(G_RMUR - G_OracleStatic) / (G_OracleGreedy - G_OracleStatic)`:
  FAIL. The deployed router remains below the test-time Oracle-Static upper
  bound.
- Utility recovery relative to Oracle-Greedy, `G_RMUR / G_OracleGreedy`:
  17.8% for q=4, near but below the 20% target.

## Decision
Do not start the six-dataset benchmark expansion from this gate. The response
objective is now stable and consistently positive, but the remaining gap to
Oracle-Static is an offline/online information gap, not a missing response
signal. The next method decision must address that gap directly or redefine the
gate in terms of a deployable reference rather than the test-time oracle.

## Hard-negative training attempt
A variant trained the response model only on the top-4 candidates returned by
the cached MUR screen, matching the inference distribution. Five-seed results:
q=4 gain .0247 [.0197, .0298] and gain over learned Static Utility .0246
[.0095, .0397]. This is worse than full-state ranking-calibrated training
(q=4 gain .0301). The final gate decision is unchanged.
