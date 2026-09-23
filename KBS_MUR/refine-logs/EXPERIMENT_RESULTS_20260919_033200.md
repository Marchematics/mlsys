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

