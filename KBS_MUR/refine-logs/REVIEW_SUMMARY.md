# Review Summary

## Outcome

- Final method-review score: **9.10/10**
- Final verdict: **READY for method implementation and experiment planning**
- Review rounds: 3
- Drift warning: none

## Score progression

| Round | Overall | Verdict | Main change requested |
|---|---:|---|---|
| 1 | 7.20 | REVISE | Protect novelty and make counterfactual cost/leakage auditable |
| 2 | 8.14 | REVISE | Fix expert protocol and deployed-state calibration trigger |
| 3 | 9.10 | READY | Editorial cleanup only |

## Stabilized thesis

MUR learns the counterfactual end-task loss marginal of a candidate at a
nonempty selected-set state and uses calibrated intervals for sequential
selection and stopping. MUR-light performs embedding-only inference; MUR-full
is a high-cost diagnostic.

## Binding review decisions

1. Hashimoto-style incremental utility is the `A=empty` closest baseline.
2. Coverage/MMR and contextual submodular policy learning are acknowledged as
   direct neighboring families, not dismissed.
3. The primary prediction expert is trained once on `expert-train`, frozen,
   and never refit after router-label generation.
4. Only MUR-light can support deployability claims.
5. Split-calibrated intervals are conditional on policy-state coverage; a
   pre-test validation trigger permits one train-only roll-in augmentation.
6. Ranking loss, Set Transformer, RL, uncertainty heads, and beam search are
   outside the primary method.
7. Traffic and Activity are expansion tasks after Gates A/B/C/D/N pass.

## Remaining empirical risk

The plan is method-ready, not result-confirmed. The paper proceeds only if
MUR-light beats `A=empty` utility and coverage baselines under redundancy,
harm, and full cost accounting.

