# Round 2 External Method Review

<details open>
<summary>Full raw review</summary>

| Dimension | Score | Rationale |
|---|---:|---|
| Problem Fidelity | 9.0 | The revised proposal preserves finite-budget multi-context selection under redundancy, harmful transfer, and stopping, with a clear ICLR boundary. |
| Method Specificity | 8.5 | MUR-light/full, feature sets, cross-fitted labels, calibration, state sampling, inference, and cost accounting are implementable. Fold construction and on-policy state control need one more decision. |
| Contribution Quality | 8.0 | The novelty contract is much stronger: nonempty-state counterfactual loss marginal plus calibrated sequential select/stop. Its empirical necessity remains to be shown. |
| Frontier Leverage | 7.8 | Split-calibrated utility intervals and test-time context routing are appropriate modern primitives without forced LLM/RL additions. |
| Feasibility | 7.4 | MUR-light fixes deployability, but may underperform without exact delta; cross-fitted labels could exceed budget on expansion datasets. |
| Validation Focus | 8.5 | Gate N and the baseline ladder directly target novelty. |
| Venue Readiness | 7.6 | Credible and focused, but the central empirical regime is not yet proven. |

**Weighted overall score: 8.14 / 10. Verdict: REVISE.**

## Previous Critical Issues

- Novelty: mostly resolved. Add the precise distinction that MUR learns the reward/marginal from frozen-predictor counterfactuals rather than learning a policy for a known structured reward.
- Deployability: resolved enough. MUR-light is main; MUR-full is high-cost diagnostic.
- Leakage: mostly resolved. Pick whether the final expert is refit after label caching.
- Stopping: resolved. Avoid universal conformal guarantee language under policy-induced state shift.
- State shift: partially resolved. Add a quantitative validation trigger for one on-policy augmentation pass.
- Cost accounting: resolved at protocol level.

## Required Actions

1. Clarify contextual submodular prediction boundary.
2. Fix one final expert protocol.
3. Add deployed-state calibration thresholds that trigger one roll-in pass.
4. Restrict deployable claims to MUR-light.
5. Put Hashimoto incremental, Static Utility, Set-Coverage/MMR, and MUR-light in the novelty stress test.

## Simplification

Keep ranking loss out unless justified; merge overlapping static/incremental definitions carefully; keep Traffic/Activity outside must-run gates.

## Modernization

Keep calibrated intervals and report coverage by state source. Do not add LLM/RL/uncertainty heads.

## Drift Warning

NONE.

## Verdict

REVISE. The proposal is ready for gates after one final protocol tightening, though the central empirical bet remains unproven.

</details>

