# Round 3 External Method Review

<details open>
<summary>Full raw review</summary>

| Dimension | Score | Rationale |
|---|---:|---|
| Problem Fidelity | 9.5 | Tightly anchored to set-conditioned selection under redundancy, harmful transfer, and budgeted stopping. |
| Method Specificity | 9.3 | Frozen expert, four partitions, provenance, MUR-light/full, calibration, state-shift trigger, and cost accounting are implementation-ready. |
| Contribution Quality | 9.0 | The contribution is sharp: learning nonempty-state counterfactual end-task marginal utility for calibrated sequential routing. |
| Frontier Leverage | 8.7 | Modern calibrated decision intervals and test-time routing are used without unnecessary modules. |
| Feasibility | 8.7 | Feasible for planning readiness; empirical signal and label cost are controlled by gates. |
| Validation Focus | 9.2 | Gates A/B/C/D/N are sufficient and claim-driven. |
| Venue Readiness | 9.0 | Ready as a method and experiment plan, with claims conditional on gates. |

**Weighted overall score: 9.10 / 10. Verdict: READY.**

All five prior protocol ambiguities are resolved: contextual-submodular distinction, one frozen expert, deployed-state trigger, MUR-light-only deployability, and Gate N baseline set.

Editorial actions incorporated before finalization:

1. Rename the gate section to five gates.
2. Separate offline before/after label generation from embedding-only MUR-light inference in Figure 2.
3. Include Gate N in the first-week target.
4. Recalibrate after any triggered train-only roll-in before test evaluation.

Simplification remains binding: ranking loss stays an ablation, MUR-full stays outside the main claim, and Traffic/Activity wait for the gates.

**Drift Warning: NONE.**

**Verdict: READY for method implementation and experiment planning.**

</details>

