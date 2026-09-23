# Pipeline Summary

**Problem**：有限预算下，从候选知识池顺序选择仍能相对于当前 selected set 降低 end-task loss 的 contexts。

**Final Method Thesis**：MUR-light 学习非空状态下的 counterfactual marginal predictive utility，并用校准区间完成 embedding-only sequential select/stop。

**Final Verdict**：READY FOR GATES

**Date**：2026-09-19

## Final Deliverables

- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Paper blueprint: `paper/PAPER_BLUEPRINT.md`
- Pilot results: `refine-logs/EXPERIMENT_RESULTS.md`

## Contribution Snapshot

- **Dominant contribution**：nonempty-state counterfactual end-task marginal value interface for sequential context routing。
- **Supporting contribution**：per-state decision bound and calibrated select/stop certificates。
- **Explicitly rejected complexity**：Set Transformer、RL、beam search、uncertainty head、joint expert-router training。

## Must-Prove Claims

- C1：set-conditioning is necessary under redundancy after controlling for relevance, standalone incremental utility, and coverage.
- C2：calibrated stopping improves the gain–harm/cost frontier under harmful candidates and larger pools.

## First Runs to Launch

1. R002：synthetic oracle-headroom smoke。
2. R003：label-cache/split/target-access audit。
3. R010–R012：Gate A, Static Utility versus Relevance across three seeds。

## Main Risks

- MUR-light lacks enough information without exact expert delta.
- Utility labels are too noisy or expensive.
- Coverage/Hashimoto baselines close the purported novelty gap.
- Learned-policy states break calibration exchangeability.

## Current Evidence

- M0 unit and oracle-headroom checks pass.
- Three-seed relation-positive pilots show MUR-light utility recovery above
  Static Utility and Coverage at medium and high redundancy.
- Three-seed harmful-candidate pilots show calibrated interval routing reduces
  negative transfer to near zero while using fewer contexts; the point router
  still needs a harm-aware policy comparison.
- These are pilot results. Electricity and formal low-redundancy controls are
- now have a leakage-audited smoke, but the learned router is inconclusive:
  the planned MUR-light directional claim does not pass. No Electricity
  confirmatory run should start until the feature/label design is revised.

## Writing Boundary

The active paper uses a single-paragraph, data-free abstract. Internal run
labels, implementation split names, hardware names, version strings, and
unexplained abbreviations remain in experiment records only.

## Next Action

Implement the controlled generator and execute M0, then Gate A. Do not start
Traffic or Activity before A/B/C/D/N pass.
