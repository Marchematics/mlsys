# PAMI strengthening plan (execution record)

Source of requirements: user PAMI acceptance table (2026-09-21) plus
`PAMI_MUR/paper/PAMI_EXTENSION_PLAN.md`. Four additions are required; nothing
else is added unless a gate fails.

## Fixed (do not change)
- Core object `m(j | A) = F(A ∪ {j}) - F(A)`.
- Method R-MUR: cached shortlist -> compact response summaries ->
  gap-weighted ranking-calibrated reranking.
- Traffic evidence: six benchmarks, 32 fixed targets, multi-target oracle and
  learned results.

## Additions and gates

### A1. Strong-backbone confirmation
Subset-capable nonlinear predictor (spatio-temporal transformer trained with
subset dropout so that *arbitrary* subsets are in-distribution; not a
full-input model masked at test time).

- Expert must be verifiably stronger than the subset ridge expert on the same
  protocol (report MAE/MSE of both, normalized).
- Datasets: METR-LA and PEMS-BAY primary; PEMS03 if the first two pass.
- Targets: 32 fixed per dataset (first 8 also reported separately).
- Seeds: 3.
- Gate G1: `G_RMUR(q=4) > G_Static` and `G_RMUR(q=4) > G_Cached` under paired
  seed/target-level CI (95%) on each dataset.

### A2. Second task family (non-traffic)
Demonstration (context-example) selection for a frozen in-context classifier.
Candidate contexts = labelled examples; fixed predictor = frozen in-context
transformer over the selected examples; utility = query-batch loss reduction.

- Benchmarks: at least 3 public image-classification benchmarks
  (CIFAR-10, CIFAR-100, SVHN, EuroSAT, DTD target set).
- Reuse R-MUR unchanged (same pair sampling, same response features, same
  ranking-calibrated reranker).
- Report task-level (benchmark) and within-task target-level (class)
  consistency with hierarchical CIs.
- Gate G2: R-MUR q=4 positive mean gain and above Static Utility on a
  majority of benchmarks, with task-level CI reported honestly (negative
  results are reported, not hidden).

### A3. General smooth-loss analysis
Derive the response-utility relation for twice-differentiable losses,
including smoothness conditions, two-sided bounds, and recovery of the
existing squared-loss identity as a special case. Numerical validation of the
bound on the second family (cross-entropy) and on the traffic expert
(squared loss, exact).

### A4. Compute / latency / memory evidence
On the strong backbone: expert-call count, wall-clock latency, peak GPU
memory for q ∈ {2, 4, 8, 16, full}; accuracy-vs-compute curve. Gate: the
response-aware stage must retain most of its gain at q = 4 while using a small
fraction of full-pool response evaluations.

### A5. Nearest-neighbour baselines
Add standard subset/context-selection competitors that are not just relevance
or MMR: diversity (determinantal / facility-location submodular greedy),
greedy mutual-information, and one learned set-scoring selector. Evaluate on
both families.

## Stop rule
No third task family, no extra backbones, no extra datasets beyond A1/A2,
unless a gate result overturns a main claim.

## Provenance
Raw runs under `PAMI_MUR/results/raw/<run_id>_<timestamp>/`, derived summaries
under `PAMI_MUR/results/derived/<run_id>/`, reports under
`PAMI_MUR/refine-logs/`. Raw runs are immutable; derived files list sources.
