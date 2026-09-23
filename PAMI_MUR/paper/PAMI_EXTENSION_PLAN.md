# PAMI-level extension plan

## Keep fixed
- Core object: `m(j | A)`.
- Core method: R-MUR with cached shortlist, compact response summaries, and
  gap-weighted ranking calibration.
- Core evidence: controlled mechanism, six traffic benchmarks, multi-target
  oracle and learned results.

## Required additions before a PAMI submission
1. **Strong-backbone confirmation**
   - subset-capable nonlinear predictor;
   - METR-LA + PEMS-BAY, fixed 8-32 targets, 3 seeds;
   - require R-MUR > Static and R-MUR > Cached.

2. **Second task family**
   - 3-5 public benchmarks with candidate contexts, subset selection, and a
     fixed predictor;
   - reuse R-MUR without changing the method;
   - report task-level and within-task target-level consistency.

3. **General-loss analysis**
   - derive a response-based relation for smooth losses:
     `m(j | A) = -<grad loss(f_A), d_{A,j}> + O(||d_{A,j}||^2) - cost`;
   - state smoothness conditions;
   - keep the existing squared-loss identity as a special case.

4. **Compute evidence**
   - expert-call count, latency, and memory for q = 2, 4, 8, full;
   - show that response-aware reranking retains most of its gain at a small
     fraction of candidate evaluations.

## Stop rule
Do not add a third task family, additional backbones, or extra datasets after
these four additions unless a new result overturns a main claim.
