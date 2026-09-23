# R103 — Strong-backbone confirmation of R-MUR (PAMI addition A1 + A4)

Run date: 2026-09-21. Expert pretraining: `results/raw/R101_expert_pretrain_20260921_0237`.
Gate run: `results/raw/R103_strong_backbone_gate_20260921_0350`.
All numbers below are copied from the derived files named next to them.

## 1. What was built

* **Subset-capable nonlinear experts** (`experiments/neural_expert.py`):
  * `SubsetSTTransformer` — per-node history embedding + learned node embedding +
    time-of-day/day-of-week embeddings, a 3-layer token transformer over the
    anchor token and the **selected** candidate tokens (key-padding mask), an
    explicit linear path from the anchor history, 499,722 parameters.
  * `SubsetDeepSets` — permutation-invariant masked-mean MLP, second backbone.
  * Trained with **subset dropout** (sizes 0..4 dominant, 15% longer, 5% full,
    5% empty) so arbitrary subsets are in distribution. It is not a
    full-context model masked at test time: unselected candidates are provably
    inert (`tests/test_neural_expert.py::test_unselected_candidates_do_not_change_prediction`).
* **Unit tests** (13 in `tests/test_neural_expert.py`): shapes, empty subset,
  inertness of unselected candidates, permutation invariance, marginal utility
  equals brute force, response features satisfy the squared-loss identity,
  batched pair evaluation equals full-mask evaluation.
* **Pretraining**: METR-LA and PEMS-BAY, 3 seeds each, 200 windows/node,
  40 epochs, early stop on a mixture of mask families. 12 checkpoints
  (6 transformer + 6 DeepSets).
* **Protocol parity**: the expert is adapted per target on exactly the 4000
  training episodes the ridge expert sees, with the epoch chosen on the
  calibration split; candidate pools, normalisation and correlations come from
  the training split only. Router training, pair sampling, response features
  and the ranking-calibrated reranker are unchanged.
* **Fast but equivalent router training** (`experiments/fast_router.py`):
  profiling showed the frozen DataLoader loops consume 70% of a cell's CPU
  time. The fast loops use the same seed, same uniform-random minibatches,
  same loss/optimizer/epochs, and move the dataset to the device once.
  `tests/test_fast_router.py` verifies (a) the fast pair sampler reproduces the
  frozen sampler **bit-identically** and (b) both trainers reach the same
  objective. They do **not** replay the DataLoader's RNG stream (PyTorch draws
  an undocumented `_base_seed` per iterator), so models are equivalent in
  distribution, not bit-identical.

## 2. The nonlinear expert is genuinely stronger than the ridge expert

Same training episodes, same held-out windows, same mask families
(`results/derived/R104_expert_strength_*`, PEMS08 from the interactive check):

| Dataset | Mask | Ridge MSE | Nonlinear MSE | ratio |
|---|---|---|---|---|
| PEMS08 | empty | 0.1005 | 0.0475 | 0.47 |
| PEMS08 | four | 0.1082 | 0.0450 | 0.42 |
| PEMS08 | full | 0.0645 | 0.0437 | 0.68 |
| METR-LA | empty | 0.471 | 0.445 | 0.94 |
| METR-LA | four | 0.417 | 0.437 | 1.05 |
| METR-LA | full | 0.403 | 0.499 | 1.24 |

The nonlinear expert is uniformly better on PEMS08 and better on the anchor-only
and four-context families on METR-LA. This is the intended stress test: a better
anchor-only predictor leaves less systematic residual for auxiliary contexts to
correct.

## 3. Gate result, METR-LA and PEMS-BAY (16 targets x 3 seeds = 48 cells each, K=16, B=4)

Source: `results/raw/R103_strong_backbone_gate_20260921_0350/METRLA/target_statistics.json`.

| policy | METR-LA | PEMS-BAY |
|---|---|---|
| anchor only | .0000 | .0000 |
| random four | .0061 | .0161 |
| relevance (correlation) | .0070 | .0107 |
| MMR | .0076 | .0113 |
| diversity (DPP) | .0091 | .0142 |
| facility location | .0040 | .0113 |
| pool all 16 | .0133 | .0224 |
| Standalone Utility | .0068 | .0138 |
| cached-only router | .0076 | .0133 |
| **R-MUR q=4** | **.0097** | **.0130** |
| **R-MUR q=8** | **.0108** | **.0141** |
| standalone oracle | .0904 | .0880 |
| sequential oracle | .1051 | .1037 |

PEMS-BAY gate: ΔStatic -.0008 [-.0036, +.0019], ΔCached -.0003 [-.0024, +.0019],
positive on 15/16 targets, above Static on 5/16.

* **Gate G1 is not passed on either dataset.** METR-LA: R-MUR q=4 positive on
  14/16 targets, above Standalone Utility on 9/16, ΔStatic +.0028
  [-.0012, +.0083], ΔCached +.0021 [-.0006, +.0053]. PEMS-BAY: positive on
  15/16, above Static on 5/16, ΔStatic -.0008 [-.0036, +.0019], ΔCached -.0003
  [-.0024, +.0019]. On PEMS-BAY every learned router is also below plain random
  selection (.0161).
* Structural headroom is preserved: per-target H_state ranges .075–.232
  (mean ≈ .15), and the sequential oracle beats the standalone oracle by .0148
  absolute.
* The learned response-free routers (.0068, .0076) are statistically
  indistinguishable from random selection (.0061), and R-MUR q=8 > q=4, i.e.
  the response stage improves as the shortlist grows: the **cached screen**,
  not the response stage, is the limiting component with a strong expert.
* Compute (A4, expert evaluations per episode, 2000 test episodes):
  q=4 → 20, q=8 → 36, full-pool reranking → 68, standalone oracle → 17.
  The shortlist keeps its intended efficiency.

## 4. Why: the cached view stops being informative

Diagnostics (`results/derived/R105_screen_diagnosis`, `R106_candidate_identity`):

* Cached-screen recall of the oracle-best candidate: **q=2: .13–.15, q=4:
  .26–.29, q=8: .49–.53** — i.e. exactly chance at q=4 and q=8. The screen's
  correlation with the true marginal is ≈ 0 (+.005, −.049, −.019 at the first
  three greedy steps).
* Variance decomposition of the standalone marginal: between-candidate
  variance is **0.03–0.13%** of the total; almost all variation is
  episode-specific, i.e. the utility is genuinely state-conditioned but the
  per-candidate average carries almost no information.
* The per-candidate mean is stable across splits for the nonlinear expert
  (r = .86/.77) but explains almost none of the variance, so candidate identity
  is not the missing ingredient.
* Consequence: any policy that ranks by a response-free score performs at the
  level of a constant prior. This matches the gate table exactly (Static ≈
  Random).

This is consistent with the general-loss analysis (PAMI addition A3): with
`g_A(x) = 2(μ(x) − f_A(x))`, the conditional expectation of the exact identity
is `⟨g_A(x), d_{A,j}⟩ − ‖d_{A,j}‖² − λc_j`. The second term is observable once
the predictor has been probed; the first requires predicting the residual of a
*predictor that is already strong*. The stronger the expert, the closer that
residual is to noise, so the ex-ante identifiable value shrinks even though the
oracle value persists.

### 4.1 Observability probe (closed form, `results/derived/R107_observability*`)

Six (dataset, target) cells (METR-LA t0/68/151, PEMS-BAY t0/124/260), 1500
training and 800 test episodes each, expert adapted exactly as in the gate run,
two feature sets per cell (state only; state + full response vector + response
magnitude + observable `⟨f_A, d⟩` interaction):

| quantity | mean | range |
|---|---|---|
| out-of-sample R² | **-.093** | [-.630, +.025] |
| within-episode rank correlation with the true marginal | **+.001** | [-.098, +.119] |
| in-sample R² (same pipeline, train data) | +.070 | [+.012, +.174] |
| label-permutation control R² | -.005 | [-.017, +.010] |
| standalone-oracle gain in these cells | +.19 to +.32 | |
| expected gain of a random four | -.03 to -.004 | |

Eleven of twelve probes have |within-episode ρ| ≤ .12. The pipeline is valid
(in-sample fit positive, permutation control at zero) but does not generalise:
the observable state carries the *scale* of the utility, not the *ordering* of
candidates within an episode, and ordering is what a router needs. This is the
quantitative form of the boundary condition in the paper's discussion and the
regime predicted by `E[m|state] = ⟨g_A(x), d⟩ − ‖d‖²`: the second term is a
within-episode constant shift, while the first requires the residual of a
predictor that has already absorbed the systematic part.

*Method note (two bugs found and fixed while building this probe, both caught
by controls rather than by the headline number):* the first version used
`np.nonzero` on an all-False mask, which silently zeroed the response features;
the second used the un-adapted expert. The reported numbers come from the
corrected version, and a permutation control plus an in-sample/out-of-sample
split are computed alongside. An earlier draft of the paper paragraph claimed
"response features change nothing" and "the probe policy is below random";
both were artefacts of those bugs and have been removed. The tie-breaking of
`argsort` also confounds raw policy-gain numbers when predictions are nearly
constant (the candidate pool is correlation-ordered), which is why the report
uses rank correlation and R² rather than policy gain for this probe.

## 5. Scope, limitations, and what would change the conclusion

* 16 targets (within the 8–32 requirement) and 3 seeds; per-target sign
  variance is large (per-target ΔStatic ranges −.0075 to +.036).
* The frozen protocol's seven scalar response summaries discard the response
  *direction*. Two enrichments were tested on target 0 (12 PCA components; and
  the theory-matched `⟨f_A, d⟩` interaction term, with a 4x wider router):
  neither changed the conclusion materially (q=4 .0131–.0152 against Static
  .0210–.0265 on that target). This is reported as an ablation, not as a
  headline.
* The negative result is about **identifiability from the router's observable
  features**, not about the oracle opportunity, and not about the ridge-based
  traffic results, which are unchanged.

## 6. Reproduction

```
# pretrain (per dataset, seed)
python experiments/train_neural_expert.py --dataset METRLA --seed 101 \
  --windows-per-node 200 --epochs 40 --out <run>/METRLA/seed101
# gate
python experiments/run_strong_backbone.py --dataset METRLA \
  --expert-root <run>/METRLA --out-root <out> --seeds 101,202,303 \
  --target-count 16 --q-values 4,8 --train-episodes 4000 \
  --calibration-episodes 1000 --test-episodes 2000 --router-epochs 20 --finetune-epochs 10
# analysis
python experiments/analyze_strong_backbone.py --run-root <out> --dataset METRLA \
  --out <derived> --primary ranked_response_q4
```
