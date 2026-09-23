# R145 — The screen is not the bottleneck: the head is

**Date:** 2026-09-23
**Run:** `results/raw/R145_noscreen_q16_20260923_1830/q16`
**Baseline:** `results/raw/R141_adapt_sweep_20260923_1500/ft10`

## Question

R142 showed that with a strong adapted backbone the router falls to the level of
random selection as the pool grows. Two explanations predict that outcome:

1. **Noisy screen.** The cached view that forms the shortlist is uncorrelated
   with the true marginal (Measurement 1), so at large `K` the shortlist is a
   nearly random subset and the response head never sees the good candidates.
   On this reading the fix is a better screen.
2. **Unidentifiable head.** The head itself cannot rank candidates from
   observable response under a strong adapted anchor (Corollary 1 / the
   identifiability result). On this reading no screen can help.

The two are separable by removing the screen: run with `q = K`, so the head
scores every candidate and the shortlist introduces no loss at all.

## Design

METR-LA, R101 transformer adapted 10 epochs, `K = 16`, budget 4, the same 16
matched cells as R141/R142. `--q-values 16` makes the runner score the full
pool; the policy is reported as `ranked_response_full`. Cost is
`B(1+K) = 68` expert rows per episode against 20 for `q=4`, i.e. the same
budget as reranking the entire pool.

## Result

| policy | mean gain | paired Δ vs pool-all | 95% CI |
|---|---|---|---|
| screened `q=4` | +.0076 | +.0017 | [−.0100, +.0178] |
| screened `q=8` | +.0087 | +.0028 | [−.0086, +.0197] |
| **no screen, full pool** | **+.0088** | +.0029 | [−.0077, +.0193] |
| cached-only router | +.0061 | +.0002 | [−.0099, +.0134] |
| random four | +.0006 | −.0053 | [−.0074, −.0032] |
| stand-alone oracle | +.0852 | +.0793 | [+.0628, +.0987] |

* **Removing the screen changes nothing**: paired full-pool minus screened `q=4`
  is **+.0012 [−.0023, +.0044]**, not significant.
* The full-pool head is still **not** significantly above random four
  (+.0082 [−.0029, +.0243]) and still not above pooling.

## Conclusion

The **screen is not the bottleneck**. Given an unrestricted view of all sixteen
candidates and the same response head, the router performs identically to the
screened version. The limiting factor is what can be identified from the
response of a predictor that has already absorbed its own residual — which is
exactly what the theory predicts, and which no screen can supply.

This also corrects the reading of Measurement 1. The cached screen's
chance-level correlation with the true marginal is a **symptom** of
unidentifiability, not its cause: replacing the screen with perfect information
about which candidates to consider does not change the outcome.

## Consequences for the papers

* Both manuscripts gain this as the closing piece of the regime-B argument: the
  identification failure survives the removal of every efficiency compromise.
* It strengthens rather than weakens the compute story: at `K=16` the shortlist
  is free in value terms (Δ +.0012, n.s.) at a 3.4× compute saving
  (20 vs 68 expert rows per episode), so the two-stage design is justified even
  though it is not the cause of the boundary.

## Incident

The runner crashed at the *summary* stage with
`KeyError: 'ranked_response_q4_gain'` because that stage hardcoded the `q4`
policy name, which does not exist when `q = K` (the runner renames it to
`ranked_response_full`). All 16 raw cells were written correctly and the
analysis above reads them directly. The bug is fixed in
`experiments/run_strong_backbone.py` (the primary policy is now resolved from
the policies actually present, and the gate key is derived from it), and the
suite still passes (63 tests).
