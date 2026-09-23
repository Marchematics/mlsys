# R143 — Adaptive rule re-derived with per-dataset provenance

**Date:** 2026-09-23

## Why

Auditing the deployable claim of the paper (the redundancy-adaptive selection
rule) turned up a provenance gap: the PEMS-BAY numbers quoted in the
manuscripts had no artifact on disk, and the METR-LA numbers did not match the
stored `R129_adaptive_rule` output. Since every number in the papers must be
recoverable from a file, the rule was re-derived per dataset and saved.

## Reproduction

```
python experiments/analyze_adaptive_rule.py \
  --run-root results/raw/R127_pool_modes_20260923_0546 \
  --out results/derived/R143_adaptive_metrla --statistic anchor_similarity
python experiments/analyze_adaptive_rule.py \
  --run-root results/raw/R128_pool_modes_pemsbay_20260923_0557 \
  --out results/derived/R143_adaptive_pemsbay --statistic anchor_similarity
```

The analyzer is deterministic (its split RNG is seeded), so re-running
reproduces `R129_adaptive_rule` exactly on METR-LA, which is the check that the
pipeline is sound.

## Result

| | METR-LA (24 cells) | PEMS-BAY (24 cells) |
|---|---|---|
| adaptive rule | **+.0081 [+.0017, +.0138]** | +.0129 [−.0066, +.0329] |
| fixed relevance | −.0006 | −.0020 |
| always pool | +.0066 | +.0098 |
| oracle pick of the two | +.0093 | +.0194 |
| rule / oracle | 87% | 67% |
| rule − pool | +.0015 | +.0031 |

## Consequence for the papers

The rule's benefit over the harmful fixed default is **resolved on METR-LA**
(interval excludes zero) and **unresolved on PEMS-BAY**; against always pooling
it is within noise on both. That is what both manuscripts now say, replacing
the previous "$+.0083\;[+.0002,+.0164]$ / $+.0137\;[-.0113,+.0387]$, 89% / 71%"
which could not be reproduced. The relevance, pooling and oracle-pick numbers
were already correct and are unchanged, which localises the discrepancy to the
split-based held-out gain of the rule itself.

The honest summary of the deployable contribution: **a label-free statistic can
prevent a harmful default; it does not beat pooling.** That is still a useful
prescription --- a practitioner applying relevance retrieval unconditionally
loses value in the low-novelty regime --- but it is a weaker claim than
"adaptive selection wins", and the papers state the weaker one.
