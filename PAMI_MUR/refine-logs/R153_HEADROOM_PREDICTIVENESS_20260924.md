# R153 — Structural headroom does not predict routing benefit (either regime)

**Date:** 2026-09-24
**Artifact:** `results/derived/R153_headroom_predictiveness/headroom_predictiveness.json`
**Script:** `experiments/analyze_headroom_predictiveness.py`

## Question

The protocol defines structural headroom `H_state = (G_seq − G_stand)/G_seq` and
treats it as the opportunity a router could capture. The natural prediction is
that targets — or datasets — with more headroom benefit more from routing. The
strong-backbone regime had already answered this negatively (`r = +.047`,
`p = .86`, on 16 targets); this round asks the same question in the regime where
routing actually works, using the six-dataset ridge protocol.

## Result

| dataset | targets | mean Δ | se | t | targets where routing wins | mean headroom | r(H, Δ) |
|---|---|---|---|---|---|---|---|
| METR-LA | 32 | +.0067 | .0014 | **+4.85** | .78 | .0959 | −.235 |
| PEMS-BAY | 32 | +.0303 | .0059 | **+5.14** | .94 | .1153 | −.043 |
| PEMS03 | 32 | +.0013 | .0009 | +1.53 | .69 | .1190 | +.203 |
| PEMS04 | 32 | −.0008 | .0020 | −0.40 | .50 | .0834 | +.266 |
| PEMS07 | 32 | +.0041 | .0014 | **+2.87** | .75 | .0891 | +.310 |
| PEMS08 | 32 | −.0026 | .0008 | **−3.06** | .38 | .0761 | +.023 |

1. **Per target, headroom is uninformative.** Pooled over 192 targets with
   dataset fixed effects, the correlation between `H_state` and the router's
   advantage over pooling is **`r = +.040`, `p = .58`** (Spearman `+.104`).
   Per dataset the coefficient ranges from `−.235` to `+.310` with no
   consistent sign.

2. **Across datasets the ordering is broken by its most informative case.**
   Mean headroom and mean advantage correlate at `r = +.60` (`p = .21`, n = 6),
   which is suggestive but unresolved — and PEMS03 has the *highest* mean
   headroom of the six (`.119`) with an advantage of `+.0013`, while PEMS-BAY
   has similar headroom (`.115`) and the largest advantage (`+.0303`).

3. **The two unresolved ties are effect-size differences, not under-powered
   ones.** The per-target standard error of the difference is comparable across
   benchmarks (`.0008`–`.0059`), while the effect ranges from `t = +5.14`
   (PEMS-BAY, 94% of targets) through `t = +1.53` (PEMS03, 69%) to
   `t = −3.06` (PEMS08, 38%). Resolving PEMS03 at its observed size would need
   roughly three times as many targets, so the honest description is "a
   genuinely small effect", not "an unresolved one".

## Why this matters

The paper's central diagnostic quantity is **not** predictive of routing
benefit, and the negative now spans both regimes: `+.047` with a strong
backbone and `+.040` in the regime where routing wins. Headroom is a necessary
condition — a router cannot capture what the oracle cannot reach — and nothing
more.

What does discriminate is the **realized share** measured under the
deployment's own screening, budget and objective: it separates the expert ladder
(`.164` to `.024`), the pool geometries and the two regimes, while the
opportunity does not. That is now the recommendation in both manuscripts: the
diagnostic protocol measures realizability, not opportunity.

This also sharpens the six-dataset reporting: the datasets differ by effect
size, and the win fraction (`.38` to `.94`) is the statistic a practitioner
should carry forward rather than the headroom.

## Manuscript actions

* `PAMI_MUR/paper/sections/6_experiments.tex` — new paragraph before the
  strong-backbone table carrying both the correlation result and the
  effect-size decomposition.
* `TKDE_MUR/paper/sections/7_experiments.tex` — the same result before regime B,
  so the two regimes are read together.
* `TKDE_MUR/paper/sections/5_diagnostics.tex` — the diagnostics section now
  states explicitly that the protocol measures realizability rather than
  opportunity, with this evidence.
