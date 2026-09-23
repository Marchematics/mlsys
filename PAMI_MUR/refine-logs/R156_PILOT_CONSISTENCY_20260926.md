# R156 — The split-half check: a high-precision filter on pilot verdicts

**Date:** 2026-09-26
**Artifact:** `results/derived/R156_pilot_consistency/pilot_consistency.json`
**Script:** `experiments/analyze_pilot_consistency.py`

## Why

R154 measured how often a pilot reproduces the full-sample verdict; R155 showed
that direction survives a time shift while the significance call partly does
not. Neither tells a practitioner how to recognise a pilot that should not be
trusted. The obvious candidate is internal consistency: split the pilot in half
and see whether the halves agree.

## Design

Six datasets, ridge regime, 32 targets each, three seeds averaged per target.
A pilot of `n` targets is drawn without replacement and split into two disjoint
halves of `n/2`; each half gets its own paired bootstrap verdict. Two
consistency definitions:

* **strict** — both halves resolve *and* point the same way;
* **sign** — both halves have the same sign, resolved or not.

The outcome measured is agreement with the full-sample (32-target) verdict, and
separately the operationally costly error: the pilot says "route" where the full
sample does not.

## Result

| n | overall agreement | strict coverage | agreement given strict | agreement given not strict | sign coverage | agreement given sign | false-route rate |
|---|---|---|---|---|---|---|---|
| 8 | .754 | .246 | **.967** | .684 | .746 | .785 | .039 |
| 16 | .854 | .347 | **.990** | .781 | .809 | .848 | .030 |
| 24 | .953 | .393 | **.999** | .923 | .842 | .947 | .027 |

1. **The strict check is a near-perfect filter.** At `n=16`, pilots whose halves
   both resolve in the same direction agree with the full-sample verdict
   **99.0%** of the time, against 78.1% for the rest. At `n=24` it is **99.9%**
   against 92.3%.
2. **It is high-precision and low-coverage.** Strict consistency holds for only
   a minority of pilots — 24.6% at `n=8`, 34.7% at `n=16`, 39.3% at `n=24` — so
   it certifies a verdict rather than producing one. The practical reading is:
   when the halves agree and both resolve, act; otherwise enlarge the pilot.
3. **The lenient sign check is nearly worthless.** Requiring merely the same sign
   covers 75–84% of pilots but raises agreement only from .754 to .785 at `n=8`
   and from .854 to .848 at `n=16` — i.e.\ no gain at all at the larger size. The
   usable rule is the strict one.
4. **The costly error is rare throughout** (2.7–3.9%) and shrinks with `n`.

## The caveat, stated

Strict consistency at `n=24` requires each 12-target half to resolve, which
happens mainly for large effects — and large effects are also the ones that
resolve on the full sample. The check is therefore partly a **signal-strength
filter**, not an independent test of correctness. That is exactly why it is
useful (signal strength is not observable from the undivided pilot) and also why
it should not be read as certifying a *small* effect: a genuinely marginal
deployment yields an inconclusive check, and the correct response there is more
labels, which is what the procedure says.

## The refined procedure

1. Label a pilot of about 24 targets.
2. Run the protocol unchanged: screening, budget, objective, router and
   pool-everything.
3. Split the pilot into two halves and compute the paired target-level interval
   on each.
4. If both halves resolve in the same direction, act on that verdict — it agrees
   with the full-sample verdict essentially always.
5. Otherwise the pilot is inconclusive: enlarge it. Do not fall back on a
   cheaper proxy (R140 probe, R153 headroom, the cached screen are all
   non-predictive).
