# R158 — How much headroom is there for an episode-level confidence gate?

**Date:** 2026-09-27
**Artifact:** `results/derived/R158_gate_ceiling.json`

## Why

Every remaining loss in the ridge regime has the same shape: the router wins on
average but loses on a substantial minority of episodes, and on PEMS08 it loses
on average. The theory's regret bound is about exactly this --- routing on an
estimate that is sometimes wrong --- so the natural method change is a
*confidence gate*: route when the estimate is trustworthy, pool when it is not.

Before building one, this measures its ceiling. Using the per-episode gains that
all 192 six-dataset cells record (2,000 episodes each), the oracle gate is the
per-episode maximum of the router's and pooling's gain. No label-free gate can
beat it; it bounds what any episode-level switching rule could achieve.

## Result

| dataset | pool | router | oracle gate | gate − pool | gate − router | episodes where the router loses |
|---|---|---|---|---|---|---|
| METR-LA | +.0080 | +.0146 | +.0567 | **+.0487** | +.0420 | .403 |
| PEMS-BAY | +.0254 | +.0557 | +.1032 | **+.0778** | +.0475 | .395 |
| PEMS03 | +.0256 | +.0269 | +.0436 | **+.0180** | +.0167 | .501 |
| PEMS04 | +.0365 | +.0357 | +.0539 | **+.0174** | +.0182 | .524 |
| PEMS07 | +.0242 | +.0284 | +.0443 | **+.0200** | +.0159 | .488 |
| PEMS08 | +.0359 | +.0333 | +.0507 | **+.0148** | +.0174 | .522 |

1. **The headroom is large.** An oracle gate would add `+.0148` to `+.0778` over
   pooling, which is **3 to 7 times** the router's own advantage --- and on
   PEMS08, where the router loses by `-.0026`, it would turn the loss into a
   `+.0148` gain.
2. **The loss is not concentrated.** The router loses on 39--52% of episodes
   across all six datasets. This is not a few catastrophic episodes that a
   safety rule could excise; it is a near-even split, which is what makes the
   gate both valuable and hard.
3. **This does not contradict R153.** R153 showed *target-level* headroom does
   not predict routing benefit; this measures *episode-level* switching
   headroom, a different quantity. The two together say: the opportunity is
   real and large, and the aggregate diagnostic that the protocol measures does
   not expose where it is.

## What this does and does not establish

**Does:** there is enough headroom for an episode-level gate to be worth
building, by a factor of 3--7 over the current method's advantage.

**Does not:** that any observable signal can realise it. The gate must decide
without labels, and this measurement deliberately uses the labels. Whether an
observable per-episode statistic --- the router's own predicted margin being the
obvious candidate --- tracks the router's per-episode correctness is a separate
question, and it is not answerable from the current artifacts because the runs
record gains but not the router's scores.

## Next step this sets up

Save per-episode router scores and margins in the six-dataset protocol (the
runner already has them in memory at selection time), then test whether
`router wins on this episode` is predictable from the predicted margin and the
candidate geometry. If it is, the gate is a genuine method improvement with a
measured ceiling of `+.015` to `+.078`; if it is not, that is the fifth
theory-guided prediction to fail and it belongs in the same table as the others.
Either way the answer is cheap once the scores are saved.

## Manuscript handling

Reported as headroom for future work, not as a result: one sentence in each
experiments section stating that the oracle episode-level gate would add
`+.015`--`.078` over pooling and that the per-episode scores needed to test a
realisable gate are not yet recorded. No claim of achieved performance is made
from an oracle quantity.
