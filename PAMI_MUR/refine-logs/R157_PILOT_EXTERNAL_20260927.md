# R157 — External validity of the pilot rule, and what this family can and cannot test

**Date:** 2026-09-27
**Artifact:** `results/derived/R157_pilot_external/pilot_external.json`
**Script:** `experiments/analyze_pilot_external.py`

## Design

The deployment procedure was validated on six traffic deployments. The
demonstration-selection family is a second domain with the same decision object,
and its runs record the class of every evaluation episode, so the pilot rule can
be re-tested there at no cost.

The split here is by **class** rather than by sensor: half the classes are the
pilot, the other half the deployment. That is a harder split than the traffic
one, because different classes are different sub-problems rather than different
draws from one population. Five benchmarks (CIFAR-10, CIFAR-100, SVHN, EuroSAT,
DTD), three seeds, 177 classes in total, and four contrasts spanning obvious and
marginal decisions.

## Result

| contrast | pooled Δ | full verdict | benchmark-level verdict agreement |
|---|---|---|---|
| router − pool | −.3006 | BELOW | 5/5 |
| router − relevance | −.2644 | BELOW | 5/5 |
| relevance − pool | −.0363 | BELOW | 5/5 |
| MMR − pool | −.0378 | BELOW | 5/5 |

Per-benchmark half-split verdict agreement, which is where the informative
variation is:

| contrast | CIFAR-10 (10 cls) | SVHN (10) | EuroSAT (10) | CIFAR-100 (100) | DTD (47) |
|---|---|---|---|---|---|
| router − pool | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| router − relevance | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| relevance − pool | **.307** | 1.000 | .985 | 1.000 | 1.000 |
| MMR − pool | **.338** | 1.000 | .978 | 1.000 | 1.000 |

1. **The pilot rule is directionally perfect in this family**: sign agreement is
   `1.000` for every contrast on every benchmark, and the pilot and deployment
   verdicts agree on 5/5 benchmarks in all four contrasts.
2. **The mechanism from R154/R156 reproduces in a second domain, quantitatively.**
   Agreement collapses exactly where the effect is marginal *relative to the
   pilot's resolution*: on CIFAR-10 the marginal contrasts are `Δ ≈ −.007` over
   10 classes, and two 5-class halves agree only 31–34% of the time, while on
   CIFAR-100 the same effect size over 100 classes agrees 100% of the time. What
   governs is effect size against resolution, not the domain or the modality.

## What this test cannot show, stated plainly

**Every contrast in this family has the same sign.** The router is below pooling
on all five benchmarks, relevance is below pooling on all five, and so on. So
"5/5 verdict agreement" is partly trivial: a rule that always answers BELOW
would score the same. The external test therefore confirms **specificity** (the
procedure does not manufacture positive calls where there are none) and cannot
test **sensitivity** (whether it would correctly call a positive deployment),
because this family contains no positive case to call.

That is a property of the demonstration family, not a gap in the analysis: as
the papers already report, candidate demonstrations contribute *labels* rather
than redundant predictions, so relevance is within a fraction of a percent of
the oracle and no learned router improves on it. A family in which the decision
is genuinely two-sided is the right place to test sensitivity, and this one is
not it.

## Consequence

The papers' external-validity statement should be the narrow, true one: the
procedure's *mechanism* --- a verdict is reliable when the effect is large
relative to the pilot's resolution --- reproduces in a second domain at a
different modality, while the operating characteristics measured in R154–R156
remain the traffic-family numbers. The class-split agreement rates are reported
alongside so the reader can see both what transferred and what did not.
