# TKDE consolidation map

Status: active as of 2026-09-24.

The project now has one submitted companion method manuscript and one active
new manuscript. There is **no planned PAMI submission**. The historical
`PAMI_MUR` tree is an evidence warehouse; its non-submitted advantages are
assigned to TKDE.

## Frozen submitted companion

Do not use the submitted IPM manuscript's main numerical tables as TKDE's
headline evidence. Shared problem notation and the basic R-MUR routing primitive
may be summarised with explicit disclosure so that TKDE remains self-contained.

## Material owned by TKDE

| asset | provenance | TKDE role |
|---|---|---|
| smooth-loss response expansion | `PAMI_MUR/refine-logs/THEORY_GENERAL_LOSS.md` | general theory |
| exact Bregman identity | same | theory centerpiece |
| cross-entropy Hessian bound and sandwich | same | classification/general-loss extension |
| response sufficiency | same | justification for response-aware selection |
| theory-shaped bilinear/signed heads | R116/R117/R119 family | method extension |
| recent selector matrix | R125, R151/R152 | strong positive baseline evidence |
| strong nonlinear subset predictor | R103 + confirmations | predictor-strength regime |
| expert/adaptation ladder | later refinement records | theory prediction test |
| observability probes | R140, R159 | identifiability mechanism |
| pool geometry and novelty | R127/R128 family | mechanism |
| label-free adaptive rule | R143 | actionable adaptation |
| budget and backward/forward search | R121/R120 | alternative explanations |
| candidate library 16→128 | R142 | scale stress test |
| shortlist removal | R145 | screen-vs-identifiability test |
| response-geometry limit case | R151 | theory stress test |
| policy-gradient selector | R109/A5 | alternative learned policy |
| harmful/anticorrelated pools | R122 family | robustness |
| CLIP second task family | R110/R112/R115/R123/R124 | cross-task evidence |
| forced-budget analysis | second-family refinement | ranking-vs-stopping separation |
| compute/latency/memory | A4 measurements | efficiency |
| pilot power | R154 | deployment sample complexity |
| time-shift pilot | R155 | temporal robustness |
| split-half consistency | R156 | pilot self-audit |
| gate ceiling | R158 | achievable episode-level benefit |
| chance-level gate signals | R159 | negative control / identifiability limit |
| claim and differentiation audits | R147, differentiation audit | reporting integrity |

## Required paper order

The manuscript should preserve the following argumentative order:

```text
state-conditioned marginal utility
        ↓
general response–utility theory
        ↓
response-aware / theory-shaped selection
        ↓
strong identifiable-regime comparison
        ↓
predictor-strength and library-size regime change
        ↓
observability + pool-geometry mechanism
        ↓
adaptive rule
        ↓
second task family
        ↓
compute + pilot deployment validation
```

The manuscript must not be framed as a negative-results paper. Negative results
are retained where they eliminate alternative explanations, but the top-level
claim is constructive: the method is strong when the decision signal is
identifiable, the theory explains that signal, and the diagnostics/adaptation
show how to act when the regime changes.
