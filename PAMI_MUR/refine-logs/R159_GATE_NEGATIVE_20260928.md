# R159 — The gate signal does not exist: the router cannot tell when it is wrong

**Date:** 2026-09-28
**Run:** `KBS_MUR/results/raw/R159_gate_20260928_1000` (36 cells: 3 datasets × 12 targets × 1 seed, 24,000 episodes per dataset)
**Artifact:** `results/derived/R159_gate/gate.json`
**Script:** `experiments/analyze_gate.py`

## Question

R158 measured the headroom for an episode-level confidence gate: taking the
better of routing and pooling episode by episode would add `+.0148` to `+.0778`
over pooling, three to seven times the router's own advantage. That is worth
building **if** per-episode router correctness is predictable from something
observable at decision time. This is that test.

Four signals are recorded per episode, each averaged over the four selection
steps, all available without labels:

| signal | what it measures |
|---|---|
| `screen_margin` | top-two margin of the cached screen --- how decisive the cheap view is |
| `pred_margin` | top-two margin of the utility model inside the shortlist |
| `pred_top1` | the utility model's top predicted utility |
| `response_norm` | prediction-change norm of the candidate it picked |

A gate routes only when the signal exceeds a threshold fitted on a held-out half
of the episodes; the other half measures what the gate realises.

## Result: every signal is at chance

AUC for predicting "the router beats pooling on this episode", where 0.5 is no
information:

| dataset | screen_margin | pred_margin | pred_top1 | response_norm |
|---|---|---|---|---|
| METR-LA | .495 | .491 | .515 | .521 |
| PEMS04 | .502 | .481 | .486 | .506 |
| PEMS08 | .496 | .482 | .488 | .494 |

Across all twelve combinations the AUC lies in `[.481, .521]`. **Not one signal
carries information about whether the router will be right on this episode.**

The realised gates follow from that: the best held-out gate adds at most
`+.0030` over pooling (PEMS04, routing 35% of episodes), against an oracle
ceiling of `+.013` to `+.048` on these datasets. The headroom is real and
**unreachable through these signals**.

| dataset | router | pool | router wins | oracle gate | best gate − pool |
|---|---|---|---|---|---|
| METR-LA | +.0158 | +.0121 | .59 | +.0550 | −.0006 |
| PEMS04 | +.0364 | +.0350 | .49 | +.0535 | +.0030 |
| PEMS08 | +.0301 | +.0341 | .47 | +.0470 | +.0006 |

## Why this is a confirmation, not a surprise

The result is what the paper's own theory predicts. The identifiable term of the
marginal is the anchor's residual; once a strong predictor has absorbed the
systematic part of that residual, what is left is noise, and noise is
unpredictable *by construction* --- including by the router's own confidence.
The predicted margin is a function of the same absorbed signal, so it cannot
carry information the signal does not have. An AUC of `.48`--`.49` for
`pred_margin` is that statement measured.

This is the fifth theory-guided expectation to fail, alongside:

1. the predictability probe cannot gate deployment (R140);
2. structural headroom does not predict routing benefit in either regime (R153);
3. the theory's own degenerate-regime policy --- least perturbation --- loses to
   pooling (R151);
4. the screen is a symptom, not the cause (R145);

and it is the one that closes the method question. The remaining loss is not a
tuning problem, a head-design problem or a gating problem: **the estimation
error is present and unobservable**, which is the boundary the paper set out to
characterise.

## What the papers should say

The method section gains a short paragraph: the oracle gate ceiling, the chance
AUCs of all four decision-time signals, and the conclusion that the headroom is
unreachable --- presented as the measured consequence of the identifiability
result rather than as a defect of a particular router. No claim of improved
performance is made; the paragraph replaces the R158 forward pointer, which
proposed exactly this experiment.

## Provenance notes

* Twelve targets per dataset, one seed, 24,000 episodes per dataset: the gate
  question is about per-episode predictability, so episodes --- not seeds --- are
  the sample, and 24,000 is far more than the AUCs need to resolve a departure
  from 0.5.
* The threshold is fitted on one half and realised on the other, so the reported
  gate means are held out. The ceiling is the oracle gate on the realised half.
* `gate_response_norm` on PEMS04 routes only 35% of episodes for +.0030; with a
  chance AUC that is threshold-fitting noise, not a signal, and is reported as
  such.
