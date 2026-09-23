# KBS vs PAMI/TKDE: differentiation plan

## Why

`KBS_MUR/paper` and `TKDE_MUR/paper` currently share 90% of their text
(sections 2, 3, 4, 5, 8 are byte-identical; only the experiments section and a
few sentences differ), and `PAMI_MUR` is a copy of the TKDE line plus four
planned additions. Under Elsevier and IEEE policy that is one piece of
research, so the two could not be submitted concurrently.

The two manuscripts therefore get separate research questions. Overlap is
allowed only where it is infrastructure (data loaders, the frozen predictor,
the definition of `m(j | A)`, the response-aware estimator), never in the
contribution, the primary metrics, or the headline evidence.

## The two papers

| | PAMI / TKDE (MUR line) | KBS (this plan) |
|---|---|---|
| Question | How should the value of a candidate be estimated, and how much of it can a router realize? | When is a marginal-utility estimate reliable enough to act on, and how should the router abstain or stop to keep harmful selections under a budget? |
| Decision object | set-conditioned marginal utility `m(j | A)` | risk-controlled selection: select / abstain / stop against a harm budget |
| Method contribution | cached shortlist + compact response summaries + gap-weighted ranking calibration (R-MUR) | calibrated lower confidence bound with a state- and candidate-dependent radius, plus an abstention/stopping controller (RC-MUR) |
| Theory | one-step routing regret; conditional set-level guarantee | finite-sample interval coverage, selective-risk bound `P(harmful selection) <= alpha`, threshold-transfer condition under shift |
| Primary metrics | prediction gain, oracle recovery | selective risk, negative-transfer rate at matched coverage, coverage vs nominal level, abstention rate, calibration gap |
| Headline evidence | six-dataset benchmark, 32 targets each, multi-target consistency | harm ladder, cross-regime and cross-dataset calibration transfer, coverage--risk curves, matched-coverage comparison against thresholds |
| Predictor | ridge subset expert, plus a strong nonlinear backbone | unchanged ridge subset expert; backbone generalization is not claimed |
| Second domain | demo selection / vision benchmarks | none; the second setting is a distribution shift, not a new domain |

## Differentiation accounting

Content that is new or rewritten for KBS:

- title, abstract, introduction: 100% new;
- related work: new sections on selective prediction, abstention, conformal
  and distribution-free risk control, and negative transfer; the context
  selection paragraph is compressed to one paragraph;
- problem formulation: keeps the `m(j | A)` definition (infrastructure) and
  adds the risk-controlled decision problem, the harm budget, and the
  coverage/abstention quantities;
- method: the risk controller is new; the estimator is described in one
  paragraph as inherited infrastructure;
- analysis: new;
- experiments: new protocols and metrics; the six-dataset gain table does not
  appear;
- discussion and conclusion: new.

Estimated share of the KBS manuscript that does not appear in the PAMI/TKDE
manuscript: 65--75%, with the remaining 25--35% confined to shared definitions
and infrastructure, which is the fraction the two papers are allowed to share.
This exceeds the 40% target set for this pass.

## Evidence plan

| Experiment | Purpose | Source |
|---|---|---|
| E1 controlled harm ladder | selective risk and abstention as the harmful fraction grows | new runs of `run_mur_conservative_sweep.py` at harmful fraction .75, plus the existing .25/.50 sweeps |
| E2 radius ablations | global vs state-conditional radius vs free threshold, at matched coverage | new runs plus the existing sweep |
| E3 deployment-state calibration | coverage, selective risk, abstention on 32 fixed targets per dataset | `diagnostics.npz` of the existing six-dataset runs (per-state predicted and true marginals); no new GPU work |
| E4 cross-dataset and cross-regime calibration transfer | does a radius fitted on one dataset/regime stay valid on another | same saved states, calibrate on dataset A and evaluate on B; controlled sweep for the regime version |
| E5 selective-routing end to end | gain, harm and coverage of the abstaining policy against static thresholds | new run of the multi-target router with the calibrated gate, two or three datasets |

## Manuscript layout (`paper_risk/`)

1. Introduction: reliability of utility estimates, abstention, harm budget.
2. Related work: selective prediction and abstention; distribution-free risk
   control; negative transfer; context selection (one paragraph).
3. Problem: set-conditioned marginal utility (definition only), the harmful
   selection event, the harm budget, and the coverage/abstention quantities.
4. Method: RC-MUR -- shortlist, response summaries, calibrated lower bound,
   radius model, abstain and stop rules.
5. Analysis: interval coverage, selective-risk bound, threshold transfer.
6. Experiments: E1--E5.
7. Discussion, 8. Conclusion.
9. Supplement: protocols, per-seed tables, proofs.

## Guardrails

- No sentence of the new manuscript is copied from the PAMI/TKDE text except
  the definitions needed to make the new paper self-contained.
- The six-dataset prediction-gain table, the multi-target consistency table,
  the candidate-scaling margin audit and the observability probe are not part
  of the KBS contribution; they are cited at most as the setting in which the
  saved deployment states were produced.
- If a reviewer asks whether the two papers overlap, the cover letter discloses
  the companion manuscript and states the different question, method, and
  primary metrics.

## Status after the first build round (2026-09-21)

### Measured differentiation

A line-level comparison of `paper_risk/sections/*.tex` against
`../../PAMI_MUR/paper/sections/*.tex`, counting content lines of at least 40
characters and ignoring comments, finds **0 shared lines out of 353**: the new
manuscript shares no prose with the MUR line. The shared material is confined
to the definition of `m(j | A)` and the description of the scorer, both of
which are paraphrased and attributed to the companion setting rather than
reused.

### Evidence already produced

`scripts/analyze_risk_control.py` (protocols P1--P7) reads the saved
deployment traces of the multi-target runs and produces:

- budget control on six benchmarks: at `alpha = .02` the calibrated gate
  retains 5.3--10.7% of decisions with realised harmful mass .0155--.0260
  against the nominal .02, and bootstrap intervals contain the nominal value
  on every dataset;
- interval certification against budget control at the same level: the
  interval gate retains 1.0--2.2% of decisions, a factor of 4.3--9.1 less
  coverage than budget control;
- threshold transfer: a threshold calibrated on five datasets realises
  .0155--.0237 on the sixth at a nominal .02, and overshoots (.070) at a
  nominal .05;
- the risk--coverage frontier of the scorer: the smallest harmful share any
  threshold can achieve is .202 on average at 2% coverage, so a share budget
  of 10% is infeasible at any useful coverage on five of six datasets;
- the comparison against uncalibrated gates: the sign gate accepts 45.0% of
  decisions with a 39.6% harmful share and a mass of .179, while the
  calibrated gate keeps 86% of its mean accepted utility and removes 87% of
  its harmful selections.

### Open items

1. Controlled harm ladder at harmful fractions .25/.50/.75 with saved
   state-level diagnostics (run `R200_harm_ladder_*`, in progress). The
   manuscript leaves a marked gap in Section 6.2 for it.
2. End-to-end selective routing: apply the calibrated gate inside the
   sequential router and report episode-level gain, negative-transfer rate and
   contexts used per budget (Section 6.6).
3. Figures for the risk manuscript: risk--coverage frontier with operating
   points, budget sweep, transfer plot.
4. Reference audit for the selective-prediction and conformal entries added to
   `paper_risk/references.bib` (marked with `TODO(reference-audit)`); the web
   search endpoint is currently unavailable, so volume/page details are not yet
   verified.
5. Submission packaging for `paper_risk` (highlights, cover letter,
   declarations) once the experiments above are complete.

## Controlled harm ladder completed (2026-09-21)

The `R200` sweep finished: three harmful fractions, five seeds each, with
per-state diagnostics. Summary written to `results/derived/R204_harm_ladder_summary`
and rendered as `paper_risk/tables/tab_harm_ladder.tex`:

| harmful fraction | calibrated gain / harm / contexts | uncalibrated gain / harm | oracle gain | $H_{\mathrm{state}}$ |
|---|---|---|---|---|
| .25 | .0722 / .0001 / 0.31 | .2734 / .0317 | .4064 | .151 |
| .50 | .0459 / .0006 / 0.18 | .2219 / .0854 | .3379 | .075 |
| .75 | .0365 / .0012 / 0.14 | .1179 / .2267 | .2130 | .000 |

Two results matter for the alternative direction: the uncalibrated gate's
negative-transfer rate grows from .032 to .227 as hazard rises while the
calibrated gate stays below .0012, and structural headroom vanishes at a
harmful fraction of .75, which shows that state conditioning is only valuable
in tasks that still contain useful contexts.

This evidence belongs to `paper_risk/`, not to the frozen KBS submission, which
is unchanged by it.
