# PAMI strengthening: consolidated results

Date: 2026-09-21. Everything below is copied from the released artifacts named
in each section; nothing is estimated. Negative results are reported as
negative.

## Deliverables

| Addition | Status | Primary artifact |
|---|---|---|
| A1 strong nonlinear backbone | done, **gate not passed** | `refine-logs/R103_STRONG_BACKBONE_20260921.md` |
| A2 second task family | done, 3/5 paired pass | `refine-logs/R110_SECOND_FAMILY_20260921.md` |
| A2b redundancy-structured variant | done | `refine-logs/R112_REDUNDANT_SECOND_FAMILY_20260921.md` |
| A3 general smooth-loss theory | done, validated | `refine-logs/THEORY_GENERAL_LOSS.md`; paper Sec. 5 + appendix |
| A4 compute / latency / memory | done | `results/derived/R108_compute_metrla/compute_report.json` |
| A5 nearest-neighbour baselines | done | `results/raw/R109_learned_policy_metrla` |
| A1b second backbone (DeepSets) | done, gate not passed (ordering favourable) | `results/raw/R111_deepsets_metrla` |

## A1. Strong backbone: the gate does not pass

Subset-capable spatio-temporal transformer (499,722 parameters), trained with
subset dropout so arbitrary subsets are in distribution, pretrained per dataset
and then adapted per target on exactly the episodes the ridge expert sees.
Expert strength verified: adapted MSE is 4–6% lower than the ridge expert on
METR-LA and 26–31% lower on PEMS-BAY across mask families
(`results/derived/R104_expert_strength_*`).

16 targets x 3 seeds x 2 datasets, K=16, B=4
(`results/raw/R103_strong_backbone_gate_20260921_0350/{METRLA,PEMSBAY}`):

| policy | METR-LA | PEMS-BAY |
|---|---|---|
| random four | .0061 | .0161 |
| relevance | .0070 | .0107 |
| MMR | .0076 | .0113 |
| DPP | .0091 | .0142 |
| pool all 16 | .0133 | .0224 |
| Standalone Utility | .0068 | .0138 |
| cached-only router | .0076 | .0133 |
| R-MUR q=4 | .0097 | .0130 |
| R-MUR q=8 | .0108 | .0141 |
| standalone oracle | .0904 | .0880 |
| sequential oracle | .1051 | .1037 |

* METR-LA: ΔStatic +.0028 [-.0012, +.0083], ΔCached +.0021 [-.0006, +.0053],
  14/16 targets positive, 9/16 above Static.
* PEMS-BAY: ΔStatic -.0008 [-.0036, +.0019], ΔCached -.0003 [-.0024, +.0019],
  15/16 positive, 5/16 above Static. Every learned router is below plain random
  selection on this dataset.
* **Structural headroom is preserved** (oracle gap .0148 / .0157 absolute;
  per-target H_state .075–.232 on METR-LA), so this is not a case of "nothing to
  select": it is a case of "nothing identifiable".

### Why: three independent diagnostics

1. **Cached screen is at chance.** Recall of the oracle-best candidate in the
   shortlist: q=2 .13–.15, q=4 .26–.29, q=8 .49–.53; screen–truth correlation
   ≈ 0 (`results/derived/R105_screen_diagnosis`).
2. **Utility is state-conditioned, not candidate-conditioned.** Between-
   candidate variance is 0.03–0.13% of the total
   (`results/derived/R106_candidate_identity`); per-candidate means are stable
   but nearly equal.
3. **No observable-feature model ranks candidates.** Ridge and small-MLP probes
   on anchor history, candidate history, candidate identity, the response
   vector, its magnitude and the base–response interaction all have *negative*
   out-of-sample R² (−.03 to −.20) and within-episode rank correlation
   |ρ| ≤ .036, while the oracle is 7x the random expectation
   (`results/derived/R107_observability`).

The theory (A3) predicts exactly this regime: with `g_A(x)=2(μ(x)−f_A(x))`,
`E[m|state] = ⟨g_A(x), d⟩ − ‖d‖²`, so the identifiable term is the residual of
a predictor that has already absorbed the systematic part. A stronger predictor
shrinks the realizable share even as the oracle share persists.

## A2. Second family: transfers where selection has value

Demonstration selection for a frozen in-context CLIP classifier, 5 benchmarks x
3 seeds x 1,602 episodes, R-MUR re-instantiated unchanged
(`results/derived/R110_demo_selection_summary/gate_table.csv`):

| benchmark | R-MUR q4 | Static | Cached | ΔStatic | pass (paired / hierarchical) |
|---|---|---|---|---|---|
| CIFAR-10 | .0643 | .0482 | .0433 | +.0160 | PASS / PASS |
| CIFAR-100 | .5247 | .5769 | .5381 | -.0522 | FAIL / FAIL |
| SVHN | .8809 | .7530 | .7688 | +.1279 | PASS / FAIL |
| EuroSAT | .1294 | .1074 | .1125 | +.0221 | PASS / FAIL |
| DTD | .3322 | .5981 | .6478 | -.2658 | FAIL / FAIL |

Boundary condition measured, not assumed: the sequential and standalone oracles
coincide to four decimals (H_state ≤ 5e-4), and kNN relevance reaches
99.4–99.9% of the oracle. The family's candidate pools contain no redundancy,
so there is no state-conditioning to exploit. The R110 protocol also required a
fix: without query-independence masking the 32 same-class queries voted on
their shared label (pre-fix run kept as evidence).

## A3. Theory: exact, bounded, and empirically informative

* Exact Bregman identity containing the squared-loss identity as a special case.
* Two-sided bound for general smooth losses with explicit constants.
* Softmax cross-entropy: Hessian spectral norm ≤ 1/2 (proved via Bhatia–Davis,
  numerically confirmed at exactly 0.5), giving
  `⟨e_y − p_A, d⟩ − ¼‖d‖² − λc ≤ m ≤ ⟨e_y − p_A, d⟩ − λc`.
* Response sufficiency: conditional on the state, the Bayes-optimal squared-loss
  utility is affine in the response — the parameterisation R-MUR uses.
* Validation: squared-loss identity exact to 3.3e-7; CE sandwich 0 violations in
  92,995 samples with Spearman .971 / R² .853 between the first-order term and
  the true marginal (`results/derived/R110_response_theory_ce/validation.json`).

## A4. Compute, latency and memory (measured, not inferred)

`results/derived/R108_compute_metrla/compute_report.json`, 2,000 episodes:

| policy | expert rows / episode | ms / episode | peak MB |
|---|---|---|---|
| cached screening only | 0 | .03 | 41 |
| R-MUR q=2 | 12 | .74 | 173 |
| R-MUR q=4 | 20 | 1.19 | 173 |
| R-MUR q=8 | 36 | 1.59 | 173 |
| R-MUR full pool | 63.5 | 2.65 | 173 |
| sequential oracle | 63.6 | 1.70 | 173 |

The shortlist keeps its efficiency (q=4 uses 32% of full-pool probes) and the
response stage adds ~83 MB of peak device memory.

## A5. Baselines

Diversity (DPP), relevance-weighted facility location, random, MMR, relevance:
implemented and evaluated in the same protocol on both the ridge and the
nonlinear backbone. A learned policy-gradient subset selector
(`experiments/policy_selector.py`, REINFORCE with a batch-mean baseline) is
running on the same 48 gate cells; on the 24 cells completed so far its mean
gain is −.0053, i.e. below random selection and below every non-learned
baseline, consistent with the observability limit rather than with a defect of
marginal-utility prediction.

## Exploration round (R116-R124, see `R124_EXPLORATION_20260923.md`)

* SOTA matrix on 48 METR-LA cells: strongest deployable baseline is **pool all**
  (+.0133); our bilinear theory-shaped head is the best *selection* policy
  (+.0117) and is statistically tied with pool-all; frozen R-MUR +.0097;
  response-free learned routers at random level; policy-gradient selector
  -.0041; oracles +.090/+.105.
* Oracle budget curve: optimal k = 3 (+.1048), not 4; random-k monotone
  (+.0264 at k=16); identifiability gap +.078.
* Backward elimination = forward greedy (+.0002), so search direction is not
  the limitation.
* Falsified hypotheses: least-disturbance selection (-.0047 vs Static),
  curvature-correction alone (neutral), consensus alignment (unstable).

## Method iteration and SOTA status (rounds 4-5)

* **Ridge regime (selection is estimable)**: R-MUR beats all 13 baselines on
  both datasets --- METR-LA +.0146 [+.0093,+.0200] against pool-all +.0080, and
  PEMS-BAY +.0557 [+.0441,+.0673] against pool-all +.0254. Eight of twelve
  training-free baselines are *significantly harmful*: relevance -.0165/-.0199,
  MMR -.0180/-.0249, DPP -.0166, k-center -.0187, facility location -.0163,
  k-means representatives -.0080, random -.0077, mutual information -.0055
  (METR-LA figures).
* **Strong backbone (selection is not estimable)**: pooling has the highest
  point estimate; mutual information ties it (+.0134 vs +.0133); frozen R-MUR
  +.0108, bilinear +.0112. **Corrected after paired testing (R141/R143):** no
  method *separates* from pooling. On METR-LA pooling ties the response-aware
  family (-.0036 [-.0091,+.0031]) while beating every response-free rule; on
  PEMS-BAY pooling significantly beats everything, including ours
  (-.0094 [-.0135,-.0053]). The earlier "nothing beats pooling" phrased a tie
  as a loss and is withdrawn.
* **Test-time adaptation is what destroys the value (R141)**: in a
  one-parameter sweep of fine-tuning epochs on 16 matched cells the structural
  oracle is invariant (+.0831 to +.0858) while the realized share collapses from
  .164 to .024 between 0 and 2 epochs (paired -.140 [-.197,-.085], resolved) and
  then plateaus. Later epochs buy 1.2% MSE and no share; the apparent recovery
  is not significant and is not claimed.
* **Mechanism**: the sign of relevance ranking is set by candidate--anchor
  redundancy, replicated on PEMS-BAY (-.033 -> +.016 across pool geometries),
  and turned into a deployable label-free rule with held-out threshold
  selection. **Re-derived with provenance (R143):** +.0081 [+.0017,+.0138] on
  METR-LA (resolved) and +.0129 [-.0066,+.0329] on PEMS-BAY (unresolved),
  versus -.0006 / -.0020 for fixed relevance and +.0066 / +.0098 for always
  pooling; the rule beats the harmful default but does not separate from
  pooling. The previously quoted +.0083/+.0137 and 89%/71% capture were not
  reproducible from any artifact and are replaced.
* **The identification probe is not a deployment gate (R140)**: at 2,000
  training and 1,200 test episodes on three targets, probe R-squared stays
  within noise of zero for both a weak and a strong expert (-.004 +- .019 and
  -.015 +- .028) and its value excess over the per-cell random baseline is
  +.048 / +.067 of the oracle against a spread of .09, versus a ladder effect of
  .205. The proposed threshold rule is not shipped. Within the neural family
  anchor MSE does order realized share perfectly (Spearman +1.0, slope 10.5
  share points per unit MSE); across families the relation breaks.
* **Method iterations all bracket the same ceiling**: scalar, curvature
  corrected, bilinear and signed bilinear heads; the learned curvature
  coefficient confirms the sign diagnosis (+.14 on CIFAR-10, p<1e-6) without
  improving performance. The binding constraint is what the router can observe.

* **The headline win is 4/6, not 6/6 (R144)**: paired against the best
  competitor --- which is the trivial pool-everything policy on every dataset ---
  R-MUR is significantly ahead on METR-LA (+.0067 [+.0044,+.0089]), PEMS-BAY
  (+.0303 [+.0232,+.0378]), PEMS03 (+.0013 [+.0002,+.0024]) and PEMS07 (+.0041
  [+.0025,+.0058]), ties on PEMS04 (-.0008 [-.0031,+.0013]) and is significantly
  *below* on PEMS08 (-.0026 [-.0036,-.0016]). All three manuscripts previously
  said "best on every dataset"; the KBS table omitted pooling entirely. Both are
  corrected, and `tab:deployable` now includes the pooling row with recomputed
  average ranks (R-MUR 1.33, pool 2.00).
* **More candidates do not help (R142)**: growing the pool 16 -> 128 raises the
  standalone oracle 40% (.0851 -> .1196, paired gap growth +.0446
  [+.0376,+.0514]) while the router falls from +.0076 to -.0025 and becomes
  indistinguishable from random selection; at 32 and 64 candidates it is
  significantly below pooling.
* **The screen is not the bottleneck (R145)**: with q=K=16, so the response head
  sees every candidate, the result moves by +.0012 [-.0023,+.0044] --- not
  significant --- and remains at random-selection level. The identification
  failure survives the removal of the efficiency compromise, which also makes
  the two-stage design free in value terms (20 vs 68 expert rows per episode).
* **A resolved component win under a strong backbone (R141)**: against the
  Standalone Utility ablation, R-MUR gains +.0122 [+.0061,+.0198] with an
  *unadapted* backbone, and only +.0024 and +.0027 (both unresolved) after 2 and
  30 adaptation epochs.

## Round 21 — the gate signal does not exist (the method question closes)

* **R159**: 24,000 episodes per dataset across three deployments, four
  decision-time signals recorded (cached screen margin, utility-model margin,
  utility-model top score, response norm). Every dataset-by-signal **AUC lies in
  [.481, .521]** — the router's own confidence carries **no** information about
  when it is wrong.
* The best held-out gate adds at most **+.0030** over pooling against an oracle
  ceiling of +.013 to +.048: the R158 headroom is real and **unreachable**.
* This is *predicted by the paper's own theory*: the identifiable term is the
  anchor's residual, a strong anchor absorbs its systematic part, and what
  remains is noise — which no function of the same absorbed signal can
  anticipate.
* Fifth theory-guided expectation to fail (after the probe, target headroom, the
  least-perturbation limit case, and screen-as-cause), and the one that closes
  the method question: the remaining loss is not a tuning, head-design or gating
  problem.

## Round 20 — measured headroom for the one method change left

* **R158**: using the per-episode gains all 192 six-dataset cells record, an
  oracle episode-level gate (take the better of routing and pooling each
  episode) would add **+.0148 to +.0778** over pooling — **3 to 7 times** the
  router's own advantage — and would turn the PEMS08 loss into a gain.
* The router loses on **39–52% of episodes** across all six datasets: a near-even
  split, not a few outliers a safety rule could excise. That is what makes a
  confidence gate both valuable and hard.
* Does not contradict R153: that measured *target*-level headroom (not
  predictive); this measures *episode*-level switching headroom (large). The
  opportunity is real; the aggregate diagnostic just does not expose where it is.
* Reported as headroom, not as a result — the gate must decide without labels and
  the runs record gains but not the per-episode scores a gate would key on.

## Round 19 — cross-paper verification and count corrections

* Added `experiments/check_cross_paper.py`: the headline claims and five shared
  number sets must agree verbatim wherever two manuscripts report the same
  protocol, so a correction applied to one paper cannot silently miss another.
  It is folded into the main checker as an eleventh suite.
* Corrected two count errors it surfaced: the KBS abstract still said "thirteen
  competitors" (now fourteen) and the TKDE detailed matrix was captioned
  "fourteen-competitor" when that table lists **eleven**; the two comparison
  sets are now named explicitly wherever both are in play.
* Differentiation audit updated: **33 of 35 components are new (94%)**, against
  the 50% requirement, and the audit now points at the two scripts that verify
  it mechanically.

## Round 18 — external validity of the procedure, and its honest limit

* **R157**: the same pilot test re-run in the demonstration family by splitting
  **classes** (177 classes, five benchmarks, four contrasts). Pilot and
  deployment verdicts agree on **5/5 benchmarks** for every contrast and sign
  agreement is **1.000** throughout.
* The mechanism reproduces quantitatively: half-split verdict agreement collapses
  where the effect is marginal *relative to the pilot's resolution* — CIFAR-10
  (10 classes, Δ≈−.007) agrees only **31–34%** of the time, while the same
  effect size over CIFAR-100's 100 classes agrees **100%**. What governs is
  effect size against resolution, not domain or modality.
* **Stated limit**: every contrast in that family has the same sign, so 5/5
  confirms specificity and cannot test sensitivity — the family contains no
  positive case, because demonstrations carry labels rather than redundant
  predictions. The operating characteristics remain the traffic-family numbers.

## Round 17 — a high-precision self-audit for the pilot

* **R156**: splitting the pilot in half and requiring both halves to resolve in
  the same direction is a near-perfect filter. At 16 targets such pilots agree
  with the full-sample verdict **99.0%** of the time against **78.1%** for the
  rest; at 24 targets **99.9%** against **92.3%**.
* It certifies rather than produces: strict consistency holds for only 35% (16
  targets) and 39% (24 targets) of pilots. Requiring merely a shared *sign* is
  worth almost nothing (85.4% → 84.8% at 16 targets).
* Caveat stated: requiring both halves to resolve selects for large effects, so
  the check is partly a signal-strength filter — a marginal deployment yields an
  inconclusive check and the response is more labels.
* The procedure is now: split the pilot, act when both halves resolve the same
  way, enlarge otherwise; never substitute a cheaper proxy.

## Round 16 — the pilot survives a time shift (direction yes, confidence partly)

* **R155**: the six-dataset protocol re-run with the router and pooling scored
  on both the calibration period (pilot) and the test period (deployment) for the
  same 192 targets. The within-dataset correlation between the two differences
  is **`+.596` (`p<1e-19`)**, every dataset individually significant
  (`+.520`–`+.848`), per-target sign agreement **81%**.
* **The significance call is weaker across periods**: dataset-level verdict
  agreement is **4/6**, against the 96% same-period pilot of R154. Both misses
  keep the correct sign (one over-confident AHEAD on a marginal positive, one
  under-confident non-call on a negative), and **no pilot recommended routing
  where routing hurt**.
* Consequence for the prescription: the procedure is directionally robust to the
  time shift and its confidence is not — enlarge the pilot rather than abandon
  the procedure. The unmeasured caveat in the papers is replaced by these
  numbers.
* Supporting change: `--extra-split validation` added to the KBS multitarget
  router driver, scoring only the two policies the decision needs; the KBS suite
  (17 tests) passes.

## Round 15 — the protocol becomes a validated decision procedure

* **R154**: pilots of $n$ targets drawn from the six ridge-regime deployments,
  400 per dataset and size, scored by the same rule as the full sample. A pilot
  of **24 targets reproduces the full-sample verdict 96% of the time**, with
  **98% sensitivity and 96% specificity** for the operationally critical
  "route" call (16 targets: 84/86/93%). Not a base-rate artefact — half the
  deployments are positive, so a constant-route rule would score specificity 0.
* Direction is easy, significance is not: four targets get the sign right
  almost always while three-way agreement is only .64. Pilot cost is ~2 minutes
  in the ridge regime, ~35 minutes with the nonlinear backbone.
* Consequence: the papers now state the four-step procedure (label ~24 targets,
  run the protocol unchanged, compute the paired interval, route only if it
  excludes zero) and rule out the cheap substitutes with the negatives from
  R140 (probe), R153 (headroom) and Measurement 1 (screen).

## Round 14 — the headroom diagnostic is not predictive, in either regime

* **R153**: over the six-dataset ridge protocol (192 targets) the within-dataset
  correlation between structural headroom `H_state` and the router's advantage
  over pooling is **`+.040` (`p=.58`)**; per dataset it ranges `−.235` to `+.310`
  with no consistent sign. Across datasets `r=+.60` (`p=.21`, n=6) and the
  ordering is broken by PEMS03 — the highest headroom (`.119`) with an advantage
  of `+.0013`, against PEMS-BAY's similar `.115` and `+.0303`. Together with the
  regime-B value (`+.047`) this makes the negative span both regimes: **headroom
  is necessary and not predictive**.
* The two unresolved regime-A cases are **effect-size differences, not
  under-powered ones**: per-target standard errors are comparable across
  benchmarks (`.0008`–`.0059`) while the effect runs from `t=+5.1` (PEMS-BAY,
  94% of targets) to `t=−3.1` (PEMS08, 38%). PEMS03 at `t=+1.5` is a genuinely
  small effect.
* Consequence for the prescription: the diagnostic protocol should be read as
  measuring the **realized share**, which does discriminate (`.164`→`.024` on
  the ladder, `.10`–`.35` vs `.06` in regime A), not the opportunity.

## Round 13 — recent-method comparison completed, one theory prediction refuted

* **A fourteenth competitor (R152)**: a DELIFT-style rule (ICLR 2025 — submodular
  coverage over k-means representatives), the composition the survey implied was
  covered but which had never been run. It is **significantly below pooling on
  all six datasets** (paired deltas −.018 to −.023, every interval resolved), so
  the verdict does not change. The first implementation degenerated to
  `kmeans_representatives` (one cluster per budget slot, so the coverage stage
  never chose); the pool is now over-segmented, and a regression test pins the
  distinction.
* **A theory prediction tested and refuted (R151)**: as the anchor absorbs its
  residual, `g_A → 0` and the score collapses to `−‖d‖²`, so the least
  perturbing policy should win. It does not — `+.0026` and `+.0037` against
  pooling's `+.0133` and `+.0224`, significantly below on both, ranking 13th and
  16th of eighteen policies. The limit's *sign* is visible; the *policy* is
  worthless, because a score that only avoids harm cannot create value.
* The response-geometry baselines were listed in the survey as implemented but
  had no raw run on the neural expert; that gap is closed.

## Verification (round 12)

Two independent read-only audits of the two experiment sections were run against
the raw runs; ~44 claim clusters verified as reproducible, and every flagged
number is now corrected, rebuilt from raw, or given a named artifact
(`R147_CLAIM_AUDIT`, `R149_TABLE_VERIFICATION`). The corrections all moved
claims toward what paired intervals support: the six-dataset headline is three
significant wins, two unresolved and one loss; the masked-mean backbone result
is a tie at `q=8` rather than a win. `experiments/check_table_artifacts.py`
enforces the table/artifact agreement mechanically and currently reports
54/54 cells across four suites.

## Bottom line for the manuscript

The PAMI additions did not confirm uniform superiority; they *delimit* the
method. The paper now states: structural headroom is real and large, the
smooth-loss theory says response is the right observation, R-MUR captures the
opportunity when the predictor leaves a systematic residual (the ridge
protocol), and the opportunity shrinks toward unidentifiability as the fixed
predictor improves or the candidate set loses redundancy. That is a defensible
PAMI-level contribution, but it is a different paper from "our router wins
everywhere", and it should be the framing the authors choose deliberately.
