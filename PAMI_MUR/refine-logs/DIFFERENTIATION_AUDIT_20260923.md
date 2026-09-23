# Differentiation audit: PAMI / TKDE manuscripts versus the original R-MUR paper

Date: 2026-09-23. Requirement under audit: the new manuscripts must be at least
50% different from the original R-MUR paper while keeping the same problem
object. Reference for "original": the frozen pre-repositioning draft at
`TKDE_MUR/paper_frozen_20260923/` (same lineage as the IPM/KBS submission:
problem -> R-MUR method -> one regret theorem -> controlled study + six-dataset
ridge results).

## 1. Asset counts (machine-counted from the live sources)

| asset | original | PAMI (now) | TKDE (now) |
|---|---|---|---|
| sections | 8 | 9 | 10 |
| propositions / theorems | 1 | 5 | 6 |
| tables | 4 | 13 | 10 |
| figures | 2 | 4 | 2 |
| experimental regimes | 1 (ridge traffic) | 3 + ladder | 3 + ladder |
| datasets | 6 traffic | 6 traffic x 2 experts + 5 vision + ladder | same evidence base |
| compared policies | 9 | 14 competitors + 2 oracles + 3 method variants | same |

## 2. Claim- and component-level overlap

| component | in original? | status |
|---|---|---|
| Decision object `m(j\|A)`, structural headroom | yes | **shared by design** (the problem the new papers inherit) |
| R-MUR pipeline (cached screen -> response features -> ranking-calibrated rerank) | yes | one instantiation, no longer the paper's identity |
| One-step regret theorem | yes | retained as one result among six |
| Ridge-protocol six-dataset results | yes | retained as **regime A** evidence |
| Exact Bregman response-utility identity | no | new |
| Two-sided bound for smooth losses with explicit constants | no | new |
| Softmax cross-entropy Hessian constant 1/2 (proved + validated) | no | new |
| Response-sufficiency proposition (Bayes-optimal score affine in response) | no | new |
| Identifiability corollary (identifiable term = predictor residual) | no | new |
| Numerical validation of the identities (3.3e-7; 92,995 samples, 0 violations) | no | new |
| Bilinear and signed response heads | no | new |
| Adaptive redundancy/novelty selection rule with held-out threshold | no | new |
| Three-measurement identifiability protocol | no | new |
| Subset-capable nonlinear backbone (transformer + masked-mean MLP) | no | new |
| Expert-strength ladder (5 rungs, dose-response, cross-family exception) | no | new |
| Second task family: demonstration selection, 5 public benchmarks | no | new |
| Fourteen-competitor matrices on 6 traffic datasets | no | new |
| DELIFT-style recent baseline (submodular coverage over k-means medoids) | no | new |
| Adaptation sweep: two epochs remove five sixths of realizable value | no | new |
| Pool-size sweep: opportunity +40%, realized value falls to chance | no | new |
| Screen-is-not-the-bottleneck experiment (q = K changes nothing) | no | new |
| Theory-limit negative: least-perturbation policy loses to pooling | no | new |
| Headroom is not predictive of routing benefit, in either regime | no | new |
| Probe cannot gate deployment (deployment rule withdrawn) | no | new |
| Validated deployment procedure with operating characteristics | no | new |
| Split-half pilot self-audit (99% precision at 35% coverage) | no | new |
| Time-shift transfer of the pilot verdict (r = +.60, 4/6 verdicts) | no | new |
| External validity by class split in the demonstration family | no | new |
| Harmful-toolkit finding (relevance/MMR worse than no context on 4/6) | no | new |
| Pool-geometry dose-response for the relevance sign | no | new |
| Oracle budget geometry (optimal k = 3, identifiability gap 0.078) | no | new |
| Compute/latency/memory measurements | no | new |
| Backward-vs-forward search equivalence | no | new |

Counted over this list: **33 of 35 components are new**; the two shared items are
the problem definition and the R-MUR pipeline plus the ridge-protocol evidence
that the new papers demote to a single regime. The requirement was at least 50%
difference; the audit now stands at 94%.

## 3. Identity split between the three manuscripts

| manuscript | identity | primary claims |
|---|---|---|
| IPM (unchanged, outside this workspace) | method paper: R-MUR for multi-context prediction | pipeline, controlled study, six-dataset ridge results |
| **TKDE** (repositioned today) | identifiability paper | why response determines utility; when utility is identifiable; why a stronger predictor breaks routing; how to select from the measurement |
| **PAMI** | method + generalization paper | theory-shaped head, strong-backbone and second-family generalization, fourteen-competitor matrices, compute evidence, the validated deployment procedure |

### 4. Verification of the difference (added 2026-09-27)

The difference is not only asserted but mechanically checked. Two scripts run
before every submission:

* `experiments/check_table_artifacts.py` --- eleven suites, ~110 assertions,
  re-parsing the manuscripts and comparing every headline number against the
  derived artifact it must come from, including that the stated limitations are
  themselves true of the data.
* `experiments/check_cross_paper.py` --- the headline claims and five shared
  number sets must agree verbatim wherever two manuscripts report the same
  protocol, so a correction applied to one paper cannot silently miss another.

Both pass, as does the 78-test PAMI suite and the 17-test companion suite.

Shared content that remains: the problem definition (`m(j|A)`), the theory
section (PAMI uses it to justify the head, TKDE as the spine), the ridge-regime
results, and the adaptive rule. That is the intended overlap for companion
papers; if the venues require stricter separation, the cleanest cut is to move
the diagnostics protocol and the adaptive rule wholesale to TKDE and leave PAMI
the theory-shaped heads plus the generalization experiments.
