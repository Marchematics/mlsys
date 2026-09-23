# Final KBS figure plan

All data figures use the shared `figures4papers_style.py` module adapted from
[ChenLiu-1996/figures4papers](https://github.com/ChenLiu-1996/figures4papers):
semantic blue/green/red palette, sans-serif type, minimal spines, print-safe
bar edges, and paired PDF/PNG export. Figure types follow the repository demos:
concept schematic (Fig.1), trend panels (Fig.2-3), frontier/scatter (Fig.4),
and grouped bars for quantitative comparison (Fig.5).

The main figures carry one claim each. Results text follows the same order.

## Figure 1 — From structural headroom to observability

**Claim:** Auxiliary-context value is state-conditioned; structural headroom
and observability govern its realization.

- **Panel A:** one candidate pool before selection; standalone utility ranks a
  redundant candidate above a complementary candidate.
- **Panel B:** after selecting the first candidate, the redundant candidate's
  true marginal collapses while the complementary candidate remains valuable.
- **Panel C:** a two-stage schematic: Oracle-Static versus Oracle-Greedy
  defines structural headroom; observable-response information determines
  whether a router can recover it.

This is the conceptual opening figure. It is a manual schematic, not an
architecture diagram.

## Figure 2 — Redundancy creates state-conditioning headroom

**Claim:** The gap between perfect standalone ranking and perfect sequential
ranking grows with redundancy.

- **Panel A:** duplicate ratio versus $H_{\mathrm{state}}$.
- **Panel B:** duplicate ratio versus Static Utility, Oracle-Static, MUR, and
  Oracle-Greedy gains.
- **Panel C:** duplicate-selection rate for Static Utility and MUR.

R070 is the source. The primary anchors are $H_{\mathrm{state}}=0,.112,.215,.286$.

## Figure 3 — Candidate growth compresses the decision margin

**Claim:** Larger candidate pools shrink the top-two utility margin. Absolute
pointwise error improves while error normalized by target scale and by the
decision margin grows, so argmax selection becomes unreliable.

- **Panel A:** candidate count versus normalized utility error (MAE / target SD).
- **Panel B:** candidate count versus maximum error divided by the positive
  top-two margin.
- **Panel C:** strict top-1 accuracy and tie-aware decision accuracy.
- **Panel D:** candidate count versus one-step regret.

The candidate-scaling run is the source. Strict top-1 counts an exact tie among
equally useful candidates as an error; decision accuracy counts a selection as
correct when it attains the best true utility.

## Figure 4 — Conservative stopping controls gain and harm

**Claim:** Calibration supplies a data-derived stopping threshold that moves a
router along a gain--harm frontier.

- **Panel A:** gain versus negative-transfer rate for MUR, five calibrated
  levels, and the free-threshold control at harmful fraction .25.
- **Panel B:** the same frontier at harmful fraction .50.
- **Panel C:** average selected contexts versus calibration level.

R072 is the source. MUR-Conservative remains a stopping analysis, not a second
ranking method.

## Figure 5 — Real-data headroom and utility predictability

**Claim:** Real tasks can contain structural headroom that a cached router does
not recover; exact response summaries identify the missing information.

- **Panel A:** Oracle-Static and Oracle-Greedy gains for Electricity and
  METR-LA, with $H_{\mathrm{state}}$ annotated for each task.
- **Panel B:** METR-LA comparison of Oracle-Static, Oracle-Greedy, Static
  Utility, and MUR; the learned policies remain near zero while oracle
  headroom is large.
- **Panel C:** R076 probe ladder: X1 cached features, X2 plus response
  summaries, X3 plus full trajectories; Spearman, positive AUC, and top-1
  accuracy.
- **Panel D:** R076 one-step regret for X1--X3.

R073--R076 are the sources. Traffic is a boundary for the current cached
  router, not a hidden appendix failure.

## Results order

1. Protocol and evaluation quantities.
2. Structural headroom under redundancy.
3. Learned recovery of that headroom.
4. Candidate-set ranking bottleneck.
5. Conservative stopping frontier.
6. Real-data structural headroom and learned-routing limits.
7. Utility observability probe.

## Final figure set (2026-09-21 submission pass)

The plan above predates the six-dataset evidence. The submitted manuscript
uses the following mapping; each figure is generated from the raw runs listed
in `figures/SOURCES.md`.

| Manuscript figure | File | Claim |
|---|---|---|
| Fig. 1 | `fig1_conceptual.pdf` | Auxiliary knowledge is state-conditioned; headroom and observability govern its realization. |
| Fig. 2 | `fig2_headroom.pdf` | Redundancy creates structural headroom, and the learned router exceeds the standalone oracle at high redundancy. |
| Fig. 3 | `fig3_multitarget_delta.pdf` | R-MUR leads the deployable methods on six benchmarks and beats both baselines on almost every target. |
| Fig. 4 | `fig3_scaling.pdf` | Candidate-pool growth compresses the decision margin, so selection accuracy falls while pointwise error improves. |
| Fig. 5 | `fig5_real_observability.pdf` | Compact predictor-response summaries make marginal utility observable; full trajectories do not add much. |
| Fig. 6 | `fig4_frontier.pdf` | Calibrated stopping traces a gain--harm frontier. |

Panels from the earlier plan that are no longer in the main text
(Electricity headroom, METR-LA router ladder) remain reproducible from their
scripts and raw runs but are not part of the submission package.
