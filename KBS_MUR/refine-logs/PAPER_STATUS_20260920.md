# KBS-MUR status after 2026-09-20 finalization pass

## Completed
- R071 rerun with five seeds and saved prediction/truth/margin arrays.
- Post-hoc candidate-scaling audit:
  - normalized MAE (MAE / target SD): .403 -> .739
  - positive top-two margin: .220 -> .024
  - median max-error/margin ratio: 1.17 -> 3.40
  - strict top-1: .488 -> .093
  - tie-aware decision accuracy: .884 -> .147
  - one-step regret: .018 -> .067 at K=16, .049 at K=64
- Main Table 2 rewritten around normalized error and decision margin.
- Fig.1 redesigned as a narrative concept figure.
- Fig.2 simplified to remove full numeric annotations.
- Fig.3 regenerated as a four-panel margin/regret diagnosis.
- Fig.4 and Fig.5 restyled through the shared figures4papers-style module.
- Real-data Table 3 split into oracle, static-utility, cached-MUR, and
  exact-response diagnostic columns.
- Abstract, Introduction, Discussion wording tightened; no internal run IDs
  or development terms in the manuscript prose.
- Appendix/reproducibility material added as a separate `supplement.tex`.
- Reference set: 62 entries, all cited; DOI/URL audit recorded.
- Math claim audit recorded; corollary proof added to supplement.

## Current artifacts
- `paper/main.pdf` (23 pages)
- `paper/supplement.pdf` (10 pages)
- `paper/references.bib` (62 entries)
- `results/derived/R071_margin_audit/summary.json`

## Remaining
- ICLR vs KBS substantial-overlap audit.
- Final language pass over Abstract, Introduction, Conclusion.
- Camera-ready reference/page/venue check.

## Claim rebalancing pass
- Title changed to `Beyond Standalone Utility: Marginal-Utility Routing for Multi-Context Prediction`.
- Abstract rewritten around the central finding: auxiliary-context value is state-conditioned.
- Contributions reframed as decision object, structural headroom, learned realization, and observability principle.
- Fig.1 redesigned without result numbers or defensive slogans.
- Sections 6.6/6.7 retitled and rewritten as positive mechanism findings.
- Discussion rewritten as a two-stage design test; limitations concentrated in one final paragraph.
- Redundant defense removed from Method and Theory while preserving precise scope statements.

## R077 response-router gate
- R-MUR with compact response summaries and top-q screening was evaluated on
  METR-LA with five seeds.
- q=2 and q=4 pass positive-gain and gain-over-learned-static gates.
- The planned 20-30% oracle headroom recovery gate fails; full-pool response
  reranking is unstable.
- Six-benchmark expansion is paused until a ranking-calibrated response-aware
  objective is developed.

## R077b ranking-calibrated response router
- Added a utility-gap-weighted pairwise ranking loss on full states.
- Five-seed METR-LA q=4 gain: .0301 [.0195, .0406], versus Static Utility.
- Full-pool response reranking becomes stable (gain .0194, CI positive).
- Strict oracle headroom recovery still fails; utility recovery relative to
  Oracle-Greedy is 17.8% for q=4.
- Six-dataset expansion remains paused.

- Hard-negative top-q training did not improve the gate; full-state ranking
  calibration remains the best R-MUR variant.

## R079 six-dataset response router
- Unified subset ridge expert and standard STAEformer split files.
- R-MUR q=4 positive on 6/6 datasets, with seed-level 95% CI above zero.
- R-MUR q=4 exceeds Static Utility on 6/6 datasets with paired seed-level CI
  above zero.
- H_state is nonzero on all six datasets (.041 to .265).
- R_oracle ranges .201 to .607; q=4 uses 25% of candidate response calls.
- STAEformer subset-interface replication remains the next backbone step.

## R080a multi-target oracle headroom (five seeds)
- 32 fixed targets per dataset, six datasets, five seeds.
- 960 target-seed runs.
- Every run has positive H_state; positive-target fraction is 1.0 on every
  dataset.
- H_state means range .0778 to .1221 across datasets.

## Narrative restructuring
- Introduction rewritten as six paragraphs around one decision object.
- Related Work compressed to four continuous paragraphs with no subsection blocks.
- Section 3 now defines marginal utility and the two oracle policies.
- Section 4 rewritten as Response-Aware Marginal-Utility Routing.
- Section 5 compressed to the one-step theorem and a conditional corollary.
- Experiments reorganised into four subsections.
- Terminology cleaned: no development run IDs, X1/X2/X3, MUR-Conservative, or
  oracle development names in manuscript prose.
- Abstract rewritten as a single paragraph with no numerical results.
