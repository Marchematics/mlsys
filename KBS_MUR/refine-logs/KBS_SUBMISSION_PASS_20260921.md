# KBS submission pass (2026-09-21)

Scope: bring `KBS_MUR/paper` to a KBS submission-ready state after the
TKDE/PAMI split. No new method, dataset, backbone, or loss was introduced.

## What changed

### Evidence pipeline

- `scripts/build_kbs_result_tables.py` (new) generates every result table in
  the manuscript directly from immutable runs and writes one LaTeX fragment
  per table into `paper/tables/`. Each fragment carries a provenance comment.
- `paper/figures/multitarget_data.py` (new) resolves the completed multi-target
  seeds at figure time; `generate_fig3_multitarget_delta.py` now plots the
  six-dataset deployable comparison plus the target-level contrast against both
  baselines.
- Figure scripts were resized to the Elsevier text block (390 pt = 5.4 in) so
  that labels print at 7 pt or larger. Fig. 1 was restacked vertically and the
  retired Electricity panel was removed from the observability figure.

### Manuscript

- Section 6 restructured into protocols, structural value, multi-target
  benchmark, response budget and component ablation, and routing-quality
  analysis. Five generated tables and six figures are now referenced from the
  narrative.
- New evidence added: deployable-method matrix with average rank and
  negative-transfer rate, multi-target consistency with target-level bootstrap
  intervals for both contrasts, and a shortlist-size ablation with the
  response-evaluation share.
- Terminology pass: `MUR` to `R-MUR`, `MUR-Conservative` to calibrated
  stopping, `MUR-rollout` to R-MUR rollout, also in the generated supplement
  tables.
- Abstract and Introduction now use the auxiliary-knowledge and knowledge-reuse
  framing that the journal screens for, and the closing paragraph states the
  decision-support role of the router.
- Float placement parameters were relaxed so figures print next to their
  references instead of at the end of the document.

### Submission package

- `paper/highlights.txt` (five bullets, all within the 85-character limit).
- `paper/cover_letter.md` (draft; decision-support contribution stated first).
- `paper/SUBMISSION_CHECKLIST.md` (per-requirement status and TODO list).
- `paper/figures/SOURCES.md` (figure to script to raw-run mapping).
- Declarations added to `main.tex`: CRediT, competing interest, data
  availability, generative AI, funding. Author-specific fields remain marked
  with TODO placeholders and are not invented.

## Verified numbers

All prose numbers in Section 6 were checked against the generated tables and
their raw sources before this pass:

- redundancy ladder and headroom (.286, .2145, .1121, .0000);
- real-data headroom: 960 target-run evaluations, all positive, mean .078--.122;
- multi-target contrasts: both deltas positive on every dataset, target-level
  bootstrap intervals above zero;
- shortlist ablation means .0346, .0422, .0443, .0435;
- candidate-scaling margin audit and observability probe values;
- gain--harm frontier values.

## Open items

1. Third multi-target seed (303) is still completing for PEMS03 and PEMSBAY
   under `refine-logs/R080b_diagnostic_completion.log`. When it finishes, run
   `scripts/build_kbs_result_tables.py` and
   `paper/figures/generate_fig3_multitarget_delta.py`, then rebuild; the
   captions and Figure 3 pick up the extra run automatically. The manuscript
   prose does not hard-code the run count.
2. Author names, affiliations, ORCID, repository URL, funding, and the
   generative-AI tool name must be filled in before upload.
3. Highlights must be pasted into a Word file named `Highlights` for the
   Editorial Manager portal.

## Independent consistency review and fixes

An independent read-only review of the manuscript against the generated tables
and raw runs produced the following findings, all of which were fixed:

| Finding | Fix |
|---|---|
| Deployable table's negative-transfer column used only the first run while the caption promised a multi-run mean | `build_kbs_result_tables.py` now averages the rate over runs per target; regenerate the table |
| Figure 3 aggregated three ragged seeds while the tables used the two complete seeds | `figures/multitarget_data.py` resolves the seeds that completed every dataset, so figure and tables agree |
| Prose quoted .391 and .260 for the observability probe; the five-run means are .390 and .259 | prose corrected |
| The shortlist claim "improves over both baselines on five of six datasets" was not checkable from any table | the ablation table now also prints the sensor-level Standalone Utility and cached-only baselines, and Section 6.1 defines that protocol |
| "METR-LA has the widest target-level variation" is false | the Discussion now reports that METR-LA converts the least headroom (lowest oracle recovery, fewest positive targets) and that PEMS-BAY has the widest spread; verified from the per-target arrays |
| Gain--harm figure caption described the trade-off backwards | caption now states that raising the calibration level increases both gain and harm, and that the highest level nearly matches the uncalibrated gain at lower harm |
| Data-availability statement claimed the supplementary material contains code | reworded to "available from the authors", with a repository link to be registered |
| "Predictor response determines ...", "rather than the capacity of the router" overclaimed | reworded to association and to a statement about the interface, which is what the probe varies |
| Electricity material in the supplement was not cross-referenced from the main text | Section 6.1 now names it as a client-load boundary audit reported in the supplement |
| Supplement said "three independent runs" and "means and standard deviations" | both corrected |
| Probe tables used `x1/x2/x3` while the text uses $\mathcal{X}_1..\mathcal{X}_3$; `MAE`/`AUC` unexpanded | generated tables and captions unified and expanded |
| `figures/latex_includes.tex` was stale (bare `MUR`, old five-figure set) | rewritten for the final six-figure set with R-MUR wording |
| Figure legends said "Static Utility" while the text says "Standalone Utility" | figures 1 and 2 relabelled and regenerated |

Every number in the reviewed list was re-checked against its source; the
remaining values were confirmed correct.
