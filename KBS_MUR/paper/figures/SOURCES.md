# Figure sources (final R080 freeze)

Four figures, all generated from immutable runs. Names keep their original
numbering, so the manuscript numbers figures by order of appearance.

| Manuscript figure | Script | Output | Source |
|---|---|---|---|
| Fig. 1 | `generate_fig1_overview.py` | `fig1_overview.pdf` | schematic; no data |
| Fig. 2 | `generate_fig2_headroom.py` | `fig2_headroom.pdf` | left: `results/raw/R070_oracle_ladder_k16_b4_rgrid_s5_20260919_1540/aggregate.json`; right: `results/raw/R080a_multitarget_oracle_*/<dataset>_s*/records.json` |
| Fig. 3 | `generate_fig3_multitarget_delta.py` | `fig3_multitarget_delta.pdf` | `results/raw/R080b_multitarget_rmur_s{101,202,303}_*/<dataset>/target*_seed*/result.json` for the five learned policies, `results/raw/R202_subset_baselines_s3_*/<dataset>/target*_seed*/result.json` for the four content-based ones (nine deployable policies in total, matching the baseline matrix) |
| Fig. 4 | `generate_fig4_mechanism.py` | `fig4_mechanism.pdf` | left: `results/derived/R071_margin_audit/summary.json`; middle and right: `results/raw/R076_traffic_utility_observability_s5_20260920_0300/summary.csv` |
| Fig. 5 | `../scripts/build_shortlist_tables.py` | `fig5_diagnostics.pdf` | `results/derived/R203_shortlist_diagnostics/summary.json` |

Notes

- `multitarget_data.py` resolves the completed runs for Fig. 3 and reproduces
  the same seed set as the manuscript tables.
- Figures are sized for the 390 pt Elsevier text block (5.4 in wide) so that
  labels print at 7 pt or larger without rescaling.
- Scripts whose output is no longer in the main text (`generate_fig1_conceptual.py`,
  `generate_fig3_scaling.py`, `generate_fig4_frontier.py`,
  `generate_fig5_real_observability.py`) are retained as the source of the
  supplementary material and of earlier drafts; they are not used by the
  manuscript build.
- The shared style module `figures4papers_style.py` follows the conventions of
  <https://github.com/ChenLiu-1996/figures4papers>.
