# KBS submission package — Response-Aware Marginal-Utility Routing

This package contains the manuscript submitted to *Knowledge-Based Systems*,
its figures, and the experiment data and code that reproduce every reported
number. The evidence is frozen at the R080 stage: the controlled mechanism
studies, the multi-target oracle audit (960 target-run evaluations) and the
multi-target learned results (576 target-run evaluations over three runs),
together with the extended baseline comparison (576 additional evaluation-only
runs).

## Layout

| Path | Content |
|---|---|
| `paper/` | manuscript sources, PDFs, tables, figures, highlights, cover letter, checklist |
| `paper/main.tex` | the manuscript source, built in the review layout (1.5 spacing, line numbers) |
| `paper/main.pdf` (39 pp) | the submission manuscript |
| `paper/supplement.tex`, `paper/supplement.pdf` (11 pp) | supplementary material |
| `paper/sections/`, `paper/tables/` | section sources and generated table fragments |
| `paper/figures/` | figure scripts, vector/raster outputs, `SOURCES.md` mapping each figure to its run |
| `paper/submission/Fig1-5.pdf` | figures as separate upload files |
| `results/raw/` | the runs cited by the manuscript: `result.json`, aggregates, per-target records, per-state arrays where needed |
| `results/derived/` | sourced summaries used by the tables and figures |
| `scripts/` | experiment runners, table builders, figure builders, the packaging script |
| `src/mur/` | library: synthetic environment, traffic data, experts, router, calibration, metrics |
| `tests/` | unit tests for the library |
| `docs/` | freeze record and the submission-pass record |
| `MANIFEST.md` | artifact history of the project |

## Reproducing the paper

Everything is regenerated from `results/` by scripts; no number is transcribed
by hand.

```bash
# 1. result tables (writes paper/tables/*.tex with provenance comments)
python scripts/build_kbs_result_tables.py
python scripts/build_shortlist_tables.py     # decomposition, screening, calibration, Fig. 5

# 2. supplementary tables
python scripts/build_supplement_tables.py

# 3. figures (writes paper/figures/*.pdf and *.png)
cd paper/figures
for s in generate_fig1_overview.py generate_fig2_headroom.py \
         generate_fig3_multitarget_delta.py generate_fig4_mechanism.py; do
  python "$s"
done
cd ..

# 4. documents
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex

# 5. rebuild this package
bash scripts/build_submission_package.sh
```

Environment used: Python 3.12 (miniconda) with numpy, scikit-learn, torch and
matplotlib; TeX Live 2022 with `elsarticle` 3.3. Rebuilding the manuscript
artefacts needs no GPU; the original runs used one CUDA device.

The bibliography contains the cited references for the manuscript, including 17
recent *Knowledge-Based Systems* articles used for scholarly positioning and
ten records for the acquisition baselines and their neighbours. Metadata was
verified against Crossref and the arXiv API; see
`docs/REFERENCE_AUDIT_20260920.md`.

The baseline matrix covers fourteen deployable policies. Their runs come from
three run sets with the same protocol: the learned policies from the
`R080b` runs, the content-based subset rules from `R202`, and the fixed-set
acquisition rules from `R205`.

## Evidence chain

| Manuscript item | Source |
|---|---|
| Controlled redundancy ladder (Fig. 2 left, Table 1) | `results/raw/R070_oracle_ladder_k16_b4_rgrid_s5_20260919_1540/aggregate.json` |
| Candidate-scaling mechanism (Fig. 4 left) | `results/derived/R071_margin_audit/summary.json` |
| Multi-target oracle audit, 960 evaluations (Fig. 2 right, Table 1) | `results/raw/R080a_multitarget_oracle_s5_*/\<dataset\>_s*/records.json` |
| Multi-target learned results, 576 evaluations (Tables 2-3, Fig. 3) | `results/raw/R080b_multitarget_rmur_s{101,202,303}_*/\<dataset\>/target*_seed*/result.json` |
| Extended baseline comparison (Table 2) | `results/raw/R202_subset_baselines_s3_20260921_0426/` |
| Fixed-set acquisition baselines (Table 2, Fig. 3) | `results/raw/R205_fixed_set_baselines_s3_20260921/<dataset>/target*_seed*/result.json` |
| Regret decomposition, screening, calibration (Tables 4-6, Fig. 5) | `results/derived/R203_shortlist_diagnostics/summary.json` |
| Observability probe (Fig. 4 middle and right) | `results/raw/R076_traffic_utility_observability_s5_20260920_0300/summary.csv` |

## Verifying the package

`checksums.sha256` covers every file. The generated artefacts are
deterministic: tables are written from the run records, and the figure PDFs
have their creation timestamp pinned, so a rebuild reproduces them byte for
byte. From the extracted package:

```bash
sha256sum -c checksums.sha256 --quiet        # or: sha256sum -c checksums.sha256
python scripts/build_kbs_result_tables.py    # rewrites paper/tables/*.tex
python scripts/build_shortlist_tables.py
cd paper/figures && for s in generate_fig*.py; do python "$s"; done
```

The five tables and five figures then match their recorded checksums exactly.
Two things are intentionally *not* byte-stable: `results/derived/*` summaries
are not regenerated by these commands, and LaTeX output embeds timestamps.

## What is not included, and why

- `diagnostics.npz` files (about 830 MB): per-state arrays saved by the
  multi-target routing runs. The manuscript tables and figures are built from
  `result.json`, the per-run `episode_gains.npz` and the derived summaries in
  `results/derived/`, all of which are included. Re-running the routers with
  the documented commands regenerates the diagnostics.
- Runs from earlier development stages and from other research lines
  (Electricity boundary audit, single-target METR-LA probes, the controlled harm
  ladder of the alternative study). They are not cited by this manuscript; the
  packaging script `scripts/build_submission_package.sh` lists exactly which
  runs are copied.
- `paper_risk/`: an alternative research direction. It is a different paper and
  is not part of this submission.
- LaTeX intermediate files: rebuilt by `latexmk`; `main.bbl` and
  `main_review.bbl` are included so the bibliography resolves without BibTeX.

## Before uploading

`paper/SUBMISSION_CHECKLIST.md` lists the author-supplied items that remain:
author names and affiliations, ORCID, and the funding statement. The manuscript
carries no generative-AI declaration; that decision is recorded in the
checklist together with the policy reference. Highlights must be pasted into a
Word file named `Highlights` for the Editorial Manager portal.
