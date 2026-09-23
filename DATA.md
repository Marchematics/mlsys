# Data and artifact policy

This snapshot is the evidence base for the three manuscripts. It is deliberately
not a byte-for-byte copy of the working tree: what is committed is what a claim
can be checked against, and everything omitted is named below with the reason
and the way back.

## Committed

| kind | where | why it is evidence |
|---|---|---|
| raw run records | `PAMI_MUR/results/raw/*/`, `KBS_MUR/results/raw/*/` | `result.json` holds every policy's metrics; `episode_gains.npz` holds the per-episode gains that the significance tests and the gate study read |
| per-step diagnostic traces | `KBS_MUR/results/raw/R159_gate_*/**/diagnostics.npz` | the gate negative (R159) is computed from the `gate_*` arrays; this is the only run whose analysis needs them |
| derived artifacts | `PAMI_MUR/results/derived/`, `KBS_MUR/results/derived/` | every headline number in the manuscripts is read from one of these, with provenance fields naming the raw runs it came from |
| experiment records | `PAMI_MUR/refine-logs/` | one file per experiment: question, design, result, provenance, and the effect on the manuscripts |
| source | `PAMI_MUR/experiments/`, `KBS_MUR/scripts/`, `KBS_MUR/src/`, `tests/` | the runners, the analyses and the two checkers |
| manuscripts | `*/paper/` | LaTeX sources, figures and compiled PDFs |

## Not committed

| kind | size | reason | how to regenerate |
|---|---|---|---|
| datasets (`data/`, STAEformer format, METR-LA/PEMS CSVs) | ~1 GB | third-party, large | download from the original sources; `experiments/pami_traffic.py` documents the expected layout |
| cached CLIP ViT-L/14 features (`PAMI_MUR/data/vision_features/*.npy`) | up to 107 MB per file | over GitHub's 100 MB file limit, and regenerable | re-encode with the frozen encoder; the demo-selection runner writes them to that path |
| trained predictor checkpoints (`*.pt` inside raw runs) | ~180 MB | training by-products, not evidence: every claim is backed by the run's recorded metrics and per-episode gains | re-run the corresponding record's command, which retrains and re-records |
| most per-step diagnostic traces (`diagnostics.npz` outside R159) | ~1.6 GB | multi-megabyte traces whose analyses are already summarised in the derived artifacts | re-run the record's command |
| other raw runs in the working tree (83 exist under `KBS_MUR/results/raw`) | ~1.3 GB | not cited by any manuscript or artifact | re-run the record's command |

## Verifying without the omitted inputs

None of the two checkers or the test suites needs the omitted data: they read the
committed artifacts and re-parse the manuscripts. To confirm the claim discipline
from a fresh clone:

```bash
cd PAMI_MUR
python experiments/check_table_artifacts.py
python experiments/check_cross_paper.py
python -m pytest tests/ -q
```

Re-running the *experiments* does need the datasets and, for the neural runs, a
GPU.

## Provenance conventions

* Raw runs are immutable; a re-run writes a new timestamped `R<n>_<name>_<date>/`
  directory rather than overwriting one.
* Derived artifacts carry a `provenance` block naming the raw runs they were
  computed from, and the generator script lives in `experiments/`.
* One run is retained but must not be cited: `KBS_MUR/results/raw/R150_ridge_delfit_*`
  is the first, degenerate implementation of the DELIFT-style baseline, which
  reduced to `kmeans_representatives` exactly. It is kept as the record of the
  error; the citable run is `R152_ridge_delfit_v2_*`.
