# KBS MUR

Project root for the two KBS-targeted manuscripts. They are different papers
and only one of them is an active submission candidate at a time.

## `paper/` — the KBS submission (frozen at R080)

**Response-Aware Marginal-Utility Routing for Multi-Context Prediction.** The
manuscript is frozen at the evidence that existed when the R080 runs closed:
the controlled mechanism studies, the multi-target oracle audit (960
target-run evaluations), and the multi-target learned results (576 target-run
evaluations, three runs). No further experiments enter this submission. See
`refine-logs/KBS_R080_FREEZE_20260921.md`.

Because the text follows the same research line as `../TKDE_MUR` and
`../../PAMI_MUR`, do not submit them concurrently; the KBS version is the
submission of record for this line unless the user decides otherwise.

## `paper_risk/` — alternative direction (not submitted)

**Risk-Controlled Context Selection.** The question is when a marginal-utility
estimate is reliable enough to act on: the router calibrates a harm budget on
deployment states by conformal risk control and abstains or stops when no
candidate can be certified. The manuscript, the analysis scripts
(`scripts/analyze_risk_control.py`, `scripts/analyze_risk_end_to_end.py`,
`scripts/build_risk_tables.py`), and the derived results are complete enough to
serve as an independent second paper if that direction is resumed. It is not
part of the current KBS submission.

## Evidence and scripts

- `results/raw/` immutable runs; `results/derived/` sourced summaries.
- `scripts/analyze_risk_control.py` builds the risk-control evidence from the
  saved deployment traces; `scripts/build_risk_tables.py` renders its tables.
- `scripts/build_kbs_result_tables.py` renders the MUR result tables.
- `refine-logs/KBS_VS_PAMI_DIFFERENTIATION.md` records how the two lines differ
  and what remains open.

No result table in this directory may contain estimated or placeholder numbers
presented as observations.
