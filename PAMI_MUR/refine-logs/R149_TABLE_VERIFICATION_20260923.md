# R149 — Mechanical table verification, and closure of the round-11 audit

**Date:** 2026-09-23
**Script:** `experiments/check_table_artifacts.py` (run before every submission)

## What the checker does

Re-parses the manuscripts and compares each printed table cell against the
derived artifact it must come from, failing on any mismatch. Four suites:

| suite | source of truth | result |
|---|---|---|
| `tab:deployable` (PAMI + TKDE, values and average rank) | `results/derived/R134_six_dataset_matrix/six_dataset_matrix.json` | **24/24** |
| `tab:headroom-real` (PAMI + TKDE) | `results/derived/R134_six_dataset_matrix/headroom.json` | **12/12** |
| `tab:consistency` (PAMI + TKDE vs the generated KBS table) | `KBS_MUR/paper/tables/tab_consistency.tex` | **12/12** |
| six-dataset paired verdicts quoted in PAMI | `results/derived/R146_six_dataset_paired/paired.json` | **6/6** |

The checker found two bugs in itself on first run (a two-decimal rank tolerance
and the papers' no-leading-zero convention); both were fixed, and neither
corresponded to a manuscript error.

## Audit items closed this round

**Rebuilt from raw.** `tab:headroom-real` had no artifact and claimed five
seeds / 960 evaluations, but every six-dataset run on disk uses seeds
101/202/303. Rebuilt over the 576 available cells
(`.../R134_six_dataset_matrix/headroom.json`), framing corrected to three
seeds. The substantive claim is unchanged and now exact: **576 of 576 cells
positive**, minimum `.0101`, means `.076`–`.119`.

**Resolved by provenance in the sibling tree.** The round-11 audits were scoped
to `PAMI_MUR/results/`; most of the evidence is in `KBS_MUR/results/`:

* `tab:redundancy` — **exact on all twelve cells** from
  `KBS_MUR/results/raw/R070_oracle_ladder_k16_b4_rgrid_s5_20260919_1540/aggregate.json`,
  which genuinely uses five seeds (101/202/303/404/505), so that table's "five
  seeds" was correct all along.
* K-growth diagnostics — **exact** from
  `KBS_MUR/results/derived/R071_margin_audit/summary.json`: `nmae_sd`
  `.4029→.7391` (K=4→64), positive top-two margin `.2204→.0236`, decision
  accuracy `.8840→.1474`.
* Response-feature comparison — **exact** from
  `KBS_MUR/results/raw/R076_traffic_utility_observability_s5_20260920_0300`
  (five-seed mean): Spearman `.1637→.3905`, AUC `.6492→.8342`, top-1
  `.1479→.2595`, regret `.0736→.0569`.
* Gain–harm frontier — the uncalibrated endpoint (`.273`/`.032`) is in
  `R204_harm_ladder_summary`; the sentence's second endpoint (".267 and .0281")
  matched **no** artifact and was replaced with the artifact's calibrated values
  (`.072`/`.0001` at harmful fraction `.25`, `.0006`/`.0012` at `.50`/`.75`).
* Per-target Pearson `.91` and `+.05` — no artifact existed because the
  analyzer never computed them. Recomputed over the sixteen R103 targets:
  `+.912` (`p<10⁻⁴`) against oracle gain and `+.047` (`p=.86`) against
  `H_state`, both as quoted. New artifact
  `results/derived/R148_per_target_correlation.json`; the recovery figures in
  the same sentence were aligned to that run (`.107` vs `.076`).

**Aligned to the generator.** `tab:consistency` had drifted: ΔStatic/ΔCached
read `.0141/.0133` and `.0316/.0275` where the generated table gives
`.0123/.0104` and `.0285/.0272`, "positive targets" read `.750` against `.812`,
and recovery was uniformly high (`.102` vs `.099`). PAMI and TKDE now carry the
generated values and the checker enforces agreement.

**One convention everywhere.** The same METR-LA router-versus-pooling contrast
appeared with two different intervals in TKDE. All six METR-LA contrasts were
restated under target-level pairing:

| policy vs pool-all (METR-LA, 16 targets) | paired Δ | 95% CI |
|---|---|---|
| response-aware `q=4` | −.0036 | [−.0110, +.0067] |
| response-aware `q=8` | −.0025 | [−.0093, +.0075] |
| cached-only router | −.0057 | [−.0115, +.0020] |
| stand-alone utility | −.0064 | [−.0115, −.0009] |
| MMR | −.0057 | [−.0089, −.0022] |
| relevance | −.0062 | [−.0093, −.0029] |

## State

Every number the two round-11 audits flagged is now corrected, rebuilt from
raw, or backed by a named artifact, and the check is mechanical rather than
manual. Both manuscripts compile with no undefined references (PAMI 19 pp,
TKDE 16 pp), KBS 45 pp, and the 63-test suite passes.
