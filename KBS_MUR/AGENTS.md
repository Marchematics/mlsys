# KBS MUR project instructions

## Pipeline Status

- language: zh
- target_venue: Knowledge-Based Systems
- project: Marginal Utility Router (MUR)
- source_boundary: This is a new paper, not a reformatted ICLR manuscript.

## Scientific boundary

1. The core object is set-conditioned marginal predictive utility
   `m(j | A)`, not task collision, finite-library prior odds, or BF-Gate.
2. Do not import the ICLR paper's `M`, collision posterior, prior correction,
   supervision law, title, abstract, figures, tables, or main claims.
3. Reuse low-level loaders, encoders, prediction experts, metrics, logging, and
   plotting infrastructure only when their semantics remain valid.
4. Never claim empirical superiority before the corresponding run and audit.
5. Keep the prediction expert fixed while generating counterfactual utility
   labels unless a separate experiment explicitly studies joint training.
6. Raw runs are immutable. Every derived artifact records its source run IDs.

## Publication-language boundary

1. The manuscript abstract is one continuous paragraph and contains no
   numerical results, dataset counts, version identifiers, run identifiers, or
   implementation labels.
2. Explain a technical term at first use. Prefer “the value a candidate adds
   to the selected context set” before introducing `m(j | A)`; define
   “negative transfer”, “context budget”, and “utility calibration” in prose.
3. Keep internal labels out of publication prose: `R001`, `R021`, `Gate N`,
   `expert-train`, `router-label`, `MUR-light`, `MUR-full`, GPU model names,
   and commit/version strings belong only in experiment records.
4. Expand or replace abbreviations on first use. `NTR`, `MAE`, `LCB`, `UCB`,
   and `MMR` do not appear in the abstract; the main text spells them out
   before using a short form in tables.
5. Do not write about Traffic, Activity, or any other expansion task as an
   observed result until its raw run and audit exist. The current evidence is
   limited to the controlled synthetic pilot and the existing Electricity
   infrastructure.
6. Do not create publication claims, supplementary files, declarations, or
   data descriptions that are absent from the verified project record.
7. Use short declarative sentences. Each paragraph has one job. Avoid slogan-
   like claims, defensive contrasts, and chained caveats.

## Method discipline

- One dominant contribution: sequential context selection by set-conditioned
  outcome utility.
- Primary model: shared context encoder, permutation-invariant selected-set
  encoder, and one marginal-utility head.
- Do not add uncertainty heads, contrastive branches, auxiliary relation
  classifiers, or RL unless a failed decision gate proves they are necessary.
- Separate harm-safe selection from completeness-safe stopping; do not state a
  stopping theorem stronger than the estimator error bound supports.

## Execution

- Run unit tests and a CPU smoke test before any GPU sweep.
- The first empirical gates are synthetic A/B/C and Electricity D.
- Stop the project if outcome utility does not beat relevance, or if
  set-conditioning does not beat static utility under controlled redundancy.
