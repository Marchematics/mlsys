# ICLR vs KBS substantial-overlap audit (2026-09-20)

## ICLR branch
- Title: `How Much Supervision Is Needed for Finite-Library In-Context Transfer?`
- Core object: task-reuse evidence and finite-library relation posterior.
- Contributions: Bayesian decomposition into relation evidence, deployment prior
  odds, and query-specific prediction gain; supervision law; prior correction;
  BF-Gate.
- Tasks: Gaussian regression, sinusoid regression, UCI Electricity clients, and
  finite Markov libraries.

## KBS branch
- Title: `Marginal-Utility Routing for Multi-Context Prediction: Structural
  Headroom and Utility Predictability`.
- Core object: set-conditioned marginal utility `m(j | A)`.
- Contributions: structural-headroom oracle gap; cached marginal-utility
  routing; normalized candidate-scaling audit; response observability probe.
- Tasks: controlled multi-context synthetic environment, UCI Electricity
  boundary audit, METR-LA oracle and cached-routing audits.

## Overlap assessment
- Shared vocabulary: auxiliary context, context selection, reference to
  few-shot prompting.
- Shared references in the inspected source: 4 low-level entries
  (`brown2020language`, `min2022rethinking`, `shazeer2017outrageously`,
  `zhou2022mixture`); no overlap in core novelty citations.
- Shared dataset: UCI Electricity.
  - ICLR uses it for client-level relation-transfer evaluation.
  - KBS uses it only as an oracle-headroom boundary audit with a different
    decision target, candidate construction, expert, and metric.
- No KBS use of finite-library posterior, balanced relation sampling, analytic
  prior correction, supervision law, BF-Gate, Gaussian action regimes, sinusoid
  experiments, or Markov sequence experiments.
- No ICLR use of `m(j | A)`, structural headroom, utility observability,
  candidate-scaling margin analysis, or METR-LA.

## Conclusion
The two manuscripts share infrastructure-level vocabulary and one real-data
boundary task. Their core problem, method, claims, and main experiments are
distinct. Electricity should remain framed as a boundary audit in KBS, with the
relation-transfer framing left to the ICLR manuscript.
