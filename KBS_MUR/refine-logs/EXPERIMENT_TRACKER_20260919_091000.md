# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | unit contracts | MUR core | synthetic unit | tests | MUST | PASS | 12 tests green on 2026-09-19 |
| R002 | M0 | oracle headroom | Base/Pool/Oracle | synthetic smoke | loss, oracle gain | MUST | PASS | oracle gain .3977; mixed-sign states 1.0; duplicate ratio 0 |
| R003 | M0 | label/provenance audit | frozen expert cache | synthetic smoke | overlap, target access, calls | MUST | TODO | fail closed |
| R004 | M0 | calibration semantics | interval MUR | synthetic calibration | coverage by state source | MUST | TODO | no test tuning |
| R010 | M1 | Gate A/C seed 1 | Relevance/Static/MUR/interval | synthetic K=8,r=.5,h=.25 | recovery, NTR | MUST | PILOT PASS | point and calibrated policies |
| R011 | M1 | Gate A/C seed 2 | Relevance/Static/MUR/interval | synthetic K=8,r=.5,h=.25 | recovery, NTR | MUST | PILOT PASS | point and calibrated policies |
| R012 | M1 | Gate A/C seed 3 | Relevance/Static/MUR/interval | synthetic K=8,r=.5,h=.25 | recovery, NTR | MUST | PILOT PASS | point and calibrated policies |
| R020 | M2 | Gate N pool scaling | four-line novelty set | synthetic K=4..32 | recovery, duplicates | MUST | PILOT PASS | K=4/8/16/32 scaling; formal bootstrap pending |
| R021 | M2 | Gate N medium redundancy | four-line novelty set | synthetic K=8,r=.5 | recovery, duplicates | MUST | PILOT PASS | 3 seeds; MUR recovery .986 vs Static .800 |
| R022 | M2 | Gate N high redundancy | four-line novelty set | synthetic K=16,r=.75 | recovery, duplicates | MUST | PILOT PASS | 3 seeds; MUR recovery .947 vs Static .713 |
| R023 | M2 | K scaling | Static/Coverage/MUR | synthetic K=4..32 | recovery, latency | MUST | PILOT PASS | controlled scaling supports MUR gap; latency audit pending |
| R024 | M2 | MUR-full diagnostic | light/full | synthetic high-r | loss, calls, latency | NICE | PILOT PASS | synthetic pilot only; no deployability claim |
| R025 | M2 | ranking loss deletion | Huber vs Huber+rank | synthetic high-r | MAE, ranking, calibration | NICE | TODO | retain only if useful |
| R040 | M3 | fixed vs stop | MUR variants | synthetic B=1..8 | gain, NTR, contexts | MUST | TODO | 3 seeds |
| R041 | M3 | harm sweep | MUR variants | harmful=0..4 | Pareto metrics | MUST | TODO | 3 seeds |
| R042 | M3 | interval alpha | calibrated MUR | alpha=.05/.1/.2 | coverage, gain, stop | MUST | TODO | calibration only |
| R043 | M3 | deployed-state audit | MUR-light | validation rollouts | MAE ratio, coverage | MUST | TODO | determines roll-in |
| R044 | M3 | one roll-in | MUR-light | train/validation | coverage | CONDITIONAL | BLOCKED | run only if trigger fires |
| R060 | M4 | Electricity builder smoke | frozen ridge expert | four partitions | overlap, headroom | MUST | PASS | zero client/week split violations; oracle headroom positive |
| R061 | M4 | Electricity router smoke | Static/MUR-light/full | train/cal/test smoke | MSE, runtime | MUST | INCONCLUSIVE | split audit passes; larger, exact-delta, and same-client diagnostics still do not pass Gate D |
| R070 | M5 | Electricity confirm seed 1 | full baseline ladder | held-out test | all primary | MUST | TODO | frozen protocol |
| R071 | M5 | Electricity confirm seed 2 | full baseline ladder | held-out test | all primary | MUST | TODO | frozen protocol |
| R072 | M5 | Electricity confirm seed 3 | full baseline ladder | held-out test | all primary | MUST | TODO | frozen protocol |
| R080 | M6 | Traffic feasibility | Base/Static/MUR/Oracle | held-out sensors | headroom, loss | CONDITIONAL | BLOCKED | after all gates |
| R090 | M6 | Activity feasibility | Base/Static/MUR/Oracle | held-out subjects | headroom, F1 | CONDITIONAL | BLOCKED | after all gates |
