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
| R020 | M2 | Gate N pool scaling | four-line novelty set | synthetic K=4..32 | recovery, duplicates | MUST | PILOT PASS | K=4/8/16/32 scaling; fixed-K redundancy sweep added |
| R021 | M2 | Gate N medium redundancy | four-line novelty set | synthetic K=8,r=.5 | recovery, duplicates | MUST | PILOT PASS | 3 seeds; MUR recovery .986 vs Static .800 |
| R022 | M2 | Gate N high redundancy | four-line novelty set | synthetic K=16,r=.75 | recovery, duplicates | MUST | PILOT PASS | 3 seeds; MUR recovery .947 vs Static .713 |
| R070 | M2 | Oracle ladder and formulation headroom | Relevance/MMR/Static/Oracle-Static/MUR/Oracle-Greedy | synthetic K=16,B=4,r=.0/.25/.5/.75 | gain, headroom, duplicate rate | MUST | PASS | five seeds; $H_{state}$ grows .000/.112/.215/.286; MUR exceeds Oracle-Static at r=.5/.75 |
| R071 | M2 | Candidate-count ranking diagnostic | Oracle ladder and state diagnostics | synthetic r=.5,K=4..64 | MAE, top-1, regret, headroom | MUST | PASS | three seeds; MAE stays small while top-1 falls from .566 to .094 at K=64 |
| R072 | M3 | MUR-Conservative confirmatory frontier | MUR, MUR-Conservative, free-threshold control | synthetic K=16,B=4,r=.5,harm=.25/.5 | gain, harm, contexts, positive-selection precision | MUST | PASS | 50 runs; calibrated and free-threshold frontiers are closely matched |
| R073 | M4 | Electricity oracle audit | Base, Oracle-Static, Oracle-Greedy, exact-delta MUR | UCI Electricity historical-context task | oracle gains, headroom, exact-delta gain | MUST | PASS / BOUNDARY | three seeds; $H_{state}=.0727\pm.0044$, exact-delta gain remains weak; no representation tuning |
| R074 | M5 | Traffic expert and oracle viability | Base, Oracle-Static, Oracle-Greedy | METR-LA multi-sensor traffic forecasting | oracle gain, state headroom | CONDITIONAL | PASS | five seeds; $H_{state}=.1095\pm.0227$ |
| R075 | M5 | Traffic full router comparison | Pool-all/Relevance/MMR/Static/MUR/Oracle ladder | METR-LA multi-sensor traffic forecasting | gain, harm, oracle gap | CONDITIONAL | BOUNDARY | five seeds; learned MUR gain -.0026 vs Static .0001 despite positive oracle headroom |
| R076 | M5 | Traffic utility observability audit | X1 cached features, X2 exact response summaries, X3 full trajectories | fixed METR-LA split/expert/pool | MAE, Spearman, AUC, top-1, regret | MUST | PASS / LIMITATION | five seeds; X2 improves Spearman .1637→.3905 and top-1 .1480→.2595; X3 adds little |
| FREEZE | -- | Experimental scope freeze | R070--R076 | all current tasks | final CI/figures/writing only | MUST | ACTIVE | no new methods, datasets, or routing variants after R076 |
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
