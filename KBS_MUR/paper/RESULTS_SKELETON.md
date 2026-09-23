# Results section skeleton

## 6 Experiments

### 6.1 Protocol and evaluation quantities

State the frozen-expert protocol, chronological real-data splits, candidate
pool construction, context budget, and the four central quantities:
prediction gain, utility recovery, negative-transfer rate, and
$H_{\mathrm{state}}$. Define Oracle-Static and Oracle-Greedy before any result.

### 6.2 Redundancy creates state-conditioning headroom

Open with the low-redundancy control: Oracle-Static and Oracle-Greedy coincide
when candidates are distinct. Then report the monotone R070 increase in
$H_{\mathrm{state}}$. Close with the direct mechanism result: MUR exceeds
Oracle-Static at the two higher redundancy levels.

### 6.3 Can a learned router recover the headroom?

Use the relevance → MMR → Static Utility → Oracle-Static → MUR → Oracle-Greedy
ladder. Emphasize that Oracle-Static already has exact standalone utility.
Report learned recovery as the realization gap, without claiming universal
dominance.

### 6.4 Candidate-set growth creates a ranking bottleneck

Use R071. Separate pointwise MAE from top-1 accuracy and one-step regret.
Explain the large-pool decline as max-over-candidates selection difficulty.

### 6.5 Conservative stopping gives a gain--harm frontier

Use R072. Present calibrated stopping and free-threshold control together.
Describe calibration as threshold selection; do not present it as a new
candidate-ranking mechanism or a simultaneous adaptive certificate.

### 6.6 Real data contains structural headroom but learned routing can fail

Place Electricity and METR-LA in the main text. Report Oracle-Static,
Oracle-Greedy, Static Utility, and MUR in one table. State that both tasks have
nonzero oracle headroom while the learned cached router does not recover it.

### 6.7 What information is missing?

Use R076. The response-summary probe improves utility predictability; full
forecast trajectories add little. Interpret this as an information-interface
limitation of cached representations. The probe is diagnostic only.

---

> **Superseded (2026-09-21).** The submitted manuscript follows the structure
> in `sections/6_experiments.tex` with the tables in `tables/` and the figures
> mapped in `figures/SOURCES.md`. This file is retained as a planning record.
