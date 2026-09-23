#!/usr/bin/env bash
# Refresh the KBS artifacts once the third multi-target seed finishes.
#
# The supervisor `finish_r080b_diagnostics.py` completes seed 303 for PEMS03
# and PEMSBAY in the background. This watcher waits for both datasets to reach
# 32 finished targets, then regenerates the result tables, Figure 3, the
# manuscript PDFs, and the per-figure submission copies.
#
# Usage: nohup bash scripts/refresh_after_third_seed.sh > /dev/null 2>&1 &
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/root/miniconda3/bin/python
LOG="$ROOT/refine-logs/R080b_third_seed_refresh.log"
BASE="$ROOT/results/raw/R080b_multitarget_rmur_s303_diag_20260921"
DEADLINE=$(( $(date +%s) + 14400 ))   # give up after four hours

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

log "watcher started"
while :; do
  done03=$(find "$BASE/PEMS03" -maxdepth 2 -name result.json 2>/dev/null | wc -l)
  donebay=$(find "$BASE/PEMSBAY" -maxdepth 2 -name result.json 2>/dev/null | wc -l)
  if [ "$done03" -ge 32 ] && [ "$donebay" -ge 32 ]; then
    log "seed 303 complete (PEMS03=$done03 PEMSBAY=$donebay); refreshing"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    log "deadline reached (PEMS03=$done03 PEMSBAY=$donebay); stopping without refresh"
    exit 0
  fi
  sleep 60
done

cd "$ROOT" || exit 1
if "$PY" scripts/build_kbs_result_tables.py >> "$LOG" 2>&1; then
  log "tables regenerated"
else
  log "table generation failed; see above"
fi

cd "$ROOT/paper/figures" || exit 1
if "$PY" generate_fig3_multitarget_delta.py >> "$LOG" 2>&1; then
  log "figure 3 regenerated"
else
  log "figure generation failed"
fi

cd "$ROOT/paper" || exit 1
if latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >> "$LOG" 2>&1; then
  log "main.pdf rebuilt: $(pdfinfo main.pdf | awk '/Pages/ {print $2}') pages"
else
  log "manuscript build failed"
fi

cp figures/fig1_conceptual.pdf submission/Fig1.pdf
cp figures/fig2_headroom.pdf submission/Fig2.pdf
cp figures/fig3_multitarget_delta.pdf submission/Fig3.pdf
cp figures/fig3_scaling.pdf submission/Fig4.pdf
cp figures/fig5_real_observability.pdf submission/Fig5.pdf
cp figures/fig4_frontier.pdf submission/Fig6.pdf
log "submission figures refreshed; done"
