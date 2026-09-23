#!/usr/bin/env bash
# R110 sweep launcher: waits for cached features and for a free GPU window,
# runs the demo-selection protocol, and retries incomplete cells.
#
#   run_r110_sweep.sh <run_dir> [max_attempts]
#
# The driver itself skips cells whose result.json already exists, so repeated
# invocations only recompute what is missing (for example after a transient
# CUDA out-of-memory from the co-tenant jobs on the shared GPU).

set -u
RUN_DIR=${1:?usage: run_r110_sweep.sh <run_dir> [max_attempts]}
MAX_ATTEMPTS=${2:-8}
PAMI=/root/icl_ess_threshold/PAMI_MUR
PY=/root/miniconda3/bin/python
FEATURES="$PAMI/data/vision_features"
DRIVER="$PAMI/experiments/demo_selection/run_r110_demo_selection.py"
BENCHMARKS="cifar10 cifar100 svhn eurosat dtd"
SEEDS="101,202,303"
MIN_FREE_MIB=${MIN_FREE_MIB:-2500}

mkdir -p "$RUN_DIR"
echo "[sweep] start $(date -Is) dir=$RUN_DIR" >> "$RUN_DIR/sweep.log"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  pending=""
  for benchmark in $BENCHMARKS; do
    if [ ! -f "$FEATURES/$benchmark/train_emb.npy" ] || [ ! -f "$FEATURES/$benchmark/test_emb.npy" ]; then
      pending="$pending $benchmark:features"
      continue
    fi
    complete=$(ls -d "$RUN_DIR/$benchmark"/seed*/result.json 2>/dev/null | wc -l)
    if [ "$complete" -lt 3 ]; then
      pending="$pending $benchmark:cells($complete/3)"
    fi
  done
  if [ -z "$pending" ]; then
    echo "[sweep] all cells complete $(date -Is)" >> "$RUN_DIR/sweep.log"
    break
  fi
  echo "[sweep] attempt $attempt pending:$pending $(date -Is)" >> "$RUN_DIR/sweep.log"

  # wait for a free-memory window on the shared GPU
  for _ in $(seq 1 60); do
    if [ ! -f "$FEATURES/dtd/test_emb.npy" ]; then
      sleep 20   # feature extraction owns the GPU; let it finish first
      continue
    fi
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${free_mib:-0}" -ge "$MIN_FREE_MIB" ]; then
      break
    fi
    echo "[sweep] waiting for GPU memory (free=${free_mib}MiB) $(date -Is)" >> "$RUN_DIR/sweep.log"
    sleep 30
  done

  OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$PY" "$DRIVER" --benchmarks cifar10,cifar100,svhn,eurosat,dtd --seeds "$SEEDS" \
    --out-root "$RUN_DIR" >> "$RUN_DIR/run.log" 2>&1
  echo "[sweep] driver exit=$? $(date -Is)" >> "$RUN_DIR/sweep.log"
  sleep 20
done

echo "[sweep] finished $(date -Is)" >> "$RUN_DIR/sweep.log"
for benchmark in $BENCHMARKS; do
  complete=$(ls -d "$RUN_DIR/$benchmark"/seed*/result.json 2>/dev/null | wc -l)
  echo "[sweep] $benchmark cells=$complete" >> "$RUN_DIR/sweep.log"
done
