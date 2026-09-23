#!/usr/bin/env bash
# Build the KBS submission package from an explicit include list.
#
# Only runs and derived files cited by the manuscript or the supplement are
# copied; development-era runs from other research lines stay out.
#
# Usage: bash scripts/build_submission_package.sh [package-name]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:-kbs_r080_20260921}"
PKG="$ROOT/package/$NAME"
DATE="$(date +%Y%m%d)"

# --- evidence cited by the manuscript and supplement ------------------------
RAW_INCLUDE=(
  "R070_oracle_ladder_k16_b4_rgrid_s5_20260919_1540"        # controlled redundancy ladder
  "R071_candidate_size_r050_b4_s5_20260920_1730"            # candidate-scaling / margin audit
  "R072_mur_conservative_h025_h050_a005_a050_s5_20260920_0000"  # controlled calibration
  "R076_traffic_utility_observability_s5_20260920_0300"     # observability probe (Fig. 4)
  "R080a_multitarget_oracle_s5_20260920_2245"               # multi-target oracle audit
  "R080a_multitarget_oracle_s5_20260920_2300"
  "R080b_multitarget_rmur_s1_20260920_2340"                 # multi-target learned runs
  "R080b_multitarget_rmur_s202_diag_20260921"
  "R080b_multitarget_rmur_s303_diag_20260921"
  "R202_subset_baselines_s3_20260921_0426"                  # coverage / DPP / k-center baselines
  "R205_fixed_set_baselines_s3_20260921"                    # fixed-set acquisition baselines
)
DERIVED_INCLUDE=(
  "R071_margin_audit"
  "R080a_multitarget_summary"
  "R080b_multitarget_rmur_summary"
  "R080b_hierarchical_summary"
  "R203_shortlist_diagnostics"                              # screening / decomposition / calibration
)

rm -rf "$PKG"
mkdir -p "$PKG"/{results/raw,results/derived,docs}

rsync -a --exclude '__pycache__' --exclude '*.aux' --exclude '*.log' \
      --exclude '*.fls' --exclude '*.fdb_latexmk' --exclude '*.out' \
      --exclude '*.spl' --exclude '*.blg' --exclude '*.synctex.gz' \
      --exclude '*.pyc' "$ROOT/paper/" "$PKG/paper/"
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ROOT/scripts/" "$PKG/scripts/"
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ROOT/src/" "$PKG/src/"
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ROOT/tests/" "$PKG/tests/"
rsync -a "$ROOT/configs/" "$PKG/configs/" 2>/dev/null || mkdir -p "$PKG/configs"

for name in "${RAW_INCLUDE[@]}"; do
  if [ -d "$ROOT/results/raw/$name" ]; then
    rsync -a --exclude 'diagnostics.npz' "$ROOT/results/raw/$name" "$PKG/results/raw/"
  else
    echo "warning: missing run $name" >&2
  fi
done
for name in "${DERIVED_INCLUDE[@]}"; do
  if [ -d "$ROOT/results/derived/$name" ]; then
    rsync -a "$ROOT/results/derived/$name" "$PKG/results/derived/"
  else
    echo "warning: missing derived set $name" >&2
  fi
done

cp "$ROOT/MANIFEST.md" "$PKG/"
cp "$ROOT/refine-logs/REFERENCE_AUDIT_20260920.md" "$PKG/docs/"
cp "$ROOT/paper/PACKAGE_README.md" "$PKG/README.md"
cp "$ROOT/refine-logs/KBS_R080_FREEZE_20260921.md" "$ROOT/refine-logs/KBS_SUBMISSION_PASS_20260921.md" "$PKG/docs/"

(cd "$PKG" && find . -type f ! -name 'checksums.sha256' -print0 | sort -z | xargs -0 sha256sum > checksums.sha256)

(cd "$ROOT/package" && rm -f "kbs_r080_submission_$DATE.tar.gz" "kbs_r080_submission_$DATE.zip" \
  && tar -czf "kbs_r080_submission_$DATE.tar.gz" "$NAME" \
  && zip -qr "kbs_r080_submission_$DATE.zip" "$NAME")

echo "package: $PKG"
du -sh "$PKG"
ls -lah "$ROOT/package/kbs_r080_submission_$DATE".*
