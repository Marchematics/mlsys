#!/usr/bin/env python3
"""Merge multi-target R-MUR dataset summaries."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/raw/R080b_multitarget_rmur_s1_20260920_2340"
OUT = ROOT / "results/derived/R080b_multitarget_rmur_summary"
DATASETS = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    rows = ["dataset,targets,targets_positive_gain,targets_gain_over_static,target_fraction_positive_gain,target_fraction_gain_over_static,H_state_mean,H_state_median,R_oracle_mean,R_deploy_mean"]
    for dataset in DATASETS:
        path = BASE / dataset / "target_statistics.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        summary[dataset] = data
        rows.append(
            f"{dataset},{data['targets']},{data['targets_positive_gain']},{data['targets_gain_over_static']},"
            f"{data['target_fraction_positive_gain']:.4f},{data['target_fraction_gain_over_static']:.4f},"
            f"{data['H_state_mean']:.4f},{data['H_state_median']:.4f},"
            f"{data['R_oracle_mean']:.4f},{data['R_deploy_mean']:.4f}"
        )
    OUT.joinpath("summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    OUT.joinpath("summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
