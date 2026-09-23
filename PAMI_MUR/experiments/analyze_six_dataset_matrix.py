#!/usr/bin/env python3
"""Six-dataset competitor matrix in the ridge regime.

Merges the frozen six-dataset R-MUR evidence (KBS_MUR R080b runs: learned
routers and oracles) with the training-free competitor runs produced here
(R125 for METR-LA/PEMS-BAY, R133 for PEMS03/04/07/08), on identical targets and
seeds, and reports one table with target-level intervals per dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KBS_RAW = ROOT / "KBS_MUR" / "results" / "raw"
sys.path.insert(0, str(ROOT / "KBS_MUR" / "scripts"))

from run_traffic_response_router_sweep import paired_ci  # noqa: E402

POLICIES = [
    "ranked_response_q4", "pool_all", "cached_mur", "static_utility",
    "mutual_information", "random", "kmeans_representatives", "kcenter",
    "facility_location", "relevance", "dpp", "mmr",
]


def load_per_target(root: Path) -> dict[int, dict[int, dict[str, float]]]:
    out: dict[int, dict[int, dict[str, float]]] = {}
    if not root.exists():
        return out
    for cell in sorted(root.glob("target*_seed*")):
        result = cell / "result.json"
        if not result.exists():
            continue
        record = json.loads(result.read_text(encoding="utf-8"))
        target = int(record.get("target_sensor", cell.name.split("_")[0].replace("target", "")))
        seed = int(record.get("seed", cell.name.split("_")[1].replace("seed", "")))
        metrics = record.get("metrics") or {
            k: v for k, v in record.get("policy_metrics", {}).items()
        }
        if not metrics:
            continue
        out.setdefault(target, {})[seed] = {
            k: float(v["mean_prediction_gain"]) for k, v in metrics.items()
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--r125-ts", required=True)
    parser.add_argument("--r133-ts", required=True)
    args = parser.parse_args()

    learned_roots = {
        ds: [KBS_RAW / f"R080b_multitarget_rmur_s1_20260920_2340" / ds,
             KBS_RAW / f"R080b_multitarget_rmur_s202_diag_20260921" / ds,
             KBS_RAW / f"R080b_multitarget_rmur_s303_diag_20260921" / ds]
        for ds in ("METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08")
    }
    competitor_roots = {
        "METRLA": ROOT / f"PAMI_MUR/results/raw/R125_ridge_baselines_METRLA_{args.r125_ts}",
        "PEMSBAY": ROOT / f"PAMI_MUR/results/raw/R125_ridge_baselines_PEMSBAY_{args.r125_ts}",
        **{ds: ROOT / f"PAMI_MUR/results/raw/R133_ridge_baselines_{ds}_{args.r133_ts}"
           for ds in ("PEMS03", "PEMS04", "PEMS07", "PEMS08")},
    }

    report = {"datasets": {}}
    for dataset in learned_roots:
        cells: dict[int, dict[int, dict[str, float]]] = {}
        for root in learned_roots[dataset]:
            for target, seeds in load_per_target(root).items():
                cells.setdefault(target, {}).update(seeds)
        for target, seeds in load_per_target(competitor_roots.get(dataset, Path("/nonexistent"))).items():
            for seed, metrics in seeds.items():
                cells.setdefault(target, {}).setdefault(seed, {}).update(metrics)
        targets = sorted(t for t, seeds in cells.items() if seeds)
        if not targets:
            continue
        entry = {"targets": len(targets), "cells": int(sum(len(cells[t]) for t in targets)), "policies": {}}
        for policy in POLICIES:
            per_target = []
            for target in targets:
                values = [m[policy] for m in cells[target].values() if policy in m]
                if values:
                    per_target.append(float(np.mean(values)))
            if not per_target:
                continue
            interval = paired_ci(np.asarray(per_target))
            entry["policies"][policy] = {
                "mean": float(np.mean(per_target)),
                "low": interval["low"],
                "high": interval["high"],
                "targets": len(per_target),
                "positive_targets": int(np.sum(np.asarray(per_target) > 0)),
                "scored_harmful": bool(interval["high"] < 0.0),
                "scored_helpful": bool(interval["low"] > 0.0),
            }
        report["datasets"][dataset] = entry

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "six_dataset_matrix.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    header = f"{'policy':24s}" + "".join(f"{ds:>11s}" for ds in report["datasets"])
    print(header)
    for policy in POLICIES:
        row = f"{policy:24s}"
        for ds, entry in report["datasets"].items():
            value = entry["policies"].get(policy)
            row += f"{value['mean']:+11.5f}" if value else f"{'--':>11s}"
        print(row)
    print()
    for ds, entry in report["datasets"].items():
        harmful = [p for p, v in entry["policies"].items() if v["scored_harmful"]]
        helpful = [p for p, v in entry["policies"].items() if v["scored_helpful"]]
        print(f"  {ds:8s} targets {entry['targets']:3d}  harmful: {', '.join(harmful) or 'none'}")
        print(f"  {'':8s}                helpful: {', '.join(helpful) or 'none'}")


if __name__ == "__main__":
    main()
