#!/usr/bin/env python3
"""Realized versus structural headroom: ridge expert against strong expert.

For the same protocol (K=16, budget 4, q=4, 16 evenly spaced targets on
METR-LA and PEMS-BAY) this compares how much of the standalone-oracle gain each
learned router realizes with the frozen ridge expert and with the
subset-capable nonlinear expert, per target, with target-level bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def target_policy_means(run_dirs: dict[int, Path], dataset: str, policy: str, targets: set[int]):
    """Per-target mean gain for one policy, averaged over seeds."""

    values: dict[int, list[float]] = {}
    for seed, root in run_dirs.items():
        data_dir = root / dataset
        if not data_dir.exists():
            continue
        for cell in sorted(data_dir.glob("target*_seed*")):
            result_path = cell / "result.json"
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            target = int(result.get("target_sensor", cell.name.split("_")[0].replace("target", "")))
            if target not in targets:
                continue
            metrics = result.get("metrics", {}).get(policy)
            if metrics is None:
                continue
            values.setdefault(target, []).append(float(metrics["mean_prediction_gain"]))
    return {target: float(np.mean(v)) for target, v in values.items()}


def bootstrap(values: np.ndarray, *, n_boot: int = 5000, seed: int = 0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--primary", default="ranked_response_q4")
    args = parser.parse_args()

    ridge_runs = {
        101: ROOT / "KBS_MUR/results/raw/R080b_multitarget_rmur_s1_20260920_2340",
        202: ROOT / "KBS_MUR/results/raw/R080b_multitarget_rmur_s202_diag_20260921",
        303: ROOT / "KBS_MUR/results/raw/R080b_multitarget_rmur_s303_diag_20260921",
    }
    strong_runs = {101: ROOT / "PAMI_MUR/results/raw/R103_strong_backbone_gate_20260921_0350"}
    targets = set(range(0, 207)) if False else set()

    # the strong run uses 16 evenly spaced targets on each dataset; take those
    report = {"protocol": {"candidate_count": 16, "budget": 4, "primary_policy": args.primary}, "datasets": {}}
    for dataset, nodes in (("METRLA", 207), ("PEMSBAY", 325)):
        strong_dir = strong_runs[101] / dataset
        if not strong_dir.exists():
            continue
        strong_targets = {
            int(json.loads((cell / "result.json").read_text())["target_sensor"])
            for cell in strong_dir.glob("target*_seed*")
            if (cell / "result.json").exists()
        }
        if not strong_targets:
            continue
        rows = {}
        for label, runs in (("ridge", ridge_runs), ("nonlinear", strong_runs)):
            means = {
                policy: target_policy_means(runs, dataset, policy, strong_targets)
                for policy in ("static_utility", "cached_mur", args.primary, "oracle_static", "oracle_greedy")
            }
            shared = sorted(set.intersection(*[set(v) for v in means.values()]))
            realized = np.asarray([
                means[args.primary][t] / means["oracle_static"][t] for t in shared
            ])
            static_fraction = np.asarray([
                means["static_utility"][t] / means["oracle_static"][t] for t in shared
            ])
            headroom = np.asarray([
                (means["oracle_greedy"][t] - means["oracle_static"][t]) / means["oracle_greedy"][t]
                for t in shared
            ])
            rmur_mean = float(np.mean([means[args.primary][t] for t in shared]))
            static_mean = float(np.mean([means["static_utility"][t] for t in shared]))
            oracle_mean = float(np.mean([means["oracle_static"][t] for t in shared]))
            r_mean, r_lo, r_hi = bootstrap(realized, seed=1)
            s_mean, s_lo, s_hi = bootstrap(static_fraction, seed=2)
            rows[label] = {
                "targets": shared,
                "rmur_gain": rmur_mean,
                "static_gain": static_mean,
                "oracle_static_gain": oracle_mean,
                "realized_fraction": {"mean": r_mean, "low": r_lo, "high": r_hi},
                "static_fraction": {"mean": s_mean, "low": s_lo, "high": s_hi},
                "H_state_mean": float(np.mean(headroom)),
            }
        report["datasets"][dataset] = rows
        print(f"[{dataset}] targets={len(rows['ridge']['targets'])}")
        for label in ("ridge", "nonlinear"):
            entry = rows[label]
            print(
                f"  {label:9s} R-MUR {entry['rmur_gain']:.4f}  static {entry['static_gain']:.4f}  "
                f"oracle {entry['oracle_static_gain']:.4f}  realized {entry['realized_fraction']['mean']:.3f} "
                f"[{entry['realized_fraction']['low']:.3f},{entry['realized_fraction']['high']:.3f}]  "
                f"H_state {entry['H_state_mean']:.3f}"
            )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "realized_vs_structural.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
