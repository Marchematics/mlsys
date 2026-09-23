#!/usr/bin/env python3
"""Aggregate and gate the strong-backbone R-MUR runs.

Reads any number of run roots containing ``target*_seed*/`` cells (partial runs
are fine) and reports policy means, target-level and seed-level paired
contrasts, hierarchical bootstrap intervals, expert cost per policy and the
gate verdict for the strong-backbone confirmation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "KBS_MUR" / "scripts"))

from run_traffic_response_router_sweep import paired_ci  # noqa: E402


def bootstrap_ci(values: np.ndarray, *, n_boot: int = 5000, seed: int = 0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
    }


def load_cells(run_roots: list[Path], dataset: str | None):
    cells: dict[tuple[int, int], dict] = {}
    for run_root in run_roots:
        base = Path(run_root)
        candidates = [base] if base.name == dataset else []
        if dataset is None:
            candidates = [p for p in base.iterdir() if p.is_dir()]
        elif not candidates:
            candidates = [base / dataset]
        for directory in candidates:
            if not directory.exists():
                continue
            for cell_dir in sorted(directory.glob("target*_seed*")):
                result_path = cell_dir / "result.json"
                gains_path = cell_dir / "episode_gains.npz"
                if not result_path.exists():
                    continue
                result = json.loads(result_path.read_text(encoding="utf-8"))
                target = int(result.get("target_sensor", cell_dir.name.split("_")[0].replace("target", "")))
                seed = int(result.get("seed", cell_dir.name.split("_")[1].replace("seed", "")))
                cell = cells.setdefault((target, seed), {"target": target, "seed": seed, "gains": {}, "metrics": {}, "expert_rows": {}})
                if result.get("metrics"):
                    cell["metrics"].update(result["metrics"])
                cell["expert_rows"].update(result.get("expert_rows", {}))
                cell["dataset"] = result.get("dataset", dataset)
                if gains_path.exists():
                    arrays = np.load(gains_path)
                    for key in arrays.files:
                        cell["gains"][key] = arrays[key]
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--primary", default="ranked_response_q4")
    args = parser.parse_args()

    cells = load_cells(args.run_root, args.dataset)
    if not cells:
        raise SystemExit("no cells found")
    policies = sorted({name for cell in cells.values() for name in cell["gains"]})
    targets = sorted({cell["target"] for cell in cells.values()})
    seeds = sorted({cell["seed"] for cell in cells.values()})

    per_target = {policy: [] for policy in policies}
    per_target_delta = {policy: {other: [] for other in ("static_utility", "cached_mur")} for policy in policies}
    positive = {policy: 0 for policy in policies}
    for target in targets:
        target_cells = [cell for cell in cells.values() if cell["target"] == target]
        for policy in policies:
            values = [float(np.mean(cell["gains"][policy])) for cell in target_cells if policy in cell["gains"]]
            if not values:
                continue
            per_target[policy].append(float(np.mean(values)))
            if np.mean(values) > 0:
                positive[policy] += 1
            for other in ("static_utility", "cached_mur"):
                deltas = [
                    float(np.mean(cell["gains"][policy])) - float(np.mean(cell["gains"][other]))
                    for cell in target_cells
                    if policy in cell["gains"] and other in cell["gains"]
                ]
                if deltas:
                    per_target_delta[policy][other].append(float(np.mean(deltas)))

    summary = {
        "dataset": args.dataset,
        "run_roots": [str(path) for path in args.run_root],
        "cells": len(cells),
        "targets": targets,
        "seeds": seeds,
        "primary_policy": args.primary,
        "policies": {},
    }
    for policy in policies:
        seed_level = []
        for cell in cells.values():
            if policy in cell["gains"] and "static_utility" in cell["gains"]:
                seed_level.append(
                    float(np.mean(cell["gains"][policy])) - float(np.mean(cell["gains"]["static_utility"]))
                )
        entry = {
            "mean_gain": float(np.mean(per_target[policy])) if per_target[policy] else float("nan"),
            "targets_positive": positive[policy],
            "targets": len(per_target[policy]),
            "delta_static_target_bootstrap": bootstrap_ci(np.asarray(per_target_delta[policy]["static_utility"]), seed=1),
            "delta_cached_target_bootstrap": bootstrap_ci(np.asarray(per_target_delta[policy]["cached_mur"]), seed=2),
            "delta_static_seed_paired": paired_ci(np.asarray(seed_level)),
            "expert_rows_per_episode": float(np.mean([
                cell["expert_rows"].get(policy, 0) / max(len(cell["gains"].get(policy, [1])), 1)
                for cell in cells.values()
                if policy in cell["gains"]
            ])),
        }
        summary["policies"][policy] = entry

    primary = summary["policies"].get(args.primary)
    if primary:
        gate = {
            "G_RMUR_mean": primary["mean_gain"],
            "delta_static": primary["delta_static_target_bootstrap"],
            "delta_cached": primary["delta_cached_target_bootstrap"],
            "targets_above_static": int(np.sum(np.asarray(per_target_delta[args.primary]["static_utility"]) > 0)),
            "targets_above_cached": int(np.sum(np.asarray(per_target_delta[args.primary]["cached_mur"]) > 0)),
            "targets": len(per_target[args.primary]),
            "pass_g1": bool(
                primary["delta_static_target_bootstrap"]["low"] > 0.0
                and primary["delta_cached_target_bootstrap"]["low"] > 0.0
            ),
        }
        summary["gate"] = gate

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    rows = ["policy,mean_gain,delta_static,delta_static_lo,delta_static_hi,delta_cached,delta_cached_lo,delta_cached_hi,expert_rows_per_episode"]
    for policy in policies:
        entry = summary["policies"][policy]
        rows.append(
            f"{policy},{entry['mean_gain']:.6f},"
            f"{entry['delta_static_target_bootstrap']['mean']:.6f},"
            f"{entry['delta_static_target_bootstrap']['low']:.6f},"
            f"{entry['delta_static_target_bootstrap']['high']:.6f},"
            f"{entry['delta_cached_target_bootstrap']['mean']:.6f},"
            f"{entry['delta_cached_target_bootstrap']['low']:.6f},"
            f"{entry['delta_cached_target_bootstrap']['high']:.6f},"
            f"{entry['expert_rows_per_episode']:.2f}"
        )
    (args.out / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(summary.get("gate", {}), indent=2))
    print(json.dumps({p: round(summary["policies"][p]["mean_gain"], 5) for p in policies}, indent=2))


if __name__ == "__main__":
    main()
