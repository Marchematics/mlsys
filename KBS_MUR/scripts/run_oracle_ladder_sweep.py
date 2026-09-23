#!/usr/bin/env python3
"""Run reproducible oracle-ladder sweeps for the controlled MUR experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_synthetic_gate_bn.py"


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--run-id", default="oracle_ladder_sweep")
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--redundancies", default="0,.25,.5,.75")
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--train-episodes", type=int, default=3000)
    parser.add_argument("--test-episodes", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mmr-gamma-grid", default="0,.25,.5,1.0")
    parser.add_argument("--save-diagnostics", action="store_true")
    args = parser.parse_args()

    redundancies = parse_floats(args.redundancies)
    seeds = parse_ints(args.seeds)
    if not redundancies or not seeds:
        raise ValueError("redundancies and seeds must be nonempty")
    if args.out_root.exists():
        raise FileExistsError(f"output root already exists: {args.out_root}")
    args.out_root.mkdir(parents=True)

    records: list[dict[str, object]] = []
    for redundancy in redundancies:
        for seed in seeds:
            label = f"k{args.candidate_count}_b{args.budget}_r{redundancy:g}_s{seed}"
            run_dir = args.out_root / label
            log_path = args.out_root / f"{label}.stdout.log"
            command = [
                sys.executable,
                str(RUNNER),
                "--seed",
                str(seed),
                "--run-id",
                args.run_id,
                "--redundancy",
                str(redundancy),
                "--candidate-count",
                str(args.candidate_count),
                "--budget",
                str(args.budget),
                "--train-episodes",
                str(args.train_episodes),
                "--test-episodes",
                str(args.test_episodes),
                "--epochs",
                str(args.epochs),
                "--device",
                args.device,
                "--mmr-gamma-grid",
                args.mmr_gamma_grid,
                *(["--save-diagnostics"] if args.save_diagnostics else []),
                "--out",
                str(run_dir),
            ]
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            metrics = result["metrics"]
            records.append(
                {
                    "candidate_count": args.candidate_count,
                    "budget": args.budget,
                    "redundancy": redundancy,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "metrics": metrics,
                    "state_diagnostics": result.get("state_diagnostics", {}),
                }
            )

    summary: dict[str, object] = {
        "protocol": {
            "candidate_count": args.candidate_count,
            "budget": args.budget,
            "redundancies": redundancies,
            "seeds": seeds,
            "train_episodes": args.train_episodes,
            "test_episodes": args.test_episodes,
            "epochs": args.epochs,
            "device": args.device,
            "mmr_gamma_grid": args.mmr_gamma_grid,
        },
        "records": records,
    }
    (args.out_root / "aggregate.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    metric_names = [
        "mean_prediction_gain",
        "mean_utility_recovery",
        "mean_duplicate_selections",
        "duplicate_selection_rate",
        "negative_transfer_rate",
        "mean_selected_contexts",
    ]
    rows: list[str] = [
        "redundancy,policy," + ",".join(f"{name}_mean,{name}_std" for name in metric_names)
    ]
    for redundancy in redundancies:
        subset = [record for record in records if record["redundancy"] == redundancy]
        policy_names = sorted(subset[0]["metrics"].keys())
        for policy in policy_names:
            if policy in {"state_conditioning_headroom", "mur_estimation_gap", "mur_headroom_recovery"}:
                continue
            values = {
                name: np.asarray(
                    [record["metrics"][policy][name] for record in subset], dtype=float
                )
                for name in metric_names
            }
            row = [f"{redundancy:g}", policy]
            for name in metric_names:
                row.extend([f"{np.nanmean(values[name]):.8f}", f"{np.nanstd(values[name], ddof=1):.8f}"])
            rows.append(",".join(row))
    (args.out_root / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    headroom_rows = [
        "redundancy,state_conditioning_headroom_mean,state_conditioning_headroom_std,"
        "mur_estimation_gap_mean,mur_estimation_gap_std,mur_headroom_recovery_mean,"
        "mur_headroom_recovery_std"
    ]
    for redundancy in redundancies:
        subset = [record for record in records if record["redundancy"] == redundancy]
        values = {
            key: np.asarray(
                [record["metrics"][key] for record in subset], dtype=float
            )
            for key in (
                "state_conditioning_headroom",
                "mur_estimation_gap",
                "mur_headroom_recovery",
            )
        }
        headroom_rows.append(
            ",".join(
                [
                    f"{redundancy:g}",
                    f"{np.nanmean(values['state_conditioning_headroom']):.8f}",
                    f"{np.nanstd(values['state_conditioning_headroom'], ddof=1):.8f}",
                    f"{np.nanmean(values['mur_estimation_gap']):.8f}",
                    f"{np.nanstd(values['mur_estimation_gap'], ddof=1):.8f}",
                    f"{np.nanmean(values['mur_headroom_recovery']):.8f}",
                    f"{np.nanstd(values['mur_headroom_recovery'], ddof=1):.8f}",
                ]
            )
        )
    (args.out_root / "headroom.csv").write_text(
        "\n".join(headroom_rows) + "\n", encoding="utf-8"
    )
    print(json.dumps({"out_root": str(args.out_root), "runs": len(records)}, indent=2))


if __name__ == "__main__":
    main()
