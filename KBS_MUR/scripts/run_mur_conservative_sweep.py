#!/usr/bin/env python3
"""Run the confirmatory MUR-Conservative gain--harm frontier sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_synthetic_gate_bn.py"


def parse_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("integer list must be nonempty")
    return values


def parse_floats(value: str) -> list[float]:
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("float list must be nonempty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--harmful-fractions", default=".25,.5")
    parser.add_argument("--alphas", default=".05,.10,.20,.30,.50")
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--redundancy", type=float, default=.5)
    parser.add_argument("--train-episodes", type=int, default=3000)
    parser.add_argument("--test-episodes", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--save-diagnostics",
        action="store_true",
        help="store per-state predicted and true marginals for post-hoc risk analysis",
    )
    args = parser.parse_args()

    harms = parse_floats(args.harmful_fractions)
    alphas = parse_floats(args.alphas)
    seeds = parse_ints(args.seeds)
    if args.max_workers < 1:
        raise ValueError("max-workers must be positive")
    if args.out_root.exists():
        raise FileExistsError(f"output root already exists: {args.out_root}")
    args.out_root.mkdir(parents=True)

    jobs: list[tuple[float, float, int]] = [
        (harm, alpha, seed) for harm in harms for alpha in alphas for seed in seeds
    ]
    pending = list(jobs)
    running: dict[subprocess.Popen[bytes], tuple[float, float, int, Path, object]] = {}
    records: list[dict[str, object]] = []

    while pending or running:
        while pending and len(running) < args.max_workers:
            harm, alpha, seed = pending.pop(0)
            label = f"h{harm:g}_a{alpha:g}_s{seed}"
            run_dir = args.out_root / label
            log_path = args.out_root / f"{label}.stdout.log"
            command = [
                sys.executable,
                str(RUNNER),
                "--seed",
                str(seed),
                "--run-id",
                "R072_mur_conservative",
                "--status",
                "confirmatory",
                "--redundancy",
                str(args.redundancy),
                "--harmful-fraction",
                str(harm),
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
                "--calibration-alpha",
                str(alpha),
                "--device",
                args.device,
                "--out",
                str(run_dir),
            ]
            if args.save_diagnostics:
                command.append("--save-diagnostics")
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT)
            running[process] = (harm, alpha, seed, run_dir, log_handle)

        finished = []
        for process, (harm, alpha, seed, run_dir, log_handle) in running.items():
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            if return_code != 0:
                raise RuntimeError(f"R072 job failed: harm={harm}, alpha={alpha}, seed={seed}")
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            records.append(
                {
                    "harmful_fraction": harm,
                    "alpha": alpha,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "metrics": result["metrics"],
                    "config": result["config"],
                }
            )
            finished.append(process)
        for process in finished:
            del running[process]
        if running and not finished:
            time.sleep(0.5)

    payload = {
        "protocol": {
            "harmful_fractions": harms,
            "alphas": alphas,
            "seeds": seeds,
            "candidate_count": args.candidate_count,
            "budget": args.budget,
            "redundancy": args.redundancy,
            "train_episodes": args.train_episodes,
            "test_episodes": args.test_episodes,
            "epochs": args.epochs,
            "max_workers": args.max_workers,
        },
        "records": records,
    }
    (args.out_root / "aggregate.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    metric_names = [
        "mean_prediction_gain",
        "negative_transfer_rate",
        "mean_selected_contexts",
        "positive_selection_precision",
    ]
    rows = [
        "harmful_fraction,alpha,policy," + ",".join(
            f"{name}_mean,{name}_std" for name in metric_names
        )
    ]
    for harm in harms:
        for alpha in alphas:
            subset = [
                record
                for record in records
                if record["harmful_fraction"] == harm and record["alpha"] == alpha
            ]
            policy_names = sorted(
                name
                for name in subset[0]["metrics"]
                if isinstance(subset[0]["metrics"][name], dict)
                and "mean_prediction_gain" in subset[0]["metrics"][name]
            )
            for policy in policy_names:
                values = {
                    name: np.asarray(
                        [record["metrics"][policy].get(name, np.nan) for record in subset],
                        dtype=float,
                    )
                    for name in metric_names
                }
                row = [f"{harm:g}", f"{alpha:g}", policy]
                for name in metric_names:
                    row.extend(
                        [
                            f"{np.nanmean(values[name]):.8f}",
                            f"{np.nanstd(values[name], ddof=1):.8f}",
                        ]
                    )
                rows.append(",".join(row))
    (args.out_root / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"out_root": str(args.out_root), "runs": len(records)}, indent=2))


if __name__ == "__main__":
    main()
