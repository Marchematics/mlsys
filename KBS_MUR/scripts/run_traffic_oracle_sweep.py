#!/usr/bin/env python3
"""Run the METR-LA oracle gate over independent expert seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_traffic_oracle_gate.py"


def parse_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("seed list must be nonempty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--validation-episodes", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=2000)
    parser.add_argument("--expert-repeats", type=int, default=3)
    parser.add_argument("--ridge-penalty", type=float, default=10.0)
    args = parser.parse_args()
    seeds = parse_ints(args.seeds)
    if args.out_root.exists():
        raise FileExistsError(f"output root already exists: {args.out_root}")
    args.out_root.mkdir(parents=True)
    records = []
    for seed in seeds:
        run_dir = args.out_root / f"seed{seed}"
        command = [
            sys.executable,
            str(RUNNER),
            "--data-path",
            str(args.data_path),
            "--seed",
            str(seed),
            "--candidate-count",
            str(args.candidate_count),
            "--budget",
            str(args.budget),
            "--history-length",
            str(args.history_length),
            "--horizon",
            str(args.horizon),
            "--train-episodes",
            str(args.train_episodes),
            "--validation-episodes",
            str(args.validation_episodes),
            "--test-episodes",
            str(args.test_episodes),
            "--expert-repeats",
            str(args.expert_repeats),
            "--ridge-penalty",
            str(args.ridge_penalty),
            "--out",
            str(run_dir),
        ]
        log_path = args.out_root / f"seed{seed}.stdout.log"
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        records.append(result)
    payload = {
        "protocol": {
            "seeds": seeds,
            "candidate_count": args.candidate_count,
            "budget": args.budget,
            "history_length": args.history_length,
            "horizon": args.horizon,
            "train_episodes": args.train_episodes,
            "validation_episodes": args.validation_episodes,
            "test_episodes": args.test_episodes,
            "expert_repeats": args.expert_repeats,
            "ridge_penalty": args.ridge_penalty,
            "data_path": str(args.data_path),
        },
        "records": records,
    }
    (args.out_root / "aggregate.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    rows = [
        "policy,mean_prediction_gain_mean,mean_prediction_gain_std,"
        "mean_utility_recovery_mean,mean_utility_recovery_std,"
        "negative_transfer_rate_mean,negative_transfer_rate_std"
    ]
    for policy in ("base_only", "oracle_static", "oracle_greedy"):
        values = {
            name: np.asarray(
                [record["metrics"][policy][name] for record in records], dtype=float
            )
            for name in (
                "mean_prediction_gain",
                "mean_utility_recovery",
                "negative_transfer_rate",
            )
        }
        rows.append(
            ",".join(
                [
                    policy,
                    f"{values['mean_prediction_gain'].mean():.8f}",
                    f"{values['mean_prediction_gain'].std(ddof=1):.8f}",
                    f"{values['mean_utility_recovery'].mean():.8f}",
                    f"{values['mean_utility_recovery'].std(ddof=1):.8f}",
                    f"{values['negative_transfer_rate'].mean():.8f}",
                    f"{values['negative_transfer_rate'].std(ddof=1):.8f}",
                ]
            )
        )
    (args.out_root / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"out_root": str(args.out_root), "runs": len(records)}, indent=2))


if __name__ == "__main__":
    main()
