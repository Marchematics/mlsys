#!/usr/bin/env python3
"""Run the fixed-redundancy candidate-pool scaling diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SWEEPER = ROOT / "scripts" / "run_oracle_ladder_sweep.py"


def parse_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("integer list must be nonempty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--candidate-counts", default="4,8,16,32,64")
    parser.add_argument("--redundancy", type=float, default=0.5)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--test-episodes", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-diagnostics", action="store_true")
    args = parser.parse_args()

    counts = parse_ints(args.candidate_counts)
    if args.out_root.exists():
        raise FileExistsError(f"output root already exists: {args.out_root}")
    args.out_root.mkdir(parents=True)
    for count in counts:
        child = args.out_root / f"k{count}"
        command = [
            sys.executable,
            str(SWEEPER),
            "--out-root",
            str(child),
            "--run-id",
            "R071_candidate_size",
            "--candidate-count",
            str(count),
            "--budget",
            str(args.budget),
            "--redundancies",
            str(args.redundancy),
            "--seeds",
            args.seeds,
            "--train-episodes",
            str(args.train_episodes),
            "--test-episodes",
            str(args.test_episodes),
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            *(["--save-diagnostics"] if args.save_diagnostics else []),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"completed {len(counts)} candidate-count sweeps under {args.out_root}")


if __name__ == "__main__":
    main()
