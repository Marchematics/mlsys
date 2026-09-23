#!/usr/bin/env python3
"""Run the full METR-LA router comparison over training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_traffic_full_router.py"


def parse_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("seed list must be nonempty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--gate-result", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
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
            "--gate-result",
            str(args.gate_result),
            "--seed",
            str(seed),
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--out",
            str(run_dir),
        ]
        log_path = args.out_root / f"seed{seed}.stdout.log"
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
        records.append(json.loads((run_dir / "result.json").read_text(encoding="utf-8")))
    payload = {"protocol": {"seeds": seeds, "epochs": args.epochs}, "records": records}
    (args.out_root / "aggregate.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    policies = [
        "base_only",
        "pool_all",
        "relevance",
        "mmr",
        "static_utility",
        "oracle_static",
        "mur",
        "oracle_greedy",
    ]
    metric_names = ["mean_prediction_gain", "negative_transfer_rate", "mean_utility_recovery"]
    rows = ["policy," + ",".join(f"{m}_mean,{m}_std" for m in metric_names)]
    for policy in policies:
        row = [policy]
        for metric in metric_names:
            values = np.asarray([r["metrics"][policy][metric] for r in records], dtype=float)
            row.extend([f"{np.nanmean(values):.8f}", f"{np.nanstd(values, ddof=1):.8f}"])
        rows.append(",".join(row))
    (args.out_root / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"out_root": str(args.out_root), "runs": len(records)}, indent=2))


if __name__ == "__main__":
    main()
