#!/usr/bin/env python3
"""Replicate the fixed Traffic observability probe across existing seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_traffic_observability_audit.py"


def parse_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("seed list must be nonempty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--states-per-episode", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    seeds = parse_ints(args.seeds)
    if args.out_root.exists():
        raise FileExistsError(f"output root already exists: {args.out_root}")
    args.out_root.mkdir(parents=True)
    records = []
    for seed in seeds:
        gate_result = args.gate_root / f"seed{seed}" / "result.json"
        run_dir = args.out_root / f"seed{seed}"
        command = [
            sys.executable,
            str(RUNNER),
            "--data-path",
            str(args.data_path),
            "--gate-result",
            str(gate_result),
            "--seed",
            str(seed),
            "--states-per-episode",
            str(args.states_per_episode),
            "--out",
            str(run_dir),
        ]
        log_path = args.out_root / f"seed{seed}.stdout.log"
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
        records.append(json.loads((run_dir / "result.json").read_text(encoding="utf-8")))
    payload = {"protocol": {"seeds": seeds, "states_per_episode": args.states_per_episode}, "records": records}
    (args.out_root / "aggregate.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    rows = ["seed,probe,mae,spearman,positive_auc,top1_accuracy,one_step_regret"]
    for record in records:
        for probe, data in record["results"].items():
            metrics = data["metrics"]
            rows.append(
                ",".join(
                    [
                        str(record["config"]["seed"]),
                        probe,
                        f"{metrics['mae']:.8f}",
                        f"{metrics['spearman']:.8f}",
                        f"{metrics['positive_auc']:.8f}",
                        f"{metrics['top1_accuracy']:.8f}",
                        f"{metrics['one_step_regret']:.8f}",
                    ]
                )
            )
    (args.out_root / "per_seed.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    summary_rows = ["probe,metric_mean,metric_std"]
    for probe in ("x1", "x2", "x3"):
        for metric in ("mae", "spearman", "positive_auc", "top1_accuracy", "one_step_regret"):
            values = np.asarray(
                [record["results"][probe]["metrics"][metric] for record in records], dtype=float
            )
            summary_rows.append(
                f"{probe}:{metric},{values.mean():.8f},{values.std(ddof=1):.8f}"
            )
    (args.out_root / "summary.csv").write_text("\n".join(summary_rows) + "\n", encoding="utf-8")
    print(json.dumps({"out_root": str(args.out_root), "runs": len(records)}, indent=2))


if __name__ == "__main__":
    main()
