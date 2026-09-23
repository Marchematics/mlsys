"""How large a pilot makes the protocol's verdict trustworthy? (R154)

The measurement protocol is meant to be run on a deployment before committing to
a router. That makes it a *decision procedure*, and the practical question is
how much of the deployment has to be labelled before its verdict can be
trusted. This script answers it with the data already on disk: for each of the
six datasets in the ridge regime, the full 32-target verdict is compared with
the verdict a pilot of n targets would have produced.

A pilot verdict is "route" when the pilot's paired target-level interval for
R-MUR minus pool-all excludes zero on the positive side, "pool" when it excludes
zero on the negative side, and "unresolved" otherwise. The full-sample verdict
is computed the same way over all 32 targets. Reported quantities:

* sign agreement -- pilot and full sample agree on the direction;
* decision agreement -- pilot gives the same AHEAD / BELOW / unresolved call;
* sensitivity and specificity for the AHEAD call, which is the one that matters
  operationally (a false AHEAD deploys a router that does not pay).

Usage
-----
python experiments/analyze_pilot_power.py --out results/derived/R154_pilot_power
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KBS_RAW = ROOT / "KBS_MUR/results/raw"
DATASETS = ["METRLA", "PEMSBAY", "PEMS03", "PEMS04", "PEMS07", "PEMS08"]
LEARNED_RUNS = [
    "R080b_multitarget_rmur_s1_20260920_2340",
    "R080b_multitarget_rmur_s202_diag_20260921",
    "R080b_multitarget_rmur_s303_diag_20260921",
]
PRIMARY = "ranked_response_q4"
SEEDS = (101, 202, 303)


def per_target_difference(dataset: str) -> dict[int, float]:
    """Target-level mean of R-MUR minus pool-all, averaged over the seeds."""

    buckets: dict[int, dict[int, float]] = {}
    for run in LEARNED_RUNS:
        for cell in sorted((KBS_RAW / run / dataset).glob("target*_seed*")):
            result = cell / "result.json"
            if not result.exists():
                continue
            record = json.loads(result.read_text())
            metrics = record.get("metrics") or {}
            if PRIMARY not in metrics or "pool_all" not in metrics:
                continue
            delta = (
                metrics[PRIMARY]["mean_prediction_gain"]
                - metrics["pool_all"]["mean_prediction_gain"]
            )
            buckets.setdefault(int(record["target_sensor"]), {})[int(record["seed"])] = float(delta)
    return {
        target: float(np.mean([seeds[s] for s in SEEDS if s in seeds]))
        for target, seeds in buckets.items()
    }


def verdict(values: np.ndarray, *, draws: int = 4000, seed: int = 0) -> str:
    """AHEAD / BELOW / UNRESOLVED from a paired target-level bootstrap."""

    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    low, high = np.quantile(means, 0.025), np.quantile(means, 0.975)
    if low > 0:
        return "AHEAD"
    if high < 0:
        return "BELOW"
    return "UNRESOLVED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilots", type=int, default=400, help="pilots sampled per size and dataset")
    parser.add_argument("--sizes", default="4,8,12,16,24")
    args = parser.parse_args()
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]

    summary = {"provenance": {"learned_runs": LEARNED_RUNS, "primary_policy": PRIMARY},
               "datasets": {}, "by_size": {}}
    rng = np.random.default_rng(20260925)
    full_verdicts = {}
    for dataset in DATASETS:
        per_target = per_target_difference(dataset)
        values = np.asarray([per_target[t] for t in sorted(per_target)], dtype=float)
        full = verdict(values)
        full_verdicts[dataset] = full
        summary["datasets"][dataset] = {
            "targets": int(values.size),
            "mean_advantage": float(values.mean()),
            "full_verdict": full,
        }

    for size in sizes:
        records = []
        for dataset in DATASETS:
            per_target = per_target_difference(dataset)
            values = np.asarray([per_target[t] for t in sorted(per_target)], dtype=float)
            full = full_verdicts[dataset]
            if size > values.size:
                continue
            for _ in range(args.pilots):
                pick = rng.choice(values.size, size=size, replace=False)
                records.append((dataset, full, verdict(values[pick], draws=1500, seed=int(rng.integers(1 << 30)))))
        same_sign = [
            (f == "UNRESOLVED" or p == "UNRESOLVED" or f == p) for _, f, p in records
        ]
        agree = [f == p for _, f, p in records]
        ahead_full = [(f, p) for _, f, p in records if f == "AHEAD"]
        not_ahead = [(f, p) for _, f, p in records if f != "AHEAD"]
        sensitivity = (
            float(np.mean([p == "AHEAD" for _, p in ahead_full])) if ahead_full else float("nan")
        )
        specificity = (
            float(np.mean([p != "AHEAD" for _, p in not_ahead])) if not_ahead else float("nan")
        )
        summary["by_size"][str(size)] = {
            "pilots": len(records),
            "sign_agreement": float(np.mean(same_sign)),
            "decision_agreement": float(np.mean(agree)),
            "sensitivity_for_AHEAD": sensitivity,
            "specificity_for_AHEAD": specificity,
            "pilot_AHEAD_rate": float(np.mean([p == "AHEAD" for _, _, p in records])),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pilot_power.json").write_text(json.dumps(summary, indent=1))

    print("full-sample verdicts:")
    for dataset in DATASETS:
        e = summary["datasets"][dataset]
        print(f"  {dataset:9s} mean Δ={e['mean_advantage']:+.4f}  {e['full_verdict']}")
    print(f"\n{'pilot n':>8} {'pilots':>7} {'sign agree':>11} {'decision agree':>15} "
          f"{'sensitivity':>12} {'specificity':>12}")
    for size in sizes:
        e = summary["by_size"].get(str(size))
        if not e:
            continue
        print(f"{size:>8} {e['pilots']:>7d} {e['sign_agreement']:>11.3f} {e['decision_agreement']:>15.3f} "
              f"{e['sensitivity_for_AHEAD']:>12.3f} {e['specificity_for_AHEAD']:>12.3f}")
    print(f"\nwrote {args.out / 'pilot_power.json'}")


if __name__ == "__main__":
    main()
