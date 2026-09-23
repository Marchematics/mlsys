"""Is an internally consistent pilot a trustworthy one? (R156)

R154/R155 leave a practical gap: a pilot is directionally reliable but its
significance call is not, and there is no way to tell from the pilot itself
whether it is one of the good ones. This script tests the obvious candidate
check --- split the pilot in half and see whether the halves agree.

Using the six-dataset ridge protocol (32 targets each, three seeds), a pilot of
``n`` targets is drawn without replacement and split into two disjoint halves.
The pilot is *internally consistent* when both halves point the same way, and
the question is whether consistency predicts agreement with the full-sample
verdict. Two consistency definitions are reported, because they trade coverage
against precision:

* ``strict`` -- both halves resolve, in the same direction;
* ``sign`` -- both halves have the same sign, resolved or not.

Usage
-----
python experiments/analyze_pilot_consistency.py --out results/derived/R156_pilot_consistency
"""

from __future__ import annotations

import argparse
import json
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


def per_target_difference(dataset: str) -> np.ndarray:
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
    return np.asarray(
        [np.mean([seeds[s] for s in SEEDS if s in seeds]) for _, seeds in sorted(buckets.items())],
        dtype=float,
    )


def verdict(values: np.ndarray, *, draws: int = 1500, seed: int = 0) -> str:
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
    parser.add_argument("--pilots", type=int, default=600)
    parser.add_argument("--sizes", default="8,16,24")
    args = parser.parse_args()
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]

    rng = np.random.default_rng(20260926)
    full, values_by_dataset = {}, {}
    for dataset in DATASETS:
        values = per_target_difference(dataset)
        values_by_dataset[dataset] = values
        full[dataset] = verdict(values, draws=4000, seed=1)

    summary = {"full_verdicts": full, "by_size": {}}
    for size in sizes:
        rows = []
        for dataset in DATASETS:
            values = values_by_dataset[dataset]
            if size > values.size or size % 2:
                continue
            half = size // 2
            for _ in range(args.pilots):
                pick = rng.choice(values.size, size=size, replace=False)
                first, second = pick[:half], pick[half:]
                v1 = verdict(values[first], seed=int(rng.integers(1 << 30)))
                v2 = verdict(values[second], seed=int(rng.integers(1 << 30)))
                pilot = verdict(values[pick], seed=int(rng.integers(1 << 30)))
                rows.append(
                    {
                        "dataset": dataset,
                        "pilot": pilot,
                        "full": full[dataset],
                        "strict": v1 == v2 and v1 != "UNRESOLVED",
                        "sign": np.sign(values[first].mean()) == np.sign(values[second].mean()),
                        "agree": pilot == full[dataset],
                        "pilot_says_ahead": pilot == "AHEAD",
                        "full_is_ahead": full[dataset] == "AHEAD",
                    }
                )

        def rate(records, key):
            return float(np.mean([r[key] for r in records])) if records else float("nan")

        consistent = [r for r in rows if r["strict"]]
        inconsistent = [r for r in rows if not r["strict"]]
        sign_ok = [r for r in rows if r["sign"]]
        sign_bad = [r for r in rows if not r["sign"]]
        summary["by_size"][str(size)] = {
            "pilots": len(rows),
            "overall_agreement": rate(rows, "agree"),
            "strict_coverage": rate(rows, "strict"),
            "strict_agreement": rate(consistent, "agree"),
            "inconsistent_agreement": rate(inconsistent, "agree"),
            "sign_coverage": rate(rows, "sign"),
            "sign_agreement": rate(sign_ok, "agree"),
            "sign_disagreement": rate(sign_bad, "agree"),
            # the operationally critical error: pilot says route where the full
            # sample says the router is behind
            "false_route_rate": float(
                np.mean([r["pilot_says_ahead"] and not r["full_is_ahead"] for r in rows])
            ),
            "false_route_rate_when_consistent": (
                float(np.mean([not r["full_is_ahead"] for r in consistent if r["pilot_says_ahead"]]))
                if any(r["pilot_says_ahead"] for r in consistent)
                else float("nan")
            ),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pilot_consistency.json").write_text(json.dumps(summary, indent=1))

    print("full-sample verdicts:", full)
    print(f"\n{'n':>4} {'agree':>7} {'strict cov':>11} {'agree|strict':>13} "
          f"{'agree|not':>10} {'sign cov':>9} {'agree|sign':>11} {'false route':>12}")
    for size in sizes:
        e = summary["by_size"].get(str(size))
        if not e:
            continue
        print(f"{size:>4} {e['overall_agreement']:>7.3f} {e['strict_coverage']:>11.3f} "
              f"{e['strict_agreement']:>13.3f} {e['inconsistent_agreement']:>10.3f} "
              f"{e['sign_coverage']:>9.3f} {e['sign_agreement']:>11.3f} {e['false_route_rate']:>12.3f}")
    print(f"\nwrote {args.out / 'pilot_consistency.json'}")


if __name__ == "__main__":
    main()
