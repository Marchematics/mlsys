"""Pool-size scaling on the strong backbone (R142).

Fixed budget, growing candidate pool. The question is whether the regime-B tie
between response-aware routing and pooling is an artefact of the small pool used
in the protocol: with 128 candidates, feeding all of them to the expert should
be infeasible and selection should separate. The design also isolates whether
the binding constraint is the size of the opportunity or its identifiability.

Usage
-----
python experiments/analyze_pool_size.py \
    --out results/derived/R142_pool_size
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = {
    16: ROOT / "results/raw/R141_adapt_sweep_20260923_1500/ft10",
    32: ROOT / "results/raw/R142_pool_size_20260923_1700/K32",
    64: ROOT / "results/raw/R142_pool_size_20260923_1700/K64",
    128: ROOT / "results/raw/R142_pool_size_20260923_1700/K128",
}
POLICIES = ["pool_all", "random", "ranked_response_q4", "ranked_response_q8",
            "cached_mur", "static_utility", "relevance", "mmr", "oracle_static", "oracle_greedy"]


def load(root: Path) -> dict:
    out = {}
    for cell in sorted(root.glob("target*_seed*/result.json")):
        record = json.loads(cell.read_text())
        metrics = record.get("metrics") or {}
        if "pool_all" not in metrics:
            continue
        key = (int(record["target_sensor"]), int(record["seed"]))
        out[key] = {p: float(v["mean_prediction_gain"]) for p, v in metrics.items()
                    if isinstance(v, dict) and "mean_prediction_gain" in v}
    return out


def bootstrap_ci(values, *, draws: int = 10000, seed: int = 20260924):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.quantile(means, 0.025)),
            "high": float(np.quantile(means, 0.975)), "n": int(values.size)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()

    levels = {k: load(v) for k, v in DEFAULT_RUNS.items()}
    for k, v in levels.items():
        print(f"K={k}: {len(v)} cells from {DEFAULT_RUNS[k]}")
    common = sorted(set.intersection(*[set(v) for v in levels.values()]))
    print(f"matched cells: {len(common)}")

    summary = {"levels": sorted(levels), "matched_cells": len(common), "by_k": {}}
    for k, cells in levels.items():
        entry = {"pool_size": k, "policy": {}}
        for policy in POLICIES:
            values = np.asarray([cells[c][policy] for c in common if policy in cells[c]], dtype=float)
            if values.size:
                entry["policy"][policy] = bootstrap_ci(values, draws=args.draws)
        for policy in ("ranked_response_q4", "ranked_response_q8", "cached_mur", "relevance", "mmr"):
            if policy not in entry["policy"]:
                continue
            for ref in ("pool_all", "random"):
                deltas = np.asarray([cells[c][policy] - cells[c][ref] for c in common
                                     if policy in cells[c] and ref in cells[c]], dtype=float)
                if deltas.size:
                    entry["policy"][policy][f"delta_vs_{ref}"] = bootstrap_ci(deltas, draws=args.draws)
        oracle = np.asarray([cells[c]["oracle_static"] for c in common], dtype=float)
        learned = np.asarray([cells[c]["ranked_response_q4"] for c in common], dtype=float)
        entry["oracle_minus_router"] = bootstrap_ci(oracle - learned, draws=args.draws)
        summary["by_k"][str(k)] = entry

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pool_size.json").write_text(json.dumps(summary, indent=1))
    with (args.out / "cells.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pool_size", "target", "seed", *POLICIES])
        for k, cells in levels.items():
            for c in common:
                writer.writerow([k, c[0], c[1], *[cells[c].get(p, "") for p in POLICIES]])

    print(f"\n{'K':>4} {'pool':>9} {'rmur_q4':>9} {'random':>9} {'oracle':>9} "
          f"{'oracle-router':>26} {'rmur-pool':>24}")
    for k in sorted(levels):
        e = summary["by_k"][str(k)]["policy"]
        g = lambda p, f="mean": e.get(p, {}).get(f, float("nan"))
        gap = summary["by_k"][str(k)]["oracle_minus_router"]
        dp = e.get("ranked_response_q4", {}).get("delta_vs_pool_all", {})
        print(f"{k:>4} {g('pool_all'):>+9.4f} {g('ranked_response_q4'):>+9.4f} {g('random'):>+9.4f} "
              f"{g('oracle_static'):>+9.4f} {gap['mean']:>+13.4f} [{gap['low']:+.4f},{gap['high']:+.4f}] "
              f"{dp.get('mean', float('nan')):>+10.4f} [{dp.get('low', float('nan')):+.4f},{dp.get('high', float('nan')):+.4f}]")
    print(f"\nwrote {args.out / 'pool_size.json'}")


if __name__ == "__main__":
    main()
